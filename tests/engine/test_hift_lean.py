"""``t2w_hift_lean``: HiFT with fused Snake activations and folded weight norms is the same vocoder (GPU).

Two Token2Wav instances (one made lean) vocode the same mels with the same source noise: eager in strict FP32 and with
the served TF32 stages, and through the HiFT graphs, for a ramped first window, a first window and steady windows with
a cached source, at several row counts.  HiFT itself is not bitwise reproducible (cuDNN's convolutions differ run to run
by ~1e-6 relative, strict FP32), so the lean vocoder must stay within 1e-5 relative L2 of the reference, an order of
magnitude inside AC-3.4's 1e-2.  The Snake kernel alone equals the reference module bit for bit on odd shapes (a
partial last block, one row).
"""

import contextlib
import os

import pytest
import torch

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")
TF32_R16 = [1, 2, "0:src", "0:res"]
BOUND = 1e-5


def _rel(a, b):
    return max(float((x - y).norm() / y.norm().clamp(min=1e-12)) for x, y in zip(a, b))


@torch.inference_mode()
def test_lean_hift_equals_the_reference_vocoder():
    from engine.models.token2wav import SOURCE_CACHE, Token2Wav, hift_forward_noise, hift_noise, hift_samples, strict_fp32
    from engine.runtime.hift_graph import HiFTGraphs
    from engine.runtime.hift_lean import make_lean

    torch.backends.cudnn.allow_tf32 = False
    ref = Token2Wav(MODEL / "assets" / "token2wav", float16=True).hift
    lean = Token2Wav(MODEL / "assets" / "token2wav", float16=True).hift
    changed = make_lean(lean)
    assert changed["snakes_fused"] == 72 and changed["weight_norms_folded"] > 0, changed
    graphs = {name: HiFTGraphs(h, decode_tf32=TF32_R16) for name, h in (("ref", ref), ("lean", lean))}
    for rows, frames, cache in ((1, 16, 0), (5, 50, 0), (16, 58, SOURCE_CACHE), (3, 58, SOURCE_CACHE)):
        g = torch.Generator(device="cuda").manual_seed(rows * 100 + frames)
        mel = torch.randn(rows, 80, frames, device="cuda", generator=g) * 2 - 4
        src = torch.randn(rows, 1, cache, device="cuda", generator=g) * 0.1
        rand_ini, noise = hift_noise(ref, list(range(rows)), hift_samples(ref, frames), "cuda")
        for tf in (False, TF32_R16):
            with strict_fp32():
                a = hift_forward_noise(ref, mel, src, rand_ini, noise, tf)
                again = hift_forward_noise(ref, mel, src, rand_ini, noise, tf)
                b = hift_forward_noise(lean, mel, src, rand_ini, noise, tf)
            print(rows, frames, "tf32" if tf else "fp32", f"lean {_rel(b, a):.1e} repeat {_rel(again, a):.1e}")
            assert _rel(b, a) <= BOUND, (rows, frames, tf)
        a, b = (graphs[n](mel, src, list(range(rows))) for n in ("ref", "lean"))
        print(rows, frames, f"graphs lean {_rel(b, a):.1e}")
        assert _rel(b, a) <= BOUND, (rows, frames, "graphs")


@torch.inference_mode()
def test_snake_kernel_equals_the_module():
    from stepaudio2.flashcosyvoice.modules.hifigan_components.layers import Snake

    from engine.runtime.hift_lean import make_lean

    for logscale in (False, True):
        m = Snake(96, alpha_logscale=logscale).cuda()
        m.alpha.data = torch.randn(96, device="cuda") * (0.5 if logscale else 3)
        holder = torch.nn.Sequential(Snake(96, alpha_logscale=logscale).cuda())
        holder[0].alpha.data = m.alpha.data.clone()
        make_lean(holder)
        for shape in ((1, 96, 1), (2, 96, 1025), (7, 96, 3000)):
            x = torch.randn(shape, device="cuda") * 4
            assert torch.equal(holder[0](x), m(x)), (logscale, shape)


@torch.inference_mode()
def test_fused_bias_resblocks_equal_the_lean_ones():
    """``make_lean(fused_bias=True)``: every ResBlock (biases, residuals and Snakes in ``bias_snake``) returns what the
    lean module returns, bit for bit, in strict FP32 and in TF32, on odd lengths; the vocoder too, through its graphs,
    wherever the lean graph itself replays bit for bit (some shapes' cuDNN kernels do not: there within ``BOUND``)."""
    from engine.models.token2wav import SOURCE_CACHE, Token2Wav, decoder_tf32, strict_fp32
    from engine.runtime.hift_graph import HiFTGraphs
    from engine.runtime.hift_lean import make_lean

    torch.backends.cudnn.allow_tf32 = False
    lean = Token2Wav(MODEL / "assets" / "token2wav", float16=True).hift
    fused = Token2Wav(MODEL / "assets" / "token2wav", float16=True).hift
    make_lean(lean)
    changed = make_lean(fused, fused_bias=True)
    assert changed["resblocks_fused"] == len(fused.resblocks) + len(fused.source_resblocks), changed
    g = torch.Generator(device="cuda").manual_seed(1)
    for blocks in ("resblocks", "source_resblocks"):
        for a, b in zip(getattr(lean, blocks), getattr(fused, blocks)):
            C = a.convs1[0].in_channels
            for shape, tf in (((3, C, 1001), False), ((1, C, 37), False), ((16, C, 2400), True)):
                x = torch.randn(shape, device="cuda", generator=g)
                with strict_fp32(), (decoder_tf32() if tf else contextlib.nullcontext()):
                    assert torch.equal(b(x), a(x)), (blocks, shape, tf)
    graphs = {name: HiFTGraphs(h, decode_tf32=TF32_R16) for name, h in (("lean", lean), ("fused", fused))}
    for rows, frames, cache in ((1, 16, 0), (5, 50, 0), (16, 58, SOURCE_CACHE), (3, 58, SOURCE_CACHE)):
        mel = torch.randn(rows, 80, frames, device="cuda", generator=g) * 2 - 4
        src = torch.randn(rows, 1, cache, device="cuda", generator=g) * 0.1
        a, again, b = (graphs[n](mel, src, list(range(rows))) for n in ("lean", "lean", "fused"))
        if all(torch.equal(x, y) for x, y in zip(a, again)):
            assert all(torch.equal(x, y) for x, y in zip(a, b)), (rows, frames)
        else:
            assert _rel(b, a) <= BOUND, (rows, frames)
