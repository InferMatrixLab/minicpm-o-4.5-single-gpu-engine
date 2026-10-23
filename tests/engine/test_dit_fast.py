"""The fast DiT solve (slot-pool caches, packed ragged rows) against the batched reference solve, per request (GPU).

Prompt setups, then steady windows past the trim point, a ramped first window (16 frames) and final flushes that keep
no caches, all in one ragged call per window; the reference runs each request alone (``BatchedToken2Wav._solve``, the
eager estimator).  FP16 flow on both sides: mels agree to FP16 noise (different attention kernels and one rounding
fewer in the gated residuals), and the slot bookkeeping matches the reference cache length after every window.
"""

import os

import pytest
import torch

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")


def _setup_inputs(t2w, feats):
    import torch.nn.functional as F

    from engine.models.token2wav import SILENCE_TOKEN

    flow = t2w.flow
    out = []
    with torch.amp.autocast("cuda", dtype=torch.float16):
        for tokens, _, spk, mels, _ in feats:
            pad = torch.full((1, 3), SILENCE_TOKEN, device=tokens.device, dtype=tokens.dtype)
            h, _, _ = flow.encoder.forward_chunk(xs=flow.input_embedding(torch.cat([tokens, pad], dim=1)), last_chunk=False, cnn_cache=None,
                                                 att_cache=None)
            h = flow.encoder_proj(h)
            out.append((h.transpose(1, 2)[0], mels.transpose(1, 2)[0], flow.spk_embed_affine_layer(F.normalize(spk, dim=1))))
    return out


def _rel(a, b):
    return float((a - b).norm() / b.norm())


@pytest.mark.parametrize("graphs", [False, True])
@torch.inference_mode()
def test_fast_solve_matches_reference(graphs):
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import StreamState, Token2Wav
    from engine.runtime.dit_fast import FastDiT
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    torch.backends.cudnn.allow_tf32 = False
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True)
    bt = BatchedToken2Wav(t2w)
    fd = FastDiT(t2w.flow.decoder, t2w.n_timesteps, n_slots=8192, use_graphs=graphs)
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[:3])
    ins = _setup_inputs(t2w, [t2w.features(cr.ref_audio_bytes()) for cr in reqs])
    steps, depth = t2w.n_timesteps, len(t2w.flow.decoder.estimator.blocks)
    blk = t2w.flow.decoder.estimator.blocks[0]
    dt = torch.float16
    ref_sts = []
    for mu, cond, spk in ins:
        st = StreamState(prompt=None, flow_cache={
            "estimator_cnn_cache": torch.zeros(steps, depth, 2, blk.conv.in_channels + blk.conv.out_channels, 2, device="cuda", dtype=dt),
            "estimator_att_cache": torch.zeros(steps, depth, 2, blk.attn.num_heads, 0, 2 * blk.attn.head_dim, device="cuda", dtype=dt)},
            hift_cache={})
        ref_sts.append(st)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        ref = [bt._solve([st], mu[None].contiguous(), spk, cond=cond[None].contiguous(), keep_cache=True)[0] for st, (mu, cond, spk) in zip(ref_sts, ins)]
    caches = [fd.new_cache() for _ in ins]  # noqa: F841
    for c, (mu, _, _) in zip(caches, ins):
        c.keep = mu.shape[1]
    fast = fd.solve(caches, [mu for mu, _, _ in ins], torch.cat([s for _, _, s in ins]), [c for _, c, _ in ins], [True] * 3)
    errs = [_rel(f, r) for f, r in zip(fast, ref)]
    assert max(errs) < 2e-2, errs
    gen = torch.Generator(device="cuda").manual_seed(0)
    for w in range(7):
        # request 0: a ramped first window (16 frames) then steady; request 2: a final flush (no caches kept) at w = 6
        Ts = [16 if w == 0 else 50, 50, 50 if w < 6 else 34]
        keep = [True, True, w < 6]
        mus = [torch.randn(80, t, device="cuda", generator=gen).to(dt) for t in Ts]
        spks = torch.cat([s for _, _, s in ins])
        with torch.amp.autocast("cuda", dtype=torch.float16):
            ref = [bt._solve([st], m[None].contiguous(), spks[i:i + 1], keep_cache=k)[0] for i, (st, m, k) in enumerate(zip(ref_sts, mus, keep))]
        for i, st in enumerate(ref_sts):
            if keep[i]:
                from engine.runtime.token2wav_batch import _trim

                st.flow_cache["estimator_att_cache"] = _trim(st.flow_cache["estimator_att_cache"], ins[i][0].shape[1], 4)
        fast = fd.solve(caches, mus, spks, None, keep)
        errs = [_rel(f, r) for f, r in zip(fast, ref)]
        assert max(errs) < 3e-2, (w, errs)
        for i, st in enumerate(ref_sts):
            assert caches[i].length == st.flow_cache["estimator_att_cache"].shape[4], (w, i)
    used = fd.pool.in_use()
    for c in caches:
        fd.release(c)
    assert fd.pool.in_use() == fd.scratch and used > fd.scratch
    assert fd.counts["graph" if graphs else "eager"] >= 7, fd.counts


@torch.inference_mode()
def test_fused_conv_kernels_equal_gather():
    """``add_ln_mod_conv`` / ``ln_mish_conv`` give ``add_ln_mod`` / ``ln_mish`` + ``conv_gather`` bit for bit (im2col, new
    caches, residual) on ragged rows of 0, 1, 2 and more tokens with strided cache views."""
    from engine.runtime.dit_kernels import add_ln_mod, add_ln_mod_conv, conv_gather, ln_mish, ln_mish_conv

    gen = torch.Generator(device="cuda").manual_seed(3)
    C, dt = 1024, torch.float16
    lens = [5, 0, 1, 2, 3, 40, 0, 17, 1, 56]
    R, N = len(lens), sum(lens)
    starts = [sum(lens[:i]) for i in range(R)]
    meta = {"tok_row": torch.tensor([r for r, t in enumerate(lens) for _ in range(t)], dtype=torch.int32, device="cuda"),
            "tok_pos": torch.tensor([p for t in lens for p in range(t)], dtype=torch.int32, device="cuda"),
            "row_start": torch.tensor(starts, dtype=torch.int32, device="cuda"), "row_len": torch.tensor(lens, dtype=torch.int32, device="cuda")}
    rn = lambda *s: torch.randn(*s, device="cuda", generator=gen).to(dt)  # noqa: E731
    pool = rn(R, 2, 2, C)  # [row, conv, frame, C]: the solve's cache views have a row stride of 4 C
    h, y, g, sh, sc = rn(N, C), rn(N, C), rn(C), rn(C), 1 + rn(C)
    ln = (1 + rn(C), rn(C), 1e-5)
    hr, a = add_ln_mod(h, g, y, sh, sc, 1e-6)
    ref_im, ref_new = torch.full((N, 3 * C), 7.0, dtype=dt, device="cuda"), torch.full((R, 2, 2, C), 7.0, dtype=dt, device="cuda")
    conv_gather(a, pool[:, 0], ref_new[:, 0], meta, ref_im)
    im, new = torch.full_like(ref_im, -3.0), torch.full_like(ref_new, -3.0)
    hf = add_ln_mod_conv(h, g, y, sh, sc, 1e-6, pool[:, 0], new[:, 0], meta, im)
    assert torch.equal(hf, hr) and torch.equal(im, ref_im) and torch.equal(new[:, 0], ref_new[:, 0])
    x = rn(N, C)
    conv_gather(ln_mish(x, ln), pool[:, 1], ref_new[:, 1], meta, ref_im)
    ln_mish_conv(x, ln, pool[:, 1], new[:, 1], meta, im)
    assert torch.equal(im, ref_im) and torch.equal(new, ref_new)


@torch.inference_mode()
def test_qk_norm_scatter_equals_per_head_kernel():
    """The one-program-per-token q/k norm (all heads, the per-head kernel's reduction order) equals the per-head kernel
    bit for bit: q, the scattered keys and values, on activations of very different scales."""
    from engine.runtime.dit_kernels import qk_norm_scatter

    gen = torch.Generator(device="cuda").manual_seed(5)
    NH, D, dt = 16, 64, torch.float16
    for N, scale in ((1, 1.0), (333, 0.01), (1792, 3.0), (4100, 40.0)):
        qkv = (torch.randn(N, 3 * NH * D, device="cuda", generator=gen) * scale
               + torch.randn(3 * NH * D, device="cuda", generator=gen)).to(dt)
        qn = tuple(torch.randn(D, device="cuda", generator=gen).to(dt) for _ in range(2)) + (1e-6,)
        kn = tuple(torch.randn(D, device="cuda", generator=gen).to(dt) for _ in range(2)) + (1e-6,)
        slots = torch.randperm(8192, device="cuda", generator=gen)[:N]
        outs = []
        for per_head in (True, False):
            q = torch.full((N, NH, D), 9.0, dtype=dt, device="cuda")
            kc, vc = torch.zeros(8192, 1, NH, D, dtype=dt, device="cuda"), torch.zeros(8192, 1, NH, D, dtype=dt, device="cuda")
            qk_norm_scatter(qkv, qn, kn, q, kc, vc, slots, NH, D, per_head=per_head)
            outs.append((q, kc, vc))
        assert all(torch.equal(a, b) for a, b in zip(*outs)), (N, scale)
