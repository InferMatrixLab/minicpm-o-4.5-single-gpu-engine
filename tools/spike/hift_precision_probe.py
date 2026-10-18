"""HiFT reduced-precision probe: waveform relative L2 against strict FP32 under the same seeded source noise, and time
per batched call, for precision variants of the vocoder (GPU, diagnostic; AC-3.4 bounds HiFT at rel-L2 <= 1e-2).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/hift_precision_probe.py [--rows 16] [--voices 32]

Inputs: 50-frame windows of real speech mels (the seed-tts reference voices' prompt mels), no source cache, one noise
seed per row.  Variants:
* ``tf32_all``: TF32 cuDNN convolutions everywhere (torch's default);
* ``tf32_decode``: f0 predictor and sine source in strict FP32, the decoder's convolutions (``_hift_decode``) in TF32;
* ``bf16_decode`` / ``fp16_decode``: f0 and source in FP32, the decoder under autocast.
"""

import argparse
import contextlib
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


@contextlib.contextmanager
def tf32(on: bool):
    prev = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = on
    try:
        yield
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = prev


def forward(hift, mel, rand_ini, noise, variant: str):
    import torch.nn.functional as F

    import engine.models.token2wav as tw

    with tf32(variant == "tf32_all"):
        # the source exactly as hift_forward_noise, in FP32 (or TF32 for tf32_all)
        _, s = tw.hift_forward_noise(hift, mel, torch.zeros(mel.shape[0], 1, 0, device=mel.device), rand_ini, noise)
    if variant in ("fp32", "tf32_all"):
        with tf32(variant == "tf32_all"):
            return tw._hift_decode(hift, mel, s)
    if variant.startswith("tf32_decode"):
        keep = {"tf32_decode": (), "tf32_decode_post32": ("conv_post",), "tf32_decode_src32": ("source_downs", "source_resblocks"),
                "tf32_decode_pre_post32": ("conv_pre", "conv_post")}[variant]
        saved = {}
        for name in keep:  # these modules in strict FP32 inside the TF32 decoder
            mod = getattr(hift, name)
            saved[name] = mod.forward

            def fp32_forward(*x, _f=mod.forward):
                with tf32(False):
                    return _f(*x)

            mod.forward = fp32_forward
        try:
            with tf32(True):
                return tw._hift_decode(hift, mel, s)
        finally:
            for name, f in saved.items():
                getattr(hift, name).forward = f
    dt = torch.bfloat16 if variant == "bf16_decode" else torch.float16
    with tf32(False), torch.amp.autocast("cuda", dtype=dt):
        return tw._hift_decode(hift, mel, s).float()
    del F


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=16)
    ap.add_argument("--voices", type=int, default=32)
    a = ap.parse_args()
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav, hift_noise, hift_samples

    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True, spk_threads=4)
    hift = t2w.hift
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(32)[:a.voices])
    mels = []
    for cr in reqs:
        m = t2w.features(cr.ref_audio_bytes())[3]  # [1, T, 80] prompt mel
        m = m.transpose(1, 2).float()
        for s0 in range(0, m.shape[2] - 50, 50):
            mels.append(m[:, :, s0:s0 + 50])
    mel = torch.cat(mels)
    n = hift_samples(hift, 50)
    rand_ini, noise = hift_noise(hift, list(range(mel.shape[0])), n, mel.device)
    print(f"{mel.shape[0]} windows of 50 mel frames from {len(reqs)} voices", flush=True)
    ref = torch.cat([forward(hift, mel[i:i + a.rows], rand_ini[i:i + a.rows], noise[i:i + a.rows], "fp32") for i in range(0, mel.shape[0], a.rows)])
    for v in ("fp32", "tf32_all", "tf32_decode", "tf32_decode_post32", "tf32_decode_src32", "tf32_decode_pre_post32", "bf16_decode", "fp16_decode"):
        out = torch.cat([forward(hift, mel[i:i + a.rows], rand_ini[i:i + a.rows], noise[i:i + a.rows], v) for i in range(0, mel.shape[0], a.rows)])
        rel = ((out - ref).norm(dim=-1) / ref.norm(dim=-1).clamp(min=1e-9)).flatten()
        b = mel[:a.rows], rand_ini[:a.rows], noise[:a.rows]
        for _ in range(3):
            forward(hift, *b, v)
        torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(10):
            forward(hift, *b, v)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t) * 100
        print(f"{v:12s} rel-L2 max {rel.max().item():.2e} mean {rel.mean().item():.2e} p99 {rel.quantile(0.99).item():.2e}  "
              f"{ms:.1f} ms per {a.rows}-row call (eager)", flush=True)


if __name__ == "__main__":
    main()
