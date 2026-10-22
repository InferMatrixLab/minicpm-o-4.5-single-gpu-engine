"""Prompt-setup rows in the decode's fast-DiT solve: one packed solve vs a setup solve plus a decode solve (GPU, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/fused_setup_bound.py [--out results/engine_profile/r18_fused_setup_bound.json]

Fixed layouts, no worker: ``n`` steady 28-token windows (56 frames, caches at the served steady length) and ``s``
prompt setups of evaluation voices (empty caches, prompt mel as condition), timed as two solves and as one solve with
CUDA events (GPU time of the calls) and host wall time, graphs on (as served).
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH


@torch.inference_mode()
def main():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--fuse-frames", type=int, default=0, help="BatchedToken2Wav fuse_setup_frames (DiT graphs take 2x that many more tokens)")
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True, spk_threads=4)
    bt = BatchedToken2Wav(t2w, use_graphs=True, compile_estimator=True, fast_dit=True, fast_slots=65536, batch_setup_encoder=True,
                          fuse_setup_frames=a.fuse_frames)
    p0 = bt.setup(t2w.features(MODEL / "assets" / "HT_ref_audio.wav"))
    torch.cuda.synchronize()
    m0, t0 = torch.cuda.memory_reserved(), time.perf_counter()
    n_graphs = bt.capture_graphs(p0)
    torch.cuda.synchronize()
    capture = {"graphs": n_graphs, "seconds": time.perf_counter() - t0, "reserved_gib": torch.cuda.memory_reserved() / 2**30,
               "capture_reserved_gib": (torch.cuda.memory_reserved() - m0) / 2**30}
    print("capture", capture, flush=True)
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).eval(32)[:24])
    feats = [t2w.features(cr.ref_audio_bytes()) for cr in reqs]
    frames = [int(f[3].shape[1]) for f in feats]
    print("prompt mel frames", frames, flush=True)
    fast = bt.fast
    dev = "cuda"
    mel = fast.mel
    # steady streams: prompt caches, then windows until the cache holds keep + 100 frames
    streams = [bt.new_stream(p, adopt=False) for p in bt.setup_many(feats[:4]) for _ in range(8)]
    for _ in range(3):
        cs = [st.flow_cache["fast"] for st in streams]
        fast.solve(cs, [torch.randn(mel, 56, device=dev, dtype=torch.float16) for _ in cs], torch.randn(len(cs), mel, device=dev, dtype=torch.float16),
                   None, [True] * len(cs))
    torch.cuda.synchronize()
    print("steady cache lengths", sorted({st.flow_cache["fast"].length for st in streams}), flush=True)

    def setup_rows(fs):
        cs = []
        for f in fs:
            c = fast.new_cache()
            c.keep = int(f[3].shape[1])
            cs.append(c)
        return cs, [torch.randn(mel, int(f[3].shape[1]), device=dev, dtype=torch.float16) for f in fs], [f[3][0].t().half() if f[3].dim() == 3 else f[3].half() for f in fs]

    def timed(fn):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t = time.perf_counter()
        e0.record()
        fn()
        e1.record()
        torch.cuda.synchronize()
        return e0.elapsed_time(e1), 1e3 * (time.perf_counter() - t)

    out = {"fuse_frames": a.fuse_frames, "capture": capture, "prompt_frames": frames, "cells": []}
    for n in (4, 8, 16, 24):
        for s in (1, 2, 4):
            sep, fused = [], []
            for rep in range(a.reps):
                fs = [feats[4 + (rep * s + j) % 20] for j in range(s)]
                dec = streams[:n]
                dmu = [torch.randn(mel, 56, device=dev, dtype=torch.float16) for _ in dec]
                dspk = torch.randn(n, mel, device=dev, dtype=torch.float16)
                sspk = torch.randn(s, mel, device=dev, dtype=torch.float16)
                for mode in ("sep", "fused"):
                    sc, smu, scond = setup_rows(fs)
                    for c, m in zip(scond, smu):
                        assert c.shape == m.shape, (c.shape, m.shape)
                    dcs = [st.flow_cache["fast"] for st in dec]
                    if mode == "sep":
                        def run():
                            fast.solve(sc, smu, sspk, scond, [True] * s)
                            fast.solve(dcs, dmu, dspk, None, [True] * n)
                    else:
                        def run():
                            fast.solve(sc + dcs, smu + dmu, torch.cat([sspk, dspk]), scond + [torch.zeros_like(m) for m in dmu], [True] * (s + n))
                    g0 = fast.counts["eager"]
                    r = timed(run)
                    if rep:  # the first repetition warms the buckets
                        (sep if mode == "sep" else fused).append(r + (fast.counts["eager"] - g0,))
                    for c in sc:
                        fast.release(c)
            med = lambda xs, i: sorted(x[i] for x in xs)[len(xs) // 2]  # noqa: E731
            cell = {"n": n, "s": s, "sep_gpu_ms": med(sep, 0), "fused_gpu_ms": med(fused, 0), "sep_wall_ms": med(sep, 1), "fused_wall_ms": med(fused, 1),
                    "sep_eager": sum(x[2] for x in sep), "fused_eager": sum(x[2] for x in fused)}
            out["cells"].append(cell)
            print(cell, flush=True)
    if a.out:
        Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
