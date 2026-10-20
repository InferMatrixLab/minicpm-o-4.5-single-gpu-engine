"""Kernel breakdown of one batched steady Token2Wav window call in the headline configuration (FP16 flow, estimator
graphs + inductor, whole-Euler graphs), per component (GPU, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/t2w_kernel_profile.py [B ...]
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH


@torch.inference_mode()
def main():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    torch.backends.cudnn.allow_tf32 = False
    Bs = [int(b) for b in sys.argv[1:]] or [1, 8, 32]
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[:max(Bs)])
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True)
    import os

    fast = os.environ.get("FAST") == "1"
    if fast:
        bt = BatchedToken2Wav(t2w, use_graphs=True, fast_dit=True, fast_slots=65536)
    else:
        bt = BatchedToken2Wav(t2w, use_graphs=True, compile_estimator=True, euler_graphs=True)
    feats = [t2w.features(cr.ref_audio_bytes()) for cr in reqs]
    p0 = bt.setup(feats[0])
    bt.capture_graphs(p0)
    bt.release_prompt(p0)
    bt.profile = {}
    for k in (1, 4):
        for _ in range(3):
            torch.cuda.synchronize()
            t = time.perf_counter()
            ps = bt.setup_many(feats[:k])
            torch.cuda.synchronize()
            print(f"setup_many({k}) {1e3 * (time.perf_counter() - t):.1f} ms", flush=True)
            for p in ps:
                bt.release_prompt(p)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
        ps = bt.setup_many(feats[:4])
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=25, max_name_column_width=90))
    for p in ps:
        bt.release_prompt(p)
    prompts = bt.setup_many(feats)
    bt.profile = {}
    gen = np.random.default_rng(0)
    toks = [[int(x) for x in gen.integers(0, 6561, 25 * 8)] for _ in reqs]
    for B in Bs:
        sts = [bt.new_stream(p, noise_seed=i) for i, p in enumerate(prompts[:B])]
        wins = [StreamChunker().push(t) for t in toks[:B]]
        times = []
        for k in range(len(wins[0]) - 1):
            torch.cuda.synchronize()
            t = time.perf_counter()
            bt.decode([(sts[i], wins[i][k].tokens, False) for i in range(B)])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t) * 1000)
        print(f"B={B}: ms per call {[round(x, 1) for x in times]}", flush=True)
        k = len(wins[0]) - 1
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
            bt.decode([(sts[i], wins[i][k].tokens, False) for i in range(B)])
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30, max_name_column_width=90))
        print("graph counts", getattr(bt.graphs, "counts", None), getattr(bt.fast, "counts", None), "stage profile", bt.profile, flush=True)
        bt.profile = {}
        for st in sts:
            bt.release_stream(st)


if __name__ == "__main__":
    main()
