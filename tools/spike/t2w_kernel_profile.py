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
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


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
    bt = BatchedToken2Wav(t2w, use_graphs=True, compile_estimator=True, euler_graphs=True)
    prompts = [t2w.prepare(cr.ref_audio_bytes()) for cr in reqs]
    bt.capture_graphs(prompts[0])
    gen = np.random.default_rng(0)
    toks = [[int(x) for x in gen.integers(0, 6561, 25 * 8)] for _ in reqs]
    for B in Bs:
        sts = [t2w.new_stream(p, noise_seed=i) for i, p in enumerate(prompts[:B])]
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
        print("graph counts", getattr(bt.graphs, "counts", None), flush=True)


if __name__ == "__main__":
    main()
