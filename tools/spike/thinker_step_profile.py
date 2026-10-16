"""Thinker decode step: graph replay time per bucket and the eager step's kernel breakdown (GPU, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/thinker_step_profile.py [buckets...]
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


@torch.inference_mode()
def main():
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderSpec, PagedKV
    from engine.runtime.thinker_decode import ThinkerDecoder

    buckets = tuple(int(b) for b in sys.argv[1:]) or (1, 8, 32)
    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    kv = PagedKV(spec, 4096)
    for b in buckets:
        dec = ThinkerDecoder(th, kv, spec, buckets=(b,), use_graphs=True)
        seqs = list(range(b))
        for s in seqs:
            kv.reserve(s, 200)
            kv.lengths[s] = 200
        toks = [1] * b
        for _ in range(3):
            toks, _ = dec.step(seqs, toks)
        torch.cuda.synchronize()
        bk = dec.buckets[b]
        n = 50
        t = time.perf_counter()
        for _ in range(n):
            bk.graph.replay()
        torch.cuda.synchronize()
        print(f"bucket {b}: graph replay {1e3 * (time.perf_counter() - t) / n:.3f} ms", flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                bk.graph.replay()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
        for s in seqs:
            kv.release(s)
        del dec


if __name__ == "__main__":
    main()
