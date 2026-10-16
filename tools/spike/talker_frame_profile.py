"""Talker frame graph: replay time per bucket and its kernel breakdown (GPU, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/talker_frame_profile.py [buckets...]
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


@torch.inference_mode()
def main():
    from engine.models.talker import Talker
    from engine.runtime.paged import DecoderSpec, PagedKV
    from engine.runtime.talker_decode import TalkerDecoder
    from engine.talker_sampler import PAD

    buckets = tuple(int(b) for b in sys.argv[1:]) or (1, 8, 32)
    ta = Talker.load(MODEL, device="cuda")
    c = ta.cfg
    print({k: getattr(c, k) for k in ("num_layers", "hidden_size", "num_heads", "num_kv_heads", "head_dim")}, "head", tuple(ta.head.shape))
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, False)
    kv = PagedKV(spec, 4096)
    for b in buckets:
        dec = TalkerDecoder(ta, kv, spec, buckets=(b,))
        seqs = list(range(b))
        for s in seqs:
            kv.reserve(s, 100)
            kv.lengths[s] = 100
        st = {"codes": torch.zeros(b, dtype=torch.int64, device="cuda"), "seeds": torch.arange(b, device="cuda"),
              "steps": torch.ones(b, dtype=torch.int64, device="cuda"), "history": torch.full((b, 16), PAD, dtype=torch.int64, device="cuda"),
              "live": torch.ones(b, dtype=torch.bool, device="cuda")}
        dec.decode_k(seqs, st, 4)
        bk = dec.buckets[b]
        torch.cuda.synchronize()
        n = 50
        t = time.perf_counter()
        for _ in range(n):
            bk.graph.replay()
        torch.cuda.synchronize()
        print(f"bucket {b}: frame replay {1e3 * (time.perf_counter() - t) / n:.3f} ms", flush=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                bk.graph.replay()
            torch.cuda.synchronize()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22))
        for s in seqs:
            kv.release(s)
        del dec


if __name__ == "__main__":
    main()
