"""Bound for overlapping the engine tick's two GPU waits (GPU, diagnostic).

Replays the Thinker decode graph (``nt`` rows) and K Talker frame graphs (``na`` rows) as one engine tick does:
serially with a host wait after each (today's tick), and on two streams with one wait at the end. The layouts are
fixed (no re-plans), so this is the upper bound of the GPU-side overlap, without the Token2Wav worker beside it.

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/tick_overlap_bound.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.paths import MODEL_PATH  # noqa: E402


def _time(fn, n=40):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return 1e3 * (time.perf_counter() - t) / n


@torch.inference_mode()
def main():
    from engine.models.talker import Talker
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderSpec, PagedKV
    from engine.runtime.talker_decode import TalkerDecoder
    from engine.runtime.thinker_decode import ThinkerDecoder
    from engine.talker_sampler import PAD, TalkerSampling

    th, ta = Thinker.load(MODEL_PATH, device="cuda"), Talker.load(MODEL_PATH, device="cuda")
    c, t = th.cfg, ta.cfg
    tspec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    aspec = DecoderSpec(t.num_layers, t.hidden_size, t.num_heads, t.num_kv_heads, t.head_dim, t.rope_theta, t.rms_eps, False)
    tkv, akv = PagedKV(tspec, 1024, 64), PagedKV(aspec, 1280, 128)
    k = 4
    print(f"{'nt':>3} {'na':>3} | thinker  talker(K={k}) | serial  overlap | saved")
    for nt, na in ((8, 16), (8, 24), (16, 16), (16, 32), (4, 32)):
        tdec = ThinkerDecoder(th, tkv, tspec, buckets=(nt,))
        adec = TalkerDecoder(ta, akv, aspec, TalkerSampling(min_new_tokens=0, max_new_tokens=1 << 30), buckets=(na,))
        ts, as_ = list(range(nt)), list(range(1000, 1000 + na))
        for s in ts:  # ~ prompt + some reply
            tkv.reserve(s, 300)
            tkv.lengths[s] = 260
        for s in as_:
            akv.reserve(s, 400)
            akv.lengths[s] = 200
        tdec.step(ts, [0] * nt)  # capture and plan
        st = {"codes": torch.zeros(na, dtype=torch.int64, device="cuda"), "seeds": torch.arange(na, device="cuda"),
              "steps": torch.ones(na, dtype=torch.int64, device="cuda"), "history": torch.full((na, 16), PAD, dtype=torch.int64, device="cuda"),
              "live": torch.ones(na, dtype=torch.bool, device="cuda")}
        adec.decode_k(as_, st, k)
        tg, ag = tdec.buckets[nt].graph, adec.buckets[na].graph
        s_t, s_a = torch.cuda.Stream(), torch.cuda.Stream()

        def thinker():
            tg.replay()
            torch.cuda.synchronize()

        def talker():
            for _ in range(k):
                ag.replay()
            torch.cuda.synchronize()

        def serial():
            thinker()
            talker()

        def overlap():
            with torch.cuda.stream(s_t):
                tg.replay()
            with torch.cuda.stream(s_a):
                for _ in range(k):
                    ag.replay()
            torch.cuda.synchronize()

        a, b, ser, ov = _time(thinker), _time(talker), _time(serial), _time(overlap)
        print(f"{nt:>3} {na:>3} | {a:7.2f}  {b:11.2f} | {ser:6.2f}  {ov:7.2f} | {100 * (1 - ov / ser):4.1f}%", flush=True)
        for s in ts:
            tkv.release(s)
        for s in as_:
            akv.release(s)
        tdec.release_pads()
        del tdec, adec
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
