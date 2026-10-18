"""The Token2Wav worker's decode call alone on the GPU (no engine beside it): ms per steady-window call at B streams,
split into encoder / DiT solve / vocode (HiFT), with the headline worker construction (GPU, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/t2w_decode_isolated.py [B ...]

Real prompt setups of B distinct seed-tts voices (``setup_many``, as the worker), then steady 28-token windows of random
codes for every stream in one ``decode`` call, timed with a synchronize around each call (``BatchedToken2Wav.profile``
splits it).  Also times ``setup_many`` per bucket of 1 / 2 / 4 prompts.  ``HIFT_TF32=1``: the TF32 HiFT upsampling
stages.  ``PROFILE=1``: kernel tables of one ``setup_many`` of 2 prompts and of one decode call at the largest B.
"""

import os
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


def _table(fn) -> None:
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA, torch.profiler.ProfilerActivity.CPU]) as prof:
        fn()
        torch.cuda.synchronize()
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=60), flush=True)


@torch.inference_mode()
def main():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    torch.backends.cudnn.allow_tf32 = False
    Bs = [int(b) for b in sys.argv[1:]] or [1, 8, 20, 32]
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True, spk_threads=4)
    bt = BatchedToken2Wav(t2w, use_graphs=True, compile_estimator=True, fast_dit=True, fast_slots=65536,
                          hift_decode_tf32=os.environ.get("HIFT_TF32") == "1")
    p0 = bt.setup(t2w.features(MODEL / "assets" / "HT_ref_audio.wav"))
    bt.capture_graphs(p0)
    bt.release_prompt(p0)
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(32)[:max(Bs)])
    feats = [t2w.features(cr.ref_audio_bytes()) for cr in reqs]
    for n in (1, 2, 4):
        ts = []
        for _ in range(3):
            torch.cuda.synchronize()
            t = time.perf_counter()
            ps = bt.setup_many(feats[:n])
            torch.cuda.synchronize()
            ts.append(1e3 * (time.perf_counter() - t))
            for p in ps:
                bt.release_prompt(p)
        print(f"setup_many n={n}: ms {[round(x, 1) for x in ts]}", flush=True)
    if os.environ.get("PROFILE") == "1":
        _table(lambda: [bt.release_prompt(p) for p in bt.setup_many(feats[:2])])
    rng = random.Random(0)
    for B in Bs:
        streams = [bt.new_stream(p, noise_seed=i, adopt=True) for i, p in enumerate(bt.setup_many(feats[:B]))]
        items = lambda: [(st, [rng.randrange(6561) for _ in range(28)], False) for st in streams]  # noqa: E731
        for _ in range(4):  # the caches reach their trimmed length; graphs warm
            bt.decode(items())
        bt.profile = {}
        ts = []
        for _ in range(8):
            torch.cuda.synchronize()
            t = time.perf_counter()
            bt.decode(items())
            torch.cuda.synchronize()
            ts.append(1e3 * (time.perf_counter() - t))
        split = {k: round(1e3 * v[1] / v[0], 2) for k, v in bt.profile.items()}
        print(f"B={B}: decode ms {[round(x, 1) for x in ts]} mean {sum(ts) / len(ts):.1f}  split ms/call {split}", flush=True)
        bt.profile = None
        if os.environ.get("PROFILE") == "1" and B == max(Bs):
            _table(lambda: bt.decode(items()))
        for st in streams:
            bt.release_stream(st)


if __name__ == "__main__":
    main()
