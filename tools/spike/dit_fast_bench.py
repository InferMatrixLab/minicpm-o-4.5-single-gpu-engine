"""Fast DiT solve: ms per steady window call at B requests (after real prompt setups) and its per-kernel profile.

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/dit_fast_bench.py [B ...]
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "engine"))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH


@torch.inference_mode()
def main():
    from test_dit_fast import _setup_inputs

    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.runtime.dit_fast import FastDiT

    Bs = [int(b) for b in sys.argv[1:]] or [1, 8, 32]
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True)
    import os
    fd = FastDiT(t2w.flow.decoder, t2w.n_timesteps, n_slots=65536, use_graphs=os.environ.get("GRAPHS") == "1")
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[:max(Bs)])
    ins = _setup_inputs(t2w, [t2w.features(cr.ref_audio_bytes()) for cr in reqs])
    print("prompt mel frames", [mu.shape[1] for mu, _, _ in ins])
    for B in Bs:
        caches = [fd.new_cache() for _ in range(B)]
        for c, (mu, _, _) in zip(caches, ins):
            c.keep = mu.shape[1]
        spks = torch.cat([s for _, _, s in ins[:B]])
        torch.cuda.synchronize()
        t = time.perf_counter()
        fd.solve(caches, [mu for mu, _, _ in ins[:B]], spks, [c for _, c, _ in ins[:B]], [True] * B)
        torch.cuda.synchronize()
        print(f"B={B} setup {1e3 * (time.perf_counter() - t):.1f} ms", flush=True)
        times = []
        for w in range(8):
            mus = [torch.randn(80, 50, device="cuda").half() for _ in range(B)]
            torch.cuda.synchronize()
            t = time.perf_counter()
            fd.solve(caches, mus, spks, None, [True] * B)
            torch.cuda.synchronize()
            times.append(1e3 * (time.perf_counter() - t))
        print(f"B={B} steady window ms {[round(x, 1) for x in times]} cache len {caches[0].length} counts {fd.counts}", flush=True)
        if B == max(Bs):
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                fd.solve(caches, mus, spks, None, [True] * B)
                torch.cuda.synchronize()
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20, max_name_column_width=70))
        for c in caches:
            fd.release(c)


if __name__ == "__main__":
    main()
