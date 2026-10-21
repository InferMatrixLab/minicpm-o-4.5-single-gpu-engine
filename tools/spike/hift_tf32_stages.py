"""TF32 on a subset of HiFT's upsampling stages: AC-3.4 chunk error and decode time (GPU, diagnostic; B24).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/hift_tf32_stages.py --out results/engine_profile/r16_hift_stages.json

The 32-utterance served-path workload of ``tests/engine/test_hift_tf32.py`` with the batched setup encoder (the
combination that reaches 1.062e-2 on one chunk with every stage in TF32) through the headline ``BatchedToken2Wav``:
strict FP32 twice (the repeat spread), then each stage subset in TF32.  Records each variant's maximum and median
chunk error against FP32 and the worst chunk, and the GPU time of one HiFT call (20 rows, 50 + lookahead mel frames,
the served steady window) per variant.
"""

import argparse
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.paths import MODEL_PATH  # noqa: E402
from tools.spike.hift_bucket_matched import FIRST_CHUNK, N_UTT, _serve  # noqa: E402
MODEL = MODEL_PATH


def _time_call(hift, stages, rows=20, frames=56, n=20):
    from engine.models.token2wav import hift_forward_noise, hift_noise, hift_samples

    mel = torch.randn(rows, 80, frames, device="cuda")
    rand_ini, noise = hift_noise(hift, list(range(rows)), hift_samples(hift, frames), "cuda")
    src = torch.zeros(rows, 1, 0, device="cuda")
    for _ in range(3):
        hift_forward_noise(hift, mel, src, rand_ini, noise, stages)
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        hift_forward_noise(hift, mel, src, rand_ini, noise, stages)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


@torch.inference_mode()
def main():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import PARTS, Token2Wav
    from engine.token2wav_chunker import StreamChunker

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--parts", action="store_true", help="stages 1.. in TF32 plus subsets of stage 0's parts")
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[:N_UTT])
    served = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    feats = [served.features(cr.ref_audio_bytes()) for cr in reqs]
    plans = []
    for j in range(N_UTT):
        speech = [int(x) for x in feats[(j + 1) % N_UTT][0][0].tolist()]
        ch = StreamChunker(first_chunk=FIRST_CHUNK)
        ws = [(w.tokens, w.last) for w in ch.push(speech)]
        plans.append(ws + [(lambda w: (w.tokens, w.last))(ch.finish())])
    hift = served.hift
    n_up = hift.num_upsamples
    variants = [tuple(c) for r in range(n_up, 0, -1) for c in itertools.combinations(range(n_up), r)] if not a.parts else \
        [tuple(range(1, n_up)) + tuple(f"0:{q}" for q in c) for r in (2, 1) for c in itertools.combinations(PARTS, r)]
    ref = _serve(served, feats, plans, False, None)[0]
    again = _serve(served, feats, plans, False, None)[0]
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y))  # noqa: E731
    keys = [(j, k) for j in range(N_UTT) for k in range(len(ref[j])) if np.linalg.norm(ref[j][k]) > 0]
    out = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain", "-uno"], capture_output=True, text=True, cwd=ROOT).stdout.strip()),
           "chunks": len(keys), "fp32_repeat_max": max(rel(again[j][k], ref[j][k]) for j, k in keys), "variants": {}}
    for v in variants:
        got = _serve(served, feats, plans, list(v), None)[0]
        errs = sorted(((rel(got[j][k], ref[j][k]), j, k) for j, k in keys), reverse=True)
        out["variants"][",".join(map(str, v))] = {"max": errs[0][0], "worst": errs[0][1:], "second": errs[1][0],
                                                  "median": float(np.median([e[0] for e in errs])), "over_1e-2": sum(e[0] > 1e-2 for e in errs)}
        print(v, out["variants"][",".join(map(str, v))], flush=True)
    out["call_ms"] = {",".join(map(str, v)) or "fp32": _time_call(hift, list(v)) for v in [()] + variants}
    print("fp32 repeat max", out["fp32_repeat_max"], "call ms", out["call_ms"], flush=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
