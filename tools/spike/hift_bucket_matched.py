"""HiFT bucket sets on identical inputs with matched repeat controls (GPU, diagnostic; round-13 review M2).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/hift_bucket_matched.py --out results/engine_profile/r14_hift_matched.json

The 32-utterance served-path workload of ``tests/engine/test_hift_tf32.py`` (real voices speaking another voice's
speech tokens, first-chunk ramp 8, steady windows, final flushes, staggered starts so calls reach 32 rows) through the
headline ``BatchedToken2Wav`` with the batched setup encoder: strict FP32 once, TF32 upsampling with the current
buckets (to 16) twice and with the extended buckets (to 32) twice.  Every decoded chunk is copied out at once, so a
later replay cannot alter it.  Records per chunk the error of each TF32 run against FP32, the repeat spread of each
bucket set, and the difference between the sets; and per run the replays per (rows bucket, mel frames).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
N_UTT, FIRST_CHUNK = 32, 8
EXTENDED = (1, 2, 4, 8, 12, 16, 20, 24, 32)


def _serve(served, feats, plans, tf32, rows):
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    bt = BatchedToken2Wav(served, use_graphs=True, fast_dit=True, fast_slots=65536, hift_decode_tf32=tf32, batch_setup_encoder=True, hift_rows=rows)
    p0 = bt.setup(feats[0])
    bt.capture_graphs(p0)
    bt.release_prompt(p0)
    prompts = bt.setup_many(feats)
    sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(prompts)]
    starts = [j % 4 for j in range(N_UTT)]
    got = {j: [] for j in range(N_UTT)}
    for tick in range(max(s + len(p) for s, p in zip(starts, plans))):
        live = [j for j, s in enumerate(starts) if s <= tick < s + len(plans[j])]
        for j, w in zip(live, bt.decode([(sts[j], *plans[j][tick - starts[j]]) for j in live])):
            got[j].append(np.array(w, dtype=np.float64))  # owned copy
    replays = {f"{k}x{F}": n for (k, F), n in sorted(bt.hift_graphs.replays.items())}
    for st in sts:
        bt.release_stream(st)
    del bt
    torch.cuda.empty_cache()
    return got, replays


@torch.inference_mode()
def main():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.token2wav_chunker import StreamChunker

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
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
    runs = {"fp32": _serve(served, feats, plans, False, None), "cur1": _serve(served, feats, plans, True, None),
            "cur2": _serve(served, feats, plans, True, None), "ext1": _serve(served, feats, plans, True, EXTENDED),
            "ext2": _serve(served, feats, plans, True, EXTENDED)}
    rel = lambda x, y: float(np.linalg.norm(x - y) / np.linalg.norm(y)) if np.linalg.norm(y) > 0 else 0.0  # noqa: E731
    pairs = {"cur_vs_fp32": ("cur1", "fp32"), "ext_vs_fp32": ("ext1", "fp32"), "cur_repeat": ("cur2", "cur1"), "ext_repeat": ("ext2", "ext1"),
             "ext_vs_cur": ("ext1", "cur1")}
    chunks = {name: [] for name in pairs}
    for j in range(N_UTT):
        for k in range(len(runs["fp32"][0][j])):
            for name, (x, y) in pairs.items():
                chunks[name].append(rel(runs[x][0][j][k], runs[y][0][j][k]))
    summary = {name: {"max": max(v), "median": float(np.median(v))} for name, v in chunks.items()}
    out = {"commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
           "dirty": bool(subprocess.run(["git", "status", "--porcelain", "-uno"], capture_output=True, text=True, cwd=ROOT).stdout.strip()),
           "config": {"utterances": N_UTT, "first_chunk": FIRST_CHUNK, "batch_setup_encoder": True, "extended_rows": EXTENDED},
           "replays": {name: r[1] for name, r in runs.items()}, "summary": summary, "per_chunk": chunks,
           "ext_vs_cur_exceeds_repeat_spread": summary["ext_vs_cur"]["max"] > max(summary["cur_repeat"]["max"], summary["ext_repeat"]["max"])}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({"summary": summary, "replays": out["replays"], "ext_vs_cur_exceeds_repeat_spread": out["ext_vs_cur_exceeds_repeat_spread"]}, indent=1))


if __name__ == "__main__":
    main()
