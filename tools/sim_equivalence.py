"""Pinned versus query-blocked WavLM SIM attention on a scored report's own batches.

    python tools/sim_equivalence.py --cell CELL [--cell CELL ...] --gpu N --out OUT.json

The rows are collected exactly as ``evals.speech.suite score`` collects them (verified cells, one measured run per
seed) and batched as the scorer batches them.  The worker (``evals/speech/sim_equivalence_worker.py``, locked
evaluation environment) embeds every batch with both attentions and compares them bit for bit.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evals.evaluator import ASSETS_LOCK, EVAL_PYTHON, SETTINGS, SGLANG_DIR, verify_evaluator  # noqa: E402
from evals.speech.suite import _env, collect  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", action="append", required=True)
    ap.add_argument("--run", default=None)
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    fp = verify_evaluator()
    col = collect([Path(c) for c in a.cell], a.run)
    files = list(json.loads(ASSETS_LOCK.read_text())["files"])
    job = {"rows": [{"id": r.key, "wav": r.wav, "ref_wav": r.ref_wav} for r in col.rows], "device": "cuda:0", "sim_batch": SETTINGS["sim"]["batch"],
           "wavlm_finetune": next(f for f in files if f.endswith("wavlm_large_finetune.pth")), "wavlm_base": next(f for f in files if f.endswith("wavlm_large.pt"))}
    with tempfile.TemporaryDirectory() as td:
        jp, op = Path(td) / "job.json", Path(td) / "out.json"
        jp.write_text(json.dumps(job))
        subprocess.run([EVAL_PYTHON, str(ROOT / "evals" / "speech" / "sim_equivalence_worker.py"), str(jp), str(op)], cwd=str(SGLANG_DIR),
                       env={**_env(), "CUDA_VISIBLE_DEVICES": str(a.gpu)}, check=True)
        res = json.loads(op.read_text())
    b = res["batches"]
    both = [x for x in b if "bitwise_equal" in x]
    res.update(evaluator_digest=fp["digest"], cells=[str(c) for c in a.cell], generation=col.generation["candidate_id"], rows=len(col.rows),
               summary={"batches": len(b), "compared": len(both), "bitwise_equal": sum(x["bitwise_equal"] for x in both),
                        "max_abs_score_diff": max((x["max_abs_score_diff"] for x in both), default=None),
                        "pinned_oom": [x["batch"] for x in b if "pinned_oom" in x], "blocked_oom": [x["batch"] for x in b if "blocked_oom" in x],
                        "blocked_peak_gib_max": max((x.get("blocked_peak_gib", 0) for x in b), default=None)})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
