"""Freeze the speech Q_ref and apply the comparative gates.

    python -m evals.speech.qref freeze --gpu L20X --out qref.json \
        --baseline <candidate_id>=<speech_quality.json> [--baseline ...]
    python -m evals.speech.qref gate --qref qref.json --quality <speech_quality.json> --out gate.json

A baseline qualifies when its pooled metrics over the seed policy (all seeds'
rows together) pass the absolute floors on the tuning rows; per-seed floor
results are reported alongside, because seed-to-seed noise is of the same order
as the floor margins.  Q_ref is the best seed-mean value per metric among qualified baselines,
frozen before any engine measurement (the file records its freeze time and the
inputs' hashes; it is never rewritten).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.speech.gates import QRef, comparative_gates, freeze_qref  # noqa: E402


def _load(path: str) -> dict:
    return json.loads(Path(path).read_text())


def qualifies(report: dict) -> bool:
    return bool(report["summary"]["absolute_floors_pooled"]["passed"])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.speech.qref")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--gpu", required=True)
    fz.add_argument("--baseline", action="append", required=True, help="CANDIDATE_ID=speech_quality.json")
    fz.add_argument("--out", required=True)
    gt = sub.add_parser("gate")
    gt.add_argument("--qref", required=True)
    gt.add_argument("--quality", required=True)
    gt.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "freeze":
        out = Path(a.out)
        if out.exists():
            raise SystemExit(f"{out} exists: a frozen Q_ref is never rewritten")
        cands, inputs = {}, {}
        for spec in a.baseline:
            cid, _, path = spec.partition("=")
            rep = _load(path)
            cands[cid] = {"qualified": qualifies(rep), "rows_sha256": rep["rows_sha256"], "seed_indices": tuple(rep["summary"]["seed_indices"]),
                          "per_seed": rep["summary"]["per_seed"]}
            inputs[cid] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(), "qualified": cands[cid]["qualified"]}
        q = freeze_qref(cands, a.gpu, time.time())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({**q.to_json(), "inputs": inputs}, indent=1))
        print(json.dumps({"qualified": {k: v["qualified"] for k, v in inputs.items()}, "values": q.values}, indent=1))
        return 0
    qd = _load(a.qref)
    qd.pop("inputs", None)
    qd["seed_indices"] = tuple(qd["seed_indices"])
    q = QRef(**qd)
    rep = _load(a.quality)
    res = comparative_gates(rep["summary"]["per_seed"], rep["rows_sha256"], tuple(rep["summary"]["seed_indices"]), q)
    res["absolute_floors"] = {s["seed_index"]: s["absolute_floors"] for s in rep["summary"]["per_seed"]}
    res["absolute_floors_passed"] = qualifies(rep)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"overall": res["overall"], "floors": res["absolute_floors_passed"], **{k: v["verdict"] for k, v in res["metrics"].items()}}, indent=1))
    return 0 if res["overall"] == "pass" and res["absolute_floors_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
