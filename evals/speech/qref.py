"""Speech qualification, Q_ref freeze and comparative gates.

    python -m evals.speech.qref qualify --quality tuning.json --effective-report results/effective_inputs/report.json --out qual.json
    python -m evals.speech.qref freeze --scope development|final --gpu L20X --out qref.json \
        --reference quality.json=qual.json [--reference ...]
    python -m evals.speech.qref gate --qref qref.json --quality engine.json --out gate.json [--headline-digest D]

* qualify: a baseline candidate qualifies when its tuning-set quality report
  (re-derived from its row sidecar) passes the absolute floors on pooled seeds
  and it is class R with eligible effective inputs.
* freeze: Q_ref per metric is the best seed-mean among references whose
  qualification passed; all references must share GPU, C, rows, seeds (exactly
  three), evaluator fingerprint and scope.  Final scope additionally needs the
  full 1088-row set.  A frozen file is never rewritten.
* gate: the engine report must match the Q_ref on GPU, C, rows, seeds,
  evaluator and scope; final scope also requires truncation evidence for every
  row and the declared headline candidate.
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

from bench.eligibility import ConfigClass  # noqa: E402
from evals.speech.detectors import RowQuality  # noqa: E402
from evals.speech.gates import QRef, comparative_gates, freeze_qref  # noqa: E402
from evals.speech.suite import FULL_ROWS, summarize  # noqa: E402

REQUIRED_SEEDS = 3


class QRefError(ValueError):
    pass


def _load(path) -> dict:
    return json.loads(Path(path).read_text())


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rederive(report_path: str) -> dict:
    """Recompute the summary from the row sidecar (checked against its recorded hash)."""
    rep = _load(report_path)
    sidecar = Path(report_path).with_suffix(".rows.jsonl")
    if _sha(sidecar) != rep["row_sidecar_sha256"]:
        raise QRefError(f"{sidecar} does not match its recorded hash")
    rows = [RowQuality(**json.loads(line)) for line in sidecar.read_text().splitlines() if line.strip()]
    return summarize(rows)


def class_r_record(candidate: dict, effective_eligible: bool) -> ConfigClass:
    """Class-R classification of a baseline candidate from its declared configuration."""
    adapter, config = candidate["adapter"], candidate["config"]
    if adapter == "vllm-omni":
        session = "duplex" if config in ("minicpmo_4_5.yaml", "minicpmo_4_5_duplex_h200.yaml", "minicpmo_4_5_duplex_mrv2.yaml") else "turn"
        return ConfigClass(engine=adapter, config=config, track="T", audio_output="stream", chunk_tokens=25,
                           precision={"thinker": "bf16", "talker": "bf16", "flow": "tf32", "hift": "fp32"}, euler_steps=10, session_mode=session,
                           canonical_inputs=effective_eligible, mps=candidate.get("mps") == "on")
    if adapter == "sglang-omni":
        return ConfigClass(engine=adapter, config=config, track="T", audio_output="full", chunk_tokens=None,
                           precision={"thinker": "bf16", "talker": "bf16", "flow": "fp16", "hift": "fp32"}, euler_steps=10, session_mode="turn",
                           canonical_inputs=effective_eligible, mps=candidate.get("mps") == "on")
    raise QRefError(f"{adapter} is not a baseline")


def qualify(quality_path: str, effective_report_path: str) -> dict:
    rep = _load(quality_path)
    if rep["generation"]["id_set"] != "tuning":
        raise QRefError("qualification uses the tuning ID set")
    summ = rederive(quality_path)
    eff = _load(effective_report_path)
    gen = rep["generation"]
    cand = gen["candidate"]
    if cand["adapter"] not in eff:
        raise QRefError(f"no effective-input record for {cand['adapter']}")
    eligible = bool(eff[cand["adapter"]]["eligible"])
    cr = class_r_record(cand, eligible)
    floors = summ["absolute_floors_pooled"]
    return {
        "candidate_id": gen["candidate_id"],
        "candidate_digest": gen["candidate_digest"],
        "gpu_sku": gen["gpu_sku"],
        "concurrency": gen["concurrency"],
        "evaluator_digest": rep["evaluator"]["digest"],
        "floors": floors,
        "per_seed_floors": {str(s["seed_index"]): s["absolute_floors"]["passed"] for s in summ["per_seed"]},
        "effective_inputs_eligible": eligible,
        "class_r": cr.to_json(),
        "qualified": bool(floors["passed"] and eligible and cr.is_class_r),
        "inputs": {"quality": {"path": quality_path, "sha256": _sha(quality_path)},
                   "effective_report": {"path": effective_report_path, "sha256": _sha(effective_report_path)}},
    }


def _identity(rep: dict) -> dict:
    g = rep["generation"]
    return {"gpu_sku": g["gpu_sku"], "concurrency": g["concurrency"], "id_set": g["id_set"], "measured_ids_sha256": g["measured_ids_sha256"],
            "seeds": sorted(int(s) for s in rep["seeds"]), "rows_per_seed": rep["rows_per_seed"], "evaluator_digest": rep["evaluator"]["digest"],
            "scope": rep["scope"]}


def _check_scope(rep: dict, scope: str) -> None:
    if rep["scope"] != scope:
        raise QRefError(f"report scope {rep['scope']} != {scope}")
    if len(rep["seeds"]) != REQUIRED_SEEDS:
        raise QRefError(f"exactly {REQUIRED_SEEDS} seeds are required, got {len(rep['seeds'])}")
    counts = set(rep["rows_per_seed"].values())
    if len(counts) != 1:
        raise QRefError("seeds have different row counts (denominators)")
    if scope == "final" and (rep["generation"]["id_set"] != "full" or counts != {FULL_ROWS}):
        raise QRefError(f"final scope needs all {FULL_ROWS} rows per seed")


def freeze(scope: str, gpu: str, refs: list[tuple[str, str]], out: Path) -> dict:
    if out.exists():
        raise QRefError(f"{out} exists: a frozen Q_ref is never rewritten")
    cands, inputs, ident = {}, {}, None
    for qpath, qualpath in refs:
        rep, qual = _load(qpath), _load(qualpath)
        _check_scope(rep, scope)
        if qual["candidate_digest"] != rep["generation"]["candidate_digest"]:
            raise QRefError(f"{qualpath} qualifies a different candidate than {qpath}")
        if qual["gpu_sku"] != rep["generation"]["gpu_sku"]:
            raise QRefError("qualification and reference were measured on different GPUs")
        i = _identity(rep)
        if ident is None:
            ident = i
        elif i != ident:
            raise QRefError(f"references disagree on {[k for k in i if i[k] != ident[k]]}")
        if not qual["qualified"]:
            inputs[rep["generation"]["candidate_id"]] = {"quality": qpath, "qualified": False}
            continue
        summ = rederive(qpath)
        cid = rep["generation"]["candidate_id"]
        cands[cid] = {"qualified": True, "rows_sha256": rep["rows_sha256"], "seed_indices": tuple(summ["seed_indices"]), "per_seed": summ["per_seed"]}
        inputs[cid] = {"quality": qpath, "quality_sha256": _sha(qpath), "qualification": qualpath, "qualification_sha256": _sha(qualpath),
                       "qualified": True}
    if ident is None:
        raise QRefError("no references")
    if ident["gpu_sku"] != gpu:
        raise QRefError(f"references were generated on {ident['gpu_sku']}, not {gpu}")
    if not cands:
        raise QRefError("no qualified reference")
    q = freeze_qref(cands, gpu, time.time())
    out.parent.mkdir(parents=True, exist_ok=True)
    doc = {**q.to_json(), "identity": ident, "inputs": inputs}
    out.write_text(json.dumps(doc, indent=1))
    return doc


def gate(qref_path: str, quality_path: str, headline_digest: str | None) -> dict:
    qd = _load(qref_path)
    ident = qd.pop("identity")
    qd.pop("inputs", None)
    qd["seed_indices"] = tuple(qd["seed_indices"])
    q = QRef(**qd)
    rep = _load(quality_path)
    _check_scope(rep, ident["scope"])
    mine = _identity(rep)
    bad = [k for k in ident if mine[k] != ident[k]]
    if bad:
        raise QRefError(f"engine report and Q_ref differ on {bad}")
    if ident["scope"] == "final":
        if not rep.get("truncation_evaluated"):
            raise QRefError("final gates need D_ref-based truncation evidence for every row")
        if headline_digest is None or rep["generation"]["candidate_digest"] != headline_digest:
            raise QRefError("final gates apply only to the declared headline configuration")
    summ = rederive(quality_path)
    res = comparative_gates(summ["per_seed"], rep["rows_sha256"], tuple(summ["seed_indices"]), q)
    res["absolute_floors_passed"] = bool(summ["absolute_floors_pooled"]["passed"])
    res["scope"] = ident["scope"]
    res["candidate_id"] = rep["generation"]["candidate_id"]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.speech.qref")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ql = sub.add_parser("qualify")
    ql.add_argument("--quality", required=True)
    ql.add_argument("--effective-report", required=True)
    ql.add_argument("--out", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--scope", choices=["development", "final"], required=True)
    fz.add_argument("--gpu", required=True, help="asserted GPU SKU; must equal the references' generation GPU")
    fz.add_argument("--reference", action="append", required=True, help="QUALITY_JSON=QUALIFICATION_JSON")
    fz.add_argument("--out", required=True)
    gt = sub.add_parser("gate")
    gt.add_argument("--qref", required=True)
    gt.add_argument("--quality", required=True)
    gt.add_argument("--headline-digest", default=None)
    gt.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "qualify":
        rec = qualify(a.quality, a.effective_report)
        Path(a.out).write_text(json.dumps(rec, indent=1))
        print(json.dumps({"candidate": rec["candidate_id"], "qualified": rec["qualified"], "floors": rec["floors"]["passed"]}))
        return 0
    if a.cmd == "freeze":
        refs = [tuple(r.split("=", 1)) for r in a.reference]
        doc = freeze(a.scope, a.gpu, refs, Path(a.out))
        print(json.dumps({"values": doc["values"], "sources": doc["sources"]}, indent=1))
        return 0
    res = gate(a.qref, a.quality, a.headline_digest)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"overall": res["overall"], "floors": res["absolute_floors_passed"], **{k: v["verdict"] for k, v in res["metrics"].items()}}))
    return 0 if res["overall"] == "pass" and res["absolute_floors_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
