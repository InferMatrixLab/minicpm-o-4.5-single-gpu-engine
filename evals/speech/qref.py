"""Speech qualification, Q_ref freeze and comparative gates, on verified evidence only.

    python -m evals.speech.qref qualify --quality tuning.json --effective-dir results/effective_inputs --out qual.json
    python -m evals.speech.qref freeze --scope development|final --gpu L20X --out qref.json \
        --reference quality.json=qual.json [--reference ...]
    python -m evals.speech.qref gate --qref qref.json --quality engine.json --out gate.json [--headline-digest D]

* qualify: the tuning-set report must pass ``verify_report``.  Floors are
  re-derived from its sidecar; effective-input eligibility is recomputed from
  the bound dumps under the recorded baseline policy (hard on prompt IDs and
  reference bytes); class R is computed from the resolved candidate semantics
  and fails closed when they are unknown.
* verify_qualification re-runs qualify on the recorded, hash-checked inputs.
* freeze/gate accept only verified reports and verified qualifications; the
  qualification must belong to the same candidate, GPU and evaluator as the
  reference (tuning -> full-row reuse is the intended transition).
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

from bench.effective_inputs import compare_sets, eligible, load_dumps  # noqa: E402
from bench.semantics import SemanticsUnknown, class_r  # noqa: E402
from evals.speech.gates import QRef, comparative_gates, freeze_qref  # noqa: E402
from evals.speech.verify_report import ReportError, verify_report  # noqa: E402

REQUIRED_SEEDS = 3
DECISIVE = ("candidate_digest", "gpu_sku", "gpu_uuid", "evaluator_digest", "floors", "effective_inputs_eligible", "class_r", "qualified")


class QRefError(ValueError):
    pass


def _sha(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified(path: str, check_sources: bool = True):
    try:
        vr = verify_report(path, check_sources=check_sources)
    except ReportError as e:
        raise QRefError(str(e)) from e
    if not vr.ok:
        raise QRefError(f"{path} fails verification: {vr.problems[:4]}")
    return vr


def qualify(quality_path: str, effective_dir: str, check_sources: bool = True) -> dict:
    vr = _verified(quality_path, check_sources)
    gen = vr.report["generation"]
    if gen["id_set"] != "tuning":
        raise QRefError("qualification uses the tuning ID set")
    cand = gen["candidate"]
    d = Path(effective_dir)
    canon_p, cand_p = d / "canonical.jsonl", d / f"{cand['adapter']}.jsonl"
    mm = compare_sets(load_dumps(cand_p), load_dumps(canon_p), strict_reference=False)
    elig = eligible(mm)
    try:
        cr = class_r(cand, elig).to_json()
        cr_ok, reason = cr["class_r"], None
    except SemanticsUnknown as e:
        cr, cr_ok, reason = None, False, str(e)
    floors = vr.summary["absolute_floors_pooled"]
    return {
        "candidate_id": cand["candidate_id"],
        "candidate_digest": cand["digest"],
        "gpu_sku": gen["gpu_sku"],
        "gpu_uuid": gen["gpu_uuid"],
        "concurrency": gen["concurrency"],
        "evaluator_digest": vr.report["evaluator"]["digest"],
        "floors": floors,
        "per_seed_floors": {str(s["seed_index"]): s["absolute_floors"]["passed"] for s in vr.summary["per_seed"]},
        "effective_inputs_eligible": elig,
        "class_r": cr,
        "class_r_unknown_reason": reason,
        "qualified": bool(floors["passed"] and elig and cr_ok),
        "inputs": {"quality": {"path": str(quality_path), "sha256": _sha(quality_path)},
                   "sidecar": {"path": str(Path(quality_path).with_suffix(".rows.jsonl")), "sha256": _sha(Path(quality_path).with_suffix(".rows.jsonl"))},
                   "canonical_dumps": {"path": str(canon_p), "sha256": _sha(canon_p)},
                   "candidate_dumps": {"path": str(cand_p), "sha256": _sha(cand_p)}},
    }


def verify_qualification(qual_path: str, check_sources: bool = True) -> dict:
    """Recompute a qualification from its recorded inputs; every decisive field must agree."""
    q = json.loads(Path(qual_path).read_text())
    for name, inp in (q.get("inputs") or {}).items():
        if not Path(inp["path"]).exists() or _sha(inp["path"]) != inp["sha256"]:
            raise QRefError(f"{qual_path}: input {name} is missing or changed")
    again = json.loads(json.dumps(qualify(q["inputs"]["quality"]["path"], str(Path(q["inputs"]["canonical_dumps"]["path"]).parent), check_sources)))
    bad = [k for k in DECISIVE if again.get(k) != q.get(k)]
    if bad:
        raise QRefError(f"{qual_path}: recorded {bad} differ from the recomputed qualification")
    return again


def _check_scope(vr, scope: str) -> None:
    if vr.report["scope"] != scope:
        raise QRefError(f"report scope {vr.report['scope']} != {scope}")
    seeds = vr.identity["seeds"]
    if len(seeds) != REQUIRED_SEEDS:
        raise QRefError(f"exactly {REQUIRED_SEEDS} seeds are required, got {len(seeds)}")
    if len(set(vr.identity["rows_per_seed"].values())) != 1:
        raise QRefError("seeds have different row counts (denominators)")


def freeze(scope: str, gpu: str, refs: list[tuple[str, str]], out: Path, check_sources: bool = True) -> dict:
    if out.exists():
        raise QRefError(f"{out} exists: a frozen Q_ref is never rewritten")
    cands, inputs, ident = {}, {}, None
    for qpath, qualpath in refs:
        vr = _verified(qpath, check_sources)
        _check_scope(vr, scope)
        qual = verify_qualification(qualpath, check_sources)
        gen = vr.report["generation"]
        for k, w in (("candidate_digest", gen["candidate"]["digest"]), ("gpu_sku", gen["gpu_sku"]), ("gpu_uuid", gen["gpu_uuid"]),
                     ("evaluator_digest", vr.report["evaluator"]["digest"])):
            if qual[k] != w:
                raise QRefError(f"{qualpath}: qualification {k}={qual[k]!r} does not match the reference ({w!r})")
        if ident is None:
            ident = vr.identity
        elif vr.identity != ident:
            raise QRefError(f"references disagree on {[k for k in ident if ident[k] != vr.identity[k]]}")
        cid = gen["candidate"]["candidate_id"]
        inputs[cid] = {"quality": qpath, "quality_sha256": _sha(qpath), "qualification": qualpath, "qualification_sha256": _sha(qualpath),
                       "qualified": qual["qualified"]}
        if qual["qualified"]:
            cands[cid] = {"qualified": True, "rows_sha256": vr.report["rows_sha256"], "seed_indices": tuple(vr.summary["seed_indices"]),
                          "per_seed": vr.summary["per_seed"]}
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


def gate(qref_path: str, quality_path: str, headline_digest: str | None, check_sources: bool = True) -> dict:
    qd = json.loads(Path(qref_path).read_text())
    ident = qd.pop("identity")
    inputs = qd.pop("inputs", {})
    for cid, inp in inputs.items():
        if _sha(inp["quality"]) != inp["quality_sha256"] or _sha(inp["qualification"]) != inp["qualification_sha256"]:
            raise QRefError(f"Q_ref source {cid} changed since freezing")
    qd["seed_indices"] = tuple(qd["seed_indices"])
    q = QRef(**qd)
    vr = _verified(quality_path, check_sources)
    _check_scope(vr, ident["scope"])
    bad = [k for k in ident if vr.identity.get(k) != ident[k]]
    if bad:
        raise QRefError(f"engine report and Q_ref differ on {bad}")
    if ident["scope"] == "final":
        if not vr.report.get("d_ref") or any(r.truncation is None for r in vr.rows):
            raise QRefError("final gates need verified D_ref-based truncation evidence for every row")
        if headline_digest is None or vr.report["generation"]["candidate"]["digest"] != headline_digest:
            raise QRefError("final gates apply only to the declared headline configuration")
    res = comparative_gates(vr.summary["per_seed"], vr.report["rows_sha256"], tuple(vr.summary["seed_indices"]), q)
    res["absolute_floors_passed"] = bool(vr.summary["absolute_floors_pooled"]["passed"])
    res["scope"] = ident["scope"]
    res["candidate_id"] = vr.report["generation"]["candidate"]["candidate_id"]
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.speech.qref")
    sub = ap.add_subparsers(dest="cmd", required=True)
    ql = sub.add_parser("qualify")
    ql.add_argument("--quality", required=True)
    ql.add_argument("--effective-dir", required=True)
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
        rec = qualify(a.quality, a.effective_dir)
        Path(a.out).write_text(json.dumps(rec, indent=1))
        print(json.dumps({"candidate": rec["candidate_id"], "qualified": rec["qualified"], "floors": rec["floors"]["passed"],
                          "eligible": rec["effective_inputs_eligible"], "class_r": (rec["class_r"] or {}).get("class_r"),
                          "unknown": rec["class_r_unknown_reason"]}))
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
