"""Speech qualification, Q_ref freeze and comparative gates, on verified evidence only.

    python -m evals.speech.qref qualify --quality tuning.json --effective-dir results/effective_inputs --out qual.json
    python -m evals.speech.qref freeze --scope development|final --gpu L20X --out qref.json \
        --reference quality.json=qual.json [--reference ...]
    python -m evals.speech.qref verify --qref qref.json
    python -m evals.speech.qref gate --qref qref.json --quality engine.json --out gate.json [--headline-digest D]

* qualify: the tuning-set report must pass ``verify_report``.  Floors are
  re-derived from its sidecar; effective-input eligibility is recomputed from
  dumps whose provenance binds them to the measured stack commit, model
  revision and current request code (hard on prompt IDs and reference bytes
  under the recorded baseline policy); class R is computed from the resolved
  candidate semantics, bound to the measured deploy files, and fails closed
  when they are unknown.
* verify_qualification re-runs qualify on the recorded, hash-checked inputs.
* freeze accepts only verified reports and verified qualifications of the same
  candidate, hardware class and evaluator.  A final Q_ref names the campaign
  qualification and the frozen winners, and must hold exactly the required
  full-set references: every tuning-qualified class-R candidate at C=32 and the
  C=1 throughput winner at C=1.  Its identity is rows, seeds, seed policy,
  evaluator, model, dataset and hardware class; each gate also binds C.
* verify_qref rebuilds the Q_ref from its recorded sources (re-verifying every
  report, sidecar, source cell and qualification, the campaign and the winners)
  and compares every decisive field.  gate uses only a verified Q_ref; a final
  gate requires every engine source cell to have been declared against this
  Q_ref's content hash (``frozen_refs.speech_qref`` in its experiment
  declaration, written before the cell ran).
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

from bench.effective_inputs import compare_sets, eligible, load_dumps, verify_provenance  # noqa: E402
from bench.semantics import SemanticsUnknown, class_r  # noqa: E402
from evals.speech.gates import QRef, comparative_gates, freeze_qref  # noqa: E402
from evals.speech.verify_report import ReportError, verify_report  # noqa: E402

REQUIRED_SEEDS = 3
DECISIVE = ("candidate_digest", "gpu_sku", "gpu_driver", "evaluator_digest", "floors", "effective_inputs_eligible", "effective_inputs_problems",
            "class_r", "qualified")
FINAL_GATE_C = (1, 32)
QREF_DECISIVE = ("scope", "gpu", "rows_sha256", "seed_indices", "values", "per_seed", "sources", "identity", "references", "campaign", "winners")


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


def _source_spec(vr) -> dict:
    cell = next(iter(vr.report["seeds"].values()))["cell"]
    return json.loads((Path(cell) / "experiment.json").read_text())["experiment"]


def launch_env_problems(vr, cand: dict) -> None:
    """Every scored run's recorded launch environment holds only the device selection, caches and the candidate's own overlay."""
    allowed = {k for k, _ in cand.get("env", [])} | {"CUDA_VISIBLE_DEVICES"}
    for s, src in vr.report["seeds"].items():
        summ = json.loads((Path(src["cell"]) / src["run"] / "summary.json").read_text())
        env = ((summ.get("runtime_evidence") or {}).get("launch_env")) or {}
        extra = sorted(k for k in env if k not in allowed and not k.startswith("HF_"))
        if extra:
            raise SemanticsUnknown(f"seed {s}: the measured launch carried environment {extra} with unknown effect on reference semantics")


def qualify(quality_path: str, effective_dir: str, check_sources: bool = True) -> dict:
    vr = _verified(quality_path, check_sources)
    gen = vr.report["generation"]
    if gen["id_set"] != "tuning":
        raise QRefError("qualification uses the tuning ID set")
    cand = gen["candidate"]
    d = Path(effective_dir)
    canon_p, cand_p = d / "canonical.jsonl", d / f"{cand['adapter']}.jsonl"
    if check_sources:
        repo = (_source_spec(vr)["fingerprints"]["repos"]).get(cand["adapter"])
    else:
        from bench.effective_inputs import STACK_REPOS
        from bench.fingerprint import repo_fingerprint

        repo = repo_fingerprint(STACK_REPOS[cand["adapter"]]) if cand["adapter"] in STACK_REPOS else None
    if canon_p.exists() and cand_p.exists():
        try:
            from bench.semantics import preprocessing_identity

            prep = preprocessing_identity(cand["adapter"], cand["config"])
        except SemanticsUnknown as e:
            prep = {"unknown": str(e)}
        prov = (verify_provenance(canon_p, "canonical", gen["model"], None, preprocessing_identity("canonical", "")) +
                verify_provenance(cand_p, cand["adapter"], gen["model"], repo, prep))
        elig = eligible(compare_sets(load_dumps(cand_p), load_dumps(canon_p), strict_reference=False)) and not prov
    else:
        prov, elig = [f"no effective-input dumps for {cand['adapter']} in {d}"], False
    try:
        if check_sources:
            launch_env_problems(vr, cand)
        cr = class_r(cand, elig, check_binding=check_sources).to_json()
        cr_ok, reason = cr["class_r"], None
    except SemanticsUnknown as e:
        cr, cr_ok, reason = None, False, str(e)
    floors = vr.summary["absolute_floors_pooled"]
    return {
        "candidate_id": cand["candidate_id"],
        "candidate_digest": cand["digest"],
        "gpu_sku": gen["gpu_sku"],
        "gpu_driver": gen["gpu_driver"],
        "gpu_memory_mib": gen.get("gpu_memory_mib"),
        "gpu_uuids": {s: v.get("gpu_uuid") for s, v in vr.report["seeds"].items()},
        "concurrency": gen["concurrency"],
        "evaluator_digest": vr.report["evaluator"]["digest"],
        "floors": floors,
        "per_seed_floors": {str(s["seed_index"]): s["absolute_floors"]["passed"] for s in vr.summary["per_seed"]},
        "effective_inputs_eligible": elig,
        "effective_inputs_problems": prov,
        "class_r": cr,
        "class_r_unknown_reason": reason,
        "qualified": bool(floors["passed"] and elig and cr_ok),
        "inputs": {"quality": {"path": str(quality_path), "sha256": _sha(quality_path)},
                   "sidecar": {"path": str(Path(quality_path).with_suffix(".rows.jsonl")), "sha256": _sha(Path(quality_path).with_suffix(".rows.jsonl"))},
                   "canonical_dumps": {"path": str(canon_p), "sha256": _sha(canon_p) if canon_p.exists() else None},
                   "candidate_dumps": {"path": str(cand_p), "sha256": _sha(cand_p) if cand_p.exists() else None}},
    }


def verify_qualification(qual_path: str, check_sources: bool = True) -> dict:
    """Recompute a qualification from its recorded inputs; every decisive field must agree."""
    q = json.loads(Path(qual_path).read_text())
    for name, inp in (q.get("inputs") or {}).items():
        if inp["sha256"] is None and not Path(inp["path"]).exists():
            continue  # recorded as absent (no dumps: not eligible), still absent
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


def _match_identity(vr, scope: str) -> dict:
    """What a Q_ref and the reports it is compared with must share; a final Q_ref spans the gated concurrencies."""
    ident = dict(vr.identity)
    if scope == "final":
        ident.pop("concurrency")
        ident.pop("measured_ids_sha256")  # the full set is ordered per C; rows_sha256 binds membership
    return ident


def required_references(campaign: str, winners: str) -> set[str]:
    """A final Q_ref covers every tuning-qualified class-R candidate at C=32 and the C=1 throughput winner at C=1."""
    from evals.campaign import verify_campaign, verify_winners

    w = verify_winners(winners)
    if Path(w["qualification"]["path"]).resolve() != Path(campaign).resolve():
        raise QRefError("the winners were selected from another campaign qualification")
    rows = verify_campaign(campaign)["candidates"]
    req = {f"{r['candidate_id']}@C32" for r in rows if r["status"] == "qualified" and (r.get("qualification") or {}).get("class_r") is True}
    return req | {f"{w['cells']['1']['throughput']['candidate_id']}@C1"}


def build_qref(scope: str, gpu: str, refs: list[tuple[str, str]], check_sources: bool = True, campaign: str | None = None,
               winners: str | None = None) -> dict:
    """The Q_ref document (without the freeze time) computed from verified references."""
    if not refs:
        raise QRefError("no references")
    if scope == "final" and not (campaign and winners):
        raise QRefError("a final Q_ref names the campaign qualification and the frozen winners that define its required references")
    cands, inputs, ident = {}, {}, None
    for qpath, qualpath in refs:
        vr = _verified(qpath, check_sources)
        _check_scope(vr, scope)
        qual = verify_qualification(qualpath, check_sources)
        gen = vr.report["generation"]
        for k, w in (("candidate_digest", gen["candidate"]["digest"]), ("gpu_sku", gen["gpu_sku"]), ("gpu_driver", gen["gpu_driver"]),
                     ("gpu_memory_mib", gen.get("gpu_memory_mib")), ("evaluator_digest", vr.report["evaluator"]["digest"])):
            if qual.get(k) != w:
                raise QRefError(f"{qualpath}: qualification {k}={qual.get(k)!r} does not match the reference ({w!r})")
        if scope == "final" and gen["concurrency"] not in FINAL_GATE_C:
            raise QRefError(f"{qpath}: final references are measured at C in {FINAL_GATE_C}")
        mi = _match_identity(vr, scope)
        if ident is None:
            ident = mi
        elif mi != ident:
            raise QRefError(f"references disagree on {[k for k in ident if ident[k] != mi.get(k)]}")
        key = f"{gen['candidate']['candidate_id']}@C{gen['concurrency']}"
        if key in inputs:
            raise QRefError(f"{key} is given twice")
        inputs[key] = {"quality": str(qpath), "quality_sha256": _sha(qpath), "sidecar_sha256": _sha(Path(qpath).with_suffix(".rows.jsonl")),
                       "qualification": str(qualpath), "qualification_sha256": _sha(qualpath), "qualified": qual["qualified"],
                       "concurrency": gen["concurrency"]}
        if qual["qualified"]:
            cands[key] = {"qualified": True, "rows_sha256": vr.report["rows_sha256"], "seed_indices": tuple(vr.summary["seed_indices"]),
                          "per_seed": vr.summary["per_seed"]}
    if ident["gpu_sku"] != gpu:
        raise QRefError(f"references were generated on {ident['gpu_sku']}, not {gpu}")
    if not cands:
        raise QRefError("no qualified reference")
    if scope == "final":
        req = required_references(campaign, winners)
        if set(cands) != req:
            raise QRefError(f"final references must be exactly the required qualified set: missing {sorted(req - set(cands))}, "
                            f"unexpected {sorted(set(cands) - req)}")
    q = freeze_qref(cands, gpu, 0.0).to_json()
    q.pop("frozen_unix")
    bind = {"campaign": {"path": str(campaign), "sha256": _sha(campaign)}, "winners": {"path": str(winners), "sha256": _sha(winners)}} if scope == "final" else {}
    return {**q, "scope": scope, "identity": ident, "references": inputs, **bind}


def freeze(scope: str, gpu: str, refs: list[tuple[str, str]], out: Path, check_sources: bool = True, campaign: str | None = None,
           winners: str | None = None) -> dict:
    if out.exists():
        raise QRefError(f"{out} exists: a frozen Q_ref is never rewritten")
    doc = {**build_qref(scope, gpu, refs, check_sources, campaign, winners), "frozen_unix": time.time()}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    return doc


def verify_qref(qref_path: str, check_sources: bool = True) -> dict:
    """Rebuild the Q_ref from its recorded references and compare every decisive field."""
    qd = json.loads(Path(qref_path).read_text())
    refs = qd.get("references") or {}
    if not refs:
        raise QRefError(f"{qref_path}: Q_ref records no references")
    for key, inp in refs.items():
        for f, h in (("quality", "quality_sha256"), ("qualification", "qualification_sha256")):
            if not Path(inp[f]).exists() or _sha(inp[f]) != inp[h]:
                raise QRefError(f"{qref_path}: reference {key} {f} is missing or changed")
    for k in ("campaign", "winners"):
        if qd.get(k) and (not Path(qd[k]["path"]).exists() or _sha(qd[k]["path"]) != qd[k]["sha256"]):
            raise QRefError(f"{qref_path}: the {k} record changed since freezing")
    again = json.loads(json.dumps(build_qref(qd.get("scope"), qd.get("gpu"), [(v["quality"], v["qualification"]) for v in refs.values()], check_sources,
                                             (qd.get("campaign") or {}).get("path"), (qd.get("winners") or {}).get("path"))))
    bad = [k for k in QREF_DECISIVE if again.get(k) != qd.get(k)]
    if bad:
        raise QRefError(f"{qref_path}: recorded {bad} differ from the Q_ref rebuilt from its references")
    if not isinstance(qd.get("frozen_unix"), (int, float)):
        raise QRefError(f"{qref_path}: no freeze time")
    return qd


def gate(qref_path: str, quality_path: str, headline_digest: str | None, check_sources: bool = True) -> dict:
    qd = verify_qref(qref_path, check_sources)
    ident, scope = qd["identity"], qd["scope"]
    q = QRef(gpu=qd["gpu"], rows_sha256=qd["rows_sha256"], seed_indices=tuple(qd["seed_indices"]), values=qd["values"], per_seed=qd["per_seed"],
             sources=qd["sources"], frozen_unix=qd["frozen_unix"])
    vr = _verified(quality_path, check_sources)
    _check_scope(vr, scope)
    mi = _match_identity(vr, scope)
    bad = [k for k in ident if mi.get(k) != ident[k]]
    if bad:
        raise QRefError(f"engine report and Q_ref differ on {bad}")
    if scope == "final":
        c = vr.report["generation"]["concurrency"]
        if c not in FINAL_GATE_C:
            raise QRefError(f"final gates apply at C in {FINAL_GATE_C}, not {c}")
        if not vr.report.get("d_ref") or any(r.truncation is None for r in vr.rows):
            raise QRefError("final gates need verified D_ref-based truncation evidence for every row")
        if headline_digest is None or vr.report["generation"]["candidate"]["digest"] != headline_digest:
            raise QRefError("final gates apply only to the declared headline configuration")
        if check_sources:
            sha = _sha(qref_path)
            for s in vr.report["seeds"].values():
                from bench.verify import verify_cell

                v = verify_cell(Path(s["cell"]))  # the binding counts only inside the verified experiment its runs carry
                if not v.ok:
                    raise QRefError(f"{s['cell']} does not verify: {v.problems[:3]}")
                spec = json.loads((Path(s["cell"]) / "experiment.json").read_text())["experiment"]
                if ((spec.get("frozen_refs") or {}).get("speech_qref") or {}).get("sha256") != sha:
                    raise QRefError(f"{s['cell']} was not declared against this frozen Q_ref (experiment.frozen_refs.speech_qref)")
    res = comparative_gates(vr.summary["per_seed"], vr.report["rows_sha256"], tuple(vr.summary["seed_indices"]), q)
    res["absolute_floors_passed"] = bool(vr.summary["absolute_floors_pooled"]["passed"])
    res["scope"] = scope
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
    fz.add_argument("--campaign", default=None, help="campaign qualification record (final scope)")
    fz.add_argument("--winners", default=None, help="frozen winners (final scope)")
    vf = sub.add_parser("verify")
    vf.add_argument("--qref", required=True)
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
                          "unknown": rec["class_r_unknown_reason"], "input_problems": rec["effective_inputs_problems"]}))
        return 0
    if a.cmd == "freeze":
        refs = [tuple(r.split("=", 1)) for r in a.reference]
        doc = freeze(a.scope, a.gpu, refs, Path(a.out), campaign=a.campaign, winners=a.winners)
        print(json.dumps({"values": doc["values"], "sources": doc["sources"]}, indent=1))
        return 0
    if a.cmd == "verify":
        doc = verify_qref(a.qref)
        print(json.dumps({"verified": True, "values": doc["values"], "sources": doc["sources"]}, indent=1))
        return 0
    res = gate(a.qref, a.quality, a.headline_digest)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"overall": res["overall"], "floors": res["absolute_floors_passed"], **{k: v["verdict"] for k, v in res["metrics"].items()}}))
    return 0 if res["overall"] == "pass" and res["absolute_floors_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
