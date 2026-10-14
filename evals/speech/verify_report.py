"""Authoritative verification of speech quality reports and duration references.

Every consumer (qualify, freeze, gate) goes through ``verify_report``: it
re-derives membership, counts, seeds and metrics from the hash-checked row
sidecar, re-verifies each seed's source cell, and requires the sidecar rows of
each seed to be exactly that cell's declared measured IDs.  Final scope needs
the canonical 1088-row set for each of exactly three seeds.  Metadata in the
report is only accepted where it equals what the evidence yields.

A duration reference (D_ref) is a document built by ``build_dref`` from verified
cells of one candidate; ``verify_dref`` recomputes every per-row mean from those
cells before the reference may be used for truncation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bench.verify import verify_cell
from bench.workload import load_seed_tts_rows, make_id_sets
from evals.speech.detectors import RowQuality
from evals.speech.suite import FULL_ROWS, summarize

FINAL_SEEDS = 3


class ReportError(ValueError):
    pass


@dataclass
class VerifiedReport:
    path: Path
    report: dict
    rows: list[RowQuality]
    summary: dict
    identity: dict
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def _evaluator_consistent(ev: dict) -> bool:
    body = {k: v for k, v in ev.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == ev.get("digest")


def verify_report(path: str | Path, check_sources: bool = True) -> VerifiedReport:
    path = Path(path)
    rep = json.loads(path.read_text())
    problems: list[str] = []
    side = path.with_suffix(".rows.jsonl")
    if not side.exists() or _sha(side) != rep.get("row_sidecar_sha256"):
        raise ReportError(f"{side} is missing or does not match its recorded hash")
    rows = [RowQuality(**r) for r in _jsonl(side)]
    keys = [(r.seed_index, r.row_id) for r in rows]
    if len(set(keys)) != len(keys):
        problems.append("sidecar has duplicate (seed, row) entries")
    by_seed: dict[int, list[str]] = {}
    for s, rid in keys:
        by_seed.setdefault(s, []).append(rid)
    seeds = sorted(by_seed)
    if [str(s) for s in seeds] != sorted(rep.get("seeds", {}), key=int):
        problems.append(f"sidecar seeds {seeds} != report seeds {sorted(rep.get('seeds', {}))}")
    if {str(s): len(v) for s, v in by_seed.items()} != rep.get("rows_per_seed"):
        problems.append("rows_per_seed claimed by the report differs from the sidecar")
    rsha = hashlib.sha256("\n".join(sorted(f"{s}:{rid}" for s, rid in keys)).encode()).hexdigest()
    if rsha != rep.get("rows_sha256"):
        problems.append("rows_sha256 claimed by the report differs from the sidecar")
    if not _evaluator_consistent(rep.get("evaluator") or {}):
        problems.append("evaluator record does not match its digest")
    gen = rep.get("generation") or {}
    if check_sources:
        for s in seeds:
            src = (rep.get("seeds") or {}).get(str(s)) or {}
            v = verify_cell(src.get("cell", ""))
            if not v.ok:
                problems.append(f"seed {s}: source cell {src.get('cell')} fails verification: {v.problems[:2]}")
                continue
            spec = v.spec
            res = json.loads((Path(src["cell"]) / "result.json").read_text())
            if res.get("experiment_digest") != src.get("experiment_digest") or res["artifacts"].get(src.get("run")) != src.get("artifacts"):
                problems.append(f"seed {s}: source binding (digest/artifacts) changed since scoring")
            if int(spec["seed_index"]) != s:
                problems.append(f"seed {s}: source cell declares seed {spec['seed_index']}")
            if sorted(by_seed[s]) != sorted(spec["measured_ids"]):
                problems.append(f"seed {s}: sidecar rows differ from the source cell's declared IDs")
            want = {"candidate_digest": spec["candidate"]["digest"], "gpu_sku": spec["hardware"].get("gpu_sku"), "gpu_uuid": spec["hardware"].get("gpu_uuid"),
                    "concurrency": spec["concurrency"], "id_set": spec["id_set"], "dataset": spec["dataset"], "model": spec["model"]}
            for k, w in want.items():
                if gen.get(k) != w:
                    problems.append(f"generation.{k}={gen.get(k)!r} contradicts source cell ({w!r})")
            if gen.get("candidate", {}).get("digest", spec["candidate"]["digest"]) != spec["candidate"]["digest"]:
                problems.append("generation candidate identity contradicts the source cell")
    if rep.get("scope") == "final":
        canon = sorted(make_id_sets(load_seed_tts_rows()).full())
        if len(seeds) != FINAL_SEEDS:
            problems.append(f"final scope needs exactly {FINAL_SEEDS} seeds, got {len(seeds)}")
        for s in seeds:
            if sorted(by_seed[s]) != canon:
                problems.append(f"final scope: seed {s} does not cover the canonical {FULL_ROWS}-row set")
        if gen.get("id_set") != "full":
            problems.append("final scope needs full-set generation")
    if rep.get("d_ref"):
        dpath = Path(rep["d_ref"]["path"])
        if not dpath.exists() or _sha(dpath) != rep["d_ref"]["sha256"]:
            problems.append("duration reference file is missing or changed")
        else:
            dproblems, dref = verify_dref(dpath)
            problems += [f"d_ref: {p}" for p in dproblems]
            for r in rows:
                if dref.get(r.row_id) != r.d_ref_s:
                    problems.append(f"row {r.row_id} uses D_ref {r.d_ref_s} not in the verified reference")
                    break
    summary = summarize(rows) if rows else {}
    identity = {"gpu_sku": gen.get("gpu_sku"), "concurrency": gen.get("concurrency"), "id_set": gen.get("id_set"),
                "measured_ids_sha256": gen.get("measured_ids_sha256"), "dataset": gen.get("dataset"), "model": gen.get("model"),
                "seeds": seeds, "rows_per_seed": {str(s): len(v) for s, v in by_seed.items()}, "evaluator_digest": (rep.get("evaluator") or {}).get("digest"),
                "scope": rep.get("scope")}
    return VerifiedReport(path=path, report=rep, rows=rows, summary=summary, identity=identity, problems=problems)


def build_dref(cells: list[str | Path], out: Path, role: str) -> dict:
    """Per-row mean output duration over the measured runs of verified cells of ONE candidate."""
    durations: dict[str, list[float]] = {}
    sources, cand = [], None
    for c in cells:
        v = verify_cell(c)
        if not v.ok:
            raise ReportError(f"{c} fails verification: {v.problems[:2]}")
        if cand is None:
            cand = v.spec["candidate"]["digest"]
        elif v.spec["candidate"]["digest"] != cand:
            raise ReportError("a duration reference comes from a single candidate")
        res = json.loads((Path(c) / "result.json").read_text())
        runs = [f"run{i}" for i in range(1, int(v.spec["k"]) + 1)]
        for run in runs:
            for r in _jsonl(Path(c) / run / "requests.jsonl"):
                durations.setdefault(r["request_id"], []).append(float(r["duration_s"]))
        sources.append({"cell": str(Path(c).resolve()), "experiment_digest": res["experiment_digest"], "runs": runs,
                        "artifacts": {r: res["artifacts"][r] for r in runs}, "seed_index": v.spec["seed_index"]})
    doc = {"role": role, "candidate_digest": cand, "sources": sources, "rows": {k: float(np.mean(v)) for k, v in sorted(durations.items())}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    return doc


def verify_dref(path: Path) -> tuple[list[str], dict]:
    doc = json.loads(Path(path).read_text())
    problems, durations = [], {}
    for src in doc.get("sources", []):
        v = verify_cell(src["cell"])
        if not v.ok:
            problems.append(f"source {src['cell']} fails verification")
            continue
        if v.spec["candidate"]["digest"] != doc.get("candidate_digest"):
            problems.append("source cell belongs to another candidate")
        res = json.loads((Path(src["cell"]) / "result.json").read_text())
        if res["experiment_digest"] != src["experiment_digest"] or any(res["artifacts"].get(r) != a for r, a in src["artifacts"].items()):
            problems.append(f"source {src['cell']} changed since the reference was built")
        if src["runs"] != [f"run{i}" for i in range(1, int(v.spec["k"]) + 1)]:
            problems.append(f"source {src['cell']}: reference must use every measured run")
        for run in src["runs"]:
            for r in _jsonl(Path(src["cell"]) / run / "requests.jsonl"):
                durations.setdefault(r["request_id"], []).append(float(r["duration_s"]))
    recomputed = {k: float(np.mean(v)) for k, v in sorted(durations.items())}
    if recomputed != doc.get("rows"):
        problems.append("per-row durations do not match the source cells")
    return problems, (doc.get("rows") or {}) if not problems else {}
