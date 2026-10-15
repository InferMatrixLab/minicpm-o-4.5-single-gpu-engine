"""Verify a measured cell from its artifacts, never from stored verdicts.

``verify_cell(dir)`` recomputes the experiment-spec digests, reconciles the
manifest with the spec, and for every declared run (run0..runK) checks the
recorded artifact hashes, the ordered request IDs, per-request validity,
trace/wire/WAV coverage and WAV sample counts, recomputes the run metrics from
the per-request records, and finally applies the run validator with the K the
spec declares.  ``ok`` is the only compliance verdict anything should trust.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import soundfile as sf

from bench.experiment import spec_base_digest, spec_digest
from bench.manifest import validate_manifest
from bench.metrics import RequestMetrics, run_metrics
from bench.protocol import ids_sha256, run_artifact_hashes

RUN_METRIC_KEYS = ("n_requests", "n_failed", "valid", "throughput_frames_s", "ttfa_p50_s", "ttfa_p95_s", "rtf_mean", "total_audio_s")


@dataclass
class CellVerification:
    cell_dir: Path
    ok: bool
    problems: list[str] = field(default_factory=list)
    spec: dict = field(default_factory=dict)
    measured_summaries: list[dict] = field(default_factory=list)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _close(a, b) -> bool:
    if isinstance(a, float) or isinstance(b, float):
        if a is None or b is None:
            return a is b
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _verify_run(run_dir: Path, label: str, spec: dict, recorded: dict, artifacts: dict | None, problems: list[str]) -> dict | None:
    p = lambda msg: problems.append(f"{label}: {msg}")  # noqa: E731
    if not run_dir.is_dir():
        p("run directory missing")
        return None
    if not artifacts:
        p("no recorded artifact hashes")
    else:
        now = run_artifact_hashes(run_dir)
        for k, v in artifacts.items():
            if now.get(k) != v:
                p(f"artifact {k} changed or missing (recorded {v}, found {now.get(k)})")
    try:
        summary = json.loads((run_dir / "summary.json").read_text())
        reqs = _jsonl(run_dir / "requests.jsonl")
        traces = _jsonl(run_dir / "traces.jsonl")
    except (OSError, json.JSONDecodeError) as e:
        p(f"unreadable artifacts: {e}")
        return None
    if summary != recorded:
        p("summary.json differs from the run summary in result.json")
    expect = {
        "concurrency": spec["concurrency"],
        "seed_index": spec["seed_index"],
        "timeout_s": spec["timeout_s"],
        "id_set": spec["id_set"],
        "candidate_id": spec["candidate"]["candidate_id"],
        "config_digest": spec["candidate"]["digest"],
        "mps_policy": spec["candidate"]["mps"],
        "gpu": spec["hardware"].get("gpu_sku"),
        "strict": spec["strict"],
        "cache_policy": spec["cache_policy"],
    }
    for k, want in expect.items():
        if summary.get(k) != want:
            p(f"run {k}={summary.get(k)!r} contradicts the spec ({want!r})")
    if spec["cache_policy"] == "restart":
        gpu_ev = ((summary.get("runtime_evidence") or {}).get("gpu") or {})
        if gpu_ev.get("gpu_uuid") != spec["hardware"].get("gpu_uuid"):
            p("run executed on a different GPU than the spec declares")
    ids = [r.get("request_id") for r in reqs]
    if ids != spec["measured_ids"]:
        p(f"request records ({len(ids)}) do not match the declared ordered IDs ({len(spec['measured_ids'])})")
    if [t.get("request_id") for t in traces] != ids:
        p("trace records do not match the request records")
    if summary.get("ids_sha256") != ids_sha256(ids) or summary.get("expected_ids_sha256") != ids_sha256(spec["measured_ids"]):
        p("run ID hashes do not match the request records and spec")
    try:
        per = [RequestMetrics(**r) for r in reqs]
        rm = run_metrics(per).to_json() if per else None
    except (TypeError, ValueError) as e:
        p(f"request records malformed: {e}")
        rm = None
    if rm is not None:
        for k in RUN_METRIC_KEYS:
            if not _close(rm[k], summary.get(k)):
                p(f"summary {k}={summary.get(k)} does not match the request records ({rm[k]})")
    if spec.get("record_raw"):
        wire = run_dir / "wire.jsonl.gz"
        try:
            with gzip.open(wire, "rt") as f:
                wire_ids = [json.loads(line)["request_id"] for line in f if line.strip()]
            if wire_ids != ids:
                p("wire capture does not cover the request records")
        except (OSError, json.JSONDecodeError, KeyError, EOFError) as e:
            p(f"wire capture missing or unreadable: {e}")
    if spec.get("save_audio"):
        for r in reqs:
            if r.get("duration_s", 0) <= 0:
                continue
            w = run_dir / "wav" / f"{r['request_id']}.wav"
            try:
                info = sf.info(str(w))
            except Exception:
                p(f"audio missing for {r['request_id']}")
                continue
            if abs(info.frames / info.samplerate - r["duration_s"]) > 1.0 / info.samplerate + 1e-9:
                p(f"audio length of {r['request_id']} ({info.frames / info.samplerate:.4f}s) != recorded duration {r['duration_s']:.4f}s")
    return summary


def reconcile_manifest(m: dict, spec: dict) -> list[str]:
    """Every identity the manifest duplicates must agree with the authoritative spec; digests are recomputed."""
    from bench.experiment import Candidate, CellSpec, digest

    out = []
    d = m.get("deploy") or {}
    if d.get("experiment") != spec or d.get("experiment_digest") != spec_digest(spec):
        out.append("deploy.experiment differs from the declared spec")
    c = spec["candidate"]
    ident = {k: c[k] for k in ("adapter", "config", "extra", "env", "deploy_files_sha256", "mps")}
    if digest(ident) != c.get("digest"):
        out.append("candidate digest does not match the candidate identity")
    mc = d.get("candidate") or {}
    for k in ("adapter", "config", "mps", "candidate_id", "digest"):
        want = c[k] if k in c else None
        got = mc.get(k)
        if k in ("extra",) or got != want:
            out.append(f"deploy.candidate.{k}={got!r} != spec {want!r}")
    if [list(x) for x in mc.get("env", [])] != [list(x) for x in c["env"]] or list(mc.get("extra", [])) != list(c["extra"]):
        out.append("deploy.candidate launch arguments/environment differ from the spec")
    cand = Candidate(adapter=c["adapter"], config=c["config"], extra=tuple(c["extra"]), env=tuple(tuple(x) for x in c["env"]), label=c.get("label", ""),
                     deploy_files_sha256=tuple(c["deploy_files_sha256"]), mps=c["mps"])
    from bench.protocol import ids_sha256 as _ids

    workload = {"dataset": spec["dataset"]["repo"], "revision": spec["dataset"]["revision"], "warmup_ids_sha256": _ids(spec["warmup_ids"]),
                "measured_ids_sha256": _ids(spec["measured_ids"])}
    cell = CellSpec(cand, spec["concurrency"], spec["k"], spec["id_set"], spec["seed_index"], spec["timeout_s"], spec["hardware"]["gpu_index"], workload)
    mcell = d.get("cell") or {}
    if mcell.get("digest") != cell.digest:
        out.append("deploy.cell digest does not match the cell recomputed from the spec")
    for k in ("concurrency", "id_set", "seed_index", "timeout_s", "gpu_index"):
        if mcell.get(k) != getattr(cell, k):
            out.append(f"deploy.cell.{k}={mcell.get(k)!r} != spec {getattr(cell, k)!r}")
    if mcell.get("candidate") != cand.identity() or mcell.get("workload") != workload:
        out.append("deploy.cell candidate/workload differ from the spec")
    if (m.get("model") or {}).get("revision") != spec["model"]["revision"] or (m.get("model") or {}).get("repo") != spec["model"]["repo"]:
        out.append("manifest model differs from the spec")
    for name, want in (spec["fingerprints"].get("repos") or {}).items():
        got = (m.get("repos") or {}).get(name) or {}
        if got.get("commit") != want.get("commit") or bool(got.get("dirty")) != bool(want.get("dirty")):
            out.append(f"manifest repo {name} differs from the spec fingerprint")
    ds = m.get("datasets") or [{}]
    if ds[0].get("repo") != spec["dataset"]["repo"] or ds[0].get("revision") != spec["dataset"]["revision"]:
        out.append("manifest dataset differs from the spec")
    if list((ds[0].get("ids") or {}).get(spec["id_set"], [])) != spec["measured_ids"]:
        out.append("manifest measured ID list differs from the spec")
    if list((ds[0].get("ids") or {}).get("warmup", [])) != spec["warmup_ids"]:
        out.append("manifest warmup ID list differs from the spec")
    if m.get("sampling") != spec["sampling"]:
        out.append("manifest sampling differs from the spec")
    hw = m.get("hardware") or {}
    sh = spec["hardware"]
    try:
        mem_ok = float(hw.get("gpu_memory_mib")) == float(sh.get("memory_mib"))
    except (TypeError, ValueError):
        mem_ok = False
    if hw.get("gpu_index") != sh.get("gpu_index") or (hw.get("gpu_sku") or "").replace("NVIDIA ", "") != sh.get("gpu_sku") \
            or hw.get("driver") != sh.get("driver") or not mem_ok:
        out.append("manifest hardware (index, SKU, driver, memory) differs from the spec")
    locks = m.get("env_lock") or {}
    for name, want in (spec["fingerprints"].get("envs") or {}).items():
        lock = locks.get(name)
        if lock is None:
            out.append(f"manifest has no lock for the required environment {name}")
            continue
        try:
            body = Path(lock["path"]).read_bytes()
        except OSError:
            out.append(f"environment lock {name} is missing")
            continue
        if hashlib.sha256(body).hexdigest() != want or lock.get("sha256") != want:
            out.append(f"environment {name} at measurement time differs from the spec fingerprint")
    return out


def verify_cell(cell_dir: str | Path, current_spec: dict | None = None) -> CellVerification:
    cell_dir = Path(cell_dir)
    v = CellVerification(cell_dir=cell_dir, ok=False)
    try:
        res = json.loads((cell_dir / "result.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        v.problems.append(f"result.json unreadable: {e}")
        return v
    try:
        declared = json.loads((cell_dir / "experiment.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        v.problems.append(f"experiment.json (the authoritative declaration) unreadable: {e}")
        return v
    spec = declared.get("experiment") or {}
    v.spec = spec
    if not spec:
        v.problems.append("declaration has no experiment spec")
        return v
    if spec_digest(spec) != declared.get("experiment_digest") or spec_base_digest(spec) != declared.get("experiment_base_digest"):
        v.problems.append("declared experiment digests do not match the declared spec")
    if res.get("experiment") != spec or res.get("experiment_digest") != declared.get("experiment_digest"):
        v.problems.append("result is stale relative to the declared experiment (an interrupted or unrecorded schedule change)")
    if current_spec is not None and spec_base_digest(current_spec) != spec_base_digest(spec):
        v.problems.append("current execution fingerprints/spec differ from the frozen experiment")
    m = res.get("manifest") or {}
    v.problems += [f"manifest: {e}" for e in validate_manifest(m)]
    v.problems += [f"manifest: {e}" for e in reconcile_manifest(m, spec)]
    k = int(spec.get("k", -1))
    runs = {r.get("label"): r for r in res.get("runs", [])}
    want = [f"run{i}" for i in range(k + 1)]
    if sorted(runs) != sorted(want):
        v.problems.append(f"result runs {sorted(runs)} != declared schedule {want}")
    measured = []
    for label in want:
        if label not in runs:
            continue
        s = _verify_run(cell_dir / label, label, spec, runs[label], (res.get("artifacts") or {}).get(label), v.problems)
        if s is None:
            continue
        if s.get("experiment_base_digest") != spec_base_digest(spec):
            v.problems.append(f"{label}: run belongs to a different experiment")
        if label != "run0":
            measured.append(s)
    v.measured_summaries = measured
    if not v.problems:
        from bench.stats import StatisticsInputError, validate_cell_runs

        try:
            validate_cell_runs(measured, expected_k=k, side="cell")
        except StatisticsInputError as e:
            v.problems.append(str(e))
    v.ok = not v.problems
    return v
