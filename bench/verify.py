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


def verify_cell(cell_dir: str | Path, current_spec: dict | None = None) -> CellVerification:
    cell_dir = Path(cell_dir)
    v = CellVerification(cell_dir=cell_dir, ok=False)
    try:
        res = json.loads((cell_dir / "result.json").read_text())
    except (OSError, json.JSONDecodeError) as e:
        v.problems.append(f"result.json unreadable: {e}")
        return v
    spec = res.get("experiment") or {}
    v.spec = spec
    if not spec:
        v.problems.append("result has no experiment spec")
        return v
    if spec_digest(spec) != res.get("experiment_digest") or spec_base_digest(spec) != res.get("experiment_base_digest"):
        v.problems.append("stored experiment digests do not match the stored spec")
    if current_spec is not None and spec_base_digest(current_spec) != spec_base_digest(spec):
        v.problems.append("current execution fingerprints/spec differ from the frozen experiment")
    m = res.get("manifest") or {}
    v.problems += [f"manifest: {e}" for e in validate_manifest(m)]
    if (m.get("deploy") or {}).get("experiment") != spec:
        v.problems.append("manifest experiment spec differs from result spec")
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
