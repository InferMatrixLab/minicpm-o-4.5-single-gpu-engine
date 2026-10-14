"""Closed-loop load protocol, run validity and sweep planning.

Closed loop at fixed concurrency C: requests go out in a fixed ID order, a new
request is dispatched as soon as a slot frees, with no think time.  A cell is
one (engine, config, C): warmup (2C disjoint IDs), run 0 (discarded), then K
measured runs over the evaluation IDs.

Caches must be cold at the start of every measured run.  ``cache_policy``:

* ``restart``  - a fresh server for every run (warmup on disjoint IDs first);
* ``flush``    - one server, and a reset hook that returns evidence of an
                 empty reference/prefix cache before each run;
* ``none``     - development smoke only; the cell is marked non-compliant.
"""

from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

import aiohttp

from bench.adapters.base import Adapter
from bench.audio import encode_wav_pcm16
from bench.client import DEFAULT_TIMEOUT_S, make_session, send_request
from bench.metrics import RequestMetrics, RequestTrace, RunMetrics, request_metrics, run_metrics
from bench.workload import CONCURRENCIES, CanonicalRequest, n_eval

CACHE_POLICIES = ("restart", "flush", "none")


async def closed_loop(
    session: aiohttp.ClientSession,
    base_url: str,
    adapter: Adapter,
    requests: list[CanonicalRequest],
    concurrency: int,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    record_raw: bool = False,
) -> list[RequestTrace]:
    """Run ``requests`` in order with ``concurrency`` slots; returns traces in request order."""
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    payloads = [adapter.build_payload(r) for r in requests]  # client-side work stays out of timing
    traces: list[RequestTrace | None] = [None] * len(requests)
    next_idx = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal next_idx
        while True:
            async with lock:
                if next_idx >= len(requests):
                    return
                i = next_idx
                next_idx += 1
            traces[i] = await send_request(
                session, base_url, adapter, requests[i], timeout_s=timeout_s, payload=payloads[i], record_raw=record_raw
            )

    await asyncio.gather(*(worker() for _ in range(min(concurrency, len(requests)))))
    return [t for t in traces if t is not None]


def ids_sha256(ids: list[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()


@dataclass
class RunMeta:
    gpu: str = ""
    engine: str = ""
    config: str = ""
    candidate_id: str = ""
    config_digest: str = ""  # candidate digest: adapter, config, args, env, deploy files
    cell_digest: str = ""  # candidate + C + workload + K + timeout
    experiment_base_digest: str = ""  # frozen experiment spec without its run schedule
    mps_policy: str = "off"  # declared; checked against runtime evidence
    id_set: str = "eval"
    ids_sha256: str = ""  # of the IDs actually measured in this run
    expected_ids_sha256: str = ""
    seed_index: int = 0
    timeout_s: float = 0.0
    strict: bool = False  # mandatory geometry was enforced
    cache_policy: str = "none"
    cache_evidence: dict = field(default_factory=dict)
    runtime_evidence: dict = field(default_factory=dict)  # launch argv/env and per-run MPS evidence
    server_startup_s: float | None = None


@dataclass
class RunResult:
    label: str
    concurrency: int
    measured: bool
    metrics: RunMetrics
    per_request: list[RequestMetrics]
    traces: list[RequestTrace] = field(repr=False, default_factory=list)
    meta: RunMeta = field(default_factory=RunMeta)
    artifacts: dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {"label": self.label, "concurrency": self.concurrency, "measured": self.measured, **asdict(self.meta), **self.metrics.to_json()}


def evaluate_run(label: str, concurrency: int, traces: list[RequestTrace], measured: bool = True, meta: RunMeta | None = None) -> RunResult:
    per = [request_metrics(t) for t in traces]
    return RunResult(
        label=label, concurrency=concurrency, measured=measured, metrics=run_metrics(per), per_request=per, traces=traces, meta=meta or RunMeta()
    )


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def run_artifact_hashes(out_dir: Path) -> dict:
    """Content hashes of one run's artifacts; WAVs are covered by a manifest hash."""
    wav_dir = out_dir / "wav"
    wavs = sorted(wav_dir.glob("*.wav")) if wav_dir.is_dir() else []
    manifest = hashlib.sha256("\n".join(f"{p.name} {_sha(p)}" for p in wavs).encode()).hexdigest()
    return {
        "summary.json": _sha(out_dir / "summary.json"),
        "requests.jsonl": _sha(out_dir / "requests.jsonl"),
        "traces.jsonl": _sha(out_dir / "traces.jsonl"),
        "wire.jsonl.gz": _sha(out_dir / "wire.jsonl.gz"),
        "wav_count": len(wavs),
        "wav_manifest_sha256": manifest,
    }


def save_run(run: RunResult, out_dir: Path, save_audio: bool = True) -> dict:
    """Raw traces are kept for every run, valid or not.  Returns the artifacts' content hashes."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(run.summary(), indent=1))
    with (out_dir / "requests.jsonl").open("w") as f:
        for m in run.per_request:
            f.write(json.dumps(asdict(m)) + "\n")
    with (out_dir / "traces.jsonl").open("w") as f:
        for t in run.traces:
            f.write(json.dumps(t.to_json()) + "\n")
    raw = [(t, getattr(t, "raw_chunks", None)) for t in run.traces]
    if any(r is not None for _, r in raw):
        with gzip.open(out_dir / "wire.jsonl.gz", "wt") as f:
            for t, chunks in raw:
                rec = {
                    "request_id": t.request_id,
                    "t_dispatch": t.t_dispatch,
                    "chunks": [[c_t - t.t_dispatch, base64.b64encode(b).decode()] for c_t, b in (chunks or [])],
                }
                f.write(json.dumps(rec) + "\n")
    if save_audio:
        wav_dir = out_dir / "wav"
        wav_dir.mkdir(exist_ok=True)
        for t in run.traces:
            pcm = t.audio.pcm()
            if pcm.size:
                (wav_dir / f"{t.request_id}.wav").write_bytes(encode_wav_pcm16(pcm, t.audio.sample_rate or 24000))
    run.artifacts = run_artifact_hashes(out_dir)
    return run.artifacts


@dataclass
class CellPlan:
    engine: str
    config: str
    concurrency: int


def plan_sweep(cells: list, seed: int) -> list:
    """Randomized run order; the seed and the executed order go into the sweep record."""
    order = list(cells)
    random.Random(seed).shuffle(order)
    return order


def check_cell_geometry(warmup: list[CanonicalRequest], evaluation: list[CanonicalRequest], concurrency: int, k_measured: int, id_set: str, strict: bool) -> None:
    eval_ids = [r.request_id for r in evaluation]
    if len(set(eval_ids)) != len(eval_ids):
        raise ValueError("measured IDs repeat within a run")
    # full-set quality generation measures every row, so its warmup necessarily overlaps;
    # those cells are quality experiments, never timing evidence
    if id_set != "full" and set(eval_ids) & {r.request_id for r in warmup}:
        raise ValueError("warmup IDs must be disjoint from evaluation IDs")
    if id_set == "full" and strict and len(evaluation) != 1088:
        raise ValueError(f"full-set quality generation needs all 1088 rows, got {len(evaluation)}")
    if k_measured < 1:
        raise ValueError("at least one measured run is required")
    if not strict:
        return
    if concurrency not in CONCURRENCIES:
        raise ValueError(f"C={concurrency} is not a mandatory concurrency {CONCURRENCIES}")
    if len(warmup) != 2 * concurrency:
        raise ValueError(f"warmup must be 2C={2 * concurrency} requests, got {len(warmup)}")
    if id_set == "eval" and len(evaluation) != n_eval(concurrency):
        raise ValueError(f"evaluation must be N_eval={n_eval(concurrency)} requests, got {len(evaluation)}")


ServerFactory = Callable[[int], object]  # run index -> context manager yielding an object with .url and .startup_s
FlushHook = Callable[[int], Awaitable[dict]]  # run index -> evidence that caches are empty


async def run_cell(
    base_url: str | None,
    adapter: Adapter,
    warmup: list[CanonicalRequest],
    evaluation: list[CanonicalRequest],
    concurrency: int,
    k_measured: int,
    out_dir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    cache_policy: str = "none",
    server_factory: ServerFactory | None = None,
    flush: FlushHook | None = None,
    meta: RunMeta | None = None,
    save_audio: bool = True,
    record_raw: bool = True,
    strict: bool = True,
    run_indices: list[int] | None = None,
    log: Callable[[str], None] = print,
) -> list[RunResult]:
    """Warmup, discarded run 0, then ``k_measured`` runs.  Every run is saved.

    ``run_indices`` limits execution to those runs (a declared K extension runs
    only the new indices; each still gets a fresh server and its own warmup).
    """
    if cache_policy not in CACHE_POLICIES:
        raise ValueError(f"cache_policy must be one of {CACHE_POLICIES}")
    if cache_policy == "restart" and server_factory is None:
        raise ValueError("cache_policy=restart needs a server_factory")
    if cache_policy == "flush" and (flush is None or base_url is None):
        raise ValueError("cache_policy=flush needs a flush hook and a base_url")
    if cache_policy != "restart" and base_url is None:
        raise ValueError("base_url is required unless servers are restarted per run")
    check_cell_geometry(warmup, evaluation, concurrency, k_measured, (meta.id_set if meta else "eval"), strict)
    base_meta = meta or RunMeta()
    base_meta.expected_ids_sha256 = ids_sha256([r.request_id for r in evaluation])
    base_meta.cache_policy = cache_policy
    base_meta.timeout_s = timeout_s
    base_meta.strict = strict
    results: list[RunResult] = []

    async def one_run(k: int, url: str, startup_s: float | None, do_warmup: bool, server_evidence: dict | None = None) -> None:
        async with make_session(connections=max(64, 4 * concurrency)) as session:
            if do_warmup:
                t0 = time.perf_counter()
                wtr = await closed_loop(session, url, adapter, warmup, concurrency, timeout_s)
                warm = evaluate_run("warmup", concurrency, wtr, measured=False, meta=base_meta)
                save_run(warm, out_dir / (f"run{k}_warmup" if cache_policy == "restart" else "warmup"), save_audio=False)
                log(f"[C={concurrency}] warmup {len(wtr)} reqs in {time.perf_counter() - t0:.1f}s valid={warm.metrics.valid}")
            evidence: dict = {}
            if cache_policy == "flush":
                evidence = await flush(k)  # type: ignore[misc]
                if not evidence.get("empty", False):
                    raise RuntimeError(f"cache flush before run{k} not verified: {evidence}")
            elif cache_policy == "restart":
                evidence = {"fresh_server": True, **(server_evidence or {})}
            trs = await closed_loop(session, url, adapter, evaluation, concurrency, timeout_s, record_raw=record_raw)
            run_meta = RunMeta(
                **{
                    **asdict(base_meta),
                    "ids_sha256": ids_sha256([t.request_id for t in trs]),
                    "cache_evidence": evidence,
                    "runtime_evidence": dict(server_evidence or {}),
                    "server_startup_s": startup_s,
                }
            )
        run = evaluate_run(f"run{k}", concurrency, trs, measured=k > 0, meta=run_meta)
        save_run(run, out_dir / f"run{k}", save_audio=save_audio)
        m = run.metrics
        log(
            f"[C={concurrency}] run{k}{' (discarded)' if k == 0 else ''}: valid={m.valid} "
            f"fps={m.throughput_frames_s:.1f} ttfa50={m.ttfa_p50_s * 1e3:.0f}ms ttfa95={m.ttfa_p95_s * 1e3:.0f}ms "
            f"rtf={m.rtf_mean:.3f} fails={m.n_failed} {m.invalid_reasons or ''}"
        )
        results.append(run)

    indices = list(range(k_measured + 1)) if run_indices is None else list(run_indices)
    if any(i < 0 or i > k_measured for i in indices):
        raise ValueError(f"run indices {indices} outside run0..run{k_measured}")
    if run_indices is not None and cache_policy != "restart":
        raise ValueError("partial run schedules need a fresh server per run")
    for k in indices:
        if cache_policy == "restart":
            with server_factory(k) as srv:  # type: ignore[misc]
                ev = srv.evidence() if hasattr(srv, "evidence") else {}
                await one_run(k, srv.url, getattr(srv, "startup_s", None), do_warmup=True, server_evidence=ev)
        else:
            await one_run(k, base_url, None, do_warmup=(k == 0))  # type: ignore[arg-type]
    return results


def measured_runs(results: list[RunResult]) -> list[RunResult]:
    return [r for r in results if r.measured]


def cell_valid(results: list[RunResult]) -> bool:
    """A cell is usable only if every measured run is valid; invalid runs are reported, never dropped."""
    runs = measured_runs(results)
    return bool(runs) and all(r.metrics.valid for r in runs)


def cell_compliant(results: list[RunResult], expected_k: int) -> bool:
    """In-memory check against the declared K; reported verdicts come from ``bench.verify.verify_cell``."""
    from bench.stats import StatisticsInputError, validate_cell_runs

    runs = measured_runs(results)
    if not cell_valid(results):
        return False
    try:
        validate_cell_runs([r.summary() for r in runs], expected_k=expected_k, side="cell")
    except StatisticsInputError:
        return False
    return True
