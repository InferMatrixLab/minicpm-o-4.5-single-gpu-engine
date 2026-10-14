"""Closed-loop load protocol, run validity and sweep planning.

Closed loop at fixed concurrency C: requests go out in a fixed ID order, a new
request is dispatched as soon as a slot frees, with no think time.  A cell is
one (engine, config, C) on a fresh server: warmup (2C disjoint IDs), run 0
(discarded), then K measured runs over the evaluation IDs.
"""

from __future__ import annotations

import asyncio
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
from bench.workload import CanonicalRequest


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


@dataclass
class RunResult:
    label: str
    concurrency: int
    measured: bool
    metrics: RunMetrics
    per_request: list[RequestMetrics]
    traces: list[RequestTrace] = field(repr=False, default_factory=list)

    def summary(self) -> dict:
        return {"label": self.label, "concurrency": self.concurrency, "measured": self.measured, **self.metrics.to_json()}


def evaluate_run(label: str, concurrency: int, traces: list[RequestTrace], measured: bool = True) -> RunResult:
    per = [request_metrics(t) for t in traces]
    return RunResult(label=label, concurrency=concurrency, measured=measured, metrics=run_metrics(per), per_request=per, traces=traces)


def save_run(run: RunResult, out_dir: Path, save_audio: bool = True) -> None:
    """Raw traces are kept for every run, valid or not."""
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.json").write_text(json.dumps(run.summary(), indent=1))
    with (out_dir / "requests.jsonl").open("w") as f:
        for m in run.per_request:
            f.write(json.dumps(asdict(m)) + "\n")
    with (out_dir / "traces.jsonl").open("w") as f:
        for t in run.traces:
            f.write(json.dumps(t.to_json()) + "\n")
    if save_audio:
        wav_dir = out_dir / "wav"
        wav_dir.mkdir(exist_ok=True)
        for t in run.traces:
            pcm = t.audio.pcm()
            if pcm.size:
                (wav_dir / f"{t.request_id}.wav").write_bytes(encode_wav_pcm16(pcm, t.audio.sample_rate or 24000))


@dataclass
class CellPlan:
    engine: str
    config: str
    concurrency: int


def plan_sweep(cells: list[CellPlan], seed: int) -> list[CellPlan]:
    """Randomized run order; the seed goes into the manifest."""
    order = list(cells)
    random.Random(seed).shuffle(order)
    return order


async def run_cell(
    base_url: str,
    adapter: Adapter,
    warmup: list[CanonicalRequest],
    evaluation: list[CanonicalRequest],
    concurrency: int,
    k_measured: int,
    out_dir: Path,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    between_runs: Callable[[int], Awaitable[None]] | None = None,
    save_audio: bool = True,
    log: Callable[[str], None] = print,
) -> list[RunResult]:
    """Warmup, discarded run 0, then ``k_measured`` runs.  Every run is saved."""
    eval_ids = [r.request_id for r in evaluation]
    if len(set(eval_ids)) != len(eval_ids):
        raise ValueError("measured IDs repeat within a run")
    if set(eval_ids) & {r.request_id for r in warmup}:
        raise ValueError("warmup IDs must be disjoint from evaluation IDs")
    results: list[RunResult] = []
    async with make_session(connections=max(64, 4 * concurrency)) as session:
        t0 = time.perf_counter()
        wtr = await closed_loop(session, base_url, adapter, warmup, concurrency, timeout_s)
        warm = evaluate_run("warmup", concurrency, wtr, measured=False)
        save_run(warm, out_dir / "warmup", save_audio=False)
        log(f"[C={concurrency}] warmup {len(wtr)} reqs in {time.perf_counter() - t0:.1f}s valid={warm.metrics.valid}")
        for k in range(k_measured + 1):
            if between_runs is not None:
                await between_runs(k)
            trs = await closed_loop(session, base_url, adapter, evaluation, concurrency, timeout_s)
            run = evaluate_run(f"run{k}", concurrency, trs, measured=k > 0)
            save_run(run, out_dir / f"run{k}", save_audio=save_audio)
            m = run.metrics
            log(
                f"[C={concurrency}] run{k}{' (discarded)' if k == 0 else ''}: valid={m.valid} "
                f"fps={m.throughput_frames_s:.1f} ttfa50={m.ttfa_p50_s * 1e3:.0f}ms ttfa95={m.ttfa_p95_s * 1e3:.0f}ms "
                f"rtf={m.rtf_mean:.3f} fails={m.n_failed} {m.invalid_reasons or ''}"
            )
            results.append(run)
    return results


def measured_runs(results: list[RunResult]) -> list[RunResult]:
    return [r for r in results if r.measured]


def cell_valid(results: list[RunResult]) -> bool:
    """A cell is usable only if every measured run is valid; invalid runs are reported, never dropped."""
    runs = measured_runs(results)
    return bool(runs) and all(r.metrics.valid for r in runs)
