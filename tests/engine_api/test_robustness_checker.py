"""The AC-6 robustness checker (tools/robustness.py) against the real EngineServer with scripted backends (CPU).

Faults it must reject: an SSE error followed by [DONE], overlapping audio, a reply without audio, a cancellation the
engine does not drain or that received no audio, frame counts that never reach a window boundary, omitted / null / stale
counters, nonfinite / negative / future values, a changed feature-process inventory, a leaked CNN row, live CUDA
allocations that do not return although reserved memory is unchanged, an unreadable feature-process RSS, a served GPU
other than the declared one, a GPU reading missing in the middle of the load, a memory excursion that recovers by the
end, and a sustained load that does not keep every client busy across row-cycle boundaries.  A clean backend passes,
and so does a fixed warmed reservation.  A full run's saved artifacts replay to the same results; edits are caught.
"""

import asyncio
import sys
from pathlib import Path

import numpy as np
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import tools.robustness as rb  # noqa: E402
from bench.adapters import get_adapter  # noqa: E402
from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets  # noqa: E402
from engine.server import AudioChunk, Done, EngineServer, TextDelta  # noqa: E402

SR = 24000


class Backend:
    """mode: ok | error_after_audio | overlap | no_audio | stuck (never drains); frames: audio_frames per reply.
    stats_fault: None | omit (no CNN-row counter) | null (null stream counter) | stale (old worker counters) |
    leak_cnn (one CNN row lost per request) | child_missing (one feature process's RSS unreadable) |
    alloc_leak (32 MiB of live CUDA tensors kept per request inside an unchanged reservation) |
    warm_reserved (a large fixed reservation; allocations return) | spike (engine RSS +5% for a few readings)."""

    def __init__(self, mode="ok", frames=None, stats_fault=None, delay=0.01, procs=2):
        self.mode, self.frames, self.n, self.live, self.fault, self.delay = mode, frames or [33], 0, 0, stats_fault, delay
        self.reads, self.procs = 0, procs

    def stats(self):
        import time

        self.reads += 1
        spike = self.fault == "spike" and 6 <= self.reads <= 8
        m = {"admitted": 1 if self.mode == "stuck" else self.live, "live": self.live, "queued": 0, "max_live": 64, "thinker_kv_pages_used": 0,
             "talker_kv_pages_used": 0, "thinker_kv_pages": 8192, "talker_kv_pages": 8192, "first_chunk": 8, "audio_live": 0, "t2w_requests": 0,
             "t2w_streams": 0, "t2w_cache_slots_used": 0, "t2w_cnn_rows_free": 64 - (self.n if self.fault == "leak_cnn" else 0),
             "t2w_dit_eager_calls": self.n, "rss_kib": 1050 if spike else 1000, "t2w_rss_kib": 2000, "feature_processes": self.procs,
             "feature_rss_by_pid": {str(11 + i): None if self.fault == "child_missing" and i == 1 else 500 for i in range(self.procs)},
             "feature_rss_kib": None if self.fault == "child_missing" else 500 * self.procs,
             "gpu_memory_allocated_bytes": 10 << 30, "gpu_memory_reserved_bytes": (40 if self.fault == "warm_reserved" else 20) << 30,
             "t2w_gpu_memory_allocated_bytes": (5 << 30) + (self.n * (32 << 20) if self.fault == "alloc_leak" else 0),
             "t2w_gpu_memory_reserved_bytes": 30 << 30, "t2w_stats_time": time.time() - (60 if self.fault == "stale" else 0)}
        if self.fault == "omit":
            del m["t2w_cnn_rows_free"]
        if self.fault == "null":
            m["t2w_streams"] = None
        return m

    async def generate(self, req):
        self.n += 1
        self.live += 1
        frames = self.frames[(self.n - 1) % len(self.frames)]  # this request's frame count, fixed at its start
        try:
            yield TextDelta("Hello.")
            x = (0.2 * np.sin(np.arange(SR // 4) * 0.05)).astype(np.float32)
            if self.mode != "no_audio":
                yield AudioChunk(x, 0)
                await asyncio.sleep(self.delay)
                if self.mode == "error_after_audio":
                    raise RuntimeError("engine error: boom")
                yield AudioChunk(x, len(x) // 2 if self.mode == "overlap" else len(x))
            yield Done("stop", {"prompt_tokens": 5, "completion_tokens": 2, "audio_frames": frames})
        finally:
            self.live -= 1


def _run(backend, fn):
    async def go():
        srv = TestServer(EngineServer(backend).app)
        await srv.start_server()
        try:
            return await fn(f"http://127.0.0.1:{srv.port}")
        finally:
            await srv.close()
    return asyncio.run(go())


def _reqs(n):
    rows = load_seed_tts_rows()
    return canonical_requests(rows, make_id_sets(rows).tuning(32)[:n])


def test_mixed_passes_clean_and_rejects_faults():
    a = get_adapter("engine")
    assert _run(Backend("ok"), lambda u: rb.scenario_mixed(u, a, n=4))["passed"]
    for mode in ("error_after_audio", "overlap", "no_audio"):
        res = _run(Backend(mode), lambda u: rb.scenario_mixed(u, a, n=4))
        assert not res["passed"], (mode, res)
        assert res["invalid_reasons"], mode


def test_cancel_requires_observed_cancellations_and_a_drained_engine():
    a = get_adapter("engine")
    payloads = [a.build_payload(r) for r in _reqs(4)]
    ok = _run(Backend("ok"), lambda u: rb.scenario_cancel(u, a, payloads))
    assert ok["passed"] and ok["cancelled_observed"] >= 2 and ok["cancelled_with_audio"] == 2, ok
    stuck = _run(Backend("stuck"), lambda u: rb.scenario_cancel(u, a, payloads))
    assert not stuck["passed"] and stuck["idle_problems"], stuck


def test_cancel_before_audio_fails():
    """A cancelling client that never received audio did not test cancellation of a speaking stream."""
    a = get_adapter("engine")
    payloads = [a.build_payload(r) for r in _reqs(4)]
    res = _run(Backend("no_audio"), lambda u: rb.scenario_cancel(u, a, payloads))
    assert not res["passed"] and res["cancelled_with_audio"] == 0, res


def test_counter_schema_rejects_omitted_null_and_stale_counters():
    for fault, needle in (("omit", "t2w_cnn_rows_free missing"), ("null", "t2w_streams=None"), ("stale", "s old"),
                          ("child_missing", "feature process 12 RSS unreadable")):
        m = {**Backend(stats_fault=fault).stats(), "cancelled_streams": 0}  # the server adds its own counter
        probs = rb.idle_problems(m)
        assert any(needle in p for p in probs), (fault, probs)
    assert rb.idle_problems({**Backend().stats(), "cancelled_streams": 0}) == []
    m = {**Backend().stats(), "cancelled_streams": 0}
    del m["feature_rss_by_pid"]["12"]
    assert any("1 of 2 processes" in p for p in rb.schema_problems(m, expect=rb.Expect(2)))


def test_counter_schema_rejects_nonfinite_negative_future_and_inconsistent_values():
    import time

    ex = rb.Expect(feature_processes=2)
    clean = {**Backend().stats(), "cancelled_streams": 0}
    assert rb.schema_problems(clean, expect=ex) == []
    cases = {"t2w_stats_time": [float("nan"), float("inf"), time.time() + 30], "feature_processes": [float("nan"), -1, 1.5],
             "t2w_cnn_rows_free": [float("inf"), -3, 2.5], "rss_kib": [0, -5], "gpu_memory_allocated_bytes": [None, float("nan")]}
    for k, vals in cases.items():
        for v in vals:
            assert rb.schema_problems({**clean, k: v}, expect=ex), (k, v)
    assert any("config declares 4" in p for p in rb.schema_problems(clean, expect=rb.Expect(4)))  # count from the config
    assert any("per-PID sum" in p for p in rb.schema_problems({**clean, "feature_rss_kib": 999}, expect=ex))
    moved = {**clean, "feature_rss_by_pid": {"11": 500, "13": 500}}  # a replaced child
    assert any("PIDs changed" in p for p in rb.schema_problems(moved, expect=rb.Expect(2, frozenset({"11", "12"}))))
    assert any("gpu_memory_used_mib" in p for p in rb.schema_problems({**clean, "gpu_memory_used_mib": None}, expect=ex, sampled=True))


def test_drain_compares_capacities_with_the_warmed_baseline():
    """A leaked CNN row leaves every 'used' counter at zero; only the comparison with the warmed baseline sees it."""
    a = get_adapter("engine")
    payloads = [a.build_payload(r) for r in _reqs(4)]
    leak = Backend(stats_fault="leak_cnn")
    base = leak.stats()  # warmed idle baseline before the cancellations
    res = _run(leak, lambda u: rb.scenario_cancel(u, a, payloads, base))
    assert not res["idle_problems"] and any("t2w_cnn_rows_free" in p for p in res["baseline_problems"]) and not res["passed"], res
    clean = Backend()
    ok = _run(clean, lambda u: rb.scenario_cancel(u, a, payloads, clean.stats()))
    assert ok["passed"], ok


def test_gpu_check_requires_the_declared_served_device(monkeypatch):
    uuid = "GPU-e7921766-67c6-bc17-6df1-eca427957832"
    monkeypatch.setattr(rb, "gpu_query", lambda u: {"sku": "NVIDIA L20X", "uuid": u, "driver": "580", "index": 2, "memory_used_mib": 1.0} if u else None)
    ok = rb.gpu_check({"device_uuid": uuid}, uuid)
    assert ok["matches"] and ok["nvidia_smi"]["sku"] == "NVIDIA L20X", ok
    other = rb.gpu_check({"device_uuid": uuid}, "GPU-712b0057-e00a-8705-40e1-6ef88f1426c1")
    assert not other["matches"] and "!= declared" in other["problems"][0], other
    assert not rb.gpu_check({}, uuid)["matches"]
    assert not rb.gpu_check({"device_uuid": uuid}, None)["matches"]


def test_mismatched_gpu_stops_the_run_before_any_load(monkeypatch, tmp_path):
    from types import SimpleNamespace

    monkeypatch.setattr(rb, "gpu_query", lambda u: {"sku": "X", "uuid": u, "driver": "1", "index": 0, "memory_used_mib": 1.0} if u else None)
    be = Backend()
    cfg = str(ROOT / "configs" / "engine" / "track_s_fast.json")

    async def go():
        from engine.server import served_identity

        es = EngineServer(be)
        es.info = {**served_identity(cfg, "engine.runtime.backend_s:TrackSBackend"), "device_uuid": "GPU-a"}
        srv = TestServer(es.app)
        await srv.start_server()
        try:
            return await rb.run(SimpleNamespace(url=f"http://127.0.0.1:{srv.port}", out=str(tmp_path), config=cfg, gpu_uuid="GPU-b", acceptance=False,
                                                sustain_min=0.1, concurrency=2, overload=4, eos_n=4))
        finally:
            await srv.close()
    rep = asyncio.run(go())
    assert rep["identity"]["matches"] and not rep["gpu"]["matches"] and not rep["overall"] and "mixed" not in rep and be.n == 0, rep


def test_sustain_refills_continuously_across_row_cycles(monkeypatch):
    """Persistent clients: the in-flight count stays at C while the deadline is ahead, and the row sequence wraps."""
    monkeypatch.setattr(rb, "gpu_memory_used_mib", lambda u: 100.0 if u else None)
    a = get_adapter("engine")
    be = Backend(delay=0.05)
    res, samples, _ = _run(be, lambda u: rb.scenario_sustain(u, a, _reqs(3), 2, 0.05, "GPU-x", be.stats(), interval_s=0.2, ref_s=0.5, expect=rb.Expect(2)))
    assert res["passed"], res
    assert res["row_cycles"] > 2 and res["in_flight_min"] == res["in_flight_max"] == 2, res
    assert res["requests"] == res["requests_started"]
    leaky = Backend(delay=0.05, stats_fault="leak_cnn")
    bad, _, _ = _run(leaky, lambda u: rb.scenario_sustain(u, a, _reqs(3), 2, 0.05, "GPU-x", leaky.stats(), interval_s=0.2, ref_s=0.5, expect=rb.Expect(2)))
    assert not bad["passed"] and bad["baseline_problems"], bad
    gone = Backend(delay=0.05)
    nogpu, _, _ = _run(gone, lambda u: rb.scenario_sustain(u, a, _reqs(3), 2, 0.05, None, gone.stats(), interval_s=0.2, ref_s=0.5, expect=rb.Expect(2)))
    assert not nogpu["passed"] and "gpu_memory_used_mib" in nogpu["missing_memory"], nogpu


def test_eos_boundary_coverage_comes_from_observed_frame_counts():
    a = get_adapter("engine")
    reqs = _reqs(6)
    covered = _run(Backend("ok", frames=[32, 33, 34, 58, 40, 41]), lambda u: rb.scenario_eos_boundary(u, a, reqs, 8, concurrency=3))
    assert covered["passed"], covered  # (f - 8) % 25 hits 24, 0, 1
    missing = _run(Backend("ok", frames=[40, 41, 42]), lambda u: rb.scenario_eos_boundary(u, a, reqs, 8, concurrency=3))
    assert not missing["passed"], missing
    faulty = _run(Backend("error_after_audio", frames=[32, 33, 34]), lambda u: rb.scenario_eos_boundary(u, a, reqs, 8, concurrency=3))
    assert not faulty["passed"]


def test_overload_counts_only_valid_or_503():
    a = get_adapter("engine")
    res = _run(Backend("error_after_audio"), lambda u: rb.scenario_overload(u, a, _reqs(3)))
    assert not res["passed"] and res["other_failures"]


def test_served_identity_matches_the_harness_resolution():
    """``/info`` (what the server loaded) and ``bench.cli.resolve_engine_deploy`` (what the harness declares) agree, and the
    checker's identity check fails for another config."""
    from bench.cli import resolve_engine_deploy
    from engine.server import served_identity

    cfg = str(ROOT / "configs" / "engine" / "track_s_fast.json")
    served = served_identity(cfg)
    want = resolve_engine_deploy(cfg)
    assert served["config_sha256"] == want["config_sha256"] and served["engine_tree_sha256"] == want["engine_tree_sha256"]
    assert rb.identity_check({**served, "backend": "engine.runtime.backend_s:TrackSBackend"}, cfg)["matches"]
    assert not rb.identity_check(served, str(ROOT / "configs" / "engine" / "track_s_proc.json"))["matches"]


def _sustain(be, monkeypatch, gpu=lambda u: 100.0 if u else None, uuid="GPU-x"):
    monkeypatch.setattr(rb, "gpu_memory_used_mib", gpu)
    a = get_adapter("engine")
    return _run(be, lambda u: rb.scenario_sustain(u, a, _reqs(3), 2, 0.05, uuid, be.stats(), interval_s=0.2, ref_s=0.5, expect=rb.Expect(2)))[0]


def test_allocator_growth_inside_unchanged_reserved_memory_fails(monkeypatch):
    """Live CUDA tensors kept per request: reserved and device memory unchanged, every capacity counter back -- only the
    allocated-bytes comparison with the warmed baseline sees it."""
    res = _sustain(Backend(delay=0.05, stats_fault="alloc_leak"), monkeypatch)
    assert not res["passed"] and any("t2w_gpu_memory_allocated_bytes" in p for p in res["baseline_problems"]), res
    assert all(v["max_dev"] == 0.0 for k, v in res["memory_envelope"].items() if "reserved" in k), res["memory_envelope"]


def test_a_fixed_warmed_reservation_passes(monkeypatch):
    res = _sustain(Backend(delay=0.05, stats_fault="warm_reserved"), monkeypatch)
    assert res["passed"], res


def test_gpu_reading_missing_in_the_middle_fails(monkeypatch):
    calls = {"n": 0}

    def gpu(u):
        calls["n"] += 1
        return None if calls["n"] == 5 else 100.0

    res = _sustain(Backend(delay=0.05), monkeypatch, gpu=gpu)
    assert not res["passed"] and any("gpu_memory_used_mib" in p for p in res["sample_schema_problems"]), res


def test_memory_excursion_that_recovers_by_the_end_fails(monkeypatch):
    """Engine RSS +5% for a few readings after the reference, back at the end: the endpoint alone would pass."""
    res = _sustain(Backend(delay=0.05, stats_fault="spike"), monkeypatch)
    env = res["memory_envelope"]["rss_kib"]
    assert not res["passed"] and env["max_dev"] >= 0.04, res


def test_replay_rebuilds_a_full_run_and_rejects_edits(monkeypatch, tmp_path):
    """tools/robustness_replay.py rebuilds every scenario's request-derived results from the saved artifacts of a full
    run; an edited trace, a deleted wire record and a missing artifact are rejected."""
    import gzip
    import json
    from types import SimpleNamespace

    import tools.robustness_replay as rr

    monkeypatch.setattr(rb, "gpu_query", lambda u: {"sku": "X", "uuid": u, "driver": "1", "index": 0, "memory_used_mib": 100.0} if u else None)
    be = Backend(delay=0.02, procs=4, frames=[32, 33, 34, 40])
    cfg = str(ROOT / "configs" / "engine" / "track_s_fast.json")

    async def go():
        from engine.server import served_identity

        es = EngineServer(be)
        es.info = {**served_identity(cfg, "engine.runtime.backend_s:TrackSBackend"), "device_uuid": "GPU-a"}
        srv = TestServer(es.app)
        await srv.start_server()
        try:
            return await rb.run(SimpleNamespace(url=f"http://127.0.0.1:{srv.port}", out=str(tmp_path), config=cfg, gpu_uuid="GPU-a", acceptance=False,
                                                sustain_min=0.03, concurrency=2, overload=4, eos_n=6))
        finally:
            await srv.close()
    rep = asyncio.run(go())
    assert rep["identity"]["matches"] and rep["gpu"]["matches"] and rep["sustain"]["requests"] > 6, rep.get("sustain")
    ok = rr.verify(tmp_path)
    assert ok["ok"], ok["problems"][:5]
    assert ok["rebuilt"]["sustain"]["requests"] == rep["sustain"]["requests"] and ok["rebuilt"]["eos_boundary"]["n"] == 6
    # an edited stored metric: the replay of its wire bytes disagrees (and the artifact hash changed)
    lines = (tmp_path / "requests.jsonl").read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if '"scenario": "sustain"' in line)
    r = json.loads(lines[i])
    r["metrics"]["stall_s"] += 1.0
    lines[i] = json.dumps(r)
    (tmp_path / "requests.jsonl").write_text("\n".join(lines) + "\n")
    bad = rr.verify(tmp_path)
    assert not bad["ok"] and any("hash" in p for p in bad["problems"]) and any("stall_s" in p for p in bad["problems"]), bad["problems"][:5]
    # a deleted wire record
    with gzip.open(tmp_path / "wire.jsonl.gz", "rt") as f:
        wires = f.read().splitlines()
    with gzip.open(tmp_path / "wire.jsonl.gz", "wt") as f:
        f.write("\n".join(wires[1:]) + "\n")
    assert any("no wire record" in p for p in rr.verify(tmp_path)["problems"])
    (tmp_path / "requests.jsonl").unlink()
    assert not rr.verify(tmp_path)["ok"]
