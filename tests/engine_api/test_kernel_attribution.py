"""Kernel-activity attribution (tools/kernel_attribution.py) on synthetic trace directories (CPU).

The round-10 counterexample: a stage whose CUDA-event span lasts a second while its kernels run for 10 ms (the rest
is a host-only delay) must show 1000 ms elapsed and 10 ms of kernel activity, not a busy GPU.  Incomplete traces
(missing, missed or uncovering captures, disagreeing markers, missing heartbeats, unregistered processes) must be
rejected, not read as zero activity.
"""

import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import tools.kernel_attribution as ka  # noqa: E402

OFFSET_NS = 5_000_000_000  # profiler clock = host clock - 5 s
IDENT = 140011243106944  # a pthread_self value


def _prof_us(host_s: float) -> float:
    return (host_s * 1e9 - OFFSET_NS) / 1000


def _markers(t: float, n: int = 16):
    """Marker host brackets from ``t`` (s) and their kernels (start 5 us into a 20 us bracket, 1 us long)."""
    ms, ks = [], []
    for i in range(n):
        t0 = int((t + i * 0.002) * 1e9)
        ms.append([t0, t0 + 20_000])
        ks.append({"ph": "X", "cat": "kernel", "name": "at::cuda::spin_kernel(long)", "ts": (t0 + 5_000 - OFFSET_NS) / 1000, "dur": 1.0, "args": {}})
    return ms, ks


def _write_dir(d: Path, kernels: list, spans: list, hb: float = 12.0, window=(10.0, 11.0), procs=(("engine", 100),), skip_capture=(), rec_extra=None,
               waits=(), gpm=True, extra_traces=()):
    (d / "procs").mkdir(parents=True)
    (d / "kernels").mkdir()
    w = {"tag": "w", "t0": window[0], "t1": window[1]}
    if gpm:  # full counter coverage of 0..30 s
        (d / "gpm.jsonl").write_text("".join(json.dumps({"t0": t, "t1": t + 0.5, "sm_util": 40.0, "sm_occupancy": 10.0, "dram_bw_util": 12.0,
                                                         "tensor_util": 3.0}) + "\n" for t in np.arange(0, 30, 0.5)))
    for proc, pid in procs:
        (d / "procs" / f"{proc}-{pid}.json").write_text(json.dumps({"proc": proc, "pid": pid, "traces": [proc, *extra_traces]}))
        rows = [{"s": "_start", "c0": 1.0, "pid": pid, "ident": IDENT}] + spans + [{"s": "_hb", "c0": hb}]
        (d / f"{proc}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        if (proc, pid) in skip_capture:
            continue
        ms, mk = _markers(window[0] - 0.1)
        me, mk2 = _markers(window[1] + 0.06)
        ev = mk + mk2
        tid = ka.thread_key(IDENT)
        for j, (s, e, launch) in enumerate(kernels):
            ev.append({"ph": "X", "cat": "kernel", "name": f"k{j}", "ts": _prof_us(s), "dur": (e - s) * 1e6, "args": {"correlation": j + 1}})
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": _prof_us(launch), "dur": 2.0, "tid": tid,
                       "args": {"correlation": j + 1}})
        for s, e in waits:  # blocking runtime calls
            ev.append({"ph": "X", "cat": "cuda_runtime", "name": "cudaStreamSynchronize", "ts": _prof_us(s), "dur": (e - s) * 1e6, "tid": tid,
                       "args": {"correlation": 10_000 + len(ev)}})
        with gzip.open(d / "kernels" / f"{proc}-{pid}-w.json.gz", "wt") as f:
            json.dump({"traceEvents": ev}, f)
        rec = {"proc": proc, "pid": pid, "window": w, "trace": f"{proc}-{pid}-w.json.gz", "markers_start": ms, "markers_end": me, **(rec_extra or {})}
        (d / "kernels" / f"{proc}-{pid}-w.markers.json").write_text(json.dumps(rec))


SPANS = [{"s": "work", "n": 4, "f": 0, "c0": 10.0, "c1": 11.0, "g0": 10.0, "g1": 11.0},  # 1 s span, 10 ms of kernels
         {"s": "work.inner", "n": 4, "f": 0, "c0": 10.0, "c1": 10.0005, "g0": 10.0, "g1": 10.01}]
KERNELS = [(10.0001, 10.0101, 10.0002 - 0.0001 - 0.00005)]  # runs 10 ms; launched inside work.inner


def test_host_only_delay_inside_a_span_is_not_kernel_activity(tmp_path):
    _write_dir(tmp_path, KERNELS, SPANS)
    out = ka.analyze_kernels(tmp_path, {"w": (10.0, 11.0)}, {"engine": 1})["w"]
    st = out["stages"]
    assert st["engine.work"]["elapsed_ms"] == pytest.approx(1000, abs=1e-6)
    assert st["engine.work"]["kernel_total_ms"] == pytest.approx(10, abs=0.01)
    assert st["engine.work"]["kernel_self_ms"] == 0  # the kernel was launched inside the inner stage
    assert st["engine.work.inner"]["kernel_self_ms"] == pytest.approx(10, abs=0.01)
    assert out["device"]["active_frac"] == pytest.approx(0.01, abs=1e-4)  # the old envelope union said 1.0
    al = out["alignment"]["engine-100"]
    assert al["halfwidth_start_us"] <= 10 and abs(al["drift_us"]) < 0.01


def test_blocking_calls_split_a_stages_host_time(tmp_path):
    _write_dir(tmp_path, KERNELS, SPANS, waits=[(10.2, 10.7)])  # half a second synchronizing inside "work"
    st = ka.analyze_kernels(tmp_path, {"w": (10.0, 11.0)}, {"engine": 1})["w"]["stages"]
    assert st["engine.work"]["blocking_ms"] == pytest.approx(500, abs=0.01)
    assert st["engine.work.inner"]["blocking_ms"] == 0


def test_kernels_outside_every_stage_are_reported_as_such(tmp_path):
    _write_dir(tmp_path, KERNELS + [(10.5, 10.6, 10.4)], SPANS[:1])
    out = ka.analyze_kernels(tmp_path, {"w": (10.0, 11.0)}, {"engine": 1})["w"]
    assert out["stages"]["engine.work"]["kernel_total_ms"] == pytest.approx(110, abs=0.1)  # both launched inside "work"
    _write_dir(tmp_path / "b", [(10.5, 10.6, 10.4)], [{"s": "work", "n": 1, "f": 0, "c0": 10.0, "c1": 10.1, "g0": 10.0, "g1": 10.1}])
    out = ka.analyze_kernels(tmp_path / "b", {"w": (10.0, 11.0)}, {"engine": 1})["w"]
    assert out["stages"]["engine.(outside stages)"]["kernel_self_frac"] == pytest.approx(0.1, abs=1e-4)
    assert out["stages"]["engine.work"]["kernel_total_ms"] == 0


@pytest.mark.parametrize("case", ["missing_capture", "missed", "error", "not_covering", "markers_disagree", "no_heartbeat", "unregistered", "trace_file"])
def test_incomplete_traces_are_rejected(tmp_path, case):
    kw = {}
    if case == "missing_capture":
        kw = {"procs": (("engine", 100), ("t2w", 200)), "skip_capture": (("t2w", 200),)}
    elif case == "missed":
        kw = {"rec_extra": {"missed": True}}
    elif case == "error":
        kw = {"rec_extra": {"error": "RuntimeError: x"}}
    elif case == "no_heartbeat":
        kw = {"hb": 10.5}
    _write_dir(tmp_path, KERNELS, SPANS, **kw)
    mf = tmp_path / "kernels" / "engine-100-w.markers.json"
    rec = json.loads(mf.read_text())
    if case == "not_covering":
        rec["markers_end"] = [[int(10.5e9), int(10.5e9) + 20_000]] * 16
    elif case == "markers_disagree":
        rec["markers_start"][3] = [rec["markers_start"][3][0] + 100_000, rec["markers_start"][3][1] + 100_000]
    elif case == "trace_file":
        (tmp_path / "kernels" / rec["trace"]).unlink()
    mf.write_text(json.dumps(rec))
    expected = {"engine": 1, "t2w": 1} if case in ("missing_capture", "unregistered") else {"engine": 1}
    with pytest.raises(ka.TraceError):
        ka.analyze_kernels(tmp_path, {"w": (10.0, 11.0)}, expected)


def test_lost_stage_files_are_rejected(tmp_path):
    _write_dir(tmp_path, KERNELS, SPANS)
    (tmp_path / "engine.jsonl").unlink()  # the whole stage file: its kernels must not become "outside stages"
    with pytest.raises(ka.TraceError, match="registered stage trace engine missing"):
        ka.analyze_kernels(tmp_path, {"w": (10.0, 11.0)}, {"engine": 1})
    _write_dir(tmp_path / "b", KERNELS, SPANS, extra_traces=("engine-talker",))  # a split-thread trace registered, its file lost
    with pytest.raises(ka.TraceError, match="engine-talker missing"):
        ka.analyze_kernels(tmp_path / "b", {"w": (10.0, 11.0)}, {"engine": 1})


def test_requested_window_must_be_the_captured_one(tmp_path):
    _write_dir(tmp_path, KERNELS, SPANS, hb=30.0)  # a valid capture of [10, 11]; heartbeats after 21
    with pytest.raises(ka.TraceError, match="does not cover"):
        ka.analyze_kernels(tmp_path, {"w": (20.0, 21.0)}, {"engine": 1})  # read as idle before the repair


def test_gpm_coverage_is_per_counter():
    rows = [{"t0": 0, "t1": 1, "sm_util": 100.0, "tensor_util": 5.0}, {"t0": 1, "t1": 2, "sm_util": None, "tensor_util": float("nan")}]
    with pytest.raises(ka.TraceError, match="sm_util"):
        ka.gpm_mean(rows, 0, 2, ("sm_util",))  # half the window has no SM evidence: not "50% utilization"
    g = ka.gpm_mean(rows, 0, 2, min_coverage=0.0)
    assert g["sm_util"] == 100.0 and g["coverage"]["sm_util"] == 0.5 and g["tensor_util"] == 5.0
    with pytest.raises(ka.TraceError, match="dram_bw_util"):  # a counter never sampled
        ka.gpm_mean([{"t0": 0, "t1": 2, "sm_util": 50.0, "sm_occupancy": 5.0, "tensor_util": 1.0}], 0, 2, ka.GPM_REQUIRED)


def test_missing_gpm_file_is_rejected(tmp_path):
    _write_dir(tmp_path, KERNELS, SPANS, gpm=False)
    with pytest.raises(ka.TraceError, match="gpm.jsonl missing"):
        ka.analyze_kernels(tmp_path, {"w": (10.0, 11.0)}, {"engine": 1})


def test_thread_key_matches_the_profiler_records():
    # observed: pthread_self 140011243106944 -> runtime records tid 395775360 (tools/spike probe, round 11)
    assert ka.thread_key(140011243106944) == 395775360


def test_critical_path_segments():
    ev = [["queued", 1.0], ["admitted", 1.01], ["w_open", 1.011], ["prefilled", 1.05], ["w_features", 1.2], ["w_setup:2", 1.21],
          ["w_setup_done", 1.24], ["text_chunk:10", 1.4], ["window_ready:0", 1.5], ["w_recv:0", 1.502], ["w_decode:0", 1.51],
          ["w_decoded:0", 1.56], ["audio_recv:0", 1.561], ["first_audio", 1.561], ["first_yield", 1.562], ["window_ready:1", 2.0],
          ["w_recv:1", 2.003], ["end", 3.0]]
    out = ka.critical_path([{"rid": "a", "events": ev}, {"rid": "b", "events": [["queued", 50.0], ["first_yield", 50.5]]}], 0, 10,
                           client={"a": (0.99, 1.565)})
    fa = out["first_audio"]
    assert out["requests"] == 1
    assert fa["prefill"]["p50_ms"] == pytest.approx(40)
    assert fa["first_text_chunk"]["p50_ms"] == pytest.approx(350)
    assert fa["decode_call"]["p50_ms"] == pytest.approx(50)
    assert fa["window0_vs_setup"]["p50_ms"] == pytest.approx(-262)
    assert fa["client_ttfa"]["p50_ms"] == pytest.approx(575)
    assert out["every_window"]["to_worker"]["n"] == 2


def _traced_backend(rec: list):
    import threading
    from types import SimpleNamespace

    from engine.runtime.backend_s import TrackSBackend

    be = object.__new__(TrackSBackend)
    be.stage_trace_dir, be.profile = "/nonexistent", False
    tl = threading.local()
    tl.trace = SimpleNamespace(begin=lambda s: s, end=lambda tok, n=0, frames=0: rec.append((tok, n, frames)))
    be.__dict__["_trace_tl"] = tl  # this thread's trace (made on first use otherwise)
    return be


def test_traced_in_process_setup_takes_feature_tuples():
    """Q10: the in-process Token2Wav thread's setup stage passes feature tuples, not LiveS rows."""
    import inspect

    from engine.runtime.backend_s import TrackSBackend

    rec = []
    be = _traced_backend(rec)
    assert be._timed_ret("setup", lambda feats: ["prompt"] * len(feats), [("tokens", "lengths", "speaker", "mel", "x")], live=False) == ["prompt"]
    assert rec == [("setup", 1, 0)]
    assert 'self._timed_ret("setup", self.t2w_batch.setup_many, [f for _, f in part], live=False)' in inspect.getsource(TrackSBackend._setups)


def test_traced_talker_stage_counts_frames():
    from types import SimpleNamespace

    rec = []
    be = _traced_backend(rec)
    rows = [SimpleNamespace(frames=3), SimpleNamespace(frames=5)]

    def step(rs):
        for r in rs:
            r.frames += 4

    be._timed("talker", step, rows)
    assert rec == [("talker", 2, 8)]


def test_inventory_owns_only_this_runs_mps_server(monkeypatch):
    import tools.engine_profile as ep

    apps = [{"pid": 11, "name": "/usr/bin/nvidia-cuda-mps-server", "mib": 60, "uid": 1000},  # this run's daemon (pid 10)
            {"pid": 21, "name": "/usr/bin/nvidia-cuda-mps-server", "mib": 60, "uid": 1000},  # another daemon of the same user
            {"pid": 5, "name": "python", "mib": 9000, "uid": 1000}]  # the engine (in the server tree)
    monkeypatch.setattr(ep, "process_tree", lambda pid: [{"pid": 4, "cmd": "launcher"}, {"pid": 5, "cmd": "engine"}])
    monkeypatch.setattr(ep, "gpu_apps", lambda uuid: apps)
    monkeypatch.setattr(ep, "_ppid", lambda pid: {11: 10, 21: 20}.get(pid))
    monkeypatch.setattr(ep, "_environ", lambda pid: {"CUDA_MPS_PIPE_DIRECTORY": {10: "/tmp/cgz-engine-mps-18950/pipe", 20: "/tmp/other/pipe"}[pid]})
    inv = ep.inventory("uuid", 4, "/tmp/cgz-engine-mps-18950/pipe")
    assert inv["own_mps_servers"] == [11]
    assert [f["pid"] for f in inv["foreign"]] == [21]  # same UID is not ownership


def _sample(t, foreign=(), error=None):
    return {"t": t, "foreign": [{"pid": p} for p in foreign], **({"error": error} if error else {})}


def _run_json(tmp_path, samples, summary=None):
    win = {"t0": 10.0, "t1": 11.0, "client": {}, "kernel": {"tag": "w", "t0": 10.0, "t1": 11.0}, "inventory_samples": samples,
           "inventory": summary if summary is not None else {"samples": len(samples), "foreign": [f for s in samples for f in s["foreign"]]}}
    (tmp_path / "run.json").write_text(json.dumps({"identity": {}, "config": {}, "expected_processes": {"engine": 1}, "windows": {"w": win}}))


@pytest.mark.parametrize("case", ["no_raw_samples", "outside_window", "foreign_raw", "summary_disagrees", "query_failed"])
def test_analysis_rejects_an_unproven_inventory(tmp_path, case):
    import tools.engine_profile as ep

    _write_dir(tmp_path / "trace", KERNELS, SPANS)
    ok = [_sample(10.2), _sample(10.5), _sample(10.9)]
    if case == "no_raw_samples":
        _run_json(tmp_path, [], {"samples": 3, "foreign": []})  # a summary without its evidence
    elif case == "outside_window":
        _run_json(tmp_path, [_sample(30.0)] * 3)
    elif case == "foreign_raw":
        _run_json(tmp_path, ok[:2] + [_sample(10.9, foreign=(21,))], {"samples": 3, "foreign": []})
    elif case == "summary_disagrees":
        _run_json(tmp_path, ok, {"samples": 2, "foreign": []})
    else:
        _run_json(tmp_path, ok[:2] + [_sample(10.9, error="RuntimeError: nvidia-smi failed (9)")])
    with pytest.raises(ka.TraceError, match="GPU inventory invalid"):
        ep.analyze(tmp_path)


def test_analysis_accepts_a_proven_inventory(tmp_path):
    import tools.engine_profile as ep

    _write_dir(tmp_path / "trace", KERNELS, SPANS)
    _run_json(tmp_path, [_sample(10.2), _sample(10.5), _sample(10.9)])
    out = ep.analyze(tmp_path)["windows"]["w"]
    assert out["kernels"]["device"]["active_frac"] == pytest.approx(0.01, abs=1e-4) and out["gpm_window"]["coverage"]["sm_util"] == 1.0


def test_untraced_control_with_foreign_work_fails(tmp_path, monkeypatch):
    """The real ``cmd_run --no-trace`` completion path (server, sampler and load stubbed): a foreign process in the
    window makes the control invalid and the command return 1."""
    import asyncio
    from types import SimpleNamespace

    import tools.engine_profile as ep

    class Proc:
        def __init__(self, cmd, **kw):
            if any("gpm_sampler.py" in str(x) for x in cmd):  # the sampler writes full counter coverage
                out = Path(cmd[cmd.index("--out") + 1])
                out.write_text("".join(json.dumps({"t0": t, "t1": t + 1, "sm_util": 50.0, "sm_occupancy": 10.0, "dram_bw_util": 10.0,
                                                   "tensor_util": 1.0}) + "\n" for t in range(0, 40)))
            self.pid = 4242

        def poll(self):
            return 0

        def send_signal(self, sig):
            pass

        def wait(self, timeout=None):
            return 0

    async def load(url, c, warm_s, measure_s, on_window, client_rows, sample_inventory=None):
        return {"t0": 10.0, "t1": 20.0, "client": {"frames_per_s": 100.0}, "inventory_samples": [_sample(11.0), _sample(15.0, foreign=(999,)), _sample(19.0)]}

    monkeypatch.setattr(ep.subprocess, "Popen", Proc)
    monkeypatch.setattr(ep, "identity", lambda out, cfg_text, gpu: {"gpu_uuid": "u", "gpu_apps_before": []})
    monkeypatch.setattr(ep, "_wait_health", lambda url, proc: None)
    monkeypatch.setattr(ep, "_load", load)
    monkeypatch.setattr(ep.time, "sleep", lambda s: None)
    monkeypatch.setattr(asyncio, "run", lambda coro: asyncio.new_event_loop().run_until_complete(coro))
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"backend": "x"}))
    a = SimpleNamespace(out=str(tmp_path / "ctl"), config=str(cfg), set=None, no_trace=True, spans_only=False, window_thread=False, gpu=2, port=18999,
                        c=[8], warm_s=1, measure_s=1, kernel_s=1.5)
    assert ep.cmd_run(a) == 1
    ctl = json.loads((tmp_path / "ctl" / "control.json").read_text())
    assert ctl["valid"] is False and any("foreign" in p for p in ctl["windows"]["8"]["inventory_problems"])


class FakeEvent:
    """A CUDA-event stand-in: completes when the test says so; elapsed time from the host clock at record."""

    def __init__(self, clock):
        self.clock, self.t, self.done = clock, None, False

    def record(self):
        self.t = self.clock()

    def query(self):
        return self.done

    def synchronize(self):
        self.done = True

    def elapsed_time(self, other):
        return 1000 * (other.t - self.t)


def _trace(tmp_path, monkeypatch):
    import engine.runtime.stage_trace as st

    now = [10.0]
    clock = lambda: now[0]  # noqa: E731
    monkeypatch.setattr(st.time, "perf_counter", clock)
    monkeypatch.setattr(st, "_FLUSHER", {st.os.getpid(): None})  # no background flusher in the test
    events = []

    def event():
        e = FakeEvent(clock)
        events.append(e)
        return e

    tr = st.StageTrace(tmp_path, "engine", proc="engine", kernel_windows=False, event=event)
    events[0].done = True  # the base event
    return tr, now, events


def _hb(tmp_path):
    rows = [json.loads(x) for x in (tmp_path / "engine.jsonl").read_text().splitlines()]
    return max(r["c0"] for r in rows if r["s"] == "_hb"), [r for r in rows if not r["s"].startswith("_")]


def test_watermark_never_passes_an_active_span(tmp_path, monkeypatch):
    tr, now, ev = _trace(tmp_path, monkeypatch)
    now[0] = 10.5
    tok = tr.begin("talker")  # a long stage: still running at 12
    now[0] = 12.0
    tr.flush()
    hb, rows = _hb(tmp_path)
    assert hb == 10.5 and not rows  # [10, 11] is not certified while the stage runs
    files = ka.load_stage_files(tmp_path)
    procs = ka.registered(tmp_path, {"engine": 1}, files)
    with pytest.raises(ka.TraceError, match="heartbeat"):
        ka.check_stage_coverage(files, procs, 10.0, 11.0)
    now[0] = 12.5
    tr.end(tok, 1)
    ev[-1].done = True
    tr.flush()
    hb, rows = _hb(tmp_path)
    assert hb == 12.5 and [r["s"] for r in rows] == ["talker"]
    files = ka.load_stage_files(tmp_path)
    ka.check_stage_coverage(files, ka.registered(tmp_path, {"engine": 1}, files), 10.0, 11.0)


def test_watermark_waits_for_a_pending_end_event_and_nested_spans(tmp_path, monkeypatch):
    tr, now, ev = _trace(tmp_path, monkeypatch)
    now[0] = 10.0
    outer = tr.begin("talker")
    now[0] = 10.2
    inner = tr.begin("talker.decode")
    now[0] = 10.4
    tr.end(inner, 1)
    ev[-1].done = True  # the inner span retires first
    now[0] = 11.5
    tr.end(outer, 1)  # ended on the host after the window, its end event still pending (the review's crossing case)
    now[0] = 12.0
    tr.flush()
    hb, rows = _hb(tmp_path)
    assert hb == 10.0 and [r["s"] for r in rows] == ["talker.decode"]  # not 11.5: the outer row is unwritten
    ev[-1].done = True
    tr.flush()
    hb, rows = _hb(tmp_path)
    assert hb == 12.0 and sorted(r["s"] for r in rows) == ["talker", "talker.decode"]


def test_failing_stage_still_ends_its_span():
    from types import SimpleNamespace

    rec = []
    be = _traced_backend(rec)

    def boom(rows):
        raise RuntimeError("stage failed")

    with pytest.raises(RuntimeError):
        be._timed("talker", boom, [SimpleNamespace(frames=0)])
    assert rec == [("talker", 1, 0)]
