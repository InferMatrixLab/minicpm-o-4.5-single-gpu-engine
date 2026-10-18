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


def _write_dir(d: Path, kernels: list, spans: list, hb: float = 12.0, window=(10.0, 11.0), procs=(("engine", 100),), skip_capture=(), rec_extra=None):
    (d / "procs").mkdir(parents=True)
    (d / "kernels").mkdir()
    w = {"tag": "w", "t0": window[0], "t1": window[1]}
    for proc, pid in procs:
        (d / "procs" / f"{proc}-{pid}.json").write_text(json.dumps({"proc": proc, "pid": pid}))
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
