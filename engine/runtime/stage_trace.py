"""Stage trace (diagnostic, off unless ``stage_trace_dir`` is configured): named stage spans and kernel-activity windows.

**Stage spans.** A stage records a CUDA event on the current stream at its start and end and the host clock
(``time.perf_counter``, CLOCK_MONOTONIC, shared by the engine and its worker processes).  Nothing synchronizes: finished
stages are resolved when their end event has completed (``Event.query``) and appended to ``<dir>/<name>.jsonl`` about
once a second.  The two event times are placed on the host clock through one base event recorded after a synchronize
when the trace is made.  The span between the events is the stage's *elapsed* time on its stream: it includes host
submission gaps, synchronizations and waits behind other work, so it is not the time kernels ran (that comes from the
kernel windows below).  Spans nest (``span``): an outer stage and the inner stages it is made of.

Row: {"s": stage, "n": rows, "f": frames (Talker) or audio frames (Token2Wav) or 0, "c0", "c1": host start/end,
"g0", "g1": stream start/end} (seconds).  A ``{"s": "_hb", "c0": t}`` row ends every flush: ``t`` is a completeness
watermark, every stage that began before ``t`` is in the file (it never passes a stage still running or waiting for its
end event; ``analyze`` requires one at or after each window's end).  One trace per thread.

**Kernel windows.** Each process that makes a trace runs one ``KernelWindows`` thread.  It polls
``<dir>/kernel_windows.json`` (a list of ``{"tag", "t0", "t1"}`` on the host clock, written by
``tools/engine_profile.py``) and, for each window, runs ``torch.profiler`` with CUDA activity only (CUPTI: every kernel,
memcpy and memset of the process with its device start/end, and the runtime call that launched it) from shortly before
``t0`` to after ``t1``.  Around the capture it launches marker kernels on its own stream between two host timestamps:
each marker's device start lies between them, which bounds the offset of the profiler's clock to the host clock (the
bound is measured, not assumed).  It writes ``<dir>/kernels/<proc>-<pid>-<tag>.json`` (the trace; ``tools/engine_profile.py`` compresses it after the run) and then
``...markers.json`` (whose presence marks a complete capture).  Every process registers itself and each stage trace it
makes in ``<dir>/procs/<proc>-<pid>.json``, so the analysis knows which processes must have windows and which stage
files must exist.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
import time
from collections import deque
from pathlib import Path

import torch

LEAD_S = 5.0  # the profiler starts this long before a window (its start-up, measured ~2 s under load, stays outside it)
MARKERS = 16  # marker kernels before and after each capture
_TRACES: list = []  # this process's traces (the flusher thread writes them every second)


class StageTrace:
    def __init__(self, directory: str | Path, name: str, proc: str | None = None, kernel_windows: bool = True, event=None):
        """``event``: a factory of CUDA-event-like objects (record, query, synchronize, elapsed_time); tests only."""
        directory = Path(directory)
        self.path = directory / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        self.lock = threading.Lock()
        self._event = event or (lambda: torch.cuda.Event(enable_timing=True))
        if event is None:
            torch.cuda.synchronize()
        self.base = self._event()
        self.base.record()
        self.base.synchronize()
        self.t_base = time.perf_counter()
        self.pending: deque = deque()
        self.active: dict[int, float] = {}  # begun spans (id of their token) -> host start
        self.watermark = self.t_base
        self.last_flush = self.t_base
        # ident: the profiler's runtime records name a thread by abs(int32(pthread_self)) (tools/kernel_attribution.py)
        self.fh.write(json.dumps({"s": "_start", "c0": round(self.t_base, 6), "pid": os.getpid(), "thread": threading.current_thread().name,
                                  "ident": threading.get_ident(), "native_id": threading.get_native_id()}) + "\n")
        self.fh.flush()
        _TRACES.append(self)
        register(directory, proc or name, name)
        if kernel_windows and os.environ.get("STAGE_TRACE_KERNEL_WINDOWS", "1") != "0":  # 0: spans only (overhead bisection)
            KernelWindows.ensure(directory, proc or name)

    def begin(self, stage: str):
        e = self._event()
        e.record()
        tok = (stage, time.perf_counter(), e)
        with self.lock:
            self.active[id(tok)] = tok[1]
        return tok

    def end(self, tok, n: int = 0, frames: int = 0) -> None:
        stage, c0, e0 = tok
        e1 = self._event()
        e1.record()
        now = time.perf_counter()
        with self.lock:
            self.active.pop(id(tok), None)
            self.pending.append((stage, n, frames, c0, now, e0, e1))
        if now - self.last_flush >= 1.0:
            self.flush(block=False)

    @contextlib.contextmanager
    def span(self, stage: str, n: int = 0):
        tok = self.begin(stage)
        try:
            yield
        finally:
            self.end(tok, n)

    def flush(self, block: bool = False) -> None:
        """Write every finished stage whose end event completed (``block``: all of them) and then a heartbeat whose
        time is a completeness watermark: every stage that began before it is in the file.  It never passes the
        earliest start of a stage still running or still waiting for its end event, and never moves back."""
        with self.lock:
            out, keep = [], deque()
            now = time.perf_counter()
            for row in self.pending:
                stage, n, frames, c0, c1, e0, e1 = row
                if not block and not e1.query():
                    keep.append(row)
                    continue
                if block:
                    e1.synchronize()
                g0 = self.t_base + self.base.elapsed_time(e0) / 1000
                g1 = self.t_base + self.base.elapsed_time(e1) / 1000
                out.append(json.dumps({"s": stage, "n": n, "f": frames, "c0": round(c0, 6), "c1": round(c1, 6), "g0": round(g0, 6), "g1": round(g1, 6)}))
            self.pending = keep
            if out:
                self.fh.write("\n".join(out) + "\n")
                self.fh.flush()  # rows are persisted before the watermark that covers them
            outstanding = [r[3] for r in keep] + list(self.active.values())
            self.watermark = max(self.watermark, min(outstanding + [now]))
            self.fh.write(json.dumps({"s": "_hb", "c0": round(self.watermark, 6)}) + "\n")
            self.fh.flush()
            self.last_flush = time.perf_counter()


_REGISTRY_LOCK = threading.Lock()
_FLUSHER: dict[int, threading.Thread] = {}


def _flush_loop() -> None:
    """Every second, flush every trace of this process (idle ones too: their heartbeats show that a window's rows are
    complete), with or without kernel windows."""
    while True:
        time.sleep(1.0)
        for tr in list(_TRACES):
            tr.flush(block=False)


def register(directory: Path, proc: str, trace_name: str) -> None:
    """``<dir>/procs/<proc>-<pid>.json``: this process and every stage trace (file) it made, so the analysis can require
    each of them (a lost stage file is not an uninstrumented process)."""
    with _REGISTRY_LOCK:
        (directory / "procs").mkdir(parents=True, exist_ok=True)
        p = directory / "procs" / f"{proc}-{os.getpid()}.json"
        cuda = torch.cuda.is_available()
        rec = json.loads(p.read_text()) if p.exists() else {
            "proc": proc, "pid": os.getpid(), "device": torch.cuda.current_device() if cuda else None,
            "device_name": torch.cuda.get_device_name(torch.cuda.current_device()) if cuda else None,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "t": time.perf_counter(), "traces": []}
        if trace_name not in rec["traces"]:
            rec["traces"].append(trace_name)
        if os.getpid() not in _FLUSHER:
            _FLUSHER[os.getpid()] = threading.Thread(target=_flush_loop, daemon=True, name="stage-trace-flush")
            _FLUSHER[os.getpid()].start()
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec))
        tmp.replace(p)


def span(trace: StageTrace | None, stage: str, n: int = 0):
    """``trace.span(stage, n)``, or nothing when tracing is off."""
    return trace.span(stage, n) if trace is not None else contextlib.nullcontext()


class KernelWindows(threading.Thread):
    """The per-process kernel-window thread (see the module docstring)."""

    _started: dict[int, KernelWindows] = {}

    @classmethod
    def ensure(cls, directory: Path, proc: str) -> None:
        if os.getpid() in cls._started:
            return
        t = cls(directory, proc)
        cls._started[os.getpid()] = t
        t.start()

    def __init__(self, directory: Path, proc: str):
        super().__init__(daemon=True, name="kernel-windows")
        self.dir, self.proc = directory, proc
        self.control = directory / "kernel_windows.json"
        self.out = directory / "kernels"
        self.device = torch.cuda.current_device()

    def run(self) -> None:
        torch.cuda.set_device(self.device)
        self.stream = torch.cuda.Stream()
        done: set = set()
        while True:
            time.sleep(0.05)
            try:
                wins = json.loads(self.control.read_text()) if self.control.exists() else []
            except (OSError, ValueError):
                continue  # being written
            now = time.perf_counter()
            for w in wins:
                if w["tag"] in done:
                    continue
                if now > w["t0"] - 0.2:  # too late to cover it: recorded as missed (the analysis rejects the window)
                    done.add(w["tag"])
                    self._write(w, {"missed": True, "now": now})
                elif now >= w["t0"] - LEAD_S:
                    done.add(w["tag"])
                    try:
                        self._capture(w)
                    except Exception as e:  # a failed capture is recorded, never fatal to the process
                        self._write(w, {"error": f"{type(e).__name__}: {e}"})

    def _markers(self) -> list[list[int]]:
        out = []
        with torch.cuda.stream(self.stream):
            for _ in range(MARKERS):
                self.stream.synchronize()
                t0 = time.perf_counter_ns()
                torch.cuda._sleep(100)
                self.stream.synchronize()
                out.append([t0, time.perf_counter_ns()])
                time.sleep(0.002)
        return out

    def _capture(self, w: dict) -> None:
        """CUDA activity over the window through the autograd profiler's kineto session, saved by kineto itself
        (``torch.profiler.profile.stop`` would also parse every event in Python: a minute for a second of engine
        kernels, on this process's GIL)."""
        from torch.autograd import _disable_profiler
        from torch.autograd import profiler as ap

        self.out.mkdir(parents=True, exist_ok=True)
        prof = ap.profile(use_device="cuda", use_kineto=True, use_cpu=False)
        prof._prepare_trace()
        prof._start_trace()
        p0 = time.perf_counter()
        m0 = self._markers()
        while time.perf_counter() < w["t1"] + 0.05:
            time.sleep(0.01)
        m1 = self._markers()
        p1 = time.perf_counter()
        res = _disable_profiler()
        getattr(ap, "_run_on_profiler_stop", lambda: None)()
        path = self.out / f"{self.proc}-{os.getpid()}-{w['tag']}.json"
        res.save(str(path))
        self._write(w, {"trace": path.name, "prof_start": p0, "prof_stop": p1, "markers_start": m0, "markers_end": m1,
                        "export_s": time.perf_counter() - p1})

    def _write(self, w: dict, rec: dict) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        p = self.out / f"{self.proc}-{os.getpid()}-{w['tag']}.markers.json"
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps({"proc": self.proc, "pid": os.getpid(), "window": w, **rec}))
        tmp.replace(p)
