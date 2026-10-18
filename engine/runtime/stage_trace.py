"""Stage trace (diagnostic, off unless ``stage_trace_dir`` is configured): named stage spans and kernel-activity windows.

**Stage spans.** A stage records a CUDA event on the current stream at its start and end and the host clock
(``time.perf_counter``, CLOCK_MONOTONIC, shared by the engine and its worker processes).  Nothing synchronizes: finished
stages are resolved when their end event has completed (``Event.query``) and appended to ``<dir>/<name>.jsonl`` about
once a second.  The two event times are placed on the host clock through one base event recorded after a synchronize
when the trace is made.  The span between the events is the stage's *elapsed* time on its stream: it includes host
submission gaps, synchronizations and waits behind other work, so it is not the time kernels ran (that comes from the
kernel windows below).  Spans nest (``span``): an outer stage and the inner stages it is made of.

Row: {"s": stage, "n": rows, "f": frames (Talker) or audio frames (Token2Wav) or 0, "c0", "c1": host start/end,
"g0", "g1": stream start/end} (seconds).  A ``{"s": "_hb", "c0": t}`` row ends every flush: every stage that ended on
the host before ``t`` is in the file (``analyze`` requires one after each window).  One trace per thread.

**Kernel windows.** Each process that makes a trace runs one ``KernelWindows`` thread.  It polls
``<dir>/kernel_windows.json`` (a list of ``{"tag", "t0", "t1"}`` on the host clock, written by
``tools/engine_profile.py``) and, for each window, runs ``torch.profiler`` with CUDA activity only (CUPTI: every kernel,
memcpy and memset of the process with its device start/end, and the runtime call that launched it) from shortly before
``t0`` to after ``t1``.  Around the capture it launches marker kernels on its own stream between two host timestamps:
each marker's device start lies between them, which bounds the offset of the profiler's clock to the host clock (the
bound is measured, not assumed).  It writes ``<dir>/kernels/<proc>-<pid>-<tag>.json`` (the trace; ``tools/engine_profile.py`` compresses it after the run) and then
``...markers.json`` (whose presence marks a complete capture).  Every process registers itself in
``<dir>/procs/<proc>-<pid>.json`` when its trace is made, so the analysis knows which processes must have windows.
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
_TRACES: list = []  # this process's traces (the kernel-window thread flushes them every second)


class StageTrace:
    def __init__(self, directory: str | Path, name: str, proc: str | None = None, kernel_windows: bool = True):
        directory = Path(directory)
        self.path = directory / f"{name}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        self.lock = threading.Lock()
        torch.cuda.synchronize()
        self.base = torch.cuda.Event(enable_timing=True)
        self.base.record()
        self.base.synchronize()
        self.t_base = time.perf_counter()
        self.pending: deque = deque()
        self.last_flush = self.t_base
        # ident: the profiler's runtime records name a thread by abs(int32(pthread_self)) (tools/kernel_attribution.py)
        self.fh.write(json.dumps({"s": "_start", "c0": round(self.t_base, 6), "pid": os.getpid(), "thread": threading.current_thread().name,
                                  "ident": threading.get_ident(), "native_id": threading.get_native_id()}) + "\n")
        self.fh.flush()
        _TRACES.append(self)
        if kernel_windows and os.environ.get("STAGE_TRACE_KERNEL_WINDOWS", "1") != "0":  # 0: spans only (overhead bisection)
            KernelWindows.ensure(directory, proc or name)

    def begin(self, stage: str):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return stage, time.perf_counter(), e

    def end(self, tok, n: int = 0, frames: int = 0) -> None:
        stage, c0, e0 = tok
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        now = time.perf_counter()
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
        with self.lock:
            out = []
            while self.pending and (block or self.pending[0][6].query()):
                stage, n, frames, c0, c1, e0, e1 = self.pending.popleft()
                if block:
                    e1.synchronize()
                g0 = self.t_base + self.base.elapsed_time(e0) / 1000
                g1 = self.t_base + self.base.elapsed_time(e1) / 1000
                out.append(json.dumps({"s": stage, "n": n, "f": frames, "c0": round(c0, 6), "c1": round(c1, 6), "g0": round(g0, 6), "g1": round(g1, 6)}))
            # every stage whose host end precedes the first one still pending is now written
            hb = self.pending[0][4] if self.pending else time.perf_counter()
            out.append(json.dumps({"s": "_hb", "c0": round(hb, 6)}))
            self.fh.write("\n".join(out) + "\n")
            self.fh.flush()
            self.last_flush = time.perf_counter()


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
        (directory / "procs").mkdir(parents=True, exist_ok=True)
        dev = torch.cuda.current_device()
        (directory / "procs" / f"{proc}-{os.getpid()}.json").write_text(json.dumps(
            {"proc": proc, "pid": os.getpid(), "device": dev, "device_name": torch.cuda.get_device_name(dev),
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "t": time.perf_counter()}))
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
        last = 0.0
        while True:
            time.sleep(0.05)
            if time.perf_counter() - last >= 1.0:  # idle threads' traces too: their heartbeats cover every window
                last = time.perf_counter()
                for tr in list(_TRACES):
                    tr.flush(block=False)
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
