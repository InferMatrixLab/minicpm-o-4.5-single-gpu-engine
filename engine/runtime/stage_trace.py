"""Stage trace (diagnostic, off unless ``stage_trace_dir`` is configured): the host and GPU interval of each named stage.

A stage records a CUDA event on the current stream at its start and end and the host clock (``time.perf_counter``,
CLOCK_MONOTONIC, shared by the engine and its worker processes).  Nothing synchronizes: finished stages are resolved
when their end event has completed (``Event.query``) and appended to ``<dir>/<proc>.jsonl`` once a second.  GPU times
are placed on the host clock through one base event recorded after a synchronize at start-up, so the engine's and the
worker's intervals share one time axis (to within the alignment error, tens of microseconds).

Row: {"s": stage, "n": rows, "f": frames (Talker) or 0, "c0", "c1": host start/end, "g0", "g1": GPU start/end} (seconds).
"""

from __future__ import annotations

import json
import time
from collections import deque
from pathlib import Path

import torch


class StageTrace:
    def __init__(self, directory: str | Path, proc: str):
        self.path = Path(directory) / f"{proc}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fh = open(self.path, "a")
        torch.cuda.synchronize()
        self.base = torch.cuda.Event(enable_timing=True)
        self.base.record()
        self.base.synchronize()
        self.t_base = time.perf_counter()
        self.pending: deque = deque()
        self.last_flush = self.t_base

    def begin(self, stage: str):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return stage, time.perf_counter(), e

    def end(self, tok, n: int = 0, frames: int = 0) -> None:
        stage, c0, e0 = tok
        e1 = torch.cuda.Event(enable_timing=True)
        e1.record()
        self.pending.append((stage, n, frames, c0, time.perf_counter(), e0, e1))
        if time.perf_counter() - self.last_flush >= 1.0:
            self.flush(block=False)

    def flush(self, block: bool = False) -> None:
        out = []
        while self.pending and (block or self.pending[0][6].query()):
            stage, n, frames, c0, c1, e0, e1 = self.pending.popleft()
            if block:
                e1.synchronize()
            g0 = self.t_base + self.base.elapsed_time(e0) / 1000
            g1 = self.t_base + self.base.elapsed_time(e1) / 1000
            out.append(json.dumps({"s": stage, "n": n, "f": frames, "c0": round(c0, 6), "c1": round(c1, 6), "g0": round(g0, 6), "g1": round(g1, 6)}))
        if out:
            self.fh.write("\n".join(out) + "\n")
            self.fh.flush()
        self.last_flush = time.perf_counter()
