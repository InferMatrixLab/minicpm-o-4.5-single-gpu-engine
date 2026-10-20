"""Common spike workload: B requests, prompt P tokens, R reply tokens, F Talker frames, W Token2Wav windows.

Every layout reports, on the same GPU and workload:
* thinker prefill (ms) for the B prompts, thinker decode step (ms) at batch B;
* Talker condition prefill (ms) and decode step (ms) at batch B;
* Thinker -> Talker handoff (ms) of the reply hidden states;
* one Token2Wav window (ms), batch 1 (reference path);
* the critical path to first audio of one request at batch B:
  prefill + R decode steps + handoff + Talker prefill + 28 frames + one window;
* peak GPU memory and the KV budget the layout reserves.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

MODEL_DIR = Path(os.environ.get("MODEL_PATH") or os.path.join(os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub"), "models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"))


@dataclass
class Workload:
    batch: int = 8
    prompt: int = 60
    reply: int = 30
    frames: int = 100
    windows: int = 3
    iters: int = 20


@dataclass
class SpikeResult:
    layout: str
    workload: dict
    thinker_prefill_ms: float = 0.0
    thinker_decode_step_ms: float = 0.0
    talker_prefill_ms: float = 0.0
    talker_decode_step_ms: float = 0.0
    handoff_ms: float = 0.0
    token2wav_window_ms: float = 0.0
    first_audio_critical_path_ms: float = 0.0
    peak_memory_gib: float = 0.0
    kv_budget_note: str = ""
    notes: list = field(default_factory=list)

    def finish(self, w: Workload) -> "SpikeResult":
        self.first_audio_critical_path_ms = (self.thinker_prefill_ms + w.reply * self.thinker_decode_step_ms + self.handoff_ms
                                             + self.talker_prefill_ms + 28 * self.talker_decode_step_ms + self.token2wav_window_ms)
        self.peak_memory_gib = torch.cuda.max_memory_allocated() / 2**30
        return self

    def save(self, out: Path) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(self), indent=1))
        print(json.dumps(asdict(self), indent=1))


def timed(fn, iters: int) -> float:
    """Median wall ms of ``fn`` (synchronised), after one warmup."""
    fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return ts[len(ts) // 2]


def token2wav_window_ms(w: Workload) -> float:
    from engine.models.token2wav import Token2Wav

    t2w = Token2Wav(MODEL_DIR / "assets" / "token2wav")
    st = t2w.new_stream(t2w.prepare(MODEL_DIR / "assets" / "HT_ref_audio.wav"))
    win = [4218, 4218, 4218] + list(range(100, 125))
    return timed(lambda: t2w.decode(st, win, False), max(3, w.iters // 4))
