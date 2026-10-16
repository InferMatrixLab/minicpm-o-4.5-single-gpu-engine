"""Scheduling policy of the Token2Wav worker loop (pure Python: CPU-testable without the models).

Each request carries its first-audio state: before its first audio its next window is due ``first_slack_s`` after it
became ready (the TTFA lane); afterwards the client plays what it has received, so its next window is due when that
audio runs out (playback deadline = time of the first audio + seconds of audio sent, the harness's stall model).

Per scheduling tick:
* continuation windows (requests that already sent audio) wait -> the tick decodes first, choosing windows by earliest
  deadline (EDF), then runs at most ``setup_calls_when_busy`` prompt-setup calls, so a continuation window never waits
  behind more than that much setup work;
* nothing continuing waits -> every pending setup runs, then the decode (the new streams' first windows ride it).

Waiting work ages: setups are taken oldest first (by the time their features became ready), and every window keeps a
fixed deadline, so under EDF a waiting window becomes the earliest as time passes; neither lane starves the other (at
least one setup call per tick, every ready window decoded unless ``max_decode_rows`` caps the call, in which case the
earliest deadlines go first).
"""

from __future__ import annotations

from dataclasses import dataclass

SR = 24000


@dataclass
class Policy:
    setup_calls_when_busy: int = 1  # setup calls per tick while continuation windows wait
    max_decode_rows: int | None = None  # windows per decode call (None: every ready window)
    first_slack_s: float = 0.25  # a first window's deadline after it became ready


@dataclass
class AudioState:
    """A request's first-audio state and playback clock."""

    t_first_audio: float | None = None  # when its first audio was sent (worker clock)
    audio_s: float = 0.0  # seconds of audio sent

    def sent(self, n_samples: int, now: float) -> None:
        if n_samples and self.t_first_audio is None:
            self.t_first_audio = now
        self.audio_s += n_samples / SR

    @property
    def started(self) -> bool:
        return self.t_first_audio is not None

    def deadline(self, t_ready: float, policy: Policy) -> float:
        """When this request's next window is due."""
        if self.t_first_audio is None:
            return t_ready + policy.first_slack_s
        return self.t_first_audio + self.audio_s


def choose_windows(ready: list, policy: Policy) -> list:
    """``ready``: (deadline, key) per request whose head window is ready; returns the keys to decode, earliest first."""
    order = sorted(ready, key=lambda x: x[0])
    if policy.max_decode_rows is not None:
        order = order[:policy.max_decode_rows]
    return [k for _, k in order]


def plan_tick(continuation_waiting: bool, setup_calls: list, policy: Policy) -> tuple[bool, list]:
    """``setup_calls``: (oldest features-ready time, call) per pending setup call.  Returns (decode first?, the calls to
    run this tick, oldest first)."""
    calls = [c for _, c in sorted(setup_calls, key=lambda x: x[0])]
    if continuation_waiting:
        return True, calls[:max(1, policy.setup_calls_when_busy)]
    return False, calls
