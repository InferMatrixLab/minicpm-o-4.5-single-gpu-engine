"""Scheduling policy of the Token2Wav worker loop (pure Python: CPU-testable without the models).

Each request carries its first-audio state: before its first audio its next window is due ``first_slack_s`` after it
became ready (the TTFA lane); afterwards the client plays what it has received, so its next window is due when that
audio runs out.  The playback clock is the harness's stall model (``bench.metrics.playback_stall_s``): playback starts at
the first audio, pauses when the buffer empties and resumes when audio arrives, so the buffer runs out at
``exhaust = max(previous exhaust, delivery time) + seconds delivered`` (an empty chunk changes nothing).  The delivery
clock is the sender's (the worker or the engine when it hands audio on); the client receives it later by the transport
and HTTP latency, so the true deadline is slightly later than this one (conservative).

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
    """A request's first-audio state and playback clock (when its buffered audio runs out)."""

    t_first_audio: float | None = None  # when its first audio was sent (sender clock)
    audio_s: float = 0.0  # seconds of audio sent
    t_exhaust: float | None = None  # when the client's buffer runs out (playback pauses while it is empty)

    def sent(self, n_samples: int, now: float) -> None:
        if not n_samples:
            return
        if self.t_first_audio is None:
            self.t_first_audio = now
        d = n_samples / SR
        self.audio_s += d
        self.t_exhaust = max(self.t_exhaust if self.t_exhaust is not None else now, now) + d

    @property
    def started(self) -> bool:
        return self.t_first_audio is not None

    def deadline(self, t_ready: float, policy: Policy) -> float:
        """When this request's next window is due."""
        if self.t_exhaust is None:
            return t_ready + policy.first_slack_s
        return self.t_exhaust

    def slack(self, now: float) -> float | None:
        """Seconds of buffered playback left (None before the first audio; negative while stalled)."""
        return None if self.t_exhaust is None else self.t_exhaust - now


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


@dataclass
class PrefillPolicy:
    """The engine tick's bounded first-chunk lane: Thinker prefills of newly admitted requests.

    ``token_budget``: prompt tokens prefilled per tick (None: every pending prefill, the rounds 4-6 behaviour);
    ``max_age_s``: a prefill that has waited this long runs regardless of the budget and of continuation slack (aging);
    ``slack_floor_s``: while any continuing stream has less buffered playback than this, only aged prefills run, so the
    tick that produces its next frames is not lengthened by new requests' prefills (None: never)."""

    token_budget: int | None = 8192
    max_age_s: float = 0.25
    slack_floor_s: float | None = 0.3


def choose_prefills(pending: list, now: float, min_slack: float | None, policy: PrefillPolicy) -> list:
    """``pending``: (admitted at, prompt tokens, key) per request awaiting its Thinker prefill; ``min_slack``: the least
    buffered playback among continuing audio streams (None if none).  Returns the keys to prefill this tick, oldest
    first: aged ones always; the others only when no continuing stream is below the slack floor, within the budget
    (at least one)."""
    order = sorted(pending, key=lambda x: x[0])
    aged = [k for t, _, k in order if now - t >= policy.max_age_s]
    if policy.slack_floor_s is not None and min_slack is not None and min_slack < policy.slack_floor_s:
        return aged
    out, used = [], 0
    for t, n, k in order:
        if out and policy.token_budget is not None and used + n > policy.token_budget and now - t < policy.max_age_s:
            break
        out.append(k)
        used += n
    return out
