"""Scheduling policy of the engine tick and the Token2Wav worker loop (pure Python: CPU-testable without the models).

Each request carries its first-audio state: before its first audio its next window is due ``first_slack_s`` after it
became ready (the TTFA lane); afterwards the client plays what it has received, so its next window is due when that
audio runs out.  The playback clock is the harness's stall model (``bench.metrics.playback_stall_s``): playback starts at
the first audio, pauses when the buffer empties and resumes when audio arrives, so the buffer runs out at
``exhaust = max(previous exhaust, delivery time) + seconds delivered`` (an empty chunk changes nothing).  The delivery
clock is the sender's (the worker or the engine when it hands audio on); the client receives it later by the transport
and HTTP latency, so the true deadline is slightly later than this one (conservative).

Worker tick (``Policy``; the plan's first-chunk-first lane guarded by continuation slack):
* first-chunk lane: pending prompt setups (the first audio of new requests waits for them) run before the decode while
  every waiting continuation window keeps at least ``first_lane_slack_s`` of playback slack (rechecked before each
  setup call); then the decode;
* otherwise the tick decodes first, choosing windows by earliest deadline (EDF), then runs at most
  ``setup_calls_when_busy`` setup calls, so a continuation that is about to run dry waits behind at most that much setup
  work.  ``first_lane_slack_s = None`` decodes first whenever a continuation window waits (the round-7 rule).

Waiting work ages: setups are taken oldest first (by the time their features became ready), and every window keeps a
fixed deadline, so under EDF a waiting window becomes the earliest as time passes; neither lane starves the other (at
least one setup call per tick, every ready window decoded unless ``max_decode_rows`` caps the call, in which case the
earliest deadlines go first).

Engine tick (``PrefillPolicy``, ``choose_prefills``): the Thinker prefills of newly admitted requests (their first
chunk) form a bounded lane: oldest first within a hard per-tick prompt-token budget (aged requests too; the oldest always
progresses, in chunks if it alone exceeds the budget), admitted only while the Thinker KV pool holds their whole
lifetime (prompt + reply), and, while a continuing stream's playback slack is below ``slack_floor_s``, only those that
have waited ``max_age_s`` (aging), after the continuing streams' decode.  Talker chunk starts are bounded the same way
(``choose_starts``: earliest playback deadline first within a per-tick token budget, at least one).
"""

from __future__ import annotations

from dataclasses import dataclass

SR = 24000


@dataclass
class Policy:
    setup_calls_when_busy: int = 1  # setup calls per tick while continuation windows wait
    max_decode_rows: int | None = None  # windows per decode call (None: every ready window)
    first_slack_s: float = 0.25  # a first window's deadline after it became ready
    first_lane_slack_s: float | None = None  # setups before the decode while every waiting continuation has this slack
    hold_slack_s: float | None = None  # continuation-only decodes wait while every ready window keeps this much slack
    hold_max_s: float = 0.1  # ... for at most this long after the oldest ready window arrived
    hold_rows: int = 16  # ... and only while fewer windows are ready
    hold_setups: bool = False  # setups that could ride the held decode wait for it too (fused setups)


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

    def deadline_r6(self, t_ready: float, policy: Policy) -> float:
        """Round 6's deadline (the ``r6`` control): first audio + seconds sent, which ignores pauses after an underrun."""
        if self.t_first_audio is None:
            return t_ready + policy.first_slack_s
        return self.t_first_audio + self.audio_s

    def slack(self, now: float) -> float | None:
        """Seconds of buffered playback left (None before the first audio; negative while stalled)."""
        return None if self.t_exhaust is None else self.t_exhaust - now


def hold_decode(ready: list, now: float, policy: Policy) -> bool:
    """Hold this tick's decode to batch more windows?  ``ready``: (deadline, receipt time, first audio sent) per request
    whose head window is ready.  Never with a first window (TTFA) or a window within ``hold_slack_s`` of its deadline;
    at most ``hold_max_s`` after the oldest window arrived and while fewer than ``hold_rows`` are ready."""
    if policy.hold_slack_s is None or not ready or len(ready) >= policy.hold_rows:
        return False
    if not all(started for _, _, started in ready):
        return False
    return min(d for d, _, _ in ready) - now >= policy.hold_slack_s and now - min(t for _, t, _ in ready) < policy.hold_max_s


def choose_windows(ready: list, policy: Policy) -> list:
    """``ready``: (deadline, key) per request whose head window is ready; returns the keys to decode, earliest first."""
    order = sorted(ready, key=lambda x: x[0])
    if policy.max_decode_rows is not None:
        order = order[:policy.max_decode_rows]
    return [k for _, k in order]


def plan_tick(continuation_waiting: bool, setup_calls: list, policy: Policy) -> tuple[bool, list]:
    """Round 6/7 order (decode first whenever a continuation waits).  ``setup_calls``: (oldest features-ready time,
    call) per pending setup call.  Returns (decode first?, the calls to run this tick, oldest first)."""
    calls = [c for _, c in sorted(setup_calls, key=lambda x: x[0])]
    if continuation_waiting:
        return True, calls[:max(1, policy.setup_calls_when_busy)]
    return False, calls


def first_lane_open(min_deadline: float | None, now: float, policy: Policy) -> bool:
    """May a setup run before the decode?  ``min_deadline``: the earliest deadline of a waiting continuation window
    (None if none waits)."""
    if min_deadline is None:
        return True
    return policy.first_lane_slack_s is not None and min_deadline - now >= policy.first_lane_slack_s


@dataclass
class PrefillPolicy:
    """The engine tick's bounded first-chunk lane: Thinker prefills of newly admitted requests.

    ``token_budget``: prompt tokens prefilled per tick, a hard bound (None: every pending prefill, the rounds 4-6
    behaviour); ``max_age_s``: a prefill that has waited this long may run while a continuing stream is below the slack
    floor (aging: priority, not a budget exemption); ``slack_floor_s``: while any continuing stream has less buffered
    playback than this, only aged prefills run, after the continuing streams' decode (None: never);
    ``talker_start_budget``: Talker chunk-start tokens per tick (None: every pending start)."""

    token_budget: int | None = 2048
    max_age_s: float = 0.25
    slack_floor_s: float | None = 0.3
    talker_start_budget: int | None = 1024


def low_slack(min_slack: float | None, policy: PrefillPolicy) -> bool:
    return policy.slack_floor_s is not None and min_slack is not None and min_slack < policy.slack_floor_s


def choose_prefills(pending: list, now: float, min_slack: float | None, policy: PrefillPolicy, free_pages: int | None = None) -> list:
    """``pending``: (admitted at, prompt tokens still to prefill, key, KV pages it still has to be granted) per request
    awaiting (the rest of) its Thinker prefill -- the pages are its lifetime need (prompt + reply) until its prefill has
    begun, 0 afterwards; ``min_slack``: the least buffered playback among continuing audio streams (None if none);
    ``free_pages``: Thinker KV pages not committed to running requests (None: unchecked).

    Returns [(key, tokens this tick)], oldest first: within ``token_budget`` tokens (the oldest always progresses: alone
    over the budget, it is prefilled in budget-sized chunks); in admission order, so a request the KV pool cannot hold
    yet waits and so do the younger ones (no starvation of long prompts); below the slack floor only aged ones."""
    out, used = [], 0
    urgent = low_slack(min_slack, policy)
    for t, n, k, pages in sorted(pending, key=lambda x: x[0]):
        if urgent and now - t < policy.max_age_s:
            break  # oldest first: every later one is younger
        if free_pages is not None:
            if pages > free_pages:
                break
            free_pages -= pages
        if policy.token_budget is None:
            out.append((k, n))
            continue
        room = policy.token_budget - used
        if n <= room:
            out.append((k, n))
            used += n
        elif not out:
            out.append((k, room))  # the oldest, alone over the budget: one chunk now, the rest in later ticks
            used = policy.token_budget
        else:
            break
        if used >= policy.token_budget:
            break
    return out


def choose_starts(pending: list, policy: PrefillPolicy) -> list:
    """``pending``: (deadline, feed tokens, key) per Talker chunk start; returns the keys to start this tick, earliest
    deadline first, within ``talker_start_budget`` tokens (at least one)."""
    out, used = [], 0
    for _, n, k in sorted(pending, key=lambda x: x[0]):
        if out and policy.talker_start_budget is not None and used + n > policy.talker_start_budget:
            break
        out.append(k)
        used += n
    return out
