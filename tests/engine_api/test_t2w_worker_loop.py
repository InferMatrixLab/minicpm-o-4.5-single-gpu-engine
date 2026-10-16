"""The production Token2Wav worker loop (engine/runtime/t2w_worker.py WorkerLoop) with a fake model and clock (CPU).

Burst setups beside continuing speech, long continuing speech, late arrivals, cancellation at every stage, slot reuse,
unusable reference audio, the earliest-deadline choice under a row cap, and the legacy (rounds 4-5) batching policy.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.runtime.t2w_sched import SR, Policy  # noqa: E402
from engine.runtime.t2w_worker import WorkerLoop, setup_calls  # noqa: E402

FRAME_S = 1 / 25  # one codec frame is 40 ms of audio


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Fut:
    def __init__(self, frames=100, ok=True, ready=True):
        self.frames, self.ok, self.ready = frames, ok, ready

    def done(self):
        return self.ready

    def result(self):
        if not self.ok:
            raise ValueError("cannot decode audio")
        return self


class Feat:
    """What ``setup_calls`` reads: ``feat[3].shape[1]`` is the prompt mel length."""

    def __init__(self, frames):
        self.shape = (1, frames)

    def __getitem__(self, i):
        return self


class FakeBT:
    """Setup / decode with virtual costs on the clock and a slot pool (8 slots per stream)."""

    def __init__(self, clock, setup_s=0.08, decode_s=0.03, slots=64, hook=None):
        self.clock, self.setup_s, self.decode_s, self.hook = clock, setup_s, decode_s, hook
        self.free = list(range(slots))
        self.calls, self.released_prompts, self.seen_slots, self.streams = [], 0, set(), 0

    def setup_many(self, feats):
        if self.hook:
            self.hook()
        self.clock.t += self.setup_s
        self.calls.append(("setup", len(feats), self.clock.t))
        return [object() for _ in feats]

    def new_stream(self, prompt, noise_seed, adopt):
        st = [self.free.pop() for _ in range(8)]
        self.seen_slots.update(st)
        self.streams += 1
        return st

    def release_stream(self, st):
        self.free.extend(st)

    def release_prompt(self, p):
        self.released_prompts += 1

    def decode(self, items):
        self.clock.t += self.decode_s
        self.calls.append(("decode", len(items), self.clock.t))
        return [np.zeros(int(len(tokens) * FRAME_S * SR), np.float32) for _, tokens, _ in items]


def _loop(mode="edf", policy=None, **bt_kw):
    clock = Clock()
    bt = FakeBT(clock, **bt_kw)
    out = []

    def resolve(f):
        return Feat(f.result().frames)

    loop = WorkerLoop(bt, out.append, resolve, lambda ready: setup_calls(ready, (512, 1024), {512: 8, 1024: 4}), policy or Policy(), mode=mode,
                      clock=clock)
    return loop, bt, clock, out


def _audio(out, seq):
    return [m for m in out if m[0] == "audio" and m[1] == seq]


def test_burst_setups_never_run_ahead_of_a_waiting_continuation():
    """A continuing stream's windows arrive every 0.5 s while 20 voices arrive at once: every tick decodes the waiting
    window before it runs a setup call, and runs at most one setup call while a continuation waits."""
    loop, bt, clock, out = _loop(slots=256)
    loop.open(0, 0, Fut())
    loop.tick()  # setup of request 0
    loop.window(0, list(range(8)), False, 0)
    loop.tick()  # its first window: first audio
    for s in range(1, 21):
        loop.open(s, s, Fut())
    t_next, k = clock.t, 1
    for _ in range(200):
        if clock.t >= t_next:
            loop.window(0, list(range(25)), False, k)
            k, t_next = k + 1, t_next + 0.5
        n_calls = len(bt.calls)
        waiting = bool(loop.reqs[0].windows)
        loop.tick()
        new = bt.calls[n_calls:]
        if waiting:
            assert new and new[0][0] == "decode", new  # the continuation first
            assert sum(c[0] == "setup" for c in new) <= 1, new
        clock.t += 0.002
    assert all(loop.reqs[s].stream is not None for s in range(1, 21))  # the burst completes (no starvation)
    offs = [m[2] for m in _audio(out, 0)]
    assert offs == sorted(offs) and len(offs) == k  # every window decoded, in order


def test_work_conserving_decode_and_the_legacy_batching_gate():
    """A single ready continuation window is decoded in the tick it is ready (edf); the legacy policy holds it until
    min_batch windows wait or max_wait passes."""
    for mode, decoded_at_once in (("edf", True), ("legacy", False)):
        loop, bt, clock, out = _loop(mode)
        for s in range(3):
            loop.open(s, s, Fut())
        loop.tick()
        for s in range(3):
            loop.window(s, list(range(8)), False, 0)
        loop.tick()  # first windows go at once in both policies
        clock.t += 0.001
        loop.window(0, list(range(25)), False, 1)
        n = len(_audio(out, 0))
        loop.tick()
        assert (len(_audio(out, 0)) == n + 1) == decoded_at_once, mode
        if not decoded_at_once:
            clock.t += 0.05  # past max_wait: the legacy gate opens
            loop.tick()
            assert len(_audio(out, 0)) == n + 1


def test_long_speech_with_late_arrivals_and_slot_reuse():
    """Requests arrive while others speak for 30 s; every stream completes with contiguous offsets and every slot comes
    back (slots are reused by the late arrivals)."""
    loop, bt, clock, out = _loop(slots=24)  # three streams at a time
    plans = {s: [list(range(8))] + [list(range(25))] * (60 if s == 0 else 6) for s in range(9)}
    opened, sent = set(), {s: 0 for s in plans}
    for _ in range(3000):
        for s in plans:
            if s not in opened and len(loop.reqs) < 3 and clock.t >= s * 0.7:
                loop.open(s, s, Fut())
                opened.add(s)
            r = loop.reqs.get(s)
            if r is not None and r.stream is not None and sent[s] < len(plans[s]) and len(r.windows) < 2:
                last = sent[s] == len(plans[s]) - 1
                loop.window(s, plans[s][sent[s]], last, sent[s])
                sent[s] += 1
        loop.tick()
        clock.t += 0.01
    done = {m[1] for m in out if m[0] == "done"}
    assert done == set(plans), (done, sent)
    for s in plans:
        a = _audio(out, s)
        offs = [m[2] for m in a]
        lens = [len(m[3]) for m in a]
        assert offs == [sum(lens[:i]) for i in range(len(lens))], s  # contiguous, no gap or overlap
    assert sorted(bt.free) == list(range(24)) and not loop.reqs
    assert bt.streams == 9 and len(bt.seen_slots) <= 24  # nine streams of 8 slots through a pool of 24: reused


def test_cancellation_at_every_stage():
    loop, bt, clock, out = _loop()
    loop.open(1, 1, Fut(ready=False))  # features pending
    loop.close(1)
    loop.open(2, 2, Fut())
    loop.tick()
    loop.open(3, 3, Fut())
    loop._resolve_ready()  # features resolved, setup pending
    loop.close(3)
    loop.window(2, list(range(8)), False, 0)
    loop.tick()
    loop.window(2, list(range(25)), False, 1)  # window queued
    loop.close(2)  # stream holds slots
    loop.tick()
    assert not loop.reqs and sorted(bt.free) == list(range(64))
    assert [m for m in out if m[1] in (1, 3)] == [] and len(_audio(out, 2)) == 1
    loop.close(2)  # after the request is gone: nothing happens
    # closed while its setup call is pending in the same tick (a hook closes it as the call starts)
    loop2, bt2, _, out2 = _loop(hook=lambda: loop2.close(7))
    loop2.open(7, 7, Fut())
    loop2.tick()
    assert 7 not in loop2.reqs and bt2.released_prompts == 1 and sorted(bt2.free) == list(range(64))


def test_unusable_reference_audio_is_a_typed_bad_request():
    loop, bt, clock, out = _loop()
    loop.open(5, 5, Fut(ok=False))
    loop.tick()
    assert out == [("error", 5, "reference audio: ValueError: cannot decode audio", "bad_request")] and not loop.reqs


def test_row_cap_serves_the_stream_closest_to_underrun():
    """B20 through the loop: a stream that stalled and was refilled ranks behind one whose buffer is about to run out."""
    loop, bt, clock, out = _loop(policy=Policy(max_decode_rows=1))
    for s in (1, 2):
        loop.open(s, s, Fut())
    loop.tick()
    loop.reqs[1].audio.sent(SR, 0.0)  # 1 s at t=0, stalled, then 4 s at t=3: buffered until t=7
    loop.reqs[1].audio.sent(4 * SR, 3.0)
    loop.reqs[2].audio.sent(int(1.2 * SR), 4.0)  # buffered until t=5.2
    clock.t = 5.0
    loop.window(1, list(range(25)), False, 2)
    loop.window(2, list(range(25)), False, 2)
    loop.tick()
    assert [m[1] for m in out if m[0] == "audio"] == [2]


def test_unknown_policy_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        WorkerLoop(None, None, None, None, Policy(), mode="fifo")
