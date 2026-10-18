"""Asynchronous decode results in the Token2Wav worker loop (``WorkerLoop(async_results=True)``) with a fake model (CPU).

A decode call returns a pending result; the loop emits each call's audio and ``done`` messages only once that call's
samples reached the host, in call order, while it keeps running setups and later calls.  Checked: emission order
across calls and within a request, ``done`` after a request's last audio, a request closed while its call is pending
(its samples dropped, the call kept until it retires), the in-flight bound, failures surfacing at completion, and the
same messages as the synchronous loop for a mixed workload.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.runtime.t2w_sched import Policy  # noqa: E402
from engine.runtime.t2w_worker import WorkerLoop, setup_calls  # noqa: E402

from test_t2w_worker_loop import Clock, FakeBT, Feat, Fut  # noqa: E402


class Pending:
    def __init__(self, wavs, fail=False):
        self.wavs, self.fail, self.is_ready, self.waited = wavs, fail, False, False

    def ready(self):
        return self.is_ready

    def result(self):
        self.waited = not self.is_ready
        self.is_ready = True
        if self.fail:
            raise RuntimeError("device error")
        return self.wavs


class AsyncBT(FakeBT):
    """``decode(items, async_result=True)`` returns a ``Pending`` the test completes; samples encode (seq, window)."""

    def __init__(self, clock, **kw):
        super().__init__(clock, **kw)
        self.pending, self.fail_next = [], False

    def decode(self, items, async_result=False):
        wavs = [np.full(int(len(tokens)), float(tokens[0]), np.float32) for _, tokens, _ in items]
        self.clock.t += self.decode_s
        self.calls.append(("decode", len(items), self.clock.t))
        if not async_result:
            return wavs
        p = Pending(wavs, fail=self.fail_next)
        self.fail_next = False
        self.pending.append(p)
        return p


def _loop(async_results=True, max_inflight=2):
    clock = Clock()
    bt = AsyncBT(clock, slots=256)
    out = []
    loop = WorkerLoop(bt, out.append, lambda f: Feat(f.result().frames), lambda ready: setup_calls(ready, (512, 1024), {512: 8, 1024: 4}),
                      Policy(), clock=clock, async_results=async_results, max_inflight=max_inflight)
    return loop, bt, clock, out


def _ready(loop, seqs):
    for s in seqs:
        loop.open(s, s, Fut())
    loop.tick()  # setups
    assert all(loop.reqs[s].stream is not None for s in seqs)


def _tok(seq, k, n=25):
    return [1000 * seq + k] * n  # the fake decoder's samples carry (seq, window)


def test_calls_are_emitted_in_call_order_after_their_copy():
    loop, bt, clock, out = _loop()
    _ready(loop, [0, 1])
    loop.window(0, _tok(0, 0), False, 0)
    loop.tick()  # call A (request 0, window 0)
    loop.window(0, _tok(0, 1), False, 1)
    loop.window(1, _tok(1, 0), False, 0)
    loop.tick()  # call B (both); setups and decodes went on without waiting
    assert len(bt.pending) == 2 and not [m for m in out if m[0] == "audio"]
    bt.pending[1].is_ready = True  # B finished first
    loop.tick()
    assert not [m for m in out if m[0] == "audio"]  # nothing ahead of A
    bt.pending[0].is_ready = True
    loop.tick()
    audio = [(m[1], int(m[3][0])) for m in out if m[0] == "audio"]
    assert audio == [(0, 0), (0, 1), (1, 1000)]  # A, then B; per request in window order
    offs = [m[2] for m in out if m[0] == "audio" and m[1] == 0]
    assert offs == [0, 25]


def test_done_follows_the_last_audio_and_closes_at_emission():
    loop, bt, clock, out = _loop()
    _ready(loop, [0])
    loop.window(0, _tok(0, 0), False, 0)
    loop.tick()
    loop.window(0, _tok(0, 1, 13), True, 1)
    loop.tick()
    assert 0 in loop.reqs  # the final window is pending: the request stays open
    for p in bt.pending:
        p.is_ready = True
    loop.tick()
    assert [m[0] for m in out if m[1] == 0 and m[0] != "mark"] == ["audio", "audio", "done"]
    assert 0 not in loop.reqs


def test_a_request_closed_while_pending_gets_no_samples():
    loop, bt, clock, out = _loop()
    _ready(loop, [0, 1])
    loop.window(0, _tok(0, 0), False, 0)
    loop.window(1, _tok(1, 0), False, 0)
    loop.tick()
    loop.close(1)  # the client went away while the call runs
    assert len(loop.inflight) == 1  # the call (and its buffer) stays until it retires
    bt.pending[0].is_ready = True
    loop.tick()
    assert [m[1] for m in out if m[0] == "audio"] == [0]
    assert not loop.inflight


def test_in_flight_calls_are_bounded():
    loop, bt, clock, out = _loop(max_inflight=2)
    _ready(loop, [0])
    for k in range(3):
        loop.window(0, _tok(0, k), False, k)
        loop.tick()
    assert len(bt.pending) == 3 and bt.pending[0].waited  # the third call waited for the first
    assert len(loop.inflight) == 2
    assert [int(m[3][0]) for m in out if m[0] == "audio"] == [0]


def test_a_failed_call_fails_its_live_requests_at_completion():
    loop, bt, clock, out = _loop()
    _ready(loop, [0, 1])
    bt.fail_next = True
    loop.window(0, _tok(0, 0), False, 0)
    loop.window(1, _tok(1, 0), False, 0)
    loop.tick()
    bt.pending[0].is_ready = True
    loop.tick()
    assert sorted(m[1] for m in out if m[0] == "error") == [0, 1] and not loop.reqs


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_same_messages_as_the_synchronous_loop(seed):
    """A mixed workload (ramp, steady and final windows, staggered requests, a cancellation): every request's audio
    and done messages are identical in content and order with and without asynchronous results."""
    rng = np.random.default_rng(seed)
    plan = []  # (tick, action, args)
    for s in range(6):
        t = int(rng.integers(0, 5))
        plan.append((t, "open", s))
        n = int(rng.integers(2, 6))
        for k in range(n):
            plan.append((t + 1 + k, "window", (s, _tok(s, k, 8 if k == 0 else 25 if k < n - 1 else 11), k == n - 1, k)))
    plan.append((6, "close", 3))
    msgs = {}
    for mode in (False, True):
        loop, bt, clock, out = _loop(async_results=mode)
        for tick in range(40):
            for t, action, arg in plan:
                if t == tick:
                    if action == "open":
                        loop.open(arg, arg, Fut())
                    elif action == "window":
                        loop.window(*arg)
                    else:
                        loop.close(arg)
            loop.tick()
            for p in bt.pending:  # the copies complete a tick later
                p.is_ready = True
            clock.t += 0.01
        msgs[mode] = {s: [(m[0], m[2], m[3].tolist()) if m[0] == "audio" else (m[0],) for m in out if m[1] == s and m[0] in ("audio", "done")]
                      for s in range(6)}
    for s in range(6):
        if s == 3:  # cancelled: the asynchronous loop may have dropped a call that the synchronous one emitted before
            assert msgs[True][s] == msgs[False][s][:len(msgs[True][s])]
            continue
        assert msgs[True][s] == msgs[False][s] and msgs[True][s][-1] == ("done",)
