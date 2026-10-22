"""The production Token2Wav worker loop (engine/runtime/t2w_worker.py WorkerLoop) with a fake model and clock (CPU).

Burst setups beside continuing speech, long continuing speech, late arrivals, cancellation at every stage, slot reuse,
unusable reference audio, the earliest-deadline choice under a row cap, and the legacy (rounds 4-5) batching policy.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

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
    with pytest.raises(ValueError):
        WorkerLoop(None, None, None, None, Policy(), mode="fifo")


def _speaking(loop, clock, seq=0, buffered_s=2.0):
    """A stream that already sent ``buffered_s`` of audio and has a continuation window waiting."""
    loop.open(seq, seq, Fut())
    loop.tick()
    loop.reqs[seq].audio.sent(int(buffered_s * SR), clock.t)
    loop.window(seq, list(range(25)), False, 1)


def test_first_lane_runs_setups_before_the_decode_while_continuations_have_slack():
    loop, bt, clock, out = _loop(policy=Policy(first_lane_slack_s=0.1), slots=256)
    _speaking(loop, clock)
    for s in range(1, 4):
        loop.open(s, s, Fut())
    n = len(bt.calls)
    loop.tick()
    kinds = [c[0] for c in bt.calls[n:]]
    assert kinds[0] == "setup" and kinds[-1] == "decode"  # new voices first, then the continuation (2 s of slack)


def test_first_lane_closes_when_a_continuation_runs_short_and_is_rechecked_per_call():
    """0.25 s of slack, 0.08 s per setup call, a 0.1 s lane floor: two setup calls fit before the decode must run, then
    at most ``setup_calls_when_busy`` after it."""
    loop, bt, clock, out = _loop(policy=Policy(first_lane_slack_s=0.1, setup_calls_when_busy=1), slots=256)
    _speaking(loop, clock, buffered_s=0.25)
    loop.calls_for = lambda ready: [(512, [r]) for r in ready]  # one setup call per voice
    for s in range(1, 6):
        loop.open(s, s, Fut())
    n = len(bt.calls)
    loop.tick()
    assert [c[0] for c in bt.calls[n:]] == ["setup", "setup", "decode", "setup"]
    # without the lane (the round-7 rule): decode first whenever a continuation waits
    loop, bt, clock, out = _loop(policy=Policy(first_lane_slack_s=None), slots=256)
    _speaking(loop, clock, buffered_s=5.0)
    loop.calls_for = lambda ready: [(512, [r]) for r in ready]
    for s in range(1, 4):
        loop.open(s, s, Fut())
    n = len(bt.calls)
    loop.tick()
    assert [c[0] for c in bt.calls[n:]] == ["decode", "setup"]


def _round6_module():
    """The round-6 policy module (07e1bd3), recovered from git; None if the commit is unavailable."""
    import importlib.util
    import subprocess
    import tempfile

    root = Path(__file__).resolve().parents[2]
    try:
        src = subprocess.run(["git", "-C", str(root), "show", "07e1bd3:engine/runtime/t2w_sched.py"], capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    f = Path(tempfile.mkdtemp()) / "t2w_sched_r6.py"
    f.write_text(src)
    spec = importlib.util.spec_from_file_location("t2w_sched_r6", f)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["t2w_sched_r6"] = mod  # dataclasses resolve their module while the class is built
    spec.loader.exec_module(mod)
    return mod


class Round6Oracle:
    """Round 6's worker tick (07e1bd3 ``_worker_main``: plan_tick, then ``decode_ready`` with its batching gate), over
    the round-6 policy module and the same fake model; the reference the ``r6`` control is checked against."""

    def __init__(self, mod, bt, clock, min_batch=8, max_wait=0.04):
        self.m, self.bt, self.clock, self.min_batch, self.max_wait = mod, bt, clock, min_batch, max_wait
        self.policy = mod.Policy()
        self.reqs, self.out = {}, []

    def open(self, seq):
        self.reqs[seq] = {"seq": seq, "stream": None, "windows": [], "audio": self.m.AudioState(), "t_feat": self.clock(), "offset": 0}

    def window(self, seq, tokens, last, index):
        if seq in self.reqs:
            self.reqs[seq]["windows"].append((tokens, last, index, self.clock()))

    def decode_ready(self):
        ready = [r for r in self.reqs.values() if r["stream"] is not None and r["windows"]]
        if not ready:
            return False
        now = self.clock()
        heads = [r["windows"][0] for r in ready]
        if not (any(h[2] == 0 for h in heads) or any(h[1] for h in heads) or len(ready) >= min(self.min_batch, len(self.reqs))
                or now - min(h[3] for h in heads) >= self.max_wait):
            return False
        dl = {r["seq"]: r["audio"].deadline(r["windows"][0][3], self.policy) for r in ready}
        sel = [self.reqs[k] for k in self.m.choose_windows([(d, k) for k, d in dl.items()], self.policy)]
        items = []
        for r in sel:
            tokens, last, _, _ = r["windows"].pop(0)
            items.append((r["stream"], tokens, last))
        wavs = self.bt.decode(items)
        sent = self.clock()
        for r, (_, _, last), wav in zip(sel, items, wavs):
            self.out.append(("audio", r["seq"], r["offset"], len(wav)))
            r["offset"] += len(wav)
            r["audio"].sent(len(wav), sent)
            if last:
                self.out.append(("done", r["seq"]))
                self.bt.release_stream(self.reqs.pop(r["seq"])["stream"])
        return True

    def tick(self):
        ready = sorted((r for r in self.reqs.values() if r["stream"] is None), key=lambda r: r["t_feat"])
        calls = [(r["t_feat"], (512, [r])) for r in ready]
        waiting = any(r["stream"] is not None and r["windows"] and r["audio"].started for r in self.reqs.values())
        decode_first, run = self.m.plan_tick(waiting, calls, self.policy)
        if decode_first:
            self.decode_ready()
        for _, part in run:
            self.bt.setup_many([object() for _ in part])
            for r in part:
                r["stream"] = self.bt.new_stream(None, noise_seed=0, adopt=True)
        if not decode_first:
            self.decode_ready()


def test_r6_control_makes_the_round6_decisions():
    """The ``r6`` mode against the round-6 implementation on one scripted trace (bursts of arrivals, continuing streams
    with a stall and a refill, a last window): the same setup/decode call sequence and the same audio."""
    mod = _round6_module()
    if mod is None:
        pytest.skip("commit 07e1bd3 not available")
    script = []  # (time, action, args)
    for s in range(6):
        script.append((0.05 * s, "open", (s,)))
    for s in range(6):
        for k in range(8):
            script.append((0.3 + 0.45 * k + 0.01 * s + (1.5 if s == 2 and k > 3 else 0.0), "window", (s, list(range(8 if k == 0 else 25)), k == 7, k)))
    for s in range(6, 14):
        script.append((1.2 + 0.07 * s, "open", (s,)))
        for k in range(3):
            script.append((1.9 + 0.07 * s + 0.4 * k, "window", (s, list(range(8 if k == 0 else 25)), k == 2, k)))
    script.sort(key=lambda x: x[0])

    def drive(tick, open_, window, clock):
        i = 0
        while clock.t < 12.0:
            while i < len(script) and script[i][0] <= clock.t:
                _, act, args = script[i]
                open_(*args) if act == "open" else window(*args)
                i += 1
            tick()
            clock.t += 0.003

    loop, bt, clock, out = _loop("r6", slots=256)
    loop.calls_for = lambda ready: [(512, [r]) for r in ready]
    drive(loop.tick, lambda s: loop.open(s, s, Fut()), loop.window, clock)
    clock2 = Clock()
    bt2 = FakeBT(clock2, slots=256)
    ref = Round6Oracle(mod, bt2, clock2)
    drive(ref.tick, ref.open, ref.window, clock2)
    norm = [(m[0], m[1], m[2], len(m[3])) if m[0] == "audio" else m for m in out]
    assert [c[:2] for c in bt.calls] == [c[:2] for c in bt2.calls]
    assert [round(c[2], 6) for c in bt.calls] == [round(c[2], 6) for c in bt2.calls]
    assert norm == ref.out and len(ref.out) > 60


class FusedBT(FakeBT):
    """FakeBT with ``decode(setups=...)``: the call carries the setups (one decode cost) and returns their prompts."""

    max_flow_streams = 4
    fuse_setup_frames = 250

    def __init__(self, clock, fail=False, **kw):
        super().__init__(clock, **kw)
        self.fail = fail

    def decode(self, items, setups=None):
        if self.fail:
            raise RuntimeError("boom")
        wavs = super().decode(items)
        if setups is None:
            return wavs
        self.calls[-1] = ("decode+setup", len(items), len(setups), self.clock.t)
        return wavs, [object() for _ in setups]


def _fused_loop(policy=None, frames=100, **bt_kw):
    clock = Clock()
    bt = FusedBT(clock, slots=256, **bt_kw)
    out = []
    loop = WorkerLoop(bt, out.append, lambda f: Feat(f.result().frames), lambda ready: [(512, [r]) for r in ready], policy or Policy(first_lane_slack_s=0.1),
                      clock=clock, fuse_setups=True)
    return loop, bt, clock, out


def test_fused_setups_ride_the_decode_call_within_its_room_and_frame_budget():
    """Three new voices (100 frames each) beside a continuing stream: the decode call carries two of them (frame budget
    250), the third runs after it as its own setup (the room is 4 requests, the budget is what binds)."""
    loop, bt, clock, out = _fused_loop()
    _speaking(loop, clock)
    for s in range(1, 4):
        loop.open(s, s, Fut())
    n = len(bt.calls)
    loop.tick()
    assert [c[:2] for c in bt.calls[n:]] == [("decode+setup", 1), ("setup", 1)] and bt.calls[n][2] == 2
    assert all(loop.reqs[s].stream is not None and loop.reqs[s].feat is None for s in (1, 2, 3))
    assert len(_audio(out, 0)) == 1  # the continuation's window was decoded in the fused call
    loop.window(1, list(range(8)), False, 0)
    loop.tick()
    assert len(_audio(out, 1)) == 1  # the fused setup's stream decodes as any other


def test_fused_setup_room_is_the_call_s_request_limit():
    loop, bt, clock, out = _fused_loop()
    bt.fuse_setup_frames = 10_000
    for s in range(3):
        loop.open(s, s, Fut())
    loop.tick()  # their setups, no decode
    for s in range(3):
        loop.reqs[s].audio.sent(2 * SR, clock.t)
        loop.window(s, list(range(25)), False, 1)
    for s in range(3, 6):
        loop.open(s, s, Fut())
    n = len(bt.calls)
    loop.tick()
    assert [c[:3] for c in bt.calls[n:]][0] == ("decode+setup", 3, 1)  # 3 windows + 1 setup = max_flow_streams


def test_a_setup_whose_first_window_waits_runs_before_the_decode():
    """A voice whose first window already arrived would wait a call for its audio if it rode the decode: it takes the
    first lane (its own setup, then the decode that carries its window); the others ride the decode."""
    loop, bt, clock, out = _fused_loop()
    _speaking(loop, clock)
    loop.open(1, 1, Fut())
    loop.open(2, 2, Fut())
    loop.window(2, list(range(8)), False, 0)
    n = len(bt.calls)
    loop.tick()
    assert [c[:3] for c in bt.calls[n:]] == [("setup", 1, bt.calls[n][2]), ("decode+setup", 2, 1)]
    assert len(_audio(out, 2)) == 1 and loop.reqs[1].stream is not None


def test_fused_setups_without_a_decode_run_on_their_own():
    loop, bt, clock, out = _fused_loop()
    loop.open(1, 1, Fut())
    loop.tick()
    assert [c[0] for c in bt.calls] == ["setup"] and loop.reqs[1].stream is not None


def test_a_failed_fused_call_fails_its_windows_and_its_setups():
    loop, bt, clock, out = _fused_loop()
    _speaking(loop, clock)
    loop.open(1, 1, Fut())
    bt.fail = True
    loop.tick()
    errs = sorted((m[1], m[2]) for m in out if m[0] == "error")
    assert errs == [(0, "token2wav: RuntimeError: boom"), (1, "prompt setup: RuntimeError: boom")] and not loop.reqs


def test_fused_setups_need_the_edf_policy():
    with pytest.raises(ValueError):
        WorkerLoop(None, None, None, None, Policy(), mode="r6", fuse_setups=True)
