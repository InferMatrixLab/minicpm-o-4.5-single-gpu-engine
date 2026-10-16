"""Token2Wav worker scheduling policy (engine/runtime/t2w_sched.py, CPU): playback deadlines, EDF window order, the
bounded setup lane while continuation windows wait, and the absence of starvation in a simulated worker loop."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.runtime.t2w_sched import SR, AudioState, Policy, choose_windows, plan_tick  # noqa: E402


def test_deadline_follows_the_first_audio_state():
    p = Policy(first_slack_s=0.25)
    a = AudioState()
    assert not a.started and a.deadline(10.0, p) == 10.25  # TTFA lane: ready time + slack
    a.sent(SR, 11.0)  # 1 s of audio at t = 11
    a.sent(SR // 2, 11.4)
    assert a.started and a.t_first_audio == 11.0 and abs(a.deadline(99.0, p) - 12.5) < 1e-9  # playback runs out at 12.5
    b = AudioState()
    b.sent(0, 5.0)  # an empty chunk is not first audio
    assert not b.started


def test_windows_are_chosen_by_earliest_deadline_and_capped():
    ready = [(3.0, "c"), (1.0, "a"), (2.0, "b"), (0.5, "z")]
    assert choose_windows(ready, Policy()) == ["z", "a", "b", "c"]
    assert choose_windows(ready, Policy(max_decode_rows=2)) == ["z", "a"]


def test_setup_lane_is_bounded_while_continuation_waits():
    calls = [(3.0, "s3"), (1.0, "s1"), (2.0, "s2")]
    first, run = plan_tick(True, calls, Policy(setup_calls_when_busy=1))
    assert first and run == ["s1"]  # decode first, then only the oldest setup call
    first, run = plan_tick(True, calls, Policy(setup_calls_when_busy=2))
    assert run == ["s1", "s2"]
    first, run = plan_tick(False, calls, Policy())
    assert not first and run == ["s1", "s2", "s3"]  # nothing continuing: every setup, oldest first
    assert plan_tick(True, calls, Policy(setup_calls_when_busy=0))[1] == ["s1"]  # never zero: setups always progress


def _simulate(policy: Policy, ticks: int = 400, arrive_every: int = 2, windows_per_req: int = 12, setup_s: float = 0.08,
              decode_s: float = 0.04, window_every_s: float = 0.5):
    """A worker loop on a virtual clock: a request arrives every ``arrive_every`` ticks; once set up its windows (1 s of
    audio each) become ready every ``window_every_s`` (the Talker runs faster than real time); each setup call costs
    ``setup_s``, each decode ``decode_s``.
    Returns per-request setup waits (ticks), the largest decode wait of a ready window (s), and the worst lateness."""
    t, reqs, setups_wait, worst_wait, worst_late, n = 0.0, {}, {}, 0.0, 0.0, 0
    for tick in range(ticks):
        if tick % arrive_every == 0:
            reqs[n] = {"setup": False, "t_feat": t, "tick_feat": tick, "audio": AudioState(), "left": windows_per_req, "t_ready": None}
            n += 1
        waiting = any(r["setup"] and r["t_ready"] is not None and r["audio"].started for r in reqs.values())
        calls = [(r["t_feat"], k) for k, r in reqs.items() if not r["setup"]]

        def decode():
            nonlocal t, worst_wait, worst_late
            ready = [(r["audio"].deadline(r["t_ready"], policy), k) for k, r in reqs.items() if r["setup"] and r["t_ready"] is not None
                     and r["t_ready"] <= t]
            sel = choose_windows(ready, policy)
            if not sel:
                return
            t += decode_s
            for k in sel:
                r = reqs[k]
                worst_wait = max(worst_wait, t - r["t_ready"])
                if r["audio"].started:
                    worst_late = max(worst_late, t - r["audio"].deadline(r["t_ready"], policy))
                r["audio"].sent(SR, t)  # each window is 1 s of audio
                r["left"] -= 1
                r["t_ready"] = None if r["left"] == 0 else r["t_ready"] + window_every_s
                if r["left"] == 0:
                    del reqs[k]

        first, run = plan_tick(waiting, calls, policy)
        if first:
            decode()
        for k in run:
            t += setup_s
            r = reqs[k]
            r["setup"], r["t_ready"] = True, t
            setups_wait[k] = tick - r["tick_feat"]
        if not first:
            decode()
        t += 0.002
    return setups_wait, worst_wait, worst_late, reqs


def test_no_starvation_under_continuous_arrivals():
    """Setups keep arriving: every one still runs (bounded wait), and continuation windows are never late."""
    waits, worst_wait, worst_late, left = _simulate(Policy(setup_calls_when_busy=1))
    assert len(waits) > 150 and max(waits.values()) <= 1, max(waits.values())  # one per tick keeps up with one per 2 ticks
    assert worst_late <= 0.0, worst_late  # every continuation window decoded before its playback deadline
    # windows arriving exactly in real time have no slack: lateness stays within one bounded setup call and a decode
    _, _, late_rt, _ = _simulate(Policy(setup_calls_when_busy=1), window_every_s=1.0)
    assert late_rt <= 0.08 + 2 * 0.04 + 1e-9, late_rt
    # a burst of arrivals larger than the lane: setups queue, but age to the front and all complete
    waits, _, late, _ = _simulate(Policy(setup_calls_when_busy=1), arrive_every=1, ticks=200)
    assert sorted(waits) == list(range(len(waits))) and max(waits.values()) < 200
    assert late <= 0.0, late


def test_capped_decode_rows_serve_the_earliest_deadlines_without_starvation():
    """With fewer rows per call than ready windows, EDF rotates through them: no window waits unboundedly."""
    _, wait_short, late, _ = _simulate(Policy(max_decode_rows=4), arrive_every=3, ticks=300)
    _, wait_long, late_long, _ = _simulate(Policy(max_decode_rows=4), arrive_every=3, ticks=1200)
    assert wait_long <= wait_short + 1e-9, (wait_short, wait_long)  # the longest wait does not grow with the run
    assert late <= 0.0 and late_long <= 0.0, (late, late_long)  # windows wait only while their audio is buffered


def test_playback_clock_follows_the_harness_stall_model_through_underruns():
    """B20: 1 s at t=0, then 1 s at t=3 (the buffer ran out at t=1: a 2 s stall) -> exhausted at t=4, not t=2.  Over a
    random delivery sequence the clock's stalls add up to the harness's ``playback_stall_s``."""
    import numpy as np

    from bench.metrics import playback_stall_s

    a = AudioState()
    a.sent(SR, 0.0)
    a.sent(SR, 3.0)
    assert a.t_exhaust == 4.0 and a.deadline(0.0, Policy()) == 4.0 and a.slack(3.5) == 0.5
    rng = np.random.default_rng(3)
    for _ in range(50):
        times = np.cumsum(rng.uniform(0.0, 1.6, 30))
        durs = rng.uniform(0.04, 1.2, 30)
        st, stalls = AudioState(), 0.0
        for t, d in zip(times, durs):
            if st.t_exhaust is not None and t > st.t_exhaust:
                stalls += t - st.t_exhaust  # the buffer was empty from t_exhaust until this delivery
            st.sent(int(round(d * SR)), float(t))
        cum = np.cumsum([int(round(d * SR)) / SR for d in durs])
        assert abs(stalls - playback_stall_s(list(times), list(cum))) < 1e-9


def test_edf_prefers_the_stream_about_to_underrun_over_a_refilled_stalled_one():
    stalled, tight = AudioState(), AudioState()
    stalled.sent(SR, 0.0)
    stalled.sent(4 * SR, 3.0)  # stalled 1 s .. 3 s, then refilled: runs out at t=7
    tight.sent(int(1.2 * SR), 4.0)  # runs out at t=5.2
    p = Policy(max_decode_rows=1)
    assert choose_windows([(stalled.deadline(0, p), "stalled"), (tight.deadline(0, p), "tight")], p) == ["tight"]
