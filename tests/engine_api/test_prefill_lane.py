"""The engine tick's bounded first-chunk lane (engine/runtime/t2w_sched.choose_prefills and
TrackSBackend._prefill_lane, CPU): oldest first within a token budget, aging, continuation slack priority, and the
scheduler knobs reaching (and being validated by) the backend constructor."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.runtime.t2w_sched import SR, AudioState, PrefillPolicy, choose_prefills  # noqa: E402


def test_budget_oldest_first_and_at_least_one():
    p = PrefillPolicy(token_budget=1000, max_age_s=1.0, slack_floor_s=None)
    pend = [(0.3, 600, "c"), (0.1, 600, "a"), (0.2, 300, "b")]
    assert choose_prefills(pend, 0.5, None, p) == ["a", "b"]  # 900 tokens; c would exceed the budget
    assert choose_prefills([(0.0, 5000, "big")], 0.1, None, p) == ["big"]  # one over-budget prompt still goes
    assert choose_prefills(pend, 0.5, None, PrefillPolicy(token_budget=None, slack_floor_s=None)) == ["a", "b", "c"]


def test_aged_prefills_go_regardless_of_budget_and_slack():
    p = PrefillPolicy(token_budget=100, max_age_s=0.25, slack_floor_s=0.3)
    pend = [(0.0, 600, "old1"), (0.05, 600, "old2"), (0.5, 50, "new")]
    assert choose_prefills(pend, 0.6, None, p) == ["old1", "old2"]  # both aged; "new" exceeds the budget behind them
    assert choose_prefills(pend, 0.6, 0.1, p) == ["old1", "old2"]  # tight continuation: only aged ones
    assert choose_prefills([(0.5, 50, "new")], 0.6, 0.1, p) == []  # tight continuation, nothing aged: wait
    assert choose_prefills([(0.5, 50, "new")], 0.6, 0.5, p) == ["new"]  # enough slack


def _row(seq, pending, t_admit=0.0, audio=None, done=False):
    return SimpleNamespace(seq=seq, pending_prefill=pending, t_admit=t_admit, prompt_ids=[0] * 100, audio=audio, finished=done, cancelled=False)


def test_backend_lane_uses_the_continuing_streams_playback_slack(monkeypatch):
    import engine.runtime.backend_s as B

    monkeypatch.setattr(B, "_now", lambda: 10.0)
    talking = AudioState()
    talking.sent(int(0.1 * SR), 9.95)  # 0.05 s of buffered playback left at t=10
    rows = [_row(1, False, audio=talking), _row(2, True, t_admit=9.9), _row(3, True, t_admit=9.5)]
    fake = SimpleNamespace(live={r.seq: r for r in rows}, audio_live={}, prefill_policy=PrefillPolicy(token_budget=8192, max_age_s=0.25, slack_floor_s=0.3))
    got = B.TrackSBackend._prefill_lane(fake, rows)
    assert [r.seq for r in got] == [1, 3]  # the decoding row, and only the aged prefill while a stream runs dry
    talking.sent(5 * SR, 10.0)  # refilled: 5 s of slack
    assert [r.seq for r in B.TrackSBackend._prefill_lane(fake, rows)] == [1, 2, 3]


def test_scheduler_knobs_are_validated_before_any_model_loads():
    import engine.runtime.backend_s as B

    for bad in ({"t2w_policy": "fifo"}, {"t2w_setup_calls_when_busy": 0}, {"t2w_max_decode_rows": 0}, {"prefill_token_budget": 0},
                {"prefill_slack_floor_ms": -1.0}):
        with pytest.raises(ValueError):
            B.TrackSBackend(**bad)
    with pytest.raises(TypeError):
        B.TrackSBackend(t2w_unknown_knob=1)  # an unknown key is still rejected
