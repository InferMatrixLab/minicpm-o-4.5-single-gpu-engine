"""The engine tick's bounded first-chunk lane (engine/runtime/t2w_sched.choose_prefills and
TrackSBackend._prefill_lane, CPU): oldest first within a token budget, aging, continuation slack priority, and the
scheduler knobs reaching (and being validated by) the backend constructor."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.runtime.t2w_sched import SR, AudioState, PrefillPolicy, choose_prefills  # noqa: E402


def test_budget_oldest_first_and_the_oldest_always_progresses():
    p = PrefillPolicy(token_budget=1000, max_age_s=1.0, slack_floor_s=None)
    pend = [(0.3, 600, "c", 0), (0.1, 600, "a", 0), (0.2, 300, "b", 0)]
    assert choose_prefills(pend, 0.5, None, p) == [("a", 600), ("b", 300)]  # 900 tokens; c would exceed the budget
    assert choose_prefills([(0.0, 5000, "big", 0)], 0.1, None, p) == [("big", 1000)]  # alone over the budget: one chunk
    assert choose_prefills(pend, 0.5, None, PrefillPolicy(token_budget=None, slack_floor_s=None)) == [("a", 600), ("b", 300), ("c", 600)]


def test_aging_is_priority_not_a_budget_exemption():
    """The review's B21 counterexample: 32 aged 8000-token prompts with 10 ms of continuation slack select one
    budget-sized chunk, never 256,000 tokens."""
    p = PrefillPolicy(token_budget=8192, max_age_s=0.25, slack_floor_s=0.3)
    pend = [(0.0, 8000, s, 0) for s in range(32)]
    got = choose_prefills(pend, 1.0, 0.01, p)
    assert sum(n for _, n in got) <= 8192 and got[0] == (0, 8000)
    p = PrefillPolicy(token_budget=100, max_age_s=0.25, slack_floor_s=0.3)
    pend = [(0.0, 600, "old1", 0), (0.05, 600, "old2", 0), (0.5, 50, "new", 0)]
    assert choose_prefills(pend, 0.6, None, p) == [("old1", 100)]
    assert choose_prefills(pend, 0.6, 0.1, p) == [("old1", 100)]  # tight continuation: aged ones only, still bounded
    assert choose_prefills([(0.5, 50, "new", 0)], 0.6, 0.1, p) == []  # tight continuation, nothing aged: wait
    assert choose_prefills([(0.5, 50, "new", 0)], 0.6, 0.5, p) == [("new", 50)]  # enough slack


def test_kv_lifetime_admission_is_in_order():
    p = PrefillPolicy(token_budget=None, slack_floor_s=None)
    pend = [(0.0, 100, "a", 50), (0.1, 100, "b", 60), (0.2, 100, "c", 5)]
    assert choose_prefills(pend, 1.0, None, p, free_pages=120) == [("a", 100), ("b", 100), ("c", 100)]
    assert choose_prefills(pend, 1.0, None, p, free_pages=112) == [("a", 100), ("b", 100)]
    assert choose_prefills(pend, 1.0, None, p, free_pages=100) == [("a", 100)]  # b waits, and c behind it (no starvation)
    assert choose_prefills(pend, 1.0, None, p, free_pages=10) == []


def _row(seq, pending, t_admit=0.0, audio=None, done=False):
    return SimpleNamespace(seq=seq, pending_prefill=pending, t_admit=t_admit, prompt_ids=[0] * 100, audio=audio, finished=done, cancelled=False,
                           prefilled=0, prefill_now=0, generated=[], req=SimpleNamespace(max_tokens=64))


def _fake_backend(rows, policy):
    import engine.runtime.backend_s as B
    from engine.runtime.paged import DecoderSpec, PagedKV

    import torch

    kv = PagedKV(DecoderSpec(1, 8, 1, 1, 8, 1e4, 1e-6, True), 64, 16, dtype=torch.float32, device="cpu")
    fake = SimpleNamespace(live={r.seq: r for r in rows}, audio_live={}, prefill_policy=policy, tkv=kv,
                           tdec=SimpleNamespace(pad_pages_outstanding=lambda: 0),
                           sched={"prefill_tokens_tick_max": 0, "prefill_chunked_ticks": 0, "prefill_kv_wait_ticks": 0})
    fake._min_slack = lambda now: B.TrackSBackend._min_slack(fake, now)
    fake._growth = B.TrackSBackend._growth
    return fake


def test_backend_lane_uses_the_continuing_streams_playback_slack(monkeypatch):
    import engine.runtime.backend_s as B

    monkeypatch.setattr(B, "_now", lambda: 10.0)
    talking = AudioState()
    talking.sent(int(0.1 * SR), 9.95)  # 0.05 s of buffered playback left at t=10
    rows = [_row(1, False, audio=talking), _row(2, True, t_admit=9.9), _row(3, True, t_admit=9.5)]
    fake = _fake_backend(rows, PrefillPolicy(token_budget=8192, max_age_s=0.25, slack_floor_s=0.3))
    got, urgent = B.TrackSBackend._prefill_lane(fake, rows)
    assert [r.seq for r in got] == [1, 3] and urgent  # the decoding row, and only the aged prefill while a stream runs dry
    talking.sent(5 * SR, 10.0)  # refilled: 5 s of slack
    got, urgent = B.TrackSBackend._prefill_lane(fake, rows)
    assert [r.seq for r in got] == [1, 2, 3] and not urgent


def test_scheduler_knobs_are_validated_before_any_model_loads():
    import engine.runtime.backend_s as B

    for bad in ({"t2w_policy": "fifo"}, {"t2w_setup_calls_when_busy": 0}, {"t2w_max_decode_rows": 0}, {"prefill_token_budget": 0},
                {"prefill_slack_floor_ms": -1.0}):
        with pytest.raises(ValueError):
            B.TrackSBackend(**bad)
    with pytest.raises(TypeError):
        B.TrackSBackend(t2w_unknown_knob=1)  # an unknown key is still rejected
