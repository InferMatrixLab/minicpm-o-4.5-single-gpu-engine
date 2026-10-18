"""The engine tick (``TrackSBackend._tick``) with timed fake stage runners (CPU): the real prefill lane, KV accounting
(``PagedKV`` on the CPU), Thinker chunking and Talker start selection; fake Thinker/Talker compute that advances a
virtual clock by its cost and delivers each Talker frame's audio at once.

Covers the bound the review's B21 counterexample broke: aged long prefills beside long continuing speech, repeated
arrivals, cancellation during a partial (chunked) prefill, Thinker KV backpressure and page reuse -- every tick's prefill
work stays within the token budget, no KV exhaustion, every request is eventually prefilled, continuing speech does not
stall; and the same workload with the bound removed does stall (the test discriminates).
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import engine.runtime.backend_s as B  # noqa: E402
from engine.runtime.paged import DecoderSpec, PagedKV  # noqa: E402
from engine.runtime.t2w_sched import SR, AudioState, PrefillPolicy  # noqa: E402

PREFILL_S_PER_TOKEN = 30e-6  # ~8B Thinker prefill on one L20X
DECODE_S, TALKER_S = 0.010, 0.012  # one Thinker decode step, one Talker K-step tick
K, FRAMES_PER_CHUNK, FRAME_S = 4, 50, 0.04  # ~10 text tokens speak ~2 s


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


class Loop:
    @staticmethod
    def call_soon_threadsafe(f, *a):
        f(*a)


class Out(list):
    def put_nowait(self, ev):
        self.append(ev)


class Stall:
    """Playback-clock bookkeeping of one stream: seconds the client's buffer was empty after its first audio."""

    def __init__(self, audio: AudioState):
        self.audio, self.s = audio, 0.0

    def sent(self, n, now):
        a = self.audio
        if a.t_exhaust is not None and now > a.t_exhaust:
            self.s += now - a.t_exhaust
        a.sent(n, now)


class FakeTDec:
    """Thinker decode: one KV token per row (real accounting), a reply of ``reply[seq]`` tokens then <|im_end|>."""

    def __init__(self, be, clock):
        self.be, self.clock = be, clock

    def pad_pages_outstanding(self):
        return 0

    def release_pads(self):
        pass

    def step(self, seqs, toks):
        self.be.tkv.layout(seqs, [1] * len(seqs), device="cpu")
        self.clock.t += DECODE_S
        out = []
        for s in seqs:
            lv = self.be.live[s]
            out.append(B.IM_END if len(lv.generated) >= lv.reply else 7)
        return out, torch.zeros(len(seqs), 8)


def make(policy: PrefillPolicy, pages=1024, page_size=64):
    clock = Clock()
    be = object.__new__(B.TrackSBackend)
    spec = DecoderSpec(1, 8, 1, 1, 8, 1e4, 1e-6, True)
    be.tkv = PagedKV(spec, pages, page_size, dtype=torch.float32, device="cpu")
    be.akv = PagedKV(spec, 64 * 19, 128, dtype=torch.float32, device="cpu")  # every utterance's lifetime commitment fits
    be.talker_commit, be.talker_pad_pages, be.talker_k, be.talker_k_max = {}, 0, K, K
    be.stage_trace_dir, be.aprefill = None, None
    be.tdec = FakeTDec(be, clock)
    be.prefill_policy = policy
    be.sched = {"prefill_token_budget": policy.token_budget, "prefill_tokens_tick_max": 0, "prefill_chunked_ticks": 0,
                "prefill_kv_wait_ticks": 0, "talker_starts_deferred_ticks": 0, "talker_kv_wait_ticks": 0}
    be.live, be.audio_live = {}, {}
    be.thinker_lazy, be.profile, be.split_threads = False, False, False
    be.t2wp = SimpleNamespace(send=lambda *a: None)
    be.tspec = SimpleNamespace(hidden_size=8)
    be.thinker = SimpleNamespace(dtype=torch.float32)
    be.device = "cpu"
    be.tok = SimpleNamespace(decode=lambda ids: "x")
    be.ticks = []  # per tick: prefill tokens executed, start time, end time

    def prefill_forward(pre):
        ns = [lv.prefill_now or len(lv.prompt_ids) - lv.prefilled for lv in pre]
        be.tkv.layout([lv.seq for lv in pre], ns, device="cpu")
        be.ticks[-1][0] += sum(ns)
        clock.t += PREFILL_S_PER_TOKEN * sum(ns)
        out = []
        for lv, n in zip(pre, ns):
            lv.prefilled += n
            lv.prefill_now = 0
            out.append(7 if lv.prefilled >= len(lv.prompt_ids) else None)
        return out

    def talker_step(rows):
        for lv in be._talker_starts([lv for lv in rows if not lv.talking and lv.chunks]):
            lv.talking, lv.chunk_frames = True, 0
        talking = [lv for lv in rows if lv.talking]
        if not talking:
            return
        clock.t += TALKER_S
        for lv in talking:
            lv.chunk_frames += K
            lv.frames += K
            lv.stall.sent(int(K * FRAME_S * SR), clock.t)
            if lv.chunk_frames >= FRAMES_PER_CHUNK:
                ch = lv.chunks.pop(0)
                lv.talking = False
                if ch.final:
                    lv.utterance_done = True

    be._prefill_forward = prefill_forward
    be._talker_step = talker_step
    return be, clock


def admit(be, clock, seq, prompt, reply, t=None, buffered_s=0.0):
    req = SimpleNamespace(max_tokens=max(reply + 1, 16), ref_audio=b"voice", modalities=["text", "audio"], seed=seq, talker_sampling={})
    lv = B.LiveS(req=req, out=Out(), loop=Loop(), prompt_ids=[1] * prompt, seq=seq)
    lv.t_admit, lv.audio, lv.reply = clock.t if t is None else t, AudioState(), reply
    lv.stall = Stall(lv.audio)
    if buffered_s:
        lv.stall.sent(int(buffered_s * SR), clock.t)
    be.live[seq] = lv
    return lv


def run(be, clock, ticks, arrivals=None):
    for i in range(ticks):
        if arrivals:
            arrivals(i)
        be.ticks.append([0, clock.t, None])
        be._tick()
        be.ticks[-1][2] = clock.t
        clock.t += 0.001
        if not be.live:
            break


def continuing(be, clock, n=4):
    """Streams already speaking (long replies), each with 2 s of buffered playback."""
    out = []
    for s in range(n):
        lv = admit(be, clock, 1000 + s, 30, 400, buffered_s=2.0)
        lv.pending_prefill, lv.prefilled = False, 30
        be.tkv.layout([lv.seq], [30], device="cpu")
        lv.generated = [7] * 11
        lv.chunk_index = 1
        lv.hidden = [torch.zeros(8)] * 11
        lv.passed = 11
        lv.chunks = [B.TextChunk([7] * 10, torch.zeros(10, 8), False) for _ in range(2)]  # text the Talker has yet to speak
        out.append(lv)
    return out


def test_aged_long_prefills_stay_bounded_beside_continuing_speech(monkeypatch):
    """The review's counterexample: 32 pending 8000-token prompts, admitted at t=0, examined at t=1 with a continuing
    stream 10 ms from running dry.  Every tick prefills at most the budget, no KV exhaustion, all 32 are prefilled in
    the end, and the continuing streams never stall."""
    pol = PrefillPolicy(token_budget=2048, max_age_s=0.25, slack_floor_s=0.3)
    be, clock = make(pol)
    monkeypatch.setattr(B, "_now", clock)
    talk = continuing(be, clock)
    clock.t = 1.0
    talk[0].stall.audio.t_exhaust = 1.01  # 10 ms of slack left on one stream
    longs = [admit(be, clock, s, 8000, 20, t=0.0) for s in range(32)]
    run(be, clock, 4000)
    assert max(t[0] for t in be.ticks) <= 2048 and be.sched["prefill_tokens_tick_max"] <= 2048
    assert be.sched["prefill_chunked_ticks"] > 0  # 8000 > 2048: chunked
    assert all(not lv.pending_prefill for lv in longs) and all(lv.utterance_done for lv in longs)
    assert max(lv.stall.s for lv in talk) <= 0.02  # at most the 10 ms it started with, never a stall caused here
    assert be.tkv.used_pages <= sum(len(be.tkv.pages.get(lv.seq, [])) for lv in be.live.values())


def test_the_same_workload_without_the_bound_stalls(monkeypatch):
    """Discrimination: with no token budget (every pending prefill in one tick, KV still checked) the continuing
    streams run dry."""
    be, clock = make(PrefillPolicy(token_budget=None, max_age_s=0.25, slack_floor_s=None))
    monkeypatch.setattr(B, "_now", clock)
    talk = continuing(be, clock)
    clock.t = 1.0
    for s in range(32):
        admit(be, clock, s, 8000, 20, t=0.0)
    run(be, clock, 4000)
    assert max(t[0] for t in be.ticks) > 2048
    assert max(lv.stall.s for lv in talk) > 0.5


def test_kv_backpressure_waits_instead_of_exhausting_the_pool(monkeypatch):
    """A pool that holds two 8000-token lifetimes: the rest wait their turn (no MemoryError), in admission order; every
    page comes back."""
    be, clock = make(PrefillPolicy(token_budget=2048, max_age_s=0.25, slack_floor_s=0.3), pages=300)
    monkeypatch.setattr(B, "_now", clock)
    reqs = [admit(be, clock, s, 8000, 20, t=0.001 * s) for s in range(8)]
    order = []
    orig = be._prefill_forward

    def spy(pre):
        order.extend(lv.seq for lv in pre if lv.seq not in order)
        live_prefills = {lv.seq for lv in be.live.values() if lv.prefilled and lv.pending_prefill}
        assert len(live_prefills | {lv.seq for lv in pre}) <= 2  # 126 pages each, 300 in the pool
        return orig(pre)

    be._prefill_forward = spy
    run(be, clock, 8000)
    assert be.sched["prefill_kv_wait_ticks"] > 0
    assert order == sorted(order) == [r.seq for r in reqs]
    assert all(r.utterance_done for r in reqs) and be.tkv.used_pages == 0 and len(be.tkv.free) == 300


def test_repeated_arrivals_all_prefilled_and_short_prompts_unaffected(monkeypatch):
    """New short requests every few ticks beside continuing speech and two long prompts: aging gets every one through,
    first-chunk prefills of short prompts are not held behind the long ones beyond the budget."""
    be, clock = make(PrefillPolicy(token_budget=2048, max_age_s=0.25, slack_floor_s=0.3))
    monkeypatch.setattr(B, "_now", clock)
    talk = continuing(be, clock, n=2)
    longs = [admit(be, clock, 500 + s, 7000, 20) for s in range(2)]
    shorts = []

    def arrive(i):
        if i % 5 == 0 and len(shorts) < 40:
            shorts.append(admit(be, clock, len(shorts), 40, 15))

    run(be, clock, 6000, arrive)
    assert len(shorts) == 40 and all(not lv.pending_prefill for lv in shorts + longs)
    assert be.sched["prefill_tokens_tick_max"] <= 2048
    assert max(lv.stall.s for lv in talk) == 0.0


def test_cancellation_during_a_partial_prefill_returns_its_pages(monkeypatch):
    be, clock = make(PrefillPolicy(token_budget=1024, max_age_s=0.25, slack_floor_s=None))
    monkeypatch.setattr(B, "_now", clock)
    lv = admit(be, clock, 1, 5000, 20)
    run(be, clock, 2)
    assert lv.pending_prefill and 0 < lv.prefilled < 5000  # chunked: part of its prompt is in the KV
    assert be.tkv.used_pages > 0
    lv.cancelled = True
    run(be, clock, 2)
    assert 1 not in be.live and be.tkv.used_pages == 0 and len(be.tkv.free) == 1024
    nxt = admit(be, clock, 2, 5000, 20)  # the pages are reused by the next request
    run(be, clock, 4000)
    assert nxt.utterance_done and be.tkv.used_pages == 0


def test_talker_starts_are_bounded_and_ordered_by_playback_deadline(monkeypatch):
    be, clock = make(PrefillPolicy(talker_start_budget=24))
    monkeypatch.setattr(B, "_now", clock)
    clock.t = 5.0
    rows = []
    for s, (started, exhaust) in enumerate([(True, 5.4), (False, None), (True, 5.05), (True, 9.0)]):
        lv = admit(be, clock, s, 10, 10, t=4.9)
        if started:
            lv.audio.sent(SR, 4.0)
            lv.audio.t_exhaust = exhaust
        lv.chunks = [B.TextChunk([1] * 10, torch.zeros(10, 8), False)]
        rows.append(lv)
    got = [lv.seq for lv in be._talker_starts(rows)]
    # deadlines: 5.4, 4.9+0.25=5.15 (first chunk), 5.05, 9.0; 12 tokens each, 24 per tick -> the two earliest
    assert sorted(got) == [1, 2]
    assert be.sched["talker_starts_deferred_ticks"] == 1
    assert [lv.seq for lv in be._talker_starts(rows[:1])] == [0]  # at least one, whatever its size


def test_backend_constructor_validates_the_new_knobs():
    for bad in ({"talker_start_budget": 0}, {"t2w_first_lane_slack_ms": -1.0}, {"t2w_policy": "r7"}):
        with pytest.raises(ValueError):
            B.TrackSBackend(**bad)
