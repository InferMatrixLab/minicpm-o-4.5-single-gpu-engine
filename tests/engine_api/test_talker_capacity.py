"""Talker KV completion safety (B22) through the production tick (CPU): the real ``TrackSBackend._tick``,
``_talker_step``, ``_take_frame`` and ``PagedKV``; a scripted Talker (zero conditioning, fixed frame counts per chunk)
and a decoder stand-in with the real decoder's KV operations (bucket padding rows, a K-frame reservation per row).

The review's counterexample: 64 utterances that each grow beyond 1024 Talker KV tokens on the 512 x 128 pool.  Without
lifetime commitments the pool runs out mid-utterance and the tick raises (every live request would fail); with them
starts wait in admission order and every utterance completes.
"""

import queue
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import engine.runtime.backend_s as B  # noqa: E402
import engine.talker_sampler as TS  # noqa: E402
from engine.runtime.paged import DecoderSpec, PagedKV  # noqa: E402
from engine.runtime.t2w_sched import AudioState, PrefillPolicy  # noqa: E402
from engine.server import BadRequest  # noqa: E402

EOS, FRAME, H = 6562, 100, 8
BUCKETS = (1, 2, 4, 8, 16, 32, 64)


class Out(list):
    def put_nowait(self, ev):
        self.append(ev)


class Loop:
    @staticmethod
    def call_soon_threadsafe(f, *a):
        f(*a)


class Chunker:
    def push(self, toks):
        return []

    def finish(self):
        return None


class FakeDecoder:
    """``TalkerDecoder.decode_k`` bookkeeping: padding rows up to the bucket reserve K frames on scratch sequences, every
    row reserves K frames up front and its length advances by K; codes follow each request's per-chunk frame plan."""

    def __init__(self, be):
        self.be, self.buckets, self.max_len, self.pad_peak = be, {b: None for b in BUCKETS}, 4096, 0

    def decode_k(self, seqs, state, k):
        kv, out = self.be.akv, []
        for a in range(0, len(seqs), max(BUCKETS)):
            part = seqs[a:a + max(BUCKETS)]
            pads = [(1 << 40) + i for i in range(min(b for b in BUCKETS if b >= len(part)) - len(part))]
            for r in part + pads:
                kv.reserve(r, k)
            self.pad_peak = max(self.pad_peak, sum(len(kv.pages[p]) for p in pads))
            for r in part:
                kv.lengths[r] += k
            for p in pads:
                kv.release(p)
        for s in seqs:
            lv = self.be.live[s]
            plan, c = lv.plan[len(lv.plan) - len(lv.chunks)], lv.chunk_frames
            col = []
            for j in range(k):
                col.append(EOS if c + j >= plan else FRAME)
                if col[-1] == EOS:
                    break
            out.append(col + [TS.PAD] * (k - len(col)))
        return torch.tensor(out, dtype=torch.int64).T


def make(pages=512, page_size=128, max_live=64):
    be = object.__new__(B.TrackSBackend)
    spec = DecoderSpec(1, H, 1, 1, H, 1e4, 1e-6, False)
    be.tkv = PagedKV(spec, 64, 64, dtype=torch.float32, device="cpu")
    be.akv = PagedKV(spec, pages, page_size, dtype=torch.float32, device="cpu")
    layout = be.akv.layout
    be.akv.layout = lambda seqs, n, device="cpu": layout(seqs, n, device="cpu")
    be.prefill_policy = PrefillPolicy(talker_start_budget=1024)
    be.sched = {"prefill_token_budget": None, "prefill_tokens_tick_max": 0, "prefill_chunked_ticks": 0, "prefill_kv_wait_ticks": 0,
                "talker_starts_deferred_ticks": 0, "talker_kv_wait_ticks": 0}
    be.live, be.audio_live, be.inbox = {}, {}, queue.Queue()
    be.thinker_lazy, be.profile, be.split_threads = False, False, False
    be.t2wp = SimpleNamespace(send=lambda *a: None)
    be.device, be.max_live, be.talker_k, be.talker_k_max = "cpu", max_live, 4, 4
    be.cfg = TS.TalkerSampling(min_new_tokens=0, max_new_tokens=1 << 30)
    be.talker = SimpleNamespace(cfg=SimpleNamespace(eos=EOS), condition=lambda ids, hidden: torch.zeros(len(ids) + 2, H),
                                code_embed=lambda codes: torch.zeros(len(codes), H), head=torch.zeros(EOS + 1, H))
    be.arun = SimpleNamespace(forward=lambda x, lay: torch.zeros(x.shape[0], H))
    be.adec = FakeDecoder(be)
    be.talker_commit, be.all = {}, []
    be.talker_pad_pages = (max(BUCKETS) - 1) * -(-4 // page_size)
    return be


def add(be, seq, plan, t, frame_cap=2048, max_tokens=256):
    """An audio request whose Thinker is done: one 10-token text chunk per entry of ``plan`` (that chunk's frames)."""
    req = SimpleNamespace(max_tokens=max_tokens, ref_audio=b"voice", modalities=["text", "audio"], seed=seq, request_id=str(seq),
                          talker_sampling={"max_new_tokens": frame_cap})
    lv = B.LiveS(req=req, out=Out(), loop=Loop(), prompt_ids=[1], seq=seq, frame_cap=frame_cap)
    lv.pending_prefill, lv.thinker_done, lv.t_admit, lv.audio, lv.chunker, lv.plan = False, True, t, AudioState(), Chunker(), plan
    lv.chunks = [B.TextChunk([7] * 10, torch.zeros(10, H), i == len(plan) - 1) for i in range(len(plan))]
    be.live[seq] = lv
    be.all.append(lv)
    return lv


@pytest.fixture(autouse=True)
def scripted_first_frame(monkeypatch):
    monkeypatch.setattr(TS, "step", lambda st, logits, eos, cfg: torch.full((logits.shape[0],), FRAME, dtype=torch.int64))


def run(be, done, max_ticks=20000, arrivals=None, cancel=None):
    """Production ticks until every request is done; per tick: arrivals, cancellations, KV and commitment invariants."""
    peak = 0
    for tick in range(max_ticks):
        for seq, plan, kw in (arrivals or {}).get(tick, []):
            add(be, seq, plan, t=float(tick), **kw)
        for seq in (cancel or {}).get(tick, []):
            be.live[seq].cancelled = True
        with torch.inference_mode():
            be._tick()
        for lv in [x for x in done if x.utterance_done or x.cancelled]:
            assert lv.seq not in be.talker_commit or lv.seq in be.live
        committed = sum(be.talker_commit.values())
        assert committed + be.talker_pad_pages <= be.akv.num_pages
        for seq, pages in be.talker_commit.items():
            assert len(be.akv.pages.get(seq, [])) <= pages
        peak = max(peak, len(be.talker_commit))
        if not be.live and not (arrivals and any(t > tick for t in arrivals)):
            return tick, peak
    raise AssertionError("requests did not finish")


def errors(lvs):
    return [ev for lv in lvs for ev in lv.out if isinstance(ev, Exception)]


def growing(n=64, chunk_frames=44, chunks=25, pages=512):
    """The review's state reached through real ticks: every utterance grows past 1024 Talker KV tokens."""
    be = make(pages=pages)
    lvs = [add(be, i, [chunk_frames] * chunks, t=float(i)) for i in range(n)]
    return be, lvs


def test_lifetime_bound_covers_the_canonical_request():
    t = B.talker_lifetime(256, 2048, 4)
    assert B.thinker_bound(256) == 261 and t == 261 + 2 * 26 + 1 + 2048 + 4
    assert -(-t // 128) == 19 and 64 * 19 + 63 <= 1280  # configs/engine/track_s_fast.json: 64 live requests always fit


def test_review_counterexample_exhausts_the_pool_without_commitments(monkeypatch):
    be, lvs = growing()
    monkeypatch.setattr(be, "_grant_talker", lambda starts: starts)  # no lifetime commitments, no commitment check
    monkeypatch.setattr(be, "_beyond_commit", lambda lv, extra: False)
    with pytest.raises(MemoryError, match="KV pool exhausted"):
        run(be, lvs)


@pytest.mark.parametrize("n,chunk_frames,chunks", [(64, 44, 25), (32, 80, 25)])  # >1024 and >2048 KV tokens per utterance
def test_growing_utterances_complete_without_pool_errors(n, chunk_frames, chunks):
    be, lvs = growing(n, chunk_frames, chunks)
    _, peak = run(be, lvs)
    assert not errors(lvs)
    assert all(lv.utterance_done and lv.frames == chunk_frames * chunks for lv in lvs)
    assert peak == (512 - 63) // 19 < n  # starts waited for capacity ...
    assert be.sched["talker_kv_wait_ticks"] > 0
    assert not be.talker_commit and len(be.akv.free) == be.akv.num_pages  # ... and every page came back
    assert be.adec.pad_peak <= be.talker_pad_pages


def test_first_starts_are_granted_in_admission_order():
    be, lvs = growing(40)
    order = []
    grant = be._grant_talker

    def spy(starts):
        before = set(be.talker_commit)
        out = grant(starts)
        order.extend(sorted(set(be.talker_commit) - before, key=lambda s: be.live[s].t_admit))
        return out

    be._grant_talker = spy
    run(be, lvs)
    assert order == sorted(order, key=lambda s: lvs[s].t_admit) == list(range(40))


def test_per_chunk_limit_carry_and_page_crossings_stay_within_the_commitment():
    be = make()
    lvs = [add(be, i, [600, 7, 130], t=float(i)) for i in range(8)]  # 600 > 500: cut at the limit, last frame carried
    run(be, lvs)
    assert not errors(lvs)
    assert all(lv.frames == 500 + 7 + 130 for lv in lvs)  # the decoder offered 600; the reference cuts at 500


def test_frame_cap_ends_the_utterance_with_finish_reason_length():
    be = make()
    lvs = [add(be, 0, [400] * 5, t=0.0, frame_cap=1000), add(be, 1, [30] * 3, t=1.0, frame_cap=1000)]
    run(be, lvs)
    assert lvs[0].frames == 1000 and lvs[0].finish_reason == "length" and lvs[0].utterance_done
    assert lvs[1].frames == 90 and lvs[1].finish_reason == "stop"
    assert not be.talker_commit and len(be.akv.free) == be.akv.num_pages


def test_cancellation_releases_the_commitment_and_later_arrivals_reuse_it():
    be, lvs = growing(30)
    arrivals = {50: [(100 + i, [44] * 25, {}) for i in range(30)]}
    run(be, lvs, arrivals=arrivals, cancel={20: list(range(0, 30, 3))})
    assert len(be.all) == 60 and not errors(be.all)
    assert all(lv.utterance_done for lv in be.all if lv.seq >= 100)
    assert not be.talker_commit and len(be.akv.free) == be.akv.num_pages


def test_requests_that_can_never_fit_are_rejected_at_admission():
    be = make()
    req = SimpleNamespace(max_tokens=4096, ref_audio=b"v", modalities=["text", "audio"], talker_sampling={})
    with pytest.raises(BadRequest, match="Talker"):
        be._check_capacity(req)  # beyond the Talker's 4096 positions
    small = make(pages=80)
    with pytest.raises(BadRequest, match="pooled tokens"):
        small._check_capacity(SimpleNamespace(max_tokens=256, ref_audio=b"v", modalities=["text", "audio"], talker_sampling={}))
    small._check_capacity(SimpleNamespace(max_tokens=256, ref_audio=None, modalities=["text"], talker_sampling={}))  # text only


def test_a_broken_bound_fails_only_its_own_request(monkeypatch):
    be, lvs = growing(4, 44, 6)
    need = be._talker_need
    monkeypatch.setattr(be, "_talker_need", lambda req: 1 if req.request_id == "2" else need(req))
    run(be, lvs)
    bad = errors([lvs[2]])
    assert len(bad) == 1 and "committed 1 pages" in str(bad[0])
    assert not errors([lvs[0], lvs[1], lvs[3]]) and all(lvs[i].utterance_done for i in (0, 1, 3))
