"""Talker sampler against the HF reference order (sampler unit tests with fixed logits)."""

import pytest
import torch
from transformers.generation.logits_process import TopKLogitsWarper, TopPLogitsWarper

from engine import talker_sampler as ts

V, EOS = 6562, 6561  # num_audio_tokens; EOS = num_audio_tokens - 1
CFG = ts.TalkerSampling()


class HFRepetitionPenalty:
    """Verbatim logic of CustomRepetitionPenaltyLogitsProcessorRepeat (modeling_minicpmo.py:4969-4991)."""

    def __init__(self, penalty, max_input_ids, past_window):
        self.penalty, self.max_input_ids, self.past_window = penalty, max_input_ids, past_window

    def __call__(self, input_ids, scores):
        if input_ids.size(1) > self.past_window:
            input_ids = input_ids.narrow(1, -self.past_window, self.past_window)
        freq = torch.nn.functional.one_hot(input_ids, scores.size(1)).sum(1)
        if freq.size(0) > self.max_input_ids:
            freq.narrow(0, self.max_input_ids, freq.size(0) - self.max_input_ids).zero_()
        alpha = torch.pow(self.penalty, freq)
        scores = scores.contiguous()
        return torch.where(scores < 0, scores.multiply(alpha), scores.divide(alpha))


def hf_processed(logits, prev, t, cfg=CFG, swap=False):
    """MiniCPMTTS.generate (modeling_minicpmo.py:4433-4458) for one row with t previous frames ``prev``."""
    x = logits.float().clone()[None] / cfg.temperature
    if t > 0:
        ids = torch.tensor([prev], dtype=torch.long)
        x = HFRepetitionPenalty(cfg.repetition_penalty, V, 16)(ids, x)
        warpers = [TopPLogitsWarper(cfg.top_p, min_tokens_to_keep=3), TopKLogitsWarper(cfg.top_k, min_tokens_to_keep=3)]
        for w in (reversed(warpers) if swap else warpers):
            x = w(ids, x)
    if t < cfg.min_new_tokens:
        x[:, EOS] = -torch.inf
    return x[0]


def _state_rows(b, t, gen):
    prev = [torch.randint(0, 300, (t,), generator=gen).tolist() for _ in range(b)]  # a small alphabet so penalties repeat
    hist = torch.full((b, 16), ts.PAD, dtype=torch.long)
    for i, p in enumerate(prev):
        tail = p[-16:]
        if tail:
            hist[i, 16 - len(tail):] = torch.tensor(tail)
    return prev, hist


@pytest.mark.parametrize("t", [1, 5, 16, 40, 60])
def test_hf_order(t):
    gen = torch.Generator().manual_seed(t)
    logits = torch.randn(6, V, generator=gen) * 3
    prev, hist = _state_rows(6, t, gen)
    got = ts.processed_logits(logits, hist, torch.full((6,), t), EOS, CFG)
    for i in range(6):
        want = hf_processed(logits[i], prev[i], t)
        assert torch.equal(torch.isinf(got[i]), torch.isinf(want))
        fin = ~torch.isinf(want)
        assert torch.allclose(got[i][fin], want[fin], rtol=1e-6, atol=1e-6)


def test_swapped_top_k_top_p_is_detected():
    gen = torch.Generator().manual_seed(7)
    logits = torch.randn(1, V, generator=gen) * 0.5  # a flat distribution: top-p alone keeps far more than 25 tokens
    prev, hist = _state_rows(1, 20, gen)
    ours = ts.processed_logits(logits, hist, torch.tensor([20]), EOS, CFG)[0]
    swapped = hf_processed(logits[0], prev[0], 20, swap=True)
    assert not torch.equal(torch.isinf(ours), torch.isinf(swapped))  # the reverse order keeps a different support


def test_frame_zero_skips_penalty_and_warpers_and_min_length_masks_eos():
    logits = torch.randn(2, V)
    got = ts.processed_logits(logits, torch.full((2, 16), ts.PAD), torch.zeros(2, dtype=torch.long), EOS, CFG)
    want = logits / CFG.temperature
    want[:, EOS] = -torch.inf
    assert torch.equal(got, want)
    for t, masked in ((49, True), (50, False)):
        x = ts.processed_logits(torch.zeros(1, V).index_fill_(1, torch.tensor([EOS]), 50.0), torch.full((1, 16), 3), torch.tensor([t]), EOS, CFG)
        assert torch.isinf(x[0, EOS]) == masked


def test_eos_forced_at_max_length():
    x = ts.processed_logits(torch.randn(1, V), torch.full((1, 16), 3), torch.tensor([CFG.max_new_tokens - 1]), EOS, CFG)
    assert torch.isfinite(x[0, EOS]) and torch.isinf(x[0, :EOS]).all()


def test_counter_rng_is_position_keyed_and_batch_independent():
    seeds = torch.tensor([11, 12, 13, 2**40 + 5])
    u1 = ts.uniform(seeds, torch.tensor([0, 1, 2, 3]))
    assert torch.equal(u1, ts.uniform(seeds, torch.tensor([0, 1, 2, 3])))
    assert u1.min() >= 0 and u1.max() < 1 and len(set(u1.tolist())) == 4
    many = ts.uniform(torch.full((20000,), 9), torch.arange(20000))
    assert abs(float(many.mean()) - 0.5) < 0.01
    logits = torch.randn(8, V)
    alone = ts.step(ts.DecodeState.new([1234]), logits[3:4], EOS, CFG)
    batch = ts.step(ts.DecodeState.new([1, 2, 3, 1234, 5, 6, 7, 8]), logits, EOS, CFG)
    assert alone[0] == batch[3]


def test_k_step_equals_single_steps_and_eos_freezes():
    gen = torch.Generator().manual_seed(3)
    logits = torch.randn(70, 4, V, generator=gen)
    logits[60, 1, EOS] = 1e4  # row 1 ends at frame 60, inside a K block
    a, b = ts.DecodeState.new([5, 6, 7, 8]), ts.DecodeState.new([5, 6, 7, 8])
    singles = torch.stack([ts.step(a, logits[i], EOS, CFG) for i in range(70)])
    ks = torch.cat([ts.decode_k(b, logits[i:i + 8], EOS, CFG) for i in range(0, 70, 8)])
    assert torch.equal(singles, ks)
    assert singles[60, 1] == EOS and (singles[61:, 1] == ts.PAD).all() and bool(b.finished[1])
    assert int(b.steps[1]) == 61 and (singles[:, 0] != ts.PAD).all()


def test_padded_and_cancelled_rows_never_change_valid_rows():
    gen = torch.Generator().manual_seed(4)
    logits = torch.randn(30, 4, V, generator=gen)
    ref = ts.DecodeState.new([1, 2, 3, 4])
    want = torch.stack([ts.step(ref, logits[i], EOS, CFG) for i in range(30)])
    s = ts.DecodeState.new([1, 2, 3, 4])
    s.active[2] = False  # cancelled before the first frame
    noisy = logits.clone()
    noisy[:, 2] = torch.randn(30, V, generator=gen) * 100
    got = torch.stack([ts.step(s, noisy[i], EOS, CFG) for i in range(30)])
    assert torch.equal(got[:, [0, 1, 3]], want[:, [0, 1, 3]]) and (got[:, 2] == ts.PAD).all()
    assert int(s.steps[2]) == 0 and (s.history[2] == ts.PAD).all()


def test_fused_top_p_top_k_equals_two_pass():
    """top_p_top_k_filter keeps exactly what top_k_filter(top_p_filter(.)) keeps (random, peaked and flat logits)."""
    from engine.talker_sampler import top_k_filter, top_p_filter, top_p_top_k_filter

    g = torch.Generator().manual_seed(0)
    mism = total = 0
    for scale in (0.3, 1.0, 3.0, 8.0):
        x = torch.randn(64, 6562, generator=g) * scale
        a = top_k_filter(top_p_filter(x, 0.85, 3), 25, 3)
        b = top_p_top_k_filter(x, 0.85, 25, 3)
        mism += int((torch.isinf(a) != torch.isinf(b)).any(dim=-1).sum())
        total += x.shape[0]
    assert mism == 0, (mism, total)
