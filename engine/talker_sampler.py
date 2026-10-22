"""Talker codec-token sampler in the HF reference order, batched and device-resident.

Reference: ``MiniCPMTTS.generate`` / ``TTSStreamingGenerator`` in the MiniCPM-o 4.5
``modeling_minicpmo.py`` / ``utils.py`` (``gen_logits``): per step

1. ``logits / temperature``;
2. frames after the first: the repetition penalty over the last 16 generated
   frames (``CustomRepetitionPenaltyLogitsProcessorRepeat``: a token seen ``n``
   times is multiplied by ``penalty**n`` when negative, divided otherwise);
3. frames after the first: top-p (``TopPLogitsWarper``, ``min_tokens_to_keep=3``),
   then top-k (``TopKLogitsWarper``, ``k = max(top_k, 3)``);
4. EOS masked while ``t < min_new_tokens``; EOS forced at ``t == max_new_tokens - 1``;
5. softmax and one draw.

The draw uses a counter-based uniform keyed by (request seed, frame index), so
a row's tokens do not depend on the batch it runs in, on padding, or on how
many frames one decode call covers.  Rows that are inactive (padding,
cancelled, finished) return ``PAD`` and never touch another row's state.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

PAD = -1
M32 = 0xFFFFFFFF


@dataclass(frozen=True)
class TalkerSampling:
    temperature: float = 0.8
    top_p: float = 0.85
    top_k: int = 25
    repetition_penalty: float = 1.05
    window: int = 16
    min_new_tokens: int = 50
    max_new_tokens: int = 2048
    min_keep: int = 3


def _mix32(x: torch.Tensor) -> torch.Tensor:
    """32-bit integer finalizer (lowbias32) on int64 lanes holding values < 2**32."""
    x = x & M32
    x = x ^ (x >> 16)
    x = (x * 0x7FEB352D) & M32
    x = x ^ (x >> 15)
    x = (x * 0x846CA68B) & M32  # wraps modulo 2**64; the low 32 bits are exact
    return x ^ (x >> 16)


def uniform(seeds: torch.Tensor, steps: torch.Tensor) -> torch.Tensor:
    """U[0, 1) per row from (seed, frame index) only."""
    s = seeds.to(torch.int64)
    h = _mix32(s & M32)
    h = _mix32(h ^ ((s >> 32) & M32) ^ 0x9E3779B9)
    h = _mix32(h ^ (steps.to(torch.int64) & M32) ^ 0x632BE5AB)
    return (h >> 8).to(torch.float32) * (1.0 / (1 << 24))


def repetition_penalty(x: torch.Tensor, history: torch.Tensor, penalty: float) -> torch.Tensor:
    """``history``: [B, W] previous frames (most recent last), ``PAD`` where empty."""
    valid = history >= 0
    freq = torch.zeros_like(x).scatter_add_(1, history.clamp(min=0), valid.to(x.dtype))
    alpha = torch.pow(torch.full_like(x, penalty), freq)
    return torch.where(x < 0, x * alpha, x / alpha)


def top_p_filter(x: torch.Tensor, top_p: float, min_keep: int) -> torch.Tensor:
    sorted_logits, sorted_idx = torch.sort(x, descending=False, dim=-1)
    cum = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    remove = cum <= (1 - top_p)
    remove[..., -min_keep:] = False
    return x.masked_fill(remove.scatter(1, sorted_idx, remove), float("-inf"))


def top_k_filter(x: torch.Tensor, top_k: int, min_keep: int) -> torch.Tensor:
    k = min(max(top_k, min_keep), x.size(-1))
    kth = torch.topk(x, k, dim=-1).values[..., -1:]
    return x.masked_fill(x < kth, float("-inf"))


def top_p_top_k_filter(x: torch.Tensor, top_p: float, top_k: int, min_keep: int) -> torch.Tensor:
    """``top_k_filter(top_p_filter(x))`` without sorting the vocabulary.

    Top-p keeps a prefix of the descending order: rank ``j`` survives when the mass ranked above it is below ``top_p``
    (equivalently its ascending cumulative mass exceeds ``1 - top_p``), plus ``min_keep``; top-k then keeps the
    ``max(top_k, min_keep)`` largest of those.  Both are a value threshold at the last kept rank, which lies within the
    top ``k`` ranks, so ``topk`` and the full-vocabulary normaliser suffice.  The masses are summed over the top ranks
    instead of the tail, so a token within float rounding of the top-p boundary can fall the other way.
    """
    k = min(max(top_k, min_keep), x.size(-1))
    vals = torch.topk(x, k, dim=-1).values  # descending
    p = torch.exp(vals - torch.logsumexp(x, dim=-1, keepdim=True))
    above = p.cumsum(dim=-1) - p
    kept = (above < top_p).sum(dim=-1).clamp(min=min(min_keep, k), max=k)
    thr = vals.gather(1, (kept - 1)[:, None])
    return x.masked_fill(x < thr, float("-inf"))


def processed_logits(logits: torch.Tensor, history: torch.Tensor, steps: torch.Tensor, eos: int, cfg: TalkerSampling) -> torch.Tensor:
    """Steps 1-4 for every row; rows at frame 0 skip the penalty and the warpers."""
    x = logits.float() / cfg.temperature
    later = steps > 0
    if bool(later.any()):
        y = top_p_top_k_filter(repetition_penalty(x, history, cfg.repetition_penalty), cfg.top_p, cfg.top_k, cfg.min_keep)
        x = torch.where(later[:, None], y, x)
    col = torch.arange(x.size(-1), device=x.device)[None, :] == eos
    x = x.masked_fill(col & (steps < cfg.min_new_tokens)[:, None], float("-inf"))
    forced = (steps >= cfg.max_new_tokens - 1)[:, None]
    return torch.where(forced & ~col, torch.full_like(x, float("-inf")), torch.where(forced & col, torch.zeros_like(x), x))


def draw(x: torch.Tensor, seeds: torch.Tensor, steps: torch.Tensor, u: torch.Tensor | None = None) -> torch.Tensor:
    """``u``: ``uniform(seeds, steps)`` when already computed (``engine.runtime.kernels.counter_uniform``)."""
    probs = torch.softmax(x, dim=-1)
    cdf = probs.cumsum(dim=-1)
    u = (uniform(seeds, steps) if u is None else u).to(cdf.dtype)[:, None] * cdf[:, -1:]
    return torch.searchsorted(cdf, u, right=True).squeeze(-1).clamp_(max=x.size(-1) - 1)


@dataclass
class DecodeState:
    """Per-row Talker sampling state: frame index, last ``window`` frames, finished flag."""

    seeds: torch.Tensor  # [B] int64
    steps: torch.Tensor  # [B] int64, next frame index
    history: torch.Tensor  # [B, W] int64, PAD where empty
    finished: torch.Tensor  # [B] bool
    active: torch.Tensor  # [B] bool: False for padding or cancelled rows

    @classmethod
    def new(cls, seeds: list[int], cfg: TalkerSampling = TalkerSampling(), device="cpu") -> "DecodeState":
        b = len(seeds)
        return cls(seeds=torch.tensor(seeds, dtype=torch.int64, device=device), steps=torch.zeros(b, dtype=torch.int64, device=device),
                   history=torch.full((b, cfg.window), PAD, dtype=torch.int64, device=device),
                   finished=torch.zeros(b, dtype=torch.bool, device=device), active=torch.ones(b, dtype=torch.bool, device=device))


def step(state: DecodeState, logits: torch.Tensor, eos: int, cfg: TalkerSampling = TalkerSampling()) -> torch.Tensor:
    """Sample one frame per row and advance the state; inactive or finished rows return ``PAD`` and do not change."""
    live = state.active & ~state.finished
    tok = draw(processed_logits(logits, state.history, state.steps, eos, cfg), state.seeds, state.steps)
    tok = torch.where(live, tok, torch.full_like(tok, PAD))
    shifted = torch.cat([state.history[:, 1:], tok[:, None]], dim=1)
    state.history = torch.where(live[:, None], shifted, state.history)
    state.steps = state.steps + live.to(torch.int64)
    state.finished = state.finished | (live & (tok == eos))
    return tok


def decode_k(state: DecodeState, logits_k: torch.Tensor, eos: int, cfg: TalkerSampling = TalkerSampling()) -> torch.Tensor:
    """K frames from precomputed per-frame logits [K, B, V]: identical to K calls of ``step``; an EOS freezes its row."""
    return torch.stack([step(state, logits_k[i], eos, cfg) for i in range(logits_k.size(0))])
