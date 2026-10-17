"""Memory-safe WavLM self-attention for the SIM scorer (evaluator setting ``sim.attention = "query_blocked_v1"``).

The pinned scorer (s3prl ``wavlm.modules.MultiheadAttention``, inference fast path) materialises the relative-position
bias of the whole batch, ``[B*H, T, T]`` fp32, keeps it across all 24 layers, and builds a gated copy plus the
key-padding sum per layer.  One 82 s output padded into a 16 + 16 batch makes each of these 34 GB.

``install()`` replaces that fast path with one that computes, per block of ``QUERY_BLOCK`` query rows over the complete
key/value sequence, exactly the elements the original computes for those rows: the relative-position bucket lookup
(``compute_bias`` on the block's positions), the GRU gate (computed once on the whole query, as before) times that
bias, plus the key-padding mask, then the same ``scaled_dot_product_attention`` call on the block's queries.  The
projections, the padding, the batch, the weights and the precision are unchanged; only the attention is evaluated
in row blocks, and each row's arithmetic does not depend on which other rows share its call.  Equivalence with the
original on every batch it can score is checked, not assumed (``tools/sim_equivalence.py``).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

QUERY_BLOCK = 256
VERSION = "query_blocked_v1"


class RelativeBias:
    """The bias rows of query positions [a, b) against all keys, from the layer that owns the bias table."""

    def __init__(self, attn, tgt_len: int, src_len: int):
        self.attn, self.tgt_len, self.src_len = attn, tgt_len, src_len

    def rows(self, a: int, b: int) -> torch.Tensor:  # [H, b - a, S]
        m = self.attn
        context = torch.arange(self.tgt_len, dtype=torch.long)[a:b, None]
        memory = torch.arange(self.src_len, dtype=torch.long)[None, :]
        bucket = m._relative_positions_bucket(memory - context, bidirectional=True).to(m.relative_attention_bias.weight.device)
        return m.relative_attention_bias(bucket).permute([2, 0, 1])


def _forward(self, query, key, value, key_padding_mask=None, incremental_state=None, need_weights=True, static_kv=False, attn_mask=None,
             before_softmax=False, need_head_weights=False, position_bias=None):
    fast = (query.device.type != "xla" and incremental_state is None and not static_kv and not torch.jit.is_scripting()
            and self.q_head_dim == self.head_dim and not need_weights and not need_head_weights and not before_softmax
            and attn_mask is None and self.bias_k is None and not self.add_zero_attn and not self.training)
    if not fast:
        raise RuntimeError("query-blocked WavLM attention covers the scorer's inference path only")
    tgt_len, bsz, embed_dim = query.size()
    src_len = key.size(0)
    H, hd = self.num_heads, self.head_dim
    if self.has_relative_attention_bias and position_bias is None:
        position_bias = RelativeBias(self, tgt_len, src_len)  # instead of the [B*H, T, S] tensor; passed on to later layers
    gate = None
    if position_bias is not None and self.gru_rel_pos:  # as the original: on the whole query, [B, H, T, 1]
        query_layer = query.transpose(0, 1)
        query_layer = query_layer.view(*(query_layer.size()[:-1] + (H, -1))).permute(0, 2, 1, 3)
        _B, _H, _L, __ = query_layer.size()
        gate_a, gate_b = torch.sigmoid(self.grep_linear(query_layer).view(_B, _H, _L, 2, 4).sum(-1, keepdim=False)).chunk(2, dim=-1)
        gate = gate_a * (gate_b * self.grep_a - 1.0) + 2.0
    k_bias = self.k_proj.bias if self.k_proj.bias is not None else torch.zeros_like(self.q_proj.bias)
    b_q, b_k, b_v = torch.cat((self.q_proj.bias, k_bias, self.v_proj.bias)).chunk(3)
    q, k, v = F._in_projection(query, key, value, self.q_proj.weight, self.k_proj.weight, self.v_proj.weight, b_q, b_k, b_v)
    kpm = F._canonical_mask(mask=key_padding_mask, mask_name="key_padding_mask", other_type=None if position_bias is None else query.dtype,
                            other_name="attn_mask", target_type=query.dtype)
    q = q.view(tgt_len, bsz * H, hd).transpose(0, 1).view(bsz, H, tgt_len, hd)
    k = k.view(src_len, bsz * H, hd).transpose(0, 1).view(bsz, H, src_len, hd)
    v = v.view(src_len, bsz * H, hd).transpose(0, 1).view(bsz, H, src_len, hd)
    if kpm is not None:
        kpm = kpm.view(bsz, 1, 1, src_len).expand(-1, H, -1, -1).reshape(bsz * H, 1, src_len)
    out = []
    for a in range(0, tgt_len, QUERY_BLOCK):
        b = min(tgt_len, a + QUERY_BLOCK)
        mask = None
        if position_bias is not None:
            rows = position_bias.rows(a, b).unsqueeze(0).expand(bsz, -1, -1, -1).reshape(bsz * H, b - a, src_len)
            mask = gate[:, :, a:b].reshape(bsz * H, -1, 1) * rows if gate is not None else rows
        if kpm is not None:
            mask = kpm if mask is None else mask + kpm
        if mask is not None:
            mask = mask.view(bsz, H, -1, src_len)
        out.append(F.scaled_dot_product_attention(q[:, :, a:b], k, v, mask, 0.0, False))
    attn = torch.cat(out, dim=2).permute(2, 0, 1, 3).reshape(bsz * tgt_len, embed_dim)
    attn = F.linear(attn, self.out_proj.weight, self.out_proj.bias).view(tgt_len, bsz, embed_dim)
    return attn, None, position_bias


def install() -> None:
    """Route every s3prl WavLM attention layer through ``_forward`` (call before the scorer is built)."""
    from s3prl.upstream.wavlm import modules

    assert math.isfinite(QUERY_BLOCK) and QUERY_BLOCK > 0
    modules.MultiheadAttention.forward = _forward
