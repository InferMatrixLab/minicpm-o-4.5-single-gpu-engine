"""MiniCPM-o 4.5 Thinker language model (Qwen3-8B): reference-precision port.

The checkpoint's ``llm.*`` weights are a Qwen3 causal LM: 36 layers, hidden
4096, 32 query / 8 KV heads of dimension 128, per-head RMSNorm on q and k
before RoPE (theta 1e6), SwiGLU MLP 12288, untied LM head, RMSNorm eps 1e-6.
Inputs are token embeddings, optionally with multimodal embeddings scattered
into placeholder positions by the caller.  ``forward`` returns the last-layer
hidden states (after the final norm, as HF ``hidden_states[-1]`` of the
generation steps that condition the Talker) and the logits.

This is the parity reference and fallback path (plain PyTorch, dense per-row KV
cache); the serving runtime uses the same weights with paged attention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from engine.models.talker import apply_rope, load_prefixed, rms_norm, rope_cos_sin


@dataclass(frozen=True)
class ThinkerConfig:
    hidden_size: int = 4096
    intermediate_size: int = 12288
    num_layers: int = 36
    num_heads: int = 32
    num_kv_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1e6
    rms_eps: float = 1e-6
    vocab_size: int = 151748

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "ThinkerConfig":
        c = json.loads((Path(model_dir) / "config.json").read_text())
        if c.get("rope_scaling") or c.get("use_sliding_window") or c.get("attention_bias") or c.get("tie_word_embeddings"):
            raise ValueError("unsupported Thinker configuration")
        return cls(hidden_size=c["hidden_size"], intermediate_size=c["intermediate_size"], num_layers=c["num_hidden_layers"],
                   num_heads=c["num_attention_heads"], num_kv_heads=c["num_key_value_heads"], head_dim=c["head_dim"],
                   rope_theta=float(c["rope_theta"]), rms_eps=c["rms_norm_eps"], vocab_size=c["vocab_size"])


@dataclass
class KVCache:
    k: torch.Tensor  # [L, B, Hkv, T, D]
    v: torch.Tensor
    lengths: torch.Tensor  # [B]


class Thinker:
    def __init__(self, weights: dict[str, torch.Tensor], cfg: ThinkerConfig):
        self.w, self.cfg = weights, cfg

    @classmethod
    def load(cls, model_dir: str | Path, device="cpu", dtype=torch.bfloat16) -> "Thinker":
        return cls(load_prefixed(model_dir, "llm.", device, dtype), ThinkerConfig.from_model_dir(model_dir))

    @property
    def dtype(self):
        return self.w["model.embed_tokens.weight"].dtype

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return F.embedding(ids, self.w["model.embed_tokens.weight"])

    def new_cache(self, batch: int, max_len: int, device=None) -> KVCache:
        c = self.cfg
        shape = (c.num_layers, batch, c.num_kv_heads, max_len, c.head_dim)
        dev = device or self.w["model.embed_tokens.weight"].device
        return KVCache(torch.zeros(shape, dtype=self.dtype, device=dev), torch.zeros(shape, dtype=self.dtype, device=dev),
                       torch.zeros(batch, dtype=torch.int64, device=dev))

    def forward(self, embeds: torch.Tensor, cache: KVCache, rows: torch.Tensor, all_logits: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        """``embeds`` [B, T, H] for cache ``rows`` at their next positions -> (final-norm hidden [B, T, H], float32 logits).

        Logits cover every position when ``all_logits`` (teacher forcing), else only the last one ([B, V]).
        """
        c, w = self.cfg, self.w
        b, t, _ = embeds.shape
        start = cache.lengths[rows]
        pos = start[:, None] + torch.arange(t, device=embeds.device)[None, :]
        cos, sin = rope_cos_sin(pos, c.head_dim, c.rope_theta, embeds.dtype)
        total = int((start + t).max())
        mask = torch.arange(total, device=embeds.device)[None, None, :] <= pos[:, :, None]
        x = embeds.to(self.dtype)
        for i in range(c.num_layers):
            p = f"model.layers.{i}."
            h = rms_norm(x, w[p + "input_layernorm.weight"], c.rms_eps)
            q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(b, t, c.num_heads, c.head_dim)
            k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(b, t, c.num_kv_heads, c.head_dim)
            v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(b, t, c.num_kv_heads, c.head_dim).transpose(1, 2)
            q = rms_norm(q, w[p + "self_attn.q_norm.weight"], c.rms_eps).transpose(1, 2)
            k = rms_norm(k, w[p + "self_attn.k_norm.weight"], c.rms_eps).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            for j in range(b):
                r, s = int(rows[j]), int(start[j])
                cache.k[i, r, :, s:s + t] = k[j]
                cache.v[i, r, :, s:s + t] = v[j]
            a = F.scaled_dot_product_attention(q, cache.k[i, rows, :, :total], cache.v[i, rows, :, :total], attn_mask=mask[:, None], enable_gqa=True)
            x = x + F.linear(a.transpose(1, 2).reshape(b, t, c.num_heads * c.head_dim), w[p + "self_attn.o_proj.weight"])
            h = rms_norm(x, w[p + "post_attention_layernorm.weight"], c.rms_eps)
            x = x + F.linear(F.silu(F.linear(h, w[p + "mlp.gate_proj.weight"])) * F.linear(h, w[p + "mlp.up_proj.weight"]),
                             w[p + "mlp.down_proj.weight"])
        cache.lengths[rows] = start + t
        hidden = rms_norm(x, w["model.norm.weight"], c.rms_eps)
        logits = F.linear(hidden if all_logits else hidden[:, -1], w["lm_head.weight"]).float()
        return hidden, logits
