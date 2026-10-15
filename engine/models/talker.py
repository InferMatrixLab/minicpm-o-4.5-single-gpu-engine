"""MiniCPM-o 4.5 Talker (``MiniCPMTTS``): conditioning, Llama backbone and codec head.

Port of ``MiniCPMTTS`` and ``MiniCPMO._generate_speech_non_streaming`` from the
model's ``modeling_minicpmo.py`` (Apache-2.0, openbmb/MiniCPM-o-4_5 revision
503e754).  Weights are read directly from the checkpoint (``tts.*``).

Conditioning (``condition_type = hidden_text_merge``): for the Thinker reply
tokens between ``<|tts_bos|>`` and ``<|tts_eos|>``,
``emb_text(ids) + normalize(projector_semantic(h))`` where ``h`` is the
Thinker's last-layer hidden state at those positions, followed by
``emb_text(text_eos)`` and ``emb_text(audio_bos)``.  The backbone is a 20-layer
Llama (``LlamaConfig`` defaults: RoPE theta 1e4, RMSNorm eps 1e-6, SiLU MLP);
frame ``t`` feeds ``emb_code(code[t-1])`` at position ``condition_len + t - 1``.
The codec head is a weight-normalised linear layer (``weight = g * v / ||v||``
per output row).

This module is the reference-precision implementation (plain PyTorch, a dense
per-row KV cache); the fused decode graph and paged attention build on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open


@dataclass(frozen=True)
class TalkerConfig:
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_layers: int = 20
    num_heads: int = 12
    num_kv_heads: int = 12
    rope_theta: float = 10000.0
    rms_eps: float = 1e-6
    num_audio_tokens: int = 6562
    audio_bos_token_id: int = 151687
    text_eos_token_id: int = 151692
    max_positions: int = 4096
    normalize_projected_hidden: bool = True

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def eos(self) -> int:
        return self.num_audio_tokens - 1

    @classmethod
    def from_model_dir(cls, model_dir: str | Path) -> "TalkerConfig":
        t = json.loads((Path(model_dir) / "config.json").read_text())["tts_config"]
        if t.get("backbone_model") != "llama" or t.get("condition_type") != "hidden_text_merge" or t.get("projector_type") != "mlp" or t.get("num_vq") != 1:
            raise ValueError(f"unsupported Talker configuration: {t.get('backbone_model')}, {t.get('condition_type')}, {t.get('projector_type')}")
        return cls(hidden_size=t["hidden_size"], intermediate_size=t["intermediate_size"], num_layers=t["num_hidden_layers"],
                   num_heads=t["num_attention_heads"], num_kv_heads=t["num_key_value_heads"], num_audio_tokens=t["num_audio_tokens"],
                   audio_bos_token_id=t["audio_bos_token_id"], text_eos_token_id=t["text_eos_token_id"],
                   max_positions=t["max_position_embeddings"], normalize_projected_hidden=t["normalize_projected_hidden"])


def load_prefixed(model_dir: str | Path, prefix: str, device="cpu", dtype=None) -> dict[str, torch.Tensor]:
    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())["weight_map"]
    files = sorted({f for k, f in index.items() if k.startswith(prefix)})
    out = {}
    for f in files:
        with safe_open(str(model_dir / f), framework="pt", device=str(device)) as fh:
            for k in fh.keys():
                if k.startswith(prefix):
                    t = fh.get_tensor(k)
                    out[k[len(prefix):]] = t.to(dtype) if dtype is not None and t.is_floating_point() else t
    return out


def rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    xf = x.float()
    return (w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype))


def rope_cos_sin(positions: torch.Tensor, head_dim: int, theta: float, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    inv = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=positions.device, dtype=torch.int64).float() / head_dim))
    freqs = positions.float()[..., None] * inv
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B, H, T, D]; cos/sin: [B, T, D]."""
    half = x.shape[-1] // 2
    rot = torch.cat([-x[..., half:], x[..., :half]], dim=-1)
    return x * cos[:, None] + rot * sin[:, None]


class Talker:
    def __init__(self, weights: dict[str, torch.Tensor], cfg: TalkerConfig):
        self.cfg = cfg
        self.w = weights
        g, v = weights["head_code.0.parametrizations.weight.original0"], weights["head_code.0.parametrizations.weight.original1"]
        self.head = (g.float() * v.float() / v.float().norm(dim=1, keepdim=True)).to(v.dtype)

    @classmethod
    def load(cls, model_dir: str | Path, device="cpu", dtype=torch.bfloat16) -> "Talker":
        return cls(load_prefixed(model_dir, "tts.", device, dtype), TalkerConfig.from_model_dir(model_dir))

    @property
    def dtype(self):
        return self.w["emb_text.weight"].dtype

    def condition(self, reply_ids: torch.Tensor, reply_hidden: torch.Tensor) -> torch.Tensor:
        """[n] Thinker reply token IDs and their [n, 4096] last-layer hidden states -> [n + 2, 768] condition embeddings."""
        w = self.w
        text = F.embedding(reply_ids, w["emb_text.weight"])
        h = F.linear(reply_hidden.to(self.dtype), w["projector_semantic.linear1.weight"], w["projector_semantic.linear1.bias"])
        h = F.linear(F.relu(h), w["projector_semantic.linear2.weight"], w["projector_semantic.linear2.bias"])
        if self.cfg.normalize_projected_hidden:
            h = F.normalize(h, p=2, dim=-1)
        tail = F.embedding(torch.tensor([self.cfg.text_eos_token_id, self.cfg.audio_bos_token_id], device=reply_ids.device), w["emb_text.weight"])
        return torch.cat([text + h, tail], dim=0)

    def code_embed(self, codes: torch.Tensor) -> torch.Tensor:
        return F.embedding(codes, self.w["emb_code.0.weight"])

    def new_cache(self, batch: int, max_len: int, device=None) -> "KVCache":
        c = self.cfg
        shape = (c.num_layers, batch, c.num_kv_heads, max_len, c.head_dim)
        dev = device or self.w["emb_text.weight"].device
        return KVCache(torch.zeros(shape, dtype=self.dtype, device=dev), torch.zeros(shape, dtype=self.dtype, device=dev),
                       torch.zeros(batch, dtype=torch.int64, device=dev))

    def forward(self, embeds: torch.Tensor, cache: "KVCache", rows: torch.Tensor) -> torch.Tensor:
        """Run ``embeds`` [B, T, 768] for cache rows ``rows`` at their next positions; returns float32 codec logits of the last step [B, V].

        All rows in one call advance by the same T (prefill of equal-length conditions, or one decode frame each).
        """
        c, w = self.cfg, self.w
        b, t, _ = embeds.shape
        start = cache.lengths[rows]
        pos = start[:, None] + torch.arange(t, device=embeds.device)[None, :]
        cos, sin = rope_cos_sin(pos, c.head_dim, c.rope_theta, embeds.dtype)
        total = int((start + t).max())
        kv_pos = torch.arange(total, device=embeds.device)
        # causal mask per row: key position <= query position, and inside the row's own history
        mask = kv_pos[None, None, :] <= pos[:, :, None]
        x = embeds.to(self.dtype)
        for i in range(c.num_layers):
            p = f"model.layers.{i}."
            h = rms_norm(x, w[p + "input_layernorm.weight"], c.rms_eps)
            q = F.linear(h, w[p + "self_attn.q_proj.weight"]).view(b, t, c.num_heads, c.head_dim).transpose(1, 2)
            k = F.linear(h, w[p + "self_attn.k_proj.weight"]).view(b, t, c.num_kv_heads, c.head_dim).transpose(1, 2)
            v = F.linear(h, w[p + "self_attn.v_proj.weight"]).view(b, t, c.num_kv_heads, c.head_dim).transpose(1, 2)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
            for j in range(b):
                r, s = int(rows[j]), int(start[j])
                cache.k[i, r, :, s:s + t] = k[j]
                cache.v[i, r, :, s:s + t] = v[j]
            keys, vals = cache.k[i, rows, :, :total], cache.v[i, rows, :, :total]
            a = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask[:, None], enable_gqa=c.num_kv_heads != c.num_heads)
            x = x + F.linear(a.transpose(1, 2).reshape(b, t, c.hidden_size), w[p + "self_attn.o_proj.weight"])
            h = rms_norm(x, w[p + "post_attention_layernorm.weight"], c.rms_eps)
            x = x + F.linear(F.silu(F.linear(h, w[p + "mlp.gate_proj.weight"])) * F.linear(h, w[p + "mlp.up_proj.weight"]),
                             w[p + "mlp.down_proj.weight"])
        cache.lengths[rows] = start + t
        x = rms_norm(x[:, -1], w["model.norm.weight"], c.rms_eps)
        return F.linear(x, self.head).float()


@dataclass
class KVCache:
    k: torch.Tensor  # [L, B, H, T, D]
    v: torch.Tensor
    lengths: torch.Tensor  # [B]

    def reset(self, rows: torch.Tensor) -> None:
        self.lengths[rows] = 0
