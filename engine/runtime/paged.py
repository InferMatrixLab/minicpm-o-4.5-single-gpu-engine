"""Paged KV cache and a FlashInfer-backed batched decoder forward (Thinker and Talker).

One ``PagedKV`` pool per model holds every live sequence's keys and values in
fixed-size pages (NHD layout, one ``[pages, page_size, H_kv, D]`` tensor pair per
layer).  ``DecoderRunner.forward`` runs a ragged batch: any mix of prefill
segments and single decode tokens, packed as ``[nnz, hidden]`` with
``qo_indptr``.  Per layer: RMSNorm, fused QKV projection, optional per-head q/k
RMSNorm (Qwen3), RoPE on the packed positions, append to the pages, causal
paged attention (FlashInfer), output projection, fused gate/up SwiGLU MLP.

The math is the reference ports' (``engine.models.thinker`` / ``talker``) with
the same weights; ``tests/engine/test_paged_runtime.py`` holds the parity check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DecoderSpec:
    num_layers: int
    hidden_size: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    rope_theta: float
    rms_eps: float
    qk_norm: bool
    prefix: str = "model."  # weight-name prefix of the decoder layers and final norm


class PagedKV:
    def __init__(self, spec: DecoderSpec, num_pages: int, page_size: int = 16, dtype=torch.bfloat16, device="cuda"):
        self.spec, self.page_size, self.num_pages = spec, page_size, num_pages
        shape = (spec.num_layers, num_pages, page_size, spec.num_kv_heads, spec.head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.free: list[int] = list(range(num_pages - 1, -1, -1))
        self.pages: dict[int, list[int]] = {}
        self.lengths: dict[int, int] = {}

    # --- accounting -----------------------------------------------------------------------------
    def pages_needed(self, seq: int, extra: int) -> int:
        have = len(self.pages.get(seq, []))
        need = -(-(self.lengths.get(seq, 0) + extra) // self.page_size)
        return max(0, need - have)

    def can_fit(self, demands: dict[int, int]) -> bool:
        return sum(self.pages_needed(s, n) for s, n in demands.items()) <= len(self.free)

    def reserve(self, seq: int, extra: int) -> None:
        self.pages.setdefault(seq, [])
        self.lengths.setdefault(seq, 0)
        n = self.pages_needed(seq, extra)
        if n > len(self.free):
            raise MemoryError(f"KV pool exhausted: need {n} pages, {len(self.free)} free")
        self.pages[seq] += [self.free.pop() for _ in range(n)]

    def release(self, seq: int) -> None:
        self.free += reversed(self.pages.pop(seq, []))
        self.lengths.pop(seq, None)

    @property
    def used_pages(self) -> int:
        return self.num_pages - len(self.free)

    # --- batch layout ---------------------------------------------------------------------------
    def layout(self, seqs: list[int], new_tokens: list[int], device="cuda") -> "BatchLayout":
        """Reserve pages for ``new_tokens[i]`` more tokens of ``seqs[i]`` and build the FlashInfer indices (lengths after append)."""
        for s, n in zip(seqs, new_tokens):
            self.reserve(s, n)
        starts = [self.lengths[s] for s in seqs]
        after = [a + n for a, n in zip(starts, new_tokens)]
        npages = [-(-L // self.page_size) for L in after]
        indices = [p for s, n in zip(seqs, npages) for p in self.pages[s][:n]]
        kv_indptr = torch.tensor([0] + list(torch.tensor(npages).cumsum(0).tolist()), dtype=torch.int32)
        last = [L - (n - 1) * self.page_size for L, n in zip(after, npages)]
        qo_indptr = torch.tensor([0] + list(torch.tensor(new_tokens).cumsum(0).tolist()), dtype=torch.int32)
        positions = torch.cat([torch.arange(a, b, dtype=torch.int32) for a, b in zip(starts, after)])
        batch_idx = torch.cat([torch.full((n,), i, dtype=torch.int32) for i, n in enumerate(new_tokens)])
        for s, L in zip(seqs, after):
            self.lengths[s] = L
        return BatchLayout(qo_indptr.to(device), kv_indptr.to(device), torch.tensor(indices, dtype=torch.int32, device=device),
                           torch.tensor(last, dtype=torch.int32, device=device), positions.to(device), batch_idx.to(device), list(new_tokens))


@dataclass
class BatchLayout:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    positions: torch.Tensor  # [nnz] absolute position of every packed token
    batch_indices: torch.Tensor  # [nnz] row of every packed token
    new_tokens: list[int] = field(default_factory=list)

    @property
    def last_token_index(self) -> torch.Tensor:
        return self.qo_indptr[1:].long() - 1


def fuse_weights(w: dict[str, torch.Tensor], spec: DecoderSpec) -> list[dict[str, torch.Tensor]]:
    """Per-layer fused QKV and gate/up matrices (the reference keeps them separate)."""
    out = []
    for i in range(spec.num_layers):
        p = f"{spec.prefix}layers.{i}."
        d = {"ln1": w[p + "input_layernorm.weight"], "ln2": w[p + "post_attention_layernorm.weight"],
             "qkv": torch.cat([w[p + "self_attn.q_proj.weight"], w[p + "self_attn.k_proj.weight"], w[p + "self_attn.v_proj.weight"]], 0).contiguous(),
             "o": w[p + "self_attn.o_proj.weight"],
             "gate_up": torch.cat([w[p + "mlp.gate_proj.weight"], w[p + "mlp.up_proj.weight"]], 0).contiguous(), "down": w[p + "mlp.down_proj.weight"]}
        if spec.qk_norm:
            d["q_norm"], d["k_norm"] = w[p + "self_attn.q_norm.weight"], w[p + "self_attn.k_norm.weight"]
        out.append(d)
    return out


class DecoderRunner:
    def __init__(self, weights: dict[str, torch.Tensor], spec: DecoderSpec, kv: PagedKV, workspace_mb: int = 256):
        import flashinfer

        self.spec, self.kv = spec, kv
        self.layers = fuse_weights(weights, spec)
        self.final_norm = weights[f"{spec.prefix}norm.weight"]
        self.workspace = torch.empty(workspace_mb << 20, dtype=torch.uint8, device=kv.k.device)
        self.prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")
        self._fi = flashinfer

    def plan(self, lay: BatchLayout) -> None:
        s = self.spec
        self.prefill.plan(lay.qo_indptr, lay.kv_indptr, lay.kv_indices, lay.kv_last_page_len, s.num_heads, s.num_kv_heads, s.head_dim,
                          self.kv.page_size, causal=True, q_data_type=self.kv.k.dtype, kv_data_type=self.kv.k.dtype)

    def forward(self, x: torch.Tensor, lay: BatchLayout) -> torch.Tensor:
        """``x`` [nnz, hidden] packed inputs -> final-norm hidden states [nnz, hidden]."""
        s, fi = self.spec, self._fi
        self.plan(lay)
        nq, nk = s.num_heads * s.head_dim, s.num_kv_heads * s.head_dim
        for i, L in enumerate(self.layers):
            h = fi.rmsnorm(x, L["ln1"], s.rms_eps)
            qkv = F.linear(h, L["qkv"])
            q = qkv[:, :nq].reshape(-1, s.num_heads, s.head_dim)
            k = qkv[:, nq:nq + nk].reshape(-1, s.num_kv_heads, s.head_dim)
            v = qkv[:, nq + nk:].reshape(-1, s.num_kv_heads, s.head_dim)
            if s.qk_norm:
                q = fi.rmsnorm(q.reshape(-1, s.head_dim), L["q_norm"], s.rms_eps).view_as(q)
                k = fi.rmsnorm(k.reshape(-1, s.head_dim), L["k_norm"], s.rms_eps).view_as(k)
            else:
                q, k = q.contiguous(), k.contiguous()
            fi.apply_rope_pos_ids_inplace(q, k, lay.positions, rope_theta=s.rope_theta)
            fi.append_paged_kv_cache(k, v.contiguous(), lay.batch_indices, lay.positions, (self.kv.k[i], self.kv.v[i]), lay.kv_indices,
                                     lay.kv_indptr, lay.kv_last_page_len)
            a = self.prefill.run(q, (self.kv.k[i], self.kv.v[i]))
            x = x + F.linear(a.reshape(-1, nq), L["o"])
            h = fi.rmsnorm(x, L["ln2"], s.rms_eps)
            gu = F.linear(h, L["gate_up"])
            g, u = gu.chunk(2, dim=-1)
            x = x + F.linear(F.silu(g) * u, L["down"])
        return fi.rmsnorm(x, self.final_norm, s.rms_eps)
