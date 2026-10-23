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

from engine.runtime.fp8 import linear
from engine.runtime.hostcopy import plan as fi_plan

_CAPTURE_STREAMS: dict = {}


def capture_stream(lane: str) -> torch.cuda.Stream:
    """The stream the CUDA graphs of one lane (``thinker``, ``talker``) are captured on.

    cuBLAS keeps a workspace per (handle, stream), and a captured GEMM keeps using the workspace of its capture stream.
    Graphs captured on torch's shared default capture stream therefore share one workspace: replayed concurrently on two
    streams (``tick_overlap``), split-K GEMMs of both write their partial sums and semaphores into the same memory and
    the GPU can hang.  Graphs of lanes that may replay concurrently are captured on different streams."""
    if lane not in _CAPTURE_STREAMS:
        _CAPTURE_STREAMS[lane] = torch.cuda.Stream()
    return _CAPTURE_STREAMS[lane]


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

    def truncate(self, seq: int, length: int) -> None:
        """Drop the tail of ``seq`` beyond ``length`` tokens (its pages stay reserved and are overwritten next)."""
        if length > self.lengths.get(seq, 0):
            raise ValueError(f"cannot extend seq {seq} by truncation")
        self.lengths[seq] = length

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
        last = torch.tensor(last, dtype=torch.int32)
        dev = lambda t: t.to(device, non_blocking=True)  # noqa: E731  (queued: no wait for the GPU; engine.runtime.hostcopy)
        return BatchLayout(dev(qo_indptr), dev(kv_indptr), dev(torch.tensor(indices, dtype=torch.int32)), dev(last), dev(positions), dev(batch_idx),
                           list(new_tokens), qo_indptr, kv_indptr, last)


@dataclass
class BatchLayout:
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_last_page_len: torch.Tensor
    positions: torch.Tensor  # [nnz] absolute position of every packed token
    batch_indices: torch.Tensor  # [nnz] row of every packed token
    new_tokens: list[int] = field(default_factory=list)
    # host copies of what FlashInfer's plan reads on the host (given device tensors it copies them back, synchronizing)
    qo_indptr_h: torch.Tensor | None = None
    kv_indptr_h: torch.Tensor | None = None
    kv_last_page_len_h: torch.Tensor | None = None

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


def hf_rms_norm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """The reference rounding: normalise in fp32, cast to the activation dtype, then scale by the weight in that dtype."""
    xf = x.float()
    return w * (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)).to(x.dtype)


def hf_rope_tables(positions: torch.Tensor, head_dim: int, theta: float, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """cos/sin [n, 1, head_dim] computed in fp32 and cast to the activation dtype (the reference rounding); the same
    for every layer of one forward, so callers compute them once."""
    d = head_dim
    inv = 1.0 / (theta ** (torch.arange(0, d, 2, device=positions.device, dtype=torch.int64).float() / d))
    freqs = positions.float()[:, None] * inv[None]
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos().to(dtype)[:, None], emb.sin().to(dtype)[:, None]


def hf_rope(q: torch.Tensor, k: torch.Tensor, positions: torch.Tensor, theta: float, tables=None) -> tuple[torch.Tensor, torch.Tensor]:
    """Rotate-half RoPE with cos/sin computed in fp32 and cast to the activation dtype (the reference rounding)."""
    d = q.shape[-1]
    cos, sin = tables if tables is not None else hf_rope_tables(positions, d, theta, q.dtype)

    def rot(x):
        return torch.cat([-x[..., d // 2:], x[..., : d // 2]], dim=-1)

    return q * cos + rot(q) * sin, k * cos + rot(k) * sin


class DecoderRunner:
    def __init__(self, weights: dict[str, torch.Tensor], spec: DecoderSpec, kv: PagedKV, workspace_mb: int = 256, numerics: str = "hf",
                 layers: list[dict] | None = None, hf_kernels: bool = False):
        """``numerics="hf"`` rounds norms and RoPE as the reference does; ``"fused"`` uses FlashInfer's fused kernels.
        ``layers``: prebuilt fused (optionally FP8, ``engine.runtime.fp8``) layer weights shared with other runners.
        ``hf_kernels`` (``prefill_hf_kernels``, ``"hf"`` numerics, bf16): the reference-rounding Triton kernels of
        ``engine.runtime.kernels`` that the fused Thinker decode runs (residual add + norm, q/k norm + RoPE + v copy,
        SiLU-and-multiply) in place of the torch elementwise ops; the same roundings, a norm's fp32 sum order aside."""
        import flashinfer

        self.numerics = numerics
        self.hf_kernels = hf_kernels and numerics == "hf" and kv.k.dtype == torch.bfloat16
        self.spec, self.kv = spec, kv
        self.layers = layers if layers is not None else fuse_weights(weights, spec)
        self.final_norm = weights[f"{spec.prefix}norm.weight"]
        self.workspace = torch.empty(workspace_mb << 20, dtype=torch.uint8, device=kv.k.device)
        self.prefill = flashinfer.BatchPrefillWithPagedKVCacheWrapper(self.workspace, "NHD")
        self._fi = flashinfer

    def plan(self, lay: BatchLayout) -> None:
        s = self.spec
        fi_plan(self.prefill, lay.qo_indptr_h, lay.kv_indptr_h, lay.kv_indices, lay.kv_last_page_len_h, s.num_heads, s.num_kv_heads, s.head_dim,
                          self.kv.page_size, causal=True, q_data_type=self.kv.k.dtype, kv_data_type=self.kv.k.dtype)

    def forward(self, x: torch.Tensor, lay: BatchLayout) -> torch.Tensor:
        """``x`` [nnz, hidden] packed inputs -> final-norm hidden states [nnz, hidden]."""
        self.plan(lay)
        return self.body(x, lay.positions, lay.batch_indices, lay.kv_indices, lay.kv_indptr, lay.kv_last_page_len, self.prefill)

    def body(self, x, positions, batch_indices, kv_indices, kv_indptr, kv_last_page_len, wrapper) -> torch.Tensor:
        """The layer stack on a planned ``wrapper`` (eager, or captured by ``engine.runtime.prefill_graph``)."""
        if self.hf_kernels:
            return self._body_hf_kernels(x, positions, batch_indices, kv_indices, kv_indptr, kv_last_page_len, wrapper)
        s, fi = self.spec, self._fi
        nq, nk = s.num_heads * s.head_dim, s.num_kv_heads * s.head_dim
        hf = self.numerics == "hf"
        norm = (lambda t, w: hf_rms_norm(t, w, s.rms_eps)) if hf else (lambda t, w: fi.rmsnorm(t, w, s.rms_eps))
        tables = hf_rope_tables(positions, s.head_dim, s.rope_theta, x.dtype) if hf else None  # once for all layers
        for i, L in enumerate(self.layers):
            h = norm(x, L["ln1"])
            qkv = linear(h, L["qkv"])
            q = qkv[:, :nq].reshape(-1, s.num_heads, s.head_dim)
            k = qkv[:, nq:nq + nk].reshape(-1, s.num_kv_heads, s.head_dim)
            v = qkv[:, nq + nk:].reshape(-1, s.num_kv_heads, s.head_dim)
            if s.qk_norm:
                q = norm(q.reshape(-1, s.head_dim), L["q_norm"]).view_as(q)
                k = norm(k.reshape(-1, s.head_dim), L["k_norm"]).view_as(k)
            q, k = q.contiguous(), k.contiguous()
            if hf:
                q, k = hf_rope(q, k, positions, s.rope_theta, tables)
                q, k = q.contiguous(), k.contiguous()
            else:
                fi.apply_rope_pos_ids_inplace(q, k, positions, rope_theta=s.rope_theta)
            fi.append_paged_kv_cache(k, v.contiguous(), batch_indices, positions, (self.kv.k[i], self.kv.v[i]), kv_indices, kv_indptr,
                                     kv_last_page_len)
            a = wrapper.run(q, (self.kv.k[i], self.kv.v[i]))
            x = x + linear(a.reshape(-1, nq), L["o"])
            h = norm(x, L["ln2"])
            gu = linear(h, L["gate_up"])
            g, u = gu.chunk(2, dim=-1)
            x = x + linear(F.silu(g) * u, L["down"])
        return norm(x, self.final_norm)

    def _body_hf_kernels(self, x, positions, batch_indices, kv_indices, kv_indptr, kv_last_page_len, wrapper) -> torch.Tensor:
        """``body`` with each residual add fused into the next norm, q/k norm + RoPE + v copy in one kernel (RoPE + v copy
        for models without q/k norms) and SiLU-and-multiply in one kernel, as ``ThinkerDecoder._step_fused``."""
        from engine.runtime.kernels import add_rms_norm_hf, qk_norm_rope_hf, rms_norm_hf, rope_hf, silu_mul_hf

        s, fi = self.spec, self._fi
        nq = s.num_heads * s.head_dim
        cos, sin = hf_rope_tables(positions, s.head_dim, s.rope_theta, x.dtype)
        h = rms_norm_hf(x, self.layers[0]["ln1"], s.rms_eps)
        last = len(self.layers) - 1
        for i, L in enumerate(self.layers):
            qkv = linear(h, L["qkv"])
            if s.qk_norm:
                q, k, v = qk_norm_rope_hf(qkv, L["q_norm"], L["k_norm"], cos, sin, s.num_heads, s.num_kv_heads, s.head_dim, s.rms_eps)
            else:
                q, k, v = rope_hf(qkv, cos, sin, s.num_heads, s.num_kv_heads, s.head_dim)
            fi.append_paged_kv_cache(k, v, batch_indices, positions, (self.kv.k[i], self.kv.v[i]), kv_indices, kv_indptr, kv_last_page_len)
            a = wrapper.run(q, (self.kv.k[i], self.kv.v[i]))
            x, h = add_rms_norm_hf(x, linear(a.reshape(-1, nq), L["o"]), L["ln2"], s.rms_eps)
            d = linear(silu_mul_hf(linear(h, L["gate_up"])), L["down"])
            x, h = add_rms_norm_hf(x, d, self.final_norm if i == last else self.layers[i + 1]["ln1"], s.rms_eps)
        return h
