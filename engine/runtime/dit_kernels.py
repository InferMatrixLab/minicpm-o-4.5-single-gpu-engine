"""Triton kernels of the fast DiT block (``engine.runtime.dit_fast``): FP16 activations, fp32 math, one rounding to FP16
where the autocast reference rounds (norm outputs feed FP16 GEMMs; normalized keys are stored as FP16 caches)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ln_mod_kernel(x_ptr, sh_ptr, sc_ptr, y_ptr, C, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < C
    x = tl.load(x_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    d = tl.where(m, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / C
    xn = d * tl.math.rsqrt(var + eps)
    sc = tl.load(sc_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sh = tl.load(sh_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(y_ptr + row * C + cols, (xn * sc + sh).to(y_ptr.dtype.element_ty), mask=m)


def ln_mod(x: torch.Tensor, shift: torch.Tensor, scale1: torch.Tensor, eps: float, out: torch.Tensor | None = None) -> torch.Tensor:
    """``modulate(LayerNorm(x), shift, scale)`` with ``scale1 = 1 + scale``; x [N, C] contiguous."""
    y = torch.empty_like(x) if out is None else out
    _ln_mod_kernel[(x.shape[0],)](x, shift, scale1, y, x.shape[1], eps, BLOCK=triton.next_power_of_2(x.shape[1]))
    return y


@triton.jit
def _add_ln_mod_kernel(h_ptr, g_ptr, y_ptr, sh_ptr, sc_ptr, hout_ptr, out_ptr, C, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < C
    h = tl.load(h_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
    y = tl.load(y_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    hn = tl.fma(g, y, h).to(hout_ptr.dtype.element_ty)  # torch.addcmul(h, g, y): one fp32 fma, rounded to the activation dtype
    tl.store(hout_ptr + row * C + cols, hn, mask=m)
    x = hn.to(tl.float32)  # then _ln_mod_kernel on the rounded sum
    mean = tl.sum(x, axis=0) / C
    d = tl.where(m, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / C
    xn = d * tl.math.rsqrt(var + eps)
    sc = tl.load(sc_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sh = tl.load(sh_ptr + cols, mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + row * C + cols, (xn * sc + sh).to(out_ptr.dtype.element_ty), mask=m)


def add_ln_mod(h: torch.Tensor, g: torch.Tensor, y: torch.Tensor, shift: torch.Tensor, scale1: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``h = torch.addcmul(h, g, y)`` (gated residual, ``g`` [C]) and ``ln_mod(h, shift, scale1, eps)`` in one kernel;
    returns the new residual and the modulated norm.  h, y [N, C] contiguous."""
    assert h.is_contiguous() and y.is_contiguous() and h.shape == y.shape
    hout, out = torch.empty_like(h), torch.empty_like(h)
    _add_ln_mod_kernel[(h.shape[0],)](h, g, y, shift, scale1, hout, out, h.shape[1], eps, BLOCK=triton.next_power_of_2(h.shape[1]))
    return hout, out


@triton.jit
def _qk_norm_scatter_kernel(qkv_ptr, qw_ptr, qb_ptr, kw_ptr, kb_ptr, q_ptr, kc_ptr, vc_ptr, slot_ptr, H, eps, NH: tl.constexpr,
                            D: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, D)
    base = qkv_ptr + row * 3 * H + head * D
    slot = tl.load(slot_ptr + row).to(tl.int64)
    for which in tl.static_range(2):  # 0: q -> q buffer, 1: k -> cache slot
        x = tl.load(base + which * H + cols).to(tl.float32)
        mean = tl.sum(x, axis=0) / D
        dlt = x - mean
        var = tl.sum(dlt * dlt, axis=0) / D
        xn = dlt * tl.math.rsqrt(var + eps)
        if which == 0:
            y = xn * tl.load(qw_ptr + cols).to(tl.float32) + tl.load(qb_ptr + cols).to(tl.float32)
            tl.store(q_ptr + (row * NH + head) * D + cols, y.to(q_ptr.dtype.element_ty))
        else:
            y = xn * tl.load(kw_ptr + cols).to(tl.float32) + tl.load(kb_ptr + cols).to(tl.float32)
            tl.store(kc_ptr + (slot * NH + head) * D + cols, y.to(kc_ptr.dtype.element_ty))
    tl.store(vc_ptr + (slot * NH + head) * D + cols, tl.load(base + 2 * H + cols))


def qk_norm_scatter(qkv: torch.Tensor, qn, kn, q_out: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor,
                    heads: int, head_dim: int) -> None:
    """Per-head LayerNorm (affine) of q into ``q_out`` [N, heads, D] and of k into its cache slot; v copied to its slot.
    ``qkv`` [N, 3 H]; caches [n_slots, 1, heads, D]; ``slots`` [N] int64.  Both norms share one eps (the reference's)."""
    N, H = qkv.shape[0], heads * head_dim
    assert qn[2] == kn[2]
    _qk_norm_scatter_kernel[(N, heads)](qkv, qn[0], qn[1], kn[0], kn[1], q_out, k_cache, v_cache, slots, H, qn[2], NH=heads, D=head_dim)


@triton.jit
def _ln_mish_kernel(x_ptr, w_ptr, b_ptr, y_ptr, C, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < C
    x = tl.load(x_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(x, axis=0) / C
    d = tl.where(m, x - mean, 0.0)
    var = tl.sum(d * d, axis=0) / C
    y = d * tl.math.rsqrt(var + eps) * tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32) + tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
    sp = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(y)))  # softplus (torch's threshold 20)
    e = tl.exp(-2.0 * sp)
    tl.store(y_ptr + row * C + cols, (y * (1.0 - e) / (1.0 + e)).to(y_ptr.dtype.element_ty), mask=m)  # y * tanh(softplus(y))


def ln_mish(x: torch.Tensor, ln, out: torch.Tensor | None = None) -> torch.Tensor:
    """``Mish(LayerNorm(x))`` (affine LayerNorm ``ln = (w, b, eps)``); x [N, C] contiguous."""
    y = torch.empty_like(x) if out is None else out
    _ln_mish_kernel[(x.shape[0],)](x, ln[0], ln[1], y, x.shape[1], ln[2], BLOCK=triton.next_power_of_2(x.shape[1]))
    return y


@triton.jit
def _conv_gather_kernel(x_ptr, cache_ptr, out_ptr, tok_row_ptr, tok_pos_ptr, row_start_ptr, cache_stride, C, BLOCK: tl.constexpr):
    """out[n] = [f(t - 2), f(t - 1), f(t)] for token n at position t of row r, where f(p) is x[row_start[r] + p] for p >= 0
    and cache[r, 2 + p] (the two cached frames) for p < 0."""
    n = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < C
    r = tl.load(tok_row_ptr + n).to(tl.int64)
    t = tl.load(tok_pos_ptr + n)
    start = tl.load(row_start_ptr + r).to(tl.int64)
    for j in tl.static_range(3):
        p = t + j - 2
        if p >= 0:
            v = tl.load(x_ptr + (start + p) * C + cols, mask=m, other=0.0)
        else:
            v = tl.load(cache_ptr + r * cache_stride + (2 + p) * C + cols, mask=m, other=0.0)
        tl.store(out_ptr + n * 3 * C + j * C + cols, v, mask=m)


@triton.jit
def _conv_tail_kernel(x_ptr, cache_ptr, new_ptr, row_start_ptr, row_len_ptr, cache_stride, new_stride, C, BLOCK: tl.constexpr):
    """new[r] = the last two frames of [cache[r, 0], cache[r, 1], x[row r]] (the real end of each row)."""
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < C
    start = tl.load(row_start_ptr + r).to(tl.int64)
    T = tl.load(row_len_ptr + r)
    for j in tl.static_range(2):
        p = T - 2 + j  # position in x (negative: a cached frame)
        if p >= 0:
            v = tl.load(x_ptr + (start + p) * C + cols, mask=m, other=0.0)
        else:
            v = tl.load(cache_ptr + r * cache_stride + (2 + p) * C + cols, mask=m, other=0.0)
        tl.store(new_ptr + r * new_stride + j * C + cols, v, mask=m)


def conv_gather(x: torch.Tensor, cache: torch.Tensor, new_cache: torch.Tensor, meta: dict, out: torch.Tensor) -> torch.Tensor:
    """im2col of a kernel-3 causal conv over packed ragged rows with two cached frames per row, and the rows' new caches.
    ``cache`` / ``new_cache`` [R, 2, C] (row stride may differ from 2 C); ``meta``: tok_row, tok_pos [N], row_start,
    row_len [R] (int32); ``out`` [N, 3 C]."""
    N, C = x.shape
    R = cache.shape[0]
    assert cache.stride(1) == C and cache.stride(2) == 1 and new_cache.stride(1) == C and new_cache.stride(2) == 1
    _conv_gather_kernel[(N,)](x, cache, out, meta["tok_row"], meta["tok_pos"], meta["row_start"], cache.stride(0), C,
                              BLOCK=triton.next_power_of_2(C))
    _conv_tail_kernel[(R,)](x, cache, new_cache, meta["row_start"], meta["row_len"], cache.stride(0), new_cache.stride(0), C,
                            BLOCK=triton.next_power_of_2(C))
    return out
