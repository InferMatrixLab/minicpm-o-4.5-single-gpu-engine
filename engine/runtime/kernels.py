"""Fused kernels with the reference's rounding (Triton).

``rms_norm_hf(x, w, eps)`` computes, in one kernel per call, what ``engine.runtime.paged.hf_rms_norm`` computes with
eight: ``w * (x.float() * rsqrt(mean(x.float() ** 2) + eps)).to(x.dtype)``, i.e. normalise in fp32, round to the
activation dtype, then the product with the weight rounded to the activation dtype.  The fp32 mean may sum in a
different order than torch's reduction, so a result can differ by one ulp of the activation dtype in rare rows
(``tests/engine/test_kernels.py`` measures it).  The decode kernels below (residual add + norm, q/k norm + RoPE + v
copy, SiLU-and-multiply) round every intermediate the reference rounds, with integer round-to-nearest-even
(``_round_bf16``) so the compiler cannot elide a rounding and contract a product into an FMA.  bf16 activations only.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _round_bf16(x):
    """fp32 -> nearest-even bf16 value, kept in fp32, by integer arithmetic: a plain ``.to(bf16).to(fp32)`` pair can be
    elided by the compiler, which then contracts the product into the following add as an FMA (not the reference)."""
    b = x.to(tl.uint32, bitcast=True)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _rms_norm_hf_kernel(x_ptr, w_ptr, y_ptr, n_cols, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    xn = _round_bf16(x * tl.math.rsqrt(var + eps))  # round to the activation dtype, as the reference
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (xn * w.to(tl.float32)).to(y_ptr.dtype.element_ty)  # the dtype product, rounded once
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


def rms_norm_hf(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    assert x.dtype == torch.bfloat16, "bf16 activations only (the roundings are bf16)"
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    n = x2.shape[1]
    _rms_norm_hf_kernel[(x2.shape[0],)](x2, w, y, n, x2.stride(0), y.stride(0), eps, BLOCK=triton.next_power_of_2(n))
    return y.view(shape)


@triton.jit
def _add_rms_norm_hf_kernel(x_ptr, r_ptr, w_ptr, s_ptr, y_ptr, n_cols, stride_x, stride_r, stride_s, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    s = (x + r).to(s_ptr.dtype.element_ty)  # the residual sum in the activation dtype, as ``x + r``
    tl.store(s_ptr + row * stride_s + cols, s, mask=mask)
    sf = s.to(tl.float32)
    var = tl.sum(sf * sf, axis=0) / n_cols
    xn = _round_bf16(sf * tl.math.rsqrt(var + eps))
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    tl.store(y_ptr + row * stride_y + cols, (xn * w.to(tl.float32)).to(y_ptr.dtype.element_ty), mask=mask)


def add_rms_norm_hf(x: torch.Tensor, r: torch.Tensor, w: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    """``s = x + r`` and ``hf_rms_norm(s, w, eps)`` in one kernel (2-D inputs); returns (s, normed)."""
    assert x.dtype == torch.bfloat16, "bf16 activations only (the roundings are bf16)"
    s, y = torch.empty_like(x), torch.empty_like(x)
    n = x.shape[1]
    _add_rms_norm_hf_kernel[(x.shape[0],)](x, r, w, s, y, n, x.stride(0), r.stride(0), s.stride(0), y.stride(0), eps,
                                           BLOCK=triton.next_power_of_2(n))
    return s, y


@triton.jit
def _qk_norm_rope_hf_kernel(qkv_ptr, qw_ptr, kw_ptr, cos_ptr, sin_ptr, q_ptr, k_ptr, v_ptr, stride_qkv, stride_cs, n_q, n_kv, eps,
                            D: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    cols = tl.arange(0, D)
    half = D // 2
    partner = (cols + half) % D
    base = qkv_ptr + row * stride_qkv + head * D
    if head >= n_q + n_kv:  # a v head: copied out contiguous
        vh = head - n_q - n_kv
        tl.store(v_ptr + (row * n_kv + vh) * D + cols, tl.load(base + cols))
    else:
        is_q = head < n_q
        x = tl.load(base + cols).to(tl.float32)
        xp = tl.load(base + partner).to(tl.float32)
        var = tl.sum(x * x, axis=0) / D
        inv = tl.math.rsqrt(var + eps)
        if is_q:
            w = tl.load(qw_ptr + cols).to(tl.float32)
            wp = tl.load(qw_ptr + partner).to(tl.float32)
        else:
            w = tl.load(kw_ptr + cols).to(tl.float32)
            wp = tl.load(kw_ptr + partner).to(tl.float32)
        dt = q_ptr.dtype.element_ty
        y = _round_bf16(_round_bf16(x * inv) * w)  # hf_rms_norm's two roundings
        yp = _round_bf16(_round_bf16(xp * inv) * wp)
        rot = tl.where(cols < half, -yp, yp)  # rotate-half
        c = tl.load(cos_ptr + row * stride_cs + cols).to(tl.float32)
        sn = tl.load(sin_ptr + row * stride_cs + cols).to(tl.float32)
        out = (_round_bf16(y * c) + _round_bf16(rot * sn)).to(dt)  # q * cos + rot(q) * sin, each op rounded (bf16 only)
        if is_q:
            tl.store(q_ptr + (row * n_q + head) * D + cols, out)
        else:
            tl.store(k_ptr + (row * n_kv + head - n_q) * D + cols, out)


def qk_norm_rope_hf(qkv: torch.Tensor, q_w: torch.Tensor, k_w: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, n_q: int, n_kv: int,
                    head_dim: int, eps: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-head ``hf_rms_norm`` of q and k, then ``hf_rope`` with the given (cos, sin) tables [n, 1, D]; v copied out.
    ``qkv`` is [n, (n_q + 2 n_kv) * D] row-major.  Returns contiguous q [n, n_q, D], k and v [n, n_kv, D]."""
    assert qkv.dtype == torch.bfloat16, "bf16 activations only (the roundings are bf16)"
    n = qkv.shape[0]
    q = torch.empty(n, n_q, head_dim, dtype=qkv.dtype, device=qkv.device)
    k = torch.empty(n, n_kv, head_dim, dtype=qkv.dtype, device=qkv.device)
    v = torch.empty_like(k)
    c2, s2 = cos.reshape(n, head_dim), sin.reshape(n, head_dim)
    _qk_norm_rope_hf_kernel[(n, n_q + 2 * n_kv)](qkv, q_w, k_w, c2, s2, q, k, v, qkv.stride(0), c2.stride(0), n_q, n_kv, eps, D=head_dim)
    return q, k, v


@triton.jit
def _silu_mul_hf_kernel(gu_ptr, y_ptr, n_cols, stride_gu, stride_y, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < n_cols
    g = tl.load(gu_ptr + row * stride_gu + cols, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(gu_ptr + row * stride_gu + n_cols + cols, mask=mask, other=0.0).to(tl.float32)
    dt = y_ptr.dtype.element_ty
    s = _round_bf16(tl.math.div_rn(g, 1.0 + libdevice.exp(-g)))  # F.silu in fp32, rounded to the activation dtype
    tl.store(y_ptr + row * stride_y + cols, (s * u).to(dt), mask=mask)


def silu_mul_hf(gu: torch.Tensor) -> torch.Tensor:
    """``F.silu(g) * u`` for ``g, u = gu.chunk(2, -1)`` with the reference's roundings, in one kernel (2-D input)."""
    assert gu.dtype == torch.bfloat16, "bf16 activations only (the roundings are bf16)"
    n, two = gu.shape
    y = torch.empty(n, two // 2, dtype=gu.dtype, device=gu.device)
    block = 1024
    _silu_mul_hf_kernel[(n, triton.cdiv(two // 2, block))](gu, y, two // 2, gu.stride(0), y.stride(0), BLOCK=block)
    return y
