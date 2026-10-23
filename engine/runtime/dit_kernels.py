"""Triton kernels of the fast DiT block (``engine.runtime.dit_fast``): FP16 activations, fp32 math, one rounding to FP16
where the autocast reference rounds (norm outputs feed FP16 GEMMs; normalized keys are stored as FP16 caches)."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


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
def _qk_norm_scatter_ref_kernel(qkv_ptr, qw_ptr, qb_ptr, kw_ptr, kb_ptr, q_ptr, kc_ptr, vc_ptr, slot_ptr, H, eps, NH: tl.constexpr,
                                D: tl.constexpr):
    """One program per (token, head): the original kernel, whose reduction order ``_qk_norm_scatter_kernel`` reproduces."""
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


@triton.jit
def _halve(x, W: tl.constexpr):
    """[NH, 2, W] -> [NH, 2, W / 2]: x[.., l] + x[.., l + W / 2], one butterfly level of a warp reduction."""
    a, b = tl.split(tl.permute(tl.reshape(x, (x.shape[0], 2, 2, W // 2)), (0, 1, 3, 2)))
    return a + b


@triton.jit
def _head_sum(a, NH: tl.constexpr):
    """[NH, 2, 16] (a head's 64 values after the first butterfly level, per warp) -> [NH]: the other levels, then
    warp 0 + warp 1."""
    a = _halve(_halve(_halve(_halve(a, 16), 8), 4), 2)
    w0, w1 = tl.split(tl.reshape(a, (NH, 2)))
    return w0 + w1


@triton.jit
def _qk_norm_scatter_kernel(qkv_ptr, qw_ptr, qb_ptr, kw_ptr, kb_ptr, q_ptr, kc_ptr, vc_ptr, slot_ptr, H, eps, NH: tl.constexpr,
                            D: tl.constexpr):
    """One program per token, all heads at once ([NH, 64]).  Each head's sums are taken in the order the per-(token, head)
    kernel takes them (``_qk_norm_scatter_ref_kernel``, one value per lane of two warps: xor-butterfly levels 16 .. 1, then
    the two warps; the first level of the squares is an fma of the lane's own square onto its partner's rounded one), so
    the outputs are the same bit for bit at a fraction of the launch (the per-head programs are latency bound)."""
    row = tl.program_id(0)
    hd = tl.arange(0, NH)[:, None] * D + tl.arange(0, D)[None, :]
    cols = tl.arange(0, D)[None, :]
    slot = tl.load(slot_ptr + row).to(tl.int64)
    base = qkv_ptr + row * 3 * H + hd
    for which in tl.static_range(2):  # 0: q -> q buffer, 1: k -> cache slot
        x = tl.load(base + which * H).to(tl.float32)
        lo, hi = tl.split(tl.permute(tl.reshape(x, (NH, 2, 2, 16)), (0, 1, 3, 2)))
        mean = _head_sum(lo + hi, NH)[:, None] / D
        dlt = x - mean
        lo, hi = tl.split(tl.permute(tl.reshape(dlt, (NH, 2, 2, 16)), (0, 1, 3, 2)))
        var = _head_sum(tl.fma(lo, lo, libdevice.mul_rn(hi, hi)), NH)[:, None] / D
        xn = dlt * tl.math.rsqrt(var + eps)
        if which == 0:
            y = xn * tl.load(qw_ptr + cols).to(tl.float32) + tl.load(qb_ptr + cols).to(tl.float32)
            tl.store(q_ptr + row * NH * D + hd, y.to(q_ptr.dtype.element_ty))
        else:
            y = xn * tl.load(kw_ptr + cols).to(tl.float32) + tl.load(kb_ptr + cols).to(tl.float32)
            tl.store(kc_ptr + slot * NH * D + hd, y.to(kc_ptr.dtype.element_ty))
    tl.store(vc_ptr + slot * NH * D + hd, tl.load(base + 2 * H))


def qk_norm_scatter(qkv: torch.Tensor, qn, kn, q_out: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slots: torch.Tensor,
                    heads: int, head_dim: int, per_head: bool = False) -> None:
    """Per-head LayerNorm (affine) of q into ``q_out`` [N, heads, D] and of k into its cache slot; v copied to its slot.
    ``qkv`` [N, 3 H]; caches [n_slots, 1, heads, D]; ``slots`` [N] int64.  Both norms share one eps (the reference's).
    ``per_head``: the original one-program-per-head kernel (also the fallback for head sizes other than 64)."""
    N, H = qkv.shape[0], heads * head_dim
    assert qn[2] == kn[2]
    if per_head or head_dim != 64 or heads & (heads - 1):
        _qk_norm_scatter_ref_kernel[(N, heads)](qkv, qn[0], qn[1], kn[0], kn[1], q_out, k_cache, v_cache, slots, H, qn[2], NH=heads, D=head_dim)
    else:
        _qk_norm_scatter_kernel[(N,)](qkv, qn[0], qn[1], kn[0], kn[1], q_out, k_cache, v_cache, slots, H, qn[2], NH=heads, D=head_dim)


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


@triton.jit
def _conv_place(v, im_ptr, cout_ptr, r, t, T, n, cout_stride, C, cols, m):
    """Token n (position t of row r, T tokens) holds frame f(t) = v of a kernel-3 causal conv's input: it is im2col
    column 2 of token n, column 1 of token n + 1 and column 0 of token n + 2 (same row), and the row's new cache frame
    t - T + 2 when it is one of the row's last two."""
    tl.store(im_ptr + n * 3 * C + 2 * C + cols, v, mask=m)
    if t + 1 < T:
        tl.store(im_ptr + (n + 1) * 3 * C + C + cols, v, mask=m)
    if t + 2 < T:
        tl.store(im_ptr + (n + 2) * 3 * C + cols, v, mask=m)
    if t + 2 >= T:
        tl.store(cout_ptr + r * cout_stride + (t - T + 2) * C + cols, v, mask=m)


@triton.jit
def _conv_row_cache(cache_ptr, im_ptr, cout_ptr, row_start_ptr, row_len_ptr, r, cache_stride, cout_stride, C, cols, m):
    """Row r's two cached frames: im2col columns 0, 1 of its first token and column 0 of its second, and its new cache
    when the row has fewer than two tokens (``_conv_gather_kernel`` / ``_conv_tail_kernel`` for p < 0)."""
    start = tl.load(row_start_ptr + r).to(tl.int64)
    T = tl.load(row_len_ptr + r)
    c0 = tl.load(cache_ptr + r * cache_stride + cols, mask=m, other=0.0)
    c1 = tl.load(cache_ptr + r * cache_stride + C + cols, mask=m, other=0.0)
    if T >= 1:
        tl.store(im_ptr + start * 3 * C + cols, c0, mask=m)
        tl.store(im_ptr + start * 3 * C + C + cols, c1, mask=m)
    if T >= 2:
        tl.store(im_ptr + (start + 1) * 3 * C + cols, c1, mask=m)
    if T == 1:
        tl.store(cout_ptr + r * cout_stride + cols, c1, mask=m)
    if T == 0:
        tl.store(cout_ptr + r * cout_stride + cols, c0, mask=m)
        tl.store(cout_ptr + r * cout_stride + C + cols, c1, mask=m)


@triton.jit
def _add_ln_mod_conv_kernel(h_ptr, g_ptr, y_ptr, sh_ptr, sc_ptr, hout_ptr, im_ptr, cache_ptr, cout_ptr, tok_row_ptr, tok_pos_ptr,
                            row_start_ptr, row_len_ptr, cache_stride, cout_stride, N, C, eps, BLOCK: tl.constexpr):
    """``_add_ln_mod_kernel`` whose normed output goes straight into the conv's im2col buffer and new cache (programs
    0..N-1: tokens; N..N+R-1: the rows' cached frames)."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < C
    if pid < N:
        row = pid
        h = tl.load(h_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + cols, mask=m, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
        hn = tl.fma(g, y, h).to(hout_ptr.dtype.element_ty)
        tl.store(hout_ptr + row * C + cols, hn, mask=m)
        x = hn.to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        d = tl.where(m, x - mean, 0.0)
        var = tl.sum(d * d, axis=0) / C
        xn = d * tl.math.rsqrt(var + eps)
        sc = tl.load(sc_ptr + cols, mask=m, other=0.0).to(tl.float32)
        sh = tl.load(sh_ptr + cols, mask=m, other=0.0).to(tl.float32)
        v = (xn * sc + sh).to(im_ptr.dtype.element_ty)
        r = tl.load(tok_row_ptr + row).to(tl.int64)
        _conv_place(v, im_ptr, cout_ptr, r, tl.load(tok_pos_ptr + row), tl.load(row_len_ptr + r), row.to(tl.int64), cout_stride, C, cols, m)
    else:
        _conv_row_cache(cache_ptr, im_ptr, cout_ptr, row_start_ptr, row_len_ptr, pid - N, cache_stride, cout_stride, C, cols, m)


def add_ln_mod_conv(h: torch.Tensor, g: torch.Tensor, y: torch.Tensor, shift: torch.Tensor, scale1: torch.Tensor, eps: float,
                    cache: torch.Tensor, new_cache: torch.Tensor, meta: dict, out: torch.Tensor) -> torch.Tensor:
    """``add_ln_mod`` followed by ``conv_gather`` of its normed output in one kernel: returns the new residual and fills
    ``out`` [N, 3 C] (the im2col) and ``new_cache`` exactly as ``conv_gather`` does."""
    N, C = h.shape
    R = cache.shape[0]
    assert h.is_contiguous() and y.is_contiguous() and h.shape == y.shape and out.is_contiguous() and out.shape == (N, 3 * C)
    assert cache.stride(1) == C and cache.stride(2) == 1 and new_cache.stride(1) == C and new_cache.stride(2) == 1
    hout = torch.empty_like(h)
    _add_ln_mod_conv_kernel[(N + R,)](h, g, y, shift, scale1, hout, out, cache, new_cache, meta["tok_row"], meta["tok_pos"], meta["row_start"],
                                      meta["row_len"], cache.stride(0), new_cache.stride(0), N, C, eps, BLOCK=triton.next_power_of_2(C))
    return hout


@triton.jit
def _ln_mish_conv_kernel(x_ptr, w_ptr, b_ptr, im_ptr, cache_ptr, cout_ptr, tok_row_ptr, tok_pos_ptr, row_start_ptr, row_len_ptr,
                         cache_stride, cout_stride, N, C, eps, BLOCK: tl.constexpr):
    """``_ln_mish_kernel`` whose output goes straight into the conv's im2col buffer and new cache (as
    ``_add_ln_mod_conv_kernel``)."""
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    m = cols < C
    if pid < N:
        row = pid
        x = tl.load(x_ptr + row * C + cols, mask=m, other=0.0).to(tl.float32)
        mean = tl.sum(x, axis=0) / C
        d = tl.where(m, x - mean, 0.0)
        var = tl.sum(d * d, axis=0) / C
        y = d * tl.math.rsqrt(var + eps) * tl.load(w_ptr + cols, mask=m, other=0.0).to(tl.float32) + tl.load(b_ptr + cols, mask=m, other=0.0).to(tl.float32)
        sp = tl.where(y > 20.0, y, tl.log(1.0 + tl.exp(y)))
        e = tl.exp(-2.0 * sp)
        v = (y * (1.0 - e) / (1.0 + e)).to(im_ptr.dtype.element_ty)
        r = tl.load(tok_row_ptr + row).to(tl.int64)
        _conv_place(v, im_ptr, cout_ptr, r, tl.load(tok_pos_ptr + row), tl.load(row_len_ptr + r), row.to(tl.int64), cout_stride, C, cols, m)
    else:
        _conv_row_cache(cache_ptr, im_ptr, cout_ptr, row_start_ptr, row_len_ptr, pid - N, cache_stride, cout_stride, C, cols, m)


def ln_mish_conv(x: torch.Tensor, ln, cache: torch.Tensor, new_cache: torch.Tensor, meta: dict, out: torch.Tensor) -> torch.Tensor:
    """``ln_mish`` followed by ``conv_gather`` of its output in one kernel (fills ``out`` and ``new_cache``)."""
    N, C = x.shape
    R = cache.shape[0]
    assert x.is_contiguous() and out.is_contiguous() and out.shape == (N, 3 * C)
    assert cache.stride(1) == C and cache.stride(2) == 1 and new_cache.stride(1) == C and new_cache.stride(2) == 1
    _ln_mish_conv_kernel[(N + R,)](x, ln[0], ln[1], out, cache, new_cache, meta["tok_row"], meta["tok_pos"], meta["row_start"], meta["row_len"],
                                   cache.stride(0), new_cache.stride(0), N, C, ln[2], BLOCK=triton.next_power_of_2(C))
    return out
