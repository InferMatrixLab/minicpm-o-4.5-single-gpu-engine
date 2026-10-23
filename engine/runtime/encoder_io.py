"""Batched cache input / output of the flow encoder's chunk forward (``t2w_encoder_batched_io``).

``BatchedToken2Wav._encoder`` builds the left-padded attention caches of a call from each stream's cache tensor and
splits the new caches back per stream, which is a handful of small copies per stream (and ``_flow_finish`` trims each
one later).  Here one kernel gathers every stream's cache into the padded ``A1`` / ``A2`` (through a table of the
streams' data pointers), and one kernel writes every stream's new cache, already trimmed as ``_trim`` would (the first
``keep`` frames and the last ``TRIM_SUFFIX``), into a tensor of its own.  Only copies: the caches are equal bit for bit.

A stream's cache is [d1 + d2, 1, nh, 2 t, dk2]: the first stack's t frames tiled twice (``repeat``), then the second
stack's 2 t frames; ``A1`` [d1, rows, nh, L1, dk2] holds the first t of each row right-aligned, ``A2`` [d2, rows, nh, L2,
dk2] the 2 t (``L2 = 2 L1``), zeros on the left.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from engine.runtime.hostcopy import to_device

BLOCK_T = 32


@triton.jit
def _gather_kernel(ptr_tab, off_ptr, a1_ptr, a2_ptr, R, NH, L1, D1: tl.constexpr, DK2: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)  # stream (row)
    layer = tl.program_id(1)
    h = tl.program_id(2)
    t1 = tl.load(off_ptr + i)  # the stream's first-stack frames (its cache holds 2 t1)
    src = tl.load(ptr_tab + i).to(tl.pointer_type(a1_ptr.dtype.element_ty))
    cols = tl.arange(0, DK2)[None, :]
    rows = tl.arange(0, BLOCK)[:, None]
    if layer < D1:
        L, n, dst = L1, t1, a1_ptr + (((layer * R + i) * NH + h) * L1).to(tl.int64) * DK2
    else:
        L, n, dst = 2 * L1, 2 * t1, a2_ptr + ((((layer - D1) * R + i) * NH + h) * 2 * L1).to(tl.int64) * DK2
    s = src + ((layer * NH + h) * 2 * t1).to(tl.int64) * DK2
    pad = L - n
    for t0 in range(0, L, BLOCK):
        t = t0 + rows
        m = (t < L) & (cols < DK2)
        v = tl.load(s + (t - pad).to(tl.int64) * DK2 + cols, mask=m & (t >= pad), other=0.0)
        tl.store(dst + t.to(tl.int64) * DK2 + cols, v, mask=m)


@triton.jit
def _scatter_kernel(ptr_tab, off_ptr, keep_ptr, len_ptr, n1_ptr, n2_ptr, s1_l, s1_r, s1_h, s1_t, s2_l, s2_r, s2_h, s2_t, NH, L1, T1,
                    D1: tl.constexpr, DK2: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    layer = tl.program_id(1)
    h = tl.program_id(2)
    t1 = tl.load(off_ptr + i)
    keep = tl.load(keep_ptr + i)
    out_len = tl.load(len_ptr + i)
    full = 2 * (t1 + T1)  # the untrimmed new cache: [c1, c1, c2] along time per layer stack
    dst = tl.load(ptr_tab + i).to(tl.pointer_type(n1_ptr.dtype.element_ty)) + ((layer * NH + h) * out_len).to(tl.int64) * DK2
    cols = tl.arange(0, DK2)[None, :]
    rows = tl.arange(0, BLOCK)[:, None]
    for q0 in range(0, out_len, BLOCK):
        q = q0 + rows
        m = (q < out_len) & (cols < DK2)
        p = tl.where(q < keep, q, full - out_len + q)  # the frame of the untrimmed cache (identity when not trimmed)
        if layer < D1:
            src = n1_ptr + (layer * s1_l + i * s1_r + h * s1_h).to(tl.int64) + ((L1 - t1 + p % (t1 + T1)) * s1_t).to(tl.int64)
        else:
            src = n2_ptr + ((layer - D1) * s2_l + i * s2_r + h * s2_h).to(tl.int64) + ((2 * L1 - 2 * t1 + p) * s2_t).to(tl.int64)
        tl.store(dst + q.to(tl.int64) * DK2 + cols, tl.load(src + cols, mask=m, other=0.0), mask=m)


def gather_caches(atts: list[torch.Tensor], off1: list[int], A1: torch.Tensor, A2: torch.Tensor) -> None:
    """Fill rows ``0 .. len(atts) - 1`` of ``A1`` [d1, R, nh, L1, dk2] and ``A2`` [d2, R, nh, 2 L1, dk2] (contiguous)."""
    d1, R, nh, L1, dk2 = A1.shape
    d2 = A2.shape[0]
    assert A1.is_contiguous() and A2.is_contiguous() and A2.shape == (d2, R, nh, 2 * L1, dk2) and A1.dtype == A2.dtype
    for a, t in zip(atts, off1):
        assert a.is_contiguous() and a.dtype == A1.dtype and a.shape == (d1 + d2, 1, nh, 2 * t, dk2) and t <= L1, (a.shape, t, L1)
    n = len(atts)
    ptrs = to_device([a.data_ptr() for a in atts], torch.int64, A1.device)
    offs = to_device(off1, torch.int32, A1.device)
    _gather_kernel[(n, d1 + d2, nh)](ptrs, offs, A1, A2, R, nh, L1, D1=d1, DK2=dk2, BLOCK=BLOCK_T)
    return atts  # the sources must live until the kernel ran: the caller keeps them (stream order)


def scatter_caches(new1: torch.Tensor, new2: torch.Tensor, off1: list[int], keeps: list[int], T1: int, L1: int, suffix: int) -> list[torch.Tensor]:
    """Each row's new cache from ``new1`` [d1, R, nh, L1 + T1, dk2] and ``new2`` [d2, R, nh, 2 (L1 + T1), dk2] (any
    strides with a contiguous last dim): ``cat([c1.repeat(1, 1, 1, 2, 1), c2])`` of its real frames, trimmed to its first
    ``keep`` and last ``suffix`` frames when longer than ``keep + suffix``."""
    d1, _, nh, _, dk2 = new1.shape
    d2 = new2.shape[0]
    assert new1.stride(4) == 1 and new2.stride(4) == 1 and new1.dtype == new2.dtype
    full = [2 * (t + T1) for t in off1]
    trimmed = [f > k + suffix for f, k in zip(full, keeps)]  # as _trim: longer than keep + suffix
    lens = [k + suffix if tr else f for f, k, tr in zip(full, keeps, trimmed)]
    kp = [k if tr else f for f, k, tr in zip(full, keeps, trimmed)]  # untrimmed rows: keep = length, the identity frame map
    dev = new1.device
    outs = [torch.empty((d1 + d2, 1, nh, n, dk2), dtype=new1.dtype, device=dev) for n in lens]
    ptrs = to_device([o.data_ptr() for o in outs], torch.int64, dev)
    _scatter_kernel[(len(outs), d1 + d2, nh)](ptrs, to_device(off1, torch.int32, dev), to_device(kp, torch.int32, dev), to_device(lens, torch.int32, dev),
                                              new1, new2, *new1.stride()[:4], *new2.stride()[:4], nh, L1, T1, D1=d1, DK2=dk2, BLOCK=BLOCK_T)
    return outs
