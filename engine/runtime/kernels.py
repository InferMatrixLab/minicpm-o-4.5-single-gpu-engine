"""Fused kernels with the reference's rounding (Triton).

``rms_norm_hf(x, w, eps)`` computes, in one kernel per call, what ``engine.runtime.paged.hf_rms_norm`` computes with
eight: ``w * (x.float() * rsqrt(mean(x.float() ** 2) + eps)).to(x.dtype)``, i.e. normalise in fp32, round to the
activation dtype, then the product with the weight rounded to the activation dtype.  The fp32 mean may sum in a
different order than torch's reduction, so a result can differ by one ulp of the activation dtype in rare rows
(``tests/engine/test_kernels.py`` measures it).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _rms_norm_hf_kernel(x_ptr, w_ptr, y_ptr, n_cols, stride_x, stride_y, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < n_cols
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / n_cols
    xn = (x * tl.math.rsqrt(var + eps)).to(y_ptr.dtype.element_ty)  # round to the activation dtype, as the reference
    w = tl.load(w_ptr + cols, mask=mask, other=0.0)
    y = (xn.to(tl.float32) * w.to(tl.float32)).to(y_ptr.dtype.element_ty)  # the dtype product, rounded once
    tl.store(y_ptr + row * stride_y + cols, y, mask=mask)


def rms_norm_hf(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    if x2.stride(-1) != 1:
        x2 = x2.contiguous()
    y = torch.empty_like(x2)
    n = x2.shape[1]
    _rms_norm_hf_kernel[(x2.shape[0],)](x2, w, y, n, x2.stride(0), y.stride(0), eps, BLOCK=triton.next_power_of_2(n))
    return y.view(shape)
