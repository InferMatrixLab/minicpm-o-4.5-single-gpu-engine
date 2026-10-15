"""FP8 (e4m3) linear layers for the Thinker (an approved approximation, qualified by AC-3.7).

Weights: per output channel scales (``max |w| / 448``), stored e4m3.  Activations: dynamic per-token scales.  The GEMM
is ``torch._scaled_mm`` with row-wise scales (sm_89+), output in the activation dtype.  Rows are padded to a multiple
of 16 for the kernel; the padding is dropped.  Graph-capturable (no host work).
"""

from __future__ import annotations

import torch

E4M3_MAX = 448.0


class Fp8Weight:
    __slots__ = ("w", "scale", "out_features")

    def __init__(self, w: torch.Tensor):
        """``w`` [out, in] (bf16/fp32) -> e4m3 weight and fp32 per-row scales."""
        wf = w.float()
        scale = (wf.abs().amax(dim=1) / E4M3_MAX).clamp_min(1e-12)
        self.w = (wf / scale[:, None]).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn).contiguous()
        self.scale = scale[None, :].contiguous()  # [1, out]
        self.out_features = w.shape[0]


def fp8_linear(x: torch.Tensor, w: Fp8Weight) -> torch.Tensor:
    """``x`` [n, in] -> ``x @ W^T`` [n, out] in ``x.dtype`` with per-token activation scales."""
    n = x.shape[0]
    pad = (-n) % 16
    xs = (x.abs().amax(dim=-1, keepdim=True).float() / E4M3_MAX).clamp_min(1e-12)
    x8 = (x.float() / xs).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    if pad:
        x8 = torch.cat([x8, x8.new_zeros(pad, x8.shape[1])])
        xs = torch.cat([xs, xs.new_ones(pad, 1)])
    y = torch._scaled_mm(x8, w.w.t(), scale_a=xs, scale_b=w.scale, out_dtype=x.dtype)
    return y[:n] if pad else y


def linear(x: torch.Tensor, w) -> torch.Tensor:
    """``F.linear`` for a bf16 weight or an ``Fp8Weight``."""
    if isinstance(w, Fp8Weight):
        return fp8_linear(x, w)
    return torch.nn.functional.linear(x, w)


def quantize_layers(layers: list[dict], names=("qkv", "o", "gate_up", "down")) -> list[dict]:
    """The fused per-layer weights (``engine.runtime.paged.fuse_weights``) with the GEMM weights in FP8."""
    return [{k: (Fp8Weight(v) if k in names else v) for k, v in L.items()} for L in layers]
