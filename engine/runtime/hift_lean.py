"""HiFT with fewer kernels per call (``t2w_hift_lean``): the same vocoder within cuDNN's run-to-run spread.

* Snake (``x + 1 / (alpha + 1e-9) * sin(x * alpha) ** 2``, 72 activations a call) is one Triton kernel instead of five
  passes over the activation (the product, ``sin``, the square, the reciprocal's product, the sum).  The kernel rounds
  each step as torch does: products and the sum with ``mul_rn`` / ``add_rn`` (no FMA contraction), ``sin`` from
  libdevice (``__nv_sinf``, torch's ``sinf``), the square as a product (torch's ``pow(x, 2)``); ``1 / (alpha + 1e-9)``
  and ``exp(alpha)`` are computed once by torch with the reference's ops.  The kernel equals the module bit for bit.
* The weight-normalised convolutions keep their weight as computed once (``remove_parametrizations`` with
  ``leave_parametrized``: the same ``_weight_norm`` the reference evaluates in every forward).

The whole vocoder is not bitwise comparable: HiFT's cuDNN convolutions already differ run to run (~1e-6 relative in
strict FP32), and the lean one stays within 1e-5 of the reference (``tests/engine/test_hift_lean.py``).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice


@triton.jit
def _snake_kernel(x_ptr, a_ptr, inv_ptr, y_ptr, C, T, BLOCK: tl.constexpr):
    row = tl.program_id(0)  # b * C + c
    t = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = t < T
    c = row % C
    a = tl.load(a_ptr + c)
    inv = tl.load(inv_ptr + c)
    x = tl.load(x_ptr + row * T + t, mask=m, other=0.0)
    s = libdevice.sin(libdevice.mul_rn(x, a))
    tl.store(y_ptr + row * T + t, libdevice.add_rn(x, libdevice.mul_rn(inv, libdevice.mul_rn(s, s))), mask=m)


def snake(x: torch.Tensor, alpha: torch.Tensor, inv: torch.Tensor) -> torch.Tensor:
    """``x`` [B, C, T] fp32; ``alpha`` (after ``exp`` for log-scale alphas) and ``inv = 1 / (alpha + 1e-9)`` [C] fp32."""
    assert x.dtype == torch.float32 and x.dim() == 3
    x = x.contiguous()
    y = torch.empty_like(x)
    B, C, T = x.shape
    block = 1024
    _snake_kernel[(B * C, triton.cdiv(T, block))](x, alpha, inv, y, C, T, BLOCK=block)
    return y


def _lean_snake_forward(self, x):
    return snake(x, self._lean_alpha, self._lean_inv)


@torch.no_grad()
def make_lean(hift) -> dict:
    """Fold HiFT's weight norms and fuse its Snake activations in place; returns what was changed."""
    from torch.nn.utils import parametrize

    wn = [mod for mod in hift.modules() if parametrize.is_parametrized(mod, "weight")]
    for mod in wn:
        parametrize.remove_parametrizations(mod, "weight", leave_parametrized=True)
    n_wn = len(wn)
    n_snake = 0
    for mod in hift.modules():
        if type(mod).__name__ == "Snake":
            alpha = mod.alpha.detach().unsqueeze(0).unsqueeze(-1)  # as the reference lines it up, then flattened
            if mod.alpha_logscale:
                alpha = torch.exp(alpha)
            mod._lean_alpha = alpha.reshape(-1).float().contiguous()
            mod._lean_inv = (1.0 / (alpha + mod.no_div_by_zero)).reshape(-1).float().contiguous()
            mod.forward = _lean_snake_forward.__get__(mod)
            n_snake += 1
    return {"weight_norms_folded": n_wn, "snakes_fused": n_snake}
