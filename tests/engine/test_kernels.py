"""Fused reference-rounding kernels against their torch definitions (GPU)."""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")


@pytest.mark.parametrize("n,width", [(1, 4096), (37, 4096), (512, 128), (8, 768)])
def test_rms_norm_hf_matches_reference(n, width):
    from engine.runtime.kernels import rms_norm_hf
    from engine.runtime.paged import hf_rms_norm

    g = torch.Generator(device="cuda").manual_seed(n)
    x = (torch.randn(n, width, device="cuda", generator=g) * 3).to(torch.bfloat16)
    w = (torch.randn(width, device="cuda", generator=g) * 0.2 + 1).to(torch.bfloat16)
    a, b = rms_norm_hf(x, w, 1e-6), hf_rms_norm(x, w, 1e-6)
    diff = (a.float() - b.float()).abs()
    ulp = b.float().abs().clamp_min(1e-30) * 2 ** -7  # one bf16 ulp (8-bit significand)
    assert bool((diff <= ulp * 1.01).all()), float((diff / ulp).max())
    assert float((a != b).float().mean()) <= 1e-3  # identical except rare one-ulp rows
