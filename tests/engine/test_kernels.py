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


def _ulp_close(a, b, rate=1e-3):
    diff = (a.float() - b.float()).abs()
    ulp = b.float().abs().clamp_min(1e-30) * 2 ** -7
    assert bool((diff <= ulp * 1.01).all()), float((diff / ulp).max())
    assert float((a != b).float().mean()) <= rate


@pytest.mark.parametrize("n", [1, 8, 33])
def test_add_rms_norm_hf_matches_reference(n):
    from engine.runtime.kernels import add_rms_norm_hf
    from engine.runtime.paged import hf_rms_norm

    g = torch.Generator(device="cuda").manual_seed(n)
    x, r = [(torch.randn(n, 4096, device="cuda", generator=g) * 3).to(torch.bfloat16) for _ in range(2)]
    w = (torch.randn(4096, device="cuda", generator=g) * 0.2 + 1).to(torch.bfloat16)
    s, y = add_rms_norm_hf(x, r, w, 1e-6)
    assert torch.equal(s, x + r)  # the residual sum is exact
    _ulp_close(y, hf_rms_norm(x + r, w, 1e-6))


@pytest.mark.parametrize("n", [1, 8, 33])
def test_qk_norm_rope_hf_matches_reference(n):
    from engine.runtime.kernels import qk_norm_rope_hf
    from engine.runtime.paged import hf_rms_norm, hf_rope, hf_rope_tables

    nq, nkv, d = 32, 8, 128
    g = torch.Generator(device="cuda").manual_seed(n)
    qkv = (torch.randn(n, (nq + 2 * nkv) * d, device="cuda", generator=g) * 2).to(torch.bfloat16)
    qw, kw = [(torch.randn(d, device="cuda", generator=g) * 0.2 + 1).to(torch.bfloat16) for _ in range(2)]
    pos = torch.randint(0, 4000, (n,), device="cuda", generator=g, dtype=torch.int32)
    tables = hf_rope_tables(pos, d, 1e6, torch.bfloat16)
    q, k, v = qk_norm_rope_hf(qkv, qw, kw, *tables, nq, nkv, d, 1e-6)
    rq = hf_rms_norm(qkv[:, :nq * d].reshape(-1, d), qw, 1e-6).view(-1, nq, d)
    rk = hf_rms_norm(qkv[:, nq * d:(nq + nkv) * d].reshape(-1, d), kw, 1e-6).view(-1, nkv, d)
    rq, rk = hf_rope(rq, rk, pos, 1e6, tables)
    assert torch.equal(v, qkv[:, (nq + nkv) * d:].reshape(-1, nkv, d))
    _ulp_close(q, rq, 2e-3)
    _ulp_close(k, rk, 2e-3)


@pytest.mark.parametrize("n", [1, 8, 33])
def test_silu_mul_hf_matches_reference(n):
    import torch.nn.functional as F

    from engine.runtime.kernels import silu_mul_hf

    g = torch.Generator(device="cuda").manual_seed(n)
    gu = (torch.randn(n, 2 * 12288, device="cuda", generator=g) * 3).to(torch.bfloat16)
    a, b = gu.chunk(2, dim=-1)
    _ulp_close(silu_mul_hf(gu), F.silu(a) * b, 1e-4)


def test_counter_uniform_equals_the_sampler_hash():
    """``counter_uniform`` is ``talker_sampler.uniform`` bit for bit: seeds over the whole int64 range (negative ones
    included), frame indices past 2**32."""
    from engine.runtime.kernels import counter_uniform
    from engine.talker_sampler import uniform

    g = torch.Generator(device="cuda").manual_seed(0)
    seeds = torch.cat([torch.randint(-2**63, 2**63 - 1, (4000,), device="cuda", generator=g, dtype=torch.int64),
                       torch.tensor([0, 1, -1, 2**32 - 1, 2**32, 2**63 - 1, -2**63], device="cuda")])
    steps = torch.randint(0, 2**40, seeds.shape, device="cuda", generator=g, dtype=torch.int64)
    steps[:300] = torch.arange(300, device="cuda")
    assert torch.equal(counter_uniform(seeds, steps), uniform(seeds, steps))


@pytest.mark.parametrize("n", [1, 8, 33, 300])
def test_rope_hf_equals_the_torch_ops(n):
    """RoPE only (the Talker: no q/k norms): no reduction, so bit for bit."""
    from engine.runtime.kernels import rope_hf
    from engine.runtime.paged import hf_rope, hf_rope_tables

    nq, nkv, d = 16, 8, 128
    g = torch.Generator(device="cuda").manual_seed(n)
    qkv = (torch.randn(n, (nq + 2 * nkv) * d, device="cuda", generator=g) * 2).to(torch.bfloat16)
    pos = torch.randint(0, 4000, (n,), device="cuda", generator=g, dtype=torch.int32)
    tables = hf_rope_tables(pos, d, 1e4, torch.bfloat16)
    q, k, v = rope_hf(qkv, *tables, nq, nkv, d)
    rq, rk = hf_rope(qkv[:, :nq * d].reshape(-1, nq, d).contiguous(), qkv[:, nq * d:(nq + nkv) * d].reshape(-1, nkv, d).contiguous(), pos, 1e4, tables)
    assert torch.equal(v, qkv[:, (nq + nkv) * d:].reshape(-1, nkv, d))
    assert torch.equal(q, rq) and torch.equal(k, rk)
