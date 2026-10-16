"""Thinker parity against HF bf16 eager on the same GPU, 64 prompts, judged against HF's own bf16 noise.

Criterion (user decision 2026-10-15): the engine is no further from HF bf16 eager than HF's own bf16 SDPA path is,
on the 1%-low and the mean per-token hidden-state cosine and on top-1 agreement over non-tied positions (paired
prompt-level bootstrap: the engine must not be significantly worse at 95%), and on greedy identity (also >= 90%, with
every divergence at an HF top-1/top-2 margin no larger than the largest one at which HF's own SDPA path diverges).  Min and mean cosine are reported.

Teacher-forced hidden states come from the paged prefill runner; greedy generation from the served graph decoder
(``ThinkerDecoder``: fused reference-rounding kernels).

References: ``tools/hfref/thinker_reference.py`` (bf16 eager) -> results/engine_parity/thinker_ref_bf16.pt, and
``tools/hfref/ac3_noise_floor.py`` -> results/engine_parity/hf_bf16_sdpa_per_prompt.pt (both on the same GPU).
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "results" / "engine_parity" / "thinker_ref_bf16.pt"
SDPA = ROOT / "results" / "engine_parity" / "hf_bf16_sdpa_per_prompt.pt"
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1" or not REF.exists() or not SDPA.exists(),
                                reason="GPU test with the HF reference files")


def not_significantly_worse(eng: list, hf: list, stat, higher_is_better=True, reps=2000, seed=0) -> tuple[bool, float, float]:
    """Paired prompt-level bootstrap of stat(engine) - stat(HF SDPA); the engine fails only if it is worse with 95% confidence."""
    rng = np.random.default_rng(seed)
    n = len(eng)
    d = []
    for _ in range(reps):
        idx = rng.integers(0, n, n)
        a, b = stat([eng[i] for i in idx]), stat([hf[i] for i in idx])
        d.append(a - b if higher_is_better else b - a)
    upper = float(np.quantile(d, 0.95))  # if even the optimistic end is below zero, the engine is worse
    return upper >= 0.0, float(stat(eng) - stat(hf)), upper


@torch.inference_mode()
def test_thinker_within_hf_noise():
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
    from engine.runtime.thinker_decode import ThinkerDecoder

    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    kv = PagedKV(spec, 4096)
    run = DecoderRunner(th.w, spec, kv)
    dec = ThinkerDecoder(th, kv, spec, buckets=(1,))  # the served decode path: graphs, fused reference-rounding kernels
    ref, hf = torch.load(REF), torch.load(SDPA)
    assert len(ref) == len(hf) == 64
    eng, divergences = [], []
    for i, r in enumerate(ref):
        t = r["tokens"].cuda()
        h = run.forward(th.embed(t), kv.layout([i], [t.numel()]))
        kv.release(i)
        cos = torch.nn.functional.cosine_similarity(h.float(), r["hidden"].cuda().float(), dim=-1).cpu()
        top1 = torch.nn.functional.linear(h, th.w["lm_head.weight"]).float().argmax(-1).cpu()
        p, seq = r["prompt_len"], 10_000 + i
        hh = run.forward(th.embed(t[:p]), kv.layout([seq], [p]))
        gen, ref_gen = [], r["tokens"][p:].tolist()
        nxt = int(torch.nn.functional.linear(hh[-1:], th.w["lm_head.weight"]).argmax())
        while len(gen) < len(ref_gen):
            gen.append(nxt)
            if nxt in (151645, 151643):
                break
            nxt = dec.step([seq], [nxt])[0][0]
        kv.release(seq)
        same = gen == ref_gen
        if not same:
            j = next((k for k in range(min(len(gen), len(ref_gen))) if gen[k] != ref_gen[k]), min(len(gen), len(ref_gen)))
            divergences.append((i, j, float(r["margin"][p + j - 1])))
        eng.append({"cos": cos, "top1": top1, "greedy_identical": same})
    untied = [r["margin"] > 0 for r in ref]
    p01 = lambda xs: float(torch.cat([x["cos"] for x in xs]).quantile(0.01))  # noqa: E731
    mean = lambda xs: float(torch.cat([x["cos"] for x in xs]).mean())  # noqa: E731

    def top1(which):
        return [{"agree": (x["top1"][u] == r["top1"][u]).float(), "n": int(u.sum())} for x, r, u in zip(which, ref, untied)]

    agree = lambda xs: float(torch.cat([x["agree"] for x in xs]).mean())  # noqa: E731
    greedy = lambda xs: float(np.mean([x["greedy_identical"] for x in xs]))  # noqa: E731
    checks = {"cos_p01": not_significantly_worse(eng, hf, p01), "cos_mean": not_significantly_worse(eng, hf, mean),
              "top1_untied": not_significantly_worse(top1(eng), top1(hf), agree), "greedy": not_significantly_worse(eng, hf, greedy)}
    report = {"engine": {"cos_min": float(torch.cat([x["cos"] for x in eng]).min()), "cos_p01": p01(eng), "cos_mean": mean(eng),
                         "top1_untied": agree(top1(eng)), "greedy_identical": greedy(eng)},
              "hf_sdpa": {"cos_min": float(torch.cat([x["cos"] for x in hf]).min()), "cos_p01": p01(hf), "cos_mean": mean(hf),
                          "top1_untied": agree(top1(hf)), "greedy_identical": greedy(hf)},
              "checks": {k: {"not_worse": v[0], "difference": v[1], "upper95": v[2]} for k, v in checks.items()}, "divergences": divergences}
    (ROOT / "results" / "engine_parity" / "thinker_parity_report.json").write_text(json.dumps(report, indent=1))
    assert all(v[0] for v in checks.values()), report["checks"]
    assert report["engine"]["greedy_identical"] >= 0.9
    # "divergences only at small margins": no engine divergence at a larger HF top-1/top-2 margin than HF's own SDPA
    # path diverges at (bf16 logits are resolved in 0.125-0.25 steps near the top logits, so a fixed 0.1 is below the
    # format's resolution)
    hf_margins = [x["diverge_margin"] for x in hf if x.get("diverge_margin") is not None]
    assert hf_margins, "the noise-floor file predates divergence records; rerun tools/hfref/ac3_noise_floor.py"
    report["hf_sdpa_divergence_margins"] = hf_margins
    (ROOT / "results" / "engine_parity" / "thinker_parity_report.json").write_text(json.dumps(report, indent=1))
    assert all(m <= max(hf_margins) for _, _, m in divergences), (divergences, hf_margins)
