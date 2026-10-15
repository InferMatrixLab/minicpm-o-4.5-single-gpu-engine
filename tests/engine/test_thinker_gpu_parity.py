"""Thinker parity against HF bf16 eager on the same GPU, 64 prompts (AC-3.1 tolerances).

Reference: ``tools/hfref/thinker_reference.py`` -> results/engine_parity/thinker_ref_bf16.pt (run on the same GPU).
"""

import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "results" / "engine_parity" / "thinker_ref_bf16.pt"
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1" or not REF.exists(),
                                reason="GPU test with the HF reference file")


@pytest.fixture(scope="module")
def engine():
    from engine.models.thinker import Thinker
    from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV

    th = Thinker.load(MODEL, device="cuda")
    c = th.cfg
    spec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
    kv = PagedKV(spec, 4096)
    return th, DecoderRunner(th.w, spec, kv), kv


@torch.inference_mode()
def test_teacher_forced_and_greedy(engine):
    th, run, kv = engine
    ref = torch.load(REF)
    assert len(ref) == 64
    agree = total = identical = 0
    worst_cos = 1.0
    bad_divergence = []
    for i, r in enumerate(ref):
        toks = r["tokens"].cuda()
        lay = kv.layout([i], [toks.numel()])
        h = run.forward(th.embed(toks), lay)
        kv.release(i)
        cos = torch.nn.functional.cosine_similarity(h.float(), r["hidden"].cuda().float(), dim=-1)
        worst_cos = min(worst_cos, float(cos.min()))
        logits = torch.nn.functional.linear(h, th.w["lm_head.weight"]).float()
        agree += int((logits.argmax(-1).cpu() == r["top1"]).sum())
        total += toks.numel()
        # greedy: engine continuation from the prompt
        p = r["prompt_len"]
        seq = 10_000 + i
        lay = kv.layout([seq], [p])
        hh = run.forward(th.embed(toks[:p]), lay)
        gen = []
        nxt = int(torch.nn.functional.linear(hh[-1:], th.w["lm_head.weight"]).argmax())
        ref_gen = r["tokens"][p:].tolist()
        for _ in range(len(ref_gen)):
            gen.append(nxt)
            if nxt in (151645, 151643):
                break
            hh = run.forward(th.embed(torch.tensor([nxt], device="cuda")), kv.layout([seq], [1]))
            nxt = int(torch.nn.functional.linear(hh[-1:], th.w["lm_head.weight"]).argmax())
        kv.release(seq)
        if gen == ref_gen:
            identical += 1
        else:
            j = next(k for k in range(min(len(gen), len(ref_gen))) if gen[k] != ref_gen[k]) if any(a != b for a, b in zip(gen, ref_gen)) else min(len(gen), len(ref_gen))
            if float(r["margin"][p + j - 1]) >= 0.1:
                bad_divergence.append((i, j, float(r["margin"][p + j - 1])))
    assert worst_cos >= 0.999, worst_cos
    assert agree / total >= 0.995, agree / total
    assert identical >= 0.9 * len(ref), identical
    assert not bad_divergence, bad_divergence
