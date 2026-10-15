"""Talker conditioning and teacher-forced codec logits against the HF chat path (AC-3.2), GPU bf16.

Reference: ``tools/hfref/talker_reference.py`` -> results/engine_parity/talker_ref_bf16.pt.  The engine rebuilds the
condition from its own Thinker hidden states over HF's reply tokens (relative L2 <= 1e-2) and replays HF's sampled
codes through its Talker (top-1 agreement >= 99%).  Shifting the hidden/token alignment by one fails.
"""

import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "results" / "engine_parity" / "talker_ref_bf16.pt"
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1" or not REF.exists(),
                                reason="GPU test with the HF reference file")


@torch.inference_mode()
def test_condition_and_teacher_forced_logits():
    from transformers import AutoTokenizer

    from engine.models.talker import Talker
    from engine.models.thinker import Thinker

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    th, ta = Thinker.load(MODEL, device="cuda"), Talker.load(MODEL, device="cuda")
    ref = [r for r in torch.load(REF) if r["codes"] is not None and r["reply_ids"] is not None]
    assert len(ref) >= 30
    worst, agree, total, shifted_worst = 0.0, 0, 0, 1.0
    for r in ref:
        text = tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True, use_tts_template=True, enable_thinking=False)
        prompt = torch.tensor(tok(text)["input_ids"], device="cuda")
        reply = r["reply_ids"].cuda()
        cache = th.new_cache(1, prompt.numel() + reply.numel() + 4)
        row = torch.tensor([0], device="cuda")
        th.forward(th.embed(prompt)[None], cache, row)
        h, _ = th.forward(th.embed(reply)[None], cache, row, all_logits=True)
        cond = ta.condition(reply, h[0])
        want = r["condition"].cuda()
        rel = float((cond.float() - want).norm() / want.norm())
        worst = max(worst, rel)
        shifted = ta.condition(reply[1:], h[0][:-1])  # hidden states one position early relative to the tokens
        shifted_worst = min(shifted_worst, float((shifted.float() - want[1:]).norm() / want[1:].norm()))
        tcache = ta.new_cache(1, cond.shape[0] + r["codes"].numel() + 4)
        lg = [ta.forward(cond[None], tcache, row)]
        codes = r["codes"].cuda()
        for c in codes[:-1]:
            lg.append(ta.forward(ta.code_embed(c.view(1, 1)), tcache, row))
        lg = torch.cat(lg)
        n = min(lg.shape[0], r["logits"].shape[0])
        agree += int((lg[:n].argmax(-1).cpu() == r["logits"][:n].argmax(-1)).sum())
        total += n
    assert worst <= 1e-2, worst
    assert agree / total >= 0.99, agree / total
    assert shifted_worst > 1e-2
