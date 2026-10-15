"""Track S Talker parity against HF streaming_generate (AC-3.5), GPU bf16.

Reference: ``tools/hfref/track_s_reference.py`` -> results/engine_parity/track_s_ref_bf16.pt.
- Teacher-forced codec logits: HF's chunk conditions and sampled frames replayed through the engine Talker in the
  reference's order (condition prefill, then every frame of the chunk; a chunk's EOS is never fed); top-1 agreement
  with HF's logits (AC-3.5 asks >= 99%; the bf16 noise analysis of AC-3.2 applies and is reported).
- Conditions: the engine rebuilds every chunk's condition from its own Thinker over HF's exact Thinker input IDs, with
  the Track S chunk rule; shapes must match exactly and values within relative L2 1e-2.
"""

import json
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
REF = ROOT / "results" / "engine_parity" / "track_s_ref_bf16.pt"
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1" or not REF.exists(),
                                reason="GPU test with the HF Track S reference file")


@torch.inference_mode()
def test_track_s_teacher_forced_and_conditions():
    from engine.models.talker import Talker
    from engine.models.thinker import Thinker
    from engine.runtime.track_s import THINKER_CHUNK

    ref = [r for r in torch.load(REF) if r["chunks"]]
    assert len(ref) >= 8
    ta, th = Talker.load(MODEL, device="cuda"), Thinker.load(MODEL, device="cuda")
    row = torch.tensor([0], device="cuda")
    agree = total = 0
    worst_cond, shape_ok = 0.0, True
    for r in ref:
        # teacher-forced Talker over HF's conditions and frames
        total_len = sum(c["condition"].shape[0] + len(c["frames"]) for c in r["chunks"]) + 8
        cache = ta.new_cache(1, total_len)
        for c in r["chunks"]:
            lg = [ta.forward(c["condition"].cuda().to(ta.dtype)[None], cache, row)]
            for f in c["frames"]:
                lg.append(ta.forward(ta.code_embed(torch.tensor([[f]], device="cuda")), cache, row))
            lg = torch.cat(lg).cpu()
            if c["logits"] is not None:
                n = min(lg.shape[0], c["logits"].shape[0])
                agree += int((lg[:n].argmax(-1) == c["logits"][:n].argmax(-1)).sum())
                total += n
        # conditions from the engine's own Thinker hidden states over HF's exact Thinker inputs
        ids, reply = r["llm_input_ids"], r["reply_ids"]
        R = reply.numel()
        # the fed reply tokens are a contiguous block of HF's Thinker inputs; the hidden state of reply[j] is the
        # output at the position where reply[j] was the input
        P = next((q for q in range(ids.numel() - R, -1, -1) if torch.equal(ids[q:q + R], reply)), None)
        assert P is not None, "reply tokens not found in the recorded Thinker inputs"
        tcache = th.new_cache(1, P + R + 4)
        hid, _ = th.forward(th.embed(ids[:P + R].cuda())[None], tcache, row, all_logits=True)
        h = hid[0, P:P + R]
        reply = reply.cuda()
        passed = 0
        for ci, c in enumerate(r["chunks"]):
            final = c["text_finished"]
            end = min(R, THINKER_CHUNK * (ci + 1)) if not final else R
            cond = ta.condition(reply[passed:end], h[passed:end])
            if not final:
                cond = torch.cat([cond[:-2], cond[-1:]], dim=0)
            want = c["condition"].cuda()
            if cond.shape != want.shape:
                shape_ok = False
            else:
                worst_cond = max(worst_cond, float((cond.float() - want).norm() / want.norm()))
            passed = end
    report = {"talker_top1_agreement": agree / max(total, 1), "frames": total, "worst_condition_rel_l2": worst_cond, "condition_shapes_match": shape_ok}
    (ROOT / "results" / "engine_parity" / "track_s_parity_report.json").write_text(json.dumps(report, indent=1))
    assert shape_ok, report
    assert worst_cond <= 1e-2, report
    assert report["talker_top1_agreement"] >= 0.99, report
