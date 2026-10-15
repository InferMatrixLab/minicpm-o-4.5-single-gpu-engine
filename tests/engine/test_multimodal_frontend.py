"""Multimodal prompt assembly and encoders against the HF reference (32 mixed image/audio/video-frame samples).

The reference runs in the model card's environment (.venv-hfref, transformers 4.51):

    .venv-hfref/bin/python tools/hfref/mm_reference.py results/engine_parity/mm_samples.pkl results/engine_parity/mm_ref_cpu_fp32.pt cpu float32

and this test recomputes the token IDs and the Thinker input embeddings with the engine's front end.
"""

import os
import pickle
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "results" / "engine_parity" / "mm_samples.pkl"
REF = ROOT / "results" / "engine_parity" / "mm_ref_cpu_fp32.pt"
pytestmark = pytest.mark.skipif(not (SAMPLES.exists() and REF.exists()), reason="reference artifacts not generated (see module docstring)")


@pytest.fixture(scope="module")
def results():
    os.environ.setdefault("HF_MODULES_CACHE", "/workspace/models/cgzheng/hf_modules")
    from engine.models.multimodal import MultimodalFrontend
    from engine.testing.mm_samples import MODEL, to_msgs

    f = MultimodalFrontend(MODEL, device="cpu", dtype=torch.float32)
    samples = pickle.loads(SAMPLES.read_bytes())
    ref = torch.load(REF)
    out = []
    for s, r in zip(samples, ref):
        inp = f.build_prompt(to_msgs(s), use_tts_template=s["tts"])
        out.append((inp["input_ids"][0].cpu(), f.embed(inp).float().cpu(), r))
    return f, out


def test_token_ids_identical_on_all_samples(results):
    _, out = results
    assert len(out) == 32
    assert all(torch.equal(ids, r["input_ids"]) for ids, _, r in out)


def test_encoder_embeddings(results):
    f, out = results
    unk = f.tokenizer.convert_tokens_to_ids("<unk>")
    worst = 1.0
    for ids, emb, r in out:
        cos = torch.nn.functional.cosine_similarity(emb, r["embeds"], dim=-1)
        worst = min(worst, float(cos.min()))
        mm = ids == unk  # placeholder positions carry the vision/audio states
        assert mm.any() and float(cos[mm].min()) >= 0.999
    assert worst >= 0.999


def test_dropped_placeholder_is_detected(results):
    f, out = results
    unk = f.tokenizer.convert_tokens_to_ids("<unk>")
    ids, _, r = out[0]
    pos = int((ids == unk).nonzero()[0])
    dropped = torch.cat([ids[:pos], ids[pos + 1:]])
    assert not torch.equal(dropped, r["input_ids"])
