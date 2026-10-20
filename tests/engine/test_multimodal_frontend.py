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

from bench import paths

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "results" / "engine_parity" / "mm_samples.pkl"
REF = ROOT / "results" / "engine_parity" / "mm_ref_cpu_fp32.pt"
pytestmark = pytest.mark.skipif(not (SAMPLES.exists() and REF.exists()), reason="reference artifacts not generated (see module docstring)")


@pytest.fixture(scope="module")
def results():
    os.environ.setdefault("HF_MODULES_CACHE", str(paths.HF_MODULES_CACHE))
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


REF_GPU = ROOT / "results" / "engine_parity" / "mm_ref_cuda_bf16.pt"


@pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1" or not REF_GPU.exists(),
                    reason="GPU test with the HF bf16 reference (tools/hfref/mm_reference.py ... cuda bfloat16)")
def test_gpu_bf16_embeddings_match_hf_bf16():
    """The served path (cuda, bf16) against HF on the same GPU in bf16: token IDs identical and placeholder embeddings
    (SigLIP + resampler, Whisper + projection) at cosine >= 0.999 (AC-3.6, literal).

    Measured 2026-10-16: images bit-identical; audio min 0.9969 (the reference env's torch picks other bf16 SDPA
    kernels for the Whisper encoder: layer-0 LayerNorm outputs are identical, its attention output is not), while HF
    bf16 itself is at 0.908 from HF fp32.  Pending a user decision analogous to AC-3.1 (judge against HF's own bf16
    noise); the CPU fp32 parity above passes the literal bound."""
    os.environ.setdefault("HF_MODULES_CACHE", str(paths.HF_MODULES_CACHE))
    from engine.models.multimodal import MultimodalFrontend
    from engine.testing.mm_samples import MODEL, to_msgs

    f = MultimodalFrontend(MODEL, device="cuda", dtype=torch.bfloat16)
    unk = f.tokenizer.convert_tokens_to_ids("<unk>")
    samples, ref = pickle.loads(SAMPLES.read_bytes()), torch.load(REF_GPU)
    worst = 1.0
    for s, r in zip(samples, ref):
        inp = f.build_prompt(to_msgs(s), use_tts_template=s["tts"])
        ids = inp["input_ids"][0].cpu()
        assert torch.equal(ids, r["input_ids"])
        cos = torch.nn.functional.cosine_similarity(f.embed(inp).float().cpu(), r["embeds"].float(), dim=-1)
        mm = ids == unk
        worst = min(worst, float(cos[mm].min()))
    assert worst >= 0.999, worst
