"""Thinker (Qwen3) port against transformers' Qwen3ForCausalLM with the checkpoint weights (teacher forced, CPU fp32).

fp32 isolates the implementation from bf16 accumulation order; the bf16 tolerances are checked on the GPU.
"""

import json
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

from engine.models.thinker import Thinker

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="checkpoint not available")


@pytest.fixture(scope="module")
def models():
    ours = Thinker.load(MODEL, dtype=torch.float32)
    c = json.loads((MODEL / "config.json").read_text())
    cfg = Qwen3Config(**{k: c[k] for k in ("hidden_size", "intermediate_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads",
                                           "head_dim", "rope_theta", "rms_norm_eps", "vocab_size", "max_position_embeddings", "hidden_act")},
                      tie_word_embeddings=False, attention_bias=False)
    with torch.device("meta"):
        ref = Qwen3ForCausalLM(cfg)
    ref.load_state_dict(ours.w, strict=True, assign=True)
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

    ref.model.rotary_emb = Qwen3RotaryEmbedding(cfg)  # non-persistent buffer, not in the checkpoint
    ref.eval()
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    return ours, ref, tok


@torch.inference_mode()
def test_teacher_forced_hidden_and_logits(models):
    ours, ref, tok = models
    text = tok.apply_chat_template([{"role": "user", "content": "Please read the following text out loud in English: The quick brown fox."}],
                                   tokenize=False, add_generation_prompt=True, use_tts_template=True, enable_thinking=False)
    ids = torch.tensor(tok(text)["input_ids"] + tok("The quick brown fox.<|tts_eos|>")["input_ids"])[None]
    out = ref(input_ids=ids, output_hidden_states=True)
    want_h, want_l = out.hidden_states[-1][0].float(), out.logits[0].float()
    cache = ours.new_cache(1, ids.shape[1] + 4)
    h1, l1 = ours.forward(ours.embed(ids[:, :20]), cache, torch.tensor([0]), all_logits=True)  # prefill split in two, then decode steps
    h2, l2 = ours.forward(ours.embed(ids[:, 20:-3]), cache, torch.tensor([0]), all_logits=True)
    hs, ls = [h1[0], h2[0]], [l1[0], l2[0]]
    for t in range(ids.shape[1] - 3, ids.shape[1]):
        h, lg = ours.forward(ours.embed(ids[:, t:t + 1]), cache, torch.tensor([0]), all_logits=True)
        hs.append(h[0])
        ls.append(lg[0])
    got_h, got_l = torch.cat(hs).float(), torch.cat(ls)
    cos = torch.nn.functional.cosine_similarity(got_h, want_h, dim=-1)
    assert cos.min() >= 0.99999, cos.min()
    assert (got_l.argmax(-1) == want_l.argmax(-1)).all()
    assert ((got_l - want_l).norm() / want_l.norm()).item() < 1e-4
