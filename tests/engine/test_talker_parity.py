"""Talker conditioning and teacher-forced codec logits against transformers' LlamaModel with the checkpoint weights."""


import pytest
import torch
import torch.nn.functional as F
from transformers import LlamaConfig, LlamaModel

from engine.models.talker import KVCache, Talker, load_prefixed  # noqa: F401

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="checkpoint not available")


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    ours = Talker.load(MODEL, dtype=torch.float32)
    c = ours.cfg
    # MiniCPMTTS.__init__ (modeling_minicpmo.py:4256-4268): LlamaConfig with these fields, defaults otherwise
    ref = LlamaModel(LlamaConfig(hidden_size=c.hidden_size, intermediate_size=c.intermediate_size, num_attention_heads=c.num_heads,
                                 num_hidden_layers=c.num_layers, num_key_value_heads=c.num_kv_heads, max_position_embeddings=c.max_positions,
                                 attn_implementation="sdpa"))
    sd = {k[len("model."):]: v for k, v in ours.w.items() if k.startswith("model.")}
    ref.load_state_dict(sd, strict=True)
    ref.eval()
    head = torch.nn.utils.parametrizations.weight_norm(torch.nn.Linear(c.hidden_size, c.num_audio_tokens, bias=False), name="weight")
    head.parametrizations.weight.original0.data = ours.w["head_code.0.parametrizations.weight.original0"]
    head.parametrizations.weight.original1.data = ours.w["head_code.0.parametrizations.weight.original1"]
    return ours, ref, head


def hf_condition(ours, ids, hidden):
    """_generate_speech_non_streaming (modeling_minicpmo.py:1314-1347)."""
    w = ours.w
    llm_embeds = F.embedding(ids, w["emb_text.weight"])
    h = F.linear(hidden, w["projector_semantic.linear1.weight"], w["projector_semantic.linear1.bias"]).relu()
    h = F.normalize(F.linear(h, w["projector_semantic.linear2.weight"], w["projector_semantic.linear2.bias"]), p=2, dim=-1)
    bos = F.embedding(torch.tensor([ours.cfg.audio_bos_token_id]), w["emb_text.weight"])
    eos = F.embedding(torch.tensor([ours.cfg.text_eos_token_id]), w["emb_text.weight"])
    return torch.cat([torch.ones([0, ours.cfg.hidden_size]), llm_embeds + h, eos, bos], dim=0)[None]


@torch.inference_mode()
def test_condition_and_teacher_forced_logits(models):
    ours, ref, head = models
    gen = torch.Generator().manual_seed(1)
    ids = torch.randint(0, 150000, (17,), generator=gen)
    hidden = torch.randn(17, 4096, generator=gen)
    cond = ours.condition(ids, hidden)
    want_cond = hf_condition(ours, ids, hidden)
    assert torch.allclose(cond, want_cond[0], rtol=1e-5, atol=1e-6)
    codes = torch.randint(0, 6561, (40,), generator=gen)
    # reference: MiniCPMTTS.generate (modeling_minicpmo.py:4359-4418) with teacher-forced codes
    out = ref(inputs_embeds=want_cond, position_ids=torch.arange(cond.shape[0])[None], use_cache=True)
    pkv, want = out.past_key_values, [head(out.last_hidden_state[:, -1]).float()]
    for t in range(1, 40):
        e = F.embedding(codes[t - 1:t], ours.w["emb_code.0.weight"])[None]
        out = ref(inputs_embeds=e, position_ids=torch.tensor([[cond.shape[0] + t - 1]]), past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        want.append(head(out.last_hidden_state[:, -1]).float())
    cache = ours.new_cache(1, 128)
    rows = torch.tensor([0])
    got = [ours.forward(cond[None], cache, rows)]
    for t in range(1, 40):
        got.append(ours.forward(ours.code_embed(codes[t - 1:t])[None], cache, rows))
    got, want = torch.cat(got), torch.cat(want)
    assert (got.argmax(-1) == want.argmax(-1)).float().mean() == 1.0
    rel = ((got - want).norm() / want.norm()).item()
    assert rel < 1e-4, rel


@torch.inference_mode()
def test_batched_rows_match_single_rows(models):
    ours, _, _ = models
    gen = torch.Generator().manual_seed(2)
    conds = [ours.condition(torch.randint(0, 150000, (n,), generator=gen), torch.randn(n, 4096, generator=gen)) for n in (9, 14)]
    codes = torch.randint(0, 6561, (2, 6), generator=gen)
    singles = []
    for j, cnd in enumerate(conds):
        cache = ours.new_cache(1, 64)
        r = torch.tensor([0])
        steps = [ours.forward(cnd[None], cache, r)] + [ours.forward(ours.code_embed(codes[j, t:t + 1])[None], cache, r) for t in range(6)]
        singles.append(torch.cat(steps))
    cache = ours.new_cache(3, 64)  # row 1 is an idle slot
    a = ours.forward(conds[0][None], cache, torch.tensor([0]))
    b = ours.forward(conds[1][None], cache, torch.tensor([2]))
    batched = [[a], [b]]
    for t in range(6):
        e = ours.code_embed(codes[:, t:t + 1])
        lg = ours.forward(e, cache, torch.tensor([0, 2]))  # rows at different positions in one call
        batched[0].append(lg[0:1])
        batched[1].append(lg[1:2])
    for j in range(2):
        assert torch.allclose(torch.cat(batched[j]), singles[j], rtol=1e-4, atol=1e-4)


@torch.inference_mode()
def test_hidden_token_misalignment_is_detected(models):
    ours, _, _ = models
    gen = torch.Generator().manual_seed(3)
    ids, hidden = torch.randint(0, 150000, (20,), generator=gen), torch.randn(21, 4096, generator=gen)
    good = ours.condition(ids, hidden[:20])
    shifted = ours.condition(ids, hidden[1:])  # hidden states one position late relative to the reply tokens
    assert ((shifted - good).norm() / good.norm()).item() > 1e-2
