"""Is the engine Talker's 96.7% teacher-forced top-1 a port difference or bf16 kernel noise?  (diagnostic, GPU)

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/hfref/talker_llama_probe.py [N]

Replays HF's conditions and sampled codes (results/engine_parity/talker_ref_bf16.pt) through (a) the engine Talker,
(b) transformers' own LlamaModel built from the tts.* weights the way MiniCPMTTS builds it (bf16, sdpa and eager), and
(c) the same LlamaModel in fp32; reports top-1 agreement with the HF chat-run logits and between the variants.
"""

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


def hf_llama(dtype, attn):
    from transformers import LlamaConfig, LlamaModel

    from engine.models.talker import TalkerConfig, load_prefixed

    c = TalkerConfig.from_model_dir(MODEL)
    cfg = LlamaConfig(hidden_size=c.hidden_size, intermediate_size=c.intermediate_size, num_attention_heads=c.num_heads,
                      num_hidden_layers=c.num_layers, num_key_value_heads=c.num_kv_heads, max_position_embeddings=c.max_positions,
                      attn_implementation=attn)
    m = LlamaModel(cfg)
    w = load_prefixed(MODEL, "tts.model.")
    missing, unexpected = m.load_state_dict(w, strict=False)
    assert not [k for k in missing if "rotary" not in k], missing
    m = m.to("cuda", dtype).eval()
    tw = load_prefixed(MODEL, "tts.", "cuda", dtype)
    head = torch._weight_norm(tw["head_code.0.parametrizations.weight.original1"], tw["head_code.0.parametrizations.weight.original0"], 0)
    return m, tw["emb_code.0.weight"], head


@torch.inference_mode()
def replay_hf(m, emb, head, cond, codes, dtype):
    o = m(inputs_embeds=cond.to(dtype)[None], position_ids=torch.arange(cond.shape[0], device="cuda")[None], use_cache=True)
    pkv, lg = o.past_key_values, [torch.nn.functional.linear(o.last_hidden_state[:, -1], head).float()]
    for i, c in enumerate(codes[:-1]):
        o = m(inputs_embeds=torch.nn.functional.embedding(c.view(1, 1).cuda(), emb), position_ids=torch.tensor([[cond.shape[0] + i]], device="cuda"),
              past_key_values=pkv, use_cache=True)
        pkv = o.past_key_values
        lg.append(torch.nn.functional.linear(o.last_hidden_state[:, -1], head).float())
    return torch.cat(lg).cpu()


@torch.inference_mode()
def replay_engine(ta, cond, codes):
    row = torch.tensor([0], device="cuda")
    cache = ta.new_cache(1, cond.shape[0] + codes.numel() + 4)
    lg = [ta.forward(cond.cuda().to(ta.dtype)[None], cache, row)]
    for c in codes[:-1]:
        lg.append(ta.forward(ta.code_embed(c.view(1, 1).cuda()), cache, row))
    return torch.cat(lg).cpu()


def main():
    from engine.models.talker import Talker

    ref = [r for r in torch.load("results/engine_parity/talker_ref_bf16.pt") if r["codes"] is not None]
    ref = ref[: int(sys.argv[1])] if len(sys.argv) > 1 else ref
    runs = {}
    ta = Talker.load(MODEL, device="cuda")
    runs["engine_bf16"] = [replay_engine(ta, r["condition"], r["codes"]) for r in ref]
    del ta
    for name, dtype, attn in (("hf_llama_bf16_sdpa", torch.bfloat16, "sdpa"), ("hf_llama_bf16_eager", torch.bfloat16, "eager"),
                              ("hf_llama_fp32_sdpa", torch.float32, "sdpa")):
        m, emb, head = hf_llama(dtype, attn)
        runs[name] = [replay_hf(m, emb, head, r["condition"].cuda(), r["codes"], dtype) for r in ref]
        del m
    want = [r["logits"] for r in ref]

    def agree(a, b):
        hit = tot = 0
        for x, y in zip(a, b):
            n = min(x.shape[0], y.shape[0])
            hit += int((x[:n].argmax(-1) == y[:n].argmax(-1)).sum())
            tot += n
        return hit / tot

    out = {f"{k}_vs_hf_chat": agree(v, want) for k, v in runs.items()}
    out.update({f"engine_vs_{k}": agree(runs["engine_bf16"], v) for k, v in runs.items() if k != "engine_bf16"})
    out["hf_llama_bf16_sdpa_vs_eager"] = agree(runs["hf_llama_bf16_sdpa"], runs["hf_llama_bf16_eager"])
    print(json.dumps(out, indent=1))
    Path("results/engine_parity/talker_llama_probe.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
