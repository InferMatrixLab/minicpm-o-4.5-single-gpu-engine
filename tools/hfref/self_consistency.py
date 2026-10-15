"""How close is HF bf16 to itself?  Eager vs SDPA attention and vs fp32, same GPU, same tokens (diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-hfref/bin/python tools/hfref/self_consistency.py results/engine_parity/thinker_ref_bf16.pt [N]
"""

import json
import sys

import torch
import torch.nn.functional as F

MODEL = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"


@torch.inference_mode()
def main():
    from transformers import AutoModel

    ref = torch.load(sys.argv[1])[: int(sys.argv[2]) if len(sys.argv) > 2 else 8]
    res = {}
    for name, dtype, attn in (("bf16_sdpa", torch.bfloat16, "sdpa"), ("fp32_eager", torch.float32, "eager")):
        m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=dtype, init_tts=False, init_vision=False, init_audio=False,
                                      attn_implementation=attn).cuda().eval()
        cos = []
        for r in ref:
            h = m.llm(input_ids=r["tokens"].cuda()[None], output_hidden_states=True).hidden_states[-1][0]
            cos.append(F.cosine_similarity(h.float(), r["hidden"].cuda().float(), dim=-1).cpu())
        c = torch.cat(cos)
        res[name] = {"vs": "bf16_eager", "min": float(c.min()), "p01": float(c.quantile(0.01)), "mean": float(c.mean())}
        print(name, json.dumps(res[name]), flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(res, open("results/engine_parity/hf_self_consistency.json", "w"), indent=1)


if __name__ == "__main__":
    main()
