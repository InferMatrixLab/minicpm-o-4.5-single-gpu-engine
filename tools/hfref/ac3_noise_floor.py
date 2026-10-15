"""HF's own bf16 noise floor for the AC-3.1/3.2 metrics (run in .venv-hfref on a GPU; diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-hfref/bin/python tools/hfref/ac3_noise_floor.py OUT.json

Thinker (64 reference prompts, tokens from thinker_ref_bf16.pt, reference = bf16 eager): bf16 SDPA and fp32 eager
runs of the same HF code -> per-token hidden cosine (min, p01, mean), top-1 agreement, greedy identity.
Talker (talker_ref_bf16.pt, HF condition + HF sampled codes, reference = HF chat run, SDPA): eager attention and fp32
runs of the same HF Talker -> teacher-forced top-1 agreement.
"""

import json
import sys

import torch
import torch.nn.functional as F

MODEL = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"
TREF = "results/engine_parity/thinker_ref_bf16.pt"
AREF = "results/engine_parity/talker_ref_bf16.pt"


def stats(c):
    c = torch.cat(c)
    return {"min": float(c.min()), "p01": float(c.quantile(0.01)), "mean": float(c.mean())}


@torch.inference_mode()
def main():
    from transformers import AutoModel

    tref, aref = torch.load(TREF), [r for r in torch.load(AREF) if r["codes"] is not None]
    out = {}
    for name, dtype, attn in (("bf16_eager", torch.bfloat16, "eager"), ("bf16_sdpa", torch.bfloat16, "sdpa"), ("fp32_eager", torch.float32, "eager")):
        m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=dtype, init_vision=False, init_audio=False,
                                      attn_implementation=attn).cuda().eval()
        cos, agree, total, identical = [], 0, 0, 0
        for r in tref:
            t = r["tokens"].cuda()
            o = m.llm(input_ids=t[None], output_hidden_states=True)
            cos.append(F.cosine_similarity(o.hidden_states[-1][0].float(), r["hidden"].cuda().float(), dim=-1).cpu())
            agree += int((o.logits[0].argmax(-1).cpu() == r["top1"]).sum())
            total += t.numel()
            p = r["prompt_len"]
            g = m.llm.generate(input_ids=t[None, :p], max_new_tokens=t.numel() - p, do_sample=False, eos_token_id=[151645, 151643], pad_token_id=0)
            identical += int(torch.equal(g[0].cpu(), r["tokens"]))
        tal_agree = tal_total = 0
        tts = m.tts
        for r in aref:
            cond = r["condition"].cuda().to(dtype)[None]
            o = tts.model(inputs_embeds=cond, position_ids=torch.arange(cond.shape[1], device="cuda")[None], use_cache=True)
            pkv, lg = o.past_key_values, [tts.head_code[0](o.last_hidden_state[:, -1]).float()]
            for i, c in enumerate(r["codes"][:-1]):
                e = tts.emb_code[0](c.view(1, 1).cuda())
                o = tts.model(inputs_embeds=e, position_ids=torch.tensor([[cond.shape[1] + i]], device="cuda"), past_key_values=pkv, use_cache=True)
                pkv = o.past_key_values
                lg.append(tts.head_code[0](o.last_hidden_state[:, -1]).float())
            lg = torch.cat(lg).cpu()
            n = min(lg.shape[0], r["logits"].shape[0])
            tal_agree += int((lg[:n].argmax(-1) == r["logits"][:n].argmax(-1)).sum())
            tal_total += n
        out[name] = {"thinker_hidden_cos": stats(cos), "thinker_top1_agreement": agree / total, "thinker_greedy_identical": identical / len(tref),
                     "talker_top1_agreement": tal_agree / tal_total}
        print(name, json.dumps(out[name]), flush=True)
        del m
        torch.cuda.empty_cache()
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
