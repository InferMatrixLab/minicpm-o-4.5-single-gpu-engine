"""HF reference for Thinker parity on 64 prompts (run in .venv-hfref on a GPU, bf16, eager attention).

    CUDA_VISIBLE_DEVICES=G .venv-hfref/bin/python tools/hfref/thinker_reference.py OUT.pt

For each prompt: the HF greedy continuation (64 new tokens, the model's terminators), then one teacher-forced pass
over prompt + continuation returning the last-layer hidden states, the top-1 token and the top-1/top-2 logit margin
at every position.  The engine test replays the same token sequences.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

MODEL = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"
GENERAL = ["What is the capital of France?", "Explain photosynthesis in two sentences.", "Write a haiku about rain.",
           "List three prime numbers greater than 100.", "Translate 'good morning' into German.", "Who wrote Pride and Prejudice?",
           "Give me a one-line summary of the theory of relativity.", "What is 17 times 23?", "Name a famous painting by Van Gogh.",
           "Describe a cat in one sentence.", "What does CPU stand for?", "Suggest a name for a coffee shop.",
           "How many continents are there?", "What is the boiling point of water in Fahrenheit?", "Tell me a fun fact about octopuses.",
           "What is the opposite of 'ancient'?"]


def prompts():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets

    rows = load_seed_tts_rows()
    tts = canonical_requests(rows, make_id_sets(rows).tuning(1)[:48])
    return [(r.messages(), True) for r in tts] + [([{"role": "user", "content": g}], False) for g in GENERAL]


def main():
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, init_tts=False, init_vision=False, init_audio=False,
                                  attn_implementation="eager").cuda().eval()
    llm = m.llm
    terms = [tok.convert_tokens_to_ids(t) for t in m.terminators]
    out = []
    for msgs, tts in prompts():
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, use_tts_template=tts, enable_thinking=False)
        ids = torch.tensor([tok(text)["input_ids"]], device="cuda")
        with torch.inference_mode():
            g = llm.generate(input_ids=ids, max_new_tokens=64, do_sample=False, eos_token_id=terms, pad_token_id=0)
            full = g[0]
            o = llm(input_ids=full[None], output_hidden_states=True)
            logits = o.logits[0].float()
            top2 = logits.topk(2, dim=-1)
        out.append({"prompt_len": ids.shape[1], "tokens": full.cpu(), "hidden": o.hidden_states[-1][0].cpu(), "top1": top2.indices[:, 0].cpu(),
                    "margin": (top2.values[:, 0] - top2.values[:, 1]).cpu(), "tts": tts})
    torch.save(out, sys.argv[1])
    print(f"wrote {len(out)} prompts to {sys.argv[1]}")


if __name__ == "__main__":
    main()
