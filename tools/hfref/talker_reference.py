"""HF reference for Talker conditioning and teacher-forced codec logits (run in .venv-hfref on a GPU, bf16).

    CUDA_VISIBLE_DEVICES=G .venv-hfref/bin/python tools/hfref/talker_reference.py OUT.pt [N]

For N canonical TTS prompts, ``MiniCPMO.chat(use_tts_template, generate_audio)`` runs with greedy Thinker decoding
(max 256 tokens) and the canonical Talker sampling (TTSSamplingParams defaults: T=0.8, top-p 0.85, top-k 25,
penalty 1.05 over 16 frames; min 50 / max 2048 frames).  Recorded per prompt: the Thinker reply tokens between
<|tts_bos|> and <|tts_eos|>, the Talker condition ``inputs_embeds`` that ``_generate_speech_non_streaming`` built,
the codes HF sampled and the codec logits at every frame (hooked on ``head_code``).
"""

import sys
import tempfile
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"


def main():
    from transformers import AutoModel

    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets

    n = int(sys.argv[2]) if len(sys.argv) > 2 else 32
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    m.init_tts()
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[48:48 + n])
    logits_log, codes_log = [], []
    hook = m.tts.head_code[0].register_forward_hook(lambda mod, inp, out: logits_log.append(out[:, -1].float().cpu()))
    orig = m.tts.generate

    def gen(*a, **k):
        o = orig(*a, **k)
        codes_log.append(o.new_ids.squeeze(-1)[0].cpu())
        return o

    m.tts.generate = gen
    bounds = []
    orig_speech = m._generate_speech_non_streaming

    def speech(outputs, tts_bound, *a, **k):
        seq = outputs["full_sequences"][0]
        bounds.append(seq[tts_bound[0]:tts_bound[1]].cpu())  # the reply tokens the condition is built from
        return orig_speech(outputs, tts_bound, *a, **k)

    m._generate_speech_non_streaming = speech
    out = []
    for i, r in enumerate(reqs):
        logits_log.clear()
        codes_log.clear()
        bounds.clear()
        torch.manual_seed(1000 + i)
        with tempfile.TemporaryDirectory() as td:
            emb_path = Path(td) / "emb.pt"
            text = m.chat(msgs=r.messages(), do_sample=False, max_new_tokens=256, use_tts_template=True, enable_thinking=False, generate_audio=True,
                          output_audio_path=str(Path(td) / "a.wav"), output_tts_inputs_embeds_path=str(emb_path))
            cond = torch.load(emb_path).float().cpu()[0]
        out.append({"row_id": r.request_id, "text": text, "condition": cond, "reply_ids": bounds[0] if bounds else None,
                    "prompt": r.messages(), "codes": codes_log[0] if codes_log else None,
                    "logits": torch.cat(logits_log) if logits_log else None})
        print(i, r.request_id, cond.shape, None if not codes_log else codes_log[0].shape, flush=True)
    hook.remove()
    torch.save(out, sys.argv[1])
    print(f"wrote {len(out)} references to {sys.argv[1]}")


if __name__ == "__main__":
    main()
