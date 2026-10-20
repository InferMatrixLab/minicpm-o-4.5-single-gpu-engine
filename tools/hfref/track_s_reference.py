"""HF reference for Track S (AC-3.5): per-chunk Talker conditions and teacher-forced codec logits of streaming_generate.

    CUDA_VISIBLE_DEVICES=G .venv-hfref/bin/python tools/hfref/track_s_reference.py OUT.pt [N]

For N canonical TTS prompts, the model card's streaming flow runs in bf16 (sdpa): reset_session,
init_token2wav_cache(reference voice at 16 kHz), streaming_prefill(user message, use_tts_template), then
streaming_generate with greedy Thinker decoding and the canonical Talker sampling (TTSSamplingParams defaults).
Recorded per prompt: the reply token IDs, every token ID the Thinker embedded (prompt, bos input and fed reply tokens,
in order), and for every Talker chunk (``TTSStreamingGenerator.generate_with_buffer``
call) the full condition it prefilled (text_eos / audio_bos included), whether the text was finished, the codec frames
it sampled (EOS excluded, as the reference keeps them) and the codec logits at every frame (hooked on ``head_code``).
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = str(MODEL_PATH)


def main():
    import librosa
    from transformers import AutoModel

    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets

    n = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    m.init_tts()
    utils = sys.modules[type(m).__module__.rsplit(".", 1)[0] + ".utils"]
    gen_cls = utils.TTSStreamingGenerator
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[64:64 + n])
    logits_log: list[torch.Tensor] = []
    chunks: list[dict] = []
    hook = m.tts.head_code[0].register_forward_hook(lambda mod, inp, out: logits_log.append(out[:, -1].float().cpu()))
    llm_ids: list[torch.Tensor] = []  # every token ID the Thinker embeds, in order (prompt, bos input, fed reply tokens)
    ids_hook = m.llm.get_input_embeddings().register_forward_pre_hook(lambda mod, args: llm_ids.append(args[0].reshape(-1).cpu()))
    orig = gen_cls.generate_with_buffer

    def wrapped(self, condition, text_finished=False, max_new_token=500):
        before_tokens = len(self.all_generated_tokens)
        before_logits = len(logits_log)
        yield from orig(self, condition, text_finished=text_finished, max_new_token=max_new_token)
        chunks.append({"condition": self.all_conditions[-1][0].float().cpu(), "text_finished": bool(text_finished),
                       "frames": [int(t.view(-1)[0]) for t in self.all_generated_tokens[before_tokens:]],
                       "logits": torch.cat(logits_log[before_logits:]) if len(logits_log) > before_logits else None})

    gen_cls.generate_with_buffer = wrapped
    out = []
    for i, r in enumerate(reqs):
        logits_log.clear()
        chunks.clear()
        llm_ids.clear()
        m.reset_session(reset_token2wav_cache=True)
        ref16, _ = librosa.load(r.row.ref_wav_path, sr=16000, mono=True)
        m.init_token2wav_cache(prompt_speech_16k=ref16)
        torch.manual_seed(2000 + i)
        sid = f"tracks-{i}"
        m.streaming_prefill(session_id=sid, msgs=r.messages(), omni_mode=False, is_last_chunk=True)
        text = ""
        for _, t in m.streaming_generate(session_id=sid, generate_audio=True, use_tts_template=True, enable_thinking=False, do_sample=False,
                                         max_new_tokens=256):
            text += t
        ids = getattr(m, "_streaming_generated_token_ids", None)
        out.append({"row_id": r.request_id, "prompt": r.messages(), "text": text,
                    "reply_ids": torch.as_tensor(ids).view(-1).cpu() if ids is not None else None,
                    "llm_input_ids": torch.cat(llm_ids) if llm_ids else None, "chunks": list(chunks)})
        print(i, r.request_id, len(chunks), [len(c["frames"]) for c in chunks], flush=True)
    hook.remove()
    ids_hook.remove()
    torch.save(out, sys.argv[1])
    print(f"wrote {len(out)} references to {sys.argv[1]}")


if __name__ == "__main__":
    main()
