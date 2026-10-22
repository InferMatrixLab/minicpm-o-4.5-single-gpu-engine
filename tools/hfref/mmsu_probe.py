"""HF greedy answers for MMSU samples (run in .venv-hfref on a GPU: the model card's transformers 4.51 stack; diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-hfref/bin/python tools/hfref/mmsu_probe.py SAMPLES.json OUT.json [bfloat16|float32] [librosa|linear|torchaudio]

``SAMPLES.json``: ``[{"id", "audio_path", "text"}]`` as the MMSU worker sends them (audio part, then the question text).
The prompt is built as ``MiniCPMO.chat`` builds it (16 kHz audio, ``"\\n".join`` of the parts, the TTS template,
thinking off); the reply is the LLM's greedy continuation of the HF multimodal embeddings, 32 new tokens (the suite's
MMSU limit), the same request the served engines answer.  The audio loader: ``librosa`` (``librosa.load(sr=16000)``, the model card's
and the engine's), ``linear`` (SGLang-Omni's ``load_audio_path``: libsndfile decode, channel mean, ``np.interp`` linear
resampling) or ``torchaudio`` (vLLM's default: libsndfile decode, channel mean, ``torchaudio.transforms.Resample``).
"""

import json
import os
import sys

import torch

MODEL = os.environ.get("MODEL_PATH") or os.path.join(os.environ.get("HF_HUB_CACHE") or os.path.expanduser("~/.cache/huggingface/hub"),
                                                     "models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


def load(path: str, how: str):
    import librosa
    import numpy as np
    import soundfile as sf

    if how == "librosa":
        return librosa.load(path, sr=16000, mono=True)[0]
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr == 16000:
        return audio.astype(np.float32)
    if how == "linear":  # sglang_omni.preprocessing.audio.resample_linear
        new_len = max(int(round(audio.shape[0] / float(sr) * 16000)), 1)
        return np.interp(np.linspace(0.0, audio.shape[0] - 1, num=new_len), np.arange(audio.shape[0], dtype=np.float64), audio).astype(np.float32)
    import torchaudio

    return torchaudio.transforms.Resample(sr, 16000)(torch.as_tensor(audio)).numpy()


@torch.inference_mode()
def main():
    from transformers import AutoModel

    samples = json.load(open(sys.argv[1]))
    dtype = getattr(torch, sys.argv[3] if len(sys.argv) > 3 else "bfloat16")
    how = sys.argv[4] if len(sys.argv) > 4 else "librosa"
    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=dtype, init_tts=False, attn_implementation="sdpa").cuda().eval()
    m.prepare_processor()
    tok = m.processor.tokenizer
    terms = [tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    out = []
    for s in samples:
        wav = load(s["audio_path"], how)
        prompt = tok.apply_chat_template([{"role": "user", "content": "<audio>./</audio>\n" + s["text"]}], tokenize=False, add_generation_prompt=True,
                                         use_tts_template=True, enable_thinking=False)
        inp = m.processor([prompt], [[]], [[wav]], [[0]], max_slice_nums=None, use_image_id=None, stream_input=False, return_tensors="pt",
                          max_length=8192).to("cuda")
        data = {k: inp.get(k) for k in ("input_ids", "audio_features", "audio_feature_lens", "image_bound", "audio_bounds", "spk_bounds", "pixel_values",
                                        "tgt_sizes")}
        e, _ = m.get_vllm_embedding(data)
        e = m.get_omni_embedding(data, input_embeddings=e, chunk_length=m.config.audio_chunk_length)
        g = m.llm.generate(inputs_embeds=e, attention_mask=torch.ones(e.shape[:2], dtype=torch.long, device="cuda"), max_new_tokens=32, do_sample=False,
                           eos_token_id=terms, pad_token_id=0)
        ids = g[0].tolist()
        out.append({"id": s["id"], "prompt_ids": inp["input_ids"][0].tolist(), "reply_ids": ids,
                    "text": tok.decode([t for t in ids if t not in terms], skip_special_tokens=True)})
    json.dump(out, open(sys.argv[2], "w"))
    print(f"wrote {len(out)} HF replies to {sys.argv[2]}")


if __name__ == "__main__":
    main()
