"""HF reference for the multimodal parity samples (run in .venv-hfref: the model card's transformers 4.51 stack).

    .venv-hfref/bin/python tools/hfref/mm_reference.py SAMPLES.pkl OUT.pt [cpu|cuda] [float32|bfloat16]
"""

import pickle
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.testing.mm_samples import MODEL, to_msgs  # noqa: E402


def main():
    samples = pickle.loads(Path(sys.argv[1]).read_bytes())
    device = sys.argv[3] if len(sys.argv) > 3 else "cpu"
    dtype = getattr(torch, sys.argv[4] if len(sys.argv) > 4 else "float32")
    from transformers import AutoModel

    m = AutoModel.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=dtype, init_tts=False, attn_implementation="sdpa").to(device).eval()
    m.prepare_processor()
    out = []
    for s in samples:
        msgs = to_msgs(s)
        from PIL import Image
        import numpy as np

        images, audios, parts = [], [], []
        cur = []
        use_tts = s["tts"]
        for c in msgs[0]["content"]:
            if isinstance(c, Image.Image):
                images.append(c)
                cur.append("<image>./</image>")
            elif isinstance(c, np.ndarray):
                audios.append(c)
                parts.append(0)
                cur.append("<audio>./</audio>")
                use_tts = True
            else:
                cur.append(c)
        prompt = m.processor.tokenizer.apply_chat_template([{"role": "user", "content": "\n".join(cur)}], tokenize=False, add_generation_prompt=True,
                                                           use_tts_template=use_tts, enable_thinking=False)
        inp = m.processor([prompt], [images], [audios], [parts], max_slice_nums=None, use_image_id=None, stream_input=False, return_tensors="pt",
                          max_length=8192).to(device)
        data = {k: inp.get(k) for k in ("input_ids", "audio_features", "audio_feature_lens", "image_bound", "audio_bounds", "spk_bounds", "pixel_values",
                                        "tgt_sizes")}
        with torch.inference_mode():
            e, _ = m.get_vllm_embedding(data)
            e = m.get_omni_embedding(data, input_embeddings=e, chunk_length=m.config.audio_chunk_length)[0]
        out.append({"input_ids": inp["input_ids"][0].cpu(), "embeds": e.float().cpu()})
    torch.save(out, sys.argv[2])
    print(f"wrote {len(out)} references to {sys.argv[2]}")


if __name__ == "__main__":
    main()
