"""Multimodal front end: HF-identical prompt assembly and input encoders for the Thinker.

The checkpoint ships its processor (``processing_minicpmo.py``: image slicing,
audio segmentation, placeholder expansion) and its encoders (SigLIP NaViT +
resampler, Whisper encoder + average pooling + projection) under Apache-2.0.
The engine uses that code as-is, built without the language model and the
Talker, so the token IDs and the embeddings scattered into the placeholder
positions are the reference's by construction:

* ``build_prompt`` follows ``MiniCPMO.chat`` (``modeling_minicpmo.py:1085-1200``):
  PIL images become ``<image>./</image>``, audio arrays ``<audio>./</audio>``
  (which also selects the TTS template), parts are joined with newlines, the
  tokenizer's chat template is applied, and the processor expands placeholders;
* ``embed`` follows ``MiniCPMO.generate``: ``get_vllm_embedding`` (token
  embeddings with the vision states scattered into ``image_bound``) then
  ``get_omni_embedding`` (audio states into ``audio_bounds``).

The resulting ``[T, 4096]`` embeddings feed the engine Thinker (``forward`` takes
embeddings).  ``tests/engine/test_multimodal_frontend.py`` checks the IDs and
the encoder outputs against the full HF model on the same GPU.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from engine.models.talker import load_prefixed

SUBMODULES = ("vpm", "resampler", "apm", "audio_projection_layer")


def _whisper_layer_forward(self, hidden_states, attention_mask, layer_head_mask=None, output_attentions=False, past_key_values=None,
                           use_cache=False):
    """``MiniCPMWhisperEncoderLayer.forward`` for transformers 5.x, where ``WhisperAttention`` returns (output, weights).

    Same computation as the checkpoint code (pre-LN attention and MLP residual blocks, fp16 clamp); only the attention
    call is adapted.  ``layer_head_mask`` is always None on the inference path.
    """
    residual = hidden_states
    hidden_states = self.self_attn_layer_norm(hidden_states)
    hidden_states, attn_weights = self.self_attn(hidden_states=hidden_states, attention_mask=attention_mask, output_attentions=output_attentions)[:2]
    hidden_states = residual + hidden_states
    residual = hidden_states
    hidden_states = self.final_layer_norm(hidden_states)
    hidden_states = self.fc2(self.activation_fn(self.fc1(hidden_states)))
    hidden_states = residual + hidden_states
    if hidden_states.dtype == torch.float16 and (torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()):
        clamp_value = torch.finfo(hidden_states.dtype).max - 1000
        hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)
    outputs = (hidden_states,)
    if output_attentions:
        outputs += (attn_weights,)
    if use_cache:
        outputs += (past_key_values,)
    return outputs


class MultimodalFrontend:
    def __init__(self, model_dir: str | Path, device="cuda", dtype=torch.bfloat16):
        from transformers import AutoConfig, AutoProcessor
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_dir = Path(model_dir)
        cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        cls = get_class_from_dynamic_module("modeling_minicpmo.MiniCPMO", str(model_dir))
        import transformers
        from importlib import import_module

        mod = import_module(cls.__module__)
        if int(transformers.__version__.split(".")[0]) >= 5:
            mod.MiniCPMWhisperEncoderLayer.forward = _whisper_layer_forward
        m = cls.__new__(cls)
        torch.nn.Module.__init__(m)
        m.config = cfg
        with torch.device(device):
            m.vpm = m.init_vision_module()
            m.vision_dim = m.vpm.embed_dim
            m.embed_dim = cfg.hidden_size
            m.resampler = m.init_resampler(m.embed_dim, m.vision_dim)
            m.apm = m.init_audio_module()
            m.audio_avg_pooler = torch.nn.AvgPool1d(cfg.audio_pool_step, stride=cfg.audio_pool_step)
            m.audio_projection_layer = mod.MultiModalProjector(in_dim=int(m.apm.config.encoder_ffn_dim // 4), out_dim=m.embed_dim)
            m.audio_encoder_layer = -1
            embed = torch.nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        w = load_prefixed(model_dir, "", device, None)
        for name in SUBMODULES:
            sub = {k[len(name) + 1:]: v for k, v in w.items() if k.startswith(name + ".")}
            getattr(m, name).load_state_dict(sub, strict=True)
        embed.weight.data = w["llm.model.embed_tokens.weight"]
        object.__setattr__(m, "llm", SimpleNamespace(model=SimpleNamespace(embed_tokens=embed), config=SimpleNamespace(hidden_size=cfg.hidden_size)))
        m.to(dtype).eval()
        embed.to(device=device, dtype=dtype)
        self.m, self.cfg, self.device, self.dtype = m, cfg, device, dtype
        self.processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

    def build_prompt(self, msgs: list[dict], use_tts_template: bool = False, enable_thinking: bool = False, max_slice_nums=None,
                     use_image_id=None, max_inp_length: int = 8192):
        """``msgs`` content is a string or a list of str / PIL.Image / 16 kHz np.ndarray parts (the HF ``chat`` input form)."""
        from PIL import Image

        images, audios, audio_parts = [], [], []
        copy_msgs = deepcopy(msgs)
        for i, msg in enumerate(copy_msgs):
            content = msg["content"] if isinstance(msg["content"], list) else [msg["content"]]
            cur = []
            for c in content:
                if isinstance(c, Image.Image):
                    images.append(c)
                    cur.append("<image>./</image>")
                elif isinstance(c, np.ndarray):
                    audios.append(c)
                    audio_parts.append(i)
                    cur.append("<audio>./</audio>")
                    use_tts_template = True
                elif isinstance(c, str):
                    cur.append(c)
            msg["content"] = "\n".join(cur)
        prompt = self.tokenizer.apply_chat_template(copy_msgs, tokenize=False, add_generation_prompt=True, use_tts_template=use_tts_template,
                                                    enable_thinking=enable_thinking)
        return self.processor([prompt], [images], [audios], [audio_parts], max_slice_nums=max_slice_nums, use_image_id=use_image_id,
                              stream_input=False, return_tensors="pt", max_length=max_inp_length).to(self.device)

    @torch.inference_mode()
    def embed(self, inputs) -> torch.Tensor:
        data = {"input_ids": inputs["input_ids"], "audio_features": inputs.get("audio_features"), "audio_feature_lens": inputs.get("audio_feature_lens"),
                "image_bound": inputs.get("image_bound"), "audio_bounds": inputs.get("audio_bounds"), "spk_bounds": inputs.get("spk_bounds"),
                "pixel_values": inputs.get("pixel_values"), "tgt_sizes": inputs.get("tgt_sizes")}
        emb, _ = self.m.get_vllm_embedding(data)
        emb = self.m.get_omni_embedding(data, input_embeddings=emb, chunk_length=self.cfg.audio_chunk_length)
        return emb[0]
