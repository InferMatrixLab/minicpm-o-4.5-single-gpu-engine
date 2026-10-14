"""Dump effective inputs for Seed-TTS rows through each stack's own preprocessing.

Run under the Python environment of the stack being dumped:

    <vllm-omni venv>/python -m bench.effective_dump canonical   --out ... [--n 32]
    <vllm-omni venv>/python -m bench.effective_dump vllm-omni   --out ... --url http://127.0.0.1:PORT
    <sglang venv>/python    -m bench.effective_dump sglang-omni --out ...

* ``canonical`` follows the HF reference (``modeling_minicpmo.py``): chat
  template with ``use_tts_template``, ``librosa.load(sr=16000)`` of the
  reference, written as 16 kHz PCM16 and passed to ``Token2wav._prepare_prompt``.
* ``vllm-omni`` takes prompt IDs from a live server (``return_token_ids``) and
  replays its reference path: vLLM ``MediaConnector`` decode, Code2Wav
  ``_normalize_reference`` (24 kHz, fixed window), PCM16 write, ``_prepare_prompt``.
* ``sglang-omni`` uses its preprocessor's chat rendering and its Code2Wav
  ``Token2Wav.prepare_prompt`` on the raw reference bytes.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.effective_inputs import EffectiveInput, array_hash, save_dumps  # noqa: E402
from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets  # noqa: E402

MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"
)
T2W_DIR = os.path.join(MODEL_PATH, "assets", "token2wav")


def _rows(n: int):
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    ids = list(sets.tuning_pool[:n])
    return canonical_requests(rows, ids, stream=False)


def _tolist(x) -> list:
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().float().cpu().reshape(-1).tolist() if x.is_floating_point() else x.detach().cpu().reshape(-1).tolist()
    return list(np.asarray(x).reshape(-1).tolist())


def _canonical_prompt_ids(tokenizer, req) -> list[int]:
    text = tokenizer.apply_chat_template(
        req.messages(), tokenize=False, add_generation_prompt=True, use_tts_template=True, enable_thinking=False
    )
    return list(tokenizer.encode(text))


def dump_canonical(n: int) -> list[EffectiveInput]:
    import librosa
    import soundfile as sf
    from token2wav import Token2wav
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    t2w = Token2wav(T2W_DIR)
    out = []
    for req in _rows(n):
        wav16, _ = librosa.load(req.row.ref_wav_path, sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav16, 16000)
            tokens, _, spk, mels, _ = t2w._prepare_prompt(f.name)
        out.append(
            EffectiveInput(
                row_id=req.request_id,
                prompt_token_ids=_canonical_prompt_ids(tok, req),
                ref_source_sha256=hashlib.sha256(req.ref_audio_bytes()).hexdigest(),
                ref_waveform_hash=array_hash(wav16.astype(np.float32)),
                ref_waveform_sample_rate=16000,
                speech_tokens=_tolist(tokens),
                speaker_embedding=_tolist(spk),
                extra={"prompt_mel_frames": int(mels.shape[1]), "ref_samples_16k": int(wav16.shape[0])},
            )
        )
    return out


def dump_vllm_omni(n: int, url: str | None) -> list[EffectiveInput]:
    import asyncio
    import urllib.request

    import soundfile as sf
    from token2wav import Token2wav
    from vllm.multimodal.media import MediaConnector

    from bench.adapters import VllmOmniAdapter
    from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_code2wav import _REF_MAX_SECONDS, _normalize_reference

    t2w = Token2wav(T2W_DIR)
    adapter = VllmOmniAdapter()
    conn = MediaConnector()
    out = []
    for req in _rows(n):
        ids = None
        if url:
            p = adapter.build_payload(req)
            p.update({"return_token_ids": True, "modalities": ["text"], "max_tokens": 1, "stream": False})
            p.pop("stream_options", None)
            r = urllib.request.urlopen(
                urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=json.dumps(p).encode(), headers={"Content-Type": "application/json"}),
                timeout=120,
            )
            ids = json.loads(r.read())["prompt_token_ids"]
        data_url = req.ref_audio_data_url()
        sent = base64.b64decode(data_url.split(",", 1)[1])
        wave, sr = asyncio.run(conn.fetch_audio_async(data_url))
        norm, nsr = _normalize_reference(wave, int(sr), max_seconds=_REF_MAX_SECONDS)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, norm.numpy(), nsr, format="WAV")
            tokens, _, spk, mels, _ = t2w._prepare_prompt(f.name)
        out.append(
            EffectiveInput(
                row_id=req.request_id,
                prompt_token_ids=list(ids or []),
                ref_source_sha256=hashlib.sha256(sent).hexdigest(),
                ref_waveform_hash=array_hash(norm.numpy().astype(np.float32)),
                ref_waveform_sample_rate=int(nsr),
                speech_tokens=_tolist(tokens),
                speaker_embedding=_tolist(spk),
                extra={"prompt_mel_frames": int(mels.shape[1]), "decoded_sr": int(sr), "ref_window_s": _REF_MAX_SECONDS},
            )
        )
    return out


def dump_sglang_omni(n: int) -> list[EffectiveInput]:
    import io

    import torch
    import torchaudio
    from transformers import AutoTokenizer

    from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav

    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    t2w = Token2Wav(Path(T2W_DIR), device=torch.device("cuda"), dtype=torch.float16)
    out = []
    for req in _rows(n):
        # sglang-omni's preprocessor renders the chat with use_tts_template when speech is enabled
        text = tok.apply_chat_template(req.messages(), add_generation_prompt=True, tokenize=False, use_tts_template=True, enable_thinking=False)
        ids = tok(text, return_tensors="pt")["input_ids"][0].tolist()
        raw = Path(req.row.ref_wav_path).read_bytes()
        prompt = t2w.prepare_prompt(io.BytesIO(raw))
        audio, sr = torchaudio.load(io.BytesIO(raw))
        speech16 = (torchaudio.transforms.Resample(sr, 16000)(audio) if sr != 16000 else audio)[0]
        fields = {k: getattr(prompt, k) for k in dir(prompt) if not k.startswith("_")}
        tokens = fields.get("prompt_tokens", fields.get("tokens"))
        spk = fields.get("speaker_embedding", fields.get("embedding"))
        out.append(
            EffectiveInput(
                row_id=req.request_id,
                prompt_token_ids=ids,
                ref_source_sha256=hashlib.sha256(raw).hexdigest(),
                ref_waveform_hash=array_hash(speech16.numpy().astype(np.float32)),
                ref_waveform_sample_rate=16000,
                speech_tokens=_tolist(tokens) if tokens is not None else None,
                speaker_embedding=_tolist(spk) if spk is not None else None,
                extra={"prompt_fields": sorted(fields)},
            )
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stack", choices=["canonical", "vllm-omni", "sglang-omni"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--url", default=None)
    a = ap.parse_args()
    if a.stack == "canonical":
        dumps = dump_canonical(a.n)
    elif a.stack == "vllm-omni":
        dumps = dump_vllm_omni(a.n, a.url)
    else:
        dumps = dump_sglang_omni(a.n)
    save_dumps(Path(a.out), dumps)
    print(f"wrote {len(dumps)} rows to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
