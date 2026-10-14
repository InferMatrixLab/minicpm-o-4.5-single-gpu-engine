"""Builders for synthetic SSE streams in each engine's wire format."""

from __future__ import annotations

import base64
import json

import numpy as np

from bench.audio import SAMPLE_RATE, encode_wav_pcm16


def tone(n: int, freq: float = 440.0, amp: float = 0.3, phase: float = 0.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(n, dtype=np.float64) / sr
    return (amp * np.sin(2 * np.pi * freq * t + phase)).astype(np.float32)


def wav_b64(samples: np.ndarray, sr: int = SAMPLE_RATE) -> str:
    return base64.b64encode(encode_wav_pcm16(samples, sr)).decode()


def header_only_wav_b64(sr: int = SAMPLE_RATE) -> str:
    return wav_b64(np.zeros(0, dtype=np.float32), sr)


def sse(obj) -> bytes:
    if obj == "[DONE]":
        return b"data: [DONE]\n\n"
    return b"data: " + json.dumps(obj).encode() + b"\n\n"


# vllm-omni ---------------------------------------------------------------
def vllm_text(text: str) -> bytes:
    return sse({"modality": "text", "choices": [{"index": 0, "delta": {"content": text}}]})


def vllm_audio(b64: str | None) -> bytes:
    return sse({"modality": "audio", "choices": [{"index": 0, "delta": {"content": b64}}]})


# new engine ---------------------------------------------------------------
def engine_text(text: str) -> bytes:
    return sse({"modality": "text", "choices": [{"index": 0, "delta": {"content": text}}]})


def engine_audio(samples: np.ndarray, offset: int, num: int | None = None) -> bytes:
    return sse(
        {
            "modality": "audio",
            "audio_offset_samples": offset,
            "num_samples": len(samples) if num is None else num,
            "choices": [{"index": 0, "delta": {"content": wav_b64(samples)}}],
        }
    )


# sglang-omni ---------------------------------------------------------------
def sglang_body(samples: np.ndarray, text: str = "hello") -> bytes:
    return json.dumps(
        {"choices": [{"index": 0, "message": {"role": "assistant", "content": text, "audio": {"data": wav_b64(samples)}}}], "usage": {"prompt_tokens": 10}}
    ).encode()
