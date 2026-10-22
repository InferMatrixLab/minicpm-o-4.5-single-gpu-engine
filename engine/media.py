"""OpenAI-style content parts -> the native multimodal form (PIL images, 16 kHz float32 audio, text).

``image_url`` (data URL, http(s) is refused, or a local path), ``input_audio`` ({"data": base64, "format"}),
``audio_url`` (data URL or path) and ``video_url`` (a local video: 1 fps frames each followed by that second of its
audio track, the frames-plus-audio form) are decoded here; malformed media raise ``BadRequest`` (HTTP 400).

Input audio is resampled to 16 kHz by the backend's ``audio_resampler``:

* ``librosa``: ``librosa.load(sr=16000)`` (the model card's loader; with this environment's librosa 1.0, ``soxr_hq``);
* ``torchaudio``: native-rate decode, channel mean, ``torchaudio.transforms.Resample`` (vLLM's default resampler);
* ``linear``: native-rate decode, channel mean, ``np.interp`` (SGLang-Omni's ``resample_linear``).

About half of MMSU's clips are 24-48 kHz, and its accuracy depends on this choice (docs/engine.md, round 17).  The
choice only concerns understanding inputs; the voice prompt (``ref_audio``) goes through Token2Wav's own front end.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np

from engine.server import BadRequest

MAX_VIDEO_FRAMES = 64
RESAMPLERS = ("librosa", "torchaudio", "linear")


def _bytes(url: str) -> bytes:
    if not isinstance(url, str) or not url:
        raise BadRequest("media part needs a URL")
    try:
        if url.startswith("data:"):
            return base64.b64decode(url.split(",", 1)[1], validate=True)
        path = url.removeprefix("file://")
        if path.startswith("/"):
            return Path(path).read_bytes()
    except (ValueError, IndexError, OSError) as e:
        raise BadRequest(f"media is not readable: {e}") from None
    raise BadRequest("media must be a data URL or a local file path")


def load_audio(src, resampler: str = "librosa") -> np.ndarray:
    """Mono float32 audio at 16 kHz from a path or a file object (any decoder failure propagates)."""
    import librosa

    if resampler == "librosa":
        return librosa.load(src, sr=16000, mono=True)[0].astype(np.float32)
    wav, sr = librosa.load(src, sr=None, mono=True)  # native rate: libsndfile (else audioread), channel mean
    wav = wav.astype(np.float32)
    if sr == 16000 or not len(wav):
        return wav
    if resampler == "linear":  # sglang_omni.preprocessing.audio.resample_linear
        n = max(int(round(wav.shape[0] / float(sr) * 16000)), 1)
        return np.interp(np.linspace(0.0, wav.shape[0] - 1, num=n), np.arange(wav.shape[0], dtype=np.float64), wav).astype(np.float32)
    if resampler == "torchaudio":  # vllm.multimodal.audio.resample_audio_torchaudio
        import torch
        import torchaudio

        return torchaudio.transforms.Resample(int(sr), 16000)(torch.as_tensor(wav)).numpy().astype(np.float32)
    raise ValueError(f"audio_resampler must be one of {RESAMPLERS}, not {resampler!r}")


def _audio(data: bytes, resampler: str = "librosa") -> np.ndarray:
    try:
        return load_audio(io.BytesIO(data), resampler)
    except Exception as e:  # any decoder failure is the client's input
        raise BadRequest(f"audio is not decodable: {e}") from None


def _image(data: bytes):
    from PIL import Image

    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise BadRequest(f"image is not decodable: {e}") from None


def _video(path: str, resampler: str = "librosa") -> list:
    import av
    from PIL import Image

    p = path.removeprefix("file://")
    if not p.startswith("/") or not Path(p).exists():
        raise BadRequest("video_url must name a local video file")
    try:
        frames, step = [], None
        with av.open(p) as c:
            s = c.streams.video[0]
            fps = float(s.average_rate or 25)
            step = max(1, int(round(fps)))
            for i, fr in enumerate(c.decode(video=0)):
                if i % step == 0:
                    frames.append(Image.fromarray(fr.to_ndarray(format="rgb24")))
                if len(frames) >= MAX_VIDEO_FRAMES:
                    break
        wav = load_audio(p, resampler)
    except Exception as e:
        raise BadRequest(f"video is not decodable: {e}") from None
    out = []
    for j, f in enumerate(frames):
        out.append(f)
        seg = wav[j * 16000:(j + 1) * 16000]
        if len(seg):
            out.append(seg.astype(np.float32))
    return out


def parts_to_native(parts: list[dict], resampler: str = "librosa") -> list:
    out = []
    for part in parts:
        t = part["type"]
        if t == "text":
            out.append(part["text"])
        elif t == "image_url":
            out.append(_image(_bytes((part.get("image_url") or {}).get("url"))))
        elif t == "input_audio":
            a = part.get("input_audio") or {}
            try:
                data = base64.b64decode(a.get("data", ""), validate=True)
            except ValueError:
                raise BadRequest("input_audio.data must be base64") from None
            out.append(_audio(data, resampler))
        elif t == "audio_url":
            out.append(_audio(_bytes((part.get("audio_url") or {}).get("url")), resampler))
        elif t == "video_url":
            out += _video((part.get("video_url") or {}).get("url", ""), resampler)
    return out
