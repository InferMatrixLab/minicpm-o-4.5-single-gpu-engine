"""``audio_resampler``: the 16 kHz resampling of understanding inputs (``engine.media.load_audio``) follows the named
framework's loader exactly, leaves 16 kHz input untouched, and an unknown name is refused before any model loads."""

import io
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from engine.media import RESAMPLERS, load_audio  # noqa: E402


def _wav(sr: int, channels: int = 1, seconds: float = 1.3) -> bytes:
    t = np.arange(int(sr * seconds)) / sr
    x = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.05 * np.sin(2 * np.pi * 7800 * t) + 0.01 * np.random.default_rng(0).standard_normal(t.size)
    x = np.stack([x, 0.5 * x], axis=1)[:, :channels]
    buf = io.BytesIO()
    sf.write(buf, x.astype(np.float32), sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()


@pytest.mark.parametrize("sr,channels", [(48000, 2), (24000, 1), (44100, 1)])
def test_each_resampler_is_its_framework_loader(sr, channels):
    import librosa
    import torch
    import torchaudio

    data = _wav(sr, channels)
    native, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    native = native.mean(axis=1)
    n = int(round(native.shape[0] / sr * 16000))
    want = {"librosa": librosa.load(io.BytesIO(data), sr=16000, mono=True)[0],
            "linear": np.interp(np.linspace(0.0, native.shape[0] - 1, num=n), np.arange(native.shape[0], dtype=np.float64), native),
            "torchaudio": torchaudio.transforms.Resample(sr, 16000)(torch.as_tensor(native)).numpy()}
    for how in RESAMPLERS:
        got = load_audio(io.BytesIO(data), how)
        assert got.dtype == np.float32 and got.shape == want[how].shape
        np.testing.assert_allclose(got, want[how].astype(np.float32), rtol=0, atol=1e-6)
    # the three differ near the 8 kHz band edge (the MMSU sensitivity), not in length or level
    assert not np.array_equal(load_audio(io.BytesIO(data), "linear"), load_audio(io.BytesIO(data), "torchaudio"))


def test_16khz_input_is_decoded_unchanged():
    data = _wav(16000, 2)
    native, _ = sf.read(io.BytesIO(data), dtype="float32", always_2d=True)
    for how in ("linear", "torchaudio"):
        np.testing.assert_array_equal(load_audio(io.BytesIO(data), how), native.mean(axis=1))


def test_unknown_resampler_is_refused_before_any_model_loads():
    import engine.runtime.backend_s as B

    with pytest.raises(ValueError, match="audio_resampler"):
        B.TrackSBackend(audio_resampler="soxr")
    with pytest.raises(ValueError, match="audio_resampler"):
        load_audio(io.BytesIO(_wav(48000)), "soxr")
