"""Token2Wav (flow-matching DiT + HiFT) streaming decode with per-request state.

Reference: ``stepaudio2.Token2wav`` from ``minicpmo-utils`` 1.0.6 (Apache-2.0),
which the MiniCPM-o 4.5 HF model uses (``modeling_minicpmo.py:261-266``), and the
HF stream driver (``init_token2wav_cache`` / ``streaming_generate``):

* the reference voice is loaded at 16 kHz, written as 16-bit PCM and passed to
  ``_prepare_prompt`` (S3 tokens, CAMPPlus speaker embedding, 24 kHz prompt mel);
* ``set_stream_cache`` runs the flow over the prompt tokens right-padded with
  three silence tokens (4218) to build the conformer/estimator caches;
* every 28-token window (see ``engine.token2wav_chunker``) goes through
  ``flow.inference_chunk`` with 10 Euler steps, the attention caches are
  trimmed to the prompt plus the last 100 frames, HiFT decodes the mel with an
  8-frame mel / 3840-sample source cache, consecutive chunks are cross-faded
  with a Hamming window, the first non-final chunk is shifted by 3840 samples of
  leading silence, and non-final chunks hold back their last 3840 samples.

The reference keeps one global stream state per model instance; here each
request owns a ``StreamState``, so many requests can share the weights.  The
flow's fixed noise buffer (``rand_noise``) is drawn at construction; seeding the
construction makes it reproducible across processes.
"""

from __future__ import annotations

import io
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

SAMPLE_RATE = 24000
MEL_CACHE = 8
SOURCE_CACHE = MEL_CACHE * 480
SILENCE_TOKEN = 4218


def clone_tree(x):
    if isinstance(x, torch.Tensor):
        return x.clone()
    if isinstance(x, dict):
        return {k: clone_tree(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x)(clone_tree(v) for v in x)
    return x


def fade_in_out(fade_in: torch.Tensor, fade_out: torch.Tensor, window: torch.Tensor) -> torch.Tensor:
    n = int(window.shape[0] / 2)
    out = fade_in.clone()
    out[..., :n] = fade_in[..., :n] * window[:n] + fade_out[..., -n:] * window[n:]
    return out


@dataclass
class Prompt:
    tokens: torch.Tensor
    tokens_len: torch.Tensor
    spk: torch.Tensor
    mels: torch.Tensor
    mels_len: torch.Tensor
    flow_cache: dict  # after set_stream_cache: the per-request starting state
    hift_cache: dict


@dataclass
class StreamState:
    prompt: Prompt
    flow_cache: dict
    hift_cache: dict
    chunks: int = 0


class Token2Wav:
    def __init__(self, model_dir: str | Path, float16: bool = False, n_timesteps: int = 10, noise_seed: int = 0):
        from stepaudio2 import Token2wav

        torch.manual_seed(noise_seed)  # the flow's rand_noise buffer is drawn here
        self.ref = Token2wav(str(model_dir), float16=float16, n_timesteps=n_timesteps)
        self.flow, self.hift = self.ref.flow, self.ref.hift
        self.float16, self.n_timesteps = float16, n_timesteps
        self.window = torch.from_numpy(np.hamming(2 * SOURCE_CACHE)).cuda()

    def prepare(self, ref_audio: str | Path | bytes) -> Prompt:
        """Reference voice -> prompt features and starting stream caches (the HF 16 kHz PCM16 path)."""
        import librosa
        import soundfile as sf

        src = io.BytesIO(ref_audio) if isinstance(ref_audio, (bytes, bytearray)) else str(ref_audio)
        wav16, _ = librosa.load(src, sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            sf.write(f.name, wav16, 16000)  # PCM_16, as in init_token2wav_cache
            tokens, tokens_len, spk, mels, mels_len = self.ref._prepare_prompt(f.name)
        pad = torch.full((1, 3), SILENCE_TOKEN, device=tokens.device, dtype=tokens.dtype)
        flow_cache = self.flow.setup_cache(torch.cat([tokens, pad], dim=1), mels, spk, n_timesteps=self.n_timesteps)
        hift_cache = dict(mel=torch.zeros(1, mels.shape[2], 0, device="cuda"), source=torch.zeros(1, 1, 0, device="cuda"),
                          speech=torch.zeros(1, 0, device="cuda"))
        return Prompt(tokens, tokens_len, spk, mels, mels_len, flow_cache, hift_cache)

    def new_stream(self, prompt: Prompt) -> StreamState:
        return StreamState(prompt, clone_tree(prompt.flow_cache), clone_tree(prompt.hift_cache))

    @torch.inference_mode()
    def decode(self, st: StreamState, window: list[int], last: bool) -> np.ndarray:
        """One window of codec tokens -> float32 waveform samples (the reference ``stream(..., return_waveform=True)``)."""
        p = st.prompt
        tok = torch.tensor([window], dtype=torch.int32, device="cuda")
        with torch.amp.autocast("cuda", dtype=torch.float16 if self.float16 else torch.float32):
            mel, st.flow_cache = self.flow.inference_chunk(token=tok, spk=p.spk, cache=st.flow_cache, last_chunk=last, n_timesteps=self.n_timesteps)
        keep = p.mels.shape[1]
        c = st.flow_cache
        if c["estimator_att_cache"].shape[4] > keep + 100:
            c["estimator_att_cache"] = torch.cat([c["estimator_att_cache"][:, :, :, :, :keep], c["estimator_att_cache"][:, :, :, :, -100:]], dim=4)
        if c["conformer_att_cache"].shape[3] > keep + 100:
            c["conformer_att_cache"] = torch.cat([c["conformer_att_cache"][:, :, :, :keep, :], c["conformer_att_cache"][:, :, :, -100:, :]], dim=3)
        h = st.hift_cache
        mel = torch.concat([h["mel"], mel], dim=2)
        speech, source = self.hift(mel, h["source"])
        if h["speech"].shape[-1] > 0:
            speech = fade_in_out(speech, h["speech"], self.window)
        first = h["speech"].shape[-1] == 0
        st.hift_cache = dict(mel=mel[..., -MEL_CACHE:].clone(), source=source[:, :, -SOURCE_CACHE:].clone(), speech=speech[:, -SOURCE_CACHE:].clone())
        if not last:
            if first:
                speech = torch.cat([torch.zeros(1, SOURCE_CACHE, device=speech.device), speech[:, :-SOURCE_CACHE]], dim=1)
            else:
                speech = speech[:, :-SOURCE_CACHE]
        st.chunks += 1
        return speech.float().cpu().numpy()[0]
