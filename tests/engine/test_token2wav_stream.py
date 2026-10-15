"""Engine Token2Wav stream decode against the HF reference ``stepaudio2.Token2wav`` stream path (GPU).

Both instances are built with the same seed, so the flow's fixed noise buffer is identical; the windows come from the
engine chunker.  Run with CUDA_VISIBLE_DEVICES set to a free GPU.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
REF_WAV = MODEL / "assets" / "HT_ref_audio.wav"
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")


def test_stream_matches_reference():
    import librosa
    import soundfile as sf
    from stepaudio2 import Token2wav

    from engine.models.token2wav import SILENCE_TOKEN, Token2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    torch.manual_seed(7)
    ref = Token2wav(str(MODEL / "assets" / "token2wav"))
    assert torch.equal(ours.flow.decoder.rand_noise, ref.flow.decoder.rand_noise)
    wav16, _ = librosa.load(str(REF_WAV), sr=16000, mono=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f:
        sf.write(f.name, wav16, 16000)
        ref.cache = None
        ref.stream_cache, ref.hift_cache_dict = ref.set_stream_cache(f.name)
    gen = np.random.default_rng(0)
    tokens = [int(x) for x in gen.integers(0, 6561, 137)]
    ch = StreamChunker()
    windows = [(w.tokens, w.last) for w in ch.push(tokens)] + [(lambda w: (w.tokens, w.last))(ch.finish())]
    st = ours.new_stream(ours.prepare(REF_WAV))
    for toks, last in windows:
        a = ours.decode(st, toks, last)
        b = ref.stream(toks, prompt_wav=None, last_chunk=last, return_waveform=True)[0]
        assert a.shape == b.shape and np.array_equal(a, b)
    assert windows[0][0][:3] == [SILENCE_TOKEN] * 3
