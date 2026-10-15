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

    from engine.models.token2wav import SILENCE_TOKEN, Token2Wav, chunk_noise_seed, strict_fp32
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    torch.manual_seed(7)
    ref = Token2wav(str(MODEL / "assets" / "token2wav"))
    assert torch.equal(ours.flow.decoder.rand_noise, ref.flow.decoder.rand_noise)
    wav16, _ = librosa.load(str(REF_WAV), sr=16000, mono=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f, strict_fp32():
        sf.write(f.name, wav16, 16000)
        ref.cache = None
        ref.stream_cache, ref.hift_cache_dict = ref.set_stream_cache(f.name)
    gen = np.random.default_rng(0)
    tokens = [int(x) for x in gen.integers(0, 6561, 137)]
    ch = StreamChunker()
    windows = [(w.tokens, w.last) for w in ch.push(tokens)] + [(lambda w: (w.tokens, w.last))(ch.finish())]
    st = ours.new_stream(ours.prepare(REF_WAV), noise_seed=11)
    a_all, b_all = [], []
    for k, (toks, last) in enumerate(windows):
        a = ours.decode(st, toks, last)
        torch.manual_seed(chunk_noise_seed(11, k))  # the reference HiFT draws its source noise from the global generator
        with strict_fp32():  # the FP32 reference of AC-3.4 (torch's default TF32 cuDNN convs move HiFT by ~30%)
            b = ref.stream(toks, prompt_wav=None, last_chunk=last, return_waveform=True)[0]
        assert a.shape == b.shape
        a_all.append(a)
        b_all.append(b)
    a, b = np.concatenate(a_all), np.concatenate(b_all)
    # with the HiFT source noise seeded alike, the remaining difference is GPU kernel nondeterminism; the engine must be within 1e-4
    assert np.linalg.norm(a - b) / np.linalg.norm(b) <= 1e-4
    assert windows[0][0][:3] == [SILENCE_TOKEN] * 3


def test_interleaved_streams_are_independent():
    """Two requests decoded window by window in alternation give the same audio as each decoded alone.

    The reference flow returns model-wide scratch buffers as its caches; a stream that kept them uncopied would be
    overwritten by the other request's chunk.
    """
    from engine.models.token2wav import Token2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    prompt = ours.prepare(REF_WAV)
    gen = np.random.default_rng(1)

    def windows(n):
        ch = StreamChunker()
        toks = [int(x) for x in gen.integers(0, 6561, n)]
        return [(w.tokens, w.last) for w in ch.push(toks)] + [(lambda w: (w.tokens, w.last))(ch.finish())]

    wa, wb = windows(90), windows(115)
    alone = []
    for j, ws in enumerate((wa, wb)):
        st = ours.new_stream(prompt, noise_seed=j)
        alone.append(np.concatenate([ours.decode(st, t, last) for t, last in ws]))
    sa, sb = ours.new_stream(prompt, noise_seed=0), ours.new_stream(prompt, noise_seed=1)
    out = ([], [])
    for i in range(max(len(wa), len(wb))):
        for j, (st, ws) in enumerate(((sa, wa), (sb, wb))):
            if i < len(ws):
                out[j].append(ours.decode(st, *ws[i]))
    for j in range(2):
        got = np.concatenate(out[j])
        assert got.shape == alone[j].shape
        assert np.linalg.norm(got - alone[j]) / np.linalg.norm(alone[j]) <= 1e-4


def test_batched_decode_matches_per_stream():
    """Batched windows of streams with different reference prompts (cache lengths), at different stages, in one call
    (steady windows, a first window and a final flush) match each stream decoded alone (FP32).

    Batched GEMMs and convolutions round differently from batch-1 ones and HiFT amplifies that, so the bound is the
    AC-3.4 HiFT bound (waveform relative L2 1e-2 against FP32 with fixed source noise); measured ~1e-3.
    """
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    batch = BatchedToken2Wav(ours)
    refs = [REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav"]
    prompts = [ours.prepare(r) for r in refs]
    assert len({p.mels.shape[1] for p in prompts}) == 3, "the test needs three different prompt lengths"
    gen = np.random.default_rng(2)
    plans = []
    for n in (160, 60, 110):
        ch = StreamChunker()
        toks = [int(x) for x in gen.integers(0, 6561, n)]
        plans.append([(w.tokens, w.last) for w in ch.push(toks)] + [(lambda w: (w.tokens, w.last))(ch.finish())])
    alone = []
    for j, (p, ws) in enumerate(zip(prompts, plans)):
        st = ours.new_stream(p, noise_seed=j)
        alone.append(np.concatenate([ours.decode(st, t, last) for t, last in ws]))
    # stream 1 starts two windows later than stream 0, stream 2 one window later
    starts = [0, 2, 1]
    sts = [ours.new_stream(p, noise_seed=j) for j, p in enumerate(prompts)]
    got = [[], [], []]
    for tick in range(max(s + len(ws) for s, ws in zip(starts, plans))):
        items, who = [], []
        for j, (s, ws) in enumerate(zip(starts, plans)):
            if s <= tick < s + len(ws):
                items.append((sts[j], *ws[tick - s]))
                who.append(j)
        for j, wav in zip(who, batch.decode(items)):
            got[j].append(wav)
    for j in range(3):
        g = np.concatenate(got[j])
        assert g.shape == alone[j].shape
        assert np.linalg.norm(g - alone[j]) / np.linalg.norm(alone[j]) <= 1e-2, j
