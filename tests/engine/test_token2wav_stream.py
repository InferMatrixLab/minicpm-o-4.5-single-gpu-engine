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


@pytest.mark.parametrize("float16,compiled", [(False, False), (True, False), (True, True)])
def test_graph_estimator_matches_eager(float16, compiled):
    """CUDA-graph estimator replays give the eager batched decode's mel (1e-4); inductor-compiled ones within the AC-3.4
    FP16 bound (5e-3).  Final windows replay padded graphs (other GEMM shapes than eager), and HiFT turns any mel change
    into a phase change, so waveforms are not compared here."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", float16=float16, noise_seed=7)
    refs = [REF_WAV, MODEL / "assets" / "system_ref_audio.wav"]
    prompts = [ours.prepare(r) for r in refs]
    eager, graph = BatchedToken2Wav(ours), BatchedToken2Wav(ours, use_graphs=True, compile_estimator=compiled)
    gen = np.random.default_rng(3)
    plans = []
    for n in (90, 70):
        ch = StreamChunker()
        plans.append([(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, n)])] + [(lambda w: (w.tokens, w.last))(ch.finish())])
    outs, mels = [], []
    for bt in (eager, graph):
        rec, flow = [], bt._flow
        bt._flow = lambda sts, ws, last, flow=flow, rec=rec: (lambda m: rec.append([x.float().cpu() for x in m]) or m)(flow(sts, ws, last))
        mels.append(rec)
        sts = [ours.new_stream(p, noise_seed=j) for j, p in enumerate(prompts)]
        got = [[], []]
        for k in range(max(len(p) for p in plans)):
            items = [(sts[j], *plans[j][k]) for j in range(2) if k < len(plans[j])]
            for j, w in zip([j for j in range(2) if k < len(plans[j])], bt.decode(items)):
                got[j].append(w)
        outs.append([np.concatenate(g) for g in got])
    for a, b in zip(outs[1], outs[0]):
        assert a.shape == b.shape
    for ga, ea in zip(mels[1], mels[0]):
        for x, y in zip(ga, ea):
            assert float((x - y).norm() / y.norm()) <= (5e-3 if compiled or float16 else 1e-4)  # AC-3.4 FP16 bound


def test_batched_setup_matches_reference_setup_cache():
    """The prompt setup on the batched solver equals the reference ``setup_cache`` (FP32): caches and the stream after."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    bt = BatchedToken2Wav(ours)
    feats = ours.features(MODEL / "assets" / "system_ref_audio.wav")
    ref, got = ours.setup(feats), bt.setup(feats)
    for k, v in ref.flow_cache.items():
        g = got.flow_cache[k]
        assert g.shape == v.shape, k
        assert float((g.float() - v.float()).norm() / v.float().norm().clamp_min(1e-12)) <= 1e-5, k
    ch = StreamChunker()
    ws = [(w.tokens, w.last) for w in ch.push([int(x) for x in np.random.default_rng(4).integers(0, 6561, 60)])]
    sa, sb = ours.new_stream(got, noise_seed=1), ours.new_stream(ref, noise_seed=1)
    a = np.concatenate([bt.decode([(sa, *w)])[0] for w in ws])
    b = np.concatenate([bt.decode([(sb, *w)])[0] for w in ws])
    assert np.linalg.norm(a - b) / np.linalg.norm(b) <= 1e-4


def test_hift_graphs_match_eager():
    """Graph-replayed HiFT (first-window and steady shapes, several rows) equals the eager seeded HiFT (same mel)."""
    from engine.models.token2wav import SOURCE_CACHE, Token2Wav, hift_forward
    from engine.runtime.hift_graph import HiFTGraphs

    ours = Token2Wav(MODEL / "assets" / "token2wav")
    hg = HiFTGraphs(ours.hift, rows=(1, 2, 4))
    g = torch.Generator(device="cuda").manual_seed(0)
    with torch.inference_mode():
        for F, src in ((50, 0), (58, SOURCE_CACHE)):
            for n in (1, 3):
                mel = torch.randn(n, 80, F, device="cuda", generator=g)
                cs = torch.randn(n, 1, src, device="cuda", generator=g) * 0.01
                a, sa = hg(mel, cs, list(range(n)))
                with __import__("engine.models.token2wav", fromlist=["strict_fp32"]).strict_fp32():
                    b, sb = hift_forward(ours.hift, mel, cs, list(range(n)))
                assert a.shape == b.shape and sa.shape == sb.shape
                assert float((a - b).norm() / b.norm()) <= 1e-4, (F, n)
    assert hg.counts["graph"] == 4
