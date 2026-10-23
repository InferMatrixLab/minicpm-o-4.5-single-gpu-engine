"""Engine Token2Wav stream decode against the HF reference ``stepaudio2.Token2wav`` stream path (GPU).

Both instances are built with the same seed, so the flow's fixed noise buffer is identical; the windows come from the
engine chunker.  Run with CUDA_VISIBLE_DEVICES set to a free GPU.
"""

import os
import tempfile

import numpy as np
import pytest
import torch

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
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
        bt._flow = lambda sts, ws, last, *rest, flow=flow, rec=rec: (lambda m: rec.append([x.float().cpu() for x in m]) or m)(flow(sts, ws, last, *rest))
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


def test_ramped_first_window_graphs_match_eager():
    """With the first-chunk ramp (first window 8 tokens + 3 lookahead) the graph path's mels equal the eager batched
    path's (FP32), first window included."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    prompts = [ours.prepare(r) for r in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav")]
    gen = np.random.default_rng(5)
    plans = []
    for n in (70, 45):
        ch = StreamChunker(first_chunk=8)
        ws = [(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, n)])] + [(lambda w: (w.tokens, w.last))(ch.finish())]
        assert len(ws[0][0]) == 11
        plans.append(ws)
    mels = []
    for bt in (BatchedToken2Wav(ours), BatchedToken2Wav(ours, use_graphs=True)):
        rec, flow = [], bt._flow
        bt._flow = lambda sts, ws, last, *rest, flow=flow, rec=rec: (lambda m: rec.append([x.float().cpu() for x in m]) or m)(flow(sts, ws, last, *rest))
        sts = [ours.new_stream(p, noise_seed=j) for j, p in enumerate(prompts)]
        for k in range(max(len(p) for p in plans)):
            bt.decode([(sts[j], *plans[j][k]) for j in range(2) if k < len(plans[j])])
        mels.append(rec)
    for ga, ea in zip(mels[1], mels[0]):
        for x, y in zip(ga, ea):
            assert float((x - y).norm() / y.norm()) <= 1e-4


def test_batched_setups_of_different_prompts_match_single():
    """setup_many over prompts of different lengths (one padded solve) equals each prompt set up alone (FP32)."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    for bt in (BatchedToken2Wav(ours), BatchedToken2Wav(ours, use_graphs=True)):
        feats = [ours.features(r) for r in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav")]
        many = bt.setup_many(feats)
        for f, m in zip(feats, many):
            one = bt.setup(f)
            for k, v in one.flow_cache.items():
                assert m.flow_cache[k].shape == v.shape, k
                assert float((m.flow_cache[k].float() - v.float()).norm() / v.float().norm().clamp_min(1e-12)) <= 1e-4, k


@pytest.mark.parametrize("graphs", [False, True])
def test_merged_final_windows_match_unmerged(graphs):
    """Final windows riding in the steady 28-token call (zero-padded) give the same mel as their own final-window call."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", noise_seed=7)
    prompts = [ours.prepare(r) for r in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav")]
    gen = np.random.default_rng(6)
    plans = []
    for n in (60, 85, 74):  # final windows of 13, 13 and 2 + 3 tokens, alongside steady windows of the others
        ch = StreamChunker()
        plans.append([(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, n)])] + [(lambda w: (w.tokens, w.last))(ch.finish())])
    mels = []
    for merge in (False, True):
        bt = BatchedToken2Wav(ours, use_graphs=graphs)
        bt.merge_finals = merge
        rec, flow = [], bt._flow
        bt._flow = lambda sts, ws, last, *rest, flow=flow, rec=rec: (lambda m: rec.extend(zip([id(s) for s in sts], [x.float().cpu() for x in m])) or m)(
            flow(sts, ws, last, *rest))
        sts = [ours.new_stream(p, noise_seed=j) for j, p in enumerate(prompts)]
        order = {id(st): j for j, st in enumerate(sts)}
        got = {j: [] for j in range(3)}
        for k in range(max(len(p) for p in plans)):
            bt.decode([(sts[j], *plans[j][k]) for j in range(3) if k < len(plans[j])])
        for sid, m in rec:
            got[order[sid]].append(m)
        mels.append(got)
    for j in range(3):
        assert len(mels[0][j]) == len(mels[1][j])
        for x, y in zip(mels[1][j], mels[0][j]):
            assert x.shape == y.shape
            assert float((x - y).norm() / y.norm()) <= 1e-4, j


@pytest.mark.parametrize("float16,first_chunk", [(False, 25), (True, 25), (False, 8)])
def test_whole_euler_graph_matches_eager(float16, first_chunk):
    """The whole-Euler graph (all steps in one replay, caches through its arenas) gives the eager batched mels
    (FP32 1e-4; FP16 within the AC-3.4 FP16 bound), for ragged prompts and staggered streams."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", float16=float16, noise_seed=7)
    prompts = [ours.prepare(r) for r in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav")]
    gen = np.random.default_rng(8)
    plans = []
    for n in (120, 95, 70):
        ch = StreamChunker(first_chunk=first_chunk)  # 8: ramped first windows (16 frames) beside 50-frame ones
        plans.append([(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, n)])] + [(lambda w: (w.tokens, w.last))(ch.finish())])
    starts = [0, 1, 2]
    mels = []
    for bt in (BatchedToken2Wav(ours), BatchedToken2Wav(ours, use_graphs=True, euler_graphs=True)):
        rec, flow = [], bt._flow
        bt._flow = lambda sts, ws, last, *rest, flow=flow, rec=rec: (lambda m: rec.extend(zip([id(s) for s in sts], [x.float().cpu() for x in m])) or m)(
            flow(sts, ws, last, *rest))
        sts = [ours.new_stream(p, noise_seed=j) for j, p in enumerate(prompts)]
        order = {id(st): j for j, st in enumerate(sts)}
        for tick in range(max(s + len(p) for s, p in zip(starts, plans))):
            items = [(sts[j], *plans[j][tick - s]) for j, s in enumerate(starts) if s <= tick < s + len(plans[j])]
            bt.decode(items)
        got = {j: [] for j in range(3)}
        for sid, m in rec:
            got[order[sid]].append(m)
        mels.append(got)
        if bt.euler is not None:
            assert bt.euler.graphs, "the whole-Euler path was not used"
    tol = 5e-3 if float16 else 1e-4
    for j in range(3):
        for x, y in zip(mels[1][j], mels[0][j]):
            assert x.shape == y.shape
            assert float((x - y).norm() / y.norm()) <= tol, j


@pytest.mark.parametrize("graphs,first_chunk", [(False, 25), (True, 25), (True, 8)])
def test_fast_dit_matches_eager(graphs, first_chunk):
    """The fast DiT solve (slot-pool caches, packed ragged rows; ``engine.runtime.dit_fast``) gives the eager batched
    FP16 mels within the AC-3.4 FP16 bound, setups included, for ragged prompts, staggered streams, ramped first
    windows and merged finals; every cache slot is returned when the streams are released."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    feats = [ours.features(r) for r in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav")]
    gen = np.random.default_rng(8)
    plans = []
    for n in (120, 95, 70):
        ch = StreamChunker(first_chunk=first_chunk)
        plans.append([(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, n)])] + [(lambda w: (w.tokens, w.last))(ch.finish())])
    starts = [0, 1, 2]
    mels = []
    for bt in (BatchedToken2Wav(ours), BatchedToken2Wav(ours, use_graphs=graphs, fast_dit=True, fast_slots=8192)):
        rec, flow = [], bt._flow
        bt._flow = lambda sts, ws, last, *rest, flow=flow, rec=rec: (lambda m: rec.extend(zip([id(s) for s in sts], [x.float().cpu() for x in m])) or m)(
            flow(sts, ws, last, *rest))
        prompts = bt.setup_many(feats)
        sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(prompts)]
        order = {id(st): j for j, st in enumerate(sts)}
        for tick in range(max(s + len(p) for s, p in zip(starts, plans))):
            items = [(sts[j], *plans[j][tick - s]) for j, s in enumerate(starts) if s <= tick < s + len(plans[j])]
            bt.decode(items)
        got = {j: [] for j in range(3)}
        for sid, m in rec:
            got[order[sid]].append(m)
        mels.append(got)
        if bt.fast is not None:
            assert bt.fast.counts["graph" if graphs else "eager"] > 0, bt.fast.counts
            for st in sts:
                bt.release_stream(st)
            assert bt.fast.pool.in_use() == bt.fast.scratch
    errs = []
    for j in range(3):
        for x, y in zip(mels[1][j], mels[0][j]):
            assert x.shape == y.shape
            errs.append(float((x - y).norm() / y.norm()))
    assert max(errs) <= 5e-3, errs  # AC-3.4 FP16 bound


@pytest.mark.parametrize("graphs", [False, True])
def test_merged_solve_matches_per_group_solves(graphs):
    """``merge_solve``: a decode whose windows fall into several groups (ramped first windows, steady windows, merged
    finals) runs one packed fast-DiT solve instead of one per group; its mels equal the per-group solves' within the
    AC-3.4 FP16 bound, and every cache slot is returned.  The waveforms agree too: HiFT runs outside the flow's FP16
    autocast on both paths (the merged path once vocoded inside it, so HiFT's eager calls ran in FP16)."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    feats = [ours.features(r) for r in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav")] * 2
    gen = np.random.default_rng(9)
    plans = []
    for n in (120, 95, 70, 140, 60, 101):
        ch = StreamChunker(first_chunk=8)
        plans.append([(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, n)])] + [(lambda w: (w.tokens, w.last))(ch.finish())])
    starts = [0, 1, 2, 2, 4, 7]
    mels, calls, wavs = [], [], []
    for merge in (False, True):
        bt = BatchedToken2Wav(ours, use_graphs=graphs, fast_dit=True, fast_slots=32768, merge_solve=merge)
        rec, vocode, solve = [], bt._vocode, bt.fast.solve
        bt._vocode = lambda items, ms, *rest, vocode=vocode, rec=rec: (rec.extend((id(it[0]), m.float().cpu()) for it, m in zip(items, ms))
                                                                     or vocode(items, ms, *rest))
        n_solve = [0]
        bt.fast.solve = lambda *a, solve=solve, n_solve=n_solve: (n_solve.__setitem__(0, n_solve[0] + 1), solve(*a))[1]
        prompts = bt.setup_many(feats)
        sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(prompts)]
        order = {id(st): j for j, st in enumerate(sts)}
        n_solve[0] = 0
        wav = {j: [] for j in range(len(sts))}
        for tick in range(max(s + len(p) for s, p in zip(starts, plans))):
            js = [j for j, s in enumerate(starts) if s <= tick < s + len(plans[j])]
            for j, w in zip(js, bt.decode([(sts[j], *plans[j][tick - starts[j]]) for j in js])):
                wav[j].append(w)
        wavs.append({j: np.concatenate(w) for j, w in wav.items()})
        got = {j: [] for j in range(len(sts))}
        for sid, m in rec:
            got[order[sid]].append(m)
        mels.append(got)
        calls.append(n_solve[0])
        for st in sts:
            bt.release_stream(st)
        assert bt.fast.pool.in_use() == bt.fast.scratch
    assert calls[1] < calls[0], calls  # some decodes had several groups
    errs = [float((x - y).norm() / y.norm()) for j in mels[0] for x, y in zip(mels[1][j], mels[0][j])]
    assert len(errs) == sum(len(p) for p in plans)
    werr = max(float(np.linalg.norm(wavs[1][j] - wavs[0][j]) / np.linalg.norm(wavs[0][j])) for j in wavs[0])
    print(f"merged solve: {calls[0]} -> {calls[1]} solve calls, max mel rel L2 {max(errs):.2e}, bitwise {max(errs) == 0.0}, wav {werr:.2e}")
    assert max(errs) <= 5e-3, errs  # AC-3.4 FP16 bound
    assert werr <= 1e-4, werr  # equal mels: only HiFT's run-to-run spread


@pytest.mark.parametrize("graphs", [False, True])
def test_fused_setups_match_separate_setups(graphs):
    """``decode(setups=...)``: prompt setups whose rows ride a decode's packed solve give the same prompt caches as
    ``setup_many`` (the later windows of those voices, and the decode's own windows, within the AC-3.4 FP16 bound), and
    every cache slot is returned."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    feats = [ours.features(r) for r in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav")] * 2
    gen = np.random.default_rng(10)
    plans = []
    for n in (120, 95, 70, 140, 60, 101):
        ch = StreamChunker(first_chunk=8)
        plans.append([(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, n)])] + [(lambda w: (w.tokens, w.last))(ch.finish())])
    late = 2  # voices 3-5 are set up in tick 2, with voices 0-2 mid-stream
    mels = []
    for fused in (False, True):
        bt = BatchedToken2Wav(ours, use_graphs=graphs, fast_dit=True, fast_slots=32768, merge_solve=True, fuse_setup_frames=2048 if fused else 0)
        rec, vocode, solve = [], bt._vocode, bt.fast.solve
        bt._vocode = lambda items, ms, *rest, vocode=vocode, rec=rec: (rec.extend((id(it[0]), m.float().cpu()) for it, m in zip(items, ms))
                                                                     or vocode(items, ms, *rest))
        sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(bt.setup_many(feats[:3]))]
        for tick in range(max(len(p) for p in plans) + late + 1):
            items = [(sts[j], *plans[j][tick - (0 if j < 3 else late + 1)]) for j in range(len(sts))
                     if 0 <= tick - (0 if j < 3 else late + 1) < len(plans[j])]
            if tick == late:
                if fused:
                    _, prompts = bt.decode(items, setups=feats[3:])
                else:
                    prompts = bt.setup_many(feats[3:])
                    bt.decode(items)
                sts += [bt.new_stream(p, noise_seed=3 + j, adopt=True) for j, p in enumerate(prompts)]
            elif items:
                bt.decode(items)
        order = {id(st): j for j, st in enumerate(sts)}
        got = {j: [] for j in range(len(sts))}
        for sid, m in rec:
            got[order[sid]].append(m)
        mels.append(got)
        for st in sts:
            bt.release_stream(st)
        assert bt.fast.pool.in_use() == bt.fast.scratch
        if graphs:
            assert bt.fast.counts["eager"] == 0, bt.fast.counts
    errs = {j: max(float((x - y).norm() / y.norm()) for x, y in zip(mels[1][j], mels[0][j])) for j in mels[0]}
    assert all(len(mels[1][j]) == len(mels[0][j]) == len(plans[j]) for j in mels[0])
    print(f"fused setups: max mel rel L2 per voice {errs}")
    assert max(errs.values()) <= 5e-3, errs  # AC-3.4 FP16 bound


@pytest.mark.parametrize("graphs", [False, True])
def test_dit_fused_residual_is_bitwise(graphs):
    """``dit_fused_residual``: the gated residual adds fused with the norms after them give the same mels, bit for bit,
    for setups (fused into a decode too), ramped first windows, steady windows and merged finals; so does
    ``dit_fused_conv`` on top of it (the im2col written by the norms)."""
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ours = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    feats = [ours.features(r) for r in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav")] * 2
    gen = np.random.default_rng(11)
    plans = []
    for n in (120, 95, 70, 140, 60, 101):
        ch = StreamChunker(first_chunk=8)
        plans.append([(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, n)])] + [(lambda w: (w.tokens, w.last))(ch.finish())])
    mels = []
    for fused, conv in ((False, False), (True, False), (True, True)):
        bt = BatchedToken2Wav(ours, use_graphs=graphs, fast_dit=True, fast_slots=32768, merge_solve=True, fuse_setup_frames=2048,
                              dit_fused_residual=fused, dit_fused_conv=conv)
        rec, vocode = [], bt._vocode
        bt._vocode = lambda items, ms, *rest, vocode=vocode, rec=rec: (rec.extend(m.float().cpu() for m in ms) or vocode(items, ms, *rest))
        sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(bt.setup_many(feats[:3]))]
        for tick in range(max(len(p) for p in plans) + 3):
            items = [(sts[j], *plans[j][tick - (0 if j < 3 else 3)]) for j in range(len(sts)) if 0 <= tick - (0 if j < 3 else 3) < len(plans[j])]
            if tick == 2:
                _, prompts = bt.decode(items, setups=feats[3:])
                sts += [bt.new_stream(p, noise_seed=3 + j, adopt=True) for j, p in enumerate(prompts)]
            elif items:
                bt.decode(items)
        mels.append(rec)
        for st in sts:
            bt.release_stream(st)
    assert len(mels[0]) == len(mels[1]) == len(mels[2]) == sum(len(p) for p in plans)
    assert all(torch.equal(x, y) for x, y in zip(mels[0], mels[1]))
    assert all(torch.equal(x, y) for x, y in zip(mels[0], mels[2]))


def test_split_prompt_features_equal_reference():
    """The feature path of the worker, split into a CPU front end (feature processes, no CUDA) and the GPU S3 tokens in
    the worker (``prompt_front_end`` + ``finish_prompt``), returns exactly the reference ``Token2wav._prepare_prompt``
    tensors."""
    import types

    import onnxruntime
    import s3tokenizer
    from stepaudio2.token2wav import Token2wav

    from engine.runtime.t2w_worker import finish_prompt, prompt_front_end

    d = MODEL / "assets" / "token2wav"
    h = types.SimpleNamespace(audio_tokenizer=s3tokenizer.load_model(str(d / "speech_tokenizer_v2_25hz.onnx")).cuda().eval(),
                              spk_model=onnxruntime.InferenceSession(str(d / "campplus.onnx"), providers=["CPUExecutionProvider"]),
                              flow=types.SimpleNamespace(up_rate=2))
    for wav in (REF_WAV, MODEL / "assets" / "system_ref_audio.wav", MODEL / "assets" / "bajie.wav"):
        with torch.inference_mode():
            a = Token2wav._prepare_prompt(h, str(wav))
            front = prompt_front_end(h, str(wav))
            assert all(not (isinstance(x, torch.Tensor) and x.is_cuda) for x in front)  # the front end stays on the CPU
            b = finish_prompt(h.audio_tokenizer, front)
            for x, y in zip(a, b):
                assert torch.equal(x, y)
