"""AC-3.4 acceptance for the served flow path: every chunk against an independent FP32 reference (GPU).

Thirty-two utterances: 32 distinct Seed-TTS reference voices (the tuning rows), codec windows from the engine chunker
(first-chunk ramp of the headline configuration, steady 25+3 windows and final flushes), random codec tokens of
70-150 frames.  Reference: ``stepaudio2`` FP32 (``Token2Wav(float16=False)``), each stream alone: prompt setup by the
reference ``setup_cache`` and every window by the reference ``flow.inference_chunk`` with the stream's own caches and
trimming (as ``Token2Wav._decode``).  Served path: the headline's ``BatchedToken2Wav`` (FP16 flow, fast DiT, CUDA
graphs), batched setups and staggered streams so steady windows, ramped first windows and merged finals share calls.
Bound: relative L2 of every chunk's mel <= 5e-3 (the AC-3.4 FP16 bound).  The vocoder is the same strict-FP32 HiFT on
both sides; its graph/eager agreement on equal mels is ``test_token2wav_stream.py::test_hift_graphs_match_eager``.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")
N_UTT, FIRST_CHUNK, BOUND = 32, 8, 5e-3


def _reference_mels(ref, feats, windows):
    """FP32 reference: setup_cache, then inference_chunk per window with the stream's own caches and trims."""
    from engine.models.token2wav import own_flow_cache, strict_fp32

    p = ref.setup(feats)
    cache = own_flow_cache(p.flow_cache, ref.n_timesteps)
    keep, out = p.mels.shape[1], []
    with torch.inference_mode(), strict_fp32():
        for tokens, last in windows:
            tok = torch.tensor([tokens], dtype=torch.int32, device="cuda")
            mel, cache = ref.flow.inference_chunk(token=tok, spk=p.spk, cache=cache, last_chunk=last, n_timesteps=ref.n_timesteps)
            cache = own_flow_cache(cache, ref.n_timesteps)
            c = cache
            if c["estimator_att_cache"].shape[4] > keep + 100:
                c["estimator_att_cache"] = torch.cat([c["estimator_att_cache"][..., :keep, :], c["estimator_att_cache"][..., -100:, :]], dim=4)
            if c["conformer_att_cache"].shape[3] > keep + 100:
                c["conformer_att_cache"] = torch.cat([c["conformer_att_cache"][:, :, :, :keep, :], c["conformer_att_cache"][:, :, :, -100:, :]], dim=3)
            out.append(mel.float().cpu())
    return out


@torch.inference_mode()
def test_served_fast_fp16_flow_against_fp32_reference():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    torch.backends.cudnn.allow_tf32 = False
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[:N_UTT])
    served = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    feats = [served.features(cr.ref_audio_bytes()) for cr in reqs]
    gen = np.random.default_rng(34)
    plans = []
    for _ in range(N_UTT):
        ch = StreamChunker(first_chunk=FIRST_CHUNK)
        ws = [(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, int(gen.integers(70, 151)))])]
        plans.append(ws + [(lambda w: (w.tokens, w.last))(ch.finish())])
    # served path: batched setups, streams started in four waves, every window of a tick in one decode call
    bt = BatchedToken2Wav(served, use_graphs=True, fast_dit=True, fast_slots=65536)
    bt.capture_graphs(bt.setup(feats[0]))
    rec, flow = [], bt._flow
    bt._flow = lambda sts, ws, last, *rest: (lambda m: rec.extend(zip([id(s) for s in sts], [x.float().cpu() for x in m])) or m)(flow(sts, ws, last, *rest))
    prompts = bt.setup_many(feats)
    sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(prompts)]
    order = {id(st): j for j, st in enumerate(sts)}
    starts = [j % 4 for j in range(N_UTT)]
    for tick in range(max(s + len(p) for s, p in zip(starts, plans))):
        items = [(sts[j], *plans[j][tick - s]) for j, s in enumerate(starts) if s <= tick < s + len(plans[j])]
        bt.decode(items)
    got = {j: [] for j in range(N_UTT)}
    for sid, m in rec:
        got[order[sid]].append(m)
    for st in sts:
        bt.release_stream(st)
    assert bt.fast.counts["graph"] > 0
    del bt, served
    torch.cuda.empty_cache()
    # independent FP32 reference, stream by stream
    ref = Token2Wav(MODEL / "assets" / "token2wav", float16=False, noise_seed=7)
    errs = []
    for j in range(N_UTT):
        want = _reference_mels(ref, feats[j], plans[j])
        assert len(want) == len(got[j]), j
        for k, (x, y) in enumerate(zip(got[j], want)):
            assert x.shape == y.shape, (j, k, x.shape, y.shape)
            errs.append((float((x - y).norm() / y.norm()), j, k))
    worst = max(errs)
    print(f"AC-3.4 served FP16/fast vs FP32: {len(errs)} chunks, max rel L2 {worst[0]:.2e} (utt {worst[1]}, chunk {worst[2]}), "
          f"median {float(np.median([e[0] for e in errs])):.2e}")
    assert worst[0] <= BOUND, worst
