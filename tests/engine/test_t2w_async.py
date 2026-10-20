"""Asynchronous decode results (``BatchedToken2Wav.decode(..., async_result=True)``) against the synchronous path (GPU).

The headline batched Token2Wav (FP16 flow, fast DiT, graphs, TF32 HiFT): eight voices set up in calls of 1, 3 and 4,
staggered streams of random codes cut by the engine chunker (first-chunk ramp, steady windows, merged finals), every
window of a tick in one call.  The asynchronous run keeps up to two calls in flight and collects the oldest only when a
third is launched (as the worker does), so later calls are queued while earlier copies are pending and graph arenas are
reused before earlier results are read.  Every window must have the same length and lie within relative L2 1e-4 of the
synchronous run (the decode is not bitwise deterministic run to run: ~2.7e-5, ``tools/spike/t2w_decode_digest.py``).
"""

import os

import numpy as np
import pytest
import torch

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")
BOUND = 1e-4


def _run(t2w, feats, plans, async_results: bool):
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    bt = BatchedToken2Wav(t2w, use_graphs=True, compile_estimator=True, fast_dit=True, fast_slots=65536, hift_decode_tf32=True)
    p0 = bt.setup(t2w.features(MODEL / "assets" / "HT_ref_audio.wav"))
    bt.capture_graphs(p0)
    bt.release_prompt(p0)
    prompts = [p for a, b in ((0, 1), (1, 4), (4, 8)) for p in bt.setup_many(feats[a:b])]
    sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(prompts)]
    starts = [j % 3 for j in range(len(sts))]
    got, inflight = [], []
    for tick in range(max(s + len(p) for s, p in zip(starts, plans))):
        live = [j for j, s in enumerate(starts) if s <= tick < s + len(plans[j])]
        items = [(sts[j], *plans[j][tick - starts[j]]) for j in live]
        if async_results:
            inflight.append((live, bt.decode(items, async_result=True)))
            if len(inflight) > 2:
                lv, pend = inflight.pop(0)
                got += list(zip(lv, pend.result()))
        else:
            got += list(zip(live, bt.decode(items)))
    for lv, pend in inflight:
        got += list(zip(lv, pend.result()))
    for st in sts:
        bt.release_stream(st)
    del bt
    torch.cuda.empty_cache()
    return got


@torch.inference_mode()
def test_async_decode_results_match_sync():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.token2wav_chunker import StreamChunker

    torch.backends.cudnn.allow_tf32 = False
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    rows = load_seed_tts_rows()
    feats = [t2w.features(cr.ref_audio_bytes()) for cr in canonical_requests(rows, make_id_sets(rows).tuning(32)[:8])]
    gen = np.random.default_rng(11)
    plans = []
    for _ in feats:
        ch = StreamChunker(first_chunk=8)
        ws = [(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, int(gen.integers(60, 140)))])]
        plans.append(ws + [(lambda w: (w.tokens, w.last))(ch.finish())])
    ref = _run(t2w, feats, plans, False)
    new = _run(t2w, feats, plans, True)
    assert [j for j, _ in ref] == [j for j, _ in new]  # the same windows, in the same order
    rel, same = [], 0
    for (_, a), (_, b) in zip(new, ref):
        assert a.shape == b.shape
        same += int(np.array_equal(a, b))
        rel.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-12)))
    print(f"async vs sync decode: {len(rel)} windows, {same} bitwise equal, max rel L2 {max(rel):.2e}")
    assert max(rel) <= BOUND
