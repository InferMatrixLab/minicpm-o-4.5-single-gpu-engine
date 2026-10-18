"""AC-3.4 HiFT bound for the TF32 decoder (``t2w_hift_decode_tf32``): the served waveform against strict FP32 (GPU).

Thirty-two utterances: 32 distinct Seed-TTS reference voices (the tuning rows); each speaks the S3 speech tokens of
another row's reference recording (real speech, so real mel and f0 contours), cut into windows by the engine chunker
(first-chunk ramp of the headline configuration, steady 25+3 windows, final flushes).  Both runs use the headline
``BatchedToken2Wav`` (FP16 flow, fast DiT, CUDA graphs including the HiFT graphs), the same batched setups, the same
staggered schedule and the same per-request noise seeds, so the mels are identical and only HiFT's decoder precision
differs.  Bound: relative L2 of every chunk's waveform and of every whole utterance <= 1e-2 against the strict-FP32
vocoder (whose graph/eager agreement with the reference is ``test_token2wav_stream.py::test_hift_graphs_match_eager``).
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")
N_UTT, FIRST_CHUNK, BOUND = 32, 8, 1e-2


EXTENDED_ROWS = (1, 2, 4, 8, 12, 16, 20, 24, 32)  # t2w_hift_rows of track_s_fast_r13


def _serve(served, feats, plans, tf32: bool, hift_rows=None, batch_setup=False):
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    bt = BatchedToken2Wav(served, use_graphs=True, fast_dit=True, fast_slots=65536, hift_decode_tf32=tf32, hift_rows=hift_rows,
                          batch_setup_encoder=batch_setup)
    p0 = bt.setup(feats[0])
    bt.capture_graphs(p0)
    bt.release_prompt(p0)
    prompts = bt.setup_many(feats)
    sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(prompts)]
    starts = [j % 4 for j in range(N_UTT)]
    got = {j: [] for j in range(N_UTT)}
    for tick in range(max(s + len(p) for s, p in zip(starts, plans))):
        live = [j for j, s in enumerate(starts) if s <= tick < s + len(plans[j])]
        wavs = bt.decode([(sts[j], *plans[j][tick - starts[j]]) for j in live])
        for j, w in zip(live, wavs):
            got[j].append(np.asarray(w, dtype=np.float64))
    assert bt.hift_graphs.counts["graph"] > 0
    if hift_rows:  # the workload reaches the buckets above 16 rows
        assert any(k > 16 for k, _ in bt.hift_graphs.replays), bt.hift_graphs.replays
    for st in sts:
        bt.release_stream(st)
    del bt
    torch.cuda.empty_cache()
    return got


@pytest.mark.parametrize("hift_rows,batch_setup", [
    (None, False), (EXTENDED_ROWS, False),
    # B24: with the batched setup encoder (the r11c candidate's combination) one of 190 chunks reaches 1.062e-2
    # (results/engine_profile/r14_hift_matched.json); recorded as a strict expected failure until repaired
    pytest.param(None, True, marks=pytest.mark.xfail(strict=True, reason="B24: TF32 HiFT + batched setup exceeds the AC-3.4 chunk bound")),
], ids=["rows_to_16", "rows_to_32", "rows_to_16_batched_setup"])
@torch.inference_mode()
def test_tf32_hift_decoder_within_the_hift_bound(hift_rows, batch_setup):
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.token2wav_chunker import StreamChunker

    torch.backends.cudnn.allow_tf32 = False
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(1)[:N_UTT])
    served = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    feats = [served.features(cr.ref_audio_bytes()) for cr in reqs]
    plans = []
    for j in range(N_UTT):
        speech = [int(x) for x in feats[(j + 1) % N_UTT][0][0].tolist()]  # another recording's speech tokens
        ch = StreamChunker(first_chunk=FIRST_CHUNK)
        ws = [(w.tokens, w.last) for w in ch.push(speech)]
        plans.append(ws + [(lambda w: (w.tokens, w.last))(ch.finish())])
    ref = _serve(served, feats, plans, tf32=False, batch_setup=batch_setup)
    again = _serve(served, feats, plans, tf32=False, batch_setup=batch_setup)
    new = _serve(served, feats, plans, tf32=True, hift_rows=hift_rows, batch_setup=batch_setup)
    chunk_errs, utt_errs, control = [], [], []
    rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))  # noqa: E731
    for j in range(N_UTT):
        assert len(new[j]) == len(ref[j]) == len(again[j])
        for k, (x, y, z) in enumerate(zip(new[j], ref[j], again[j])):
            assert x.shape == y.shape == z.shape
            if np.linalg.norm(y) > 0:
                chunk_errs.append((rel(x, y), j, k))
                control.append(rel(z, y))
        utt_errs.append((rel(np.concatenate(new[j]), np.concatenate(ref[j])), j))
    worst_c, worst_u = max(chunk_errs), max(utt_errs)
    print(f"HiFT TF32 decoder (graph rows {hift_rows or 'default'}) vs strict FP32: {len(chunk_errs)} chunks, max rel L2 {worst_c[0]:.2e} (utt {worst_c[1]}, chunk {worst_c[2]}), "
          f"median {float(np.median([e[0] for e in chunk_errs])):.2e}; utterances max {worst_u[0]:.2e} (utt {worst_u[1]}); "
          f"strict-FP32 run to run max {max(control):.2e}")
    assert max(control) <= 1e-4  # two strict-FP32 runs agree far inside the bound: the difference measured is the precision
    assert worst_c[0] <= BOUND and worst_u[0] <= BOUND, (worst_c, worst_u)
