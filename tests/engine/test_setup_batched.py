"""Batched prompt-setup encoder (``BatchedToken2Wav.batch_setup_encoder``) against the per-prompt reference forward (GPU).

Sixteen Seed-TTS reference voices of 80-158 prompt tokens, in setup calls of 2, 3, 4 and 8 prompts of mixed lengths.
For every prompt the padded batch must give the reference ``forward_chunk``'s encoder output, conv caches (gathered at
the prompt's real end) and attention caches (cut to its real length) with the same shapes and a relative L2 <= 1e-3
(FP16 GEMMs of another shape; measured 3.2e-4), while two per-prompt runs agree bitwise (the control: the difference
measured is the batching).  The served chunks of batched setups are checked against the FP32 reference in
``test_ac34_fp32.py`` (``setup_batched``).
"""

import os

import pytest
import torch

from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH
pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or os.environ.get("ENGINE_GPU_TESTS") != "1",
                                reason="GPU test (set ENGINE_GPU_TESTS=1 on a free GPU)")
BOUND = 1e-3


@torch.inference_mode()
def test_batched_setup_encoder_matches_per_prompt():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav, strict_fp32
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    torch.backends.cudnn.allow_tf32 = False
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(32)[:16])
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True)
    bt = BatchedToken2Wav(t2w, fast_dit=True)
    feats = [t2w.features(cr.ref_audio_bytes()) for cr in reqs]
    assert len({f[0].shape[1] for f in feats}) > 8  # mixed lengths
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm())  # noqa: E731
    worst, control = 0.0, 0.0
    with strict_fp32(), torch.amp.autocast("cuda", dtype=torch.float16):
        for g in (2, 3, 4, 8):
            for a in range(0, len(feats) - g + 1, g):
                part = feats[a:a + g]
                one = [bt._setup_encoder_one(f) for f in part]
                again = [bt._setup_encoder_one(f) for f in part]
                for x, y, z in zip(bt._setup_encoder_batched(part), one, again):
                    for i in range(3):
                        assert x[i].shape == y[i].shape, (g, i, x[i].shape, y[i].shape)
                        worst = max(worst, rel(x[i], y[i]))
                        control = max(control, rel(z[i], y[i]))
    print(f"batched setup encoder vs per prompt: max rel L2 {worst:.2e}; per-prompt run to run {control:.1e}")
    assert control == 0.0
    assert worst <= BOUND
