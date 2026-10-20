"""Batched vs per-prompt Token2Wav prompt-setup encoder (GPU, diagnostic): relative L2 of the encoder output, conv and
attention caches per prompt, a per-prompt run-to-run control, and setup_many wall time per bucket (1/2/4 prompts).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/setup_batch_probe.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH


@torch.inference_mode()
def main():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav, strict_fp32
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    torch.backends.cudnn.allow_tf32 = False
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True, spk_threads=4)
    bt = BatchedToken2Wav(t2w, use_graphs=True, compile_estimator=True, fast_dit=True, fast_slots=65536, hift_decode_tf32=True)
    p0 = bt.setup(t2w.features(MODEL / "assets" / "HT_ref_audio.wav"))
    bt.capture_graphs(p0)
    bt.release_prompt(p0)
    rows = load_seed_tts_rows()
    reqs = canonical_requests(rows, make_id_sets(rows).tuning(32)[:16])
    feats = [t2w.features(cr.ref_audio_bytes()) for cr in reqs]
    print("prompt tokens", [f[0].shape[1] for f in feats], flush=True)
    rel = lambda a, b: float((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12))  # noqa: E731
    worst = {"h": 0.0, "cnn": 0.0, "att": 0.0}
    ctrl = {"h": 0.0, "cnn": 0.0, "att": 0.0}
    with strict_fp32(), torch.amp.autocast("cuda", dtype=bt.dtype):
        for g in (2, 3, 4):
            for a in range(0, len(feats) - g + 1, g):
                part = feats[a:a + g]
                one = [bt._setup_encoder_one(f) for f in part]
                again = [bt._setup_encoder_one(f) for f in part]
                bat = bt._setup_encoder_batched(part)
                for x, y, z in zip(bat, one, again):
                    for k, i in (("h", 0), ("cnn", 1), ("att", 2)):
                        assert x[i].shape == y[i].shape, (k, x[i].shape, y[i].shape)
                        worst[k] = max(worst[k], rel(x[i], y[i]))
                        ctrl[k] = max(ctrl[k], rel(z[i], y[i]))
    print("batched vs per-prompt max rel L2", {k: f"{v:.2e}" for k, v in worst.items()}, " per-prompt run to run", {k: f"{v:.2e}" for k, v in ctrl.items()}, flush=True)
    for mode in (False, True):
        bt.batch_setup_encoder = mode
        for n in (1, 2, 4):
            ts = []
            for r in range(6):
                torch.cuda.synchronize()
                t = time.perf_counter()
                ps = bt.setup_many(feats[4 * r % 12:4 * r % 12 + n])
                torch.cuda.synchronize()
                ts.append(1e3 * (time.perf_counter() - t))
                for p in ps:
                    bt.release_prompt(p)
            print(f"batch_setup_encoder={mode} setup_many n={n}: ms {[round(x, 1) for x in ts[1:]]}", flush=True)


if __name__ == "__main__":
    main()
