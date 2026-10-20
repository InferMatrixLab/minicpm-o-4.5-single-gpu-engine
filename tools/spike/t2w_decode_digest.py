"""A fixed Token2Wav workload through the headline ``BatchedToken2Wav`` and a SHA-256 of every output (GPU, diagnostic):
run it in two source trees to show that a change is bitwise exact.

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/spike/t2w_decode_digest.py [--batch-setup] > digest.txt

Eight tuning voices set up in calls of 1, 3 and 4 prompts, then staggered streams of random codes cut by the engine
chunker (first-chunk ramp 8, steady windows, merged finals), every window of a tick in one decode call; prints the
digest of each setup's caches, of every decoded waveform and of all of them together.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path.cwd()))
from bench.paths import MODEL_PATH  # noqa: E402
MODEL = MODEL_PATH


@torch.inference_mode()
def main():
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.models.token2wav import Token2Wav
    from engine.runtime.token2wav_batch import BatchedToken2Wav
    from engine.token2wav_chunker import StreamChunker

    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-setup", action="store_true")
    ap.add_argument("--dump", default=None, help="also save every decoded window (npz) for a numerical comparison")
    a = ap.parse_args()
    torch.backends.cudnn.allow_tf32 = False
    t2w = Token2Wav(MODEL / "assets" / "token2wav", float16=True, noise_seed=7)
    kw = {"batch_setup_encoder": True} if a.batch_setup else {}
    bt = BatchedToken2Wav(t2w, use_graphs=True, compile_estimator=True, fast_dit=True, fast_slots=65536, hift_decode_tf32=True, **kw)
    p0 = bt.setup(t2w.features(MODEL / "assets" / "HT_ref_audio.wav"))
    bt.capture_graphs(p0)
    bt.release_prompt(p0)
    rows = load_seed_tts_rows()
    feats = [t2w.features(cr.ref_audio_bytes()) for cr in canonical_requests(rows, make_id_sets(rows).tuning(32)[:8])]
    total = hashlib.sha256()
    prompts = []
    for a0, b0 in ((0, 1), (1, 4), (4, 8)):
        for p in bt.setup_many(feats[a0:b0]):
            h = hashlib.sha256()
            for k in ("conformer_cnn_cache", "conformer_att_cache"):
                h.update(p.flow_cache[k].float().cpu().numpy().tobytes())
            print("setup", len(prompts), h.hexdigest()[:16])
            total.update(h.digest())
            prompts.append(p)
    gen = np.random.default_rng(11)
    plans = []
    for _ in prompts:
        ch = StreamChunker(first_chunk=8)
        ws = [(w.tokens, w.last) for w in ch.push([int(x) for x in gen.integers(0, 6561, int(gen.integers(60, 140)))])]
        plans.append(ws + [(lambda w: (w.tokens, w.last))(ch.finish())])
    sts = [bt.new_stream(p, noise_seed=j, adopt=True) for j, p in enumerate(prompts)]
    starts = [j % 3 for j in range(len(sts))]
    n, wavs = 0, []
    for tick in range(max(s + len(p) for s, p in zip(starts, plans))):
        live = [j for j, s in enumerate(starts) if s <= tick < s + len(plans[j])]
        for j, w in zip(live, bt.decode([(sts[j], *plans[j][tick - starts[j]]) for j in live])):
            total.update(np.ascontiguousarray(w).tobytes())
            wavs.append(np.asarray(w))
            n += 1
    print(f"{n} windows; all outputs sha256 {total.hexdigest()}")
    if a.dump:
        np.savez(a.dump, *wavs)


if __name__ == "__main__":
    main()
