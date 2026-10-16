"""Does a feature process's memory grow with the number of distinct input lengths, and is that growth bounded?

    .venv-engine/bin/python tools/spike/feature_shape_cache_probe.py [n_lengths] [component]

Runs ``n_lengths`` synthetic voices of distinct lengths twice (component: frontend | full; full needs a GPU) and prints
RSS during each pass.  A second pass flat at the first pass's end means the growth is per-length caching; a first pass
that keeps rising linearly with no cap means unbounded per-length state.
"""

import io
import os
import sys
import tempfile
import wave
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
MODEL = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"


def voice(n_samples, sr=24000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples) / sr
    x = 0.3 * np.sin(2 * np.pi * 180 * t) + 0.02 * rng.standard_normal(n_samples)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())
    return buf.getvalue()


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 800
    comp = sys.argv[2] if len(sys.argv) > 2 else "frontend"
    from engine.runtime import t2w_worker as W

    W._feat_init(MODEL, 4, comp == "full")
    lengths = [int(24000 * 2.0) + 37 * i for i in range(n)]  # 2.0 s upwards, every length distinct
    voices = [voice(L, seed=i) for i, L in enumerate(lengths)]

    def job(v):
        if comp == "full":
            W._feat_job(v)
            return
        import librosa
        import soundfile as sf
        import torch

        wav16, _ = librosa.load(io.BytesIO(v), sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, torch.inference_mode():
            sf.write(f.name, wav16, 16000)
            front = W.prompt_front_end(W._FEAT, f.name)
            front[2].result()
        W._trim_heap()

    for v in voices[:10]:
        job(v)
    r0 = W._rss_kib(os.getpid())
    for p in (1, 2):
        for i, v in enumerate(voices):
            job(v)
            if (i + 1) % (n // 8) == 0:
                print(f"pass {p} {i + 1}/{n} rss_mb {W._rss_kib(os.getpid()) / 1024:.0f} delta_mb {(W._rss_kib(os.getpid()) - r0) / 1024:+.1f}", flush=True)


if __name__ == "__main__":
    main()
