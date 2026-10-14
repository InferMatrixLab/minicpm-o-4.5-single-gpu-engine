import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.speech.calibrate import INSERT_SILENCE_S, inject  # noqa: E402
from evals.speech.detectors import silence_profile  # noqa: E402
from evals.speech.suite import GenRow  # noqa: E402

SR = 24000


def _rows(tmp_path, n=20):
    rows = []
    for i in range(n):
        t = np.arange(3 * SR) / SR
        x = (0.3 * np.sin(2 * np.pi * (200 + 10 * i) * t)).astype(np.float32)
        p = tmp_path / f"g{i}.wav"
        sf.write(p, x, SR, subtype="PCM_16")
        rows.append(GenRow(f"0:r{i}", f"r{i}", 0, "text", f"/refs/spk{i % 5}.wav", str(p), "text", 3.0))
    return rows


def test_silence_substitution_hits_requested_fraction(tmp_path):
    rows = _rows(tmp_path)
    out = inject("silence_substitution", 0.1, rows, tmp_path / "fx")
    flagged = [r for r in out if silence_profile(sf.read(r.wav, dtype="float32")[0]).flagged]
    assert len(flagged) == 2  # 10% of 20 rows
    assert all(abs(r.duration_s - (3.0 + INSERT_SILENCE_S)) < 1e-3 for r in flagged)
    assert all(r.key.startswith("silence_substitution@0.1|") for r in out)


def test_speaker_swap_changes_reference_only(tmp_path):
    rows = _rows(tmp_path)
    out = inject("speaker_swap", 0.2, rows, tmp_path / "fx")
    changed = [(a, b) for a, b in zip(rows, out) if a.ref_wav != b.ref_wav]
    assert len(changed) == 4
    assert all(a.wav == b.wav for a, b in zip(rows, out))


def test_additive_noise_snr(tmp_path):
    rows = _rows(tmp_path, n=2)
    out = inject("additive_noise", 10.0, rows, tmp_path / "fx")
    clean = sf.read(rows[0].wav, dtype="float32")[0].astype(np.float64)
    noisy = sf.read(out[0].wav, dtype="float32")[0].astype(np.float64)
    snr = 10 * np.log10(np.mean(clean**2) / np.mean((noisy - clean) ** 2))
    assert abs(snr - 10.0) < 0.5
