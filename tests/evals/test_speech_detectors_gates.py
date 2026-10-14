import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.speech.detectors import RowQuality, corpus_metrics, repetition_flag, silence_profile, truncation_flag, word_recall  # noqa: E402
from evals.speech.gates import FLOOR_FILTERED_WER, FLOOR_SIM, FLOOR_UTMOS, QRef, absolute_floors, comparative_gates, freeze_qref  # noqa: E402

SR = 24000


def tone(sec, amp=0.3):
    t = np.arange(int(sec * SR)) / SR
    return (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def zeros(sec):
    return np.zeros(int(sec * SR), dtype=np.float32)


def test_silence_profile_edges_and_internal():
    x = np.concatenate([zeros(0.5), tone(1.0), zeros(2.5), tone(1.0), zeros(1.2)])
    p = silence_profile(x)
    assert p.leading_s == pytest.approx(0.5, abs=0.021)
    assert p.trailing_s == pytest.approx(1.2, abs=0.021)
    assert p.max_internal_s == pytest.approx(2.5, abs=0.041)
    assert p.flagged
    clean = silence_profile(np.concatenate([zeros(0.3), tone(2.0), zeros(1.9), tone(1.0), zeros(0.4)]))
    assert not clean.flagged  # 1.9 s internal and <1 s edges are allowed
    assert silence_profile(np.concatenate([zeros(1.1), tone(1.0)])).flagged
    assert silence_profile(zeros(1.0)).silent_fraction == 1.0


def test_repetition_requires_three_consecutive_trigrams_absent_from_target():
    hyp = "the cat sat the cat sat the cat sat on the mat"
    assert repetition_flag(hyp, "the cat sat on the mat")
    assert not repetition_flag("the cat sat the cat sat on the mat", "the cat sat on the mat")  # only twice
    assert not repetition_flag(hyp, hyp)  # repetition present in the target itself
    assert not repetition_flag("a b c d e f g h i", "a b c d e f g h i")


def test_truncation_needs_low_recall_and_short_duration():
    assert word_recall(7, 1, 2) == pytest.approx(0.7)
    assert truncation_flag(0.7, 2.0, 4.0) is True
    assert truncation_flag(0.7, 3.0, 4.0) is False  # duration not short enough
    assert truncation_flag(0.9, 1.0, 4.0) is False  # recall fine
    assert truncation_flag(0.5, 1.0, None) is None


def row(i, wer=0.0, s=0, d=0, ins=0, h=10, sim=60.0, ut=4.3, sil=False, rep=False, trunc=False, seed=0, exact=True):
    return RowQuality(
        row_id=f"r{i}", seed_index=seed, target_text="t", thinker_text="t" if exact else "x", hyp_text="h", ref_norm="r", hyp_norm="h",
        wer=wer, substitutions=s, deletions=d, insertions=ins, hits=h, sim=sim, utmos=ut, duration_s=3.0, d_ref_s=3.0,
        leading_silence_s=0.1, trailing_silence_s=0.1, max_internal_silence_s=0.2, silent_fraction=0.05, catastrophic=wer > 0.5,
        silence=sil, repetition=rep, truncation=trunc, thinker_exact_match=exact,
    )


def test_corpus_metrics_and_filtered_wer():
    rows = [row(i, s=0, h=10) for i in range(98)] + [row(98, wer=0.1, s=1, h=9), row(99, wer=0.8, s=8, h=2)]
    m = corpus_metrics(rows)
    assert m["rows"] == 100
    assert m["wer_corpus"] == pytest.approx(9 / 1000)
    assert m["wer_filtered_corpus"] == pytest.approx(1 / 990)
    assert m["catastrophic_count"] == 1
    with pytest.raises(ValueError):
        corpus_metrics([])


def test_absolute_floors():
    ok = {"wer_filtered_corpus": 0.02, "sim_mean": 60.0, "utmos_mean": 4.3}
    assert absolute_floors(ok)["passed"]
    assert FLOOR_FILTERED_WER == pytest.approx(0.026875) and FLOOR_SIM == pytest.approx(41.58099) and FLOOR_UTMOS == pytest.approx(4.163434)
    for k, v in (("wer_filtered_corpus", 0.03), ("sim_mean", 41.0), ("utmos_mean", 4.1)):
        assert not absolute_floors({**ok, k: v})["passed"]


def _seed_metrics(wer=0.02, cat=1, sim=60.0, ut=4.3, sil=0.01, rep=0.0, trunc=0.0):
    return {"wer_corpus": wer, "catastrophic_count": cat, "sim_mean": sim, "utmos_mean": ut, "silence_rate": sil, "repetition_rate": rep,
            "truncation_rate": trunc}


def _qref(**over):
    per = [_seed_metrics(**over), _seed_metrics(**over), _seed_metrics(**over)]
    return freeze_qref({"base:a": {"qualified": True, "rows_sha256": "R", "seed_indices": (0, 1, 2), "per_seed": per}}, "L20X", 0.0)


def test_comparative_gates_pass_fail_inconclusive():
    q = _qref()
    same = [_seed_metrics()] * 3
    assert comparative_gates(same, "R", (0, 1, 2), q)["overall"] == "pass"
    worse_sim = [_seed_metrics(sim=58.5)] * 3
    r = comparative_gates(worse_sim, "R", (0, 1, 2), q)
    assert r["overall"] == "fail" and r["metrics"]["sim_mean"]["verdict"] == "fail"
    noisy = [_seed_metrics(ut=4.25), _seed_metrics(ut=4.35), _seed_metrics(ut=4.31)]  # 0.1 spread > 0.05 margin
    r = comparative_gates(noisy, "R", (0, 1, 2), q)
    assert r["metrics"]["utmos_mean"]["verdict"] == "inconclusive" and r["overall"] == "inconclusive"
    with pytest.raises(ValueError):
        comparative_gates(same, "other-rows", (0, 1, 2), q)
    with pytest.raises(ValueError):
        comparative_gates(same, "R", (0, 1), q)


def test_qref_takes_best_qualified_value_per_metric():
    per_a = [_seed_metrics(wer=0.02, sim=60.0)] * 3
    per_b = [_seed_metrics(wer=0.03, sim=62.0)] * 3
    per_c = [_seed_metrics(wer=0.001, sim=70.0)] * 3  # unqualified: ignored
    q = freeze_qref({"a": {"qualified": True, "rows_sha256": "R", "seed_indices": (0, 1, 2), "per_seed": per_a},
                     "b": {"qualified": True, "rows_sha256": "R", "seed_indices": (0, 1, 2), "per_seed": per_b},
                     "c": {"qualified": False, "rows_sha256": "R", "seed_indices": (0, 1, 2), "per_seed": per_c}}, "L20X", 0.0)
    assert q.values["wer_corpus"] == pytest.approx(0.02) and q.sources["wer_corpus"] == "a"
    assert q.values["sim_mean"] == pytest.approx(62.0) and q.sources["sim_mean"] == "b"
    with pytest.raises(ValueError):
        freeze_qref({"a": {"qualified": False, "rows_sha256": "R", "seed_indices": (0,), "per_seed": per_a}}, "L20X", 0.0)
