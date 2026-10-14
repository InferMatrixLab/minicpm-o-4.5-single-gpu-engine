import numpy as np
import pytest

from bench.effective_inputs import EffectiveInput, compare, eligible
from bench.eligibility import ConfigClass, EnvelopeEntry, EnvelopeError, envelope_winner, validate_envelope
from bench.stats import StatisticsInputError, bootstrap_ratio, duration_guard, estimate, per_run_values, verdict


def runs(values, metric="throughput_frames_s", prefix="run"):
    return [{"label": f"{prefix}{i + 1}", "measured": True, "valid": True, "n_requests": 320, metric: v} for i, v in enumerate(values)]


def test_pass_rule_and_inconclusive():
    assert verdict(2.3, 2.1) == "pass"
    assert verdict(2.1, 1.9) == "inconclusive"
    assert verdict(1.9, 1.7) == "fail"


def test_estimate_point_is_ratio_of_means_and_orientation():
    new = runs([400, 410, 390, 405, 395])
    base = runs([100, 102, 98, 101, 99])
    e = estimate(new, base, "throughput_frames_s", replicates=2000)
    assert e.point == pytest.approx(400 / 100)
    assert e.lcb95 > 3.8 and e.verdict == "pass"
    # latency: best / new
    new_t = runs([0.1, 0.11, 0.09], metric="ttfa_p50_s")
    base_t = runs([0.5, 0.5, 0.5], metric="ttfa_p50_s")
    et = estimate(new_t, base_t, "ttfa_p50_s", replicates=2000)
    assert et.point == pytest.approx(0.5 / 0.1)


def test_noisy_ratio_is_inconclusive():
    new = runs([180.0, 230.0, 200.0, 260.0, 170.0])  # mean 208
    base = runs([100.0] * 5)
    e = estimate(new, base, "throughput_frames_s", replicates=4000)
    assert e.point == pytest.approx(2.08)
    assert e.lcb95 < 2.0
    assert e.verdict == "inconclusive"
    assert e.interval_too_wide


def test_request_level_bootstrap_rejected():
    req_records = [{"request_id": f"r{i}", "ttfa_s": 0.1} for i in range(320)]
    with pytest.raises(StatisticsInputError):
        per_run_values(req_records, "ttfa_s")
    bad = runs([1.0, 2.0])
    bad[0]["request_id"] = "x"
    with pytest.raises(StatisticsInputError):
        per_run_values(bad, "throughput_frames_s")


def test_invalid_or_discarded_runs_rejected():
    r = runs([1.0, 2.0])
    r[1]["valid"] = False
    with pytest.raises(StatisticsInputError):
        per_run_values(r, "throughput_frames_s")
    r = runs([1.0, 2.0])
    r[0]["measured"] = False
    with pytest.raises(StatisticsInputError):
        per_run_values(r, "throughput_frames_s")


def test_bootstrap_resamples_complete_runs_independently():
    new = np.array([1.0, 2.0, 3.0])
    base = np.array([1.0, 1.0])
    b = bootstrap_ratio(new, base, True, replicates=1000, seed=3)
    # only run-level means can appear: means of 3 values from {1,2,3}
    assert set(np.round(b * 3).astype(int)) <= set(range(3, 10))


def test_duration_guard():
    ref = {f"r{i}": 4.0 for i in range(100)}
    ok = duration_guard({k: 4.1 for k in ref}, ref)
    assert ok.passed
    long = duration_guard({k: 4.0 * 1.08 for k in ref}, ref)  # corpus ratio 1.08 is rejected
    assert not long.passed and long.corpus_ratio == pytest.approx(1.08)
    tail = {k: 4.0 for k in ref}
    for i in range(3):
        tail[f"r{i}"] = 8.0
    for i in range(3, 6):
        tail[f"r{i}"] = 0.5
    g = duration_guard(tail, ref)
    assert not g.passed and g.tail_fraction == pytest.approx(0.06)
    sil = duration_guard(ref, ref, {k: 0.10 for k in ref}, {k: 0.07 for k in ref})
    assert not sil.passed


def cls(**kw):
    base = dict(
        engine="vllm-omni",
        config="minicpmo_4_5_turn_mrv2.yaml",
        track="T",
        audio_output="stream",
        chunk_tokens=25,
        flow_precision="tf32",
        thinker_precision="bf16",
        euler_steps=10,
        session_mode="turn",
        canonical_inputs=True,
    )
    base.update(kw)
    return ConfigClass(**base)


def test_class_r():
    assert cls().is_class_r
    assert ConfigClass(**{**cls().__dict__, "engine": "sglang-omni", "flow_precision": "fp16", "audio_output": "full", "chunk_tokens": None}).is_class_r
    assert not cls(track="S").is_class_r
    assert not cls(session_mode="duplex").is_class_r
    assert not cls(canonical_inputs=False).is_class_r
    assert not cls(euler_steps=8).is_class_r
    assert not cls(chunk_tokens=10).is_class_r
    assert not cls(flow_precision="fp16").is_class_r  # vllm-omni ships tf32


def test_envelope_validator_rejects_ineligible_and_wrong_hardware():
    good = EnvelopeEntry("H200", 8, "throughput_frames_s", cls(), 1000.0, qualified=True)
    validate_envelope([good], "H200", require_final=True)
    for bad in (cls(track="S"), cls(session_mode="duplex"), cls(canonical_inputs=False)):
        with pytest.raises(EnvelopeError):
            validate_envelope([EnvelopeEntry("H200", 8, "throughput_frames_s", bad, 1.0, qualified=True)], "H200")
    l20 = EnvelopeEntry("L20X", 8, "throughput_frames_s", cls(), 1000.0, qualified=True, provisional=True)
    with pytest.raises(EnvelopeError):
        validate_envelope([l20], "H200", require_final=True)
    with pytest.raises(EnvelopeError):
        validate_envelope([l20], "L20X", require_final=True)
    validate_envelope([l20], "L20X")
    with pytest.raises(EnvelopeError):
        validate_envelope([EnvelopeEntry("H200", 8, "x", cls(), 1.0, qualified=False)], "H200")
    a = EnvelopeEntry("H200", 8, "ttfa_p50_s", cls(), 0.5, qualified=True)
    b = EnvelopeEntry("H200", 8, "ttfa_p50_s", cls(config="b"), 0.4, qualified=True)
    assert envelope_winner([a, b], higher_is_better=False) is b


def _ei(ids=(1, 2, 3, 151644), src="s", wav="h", toks=(5, 6), spk=(1.0, 2.0)):
    return EffectiveInput("r1", list(ids), ref_source_sha256=src, ref_waveform_hash=wav, ref_waveform_sample_rate=16000, speech_tokens=list(toks), speaker_embedding=list(spk))


def test_non_canonical_prompt_ids_rejected():
    canon = _ei()
    assert compare(_ei(spk=(1.0, 2.0005)), canon) == []
    mm = compare(_ei(ids=(1, 2, 4, 151644)), canon)
    assert [m.field for m in mm] == ["prompt_token_ids"] and mm[0].hard
    assert not eligible(mm)
    mm = compare(_ei(src="other"), canon)
    assert [m.field for m in mm] == ["ref_source_sha256"] and not eligible(mm)


def test_baseline_reference_preprocessing_is_documented_not_disqualifying():
    canon = _ei()
    cand = _ei(wav="h2", toks=(5, 7, 8), spk=(1.0, 2.1))
    mm = compare(cand, canon)
    assert {m.field for m in mm} == {"ref_waveform_hash", "speech_tokens", "speaker_embedding"}
    assert eligible(mm) and not any(m.hard for m in mm)
    # the new engine is held to the HF reference path exactly
    strict = compare(cand, canon, strict_reference=True)
    assert not eligible(strict)
    assert eligible(compare(_ei(spk=(1.0, 2.0005)), canon, strict_reference=True))
