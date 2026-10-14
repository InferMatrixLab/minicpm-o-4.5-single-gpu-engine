"""Speech report verification, qualification, Q_ref freeze and gates (B8, B10)."""

import hashlib
import json
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.semantics import SemanticsUnknown, class_r  # noqa: E402
from bench.workload import load_seed_tts_rows, make_id_sets  # noqa: E402
from evals.speech.detectors import RowQuality  # noqa: E402
from evals.speech.qref import QRefError, freeze, gate, qualify, verify_qualification  # noqa: E402
from evals.speech.verify_report import verify_report  # noqa: E402

EFF = ROOT / "results" / "effective_inputs"
SETS = make_id_sets(load_seed_tts_rows())
SGL = {"adapter": "sglang-omni", "config": "speech", "extra": [], "env": [], "mps": "off", "deploy_files_sha256": ["x"], "digest": "D1",
       "candidate_id": "sglang-omni:speech:1", "label": ""}
EV = {"x": 1}
EV["digest"] = hashlib.sha256(json.dumps({"x": 1}, sort_keys=True).encode()).hexdigest()

pytestmark = pytest.mark.skipif(not (EFF / "canonical.jsonl").exists(), reason="effective-input dumps not available")


def _row(rid, seed, utmos=4.3, trunc=False):
    return RowQuality(row_id=rid, seed_index=seed, target_text="t", thinker_text="t", hyp_text="h", ref_norm="a b", hyp_norm="a b", wer=0.0,
                      substitutions=0, deletions=0, insertions=0, hits=20, sim=60.0, utmos=utmos, duration_s=3.0, d_ref_s=3.0 if trunc is not None else None,
                      leading_silence_s=0.1, trailing_silence_s=0.1, max_internal_silence_s=0.1, silent_fraction=0.05, catastrophic=False,
                      silence=False, repetition=False, truncation=trunc, thinker_exact_match=True)


def report(tmp_path, name, *, seeds=(0, 1, 2), ids=None, id_set="tuning", scope="development", cand=None, utmos=4.3, gpu="L20X", c=8,
           evaluator=None, truncation=False, claim=None):
    ids = ids if ids is not None else list(SETS.tuning(8))
    q = [_row(rid, s, utmos=utmos, trunc=truncation) for s in seeds for rid in ids]
    path = tmp_path / f"{name}.json"
    side = path.with_suffix(".rows.jsonl")
    side.write_text("".join(json.dumps(x.to_json()) + "\n" for x in q))
    cand = cand or SGL
    rep = {
        "scope": scope,
        "generation": {"candidate_id": cand["candidate_id"], "candidate_digest": cand["digest"], "candidate": cand, "gpu_sku": gpu,
                       "gpu_driver": "570", "concurrency": c, "id_set": id_set, "measured_ids_sha256": "ids", "dataset": {"repo": "d"},
                       "model": {"revision": "m"}},
        "seeds": {str(s): {"cell": f"c{s}", "gpu_uuid": f"GPU-{s}"} for s in seeds},
        "rows_per_seed": claim or {str(s): len(ids) for s in seeds},
        "rows_sha256": hashlib.sha256("\n".join(sorted(f"{s}:{r}" for s in seeds for r in ids)).encode()).hexdigest(),
        "evaluator": evaluator or EV,
        "row_sidecar_sha256": hashlib.sha256(side.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(rep))
    return str(path)


def qual(tmp_path, rep_path):
    rec = qualify(rep_path, str(EFF), check_sources=False)
    p = Path(rep_path).with_suffix(".qual.json")
    p.write_text(json.dumps(rec))
    return str(p), rec


# --- report verification (B8) -------------------------------------------------------
def test_relabelled_counts_and_scope_are_rejected(tmp_path):
    rep = report(tmp_path, "r20", ids=list(SETS.tuning(8))[:20])
    r = json.loads(Path(rep).read_text())
    r["scope"], r["generation"]["id_set"], r["rows_per_seed"] = "final", "full", {"0": 1088, "1": 1088, "2": 1088}
    Path(rep).write_text(json.dumps(r))
    v = verify_report(rep, check_sources=False)
    assert not v.ok and any("rows_per_seed" in p for p in v.problems) and any("canonical" in p for p in v.problems)


def test_duplicates_and_seed_map_inconsistency(tmp_path):
    rep = report(tmp_path, "dup")
    side = Path(rep).with_suffix(".rows.jsonl")
    lines = side.read_text().splitlines()
    side.write_text("\n".join(lines + lines[:1]) + "\n")
    r = json.loads(Path(rep).read_text())
    r["row_sidecar_sha256"] = hashlib.sha256(side.read_bytes()).hexdigest()
    r["seeds"] = {"0": {}, "1": {}, "5": {}}
    Path(rep).write_text(json.dumps(r))
    v = verify_report(rep, check_sources=False)
    assert any("duplicate" in p for p in v.problems) and any("seeds" in p for p in v.problems)


def test_full_scope_needs_the_canonical_set(tmp_path):
    full = list(SETS.full())
    ok = report(tmp_path, "full", ids=full, id_set="full", scope="final", truncation=True)
    assert verify_report(ok, check_sources=False).ok
    fake = [f"x{i}" for i in range(1088)]  # 1088 rows that are not the canonical rows
    bad = report(tmp_path, "fake", ids=fake, id_set="full", scope="final", truncation=True)
    assert not verify_report(bad, check_sources=False).ok


REAL = ROOT / "results" / "quality_v2" / "sglang_speech_c8_tuning" / "speech_quality.json"


@pytest.mark.skipif(not REAL.exists(), reason="real development report not available")
@pytest.mark.parametrize("mutate", [lambda g: g["model"].update(revision="foreign"), lambda g: g["dataset"].update(revision="foreign"),
                                    lambda g: g.update(concurrency=32), lambda g: g.update(gpu_sku="H200"), lambda g: g.update(gpu_driver="0"),
                                    lambda g: g["seeds"]["1"].update(gpu_uuid="GPU-foreign")])
def test_real_report_source_binding(tmp_path, mutate):
    from evals.speech.suite import collect

    dst = tmp_path / "rep.json"
    shutil.copy(REAL.with_suffix(".rows.jsonl"), dst.with_suffix(".rows.jsonl"))
    r = json.loads(REAL.read_text())
    # rebind the round-2 report to the current generation identity (hardware class + per-seed device), from its own source cells
    col = collect([v["cell"] for _, v in sorted(r["seeds"].items())])
    r["generation"], r["seeds"] = col.generation, col.seeds
    dst.write_text(json.dumps(r))
    assert verify_report(dst).ok, verify_report(dst).problems[:3]
    g = {**r["generation"], "seeds": r["seeds"]}
    mutate(g)
    r["seeds"] = g.pop("seeds")
    r["generation"] = g
    dst.write_text(json.dumps(r))
    assert not verify_report(dst).ok


def test_foreign_duration_reference_rejected(tmp_path):
    rep = report(tmp_path, "dr", truncation=True)
    dref = tmp_path / "dref.json"
    dref.write_text(json.dumps({"role": "throughput winner", "candidate_digest": "X", "sources": [], "rows": {"r": 3.0}}))
    r = json.loads(Path(rep).read_text())
    r["d_ref"] = {"path": str(dref), "sha256": hashlib.sha256(dref.read_bytes()).hexdigest()}
    Path(rep).write_text(json.dumps(r))
    v = verify_report(rep, check_sources=False)
    assert any("d_ref" in p for p in v.problems)  # durations not backed by verified source cells


# --- class R and qualification (B10) -----------------------------------------------------
def test_class_r_from_resolved_semantics(tmp_path):
    base = {"adapter": "vllm-omni", "extra": [], "env": [], "mps": "off"}
    assert class_r({**base, "config": "minicpmo_4_5_turn_mrv2.yaml"}, True).is_class_r
    absolute = "/workspace/cgzheng/cgzheng/vllm-omni/vllm_omni/deploy/minicpmo_4_5.yaml"
    assert not class_r({**base, "config": absolute}, True).is_class_r  # default duplex by absolute path
    renamed = tmp_path / "my_turn_profile.yaml"
    shutil.copy(absolute, renamed)
    assert not class_r({**base, "config": str(renamed)}, True).is_class_r  # renamed duplex
    steps = tmp_path / "steps8.yaml"
    steps.write_text(f"base_config: /workspace/cgzheng/cgzheng/vllm-omni/vllm_omni/deploy/minicpmo_4_5_turn_mrv2.yaml\n"
                     "connectors:\n  connector_of_shared_memory:\n    extra:\n      token2wav_n_timesteps: 8\n")
    assert not class_r({**base, "config": str(steps)}, True).is_class_r
    fp16 = tmp_path / "fp16.yaml"
    fp16.write_text(f"base_config: /workspace/cgzheng/cgzheng/vllm-omni/vllm_omni/deploy/minicpmo_4_5_turn_mrv2.yaml\n"
                    "connectors:\n  connector_of_shared_memory:\n    extra:\n      token2wav_float16: true\n")
    assert not class_r({**base, "config": str(fp16)}, True).is_class_r
    with pytest.raises(SemanticsUnknown):
        class_r({**base, "config": "minicpmo_4_5_turn_mrv2.yaml", "extra": ["--quantization", "fp8"]}, True)
    with pytest.raises(SemanticsUnknown):
        class_r({**base, "config": "minicpmo_4_5_turn_mrv2.yaml", "env": [["MINICPMO_CODE2WAV_TF32", "0"]]}, True)
    with pytest.raises(SemanticsUnknown):  # declared MPS must match the launch
        class_r({**base, "config": "minicpmo_4_5_turn_mrv2.yaml", "mps": "on"}, True)


def test_qualification_recomputed_and_stale_records_rejected(tmp_path):
    rep = report(tmp_path, "good")
    qp, rec = qual(tmp_path, rep)
    assert rec["qualified"] and rec["effective_inputs_eligible"]
    assert verify_qualification(qp, check_sources=False)["qualified"]
    for forge in ({"evaluator_digest": "other"}, {"effective_inputs_eligible": False}, {"class_r": {"class_r": False}}, {"gpu_sku": "H200"}):
        q = json.loads(Path(qp).read_text())
        q.update(forge)
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps(q))
        with pytest.raises(QRefError):
            verify_qualification(str(bad), check_sources=False)
    q = json.loads(Path(qp).read_text())
    q["inputs"]["quality"]["sha256"] = "0" * 64
    bad = tmp_path / "stale.json"
    bad.write_text(json.dumps(q))
    with pytest.raises(QRefError):
        verify_qualification(str(bad), check_sources=False)
    duplex = dict(SGL, adapter="vllm-omni", config="minicpmo_4_5.yaml")
    _, rec = qual(tmp_path, report(tmp_path, "duplex", cand=duplex))
    assert not rec["qualified"]


# --- freeze / gate --------------------------------------------------------------------------
def test_freeze_and_gate_scope_checks(tmp_path):
    ref = report(tmp_path, "ref")
    qp, _ = qual(tmp_path, ref)
    with pytest.raises(QRefError):
        freeze("development", "H200", [(ref, qp)], tmp_path / "q_h200.json", check_sources=False)
    freeze("development", "L20X", [(ref, qp)], tmp_path / "qref.json", check_sources=False)
    with pytest.raises(QRefError):
        freeze("development", "L20X", [(ref, qp)], tmp_path / "qref.json", check_sources=False)
    qref = str(tmp_path / "qref.json")
    assert gate(qref, report(tmp_path, "eng"), None, check_sources=False)["overall"] in ("pass", "inconclusive")
    for kw in ({"c": 32}, {"gpu": "H200"}, {"seeds": (0, 1, 3)}, {"scope": "final"}, {"ids": list(SETS.tuning(8))[:300]}):
        with pytest.raises(QRefError):
            gate(qref, report(tmp_path, f"e{list(kw)[0]}", **kw), None, check_sources=False)
    one = report(tmp_path, "one", seeds=(0,), ids=list(SETS.tuning(8))[:1])
    qo, _ = qual(tmp_path, one)
    with pytest.raises(QRefError):
        freeze("development", "L20X", [(one, qo)], tmp_path / "q1.json", check_sources=False)
    other_eval = report(tmp_path, "oe", evaluator={**EV, "x": 2, "digest": EV["digest"]})
    with pytest.raises(QRefError):  # evaluator record no longer matches its digest
        freeze("development", "L20X", [(other_eval, qp)], tmp_path / "q2.json", check_sources=False)


def test_final_gate_needs_headline_and_truncation(tmp_path):
    full = list(SETS.full())
    qp, _ = qual(tmp_path, report(tmp_path, "tq"))
    ref = report(tmp_path, "fref", ids=full, id_set="full", scope="final", truncation=True)
    with pytest.raises(QRefError):  # final references need D_ref-backed truncation in the engine report, checked at gate
        freeze("final", "L20X", [(report(tmp_path, "short", ids=full[:20], id_set="full", scope="final"), qp)], tmp_path / "x.json",
               check_sources=False)
    freeze("final", "L20X", [(ref, qp)], tmp_path / "fq.json", check_sources=False)
    eng = report(tmp_path, "feng", ids=full, id_set="full", scope="final", truncation=None)
    with pytest.raises(QRefError):
        gate(str(tmp_path / "fq.json"), eng, SGL["digest"], check_sources=False)
