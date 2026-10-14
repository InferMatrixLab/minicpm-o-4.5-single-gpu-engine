"""Q_ref qualification/freeze/gate scoping (B8)."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.speech.detectors import RowQuality  # noqa: E402
from evals.speech.qref import QRefError, freeze, gate, qualify  # noqa: E402


def _row(i, seed, wer=0.0, utmos=4.3, trunc=False):
    return RowQuality(row_id=f"r{i}", seed_index=seed, target_text="t", thinker_text="t", hyp_text="h", ref_norm="a b", hyp_norm="a b",
                      wer=wer, substitutions=0, deletions=0, insertions=0, hits=20, sim=60.0, utmos=utmos, duration_s=3.0, d_ref_s=3.0,
                      leading_silence_s=0.1, trailing_silence_s=0.1, max_internal_silence_s=0.1, silent_fraction=0.05, catastrophic=False,
                      silence=False, repetition=False, truncation=trunc, thinker_exact_match=True)


def report(tmp_path, name, *, seeds=(0, 1, 2), rows=20, gpu="L20X", c=8, id_set="tuning", scope="development", evaluator="E1",
           cand_digest="D1", cand=None, utmos=4.3, truncation=True):
    q = [_row(i, s, utmos=utmos, trunc=False if truncation else None) for s in seeds for i in range(rows)]
    path = tmp_path / f"{name}.json"
    side = path.with_suffix(".rows.jsonl")
    side.write_text("".join(json.dumps(x.to_json()) + "\n" for x in q))
    rep = {
        "scope": scope,
        "generation": {"candidate_id": f"{name}:cfg:1", "candidate_digest": cand_digest, "gpu_sku": gpu, "gpu_uuid": "GPU-x", "concurrency": c,
                       "id_set": id_set, "measured_ids_sha256": f"ids-{rows}",
                       "candidate": cand or {"adapter": "sglang-omni", "config": "speech", "mps": "off", "label": ""}},
        "seeds": {str(s): {"cell": f"c{s}"} for s in seeds},
        "rows_per_seed": {str(s): rows for s in seeds},
        "rows_sha256": hashlib.sha256("\n".join(sorted(f"{s}:r{i}" for s in seeds for i in range(rows))).encode()).hexdigest(),
        "evaluator": {"digest": evaluator},
        "truncation_evaluated": truncation,
        "row_sidecar_sha256": hashlib.sha256(side.read_bytes()).hexdigest(),
    }
    path.write_text(json.dumps(rep))
    return str(path)


def effective(tmp_path, eligible=True):
    p = tmp_path / "eff.json"
    p.write_text(json.dumps({"sglang-omni": {"eligible": eligible}, "vllm-omni": {"eligible": eligible}}))
    return str(p)


def qual(tmp_path, rep_path, eligible=True):
    rec = qualify(rep_path, effective(tmp_path, eligible))
    p = Path(rep_path).with_suffix(".qual.json")
    p.write_text(json.dumps(rec))
    return str(p), rec


def test_qualify_requires_floors_class_r_and_effective_inputs(tmp_path):
    _, rec = qual(tmp_path, report(tmp_path, "good"))
    assert rec["qualified"]
    _, rec = qual(tmp_path, report(tmp_path, "lowmos", utmos=4.0))
    assert not rec["qualified"]
    _, rec = qual(tmp_path, report(tmp_path, "inel"), eligible=False)
    assert not rec["qualified"]
    duplex = report(tmp_path, "dup", cand={"adapter": "vllm-omni", "config": "minicpmo_4_5.yaml", "mps": "off", "label": ""})
    _, rec = qual(tmp_path, duplex)
    assert not rec["qualified"]  # duplex session mode is not class R
    side = Path(report(tmp_path, "tamper")).with_suffix(".rows.jsonl")
    side.write_text(side.read_text().replace('"utmos": 4.3', '"utmos": 4.9'))
    with pytest.raises(QRefError):
        qualify(str(side.with_suffix("").with_suffix(".json")), effective(tmp_path))


def test_wrong_gpu_freeze_rejected(tmp_path):
    rep = report(tmp_path, "a")
    q, _ = qual(tmp_path, rep)
    with pytest.raises(QRefError):
        freeze("development", "H200", [(rep, q)], tmp_path / "qref.json")  # report was generated on L20X
    doc = freeze("development", "L20X", [(rep, q)], tmp_path / "qref.json")
    assert doc["identity"]["gpu_sku"] == "L20X"
    with pytest.raises(QRefError):
        freeze("development", "L20X", [(rep, q)], tmp_path / "qref.json")  # never rewritten


def test_one_seed_and_missing_qualification_rejected(tmp_path):
    one = report(tmp_path, "one", seeds=(0,), rows=1)
    q1, _ = qual(tmp_path, one)
    with pytest.raises(QRefError):
        freeze("development", "L20X", [(one, q1)], tmp_path / "q1.json")
    bad = report(tmp_path, "bad", utmos=4.0)
    qb, _ = qual(tmp_path, bad)
    with pytest.raises(QRefError):
        freeze("development", "L20X", [(bad, qb)], tmp_path / "q2.json")  # no qualified reference
    good = report(tmp_path, "good")
    qg, _ = qual(tmp_path, good)
    with pytest.raises(QRefError):
        freeze("development", "L20X", [(good, qb)], tmp_path / "q3.json")  # qualification of another candidate


def test_gate_rejects_scope_mismatches(tmp_path):
    ref = report(tmp_path, "ref")
    q, _ = qual(tmp_path, ref)
    freeze("development", "L20X", [(ref, q)], tmp_path / "qref.json")
    qp = str(tmp_path / "qref.json")
    assert gate(qp, report(tmp_path, "eng"), None)["overall"] in ("pass", "inconclusive")
    for kw in ({"c": 32}, {"gpu": "H200"}, {"evaluator": "E2"}, {"rows": 19}, {"seeds": (0, 1, 3)}, {"scope": "final"}):
        with pytest.raises(QRefError):
            gate(qp, report(tmp_path, f"eng_{len(kw)}_{list(kw)[0]}", **kw), None)


def test_final_scope_requirements(tmp_path):
    ref = report(tmp_path, "fref", rows=1088, id_set="full", scope="final")
    q, _ = qual(tmp_path, report(tmp_path, "fqual"))
    with pytest.raises(QRefError):
        freeze("final", "L20X", [(report(tmp_path, "short", rows=20, id_set="full", scope="final"), q)], tmp_path / "x.json")
    # the qualification belongs to another candidate digest only if digests differ; same digest here
    freeze("final", "L20X", [(ref, q)], tmp_path / "fq.json")
    fq = str(tmp_path / "fq.json")
    eng = report(tmp_path, "feng", rows=1088, id_set="full", scope="final", cand_digest="HEAD")
    with pytest.raises(QRefError):
        gate(fq, eng, None)  # no declared headline
    with pytest.raises(QRefError):
        gate(fq, eng, "OTHER")
    assert gate(fq, eng, "HEAD")["scope"] == "final"
    notrunc = report(tmp_path, "fnt", rows=1088, id_set="full", scope="final", cand_digest="HEAD", truncation=False)
    with pytest.raises(QRefError):
        gate(fq, notrunc, "HEAD")
