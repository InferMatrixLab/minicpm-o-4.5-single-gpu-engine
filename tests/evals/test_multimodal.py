"""Multimodal suite evidence boundaries (B11, B12, B13) and the calibrated fixture (M1)."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.multimodal import suite as mm  # noqa: E402
from evals.multimodal.sglang_mm_worker import build_payload  # noqa: E402

BENCH_N = {"mmmu": 40, "mmsu": 40, "daily_omni": 40}


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Frozen expected manifests in a temp dir, and a re-scorer that grades a response correct iff it is that item's gold "A:<id>"."""
    exp = tmp_path / "expected"
    exp.mkdir()
    monkeypatch.setattr(mm, "EXPECTED_DIR", exp)
    monkeypatch.setattr(mm, "FULL_COUNTS", BENCH_N)
    for subset in ("full", "dev"):
        for b, n in BENCH_N.items():
            (exp / f"{subset}_{b}.json").write_text(json.dumps({"items": [[f"{b}{i}", f"h{i}"] for i in range(n)]}))

    def fake_worker(bench, job):
        assert job["mode"] == "rescore"
        per = [{"id": k, "is_correct": v == f"A:{k}", "is_success": bool(v), "raw": v} for k, v in job["responses"].items()]
        return {"per_sample": per, "accuracy": sum(p["is_correct"] for p in per) / len(per)}

    monkeypatch.setattr(mm, "_worker", fake_worker)
    return tmp_path


def make_report(d, *, acc=0.9, scope="development", subset="full", benches=mm.BENCHES, mps="off", observed="off", digest="H", n=None):
    d.mkdir(parents=True, exist_ok=True)
    rep = {"scope": scope, "subset": subset, "concurrency": 16, "candidate": {"digest": digest, "candidate_id": f"c:{digest}", "mps": mps},
           "gpu": {"gpu_sku": "L20X", "gpu_uuid": "GPU-1", "driver": "570"}, "evaluator": {"digest": "E"}, "benches": {},
           "server_evidence": {"mps": {"observed": observed}, "gpu": {"gpu_uuid": "GPU-1"}}}
    for b in benches:
        k = n if n is not None else BENCH_N[b]
        good = int(round(acc * k))
        per = [{"id": f"{b}{i}", "is_correct": i < good, "is_success": True, "raw": f"A:{b}{i}" if i < good else "B"} for i in range(k)]
        side = d / f"{b}.per_sample.jsonl"
        side.write_text("".join(json.dumps(p) + "\n" for p in per))
        rep["benches"][b] = {"accuracy": good / k, "per_sample_sha256": hashlib.sha256(side.read_bytes()).hexdigest(),
                             "expected_sha256": hashlib.sha256((mm.EXPECTED_DIR / f"{subset}_{b}.json").read_bytes()).hexdigest()}
    (d / "report.json").write_text(json.dumps(rep))
    return d


def test_verified_report_and_tampering(env):
    d = make_report(env / "r")
    _, derived, problems = mm.verify_mm_report(d)
    assert not problems and all(v["coverage"]["complete"] for v in derived.values())
    rep = json.loads((d / "report.json").read_text())
    rep["benches"]["mmmu"]["accuracy"] = 0.99
    (d / "report.json").write_text(json.dumps(rep))
    assert any("accuracy" in p for p in mm.verify_mm_report(d)[2])
    d2 = make_report(env / "r2")
    (d2 / "mmsu.per_sample.jsonl").unlink()
    assert any("sidecar" in p for p in mm.verify_mm_report(d2)[2])
    d3 = make_report(env / "r3")
    side = d3 / "daily_omni.per_sample.jsonl"
    side.write_text(side.read_text().replace('"raw": "B"', '"raw": "C"', 1))
    assert any("sidecar" in p for p in mm.verify_mm_report(d3)[2])


def test_subset_and_empty_reports_are_not_complete(env):
    d = make_report(env / "sub", n=10)  # a "full" report that only answered 10 samples
    _, derived, _ = mm.verify_mm_report(d)
    assert not derived["mmmu"]["coverage"]["complete"]
    assert not mm.coverage([], [], [])["complete"]
    dup = [{"id": "x0", "is_success": True}, {"id": "x0", "is_success": True}]
    assert not mm.coverage(["x0", "x0"], dup, ["x0", "x0"])["complete"]
    (mm.EXPECTED_DIR / "full_mmmu.json").write_text(json.dumps({"items": [["a", "1"]] * 2}))
    with pytest.raises(mm.EvidenceError):
        mm.load_expected("full", "mmmu")


def _qual(env, monkeypatch, digest="H", qualified=True):
    rec = {"qualified": qualified, "candidate_digest": digest, "gpu_sku": "L20X", "gpu_driver": "570"}
    import evals.speech.qref as sq

    monkeypatch.setattr(sq, "verify_qualification", lambda p, check_sources=True: rec)
    p = env / f"q_{digest}.json"
    p.write_text(json.dumps(rec))
    return str(p)


def test_freeze_requires_verified_qualified_references(env, monkeypatch):
    rep = make_report(env / "base", scope="final")
    q = _qual(env, monkeypatch)
    args = SimpleNamespace(scope="final", reference=[f"{rep}={q}"], out=str(env / "qref.json"))
    assert mm.cmd_freeze(args) == 0
    with pytest.raises(SystemExit):  # never rewritten
        mm.cmd_freeze(args)
    bad = make_report(env / "bad", scope="final")
    r = json.loads((bad / "report.json").read_text())
    r["benches"]["mmmu"]["accuracy"] = 0.99
    (bad / "report.json").write_text(json.dumps(r))
    with pytest.raises(SystemExit):
        mm.cmd_freeze(SimpleNamespace(scope="final", reference=[f"{bad}={q}"], out=str(env / "q2.json")))
    one = make_report(env / "one", scope="final", benches=("mmmu",))
    with pytest.raises(SystemExit):  # a final reference must cover every benchmark
        mm.cmd_freeze(SimpleNamespace(scope="final", reference=[f"{one}={q}"], out=str(env / "q3.json")))
    unq = _qual(env, monkeypatch, qualified=False)
    with pytest.raises(SystemExit):
        mm.cmd_freeze(SimpleNamespace(scope="final", reference=[f"{rep}={unq}"], out=str(env / "q4.json")))


def test_certification_rules(env, monkeypatch):
    base = make_report(env / "base", scope="final", acc=0.9)
    q = _qual(env, monkeypatch)
    mm.cmd_freeze(SimpleNamespace(scope="final", reference=[f"{base}={q}"], out=str(env / "qref.json")))
    qref = str(env / "qref.json")
    eng = make_report(env / "eng", scope="final", acc=0.9, digest="ENG")
    assert mm.certify(str(eng), qref, "ENG")["certified"]
    assert not mm.certify(str(eng), qref, "OTHER")["certified"]  # wrong headline
    assert not mm.certify(str(eng), qref, None)["certified"]
    low = make_report(env / "low", scope="final", acc=0.85, digest="ENG")
    assert not mm.certify(str(low), qref, "ENG")["certified"]  # more than 1 pp below Q_ref
    one = make_report(env / "eng1", scope="final", benches=("mmmu",), digest="ENG")
    assert not mm.certify(str(one), qref, "ENG")["certified"]
    dev = make_report(env / "dev", scope="development", digest="ENG")
    assert not mm.certify(str(dev), qref, "ENG")["certified"]
    mps = make_report(env / "mps", scope="final", digest="ENG", mps="on", observed="off")
    assert not mm.certify(str(mps), qref, "ENG")["certified"]
    dev_ref = make_report(env / "devbase", scope="development")
    mm.cmd_freeze(SimpleNamespace(scope="development", reference=[f"{dev_ref}={q}"], out=str(env / "qdev.json")))
    assert not mm.certify(str(eng), str(env / "qdev.json"), "ENG")["certified"]  # development reference


def test_calibrated_shuffle_fixture(env):
    d = make_report(env / "clean", acc=0.9)
    args = SimpleNamespace(report=str(d), out=str(env / "cal"))
    assert mm.cmd_calibrate(args) == 0
    cal = json.loads((env / "cal" / "calibration.json").read_text())
    for b, v in cal["benches"].items():
        assert v["selected"]["drop"] >= v["required_drop"] and v["rejected"]
        assert Path(v["selected"]["fixture"]).exists()


def test_request_formats_place_media_before_text(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    mmmu = SimpleNamespace(prompt="Q?", image_data_uris=("data:image/png;base64,AA", "data:image/png;base64,BB"))
    p = build_payload({"bench": "mmmu", "model": "m", "max_tokens": 8, "format": "openai"}, mmmu, None)
    assert [c["type"] for c in p["messages"][0]["content"]] == ["image_url", "image_url", "text"]
    s = build_payload({"bench": "mmmu", "model": "m", "max_tokens": 8, "format": "sglang"}, mmmu, None)
    assert s["images"] == list(mmmu.image_data_uris) and s["messages"][0]["content"] == "Q?"
    p = build_payload({"bench": "mmsu", "model": "m", "max_tokens": 8, "format": "openai"}, SimpleNamespace(audio_path=str(wav)), "Listen")
    assert [c["type"] for c in p["messages"][0]["content"]] == ["input_audio", "text"]


def test_vllm_evaluator_env_drift_rejected(tmp_path, monkeypatch):
    lock = tmp_path / "vllm.lock"
    lock.write_text(mm.VLLM_LOCK.read_text() + "drift==1\n")
    monkeypatch.setattr(mm, "VLLM_LOCK", lock)
    with pytest.raises(mm.EvaluatorMismatch):
        mm.verify_mm_evaluator()


def test_foreign_dataset_revision_refused(tmp_path):
    """The real pinned loader in the eval env refuses a foreign revision and records the consumed revision."""
    import subprocess

    code = ("import sys, os; sys.path.insert(0, os.getcwd()); sys.path.insert(0, %r);"
            "import sglang_mm_worker as w; w._pin_loaders(); import benchmarks.dataset.mmmu as m;"
            "\ntry:\n m.load_dataset('zhaochenyang20/mmmu-ci-50', split='validation', revision='0'*40)\n print('ACCEPTED')\n"
            "except RuntimeError as e:\n print('REFUSED', e)\n"
            "try:\n m.load_dataset('someone/else', split='validation')\n print('ACCEPTED')\nexcept RuntimeError as e:\n print('REFUSED', e)\n"
            ) % str(Path(mm.ROOT) / "evals" / "multimodal")
    out = subprocess.run([mm.EVAL_PYTHON, "-c", code], cwd=str(mm.SGLANG_DIR), env=mm._env(), capture_output=True, text=True).stdout
    assert out.count("REFUSED") == 2, out


def test_daily_omni_payload_matches_native_vllm_requests():
    pytest.importorskip("vllm_omni")
    from transformers import AutoTokenizer

    from evals.multimodal import daily_omni_worker as w
    from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniDataset

    job = {"model": "m", "max_samples": 3, "video_dir": str(mm.DAILY_ROOT / "Videos"), "qa_json": str(mm.DAILY_ROOT / "qa.json"),
           "tokenizer": mm.MODEL_PATH}
    ours = [w.payload(job, r) for r in w.load_requests(job)]
    ds = DailyOmniDataset(qa_json_path=job["qa_json"], video_dir=job["video_dir"], input_mode="all", pack_mode="minicpm-interleave",
                          inline_local_video=True, random_seed=0)
    native = ds.sample(AutoTokenizer.from_pretrained(mm.MODEL_PATH, trust_remote_code=True), 3, output_len=512, no_oversample=True)
    for p, n in zip(ours, native):
        assert p["messages"] == n.omni_chat_messages
        parts = [c["type"] for c in p["messages"][0]["content"] if isinstance(c, dict)]
        assert "image_url" in parts and "audio_url" in parts
        assert p["modalities"] == ["text"] and p["chat_template_kwargs"] == {"enable_thinking": False}
    assert [k for k, _ in w.manifest(job)["items"]] == [f"{r.daily_omni_video_id}#{i}" for i, r in enumerate(native)]
