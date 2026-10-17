"""Multimodal suite evidence boundaries (B11, B12, B13) and the calibrated fixture (M1)."""

import hashlib
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.multimodal import suite as mm  # noqa: E402
from evals.multimodal.sglang_mm_worker import build_payload  # noqa: E402

BENCH_N = {"mmmu": 40, "mmsu": 40, "daily_omni": 40}
EVAL = {"version": 2, "sglang": {"commit": "s"}, "eval_env": "e", "vllm_omni": {"commit": "v"}, "expected": {}, "workers": {}, "floors": {"mmmu": 0.64},
        "max_tokens": {"mmmu": 2048}}
EVAL["digest"] = hashlib.sha256(json.dumps(EVAL, sort_keys=True).encode()).hexdigest()
GPU = {"gpu_index": 1, "gpu_sku": "L20X", "gpu_uuid": "GPU-1", "driver": "570", "memory_mib": "143771"}
CONSUMED: dict = {}


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

    def fake_consumed(subset, bench):
        exp_doc = mm.load_expected(subset, bench)
        if CONSUMED.get((subset, bench), exp_doc["items"]) != exp_doc["items"]:
            raise mm.EvidenceError(f"{bench}/{subset}: consumed data differs from the frozen manifest")
        return exp_doc

    CONSUMED.clear()
    import bench.semantics as sem

    monkeypatch.setattr(sem, "check_deploy_binding", lambda c: None)  # fixture candidates have no deploy files; see test_real_deploy_binding
    monkeypatch.setattr(mm, "_worker", fake_worker)
    monkeypatch.setattr(mm, "check_consumed", fake_consumed)
    monkeypatch.setattr(mm, "verify_mm_evaluator", lambda: EVAL)
    return tmp_path


def candidate(label="base", mps="off"):
    from bench.experiment import Candidate

    c = Candidate.from_spec({"adapter": "vllm-omni", "config": f"{label}.yaml", "label": label, "mps": mps}, ("fake",))
    return c, {**c.identity(), "candidate_id": c.candidate_id, "digest": c.digest, "label": label}


def digest_of(label):
    return candidate(label)[1]["digest"]


def make_report(d, *, acc=0.9, scope="development", subset="full", benches=mm.BENCHES, mps="off", observed="off", label="base", n=None,
                started=None, frozen=None):
    from bench.cli import server_spec

    d.mkdir(parents=True, exist_ok=True)
    c, cd = candidate(label, mps)
    from bench.cli import ADAPTER_PYTHON, DEFAULT_REPOS

    gen = {"candidate": cd, "model": {"repo": "openbmb/MiniCPM-o-4_5", "revision": "r"},
           "fingerprints": {"repos": {"vllm-omni": {"commit": "c", "dirty": False}}}, "adapter_python": ADAPTER_PYTHON["vllm-omni"],
           "repos_checked": sorted(DEFAULT_REPOS), "port": 18101, "hardware": dict(GPU),
           "requests": {b: mm.request_policy("vllm-omni", b, 16) for b in benches}, "evaluator_digest": EVAL["digest"], "frozen_refs": dict(frozen or {})}
    gen["digest"] = hashlib.sha256(json.dumps(gen, sort_keys=True).encode()).hexdigest()
    rep = {"scope": scope, "subset": subset, "concurrency": 16, "candidate": cd, "generation": gen, "gpu": dict(GPU), "evaluator": EVAL,
           "started_unix": started if started is not None else time.time(), "benches": {},
           "server_evidence": {"mps": {"observed": observed}, "gpu": dict(GPU), "launch_argv": server_spec(c, 18101, 1).cmd, "launch_env": {},
                               "process_tree": [10, 11], "gpu_workers": [{"pid": 11}]}}
    for b in benches:
        k = n if n is not None else BENCH_N[b]
        good = int(round(acc * k))
        per = [{"id": f"{b}{i}", "is_correct": i < good, "is_success": True, "raw": f"A:{b}{i}" if i < good else "B", "generation_digest": gen["digest"]}
               for i in range(k)]
        side = d / f"{b}.per_sample.jsonl"
        side.write_text("".join(json.dumps(p) + "\n" for p in per))
        rep["benches"][b] = {"accuracy": good / k, "per_sample_sha256": hashlib.sha256(side.read_bytes()).hexdigest(), "generation_digest": gen["digest"],
                             "expected_sha256": hashlib.sha256((mm.EXPECTED_DIR / f"{subset}_{b}.json").read_bytes()).hexdigest()}
    (d / "report.json").write_text(json.dumps(rep))
    (d / "generation.json").write_text(json.dumps(gen))
    return d


def edit(d, fn):
    rep = json.loads((d / "report.json").read_text())
    fn(rep)
    (d / "report.json").write_text(json.dumps(rep))
    return d


def test_verified_report_and_tampering(env):
    d = make_report(env / "r")
    _, derived, problems = mm.verify_mm_report(d)
    assert not problems and all(v["coverage"]["complete"] for v in derived.values())
    edit(d, lambda r: r["benches"]["mmmu"].update(accuracy=0.99))
    assert any("accuracy" in p for p in mm.verify_mm_report(d)[2])
    d2 = make_report(env / "r2")
    (d2 / "mmsu.per_sample.jsonl").unlink()
    assert any("sidecar" in p for p in mm.verify_mm_report(d2)[2])
    d3 = make_report(env / "r3")
    side = d3 / "daily_omni.per_sample.jsonl"
    side.write_text(side.read_text().replace('"raw": "B"', '"raw": "C"', 1))
    assert any("sidecar" in p for p in mm.verify_mm_report(d3)[2])


@pytest.mark.parametrize("name,fn", [
    ("candidate config", lambda r: (r["candidate"].update(config="other.yaml"), r["generation"]["candidate"].update(config="other.yaml"))),
    ("candidate digest", lambda r: r["candidate"].update(digest="FOREIGN")),
    ("gpu sku", lambda r: r["gpu"].update(gpu_sku="H200")),
    ("gpu driver", lambda r: r["gpu"].update(driver="999")),
    ("gpu memory", lambda r: r["gpu"].update(memory_mib="1")),
    ("evaluator digest", lambda r: r["evaluator"].update(digest="FOREIGN")),
    ("request policy", lambda r: r.update(concurrency=4)),
    ("launch", lambda r: r["server_evidence"].update(launch_argv=["bash", "other.sh"])),
    ("frozen generation", lambda r: r["generation"].update(port=1)),
    ("dirty checkout", lambda r: r["generation"]["fingerprints"]["repos"]["vllm-omni"].update(dirty=True)),
    ("foreign interpreter", lambda r: r["generation"].update(adapter_python="/foreign/python")),
    ("no workers", lambda r: r["server_evidence"].update(gpu_workers=[], process_tree=[])),
    ("consumption record", lambda r: r["benches"]["mmmu"].update(consumed=[{"repo": "foreign"}])),
])
def test_identity_relabelling_rejected(env, name, fn):
    d = make_report(env / name.replace(" ", "_"))
    if name == "consumption record":
        doc = json.loads((mm.EXPECTED_DIR / "full_mmmu.json").read_text())
        doc["consumed"] = [{"repo": "MMMU/MMMU"}]
        (mm.EXPECTED_DIR / "full_mmmu.json").write_text(json.dumps(doc))
        d = make_report(env / "consumption_record2")
        edit(d, lambda r: r["benches"]["mmmu"].update(consumed=[{"repo": "MMMU/MMMU"}]))
        assert not any("consumption" in p for p in mm.verify_mm_report(d)[2])
    if name.startswith("frozen generation"):
        edit(d, fn)  # report and generation.json now disagree
    else:
        edit(d, fn)
        if "generation" in str(fn.__code__.co_names) or name in ("dirty checkout", "foreign interpreter"):
            g = json.loads((d / "report.json").read_text())["generation"]
            body = {k: v for k, v in g.items() if k != "digest"}
            g["digest"] = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
            edit(d, lambda r: r.update(generation=g))
            (d / "generation.json").write_text(json.dumps(g))  # a consistent frozen declaration of the foreign execution
    assert mm.verify_mm_report(d)[2], name


def test_real_deploy_binding(tmp_path):
    from bench.cli import make_candidate
    from bench.semantics import SemanticsUnknown, check_deploy_binding

    c = make_candidate({"adapter": "vllm-omni", "config": "minicpmo_4_5_turn.yaml", "mps": "off"})
    ident = {**c.identity(), "candidate_id": c.candidate_id, "digest": c.digest}
    check_deploy_binding(ident)
    with pytest.raises(SemanticsUnknown):
        check_deploy_binding({**ident, "deploy_files_sha256": ["FOREIGN"]})


def test_rescorer_and_content_must_match_the_recorded_evaluator(env, monkeypatch):
    d = make_report(env / "ev")
    other = {"x": 2}
    other["digest"] = hashlib.sha256(json.dumps({"x": 2}, sort_keys=True).encode()).hexdigest()
    monkeypatch.setattr(mm, "verify_mm_evaluator", lambda: other)
    assert any("evaluator" in p for p in mm.verify_mm_report(d)[2])
    monkeypatch.setattr(mm, "verify_mm_evaluator", lambda: EVAL)
    CONSUMED[("full", "mmmu")] = [[f"mmmu{i}", "changed"] for i in range(BENCH_N["mmmu"])]  # same IDs, different content
    assert any("consumed" in p for p in mm.verify_mm_report(d)[2])


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


def _qual(env, monkeypatch, label="base", qualified=True):
    rec = {"qualified": qualified, "candidate_digest": digest_of(label), "gpu_sku": "L20X", "gpu_driver": "570", "gpu_memory_mib": "143771"}
    import evals.speech.qref as sq

    monkeypatch.setattr(sq, "verify_qualification", lambda p, check_sources=True: json.loads(Path(p).read_text()))
    p = env / f"q_{label}_{qualified}.json"
    p.write_text(json.dumps(rec))
    return str(p)


def freeze(env, refs, name="qref.json", scope="final"):
    args = SimpleNamespace(scope=scope, reference=refs, out=str(env / name))
    assert mm.cmd_freeze(args) == 0
    return str(env / name)


def test_freeze_requires_verified_qualified_full_references(env, monkeypatch):
    rep = make_report(env / "base", scope="final")
    q = _qual(env, monkeypatch)
    qref = freeze(env, [f"{rep}={q}"])
    assert mm.verify_mm_qref(qref)["values"]["mmmu"] == 0.9
    with pytest.raises(SystemExit):  # never rewritten
        mm.cmd_freeze(SimpleNamespace(scope="final", reference=[f"{rep}={q}"], out=qref))
    bad = edit(make_report(env / "bad", scope="final"), lambda r: r["benches"]["mmmu"].update(accuracy=0.99))
    with pytest.raises(SystemExit):
        freeze(env, [f"{bad}={q}"], "q2.json")
    one = make_report(env / "one", scope="final", benches=("mmmu",))
    with pytest.raises(SystemExit):  # a final reference must cover every benchmark
        freeze(env, [f"{one}={q}"], "q3.json")
    with pytest.raises(SystemExit):
        freeze(env, [f"{rep}={_qual(env, monkeypatch, qualified=False)}"], "q4.json")
    dev = make_report(env / "devsub", scope="final", subset="dev")
    with pytest.raises(SystemExit):  # a final Q_ref is frozen from the full sets
        freeze(env, [f"{dev}={q}"], "q5.json")
    with pytest.raises(SystemExit):
        freeze(env, [], "q6.json")


def test_certification_rules(env, monkeypatch):
    base = make_report(env / "base", scope="final", acc=0.9, started=1.0)
    qref = freeze(env, [f"{base}={_qual(env, monkeypatch)}"])
    eng = make_report(env / "eng", scope="final", acc=0.9, label="eng", frozen={"mm_qref": mm._sha(Path(qref))})
    assert mm.certify(str(eng), qref, digest_of("eng"))["certified"]
    assert not mm.certify(str(eng), qref, digest_of("other"))["certified"]  # wrong headline
    assert not mm.certify(str(eng), qref, None)["certified"]
    low = make_report(env / "low", scope="final", acc=0.85, label="eng", frozen={"mm_qref": mm._sha(Path(qref))})
    res = mm.certify(str(low), qref, digest_of("eng"))
    assert not res["certified"] and all("relative" in v["failed"] for v in res["benches"].values())
    one = make_report(env / "eng1", scope="final", benches=("mmmu",), label="eng")
    assert not mm.certify(str(one), qref, digest_of("eng"))["certified"]
    dev = make_report(env / "dev", scope="development", label="eng")
    assert not mm.certify(str(dev), qref, digest_of("eng"))["certified"]
    mps = make_report(env / "mps", scope="final", label="eng", mps="on", observed="off")
    assert not mm.certify(str(mps), qref, digest_of("eng"))["certified"]
    undeclared = make_report(env / "early", scope="final", acc=0.9, label="eng")  # its generation never named this Q_ref
    assert not mm.certify(str(undeclared), qref, digest_of("eng"))["certified"]
    q = json.loads(Path(qref).read_text())
    q["frozen_unix"] = -1  # an editable timestamp is not the binding
    Path(qref).write_text(json.dumps(q))
    assert not mm.certify(str(undeclared), qref, digest_of("eng"))["certified"]
    dev_ref = make_report(env / "devbase", scope="development")
    qdev = freeze(env, [f"{dev_ref}={_qual(env, monkeypatch)}"], "qdev.json", scope="development")
    assert not mm.certify(str(eng), qdev, digest_of("eng"))["certified"]  # development reference


@pytest.mark.parametrize("name,fn", [
    ("lowered values, empty references", lambda q: q.update(values={b: 0.10 for b in q["values"]}, references={})),
    ("lowered values", lambda q: q.update(values={b: 0.85 for b in q["values"]})),
    ("winner", lambda q: q["sources"].update(mmmu="someone")),
    ("identity", lambda q: q["identity"]["mmmu"].update(gpu_sku="H200")),
])
def test_qref_rebuilt_from_its_references(env, monkeypatch, name, fn):
    base = make_report(env / "base", scope="final", acc=0.9, started=1.0)
    qref = freeze(env, [f"{base}={_qual(env, monkeypatch)}"])
    low = make_report(env / "low", scope="final", acc=0.85, label="eng")
    assert not mm.certify(str(low), qref, digest_of("eng"))["certified"]
    q = json.loads(Path(qref).read_text())
    fn(q)
    Path(qref).write_text(json.dumps(q))
    assert not mm.certify(str(low), qref, digest_of("eng"))["certified"], name


def test_reference_sidecars_are_reverified(env, monkeypatch):
    base = make_report(env / "base", scope="final", acc=0.9, started=1.0)
    qref = freeze(env, [f"{base}={_qual(env, monkeypatch)}"])
    eng = make_report(env / "eng", scope="final", acc=0.9, label="eng", frozen={"mm_qref": mm._sha(Path(qref))})
    (base / "mmmu.per_sample.jsonl").unlink()
    assert not mm.certify(str(eng), qref, digest_of("eng"))["certified"]


def test_calibrated_shuffle_fixture(env):
    d = make_report(env / "clean", acc=0.9)
    assert mm.cmd_calibrate(SimpleNamespace(report=str(d), out=str(env / "cal"))) == 0
    cal = json.loads((env / "cal" / "calibration.json").read_text())
    for b, v in cal["benches"].items():
        assert v["selected"]["drop"] >= v["required_drop"] and v["rejected"]
        assert v["clean_control"]["checks"]["relative"] and "relative" in v["selected"]["verdict"]["failed"]
        assert Path(v["selected"]["fixture"], "report.json").exists()


def test_calibration_fails_when_the_production_gate_is_disabled(env, monkeypatch):
    d = make_report(env / "clean", acc=0.9)
    real = mm.compare_bench
    monkeypatch.setattr(mm, "compare_bench", lambda *a: {**real(*a), "passed": True, "failed": [], "checks": {k: True for k in real(*a)["checks"]}})
    assert mm.cmd_calibrate(SimpleNamespace(report=str(d), out=str(env / "cal"))) == 1


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


def test_daily_omni_sglang_form_is_the_native_video_request():
    """sglang-omni gets MiniCPM-o's official omni form: one video_url with its audio, the same question text."""
    pytest.importorskip("vllm_omni")
    from evals.multimodal import daily_omni_worker as w

    job = {"model": "m", "max_samples": 3, "video_dir": str(mm.DAILY_ROOT / "Videos"), "qa_json": str(mm.DAILY_ROOT / "qa.json"),
           "tokenizer": mm.MODEL_PATH}
    for r in w.load_requests(job):
        native = w.payload(job, r)
        sgl = w.payload({**job, "format": "sglang"}, r)
        parts = sgl["messages"][0]["content"]
        assert parts[0]["type"] == "video_url" and parts[0]["video_url"]["use_audio"] is True
        assert Path(parts[0]["video_url"]["url"]).name == f"{r.daily_omni_video_id}_video.mp4" and Path(parts[0]["video_url"]["url"]).exists()
        assert parts[1:] == [c for c in native["messages"][0]["content"] if c["type"] == "text"]
        assert "mm_processor_kwargs" not in sgl and sgl["chat_template_kwargs"] == {"enable_thinking": False}


def test_launch_from_a_worktree_copy_of_the_serve_script(tmp_path):
    """A serve script launched from another checkout (frozen campaign worktree) counts as the declared launch only when
    it is scripts/<name> there with the same bytes; a modified copy or another script does not."""
    from evals.multimodal.suite import same_launch

    here = Path(__file__).resolve().parents[2] / "scripts" / "serve_vllm_omni.sh"
    want = ["bash", str(here), "minicpmo_4_5_turn.yaml", "18620"]
    wt = tmp_path / "wt" / "scripts"
    wt.mkdir(parents=True)
    (wt / "serve_vllm_omni.sh").write_bytes(here.read_bytes())
    assert same_launch(["bash", str(wt / "serve_vllm_omni.sh"), "minicpmo_4_5_turn.yaml", "18620"], want)
    assert not same_launch(["bash", str(wt / "serve_vllm_omni.sh"), "minicpmo_4_5_turn_mrv2.yaml", "18620"], want)
    (wt / "serve_vllm_omni.sh").write_bytes(here.read_bytes() + b"\nexport X=1\n")
    assert not same_launch(["bash", str(wt / "serve_vllm_omni.sh"), "minicpmo_4_5_turn.yaml", "18620"], want)
    (wt / "other.sh").write_bytes(here.read_bytes())
    assert not same_launch(["bash", str(wt / "other.sh"), "minicpmo_4_5_turn.yaml", "18620"], want)


# --- B12d / B15-MM / evaluator v2 (round 9) ----------------------------------------------------------------------------
def test_metadata_only_reference_attachment_is_rejected(env, monkeypatch):
    """The review's B12d probe: a report generated without the Q_ref declaration, the reference then attached to its
    generation metadata (digest recomputed in report.json and generation.json, sidecars untouched)."""
    base = make_report(env / "base", scope="final", acc=0.9, started=1.0)
    qref = freeze(env, [f"{base}={_qual(env, monkeypatch)}"])
    late = make_report(env / "late", scope="final", acc=0.9, label="eng")
    assert not mm.certify(str(late), qref, digest_of("eng"))["certified"]

    def attach(rep):
        g = rep["generation"]
        g["frozen_refs"] = {"mm_qref": mm._sha(Path(qref))}
        g["digest"] = hashlib.sha256(json.dumps({k: v for k, v in g.items() if k != "digest"}, sort_keys=True).encode()).hexdigest()
        for r in rep["benches"].values():
            r["generation_digest"] = g["digest"]  # even the records: the responses still name the original declaration
        (late / "generation.json").write_text(json.dumps(g))

    edit(late, attach)
    res = mm.certify(str(late), qref, digest_of("eng"))
    assert not res["certified"] and any("generation_digest" in p for p in res["problems"]), res["problems"]
    declared = make_report(env / "declared", scope="final", acc=0.9, label="eng", frozen={"mm_qref": mm._sha(Path(qref))})
    assert mm.certify(str(declared), qref, digest_of("eng"))["certified"]


def test_evaluator_v1_records_map_to_their_multimodal_components():
    comp = {k: EVAL[k] for k in mm.MM_COMPONENTS}
    v1 = {"speech_evaluator_digest": "old-speech", **comp}
    v1["digest"] = hashlib.sha256(json.dumps(v1, sort_keys=True).encode()).hexdigest()
    assert mm.mm_evaluator_digest(v1) == mm.mm_evaluator_digest(EVAL) == EVAL["digest"]
    changed = {**v1, "workers": {"w": "x"}}
    changed["digest"] = hashlib.sha256(json.dumps({k: v for k, v in changed.items() if k != "digest"}, sort_keys=True).encode()).hexdigest()
    assert mm.mm_evaluator_digest(changed) not in (None, EVAL["digest"])  # a multimodal component differs: another evaluator
    assert mm.mm_evaluator_digest({**v1, "digest": "forged"}) is None
    assert mm.mm_evaluator_digest({k: v for k, v in v1.items() if k != "expected"}) is None


def _engine_generation(monkeypatch):
    import bench.cli as cli

    py = str(Path(".venv/bin/python").absolute())
    c = cli.make_candidate({"adapter": "engine", "config": "track_s_fast", "env": {"ENGINE_PYTHON": py}, "mps": "off", "label": "mm-override"})
    gpu = {"gpu_index": 2, "gpu_sku": "L20X", "gpu_uuid": "GPU-fixture", "driver": "570", "memory_mib": "143771"}
    g = mm.generation_identity(c, gpu, ["mmmu"], 1, 18990, EVAL["digest"])
    ss = cli.server_spec(c, 18990, 2)
    rep = {"generation": g, "candidate": g["candidate"], "gpu": gpu, "concurrency": 1, "evaluator": EVAL, "benches": {"mmmu": {}},
           "server_evidence": {"gpu": gpu, "gpu_workers": [{"pid": 11}], "process_tree": [10, 11], "launch_argv": ss.cmd, "launch_env": ss.env,
                               "served_identity": dict(ss.expected_identity), "served_identity_problems": []}}
    return py, g, rep


def _redigest(g):
    g["digest"] = hashlib.sha256(json.dumps({k: v for k, v in g.items() if k != "digest"}, sort_keys=True).encode()).hexdigest()


def _b15(problems):
    return [p for p in problems if "interpreter" in p or "served identity" in p or "server's identity" in p]


def test_engine_generation_binds_the_candidate_interpreter_and_served_identity(monkeypatch):
    py, g, rep = _engine_generation(monkeypatch)
    assert g["adapter_python"] == py and g["served_identity"]["python"] == py  # the override, not the default interpreter
    assert not _b15(mm.generation_problems(rep)), mm.generation_problems(rep)
    # the round-7 behaviour: the default interpreter recorded despite the override
    import bench.cli as cli

    old = json.loads(json.dumps(rep))
    old["generation"]["adapter_python"] = cli.ADAPTER_PYTHON["engine"]
    _redigest(old["generation"])
    old["candidate"] = old["generation"]["candidate"]
    assert cli.ADAPTER_PYTHON["engine"] != py and any("interpreter" in p for p in mm.generation_problems(old))
    # the server reported other packages
    wrong = json.loads(json.dumps(rep))
    wrong["server_evidence"]["served_identity"]["python_dists_sha256"] = "wrong-packages"
    assert any("server's identity" in p for p in mm.generation_problems(wrong))
    # the candidate's environment changed packages after the declaration
    real = cli.served_identity_expected
    monkeypatch.setattr(cli, "served_identity_expected", lambda config, env=None: {**real(config, env), "python_dists_sha256": "drifted"})
    assert any("resolution now" in p for p in mm.generation_problems(rep))
    # an engine declaration without a served identity
    bare = json.loads(json.dumps(rep))
    bare["generation"].pop("served_identity")
    _redigest(bare["generation"])
    assert any("no served identity" in p for p in mm.generation_problems(bare))
