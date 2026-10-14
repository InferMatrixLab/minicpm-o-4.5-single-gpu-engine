import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals.multimodal import suite as mm  # noqa: E402
from evals.multimodal.sglang_mm_worker import build_payload  # noqa: E402


def _out(n=10, fail=0, correct=8):
    per = [{"id": f"s{i}", "expected": "A", "predicted": "A" if i < correct else "B", "is_correct": i < correct, "is_success": i >= fail}
           for i in range(n)]
    return {"ids": [f"s{i}" for i in range(n)], "per_sample": per}


def test_coverage_requires_every_declared_sample():
    assert mm.coverage(_out())["complete"]
    assert not mm.coverage(_out(fail=1))["complete"]
    o = _out()
    o["per_sample"] = o["per_sample"][:-1]
    assert not mm.coverage(o)["complete"]


def _report(acc, complete=True, ident="X"):
    return {"benches": {"mmmu": {"accuracy": acc, "coverage": {"complete": complete}, "identity": ident}}}


def test_floors_and_relative_gate():
    assert mm.gates(_report(0.70), None)["mmmu"]["passed"]
    assert not mm.gates(_report(0.60), None)["mmmu"]["passed"]
    assert not mm.gates(_report(0.70, complete=False), None)["mmmu"]["passed"]
    q = {"values": {"mmmu": 0.72}, "identity": {"mmmu": "X"}}
    assert mm.gates(_report(0.711), q)["mmmu"]["passed"]
    assert not mm.gates(_report(0.705), q)["mmmu"]["passed"]  # more than 1 pp below Q_ref
    assert not mm.gates(_report(0.75, ident="Y"), q)["mmmu"]["passed"]  # Q_ref from another dataset/evaluator


def test_shuffled_answer_fixture_fails(tmp_path):
    per = [{"id": f"s{i}", "expected": "ABCD"[i % 4], "predicted": "ABCD"[i % 4], "is_correct": True, "is_success": True} for i in range(200)]
    (tmp_path / "mmmu.per_sample.jsonl").write_text("".join(json.dumps(p) + "\n" for p in per))
    rep = {"benches": {"mmmu": {"accuracy": 1.0}}}
    (tmp_path / "report.json").write_text(json.dumps(rep))
    args = SimpleNamespace(report=str(tmp_path / "report.json"), out=str(tmp_path / "shuffled.json"))
    assert mm.cmd_calibrate(args) == 0
    res = json.loads((tmp_path / "shuffled.json").read_text())
    assert res["mmmu"]["shuffled_accuracy"] < 0.4 and res["mmmu"]["rejected"]


def test_freeze_needs_qualified_baselines(tmp_path):
    rep = {"scope": "development", "candidate": {"digest": "D", "candidate_id": "b:x:1"},
           "benches": {"mmmu": {"accuracy": 0.7, "coverage": {"complete": True}, "identity": {"ids": 1}}}}
    rp = tmp_path / "r.json"
    rp.write_text(json.dumps(rep))
    qp = tmp_path / "q.json"
    qp.write_text(json.dumps({"qualified": False, "candidate_digest": "D"}))
    with pytest.raises(SystemExit):
        mm.cmd_freeze(SimpleNamespace(out=str(tmp_path / "f1.json"), scope="development", reference=[f"{rp}={qp}"]))
    qp.write_text(json.dumps({"qualified": True, "candidate_digest": "D"}))
    assert mm.cmd_freeze(SimpleNamespace(out=str(tmp_path / "f2.json"), scope="development", reference=[f"{rp}={qp}"])) == 0
    with pytest.raises(SystemExit):
        mm.cmd_freeze(SimpleNamespace(out=str(tmp_path / "f2.json"), scope="development", reference=[f"{rp}={qp}"]))


def test_request_formats_place_media_before_text(tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF....")
    mmmu = SimpleNamespace(prompt="Q?", image_data_uris=("data:image/png;base64,AA", "data:image/png;base64,BB"))
    p = build_payload({"bench": "mmmu", "model": "m", "max_tokens": 8, "format": "openai"}, mmmu, None)
    assert [c["type"] for c in p["messages"][0]["content"]] == ["image_url", "image_url", "text"]
    s = build_payload({"bench": "mmmu", "model": "m", "max_tokens": 8, "format": "sglang"}, mmmu, None)
    assert s["images"] == list(mmmu.image_data_uris) and s["messages"][0]["content"] == "Q?"
    mmsu = SimpleNamespace(audio_path=str(wav))
    p = build_payload({"bench": "mmsu", "model": "m", "max_tokens": 8, "format": "openai"}, mmsu, "Listen")
    assert [c["type"] for c in p["messages"][0]["content"]] == ["input_audio", "text"]
    assert p["messages"][0]["content"][0]["input_audio"]["format"] == "wav" and p["modalities"] == ["text"]


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
        assert p["messages"] == n.omni_chat_messages  # sampled frames + audio track, exactly as vllm-omni builds them
        parts = [c["type"] for c in p["messages"][0]["content"] if isinstance(c, dict)]
        assert "image_url" in parts and "audio_url" in parts
        assert p["modalities"] == ["text"] and p["chat_template_kwargs"] == {"enable_thinking": False}
