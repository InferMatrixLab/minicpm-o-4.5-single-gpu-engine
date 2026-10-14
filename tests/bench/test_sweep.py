"""End-to-end cells against a fake engine server: experiment specs, verification, resume, extension.

Everything is real (server processes, manifests, artifacts, verification) except the
nvidia-smi observation of GPU contexts, which is replaced by a direct context for
the fake server's processes.
"""

import asyncio
import gzip
import json
import socket
import sys
from pathlib import Path

import pytest

import bench.cli as cli
import bench.servers as servers
from bench.experiment import Candidate, unique_candidates
from bench.servers import ServerSpec
from bench.verify import verify_cell

FAKE = Path(__file__).resolve().parent / "fake_engine_server.py"
GPU = 1


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    calls = []

    def fake_spec(cand, port, gpu):
        calls.append({"candidate_id": cand.candidate_id, "extra": list(cand.extra), "env": cand.env_dict()})
        return ServerSpec(name=cand.candidate_id, cmd=[sys.executable, str(FAKE), str(port)], port=port, gpu=gpu, env=cand.env_dict())

    def fake_gpu_processes(gpu):
        # the fake server's own processes hold a direct (non-MPS) context
        out = servers.subprocess.run(["pgrep", "-f", str(FAKE)], capture_output=True, text=True).stdout.split()
        return [{"pid": int(p), "type": "C", "process_name": "python", "used_memory": "1 MiB"} for p in out]

    monkeypatch.setattr(cli, "server_spec", fake_spec)
    monkeypatch.setattr(servers, "gpu_processes", fake_gpu_processes)
    monkeypatch.setattr(servers, "gpu_memory_used_mib", lambda gpu: 0.0)  # real GPUs may be busy with other cells
    monkeypatch.setattr(cli, "deploy_files_sha256", lambda resolved: ("fake",))
    monkeypatch.setattr(cli, "make_candidate", lambda d: Candidate.from_spec(d, ("fake",)))
    return calls


def _cell(tmp_path, k=1, label="", extra=(), env=None, mps="off", seed=0):
    cand = Candidate.from_spec({"adapter": "engine", "config": "fake", "extra": list(extra), "env": env or {}, "label": label, "mps": mps}, ("fake",))
    cell = cli.make_cell(cand, 1, k, "eval", seed, 180.0, GPU)
    out = tmp_path / f"cell_{label or 'x'}_{k}_s{seed}"
    res = asyncio.run(cli.run_one_cell(cell=cell, out=out, port=_free_port(), url=None, cache_policy="restart", run_order_seed=0,
                                       save_audio=True, strict=True))
    return cell, out, res


def test_real_cell_verifies_and_tamper_cases_fail(tmp_path):
    cell, out, res = _cell(tmp_path, k=1)
    assert res["cell_compliant"], res["verification"]
    assert verify_cell(out).ok
    res_path = out / "result.json"
    orig = res_path.read_text()

    def tampered(fn):
        r = json.loads(orig)
        fn(r)
        res_path.write_text(json.dumps(r))
        ok = verify_cell(out).ok
        res_path.write_text(orig)
        return ok

    # an invalid measured run, a stale digest, a foreign spec, a mismatched manifest, a dropped tail run
    assert not tampered(lambda r: r["runs"][1].update(valid=False))
    assert not tampered(lambda r: r["experiment"].update(concurrency=32))
    assert not tampered(lambda r: r["manifest"]["deploy"]["experiment"].update(concurrency=32))
    assert not tampered(lambda r: r.update(runs=r["runs"][:1]))
    assert not tampered(lambda r: r["experiment"]["fingerprints"].update(harness_tree_sha256="0" * 64))
    assert verify_cell(out).ok


@pytest.mark.parametrize("damage", ["empty_summary", "truncated_requests", "missing_request_line", "missing_wire", "missing_wav", "empty_traces"])
def test_artifact_damage_fails_verification(tmp_path, damage):
    _, out, res = _cell(tmp_path, k=1)
    run = out / "run1"
    if damage == "empty_summary":
        (run / "summary.json").write_text("{}")
    elif damage == "truncated_requests":
        data = (run / "requests.jsonl").read_bytes()
        (run / "requests.jsonl").write_bytes(data[: len(data) // 2])
    elif damage == "missing_request_line":
        lines = (run / "requests.jsonl").read_text().splitlines()
        (run / "requests.jsonl").write_text("\n".join(lines[1:]) + "\n")
    elif damage == "missing_wire":
        (run / "wire.jsonl.gz").unlink()
    elif damage == "missing_wav":
        next((run / "wav").glob("*.wav")).unlink()
    elif damage == "empty_traces":
        (run / "traces.jsonl").write_text("")
    v = verify_cell(out)
    assert not v.ok and v.problems


def test_resume_compares_current_fingerprints(tmp_path, monkeypatch):
    cell, out, res = _cell(tmp_path, k=1)
    current = cli.build_experiment(cell, "performance", "restart", True, True, True, False)
    assert cli.resume_state(out, current) == "reuse"
    (out / "run1" / "traces.jsonl").write_text("")
    assert cli.resume_state(out, current) == "retry"
    for key, val in (("harness_tree_sha256", "1" * 64), ("launch_scripts", {"env.sh": "2" * 64}), ("envs", {"bench": "3" * 64})):
        changed = json.loads(json.dumps(current))
        changed["fingerprints"][key] = val
        with pytest.raises(cli.SweepError):
            cli.resume_state(out, changed)
    changed = json.loads(json.dumps(current))
    changed["fingerprints"]["repos"]["vllm-omni"]["commit"] = "deadbeef"
    with pytest.raises(cli.SweepError):
        cli.resume_state(out, changed)
    changed = json.loads(json.dumps(current))
    changed["model"]["revision"] = "main"
    with pytest.raises(cli.SweepError):
        cli.resume_state(out, changed)


def test_extension_is_declared_before_execution(tmp_path):
    cell, out, res = _cell(tmp_path, k=1)
    args = type("A", (), {"cell": str(out), "k": 2, "reason": "interval rule", "port": _free_port()})()
    assert asyncio.run(cli._extend(args)) == 0
    r = json.loads((out / "result.json").read_text())
    assert r["experiment"]["k"] == 2 and r["experiment"]["k_updates"][0]["from"] == 1
    assert [x["label"] for x in r["runs"]] == ["run0", "run1", "run2"]
    assert verify_cell(out).ok
    # dropping the (declared) new tail run is detected
    r["runs"] = r["runs"][:2]
    (out / "result.json").write_text(json.dumps(r))
    assert not verify_cell(out).ok


def test_variants_keep_their_own_arguments(tmp_path, fake_runtime):
    spec = {"gpu": GPU, "k": 1, "run_order_seed": 3, "concurrencies": [1], "port": _free_port(),
            "configs": [{"adapter": "engine", "config": "fake", "label": "mps-off", "extra": []},
                        {"adapter": "engine", "config": "fake", "label": "v2", "env": {"OMNI_X": "1"}, "extra": ["--flag-on"]}]}
    p = tmp_path / "spec.json"
    p.write_text(json.dumps(spec))
    assert asyncio.run(cli._sweep(type("A", (), {"spec": str(p), "out": str(tmp_path / "out")})())) == 0
    by_label = {}
    for c in fake_runtime:
        by_label.setdefault(c["candidate_id"].split(":")[1], c)
    assert by_label["fake-mps-off"]["extra"] == [] and by_label["fake-mps-off"]["env"] == {}
    assert by_label["fake-v2"]["extra"] == ["--flag-on"] and by_label["fake-v2"]["env"] == {"OMNI_X": "1"}
    # a second sweep run reuses both verified cells without executing anything
    fake_runtime.clear()
    assert asyncio.run(cli._sweep(type("A", (), {"spec": str(p), "out": str(tmp_path / "out")})())) == 0
    assert fake_runtime == []
    # a different spec cannot reuse the directory
    spec["k"] = 2
    p.write_text(json.dumps(spec))
    with pytest.raises(cli.SweepError):
        asyncio.run(cli._sweep(type("A", (), {"spec": str(p), "out": str(tmp_path / "out")})()))


def test_duplicate_candidates_rejected():
    a = Candidate.from_spec({"adapter": "sglang-omni", "config": "speech"}, ("x",))
    with pytest.raises(ValueError):
        unique_candidates([a, Candidate.from_spec({"adapter": "sglang-omni", "config": "speech"}, ("x",))])
    with pytest.raises(ValueError):
        unique_candidates([a, Candidate.from_spec({"adapter": "sglang-omni", "config": "speech", "label": "x"}, ("x",))])
    assert Candidate.from_spec({"adapter": "a", "config": "c", "mps": "on"}, ()).digest != Candidate.from_spec({"adapter": "a", "config": "c"}, ()).digest


def test_declared_mps_must_match_observation(tmp_path, monkeypatch):
    _, out, res = _cell(tmp_path, k=1, mps="on")  # fake server holds a direct context: observed "off"
    assert not res["cell_compliant"]
    assert any("MPS" in p for p in res["verification"]["problems"])
