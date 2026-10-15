"""Producer-to-consumer evidence joins on real harness cells (B8a, B8b, B12b, B14).

The cells are produced by the real harness against the fake engine server and
pass ``verify_cell``; only the neural scorers (ASR/SIM/UTMOS) and the nvidia-smi
context observation are replaced.  Source checks stay enabled throughout.
"""

import asyncio
import hashlib
import json
import shutil
import socket
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "bench"))

import bench.cli as cli  # noqa: E402
import bench.servers as servers  # noqa: E402
from bench.experiment import Candidate  # noqa: E402
from bench.servers import ServerSpec  # noqa: E402
from evals import campaign  # noqa: E402
from evals.speech import qref as sq  # noqa: E402
from evals.speech import suite  # noqa: E402
from evals.speech import verify_report as vrm  # noqa: E402

FAKE = ROOT / "tests" / "bench" / "fake_engine_server.py"
GPU = 1
EV = {"x": 1}
EV["digest"] = hashlib.sha256(json.dumps({"x": 1}, sort_keys=True).encode()).hexdigest()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture(autouse=True)
def fake_runtime(monkeypatch):
    def fake_spec(cand, port, gpu):
        return ServerSpec(name=cand.candidate_id, cmd=[sys.executable, str(FAKE), str(port)], port=port, gpu=gpu, env=cand.env_dict())

    def fake_gpu_processes(gpu):
        out = servers.subprocess.run(["pgrep", "-f", str(FAKE)], capture_output=True, text=True).stdout.split()
        return [{"pid": int(p), "type": "C", "process_name": "python", "used_memory": "1 MiB"} for p in out]

    monkeypatch.setattr(cli, "server_spec", fake_spec)
    monkeypatch.setattr(servers, "gpu_processes", fake_gpu_processes)
    monkeypatch.setattr(servers, "gpu_memory_used_mib", lambda gpu: 0.0)
    monkeypatch.setattr(cli, "deploy_files_sha256", lambda resolved: ("fake",))
    monkeypatch.setattr(cli, "make_candidate", lambda d: Candidate.from_spec(d, ("fake",)))
    campaign._VERIFIED.clear()
    campaign._WINNERS.clear()


def cell(tmp_path, *, label="a", seed=0, c=1, k=1, id_set="tuning", kind="quality"):
    cand = Candidate.from_spec({"adapter": "engine", "config": "fake", "label": label, "env": {"LBL": label}}, ("fake",))  # env: distinct digests
    cs = cli.make_cell(cand, c, k, id_set, seed, 180.0, GPU)
    out = tmp_path / f"{label}_{kind}_{id_set}_c{c}_k{k}_s{seed}"
    asyncio.run(cli.run_one_cell(cell=cs, out=out, port=_free_port(), url=None, cache_policy="restart", run_order_seed=0, save_audio=True,
                                 strict=True, kind=kind))
    return out


def fake_scored(rows):
    asr = {r.key: {"ok": True, "hyp_text": r.target_text, "ref_norm": r.target_text.lower(), "hyp_norm": r.target_text.lower(), "wer": 0.0,
                   "substitutions": 0, "deletions": 0, "insertions": 0, "hits": len(r.target_text.split())} for r in rows}
    return {"evaluator": EV, "execution_evidence": {}, "tasks": {"asr": {"rows": asr}, "sim": {"rows": {r.key: 60.0 for r in rows}},
                                                                 "utmos": {"rows": {r.key: 4.3 for r in rows}}}}


def score(cells, out, scope="development", d_ref=None):
    col = suite.collect(cells)
    rows = suite.load_dref(str(d_ref), col) if d_ref else None
    q = suite.build_quality(col.rows, fake_scored(col.rows), rows)
    suite.write_report(Path(out), scope, col, fake_scored(col.rows), q, str(d_ref) if d_ref else None, 0.0)
    return Path(out)


def mutate(src: Path, dst: Path, fn) -> Path:
    shutil.copy(src.with_suffix(".rows.jsonl"), dst.with_suffix(".rows.jsonl"))
    r = json.loads(src.read_text())
    fn(r)
    dst.write_text(json.dumps(r))
    return dst


# --- B8b: generation identity and measured-run binding ------------------------------------------
def test_report_identity_mutations_fail(tmp_path):
    cells = [cell(tmp_path, seed=s) for s in range(3)]
    rep = score(cells, tmp_path / "q.json")
    assert vrm.verify_report(rep).ok, vrm.verify_report(rep).problems[:3]
    cases = {
        "nested candidate fields": lambda r: r["generation"]["candidate"].update(config="foreign.yaml", extra=["--x"], env=[["A", "1"]]),
        "measured-ID hash": lambda r: r["generation"].update(measured_ids_sha256="FOREIGN"),
        "run0 source": lambda r: [v.update(run="run0", artifacts=json.loads((Path(v["cell"]) / "result.json").read_text())["artifacts"]["run0"])
                                  for v in r["seeds"].values()],
    }
    for name, fn in cases.items():
        bad = mutate(rep, tmp_path / f"bad_{len(name)}.json", fn)
        assert not vrm.verify_report(bad).ok, name
    side = rep.with_suffix(".rows.jsonl")  # a row's generation facts must be the source run's
    lines = side.read_text().splitlines()
    row = json.loads(lines[0])
    row["duration_s"] += 0.5
    lines[0] = json.dumps(row)
    bad = tmp_path / "dur.json"
    (tmp_path / "dur.rows.jsonl").write_text("\n".join(lines) + "\n")
    r = json.loads(rep.read_text())
    r["row_sidecar_sha256"] = hashlib.sha256((tmp_path / "dur.rows.jsonl").read_bytes()).hexdigest()
    bad.write_text(json.dumps(r))
    assert any("duration" in p for p in vrm.verify_report(bad).problems)


# --- B8a: structured duration references ----------------------------------------------------------
def _winners_stub(monkeypatch, digest, c, hardware):
    doc = {"gpu_class": vrm.gpu_class(hardware), "cells": {str(c): {"throughput": {"candidate_digest": digest, "candidate_id": "w"}}}}
    monkeypatch.setattr(campaign, "verify_winners", lambda path, check_sources=True: doc)


def test_structured_dref_end_to_end(tmp_path, monkeypatch):
    full = [cell(tmp_path, seed=s, id_set="full") for s in range(3)]
    spec = json.loads((full[0] / "experiment.json").read_text())["experiment"]
    winners = tmp_path / "winners.json"
    winners.write_text("{}")
    _winners_stub(monkeypatch, spec["candidate"]["digest"], 1, spec["hardware"])
    dref = tmp_path / "dref.json"
    vrm.build_dref(full, dref, "truncation", winners)
    rep = score(full, tmp_path / "final.json", scope="final", d_ref=dref)
    v = vrm.verify_report(rep)
    assert v.ok, v.problems[:3]
    assert all(r.d_ref_s and r.d_ref_s > 0 and r.truncation is False for r in v.rows)  # the CLI path consumes the verified row map
    # a valid but inappropriate source: one tuning-quality K=1 cell offered as the evaluation guard or as truncation
    tune = cell(tmp_path, seed=0)
    for role in ("guard", "truncation"):
        with pytest.raises(vrm.ReportError):
            vrm.build_dref([tune], tmp_path / f"bad_{role}.json", role, winners)
    with pytest.raises(vrm.ReportError):  # duplicate source
        vrm.build_dref([full[0], full[0], full[1]], tmp_path / "dup.json", "truncation", winners)
    doc = json.loads(dref.read_text())
    for name, fn in {"duplicate": lambda d: d["sources"].append(d["sources"][0]), "role": lambda d: d.update(role="guard"),
                     "concurrency": lambda d: d.update(concurrency=32), "duration": lambda d: d["rows"].update({next(iter(d["rows"])): 9.0})}.items():
        bad = json.loads(json.dumps(doc))
        fn(bad)
        p = tmp_path / f"dref_{name}.json"
        p.write_text(json.dumps(bad))
        assert vrm.verify_dref(p)[0], name
    problems, _ = vrm.verify_dref(dref, {"concurrency": 32})
    assert problems  # wrong C for the consumer
    _winners_stub(monkeypatch, "another-candidate", 1, spec["hardware"])
    assert any("throughput winner" in p for p in vrm.verify_dref(dref)[0])
    _winners_stub(monkeypatch, spec["candidate"]["digest"], 1, {**spec["hardware"], "gpu_sku": "H200"})
    assert any("selected on" in p for p in vrm.verify_dref(dref)[0])


def test_final_scope_needs_dref(tmp_path, monkeypatch):
    full = [cell(tmp_path, seed=s, id_set="full") for s in range(3)]
    rep = score(full, tmp_path / "nodref.json", scope="final")
    assert any("duration reference" in p for p in vrm.verify_report(rep).problems)


# --- B14: campaign inventory, joins and attempts -------------------------------------------------
def _sweep(tmp_path, name, seed, labels=("a", "b")):
    spec = {"description": name, "kind": "quality", "id_set": "tuning", "seed_index": seed, "gpu": GPU, "port": _free_port(), "k": 1,
            "run_order_seed": seed, "timeout_s": 180, "save_audio": True, "concurrencies": [8],
            "configs": [{"adapter": "engine", "config": "fake", "label": lab, "env": {"LBL": lab}} for lab in labels]}
    sp = tmp_path / f"{name}.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / name
    asyncio.run(cli._sweep(type("A", (), {"spec": str(sp), "out": str(out)})()))
    return str(out)


def test_campaign_joins_and_attempts(tmp_path):
    sweeps = [_sweep(tmp_path, f"s{s}", s) for s in range(3)]
    groups = campaign.group_cells(sweeps)
    assert len(groups) == 2 and all(sorted(g["cells"]) == [0, 1, 2] for g in groups.values())
    out = tmp_path / "qual"
    ga, gb = sorted(groups.values(), key=lambda g: g["candidate"]["label"])
    for g in (ga, gb):
        d = out / campaign._slug(g["candidate"])
        d.mkdir(parents=True)
        score([g["cells"][s] for s in range(3)], d / "speech_quality.json")
        (d / "qualification.json").write_text(json.dumps(sq.qualify(str(d / "speech_quality.json"), str(tmp_path / "noeff"))))
    assert campaign.record(ga["candidate"], ga["cells"], out, [0, 1, 2])["status"] == "not-qualified"  # engine adapter: never class R
    # cross-candidate substitution: b's directory holds a's unchanged, valid report and qualification
    da, db = out / campaign._slug(ga["candidate"]), out / campaign._slug(gb["candidate"])
    for f in ("speech_quality.json", "speech_quality.rows.jsonl", "qualification.json"):
        shutil.copy(da / f, db / f)
    assert campaign.record(gb["candidate"], gb["cells"], out, [0, 1, 2])["status"] == "unverified"
    # same candidate, a different attempt's cells
    other = {**ga["cells"], 1: cell(tmp_path, label="a", seed=1, c=8)}
    assert campaign.record(ga["candidate"], other, out, [0, 1, 2])["status"] == "unverified"
    # an invalid completed attempt next to a valid rerun, and an archived attempt, are both recorded failures
    bad = Path(sweeps[1]) / "zz_invalid"
    shutil.copytree(ga["cells"][1], bad)
    (bad / "run1" / "requests.jsonl").write_text("")
    arch = Path(sweeps[2]) / "attempts" / "old"
    shutil.copytree(ga["cells"][2], arch)
    (arch / "result.json").unlink()
    g2 = campaign.group_cells(sweeps)[ga["candidate"]["digest"]]
    cells_failed = {f["cell"] for f in g2["failed_attempts"]}
    assert str(bad) in cells_failed and str(arch) in cells_failed and sorted(g2["cells"]) == [0, 1, 2]
    with pytest.raises(campaign.CampaignError):  # a second verified cell of the same seed is ambiguous
        shutil.copytree(ga["cells"][0], Path(sweeps[0]) / "zz_dup")
        campaign.group_cells(sweeps)
    with pytest.raises(campaign.CampaignError):
        campaign.group_cells([])
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(campaign.CampaignError):
        campaign.group_cells([str(empty)])


def test_campaign_reports_missing_seeds(tmp_path):
    sweeps = [_sweep(tmp_path, f"m{s}", s, labels=("a",)) for s in range(2)]  # seed 2 never ran
    g = next(iter(campaign.group_cells(sweeps).values()))
    assert campaign.record(g["candidate"], g["cells"], tmp_path / "q", g["expected_seeds"])["status"] == "incomplete"


# --- B12b: speech Q_ref is rebuilt from its sources ------------------------------------------------
def test_speech_qref_verified_at_consumption(tmp_path, monkeypatch):
    cells = [cell(tmp_path, seed=s) for s in range(3)]
    rep = score(cells, tmp_path / "ref.json")
    qual = tmp_path / "qual.json"
    rec = sq.qualify(str(rep), str(tmp_path / "noeff"))
    rec["qualified"] = True  # the engine adapter is never class R; force a qualified reference for the freeze logic

    def fake_verify_qualification(p, check_sources=True):
        return json.loads(Path(p).read_text())

    monkeypatch.setattr(sq, "verify_qualification", fake_verify_qualification)
    qual.write_text(json.dumps(rec))
    qref = tmp_path / "qref.json"
    sq.freeze("development", "L20X", [(str(rep), str(qual))], qref)
    assert sq.verify_qref(str(qref))["values"]
    eng = score([cell(tmp_path, label="e", seed=s) for s in range(3)], tmp_path / "eng.json")
    assert sq.gate(str(qref), str(eng), None)["overall"] in ("pass", "inconclusive")
    doc = json.loads(qref.read_text())
    for name, fn in {"values": lambda d: d["values"].update(utmos_mean=1.0), "winner": lambda d: d["sources"].update(utmos_mean="other"),
                     "noise": lambda d: d["per_seed"].update(utmos_mean=[1.0, 1.0, 1.0]), "inputs": lambda d: d.update(references={}),
                     "scope": lambda d: d.update(scope="final")}.items():
        bad = json.loads(json.dumps(doc))
        fn(bad)
        p = tmp_path / f"qref_{name}.json"
        p.write_text(json.dumps(bad))
        with pytest.raises(sq.QRefError):
            sq.gate(str(p), str(eng), None)
    side = rep.with_suffix(".rows.jsonl")  # changed sidecar, unchanged report JSON
    side.write_text(side.read_text().replace('"utmos": 4.3', '"utmos": 4.9', 1))
    with pytest.raises(sq.QRefError):
        sq.gate(str(qref), str(eng), None)


def test_final_qref_freeze_to_gate(tmp_path, monkeypatch):
    full = [cell(tmp_path, seed=s, id_set="full") for s in range(3)]
    spec = json.loads((full[0] / "experiment.json").read_text())["experiment"]
    winners = tmp_path / "winners.json"
    winners.write_text("{}")
    _winners_stub(monkeypatch, spec["candidate"]["digest"], 1, spec["hardware"])
    dref = tmp_path / "dref.json"
    vrm.build_dref(full, dref, "truncation", winners)
    ref = score(full, tmp_path / "fref.json", scope="final", d_ref=dref)
    tq = score([cell(tmp_path, seed=s) for s in range(3)], tmp_path / "tq.json")
    rec = sq.qualify(str(tq), str(tmp_path / "noeff"))
    rec["qualified"] = True  # the engine adapter is never class R; force a qualified reference for the freeze logic
    qual = tmp_path / "tq.qual.json"
    qual.write_text(json.dumps(rec))
    monkeypatch.setattr(sq, "verify_qualification", lambda p, check_sources=True: json.loads(Path(p).read_text()))
    qref = tmp_path / "fq.json"
    sq.freeze("final", "L20X", [(str(ref), str(qual))], qref)
    eng_cells = [cell(tmp_path, label="e", seed=s, id_set="full") for s in range(3)]  # declared after the freeze
    eng = score(eng_cells, tmp_path / "feng.json", scope="final", d_ref=dref)
    head = json.loads((eng_cells[0] / "experiment.json").read_text())["experiment"]["candidate"]["digest"]
    assert sq.gate(str(qref), str(eng), head)["overall"] in ("pass", "inconclusive")
    with pytest.raises(sq.QRefError):
        sq.gate(str(qref), str(eng), None)  # not the declared headline
    with pytest.raises(sq.QRefError):
        sq.gate(str(qref), str(ref), spec["candidate"]["digest"])  # its cells were declared before the Q_ref was frozen
