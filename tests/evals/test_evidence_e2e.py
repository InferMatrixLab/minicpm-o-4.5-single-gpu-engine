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
def _winners_stub(monkeypatch, digest, c, hardware, cell_dir):
    doc = {"gpu_class": vrm.gpu_class(hardware), "cells": {str(c): {"throughput": {"candidate_digest": digest, "candidate_id": "w", "cell": str(cell_dir)}}}}
    monkeypatch.setattr(campaign, "verify_winners", lambda path, check_sources=True: doc)


def test_structured_dref_end_to_end(tmp_path, monkeypatch):
    full = [cell(tmp_path, seed=s, id_set="full") for s in range(3)]
    spec = json.loads((full[0] / "experiment.json").read_text())["experiment"]
    winners = tmp_path / "winners.json"
    winners.write_text("{}")
    _winners_stub(monkeypatch, spec["candidate"]["digest"], 1, spec["hardware"], full[0])
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
    _winners_stub(monkeypatch, "another-candidate", 1, spec["hardware"], full[0])
    assert any("throughput winner" in p for p in vrm.verify_dref(dref)[0])
    _winners_stub(monkeypatch, spec["candidate"]["digest"], 1, {**spec["hardware"], "gpu_sku": "H200"}, full[0])
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


def _sweep_spec(tmp_path, name, *, seed, kind, id_set, k, concurrencies, labels, frozen_refs=None):
    spec = {"description": name, "kind": kind, "id_set": id_set, "seed_index": seed, "gpu": GPU, "port": _free_port(), "k": k, "run_order_seed": seed,
            "timeout_s": 180, "save_audio": True, "concurrencies": concurrencies,
            "configs": [{"adapter": "engine", "config": "fake", "label": lab, "env": {"LBL": lab}} for lab in labels]}
    if frozen_refs:
        spec["frozen_refs"] = frozen_refs
    sp = tmp_path / f"{name}.json"
    sp.write_text(json.dumps(spec))
    out = tmp_path / name
    asyncio.run(cli._sweep(type("A", (), {"spec": str(sp), "out": str(out)})()))
    return str(out)


def test_full_chain_winners_dref_final_gate(tmp_path, monkeypatch):
    """qualification -> tuning winners -> winner-bound D_ref -> final reports -> complete final Q_ref -> declared engine gate.

    Only the neural scorers are faked; class R and the preprocessing identity of the fake 'engine' baseline are stubbed
    (the fake server is not a real baseline), everything else is the production path with source checks enabled.
    """
    from bench.eligibility import ConfigClass
    from bench.effective_inputs import write_provenance
    import bench.semantics as sem

    eff = tmp_path / "eff"
    eff.mkdir()
    model = {"repo": "openbmb/MiniCPM-o-4_5", "revision": "503e754207c94da6bb26850b4469f367c9ea3582"}
    for stack in ("canonical", "engine"):
        shutil.copy(ROOT / "results" / "effective_inputs_v1" / "canonical.jsonl" if (ROOT / "results" / "effective_inputs_v1").exists()
                    else ROOT / "results" / "effective_inputs" / "canonical.jsonl", eff / f"{stack}.jsonl")
        write_provenance(eff / f"{stack}.jsonl", stack, model, preprocessing={"adapter": stack})
    monkeypatch.setattr(campaign, "EFFECTIVE_DIR", eff)
    monkeypatch.setattr(campaign, "CONCURRENCIES", (1, 32))
    monkeypatch.setattr(sem, "preprocessing_identity", lambda adapter, config: {"adapter": adapter})
    monkeypatch.setattr(sq, "class_r", lambda cand, elig, check_binding=True: ConfigClass(
        engine="vllm-omni", config=cand["config"], track="T", audio_output="stream", chunk_tokens=25,
        precision={"thinker": "bf16", "talker": "bf16", "flow": "tf32", "hift": "fp32"}, euler_steps=10, session_mode="turn", canonical_inputs=elig))
    labels = ("a", "b")
    qual_sweeps = [_sweep_spec(tmp_path, f"q{s}", seed=s, kind="quality", id_set="tuning", k=1, concurrencies=[8], labels=labels) for s in range(3)]
    qdir = tmp_path / "qualification"
    for g in campaign.group_cells(qual_sweeps).values():
        d = qdir / campaign._slug(g["candidate"])
        d.mkdir(parents=True)
        score([g["cells"][s] for s in range(3)], d / "speech_quality.json")
        (d / "qualification.json").write_text(json.dumps(sq.qualify(str(d / "speech_quality.json"), str(eff))))
    rows = campaign.campaign_rows(qual_sweeps, qdir)
    assert all(r["status"] == "qualified" for r in rows), [r.get("reason") for r in rows]
    camp = qdir / "campaign_qualification.json"
    camp.write_text(json.dumps({"scope": "development", "id_set": "tuning", "sweeps": qual_sweeps, "candidates": rows}))
    tune = [_sweep_spec(tmp_path, "tune", seed=0, kind="performance", id_set="tuning", k=campaign.TUNE_K, concurrencies=[1, 32], labels=labels)]
    winners = tmp_path / "winners.json"
    winners.write_text(json.dumps(campaign.compute_winners(camp, tune)))
    w = campaign.verify_winners(winners)
    top1, top32 = w["cells"]["1"]["throughput"]["candidate_id"].split(":")[1].split("-")[-1], w["cells"]["32"]["throughput"]["candidate_id"].split(":")[1].split("-")[-1]
    full32 = [_sweep_spec(tmp_path, f"f32s{s}", seed=s, kind="quality", id_set="full", k=1, concurrencies=[32], labels=labels) for s in range(3)]
    full1 = [_sweep_spec(tmp_path, f"f1s{s}", seed=s, kind="quality", id_set="full", k=1, concurrencies=[1], labels=(top1,)) for s in range(3)]
    inv, chosen, _ = campaign.select_cells(full32 + full1)
    drefs = {}
    for c, top in ((32, top32), (1, top1)):
        dg = next(d for (d, s, cc) in inv if cc == c and inv[(d, s, cc)]["candidate_id"].split(":")[1].endswith(top))
        drefs[c] = tmp_path / f"dref_c{c}.json"
        vrm.build_dref([chosen[(dg, s, c)]["cell"] for s in range(3)], drefs[c], "truncation", winners)
    refs = []
    for (dg, s, c), a in sorted(chosen.items()):
        if s:
            continue
        cells = [chosen[(dg, x, c)]["cell"] for x in range(3)]
        rep = score(cells, tmp_path / f"final_{dg[:8]}_c{c}.json", scope="final", d_ref=drefs[c])
        qual = next(r for r in rows if r["digest"] == dg)["qualification"]["path"]
        refs.append((str(rep), qual))
    qref = tmp_path / "qref_final.json"
    with pytest.raises(sq.QRefError):  # an incomplete reference set is refused
        sq.freeze("final", "L20X", refs[:-1], tmp_path / "incomplete.json", campaign=str(camp), winners=str(winners))
    sq.freeze("final", "L20X", refs, qref, campaign=str(camp), winners=str(winners))
    assert sq.verify_qref(str(qref))["values"]
    eng = [_sweep_spec(tmp_path, f"e{s}", seed=s, kind="quality", id_set="full", k=1, concurrencies=[32], labels=("eng",),
                       frozen_refs={"speech_qref": str(qref)}) for s in range(3)]
    ecells = [next(iter(campaign.select_cells([e])[1].values()))["cell"] for e in eng]
    erep = score(ecells, tmp_path / "engine_final.json", scope="final", d_ref=drefs[32])
    head = json.loads((Path(ecells[0]) / "experiment.json").read_text())["experiment"]["candidate"]["digest"]
    assert sq.gate(str(qref), str(erep), head)["overall"] in ("pass", "inconclusive")
    q = json.loads(qref.read_text())
    q["frozen_unix"] = -1  # editing the timestamp neither helps nor matters: the binding is the declared hash
    (tmp_path / "qref_edit.json").write_text(json.dumps(q))
    with pytest.raises(sq.QRefError):
        sq.gate(str(tmp_path / "qref_edit.json"), str(erep), head)
    undeclared = [_sweep_spec(tmp_path, f"u{s}", seed=s, kind="quality", id_set="full", k=1, concurrencies=[32], labels=("eng2",)) for s in range(3)]
    ucells = [next(iter(campaign.select_cells([u])[1].values()))["cell"] for u in undeclared]
    urep = score(ucells, tmp_path / "undeclared_final.json", scope="final", d_ref=drefs[32])
    uhead = json.loads((Path(ucells[0]) / "experiment.json").read_text())["experiment"]["candidate"]["digest"]
    with pytest.raises(sq.QRefError):
        sq.gate(str(qref), str(urep), uhead)
    # B12c: attaching the Q_ref to completed, undeclared cells afterwards (inside the experiment, with its digests
    # recomputed, or beside it) fails verification and the gate: the runs carry the base digest of the declared spec
    from bench.experiment import spec_base_digest, spec_digest
    from bench.verify import verify_cell

    sha = sq._sha(str(qref))
    for c in ucells:
        d = json.loads((Path(c) / "experiment.json").read_text())
        d["experiment"]["frozen_refs"] = {"speech_qref": {"path": str(qref), "sha256": sha}}
        d["experiment_digest"], d["experiment_base_digest"] = spec_digest(d["experiment"]), spec_base_digest(d["experiment"])
        d["frozen_refs"] = d["experiment"]["frozen_refs"]
        (Path(c) / "experiment.json").write_text(json.dumps(d))
        assert not verify_cell(Path(c)).ok
    with pytest.raises(sq.QRefError):
        sq.gate(str(qref), str(urep), uhead)
    # declared cells carry the binding inside the verified experiment
    for c in ecells:
        assert json.loads((Path(c) / "experiment.json").read_text())["experiment"]["frozen_refs"]["speech_qref"]["sha256"] == sha
        assert verify_cell(Path(c)).ok
    # a tuning cell with a schedule other than its sweep's declaration is never selected
    rec = json.loads((Path(tune[0]) / "sweep.json").read_text())
    rec["spec"]["k"] = 10
    from bench.experiment import digest as _digest

    rec["spec_digest"] = _digest(rec["spec"])
    (Path(tune[0]) / "sweep.json").write_text(json.dumps(rec))
    campaign._WINNERS.clear()
    assert not campaign.select_cells(tune)[1]
