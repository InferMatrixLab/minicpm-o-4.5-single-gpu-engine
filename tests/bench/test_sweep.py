"""Sweep scheduling, candidate identity and resume validation."""

import asyncio
import json

import pytest

import bench.cli as cli
from bench.experiment import Candidate, unique_candidates


def _fake_make_candidate(d):
    return Candidate.from_spec(d, ("deadbeef",))


class _Recorder:
    def __init__(self, compliant=True):
        self.calls = []
        self.compliant = compliant

    async def __call__(self, *, cell, out, **kw):
        self.calls.append((cell.candidate.candidate_id, list(cell.candidate.extra), dict(cell.candidate.env), cell.concurrency))
        out.mkdir(parents=True, exist_ok=True)
        for i in range(cell.k + 1):
            (out / f"run{i}").mkdir(exist_ok=True)
            (out / f"run{i}" / "summary.json").write_text("{}")
            (out / f"run{i}" / "traces.jsonl").write_text("")
        res = {"cell_spec": cell.to_json(), "manifest": {}, "manifest_errors": [], "cell_valid": True, "cell_compliant": self.compliant}
        (out / "result.json").write_text(json.dumps(res))
        return res


def _spec(tmp_path, **over):
    spec = {
        "gpu": 1, "k": 2, "run_order_seed": 3, "concurrencies": [1],
        "configs": [
            {"adapter": "sglang-omni", "config": "speech", "label": "mps-off", "extra": []},
            {"adapter": "sglang-omni", "config": "speech", "label": "mps-on", "env": {"OMNI_MPS": "1"}, "extra": ["--flag-on"]},
        ],
    }
    spec.update(over)
    p = tmp_path / f"spec{len(list(tmp_path.glob('spec*')))}.json"
    p.write_text(json.dumps(spec))
    return p


def _run(monkeypatch, spec_path, out, rec):
    monkeypatch.setattr(cli, "make_candidate", _fake_make_candidate)
    monkeypatch.setattr(cli, "validate_manifest", lambda m: [])
    monkeypatch.setattr(cli, "run_one_cell", rec)
    return asyncio.run(cli._sweep(type("A", (), {"spec": str(spec_path), "out": str(out)})()))


def test_variants_keep_their_own_arguments(tmp_path, monkeypatch):
    rec = _Recorder()
    assert _run(monkeypatch, _spec(tmp_path), tmp_path / "out", rec) == 0
    by_label = {c[0].split(":")[1]: c for c in rec.calls}
    assert by_label["speech-mps-off"][1] == [] and by_label["speech-mps-off"][2] == {}
    assert by_label["speech-mps-on"][1] == ["--flag-on"] and by_label["speech-mps-on"][2] == {"OMNI_MPS": "1"}
    assert len({c[0] for c in rec.calls}) == 2
    dirs = sorted(p.name for p in (tmp_path / "out").iterdir() if p.is_dir())
    assert len(dirs) == 2


def test_duplicate_candidates_rejected():
    a = Candidate.from_spec({"adapter": "sglang-omni", "config": "speech"}, ("x",))
    with pytest.raises(ValueError):
        unique_candidates([a, Candidate.from_spec({"adapter": "sglang-omni", "config": "speech"}, ("x",))])
    with pytest.raises(ValueError):  # a different label on an identical launch is still the same experiment
        unique_candidates([a, Candidate.from_spec({"adapter": "sglang-omni", "config": "speech", "label": "x"}, ("x",))])


def test_resume_reuses_only_matching_compliant_cells(tmp_path, monkeypatch):
    out = tmp_path / "out"
    spec = _spec(tmp_path)
    assert _run(monkeypatch, spec, out, _Recorder()) == 0
    rec = _Recorder()
    assert _run(monkeypatch, spec, out, rec) == 0
    assert rec.calls == []  # both cells reused
    # a changed spec (K, GPU, seed, config or arguments) cannot reuse this directory
    for over in ({"k": 3}, {"gpu": 2}, {"run_order_seed": 9},
                 {"configs": [{"adapter": "sglang-omni", "config": "speech", "extra": ["--other"]}]}):
        with pytest.raises(cli.SweepError):
            _run(monkeypatch, _spec(tmp_path, **over), out, _Recorder())


def test_cell_dir_from_other_experiment_refused(tmp_path, monkeypatch):
    out = tmp_path / "out"
    spec = _spec(tmp_path)
    assert _run(monkeypatch, spec, out, _Recorder()) == 0
    cell_dir = next(p for p in out.iterdir() if p.is_dir() and p.name[:2].isdigit())
    res = json.loads((cell_dir / "result.json").read_text())
    res["cell_spec"]["digest"] = "different"
    (cell_dir / "result.json").write_text(json.dumps(res))
    with pytest.raises(cli.SweepError):
        _run(monkeypatch, spec, out, _Recorder())


def test_noncompliant_or_incomplete_cells_rerun_and_fail(tmp_path, monkeypatch):
    out = tmp_path / "out"
    spec = _spec(tmp_path)
    assert _run(monkeypatch, spec, out, _Recorder(compliant=False)) == 1  # noncompliant cells fail the sweep
    rec = _Recorder()
    assert _run(monkeypatch, spec, out, rec) == 0
    assert len(rec.calls) == 2  # retried, not reused
    attempts = list((out / "attempts").iterdir())
    assert len(attempts) == 2  # failed attempts are kept
    # missing run artifacts force a retry too
    cell_dir = next(p for p in out.iterdir() if p.is_dir() and p.name[:2].isdigit())
    (cell_dir / "run2" / "traces.jsonl").unlink()
    rec = _Recorder()
    assert _run(monkeypatch, spec, out, rec) == 0
    assert len(rec.calls) == 1
