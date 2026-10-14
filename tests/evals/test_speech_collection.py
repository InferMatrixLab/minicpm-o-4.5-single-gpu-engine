"""Speech collection is bound to verified source cells (B7)."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from test_sweep import _cell, fake_runtime  # noqa: E402,F401

from evals.speech.suite import CoverageError, collect  # noqa: E402


@pytest.fixture
def two_seeds(tmp_path, fake_runtime):  # noqa: F811
    _, a, _ = _cell(tmp_path, seed=0)
    _, b, _ = _cell(tmp_path, seed=1)
    return a, b


def test_collect_derives_seeds_from_specs(two_seeds):
    a, b = two_seeds
    col = collect([a, b])
    assert sorted(col.seeds) == ["0", "1"]
    assert len(col.rows) == 640 and len({r.wav for r in col.rows}) == 640
    assert col.generation["concurrency"] == 1


def test_same_cell_relabelled_as_several_seeds_rejected(two_seeds):
    a, _ = two_seeds
    with pytest.raises(CoverageError):
        collect([a, a, a])


def test_deleted_request_line_rejected(two_seeds):
    a, b = two_seeds
    p = a / "run1" / "requests.jsonl"
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:1]) + "\n")
    with pytest.raises(CoverageError):
        collect([a, b])


def test_missing_trace_text_rejected(two_seeds):
    a, b = two_seeds
    p = a / "run1" / "traces.jsonl"
    recs = [json.loads(x) for x in p.read_text().splitlines()]
    del recs[5]["text"]
    p.write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    with pytest.raises(CoverageError):
        collect([a, b])


def test_mixed_candidates_and_duplicate_seed_rejected(tmp_path, fake_runtime):  # noqa: F811
    _, a, _ = _cell(tmp_path, seed=0)
    _, b, _ = _cell(tmp_path, seed=1, label="other")
    with pytest.raises(CoverageError):
        collect([a, b])
    _, c, _ = _cell(tmp_path, seed=0, label="dup")
    _, d, _ = _cell(tmp_path / "x", seed=0, label="dup")
    with pytest.raises(CoverageError):
        collect([c, d])


def test_seed_metadata_tamper_rejected(two_seeds):
    a, b = two_seeds
    r = json.loads((a / "result.json").read_text())
    r["experiment"]["seed_index"] = 7
    (a / "result.json").write_text(json.dumps(r))
    with pytest.raises(CoverageError):
        collect([a, b])
