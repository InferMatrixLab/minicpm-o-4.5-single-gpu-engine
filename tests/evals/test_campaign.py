"""Campaign tooling: generated specs, candidate grouping and evidence-derived rows."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from evals import campaign  # noqa: E402


def test_committed_specs_match_generator():
    assert campaign.main(["specs", "--check"]) == 0
    specs = [campaign.sweep_spec(s) for s in campaign.SEEDS]
    assert len({s["gpu"] for s in specs}) == 3 and [s["seed_index"] for s in specs] == [0, 1, 2]
    cands = specs[0]["configs"]
    assert sum(c["mps"] == "on" for c in cands) == 4 and sum(c["mps"] == "off" for c in cands) == 5
    assert all(Path(c["config"]).exists() for c in cands if c["mps"] == "on" and c["adapter"] == "vllm-omni")


def _exp(root, name, digest, seed, completed=True):
    d = root / name
    d.mkdir(parents=True)
    if completed:
        (d / "result.json").write_text("{}")
    (d / "experiment.json").write_text(json.dumps({"experiment": {"seed_index": seed, "candidate": {"digest": digest, "candidate_id": f"c:{digest}",
                                                                                                     "adapter": "vllm-omni", "config": "x.yaml"}}}))


def test_grouping_and_incomplete_candidates(tmp_path):
    for s in range(3):
        _exp(tmp_path / f"s{s}", "a", "AAA", s)
    _exp(tmp_path / "s0", "b", "BBB", 0)
    g = campaign.group_cells([str(tmp_path / f"s{s}") for s in range(3)])
    assert sorted(g["AAA"]["cells"]) == [0, 1, 2] and sorted(g["BBB"]["cells"]) == [0]
    assert campaign.record(g["BBB"]["candidate"], g["BBB"]["cells"], tmp_path / "out")["status"] == "incomplete"
    assert campaign.record(g["AAA"]["candidate"], g["AAA"]["cells"], tmp_path / "out")["status"] == "unscored"
    _exp(tmp_path / "s1", "a-failed", "AAA", 1, completed=False)  # a server that never completed: recorded, not a seed
    g = campaign.group_cells([str(tmp_path / f"s{s}") for s in range(3)])
    assert sorted(g["AAA"]["cells"]) == [0, 1, 2] and g["AAA"]["failed_attempts"] == [{"seed_index": 1, "cell": str(tmp_path / "s1" / "a-failed")}]
    _exp(tmp_path / "s9", "a2", "AAA", 1)
    with pytest.raises(SystemExit):  # one candidate's seed observed twice
        campaign.group_cells([str(tmp_path / "s1"), str(tmp_path / "s9")])
