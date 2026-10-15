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


def test_generated_stages_cover_every_concurrency_once():
    specs = campaign.all_specs()
    tune = [specs[f"tune_gpu{lane['gpu']}.json"] for lane in campaign.TUNE_LANES]
    assert sorted(c for t in tune for c in t["concurrencies"]) == list(campaign.CONCURRENCIES)
    assert all(t["kind"] == "performance" and t["id_set"] == "tuning" and t["seed_index"] == 0 for t in tune)
    assert all(campaign.candidate_key(c) not in campaign.NOT_QUALIFIED for c in tune[0]["configs"]) and len(tune[0]["configs"]) == 7
    full = [specs[f"fullq_c32_seed{s['seed_index']}.json"] for s in campaign.SEEDS]
    assert [f["seed_index"] for f in full] == [0, 1, 2] and all(f["id_set"] == "full" and f["concurrencies"] == [32] for f in full)


def test_stage_specs_follow_the_winners():
    cands = campaign.tuning_candidates()
    d = [campaign._digest(c) for c in cands]
    cells = {str(c): {m: {"candidate_digest": d[i % 2]} for i, m in enumerate(campaign.METRICS)} for c in campaign.CONCURRENCIES}
    st = campaign.stage_specs({"cells": cells})
    for c in campaign.CONCURRENCIES:
        e = st[f"eval_c{c}.json"]
        assert e["k"] == campaign.EVAL_K and e["id_set"] == "eval" and e["concurrencies"] == [c] and len(e["configs"]) == 2
    assert all(st[f"fullq_c1_seed{s}.json"]["configs"] == [cands[0]] for s in range(3))
