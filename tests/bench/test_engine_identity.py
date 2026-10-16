"""The engine's configuration and implementation are part of its measured identity (B15).

An engine candidate's digest binds the launcher, the JSON bytes the launcher reads and the engine source tree; the
experiment's execution fingerprints add the engine tree and this repository's commit/dirty hash.  So a config
precision/ramp change or a kernel change yields a new candidate and experiment, and a declared cell cannot be resumed
under the changed code.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import bench.cli as cli  # noqa: E402
from bench.experiment import spec_base_digest  # noqa: E402


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "configs" / "engine").mkdir(parents=True)
    (root / "engine" / "runtime").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "scripts" / "serve_engine.sh").write_text((ROOT / "scripts" / "serve_engine.sh").read_text())
    (root / "configs" / "engine" / "head.json").write_text(json.dumps({"first_chunk": 25, "float16_flow": False}))
    (root / "engine" / "runtime" / "dit_fast.py").write_text("SCALE = 1.0\n")
    return root


def _cand():
    return cli.make_candidate({"adapter": "engine", "config": "head", "label": "x", "mps": "on", "env": {"ENGINE_MPS": "1"}})


def test_config_bytes_and_engine_source_change_the_candidate(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setattr(cli, "ROOT", root)
    a = _cand()
    res = cli.resolved_deploy("engine", "head", [], {})
    assert res["settings"] == {"first_chunk": 25, "float16_flow": False} and res["config_path"] == str(root / "configs/engine/head.json")
    (root / "configs" / "engine" / "head.json").write_text(json.dumps({"first_chunk": 8, "float16_flow": True}))
    b = _cand()
    assert a.digest != b.digest  # the reproduced review case: same path, new precision/ramp
    (root / "engine" / "runtime" / "dit_fast.py").write_text("SCALE = 2.0\n")
    c = _cand()
    assert c.digest != b.digest  # a kernel change outside configs
    assert _cand().digest == c.digest  # stable when nothing changes
    # a path config resolves like the launcher (an absolute path or *.json is used as is)
    other = tmp_path / "other.json"
    other.write_text("{}")
    assert cli.resolved_deploy("engine", str(other), [], {})["config_sha256"] is not None


def test_engine_fingerprints_and_resume_bind_the_implementation(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    monkeypatch.setattr(cli, "ROOT", root)
    monkeypatch.setattr(cli, "repo_fingerprint", lambda p, paths=(): {"commit": "c0", "dirty": False, "diff_sha256": "d0"} if Path(p) == root
                        else {"commit": "x", "dirty": False, "diff_sha256": "x"})
    fp = cli.execution_fingerprints("engine")
    assert fp["engine"]["repo"]["commit"] == "c0" and fp["engine"]["tree_sha256"]
    assert "engine" not in cli.execution_fingerprints("vllm-omni")  # baseline experiments keep their earlier fields
    cell = cli.make_cell(_cand(), 1, 2, "tuning", 0, 180.0, 0)
    spec = cli.build_experiment(cell, "performance", "restart", True, True, True, False)
    (root / "engine" / "runtime" / "dit_fast.py").write_text("SCALE = 3.0\n")
    cell2 = cli.make_cell(_cand(), 1, 2, "tuning", 0, 180.0, 0)
    spec2 = cli.build_experiment(cell2, "performance", "restart", True, True, True, False)
    assert spec_base_digest(spec) != spec_base_digest(spec2)  # resume compares base digests: the old cell is set aside
    # frozen references are inside the spec (and only when declared: undeclared specs are unchanged)
    spec3 = cli.build_experiment(cell2, "performance", "restart", True, True, True, False, {"speech_qref": {"path": "/q", "sha256": "ab"}})
    assert spec3["frozen_refs"]["speech_qref"]["sha256"] == "ab" and spec_base_digest(spec3) != spec_base_digest(spec2)
    assert "frozen_refs" not in spec2
