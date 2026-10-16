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


def test_reserved_flags_missing_config_and_relative_paths(tmp_path, monkeypatch):
    """Extras may not override --config/--backend (the launcher's argparse takes the last one); a missing config fails at
    declaration; a relative *.json resolves against the repository root, as the launcher (which runs there) does."""
    import pytest

    root = _tree(tmp_path)
    monkeypatch.setattr(cli, "ROOT", root)
    for extra in (["--config", str(root / "configs/engine/head.json")], ["--config=/x.json"], ["--backend", "a:B"], ["--backend=a:B"]):
        with pytest.raises(ValueError, match="may not override"):
            cli.make_candidate({"adapter": "engine", "config": "head", "extra": extra})
    with pytest.raises(ValueError, match="does not exist"):
        cli.make_candidate({"adapter": "engine", "config": "nope"})
    other = tmp_path / "elsewhere"
    other.mkdir()
    monkeypatch.chdir(other)
    rel = cli.resolve_engine_deploy("configs/engine/head.json")
    assert rel["config_path"] == str(root / "configs/engine/head.json") and rel["config_sha256"]
    assert rel["backend"] == cli.ENGINE_DEFAULT_BACKEND
    (root / "configs" / "engine" / "tb.json").write_text(json.dumps({"backend": "engine.runtime.backend_s:TrackSBackend"}))
    assert cli.served_identity_expected("tb")["backend"] == "engine.runtime.backend_s:TrackSBackend"


def test_served_identity_mismatch_fails_before_load(tmp_path, monkeypatch):
    """A server whose /info differs from the declaration is stopped by Server.start before any request is sent."""
    import socket

    import pytest

    import bench.servers
    from bench.servers import Server, ServerSpec

    monkeypatch.setattr(bench.servers, "gpu_memory_used_mib", lambda gpu: None)  # the fake holds no GPU memory

    fake = ROOT / "tests" / "bench" / "fake_engine_server.py"
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    want = cli.served_identity_expected("fake")
    bad = {**want, "config_sha256": "0" * 64}
    srv = Server(ServerSpec(name="fake", cmd=[sys.executable, str(fake), str(port)], port=port, gpu=0, info_path="/info", expected_identity=bad),
                 log_path=tmp_path / "srv.log")
    with pytest.raises(RuntimeError, match="served identity differs"):
        srv.start()
    from bench.servers import _healthy

    assert not _healthy(f"http://127.0.0.1:{port}/health")  # stopped before any load
    ok = Server(ServerSpec(name="fake", cmd=[sys.executable, str(fake), str(port)], port=port, gpu=0, info_path="/info", expected_identity=want),
                log_path=tmp_path / "srv2.log")
    with ok:
        ev = ok.evidence()
    assert ev["served_identity_problems"] == [] and ev["served_identity"]["engine_tree_sha256"] == want["engine_tree_sha256"]


def _fake_python(dirpath: Path, dists: str) -> Path:
    """An executable that answers the distributions query with ``dists`` (from a file next to it, so it can drift)."""
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "dists").write_text(dists)
    py = dirpath / "python"
    py.write_text('#!/bin/bash\ncat "$(dirname "$0")/dists"\n')
    py.chmod(0o755)
    return py


def test_interpreter_is_resolved_once_and_bound_to_launch_fingerprint_and_served_identity(tmp_path, monkeypatch):
    """B15: the candidate's ENGINE_PYTHON (else this tree's .venv-engine, else the main checkout's) is launched,
    fingerprinted, put in the manifest and declared as served identity (path and installed distributions)."""
    cli.env_distributions_sha256.cache_clear()
    py = _fake_python(tmp_path / "envA" / "bin", "dists-A")
    env = {"ENGINE_PYTHON": str(py)}
    assert cli.engine_python(env) == str(py)
    assert cli.engine_python({"ENGINE_PYTHON": str(tmp_path / "missing" / "python")}) == cli.ENGINE_PYTHON  # the launcher's fallback
    assert cli.engine_python({}) in (str(cli.ROOT / ".venv-engine" / "bin" / "python"), cli.ENGINE_PYTHON)
    want = cli.served_identity_expected("track_s_fast", env)
    assert want["python"] == str(py) and want["python_dists_sha256"] == "dists-A"
    cand = cli.make_candidate({"adapter": "engine", "config": "track_s_fast", "label": "envA", "mps": "off", "env": env})
    spec = cli.server_spec(cand, 18999, 0)
    assert spec.env["ENGINE_PYTHON"] == str(py) and spec.expected_identity == want
    seen = []
    monkeypatch.setattr(cli, "env_fingerprint", lambda p: seen.append(p) or {"freeze_sha256": f"freeze:{p}"})
    fp = cli.execution_fingerprints("engine", env)
    assert fp["envs"]["engine"] == f"freeze:{py}" and str(py) in seen
    assert cli.adapter_python("engine", env) == str(py)


def test_package_drift_at_the_same_path_changes_the_resume_digest(tmp_path, monkeypatch):
    cli.env_distributions_sha256.cache_clear()
    py = _fake_python(tmp_path / "envB" / "bin", "dists-1")
    env = {"ENGINE_PYTHON": str(py)}
    cand = cli.make_candidate({"adapter": "engine", "config": "track_s_fast", "label": "envB", "mps": "off", "env": env})
    cell = cli.make_cell(cand, 1, 2, "tuning", 0, 180.0, 0)
    monkeypatch.setattr(cli, "env_fingerprint", lambda p: {"freeze_sha256": Path(p).with_name("dists").read_text() if p == str(py) else "bench"})
    spec1 = cli.build_experiment(cell, "performance", "restart", True, True, True, False)
    (py.parent / "dists").write_text("dists-2")  # same interpreter path, other packages
    cli.env_distributions_sha256.cache_clear()
    spec2 = cli.build_experiment(cell, "performance", "restart", True, True, True, False)
    assert spec1["fingerprints"]["engine"]["served_identity"]["python_dists_sha256"] == "dists-1"
    assert spec2["fingerprints"]["engine"]["served_identity"]["python_dists_sha256"] == "dists-2"
    assert spec_base_digest(spec1) != spec_base_digest(spec2)  # a declared cell is not resumed under the drifted environment


def test_foreign_interpreter_or_drifted_packages_stop_the_server_before_load(tmp_path, monkeypatch):
    import os
    import socket

    import pytest

    import bench.servers
    from bench.servers import Server, ServerSpec

    monkeypatch.setattr(bench.servers, "gpu_memory_used_mib", lambda gpu: None)
    fake = ROOT / "tests" / "bench" / "fake_engine_server.py"
    want = cli.served_identity_expected("fake")
    for override in ({"python": "/different/env/bin/python"}, {"python_dists_sha256": "0" * 64}):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]
        srv = Server(ServerSpec(name="fake", cmd=[sys.executable, str(fake), str(port)], port=port, gpu=0, info_path="/info", expected_identity=want,
                                env={"FAKE_INFO_OVERRIDE": json.dumps(override)}), log_path=tmp_path / f"srv{port}.log")
        assert os.environ.get("FAKE_INFO_OVERRIDE") is None
        with pytest.raises(RuntimeError, match="served identity differs"):
            srv.start()
