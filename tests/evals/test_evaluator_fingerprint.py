"""The evaluator is verified before any scorer model loads (B9)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import evals.evaluator as ev  # noqa: E402


def test_current_evaluator_verifies():
    rec = ev.verify_evaluator(check_weights=False)
    assert rec["scorer_checkout"]["commit"] == ev.SGLANG_COMMIT and len(rec["digest"]) == 64
    assert rec["settings"]["asr"]["batch_size"] == 16 and rec["settings"]["asr"]["dtype"] == "bfloat16"


def test_environment_drift_rejected(tmp_path, monkeypatch):
    lock = tmp_path / "eval-env.lock"
    lock.write_text(ev.LOCKS["eval-env"][1].read_text() + "extra-package==1.0\n")
    monkeypatch.setitem(ev.LOCKS, "eval-env", (ev.LOCKS["eval-env"][0], lock))
    with pytest.raises(ev.EvaluatorMismatch, match="eval-env"):
        ev.verify_evaluator(check_weights=False)


def test_checkout_and_checkpoint_mismatch_rejected(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "SGLANG_COMMIT", "0" * 40)
    with pytest.raises(ev.EvaluatorMismatch, match="scorer checkout"):
        ev.verify_evaluator(check_weights=False)
    monkeypatch.undo()
    import json

    lock = json.loads(ev.ASSETS_LOCK.read_text())
    first = next(iter(lock["files"]))
    lock["files"][first]["sha256"] = "f" * 64
    p = tmp_path / "assets.lock.json"
    p.write_text(json.dumps(lock))
    monkeypatch.setattr(ev, "ASSETS_LOCK", p)
    with pytest.raises(ev.EvaluatorMismatch, match="checkpoint"):
        ev.verify_evaluator(check_weights=False)
