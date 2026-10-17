"""Evaluator identity: verified before any scorer model is loaded.

The fingerprint covers the pinned sglang-omni scorer checkout (clean, at the
pinned commit), both evaluation environments (their package freezes must equal
the committed locks), every scorer checkpoint and ASR/Whisper weight file
(SHA-256 against ``evals/assets.lock.json``), the evaluator source files, and
the effective settings (batch sizes, precision, decoding).  Quality reports and
Q_refs carry its digest; references and gates only combine equal fingerprints.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SGLANG_DIR = Path("/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main")
SGLANG_COMMIT = "22d4ace88f9547777b819a69d7691204bb9ce847"
EVAL_PYTHON = "/workspace/cgzheng/cgzheng/omni-baselines/eval-env/bin/python"
ASR_PYTHON = "/workspace/cgzheng/cgzheng/omni-baselines/asr-env/bin/python"
LOCKS = {"eval-env": (EVAL_PYTHON, ROOT / "evals" / "eval-env.lock"), "asr-env": (ASR_PYTHON, ROOT / "evals" / "asr-env.lock")}
ASSETS_LOCK = ROOT / "evals" / "assets.lock.json"
SOURCES = ("evals/speech/asr_worker.py", "evals/speech/scorer_worker.py", "evals/speech/wavlm_blocked.py", "evals/speech/detectors.py",
           "evals/speech/gates.py", "evals/speech/suite.py", "evals/evaluator.py")
SETTINGS = {
    "asr": {"model": "Qwen/Qwen3-ASR-1.7B", "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5", "batch_size": 16, "dtype": "bfloat16",
            "decoding": "greedy", "language": "English"},
    "whisper": {"model": "openai/whisper-large-v3", "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1", "batch_size": 16,
                "dtype": "bfloat16", "decoding": "greedy", "language": "en"},
    "wer_normalizer": "sglang-omni benchmarks.tasks.asr.apply_wer (whisper EnglishTextNormalizer + jiwer)",
    # attention v1 (round 9): the same batches and arithmetic, the attention in 256-row query blocks (an 82 s output did not fit)
    "sim": {"scorer": "WavLMSpeakerSimilarity", "scale": "x100", "batch": 16, "attention": "query_blocked_v1", "query_block": 256},
    "utmos": {"scorer": "UTMOSScorer"},
}


class EvaluatorMismatch(RuntimeError):
    pass


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _freeze(python: str) -> str:
    return subprocess.run(["uv", "pip", "freeze", "--python", python], capture_output=True, text=True, check=True).stdout


def verify_evaluator(check_weights: bool = True) -> dict:
    """Raise ``EvaluatorMismatch`` on any deviation; return the fingerprint record."""
    problems = []
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=SGLANG_DIR, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=SGLANG_DIR, capture_output=True, text=True).stdout.strip()
    if commit != SGLANG_COMMIT:
        problems.append(f"scorer checkout at {commit}, pinned {SGLANG_COMMIT}")
    if dirty:
        problems.append("scorer checkout has modified tracked files")
    envs = {}
    for name, (py, lock) in LOCKS.items():
        frozen = _freeze(py)
        if frozen != lock.read_text():
            problems.append(f"{name} packages differ from {lock.name}")
        envs[name] = hashlib.sha256(frozen.encode()).hexdigest()
    assets = json.loads(ASSETS_LOCK.read_text())
    files = {}
    for path, info in assets["files"].items():
        p = Path(path)
        got = _sha_file(p) if p.exists() else None
        if got != info["sha256"]:
            problems.append(f"scorer checkpoint {p} hash mismatch")
        files[path] = got
    models = {}
    for name, m in assets["models"].items():
        d = Path(m["dir"])
        models[name] = {"dir": str(d)}
        if check_weights:
            for fname, want in m["files"].items():
                p = (d / fname).resolve()
                if not p.exists() or _sha_file(p) != want:
                    problems.append(f"{name} file {fname} hash mismatch")
            models[name]["files_sha256"] = hashlib.sha256(json.dumps(m["files"], sort_keys=True).encode()).hexdigest()
    sources = {s: _sha_file(ROOT / s) for s in SOURCES}
    if problems:
        raise EvaluatorMismatch("; ".join(problems))
    record = {"scorer_checkout": {"path": str(SGLANG_DIR), "commit": commit}, "envs": envs, "checkpoints": files, "models": models,
              "sources": sources, "settings": SETTINGS}
    record["digest"] = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
    return record
