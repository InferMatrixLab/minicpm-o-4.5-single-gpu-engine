"""Local locations of the checkpoint, datasets, baseline checkouts and their Python environments.

Every location can be overridden by its environment variable (``scripts/env.sh`` lists them).  The defaults assume the
Hugging Face cache layout and the baseline checkouts next to this repository.  Standard library only: the harness also
runs inside the baseline environments.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _path(var: str, default) -> Path:
    return Path(os.environ.get(var) or default)


HF_HOME = _path("HF_HOME", Path.home() / ".cache" / "huggingface")
HF_HUB_CACHE = _path("HF_HUB_CACHE", HF_HOME / "hub")
HF_MODULES_CACHE = _path("HF_MODULES_CACHE", HF_HOME / "modules")
HF_DATASETS_CACHE = _path("HF_DATASETS_CACHE", HF_HOME / "datasets")


def hub_snapshot(repo: str, revision: str, kind: str = "models") -> Path:
    """The Hugging Face cache directory of one pinned snapshot."""
    return HF_HUB_CACHE / f"{kind}--{repo.replace('/', '--')}" / "snapshots" / revision


MODEL_REPO = "openbmb/MiniCPM-o-4_5"
MODEL_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"
MODEL_PATH = _path("MODEL_PATH", hub_snapshot(MODEL_REPO, MODEL_REVISION))

SEED_TTS_EVAL = hub_snapshot("zhaochenyang20/seed-tts-eval", "8f5e1aa2a35d42f42e940074c1983358b9491f89", "datasets")
DAILY_OMNI = hub_snapshot("liarliar/Daily-Omni", "bf5a6ee4c829510b7a14869e3104efeafc330b07", "datasets")
QWEN3_ASR = hub_snapshot("Qwen/Qwen3-ASR-1.7B", "7278e1e70fe206f11671096ffdd38061171dd6e5")
WHISPER_LARGE_V3 = hub_snapshot("openai/whisper-large-v3", "06f233fe06e710322aca913c1bc4249a0d71fce1")

VLLM_OMNI_REPO = _path("OMNI_VLLM_REPO", ROOT.parent / "vllm-omni")
SGLANG_OMNI_REPO = _path("OMNI_SGLANG_REPO", ROOT.parent / "sglang-omni")
VLLM_OMNI_PYTHON = _path("OMNI_VLLM_PYTHON", VLLM_OMNI_REPO / ".venv" / "bin" / "python")
ENGINE_PYTHON = _path("ENGINE_PYTHON", ROOT / ".venv-engine" / "bin" / "python")
#: sglang-omni's speech evaluation tools (Seed-TTS WER and speaker similarity) and the ASR server environment
EVAL_ENV = _path("OMNI_EVAL_ENV", ROOT.parent / "omni-eval-env")
ASR_ENV = _path("OMNI_ASR_ENV", ROOT.parent / "omni-asr-env")
