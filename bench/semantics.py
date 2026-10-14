"""Reference semantics of a baseline candidate, from its resolved deployment.

Class R is decided from what the server will actually run, never from a file
name: the vllm-omni deploy YAML is resolved with vllm-omni's own
``resolve_deploy_yaml`` (inheritance and overlays), then the CLI arguments and
launch environment are applied.  Any argument or environment variable whose
effect on the classified semantics is unknown makes the result "unknown",
which is never class R.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from bench.eligibility import ConfigClass

VLLM_PYTHON = "/workspace/cgzheng/cgzheng/vllm-omni/.venv/bin/python"
VLLM_DEPLOY_DIR = Path("/workspace/cgzheng/cgzheng/vllm-omni/vllm_omni/deploy")
SGLANG_CONFIG = Path("/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main/sglang_omni/models/minicpm_o/config.py")
SGLANG_CONFIG_SHA256 = None  # filled from the pinned checkout at first use

#: CLI arguments known not to change the classified semantics (value-taking flags list their arity).
SEMANTIC_NEUTRAL_ARGS = {"vllm-omni": {}, "sglang-omni": {"--thinker.factory.max_seq_len": 1}}
#: Environment variables that change classified semantics (reject unless at their default).
SEMANTIC_ENV = {"MINICPMO_CODE2WAV_TF32", "MINICPMO_TOKEN2WAV_TRT", "MINICPMO_TOKEN2WAV_TRT_DTYPE"}


class SemanticsUnknown(ValueError):
    pass


def resolve_vllm_config(config: str) -> dict:
    path = Path(config) if config.startswith("/") else VLLM_DEPLOY_DIR / config
    code = ("import json,sys; from vllm_omni.config.stage_config import resolve_deploy_yaml; "
            "print(json.dumps(resolve_deploy_yaml(sys.argv[1]), default=str))")
    out = subprocess.run([VLLM_PYTHON, "-c", code, str(path)], capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[-1])


def _check_args(adapter: str, extra: list[str]) -> None:
    allowed = SEMANTIC_NEUTRAL_ARGS.get(adapter, {})
    i = 0
    while i < len(extra):
        flag = extra[i].split("=", 1)[0]
        if flag not in allowed:
            raise SemanticsUnknown(f"argument {extra[i]!r} has unknown effect on reference semantics")
        i += 1 + (0 if "=" in extra[i] else allowed[flag])


def vllm_semantics(resolved: dict, extra: list[str], env: dict) -> dict:
    _check_args("vllm-omni", extra)
    bad_env = sorted(k for k in env if k in SEMANTIC_ENV)
    if bad_env:
        raise SemanticsUnknown(f"environment {bad_env} changes Code2Wav semantics")
    if "session_mode" not in resolved:
        raise SemanticsUnknown("deploy config does not declare session_mode")
    conn = ((resolved.get("connectors") or {}).get("connector_of_shared_memory") or {}).get("extra") or {}
    if "codec_chunk_frames" not in conn:
        raise SemanticsUnknown("deploy config does not declare codec_chunk_frames")
    stages = {int(s["stage_id"]): s for s in resolved.get("stages", [])}
    if set(stages) != {0, 1, 2}:
        raise SemanticsUnknown(f"unexpected stage layout {sorted(stages)}")
    for sid, name in ((0, "thinker"), (1, "talker")):
        s = stages[sid]
        if s.get("quantization") not in (None, "", "none"):
            raise SemanticsUnknown(f"{name} quantization {s['quantization']} (class R requires bf16)")
        if s.get("dtype") not in (None, "auto", "bfloat16", "bf16"):
            raise SemanticsUnknown(f"{name} dtype {s['dtype']}")
    if conn.get("token2wav_trt") or conn.get("token2wav_float16"):
        flow = "fp16"
    elif conn.get("token2wav_allow_tf32", conn.get("code2wav_allow_tf32", True)) in (True, "true", "1", 1):
        flow = "tf32"
    else:
        flow = "fp32"
    return {
        "track": "T",  # vllm-omni conditions the Talker on the completed Thinker reply in turn and duplex modes alike
        "audio_output": "stream",
        "chunk_tokens": int(conn["codec_chunk_frames"]),
        "precision": {"thinker": "bf16", "talker": "bf16", "flow": flow, "hift": "fp32"},
        "euler_steps": int(conn.get("token2wav_n_timesteps", 10)),
        "session_mode": str(resolved["session_mode"]),
        "mps": bool(resolved.get("cuda_mps", False)),
    }


SGLANG_NEUTRAL_ENV = {"OMNI_SGLANG_MPS", "OMNI_SGLANG_MAX_SEQ"}


def sglang_semantics(extra: list[str], env: dict) -> dict:
    _check_args("sglang-omni", extra)
    unknown = sorted(k for k in env if k not in SGLANG_NEUTRAL_ENV)
    if unknown:
        raise SemanticsUnknown(f"environment {unknown} has unknown effect on reference semantics")
    text = SGLANG_CONFIG.read_text()
    if 'dtype="float16"' not in text or "def speech_stages" not in text:
        raise SemanticsUnknown("pinned sglang-omni code2wav stage changed")
    return {"track": "T", "audio_output": "full", "chunk_tokens": None,
            "precision": {"thinker": "bf16", "talker": "bf16", "flow": "fp16", "hift": "fp32"}, "euler_steps": 10, "session_mode": "turn",
            "mps": env.get("OMNI_SGLANG_MPS") == "1"}


def candidate_semantics(candidate: dict) -> dict:
    """``candidate``: the identity dict stored in an experiment spec (adapter, config, extra, env, mps)."""
    env = {k: v for k, v in candidate.get("env", [])} if isinstance(candidate.get("env"), list) else dict(candidate.get("env") or {})
    extra = list(candidate.get("extra") or [])
    if candidate["adapter"] == "vllm-omni":
        sem = vllm_semantics(resolve_vllm_config(candidate["config"]), extra, env)
    elif candidate["adapter"] == "sglang-omni":
        sem = sglang_semantics(extra, env)
    else:
        raise SemanticsUnknown(f"{candidate['adapter']} is not a baseline")
    want_mps = candidate.get("mps", "off") == "on"
    if sem["mps"] != want_mps:
        raise SemanticsUnknown(f"declared MPS {candidate.get('mps')} but the launch configures mps={sem['mps']}")
    return sem


def class_r(candidate: dict, canonical_inputs: bool) -> ConfigClass:
    """Fail closed: unknown semantics raise ``SemanticsUnknown``."""
    s = candidate_semantics(candidate)
    return ConfigClass(engine=candidate["adapter"], config=candidate["config"], track=s["track"], audio_output=s["audio_output"],
                       chunk_tokens=s["chunk_tokens"], precision=s["precision"], euler_steps=s["euler_steps"], session_mode=s["session_mode"],
                       canonical_inputs=canonical_inputs, mps=s["mps"])
