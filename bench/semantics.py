"""Reference semantics of a baseline candidate, from its resolved deployment.

Class R is decided from what the server will actually run, never from a file
name.  For vllm-omni the deploy YAML goes through vllm-omni's own resolver
(``StageConfigFactory``: inheritance, CUDA platform overrides, pipeline-wide
fields such as ``dtype``/``quantization`` propagated into every stage, stage
overrides), and the classification reads the final per-stage engine arguments,
runtime environments and the connector settings Code2Wav consumes.  The deploy
files are first checked against the hashes the measured candidate recorded, so
today's files can only classify the configuration that was measured.

Fail closed: a launch argument, environment variable (candidate or stage) or
setting whose effect on the classified semantics is unknown raises
``SemanticsUnknown``, which is never class R.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

from bench.eligibility import ConfigClass

VLLM_REPO = Path("/workspace/cgzheng/cgzheng/vllm-omni")
VLLM_PYTHON = str(VLLM_REPO / ".venv/bin/python")
VLLM_DEPLOY_DIR = VLLM_REPO / "vllm_omni" / "deploy"
SGLANG_REPO = Path("/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main")
SGLANG_CONFIG = SGLANG_REPO / "sglang_omni" / "models" / "minicpm_o" / "config.py"
MODEL_PATH = os.environ.get(
    "MODEL_PATH", "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"
)

#: CLI arguments known not to change the classified semantics (value-taking flags list their arity).
SEMANTIC_NEUTRAL_ARGS = {"vllm-omni": {}, "sglang-omni": {"--thinker.factory.max_seq_len": 1}}
#: Launch-environment variables known not to change the classified semantics; anything else is unknown.
NEUTRAL_LAUNCH_ENV = {"vllm-omni": set(), "sglang-omni": {"OMNI_SGLANG_MPS", "OMNI_SGLANG_MAX_SEQ"}}
#: Stage-runtime environment variables of the pinned vllm-omni profiles: scheduling only, no numerics.
NEUTRAL_STAGE_ENV = {"VLLM_OMNI_CODEC_FIRST_CHUNK_EXPRESS", "VLLM_OMNI_CODEC_FIRST_CHUNK_EXPRESS_SLACK_S"}
#: Official stream geometry: 25-token chunks, no special first chunk, 3 frames of left context.
OFFICIAL_GEOMETRY = {"codec_chunk_frames": 25, "initial_codec_chunk_frames": 0, "codec_left_context_frames": 3}
SHIPPED_DTYPES = (None, "auto", "bfloat16", "bf16")

_RESOLVE = r"""
import json, sys
from vllm_omni.config import stage_config as sc
from vllm_omni.config.config_factory import StageConfigFactory
orig = sc._apply_platform_overrides
sc._apply_platform_overrides = lambda deploy, platform=None: orig(deploy, "cuda")  # the measured servers run on CUDA
out = []
for path in sys.argv[2:]:
    stages, _ = StageConfigFactory.create_legacy_stage_configs_from_model(sys.argv[1], trust_remote_code=True, cli_overrides={},
                                                                           deploy_config_path=path)
    deploy = orig(sc.load_deploy_config(path), "cuda")
    out.append({"stages": [{"stage_id": s.stage_id, "engine_args": s.yaml_engine_args, "runtime": s.yaml_runtime} for s in stages],
                "connectors": deploy.connectors, "session_mode": deploy.session_mode, "cuda_mps": bool(deploy.cuda_mps)})
print(json.dumps(out, default=str))
"""


class SemanticsUnknown(ValueError):
    pass


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def deploy_chain(config: str) -> list[dict]:
    """The YAML inheritance chain exactly as the harness hashes it for the candidate identity."""
    import yaml

    path = Path(config) if config.startswith("/") else VLLM_DEPLOY_DIR / config
    chain, seen = [], set()
    while path is not None and path not in seen:
        seen.add(path)
        text = path.read_text()
        chain.append({"file": str(path), "sha256": _sha(text)})
        base = (yaml.safe_load(text) or {}).get("base_config")
        path = (path.parent / base) if base else None
    return chain


_CACHE: dict[tuple, dict] = {}


def _key(config: str) -> tuple:
    path = Path(config) if config.startswith("/") else VLLM_DEPLOY_DIR / config
    return (str(path), tuple(c["sha256"] for c in deploy_chain(config)))


def resolve_vllm_configs(configs: list[str]) -> None:
    """Resolve several deploy configs in one resolver process (memoized by path and file hashes)."""
    todo = sorted({_key(c) for c in configs if _key(c) not in _CACHE})
    if not todo:
        return
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "", "HF_HUB_OFFLINE": "1",
           "HF_MODULES_CACHE": os.environ.get("HF_MODULES_CACHE", "/workspace/models/cgzheng/hf_modules")}
    out = subprocess.run([VLLM_PYTHON, "-c", _RESOLVE, MODEL_PATH, *(k[0] for k in todo)], capture_output=True, text=True, env=env, cwd=str(VLLM_REPO))
    if out.returncode != 0:
        raise SemanticsUnknown(f"vllm-omni resolver failed for {[k[0] for k in todo]}: {out.stderr.strip().splitlines()[-1:]}")
    for k, r in zip(todo, json.loads(out.stdout.strip().splitlines()[-1])):
        _CACHE[k] = r


def resolve_vllm_config(config: str) -> dict:
    """Final per-stage engine args, runtimes and connectors from the pinned vllm-omni resolver."""
    resolve_vllm_configs([config])
    return json.loads(json.dumps(_CACHE[_key(config)]))


def _check_args(adapter: str, extra: list[str]) -> None:
    allowed = SEMANTIC_NEUTRAL_ARGS.get(adapter, {})
    i = 0
    while i < len(extra):
        flag = extra[i].split("=", 1)[0]
        if flag not in allowed:
            raise SemanticsUnknown(f"argument {extra[i]!r} has unknown effect on reference semantics")
        i += 1 + (0 if "=" in extra[i] else allowed[flag])


def _check_launch_env(adapter: str, env: dict) -> None:
    unknown = sorted(k for k in env if k not in NEUTRAL_LAUNCH_ENV.get(adapter, set()))
    if unknown:
        raise SemanticsUnknown(f"environment {unknown} has unknown effect on reference semantics")


def _precision(args: dict, name: str) -> str:
    if args.get("quantization") not in (None, "", "none"):
        return str(args["quantization"])
    if args.get("kv_cache_dtype") not in (None, "auto"):
        return f"kv:{args['kv_cache_dtype']}"
    dtype = args.get("dtype")
    if dtype in SHIPPED_DTYPES:
        return "bf16"  # the checkpoint's bfloat16 (``auto``) is the shipped default
    return str(dtype)


def vllm_semantics(resolved: dict, extra: list[str], env: dict) -> dict:
    _check_args("vllm-omni", extra)
    _check_launch_env("vllm-omni", env)
    stages = {int(s["stage_id"]): s for s in resolved.get("stages", [])}
    if set(stages) != {0, 1, 2}:
        raise SemanticsUnknown(f"unexpected stage layout {sorted(stages)}")
    for sid, s in stages.items():
        unknown = sorted(k for k in ((s.get("runtime") or {}).get("env") or {}) if k not in NEUTRAL_STAGE_ENV)
        if unknown:
            raise SemanticsUnknown(f"stage {sid} environment {unknown} has unknown effect on reference semantics")
    conns = resolved.get("connectors") or {}
    if set(conns) != {"connector_of_shared_memory"}:
        raise SemanticsUnknown(f"unexpected connector layout {sorted(conns)}")
    conn = conns["connector_of_shared_memory"].get("extra") or {}
    geometry = {k: int(conn.get(k, d)) for k, d in (("codec_chunk_frames", 25), ("initial_codec_chunk_frames", 0), ("codec_left_context_frames", 3))}
    if geometry["initial_codec_chunk_frames"] == geometry["codec_chunk_frames"]:
        geometry["initial_codec_chunk_frames"] = 0  # an explicit first chunk of the regular size is the regular geometry
    s2 = stages[2]["engine_args"]
    if conn.get("token2wav_trt") or conn.get("token2wav_float16") or s2.get("dtype") not in SHIPPED_DTYPES + ("float32", "fp32"):
        flow = "fp16"
    elif conn.get("token2wav_allow_tf32", conn.get("code2wav_allow_tf32", True)) in (True, "true", "1", 1, "tf32", "flow"):
        flow = "tf32"
    else:
        flow = "fp32"
    session = {str(s["engine_args"].get("session_mode")) for s in stages.values()}
    if len(session) != 1 or session != {str(resolved.get("session_mode"))}:
        raise SemanticsUnknown(f"stages disagree on session mode: {sorted(session)}")
    mps = {bool((s.get("runtime") or {}).get("cuda_mps", False)) for s in stages.values()}
    if len(mps) != 1:
        raise SemanticsUnknown("stages disagree on CUDA MPS")
    return {
        "track": "T",  # vllm-omni conditions the Talker on the completed Thinker reply in turn and duplex modes alike
        "audio_output": "stream",
        "chunk_tokens": geometry["codec_chunk_frames"] if geometry == OFFICIAL_GEOMETRY else f"nonofficial:{json.dumps(geometry, sort_keys=True)}",
        "chunk_geometry": geometry,
        "precision": {"thinker": _precision(stages[0]["engine_args"], "thinker"), "talker": _precision(stages[1]["engine_args"], "talker"),
                      "flow": flow, "hift": "fp32"},
        "euler_steps": int(conn.get("token2wav_n_timesteps", 10)),
        "session_mode": session.pop(),
        "mps": mps.pop(),
        "resolved_sha256": hashlib.sha256(json.dumps(resolved, sort_keys=True).encode()).hexdigest(),
    }


def sglang_files() -> dict:
    root = Path(__file__).resolve().parents[1]
    return {"serve_script_sha256": _sha((root / "scripts" / "serve_sglang_omni.sh").read_text()),
            "model_config_py_sha256": _sha(SGLANG_CONFIG.read_text())}


def sglang_semantics(extra: list[str], env: dict) -> dict:
    _check_args("sglang-omni", extra)
    _check_launch_env("sglang-omni", env)
    text = SGLANG_CONFIG.read_text()
    if 'dtype="float16"' not in text or "def speech_stages" not in text:
        raise SemanticsUnknown("pinned sglang-omni code2wav stage changed")
    return {"track": "T", "audio_output": "full", "chunk_tokens": None, "chunk_geometry": None,
            "precision": {"thinker": "bf16", "talker": "bf16", "flow": "fp16", "hift": "fp32"}, "euler_steps": 10, "session_mode": "turn",
            "mps": env.get("OMNI_SGLANG_MPS") == "1", "resolved_sha256": hashlib.sha256(json.dumps(sglang_files(), sort_keys=True).encode()).hexdigest()}


def check_deploy_binding(candidate: dict) -> None:
    """Today's deploy files must be the bytes the measured candidate recorded."""
    recorded = list(candidate.get("deploy_files_sha256") or [])
    if candidate["adapter"] == "vllm-omni":
        now = [c["sha256"] for c in deploy_chain(candidate["config"])]
    elif candidate["adapter"] == "sglang-omni":
        now = [v for _, v in sorted(sglang_files().items())]
    else:
        raise SemanticsUnknown(f"{candidate['adapter']} is not a baseline")
    if now != recorded:
        raise SemanticsUnknown(f"deploy files of {candidate.get('candidate_id', candidate['config'])} differ from the measured candidate's")


def candidate_semantics(candidate: dict, check_binding: bool = True) -> dict:
    """``candidate``: the identity dict stored in an experiment spec (adapter, config, extra, env, mps, deploy hashes)."""
    env = {k: v for k, v in candidate.get("env", [])} if isinstance(candidate.get("env"), list) else dict(candidate.get("env") or {})
    extra = list(candidate.get("extra") or [])
    if candidate["adapter"] not in ("vllm-omni", "sglang-omni"):
        raise SemanticsUnknown(f"{candidate['adapter']} is not a baseline")
    if check_binding:
        check_deploy_binding(candidate)
    if candidate["adapter"] == "vllm-omni":
        sem = vllm_semantics(resolve_vllm_config(candidate["config"]), extra, env)
    else:
        sem = sglang_semantics(extra, env)
    want_mps = candidate.get("mps", "off") == "on"
    if sem["mps"] != want_mps:
        raise SemanticsUnknown(f"declared MPS {candidate.get('mps')} but the launch configures mps={sem['mps']}")
    return sem


def class_r(candidate: dict, canonical_inputs: bool, check_binding: bool = True) -> ConfigClass:
    """Fail closed: unknown semantics raise ``SemanticsUnknown``."""
    s = candidate_semantics(candidate, check_binding)
    return ConfigClass(engine=candidate["adapter"], config=candidate["config"], track=s["track"], audio_output=s["audio_output"],
                       chunk_tokens=s["chunk_tokens"], precision=s["precision"], euler_steps=s["euler_steps"], session_mode=s["session_mode"],
                       canonical_inputs=canonical_inputs, mps=s["mps"])
