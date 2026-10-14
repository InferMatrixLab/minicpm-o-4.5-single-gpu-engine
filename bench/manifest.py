"""Reproducibility manifest embedded in every result JSON, and its validator."""

from __future__ import annotations

import hashlib
import os
import socket
import subprocess
import time
from pathlib import Path

from bench.workload import MODEL_REPO, MODEL_REVISION

REQUIRED_HARDWARE_KEYS = (
    "gpu_sku",
    "gpu_memory_mib",
    "driver",
    "clocks",
    "power_limit_w",
    "mps",
    "cpu_affinity",
)


def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=60, check=True).stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def git_info(path: str | Path) -> dict:
    path = str(path)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=path)
    status = _run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=path)
    return {"path": path, "commit": commit or None, "dirty": bool(status) if commit else None}


def env_lock(python: str, out_dir: Path, name: str) -> dict:
    """Freeze the environment of ``python`` to a file and hash it."""
    frozen = _run(["uv", "pip", "freeze", "--python", python])
    if not frozen:
        frozen = _run([python, "-m", "pip", "freeze"])
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"env-{name}.lock"
    data = (frozen + "\n").encode()
    path.write_bytes(data)
    return {"python": python, "path": str(path), "sha256": hashlib.sha256(data).hexdigest(), "packages": len(frozen.splitlines())}


def _mps_state() -> dict:
    procs = _run(["pgrep", "-a", "nvidia-cuda-mps"])
    return {
        "daemon_running": bool(procs),
        "pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "active_thread_percentage": os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"),
    }


def gpu_info(index: int) -> dict:
    q = "name,memory.total,driver_version,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.limit,compute_mode,pci.bus_id"
    line = _run(["nvidia-smi", "-i", str(index), f"--query-gpu={q}", "--format=csv,noheader,nounits"])
    fields = [f.strip() for f in line.split(",")] if line else []
    if len(fields) < 10:
        return {"gpu_index": index, "error": "nvidia-smi query failed"}
    name, mem, driver, sm, memclk, max_sm, max_mem, power, compute_mode, bus = fields[:10]

    def num(x: str):
        try:
            return float(x)
        except ValueError:
            return None

    return {
        "gpu_index": index,
        "gpu_sku": name,
        "gpu_memory_mib": num(mem),
        "driver": driver,
        "clocks": {"sm_mhz": num(sm), "mem_mhz": num(memclk), "max_sm_mhz": num(max_sm), "max_mem_mhz": num(max_mem)},
        "power_limit_w": num(power),
        "compute_mode": compute_mode,
        "pci_bus_id": bus,
        "mps": _mps_state(),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "hostname": socket.gethostname(),
    }


def build_manifest(
    *,
    repos: dict[str, str | Path],
    env_pythons: dict[str, str],
    out_dir: Path,
    gpu_index: int,
    deploy: dict,
    sampling: dict,
    datasets: list[dict],
    run_order_seed: int,
    extra: dict | None = None,
) -> dict:
    return {
        "created_unix": time.time(),
        "repos": {k: git_info(v) for k, v in repos.items()},
        "env_lock": {k: env_lock(v, out_dir, k) for k, v in env_pythons.items()},
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION},
        "datasets": datasets,
        "deploy": deploy,
        "sampling": sampling,
        "hardware": gpu_info(gpu_index),
        "run_order_seed": run_order_seed,
        "extra": extra or {},
    }


def validate_manifest(m: dict) -> list[str]:
    """Return a list of problems; empty means valid."""
    errs: list[str] = []
    for key in ("repos", "env_lock", "model", "datasets", "deploy", "sampling", "hardware", "run_order_seed"):
        if key not in m:
            errs.append(f"missing {key}")
    if errs:
        return errs
    if not m["repos"]:
        errs.append("repos is empty")
    for name, info in m["repos"].items():
        if not info.get("commit"):
            errs.append(f"repo {name} has no commit")
        if info.get("dirty") is None:
            errs.append(f"repo {name} has no dirty flag")
    for name, lock in m["env_lock"].items():
        if not lock.get("sha256") or not lock.get("packages"):
            errs.append(f"env lock {name} is empty")
            continue
        try:
            actual = hashlib.sha256(Path(lock["path"]).read_bytes()).hexdigest()
        except OSError:
            errs.append(f"env lock {name} file {lock.get('path')} is missing")
            continue
        if actual != lock["sha256"]:
            errs.append(f"env lock {name} hash does not match its file")
    if m["model"].get("revision") != MODEL_REVISION:
        errs.append(f"model revision {m['model'].get('revision')} != {MODEL_REVISION}")
    if not m["datasets"]:
        errs.append("datasets is empty")
    for d in m["datasets"]:
        if not d.get("revision"):
            errs.append(f"dataset {d.get('repo')} has no revision")
        if not d.get("ids"):
            errs.append(f"dataset {d.get('repo')} has no id lists")
    if not m["deploy"].get("resolved"):
        errs.append("deploy has no resolved config")
    cand = m["deploy"].get("candidate") or {}
    if not cand.get("candidate_id") or not cand.get("digest"):
        errs.append("deploy has no candidate identity")
    if not (m["deploy"].get("cell") or {}).get("digest"):
        errs.append("deploy has no cell specification digest")
    if not m["sampling"]:
        errs.append("sampling is empty")
    hw = m["hardware"]
    for key in REQUIRED_HARDWARE_KEYS:
        if hw.get(key) in (None, "", []):
            errs.append(f"hardware.{key} missing")
    if not isinstance(m["run_order_seed"], int):
        errs.append("run_order_seed must be an int")
    return errs
