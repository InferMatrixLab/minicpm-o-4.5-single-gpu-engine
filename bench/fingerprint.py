"""Content fingerprints of everything an experiment's result depends on."""

from __future__ import annotations

import functools
import hashlib
import subprocess
from pathlib import Path


def _run(cmd: list[str], cwd: str | None = None) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=120, check=True).stdout
    except (subprocess.SubprocessError, FileNotFoundError):
        return ""


def file_sha256(path: str | Path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha256(root: str | Path, subdirs: tuple[str, ...], suffixes: tuple[str, ...] = (".py", ".sh", ".yaml", ".json")) -> str:
    """Content hash of every source file under ``root/subdir`` (path + bytes), ignoring caches."""
    h = hashlib.sha256()
    root = Path(root)
    for sub in subdirs:
        for p in sorted((root / sub).rglob("*")):
            if p.is_file() and p.suffix in suffixes and "__pycache__" not in p.parts:
                h.update(str(p.relative_to(root)).encode())
                h.update(b"\0")
                h.update(p.read_bytes())
                h.update(b"\0")
    return h.hexdigest()


def repo_fingerprint(path: str | Path) -> dict:
    """Commit plus a hash of the tracked working-tree diff (dirty content, not just a flag)."""
    path = str(path)
    commit = _run(["git", "rev-parse", "HEAD"], cwd=path).strip()
    diff = _run(["git", "diff", "HEAD", "--binary"], cwd=path)
    return {"commit": commit or None, "dirty": bool(diff.strip()), "diff_sha256": hashlib.sha256(diff.encode()).hexdigest()}


@functools.lru_cache(maxsize=16)
def env_fingerprint(python: str) -> dict:
    frozen = _run(["uv", "pip", "freeze", "--python", python])
    return {"python": python, "freeze_sha256": hashlib.sha256(frozen.encode()).hexdigest(), "packages": len(frozen.splitlines())}


def gpu_identity(index: int) -> dict:
    out = _run(["nvidia-smi", "-i", str(index), "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv,noheader,nounits"]).strip()
    parts = [p.strip() for p in out.split(",")] if out else []
    if len(parts) < 4:
        return {"gpu_index": index, "error": "nvidia-smi query failed"}
    return {"gpu_index": index, "gpu_sku": parts[0].replace("NVIDIA ", ""), "gpu_uuid": parts[1], "driver": parts[2], "memory_mib": parts[3]}
