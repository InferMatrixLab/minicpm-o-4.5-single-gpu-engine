"""Start and stop serving processes for measurement cells.

Servers run directly on a GPU the caller already holds (a manual canhazgpu
reservation), so a restart between runs cannot lose the GPU to the queue.
Each server is its own process group and is stopped as a group.
"""

from __future__ import annotations

import os
import signal
import subprocess
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass
class ServerSpec:
    name: str
    cmd: list[str]
    port: int
    gpu: int
    health_path: str = "/health"
    startup_timeout_s: float = 2400.0
    env: dict = field(default_factory=dict)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def _healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            return r.status == 200
    except Exception:
        return False


def gpu_memory_used_mib(gpu: int) -> float | None:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=30,
            check=True,
        ).stdout.strip()
        return float(out)
    except Exception:
        return None


RELEVANT_ENV_PREFIXES = ("CUDA_", "HF_", "VLLM", "SGLANG", "OMNI_", "PYTORCH_", "NCCL_", "TORCH")


def gpu_compute_apps(gpu: int) -> list[dict]:
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30, check=True,
        ).stdout
    except Exception:
        return []
    apps = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 3:
            apps.append({"pid": int(parts[0]) if parts[0].isdigit() else parts[0], "process_name": parts[1], "used_mib": parts[2]})
    return apps


def mps_evidence(gpu: int, pgid: int | None) -> dict:
    """Whether the GPU is served through an MPS server, from the compute apps seen after startup."""
    apps = gpu_compute_apps(gpu)
    mps_server = any("nvidia-cuda-mps-server" in a["process_name"] for a in apps)
    group_pids = set()
    if pgid is not None:
        try:
            out = subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True, text=True, timeout=10).stdout
            group_pids = {int(x) for x in out.split()}
        except Exception:
            pass
    direct = [a for a in apps if a["pid"] in group_pids]
    return {"compute_apps": apps, "mps_server_on_gpu": mps_server, "server_group_direct_contexts": len(direct), "mps_active": mps_server}


class Server:
    def __init__(self, spec: ServerSpec, log_path: Path) -> None:
        self.spec = spec
        self.log_path = log_path
        self.proc: subprocess.Popen | None = None
        self.pgid: int | None = None
        self.startup_s: float | None = None
        self._env_overlay: dict = {}

    @property
    def url(self) -> str:
        return self.spec.url

    def start(self) -> "Server":
        if _healthy(self.spec.url + self.spec.health_path):
            raise RuntimeError(f"port {self.spec.port} already serves a healthy server")
        env = os.environ.copy()
        env.update(self.spec.env)
        env["CUDA_VISIBLE_DEVICES"] = str(self.spec.gpu)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = self.log_path.open("ab")
        t0 = time.monotonic()
        self._env_overlay = {k: v for k, v in env.items() if k.startswith(RELEVANT_ENV_PREFIXES)}
        self.proc = subprocess.Popen(self.spec.cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, cwd=str(ROOT))
        self.pgid = self.proc.pid  # start_new_session: the leader's pid is the group id
        deadline = t0 + self.spec.startup_timeout_s
        try:
            while time.monotonic() < deadline:
                if self.proc.poll() is not None:
                    raise RuntimeError(f"{self.spec.name} exited with {self.proc.returncode} during startup; see {self.log_path}")
                if _healthy(self.spec.url + self.spec.health_path):
                    self.startup_s = time.monotonic() - t0
                    return self
                time.sleep(5)
            raise RuntimeError(f"{self.spec.name} not healthy after {self.spec.startup_timeout_s}s; see {self.log_path}")
        except BaseException:
            self.stop()  # never leave stage children behind
            raise

    def evidence(self) -> dict:
        """Launch identity and runtime state of this server, captured after startup."""
        return {
            "server_pid": self.proc.pid if self.proc else None,
            "server_pgid": self.pgid,
            "argv": list(self.spec.cmd),
            "env": dict(sorted(self._env_overlay.items())),
            "gpu": self.spec.gpu,
            "startup_s": self.startup_s,
            "mps": mps_evidence(self.spec.gpu, self.pgid),
        }

    def stop(self, timeout_s: float = 120.0, gpu_free_mib: float = 1024.0) -> None:
        if self.proc is None:
            return
        pgid = self.pgid
        if pgid is not None:
            for sig, wait in ((signal.SIGTERM, timeout_s), (signal.SIGKILL, 30.0)):
                try:
                    os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                end = time.monotonic() + wait
                while time.monotonic() < end and self._group_alive(pgid):
                    time.sleep(1)
                if not self._group_alive(pgid):
                    break
        try:
            self.proc.wait(timeout=10)  # reap the leader
        except Exception:
            pass
        self.proc = None
        end = time.monotonic() + 300
        while time.monotonic() < end:
            used = gpu_memory_used_mib(self.spec.gpu)
            if used is None or used < gpu_free_mib:
                return
            time.sleep(3)
        raise RuntimeError(f"GPU {self.spec.gpu} still holds memory after stopping {self.spec.name}")

    @staticmethod
    def _group_alive(pgid: int) -> bool:
        try:
            os.killpg(pgid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def vllm_omni_spec(config: str, port: int, gpu: int, extra: list[str] | None = None) -> ServerSpec:
    return ServerSpec(name=f"vllm-omni:{config}", cmd=["bash", str(ROOT / "scripts" / "serve_vllm_omni.sh"), config, str(port), *(extra or [])], port=port, gpu=gpu)


def sglang_omni_spec(port: int, gpu: int, extra: list[str] | None = None) -> ServerSpec:
    return ServerSpec(name="sglang-omni:speech", cmd=["bash", str(ROOT / "scripts" / "serve_sglang_omni.sh"), str(port), *(extra or [])], port=port, gpu=gpu)
