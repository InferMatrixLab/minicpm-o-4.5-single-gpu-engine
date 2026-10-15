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


RELEVANT_ENV_PREFIXES = ("CUDA_", "HF_", "VLLM", "SGLANG", "OMNI_", "PYTORCH_", "NCCL_", "TORCH", "MINICPMO", "ENGINE_")
#: Inherited variables that could change a server's semantics are never passed through; a launch sets them explicitly.
SCRUBBED_ENV_PREFIXES = ("MINICPMO", "VLLM", "SGLANG", "OMNI_", "TORCHINDUCTOR", "PYTORCH_", "ENGINE_", "NCCL_")


def gpu_processes(gpu: int) -> list[dict] | None:
    """Processes holding a context on ``gpu`` with their context type ("C", or "M+C" for MPS clients).

    Returns None when the query fails (missing observation, never "no processes").
    """
    import xml.etree.ElementTree as ET

    try:
        out = subprocess.run(["nvidia-smi", "-q", "-x", "-i", str(gpu)], capture_output=True, text=True, timeout=30, check=True).stdout
        root = ET.fromstring(out)
    except Exception:
        return None
    procs = []
    for p in root.iter("process_info"):
        pid = (p.findtext("pid") or "").strip()
        procs.append({"pid": int(pid) if pid.isdigit() else pid, "type": (p.findtext("type") or "").strip(),
                      "process_name": (p.findtext("process_name") or "").strip(), "used_memory": (p.findtext("used_memory") or "").strip()})
    return procs


def _ppid_map() -> dict[int, int]:
    out = {}
    for d in Path("/proc").iterdir():
        if d.name.isdigit():
            try:
                fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
                out[int(d.name)] = int(fields[1])
            except (OSError, IndexError, ValueError):
                pass
    return out


def descendants(pid: int, ppid: dict[int, int] | None = None) -> set[int]:
    ppid = ppid if ppid is not None else _ppid_map()
    children: dict[int, list[int]] = {}
    for c, p in ppid.items():
        children.setdefault(p, []).append(c)
    out, stack = {pid}, [pid]
    while stack:
        for c in children.get(stack.pop(), []):
            if c not in out:
                out.add(c)
                stack.append(c)
    return out


def process_info(pid: int) -> dict:
    """Resolved command line and relevant environment of a running process."""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().split(b"\0")
        env_raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError:
        return {"pid": pid, "error": "not readable"}
    env = {}
    for kv in env_raw:
        k, _, v = kv.decode(errors="replace").partition("=")
        if k.startswith(RELEVANT_ENV_PREFIXES):
            env[k] = v
    return {"pid": pid, "cmdline": [c.decode(errors="replace") for c in cmd if c], "env": dict(sorted(env.items()))}


def classify_mps(server_pids: set[int], gpu_procs: list[dict] | None) -> dict:
    """Attribute GPU contexts to this server's processes and derive the observed MPS state.

    observed: "on" (every worker context is an MPS client), "off" (every worker
    context is direct), "mixed", or "unknown" (no observation or no worker on the GPU).
    MPS servers and other processes on the GPU that are not this server's
    workers never decide the state.
    """
    if gpu_procs is None:
        return {"observed": "unknown", "reason": "gpu process query failed", "workers": [], "foreign": []}
    workers = [p for p in gpu_procs if p["pid"] in server_pids]
    foreign = [p for p in gpu_procs if p["pid"] not in server_pids]
    if not workers:
        return {"observed": "unknown", "reason": "no context of this server on the GPU", "workers": [], "foreign": foreign}
    mps = [w for w in workers if "M" in w["type"]]
    observed = "on" if len(mps) == len(workers) else ("off" if not mps else "mixed")
    return {"observed": observed, "workers": workers, "foreign": foreign,
            "mps_servers_on_gpu": [p for p in gpu_procs if "nvidia-cuda-mps-server" in p["process_name"]]}


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
        env = {k: v for k, v in os.environ.items() if not k.startswith(SCRUBBED_ENV_PREFIXES)}
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
                time.sleep(1)
            raise RuntimeError(f"{self.spec.name} not healthy after {self.spec.startup_timeout_s}s; see {self.log_path}")
        except BaseException:
            self.stop()  # never leave stage children behind
            raise

    def evidence(self) -> dict:
        """Launch inputs and the resolved runtime state of this server, captured after startup."""
        from bench.fingerprint import gpu_identity

        pids = descendants(self.proc.pid) if self.proc else set()
        gpu_procs = gpu_processes(self.spec.gpu)
        on_gpu = {p["pid"] for p in gpu_procs or []}
        return {
            "server_pid": self.proc.pid if self.proc else None,
            "server_pgid": self.pgid,
            "launch_argv": list(self.spec.cmd),
            "launch_env": dict(sorted(self._env_overlay.items())),
            "gpu": gpu_identity(self.spec.gpu),
            "startup_s": self.startup_s,
            "process_tree": sorted(pids),
            "gpu_workers": [process_info(pid) for pid in sorted(pids & on_gpu)],
            "mps": classify_mps(pids, gpu_procs),
        }

    def stop(self, timeout_s: float = 120.0, gpu_free_mib: float = 1024.0) -> None:
        if self.proc is None:
            return
        pgid = self.pgid
        if pgid is not None:
            # The leader first: servers that own resources outside the group (sglang-omni's MPS daemon) retire their
            # workers in order; signalling every worker at once races that teardown. Then the group, then SIGKILL.
            for target, sig, wait in (("leader", signal.SIGTERM, timeout_s), ("group", signal.SIGTERM, 30.0), ("group", signal.SIGKILL, 30.0)):
                try:
                    if target == "leader":
                        self.proc.send_signal(sig)
                    else:
                        os.killpg(pgid, sig)
                except ProcessLookupError:
                    break
                end = time.monotonic() + wait
                while time.monotonic() < end:
                    if self.proc.poll() is not None and target == "leader":  # reaped; orphaned workers get a short grace
                        end = min(end, time.monotonic() + 15.0)
                    if not self._group_alive(pgid):
                        break
                    time.sleep(0.5)
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
        """True while any non-zombie process of the group exists."""
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        for d in Path("/proc").iterdir():
            if not d.name.isdigit():
                continue
            try:
                fields = (d / "stat").read_text().rsplit(")", 1)[1].split()
            except (OSError, IndexError):
                continue
            if fields[0] != "Z" and int(fields[2]) == pgid:
                return True
        return False

    def __enter__(self) -> "Server":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()


def vllm_omni_spec(config: str, port: int, gpu: int, extra: list[str] | None = None) -> ServerSpec:
    return ServerSpec(name=f"vllm-omni:{config}", cmd=["bash", str(ROOT / "scripts" / "serve_vllm_omni.sh"), config, str(port), *(extra or [])], port=port, gpu=gpu)


def sglang_omni_spec(port: int, gpu: int, extra: list[str] | None = None) -> ServerSpec:
    return ServerSpec(name="sglang-omni:speech", cmd=["bash", str(ROOT / "scripts" / "serve_sglang_omni.sh"), str(port), *(extra or [])], port=port, gpu=gpu)
