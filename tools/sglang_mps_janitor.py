"""Archive abandoned sglang-omni MPS state dirs (sglang's own documented cleanup), only when provably abandoned.

Abandoned: every owner marker PID is dead, the recorded control/server PIDs (if any) are dead, and no live process has
CUDA_MPS_PIPE_DIRECTORY pointing at the dir.  Its logs and owner markers are copied under results/l20x/mps_probe/abandoned/, then the dir is removed.
"""
import getpass, json, os, shutil, tempfile, time
from pathlib import Path

ROOT = Path(tempfile.gettempdir()) / f"sglang-omni-mps-{getpass.getuser()}"  # sglang_omni.mps.runtime's state dir
DEST = Path(__file__).resolve().parents[1] / "results" / "l20x" / "mps_probe" / "abandoned"


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def pipe_users(pipe: str) -> list[int]:
    out = []
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            env = (p / "environ").read_bytes().split(b"\0")
        except OSError:
            continue
        if f"CUDA_MPS_PIPE_DIRECTORY={pipe}".encode() in env:
            out.append(int(p.name))
    return out


def abandoned(d: Path) -> tuple[bool, dict]:
    owners = [int(x.name) for x in (d / "owners").glob("*") if x.name.isdigit()]
    pidf = d / "pipe" / "nvidia-cuda-mps-control.pid"
    daemon = int(pidf.read_text().strip()) if pidf.exists() and pidf.read_text().strip().isdigit() else None
    users = pipe_users(str(d / "pipe"))
    info = {"owners": owners, "owners_alive": [o for o in owners if alive(o)], "daemon": daemon, "daemon_alive": bool(daemon and alive(daemon)),
            "pipe_users": users}
    return (not info["owners_alive"] and not info["daemon_alive"] and not users), info


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    while True:
        for d in sorted(ROOT.glob("GPU-*")) if ROOT.exists() else []:
            ok, info = abandoned(d)
            if ok:
                time.sleep(1.0)
                ok, info = abandoned(d)  # re-check: a new server may have just claimed it
            if ok:
                dst = DEST / f"{d.name}.{time.strftime('%Y%m%d-%H%M%S')}"
                dst.mkdir(parents=True, exist_ok=True)
                for sub in ("log", "owners"):  # regular files only: the pipe dir holds FIFOs and sockets
                    if (d / sub).is_dir():
                        shutil.copytree(d / sub, dst / sub, dirs_exist_ok=True)
                (dst / "pipe_listing.txt").write_text("\n".join(sorted(x.name for x in (d / "pipe").iterdir())) if (d / "pipe").is_dir() else "")
                (dst / "archived.json").write_text(json.dumps({"unix": time.time(), **info}, indent=1))
                shutil.rmtree(d)
                print(f"[{time.strftime('%F %T')}] archived abandoned {d.name} -> {dst} {info}", flush=True)
        time.sleep(2.0)


if __name__ == "__main__":
    main()
