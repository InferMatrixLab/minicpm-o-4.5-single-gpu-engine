"""Stage and kernel-activity attribution of the served engine under closed-loop load (diagnostic; tasks 8/14-17).

    .venv/bin/python tools/engine_profile.py run --gpu 2 --port 18950 --c 1 8 16 32 --out results/engine_profile/r11_base \
        [--config configs/engine/track_s_fast_hift.json] [--warm-s 20] [--measure-s 40] [--kernel-s 3] [--set key=json ...]
    .venv/bin/python tools/engine_profile.py analyze results/engine_profile/r11_base

``run`` writes the engine config plus ``stage_trace_dir`` (``engine.runtime.stage_trace``) into the output directory,
records the run identity (source commit and dirty state, config digest, engine interpreter and torch, GPU UUID, the
compute processes already on that GPU), starts ``scripts/serve_engine.sh`` under a private MPS daemon on the GPU and
the GPM sampler (``tools/gpm_sampler.py``), and for each C drives C closed-loop clients with the canonical speech
requests of the tuning set through the harness client (``bench.client.send_request``).  Per C it records the measured
window on the host clock (CLOCK_MONOTONIC, the clock of the trace), the client metrics of the requests that complete
inside it, each request's client dispatch / first audio / engine request id (``client.jsonl``), the process inventory
of the GPU during the window, and one kernel window of ``--kernel-s`` seconds ending a second before the measured
window ends (every GPU process of the engine captures its CUDA activity over it; the next C starts once every capture
is written, and the captures are compressed after the server stops).  The window throughput is diagnostic:
completion-window accounting over repeated tuning IDs, not the AC-1 protocol.

``analyze`` writes ``attribution.json``:
* ``stages``: per C over the measured window, per stage span (``<proc>.<stage>``, nested ``outer.inner``): spans per
  second, rows, host milliseconds and *stream-elapsed* milliseconds between the stage's CUDA events.  Elapsed time
  includes submission gaps, synchronization and waits behind other work: it is not GPU busy time;
* ``kernels``: per C over its kernel window (``tools/kernel_attribution.py``): kernel activity per process and
  device-wide, GPM counters, and per stage the time its kernels ran and their launch-to-start delay;
* ``critical_path``: per C, the first audio's path per request (engine queue, features, setup, prefill, first text
  chunk, Talker to the first window, transport each way, worker wait, decode, HTTP) and every window's transport legs.
It refuses an incomplete trace (``kernel_attribution.TraceError``).
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ENGINE_PY = os.environ.get("ENGINE_PYTHON", str(ROOT / ".venv-engine" / "bin" / "python"))


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _sh(cmd: list[str]) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT).stdout.strip()


def gpu_apps(uuid: str) -> list[dict]:
    """Compute processes NVML reports on the GPU (under MPS: the MPS server stands for its clients)."""
    out = []
    for line in _sh(["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"]).splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 4 and parts[0] == uuid:
            uid = None
            try:
                uid = next(int(x.split()[1]) for x in open(f"/proc/{parts[1]}/status") if x.startswith("Uid:"))
            except (OSError, StopIteration, ValueError):
                pass
            out.append({"pid": int(parts[1]), "name": parts[2], "mib": int(parts[3]) if parts[3].isdigit() else None, "uid": uid})
    return out


def process_tree(pid: int) -> list[dict]:
    """The server's process tree (pid, command)."""
    out, todo = [], [pid]
    while todo:
        p = todo.pop()
        try:
            cmd = open(f"/proc/{p}/cmdline").read().replace("\0", " ").strip()
        except OSError:
            continue
        out.append({"pid": p, "cmd": cmd[:200]})
        for t in Path(f"/proc/{p}/task").glob("*/children"):
            try:
                todo += [int(x) for x in t.read_text().split()]
            except OSError:
                pass
    return out


def inventory(uuid: str, server_pid: int) -> dict:
    """The GPU's compute processes during a window and whether any is foreign: not in the server's tree and not an MPS
    server of this user (the private MPS daemon runs detached from the tree)."""
    tree = process_tree(server_pid)
    mine = {p["pid"] for p in tree}
    apps = gpu_apps(uuid)
    foreign = [a for a in apps if a["pid"] not in mine and not (a["name"].endswith("nvidia-cuda-mps-server") and a["uid"] == os.getuid())]
    return {"gpu_apps": apps, "server_tree": tree, "foreign": foreign}


def identity(out: Path, cfg_text: str, gpu: int) -> dict:
    uuid = _sh(["nvidia-smi", "-i", str(gpu), "--query-gpu=uuid", "--format=csv,noheader"])
    return {"commit": _sh(["git", "rev-parse", "HEAD"]), "dirty": _sh(["git", "status", "--porcelain", "-uno"]).splitlines(),
            "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(), "engine_python": ENGINE_PY,
            "engine_python_version": _sh([ENGINE_PY, "-c", "import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda)"]),
            "gpu": gpu, "gpu_uuid": uuid, "gpu_name": _sh(["nvidia-smi", "-i", str(gpu), "--query-gpu=name", "--format=csv,noheader"]),
            "gpu_apps_before": gpu_apps(uuid), "host": os.uname().nodename}


def wait_captures(tr: Path, tag: str, timeout_s: float = 600) -> float:
    """Seconds until every registered process has written its capture of window ``tag`` (raises after the timeout)."""
    t = time.time()
    while True:
        procs = [json.loads(p.read_text()) for p in (tr / "procs").glob("*.json")]
        if all((tr / "kernels" / f"{p['proc']}-{p['pid']}-{tag}.markers.json").exists() for p in procs):
            return time.time() - t
        if time.time() - t > timeout_s:
            raise SystemExit(f"captures of window {tag} not written after {timeout_s} s")
        time.sleep(1)


def expected_processes(cfg: dict) -> dict:
    exp = {"engine": 1}
    if cfg.get("t2w_process"):
        exp["t2w"] = 1
        if cfg.get("feature_processes", 0) and cfg.get("feature_gpu_tokens", True):
            exp["feature"] = int(cfg["feature_processes"])
    return exp


def analyze(d: Path) -> dict:
    from tools.kernel_attribution import analyze_kernels, critical_path, load_stage_files

    run = json.loads((d / "run.json").read_text())
    tr = d / "trace"
    files = load_stage_files(tr)
    tags = {w["kernel"]["tag"]: (w["kernel"]["t0"], w["kernel"]["t1"]) for w in run["windows"].values() if "kernel" in w}
    kern = analyze_kernels(tr, tags, run["expected_processes"]) if tags else {}  # a spans-only run: no kernel windows
    reqs = _rows(tr / "requests.jsonl")
    client = {r["engine_id"]: (r["t_dispatch"], r["t_first_audio"]) for r in _rows(d / "client.jsonl") if r.get("engine_id")}
    out = {"identity": run["identity"], "config": run["config"], "windows": {}}
    for c, win in run["windows"].items():
        w0, w1 = win["t0"], win["t1"]
        span = w1 - w0
        stages = {}
        for stem, f in files.items():
            proc = stem.split("-")[0] if stem.startswith("feature-") else stem
            for r in f["rows"]:
                if r["c1"] > w0 and r["c0"] < w1:
                    st = stages.setdefault(f"{proc}.{r['s']}", {"spans": 0, "rows": 0, "frames": 0, "host": 0.0, "elapsed": 0.0})
                    st["spans"] += 1
                    st["rows"] += r["n"]
                    st["frames"] += r["f"]
                    st["host"] += r["c1"] - r["c0"]
                    st["elapsed"] += r["g1"] - r["g0"]
        ktag = win.get("kernel", {}).get("tag")
        k0, k1 = tags.get(ktag, (w1, w1))
        probe = {}  # the capture's perturbation: rates and elapsed times inside the kernel window vs the rest of the window
        for name in ("engine.thinker", "engine.talker", "t2w.decode", "t2w.setup"):
            stem, s = name.split(".")
            rs = [r for r in files.get(stem, {"rows": []})["rows"] if r["s"] == s and w0 <= r["c0"] < w1]
            ins = [r for r in rs if k0 <= r["c0"] < k1]
            outs = [r for r in rs if not k0 <= r["c0"] < k1]
            probe[name] = {"inside_per_s": len(ins) / (k1 - k0) if k1 > k0 else None, "outside_per_s": len(outs) / (span - (k1 - k0)),
                           "inside_elapsed_ms": 1000 * sum(r["g1"] - r["g0"] for r in ins) / len(ins) if ins else None,
                           "outside_elapsed_ms": 1000 * sum(r["g1"] - r["g0"] for r in outs) / len(outs) if outs else None}
        out["windows"][c] = {
            "window_s": span, "client": win.get("client"), "inventory_foreign": win["inventory"]["foreign"], "capture_perturbation": probe,
            "stages": {k: {"spans_per_s": v["spans"] / span, "rows_mean": v["rows"] / v["spans"], "frames_per_s": v["frames"] / span,
                           "host_ms": 1000 * v["host"] / v["spans"], "elapsed_ms": 1000 * v["elapsed"] / v["spans"]} for k, v in sorted(stages.items())},
            "kernels": kern.get(ktag), "critical_path": critical_path(reqs, w0, w1, client)}
    (d / "attribution.json").write_text(json.dumps(out, indent=1))
    return out


def _print(out: dict) -> None:
    f = lambda x, p=2: "-" if x is None else f"{x:.{p}f}"  # noqa: E731
    for c, r in out["windows"].items():
        cl = r.get("client") or {}
        k = r["kernels"]
        if k is None:  # spans only: stage host/elapsed times and the critical path
            print(f"C={c}  client fps {f(cl.get('frames_per_s'), 0)} ttfa_p50 {f(cl.get('ttfa_p50_s'), 3)} rtf {f(cl.get('rtf_mean'), 3)}  (spans only)")
            for s, v in r["stages"].items():
                print(f"  {s:28s} {v['spans_per_s']:7.1f}/s rows {f(v['rows_mean'], 1):>5s} host {f(v['host_ms']):>7s} elapsed {f(v['elapsed_ms']):>7s} ms")
            cp = r["critical_path"]["first_audio"]
            print("  first audio (p50 / p95 ms): " + ", ".join(f"{s} {f(v['p50_ms'], 1)}/{f(v['p95_ms'], 1)}" for s, v in cp.items()))
            continue
        g = k.get("gpm", {})
        print(f"C={c}  client fps {f(cl.get('frames_per_s'), 0)} ttfa_p50 {f(cl.get('ttfa_p50_s'), 3)} rtf {f(cl.get('rtf_mean'), 3)}  "
              f"kernel window {k['window_s']:.1f}s: device kernel-active {k['device']['active_frac']:.3f} busy-sum {k['device']['busy_sum_frac']:.3f}  "
              f"GPM sm {f(g.get('sm_util'), 1)}% occ {f(g.get('sm_occupancy'), 1)}% dram {f(g.get('dram_bw_util'), 1)}% tensor {f(g.get('tensor_util'), 1)}%"
              f"  foreign {len(r['inventory_foreign'])}")
        for p, v in k["processes"].items():
            print(f"  {p:16s} kernel-active {v['active_frac']:.3f}  kernels/s {v['kernels'] / k['window_s']:.0f}")
        for s, v in k["stages"].items():
            print(f"  {s:28s} {v['spans_per_s']:7.1f}/s rows {f(v['rows_mean'], 1):>5s} host {f(v['host_ms']):>7s} elapsed {f(v['elapsed_ms']):>7s} "
                  f"kernel self {f(v['kernel_self_ms']):>7s} total {f(v['kernel_total_ms']):>7s} ms  frac self {v['kernel_self_frac']:.3f}  blocking {f(v.get('blocking_ms'))} ms  "
                  f"launch->start p50 {f(v['launch_to_start_ms_p50'])} ms")
        print("  capture perturbation (inside / outside the kernel window): " + ", ".join(
            f"{s} {f(v['inside_per_s'], 1)}/{f(v['outside_per_s'], 1)} per s, {f(v['inside_elapsed_ms'])}/{f(v['outside_elapsed_ms'])} ms"
            for s, v in r["capture_perturbation"].items()))
        cp = r["critical_path"]["first_audio"]
        print("  first audio (p50 / p95 ms): " + ", ".join(f"{s} {f(v['p50_ms'], 1)}/{f(v['p95_ms'], 1)}" for s, v in cp.items()))
        ew = r["critical_path"]["every_window"]
        print("  every window (p50 / p95 ms): " + ", ".join(f"{s} {f(v['p50_ms'], 1)}/{f(v['p95_ms'], 1)}" for s, v in ew.items()))


async def _load(url: str, c: int, warm_s: float, measure_s: float, on_window, client_rows: list) -> dict:
    import numpy as np

    from bench.adapters import get_adapter
    from bench.client import make_session, send_request
    from bench.metrics import request_metrics
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets

    rows = load_seed_tts_rows()
    ids = make_id_sets(rows).tuning(32)
    reqs = canonical_requests(rows, ids)
    adapter = get_adapter("engine")
    payloads = [adapter.build_payload(r) for r in reqs]
    engine_ids: dict = {}
    handle = adapter.handle_sse_data

    def capture_id(t, data, trace, asm):  # the engine's request id (the chunk "id") joins its timeline to the client
        if trace.request_id not in engine_ids:
            try:
                engine_ids[trace.request_id] = json.loads(data).get("id")
            except (ValueError, AttributeError):
                pass
        return handle(t, data, trace, asm)

    adapter.handle_sse_data = capture_id
    t_start = time.perf_counter()
    w0, w1 = t_start + warm_s, t_start + warm_s + measure_s
    on_window(w0, w1)
    done: list = []
    nxt = [0]

    async def client(session):
        while time.perf_counter() < w1:
            i = nxt[0] % len(reqs)
            nxt[0] += 1
            tr = await send_request(session, url, adapter, SimpleNamespace(request_id=f"p{c}-{nxt[0]}"), timeout_s=300, payload=payloads[i])
            done.append(tr)

    async with make_session() as session:
        await asyncio.gather(*(client(session) for _ in range(c)))
    ms = [request_metrics(t) for t in done]
    for t, m in zip(done, ms):
        client_rows.append({"c": c, "client_id": t.request_id, "engine_id": engine_ids.get(t.request_id), "ok": m.ok, "t_dispatch": m.t_dispatch,
                            "t_first_audio": None if m.ttfa_s is None else m.t_dispatch + m.ttfa_s, "t_complete": m.t_complete})
    inside = [m for m in ms if m.ok and m.t_dispatch >= w0 and m.t_complete is not None and m.t_complete <= w1]
    audio = sum(m.duration_s for m in ms if m.ok and m.t_complete is not None and w0 <= m.t_complete <= w1)
    client = {"requests": len(ms), "invalid": sum(not m.ok for m in ms), "inside": len(inside),
              "frames_per_s": 25 * audio / measure_s,  # audio of the requests completing in the window, over the window
              "ttfa_p50_s": float(np.percentile([m.ttfa_s for m in inside], 50)) if inside else None,
              "rtf_mean": float(np.mean([m.rtf for m in inside])) if inside else None}
    return {"t0": w0, "t1": w1, "client": client}


def _wait_health(url: str, proc, timeout_s: float = 1800) -> None:
    t = time.time()
    while time.time() - t < timeout_s:
        if proc.poll() is not None:
            raise SystemExit(f"server exited with {proc.returncode}")
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as r:
                if r.status == 200:
                    return
        except OSError:
            pass
        time.sleep(5)
    raise SystemExit("server did not become healthy")


def cmd_run(a) -> int:
    out = Path(a.out).absolute()
    if out.exists():
        raise SystemExit(f"{out} exists")
    tr = out / "trace"
    tr.mkdir(parents=True)
    cfg = json.loads(Path(a.config).read_text())
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    if not a.no_trace:
        cfg["stage_trace_dir"] = str(tr)
    cfg_text = json.dumps(cfg, indent=1)
    (out / "config.json").write_text(cfg_text)
    ident = identity(out, cfg_text, a.gpu)
    if ident["gpu_apps_before"]:
        raise SystemExit(f"GPU {a.gpu} is not idle: {ident['gpu_apps_before']}")
    url = f"http://127.0.0.1:{a.port}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu), "ENGINE_MPS": "1"}
    if a.spans_only and not a.window_thread:
        env["STAGE_TRACE_KERNEL_WINDOWS"] = "0"
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(["bash", str(ROOT / "scripts" / "serve_engine.sh"), str(out / "config.json"), str(a.port)], env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    (out / "server.pid").write_text(str(proc.pid))
    gpm = subprocess.Popen([ENGINE_PY, str(ROOT / "tools" / "gpm_sampler.py"), "--gpu", str(a.gpu), "--out", str(tr / "gpm.jsonl")],
                           stdout=subprocess.DEVNULL, stderr=open(out / "gpm.log", "w"))
    run = {"identity": ident, "config": {"source": a.config, "set": a.set or [], "gpu": a.gpu}, "expected_processes": expected_processes(cfg),
           "windows": {}}
    kwin: list = []
    client_rows: list = []
    try:
        _wait_health(url, proc)
        for c in a.c:
            def on_window(w0, w1, c=c):
                if a.no_trace or a.spans_only:
                    return
                tag = str(c) if not any(w["tag"].split("#")[0] == str(c) for w in kwin) else f"{c}#{sum(w['tag'].split('#')[0] == str(c) for w in kwin) + 1}"
                kwin.append({"tag": tag, "t0": w1 - 1.0 - a.kernel_s, "t1": w1 - 1.0})
                tmp = tr / "kernel_windows.tmp"
                tmp.write_text(json.dumps(kwin))
                tmp.replace(tr / "kernel_windows.json")

            res = asyncio.run(_load(url, c, a.warm_s, a.measure_s, on_window, client_rows))
            res["inventory"] = inventory(ident["gpu_uuid"], proc.pid)
            if not (a.no_trace or a.spans_only):
                res["kernel"] = kwin[-1]
                res["captures_wait_s"] = wait_captures(tr, kwin[-1]["tag"])  # before the next C: an export never overlaps a window
            key = str(c) if str(c) not in run["windows"] else f"{c}#{sum(k.split('#')[0] == str(c) for k in run['windows']) + 1}"
            run["windows"][key] = res  # a repeated C keeps every window
            print(json.dumps({"c": c, **res["client"], "foreign": res["inventory"]["foreign"], "captures_wait_s": res.get("captures_wait_s")}), flush=True)
        time.sleep(3)  # trace rows and heartbeats of the last window are flushed within a second
    finally:
        (out / "run.json").write_text(json.dumps(run, indent=1))
        (out / "client.jsonl").write_text("".join(json.dumps(r) + "\n" for r in client_rows))
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)  # serve_engine.sh stops the server and its MPS daemon
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
        gpm.send_signal(signal.SIGTERM)
        gpm.wait(timeout=30)
    if a.no_trace:  # a control run: client metrics only
        return 0
    for f in sorted((tr / "kernels").glob("*.json")):  # the captures, compressed once the server is gone
        if not f.name.endswith(".markers.json"):
            with open(f, "rb") as fin, gzip.open(str(f) + ".gz", "wb", compresslevel=3) as fout:
                shutil.copyfileobj(fin, fout)
            f.unlink()
    _print(analyze(out))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--gpu", type=int, required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--c", type=int, nargs="+", default=[1, 8, 16, 32])
    r.add_argument("--out", required=True)
    r.add_argument("--config", default=str(ROOT / "configs" / "engine" / "track_s_fast_hift.json"))
    r.add_argument("--warm-s", type=float, default=20.0)
    r.add_argument("--measure-s", type=float, default=40.0)
    r.add_argument("--kernel-s", type=float, default=1.5)
    r.add_argument("--set", action="append", help="KEY=JSON override of the engine config")
    r.add_argument("--no-trace", action="store_true", help="control: no stage trace or kernel windows (client metrics only)")
    r.add_argument("--spans-only", action="store_true", help="control: stage spans and timelines, no kernel windows (client metrics only)")
    r.add_argument("--window-thread", action="store_true", help="with --spans-only: keep the kernel-window threads running (no window is ever set)")
    an = sub.add_parser("analyze")
    an.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    _print(analyze(Path(a.dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
