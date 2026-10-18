"""Stage attribution of the served engine under closed-loop load (diagnostic; task8/task17 performance attribution).

    .venv/bin/python tools/engine_profile.py run --gpu 2 --port 18950 --c 1 8 16 32 --out results/engine_profile/r10_base \
        [--config configs/engine/track_s_fast.json] [--warm-s 20] [--measure-s 40] [--set key=json ...]
    .venv/bin/python tools/engine_profile.py analyze results/engine_profile/r10_base

``run`` writes the given engine config plus ``stage_trace_dir`` (``engine.runtime.stage_trace``; no synchronization in
the engine) into the output directory, starts ``scripts/serve_engine.sh`` under a private MPS daemon on the GPU, and
for each C drives C closed-loop clients with the canonical speech requests of the tuning set through the harness
client (``bench.client.send_request``).  It records the measured window of each C on the host clock (CLOCK_MONOTONIC,
the clock of the trace) and the client metrics of the requests that complete inside it, then stops the server.

``analyze`` reads ``trace/engine.jsonl`` and ``trace/t2w.jsonl`` and, per C over the measured window, reports for each
stage the calls per second, mean rows, host and GPU milliseconds per call, the fraction of the window it occupies
on the host and on the GPU stream (the union of its intervals), frames produced per second (Talker frames, Token2Wav
audio frames); per process the union of all stage GPU intervals, the engine thread's union of host intervals, and the
union over both processes.  Writes ``attribution.json``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _union(iv: list[tuple[float, float]], w0: float, w1: float) -> float:
    tot, cur0, cur1 = 0.0, None, None
    for a, b in sorted((max(a, w0), min(b, w1)) for a, b in iv if b > w0 and a < w1):
        if cur1 is None or a > cur1:
            if cur1 is not None:
                tot += cur1 - cur0
            cur0, cur1 = a, b
        else:
            cur1 = max(cur1, b)
    if cur1 is not None:
        tot += cur1 - cur0
    return tot


def _rows(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def analyze(d: Path) -> dict:
    run = json.loads((d / "run.json").read_text())
    procs = {name: _rows(d / "trace" / f"{name}.jsonl") for name in ("engine", "t2w")}
    out = {"config": run["config"], "windows": {}}
    for c, win in run["windows"].items():
        w0, w1 = win["t0"], win["t1"]
        span = w1 - w0
        res = {"window_s": span, "client": win.get("client"), "stages": {}, "processes": {}}
        allg = []
        for name, rows in procs.items():
            inw = [r for r in rows if r["g1"] > w0 and r["g0"] < w1]
            g = [(r["g0"], r["g1"]) for r in inw]
            h = [(r["c0"], r["c1"]) for r in inw]
            allg += g
            res["processes"][name] = {"gpu_busy": _union(g, w0, w1) / span, "host_busy": _union(h, w0, w1) / span}
            for s in sorted({r["s"] for r in inw}):
                rs = [r for r in inw if r["s"] == s]
                res["stages"][f"{name}.{s}"] = {
                    "calls_per_s": len(rs) / span, "rows_mean": sum(r["n"] for r in rs) / len(rs),
                    "host_ms": 1000 * sum(r["c1"] - r["c0"] for r in rs) / len(rs), "gpu_ms": 1000 * sum(r["g1"] - r["g0"] for r in rs) / len(rs),
                    "host_frac": _union([(r["c0"], r["c1"]) for r in rs], w0, w1) / span,
                    "gpu_frac": _union([(r["g0"], r["g1"]) for r in rs], w0, w1) / span,
                    "frames_per_s": sum(r["f"] for r in rs) / span}
        res["gpu_busy_any"] = _union(allg, w0, w1) / span
        out["windows"][c] = res
    (d / "attribution.json").write_text(json.dumps(out, indent=1))
    return out


def _print(out: dict) -> None:
    for c, r in out["windows"].items():
        cl = r.get("client") or {}
        print(f"C={c}  window {r['window_s']:.0f}s  client fps {cl.get('frames_per_s', float('nan')):.0f}  ttfa_p50 {cl.get('ttfa_p50_s', float('nan')):.3f}"
              f"  rtf {cl.get('rtf_mean', float('nan')):.3f}  gpu busy any {r['gpu_busy_any']:.2f}")
        for p, v in r["processes"].items():
            print(f"  {p:7s} gpu {v['gpu_busy']:.2f} host {v['host_busy']:.2f}")
        for s, v in r["stages"].items():
            print(f"  {s:16s} {v['calls_per_s']:7.1f}/s rows {v['rows_mean']:5.1f} host {v['host_ms']:6.2f} ms gpu {v['gpu_ms']:6.2f} ms"
                  f"  host {v['host_frac']:.2f} gpu {v['gpu_frac']:.2f}  frames/s {v['frames_per_s']:.0f}")


async def _load(url: str, c: int, warm_s: float, measure_s: float) -> dict:
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
    t_start = time.perf_counter()
    w0, w1 = t_start + warm_s, t_start + warm_s + measure_s
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
    (out / "trace").mkdir(parents=True)
    cfg = json.loads(Path(a.config).read_text())
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        cfg[k] = json.loads(v)
    cfg["stage_trace_dir"] = str(out / "trace")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    url = f"http://127.0.0.1:{a.port}"
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(a.gpu), "ENGINE_MPS": "1"}
    log = open(out / "server.log", "w")
    proc = subprocess.Popen(["bash", str(ROOT / "scripts" / "serve_engine.sh"), str(out / "config.json"), str(a.port)], env=env, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True)
    (out / "server.pid").write_text(str(proc.pid))
    run = {"config": {"source": a.config, "set": a.set or [], "gpu": a.gpu}, "windows": {}}
    try:
        _wait_health(url, proc)
        for c in a.c:
            run["windows"][str(c)] = asyncio.run(_load(url, c, a.warm_s, a.measure_s))
            print(json.dumps({"c": c, **run["windows"][str(c)]["client"]}), flush=True)
            time.sleep(3)  # trace rows of the window are flushed within a second
    finally:
        (out / "run.json").write_text(json.dumps(run, indent=1))
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)  # serve_engine.sh stops the server and its MPS daemon
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
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
    r.add_argument("--config", default=str(ROOT / "configs" / "engine" / "track_s_fast.json"))
    r.add_argument("--warm-s", type=float, default=20.0)
    r.add_argument("--measure-s", type=float, default=40.0)
    r.add_argument("--set", action="append", help="KEY=JSON override of the engine config")
    an = sub.add_parser("analyze")
    an.add_argument("dir")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        return cmd_run(a)
    _print(analyze(Path(a.dir)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
