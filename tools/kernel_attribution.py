"""Kernel-activity attribution of a traced engine run (diagnostic; ``tools/engine_profile.py analyze``).

Inputs, all under one trace directory (``engine.runtime.stage_trace``):
* ``procs/<proc>-<pid>.json``: every process that made a stage trace (engine, t2w worker, feature processes);
* ``<name>.jsonl``: stage spans per thread (``_start`` row: pid and thread ident; ``_hb`` heartbeats);
* ``kernels/<proc>-<pid>-<tag>.json[.gz]`` + ``.markers.json``: the CUDA activity (CUPTI) of each process over each
  kernel window, with the marker kernels that align its clock to the host clock;
* ``requests.jsonl``: per-request timelines (engine and worker events on the shared host clock);
* ``gpm.jsonl``: device-wide hardware counters (``tools/gpm_sampler.py``).

What it reports for each kernel window, on the host clock:
* **kernel activity**: the union of the intervals in which a kernel, memcpy or memset of a process ran, per process
  and over all processes (the device-wide "some kernel is running" fraction), and the summed busy time (above the
  union when processes overlap under MPS);
* **hardware counters** over the same interval: SM utilization, occupancy, DRAM bandwidth, tensor-pipe utilization.
  Kernel activity says when something ran; the counters say how much of the GPU it used.  Neither alone shows
  saturation;
* **stages**: each kernel is attributed to the innermost stage span of its process and launching thread whose host
  interval contains its launch (the CUPTI correlation of kernel and runtime call).  Per stage: spans, host time, the
  stream-elapsed time between its CUDA events (it includes submission gaps, synchronization and waits), and the time
  its kernels ran (self: launched directly in the stage; total: with its inner stages), plus the delay from each launch
  call to its first kernel's start (queueing behind earlier work; a graph launch counts once);
* **blocking calls**: per stage (innermost, self), the host time spent inside runtime calls that wait on the GPU
  (synchronizations, copies), so a stage's host time splits into those waits and the rest (Python and launches);
* **alignment**: per process, the half-width of the offset interval the markers allow before and after the capture,
  and the drift between the two.

Rejected (``TraceError``), never read as zero: a registered process without a complete capture of a window, a missed
or failed capture, a capture that does not cover its window, markers that disagree, a stage file without a heartbeat
after the window, a process count below the one expected.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

ACTIVITY = ("kernel", "gpu_memcpy", "gpu_memset")
LAUNCH = ("cuda_runtime", "cuda_driver")
MARKER = "spin_kernel"
BLOCKING = ("Synchronize", "cudaMemcpy", "cuMemcpy")  # runtime/driver calls that can block the host on the GPU
ALIGN_TOL_NS = 1000  # timestamp granularity allowed when intersecting marker bounds


class TraceError(Exception):
    pass


def thread_key(ident: int) -> int:
    """The thread id the profiler's runtime records carry: abs(int32(pthread_self))."""
    v = ident & 0xFFFFFFFF
    return abs(v - (1 << 32) if v >= 1 << 31 else v)


def union(iv, w0: float, w1: float) -> float:
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


def _clip_sum(iv, w0: float, w1: float) -> float:
    return sum(max(0.0, min(b, w1) - max(a, w0)) for a, b in iv)


def _pct(xs: list, q: float):
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    i = int(k)
    return xs[i] if i + 1 >= len(xs) else xs[i] + (xs[i + 1] - xs[i]) * (k - i)


def alignment(markers_start: list, markers_end: list, spins: list) -> tuple:
    """Offset (ns) from profiler time to host time from marker kernels: each marker's device start lies between the host
    timestamps around its launch and completion.  ``spins``: (ts_us, dur_us) of the marker kernels in time order.
    Returns (function profiler-us -> host seconds, {half-widths and drift in microseconds})."""
    if len(spins) != len(markers_start) + len(markers_end):
        raise TraceError(f"{len(spins)} marker kernels for {len(markers_start) + len(markers_end)} markers")
    out = []
    for ms, sp in ((markers_start, spins[:len(markers_start)]), (markers_end, spins[len(markers_start):])):
        lo = max(t0 - ts * 1000 for (t0, _), (ts, _) in zip(ms, sp))
        hi = min(t1 - (ts + dur) * 1000 for (_, t1), (ts, dur) in zip(ms, sp))
        if lo > hi + ALIGN_TOL_NS:
            raise TraceError(f"marker bounds disagree by {(lo - hi) / 1000:.1f} us")
        mid = (lo + hi) / 2
        out.append((mid, max(0.0, hi - lo) / 2, sum(ts for ts, _ in sp) / len(sp)))
    (m0, h0, a0), (m1, h1, a1) = out
    slope = (m1 - m0) / (a1 - a0) if a1 != a0 else 0.0

    def to_host(ts_us: float) -> float:
        return (ts_us * 1000 + m0 + slope * (ts_us - a0)) / 1e9

    return to_host, {"halfwidth_start_us": h0 / 1000, "halfwidth_end_us": h1 / 1000, "drift_us": (m1 - m0) / 1000}


def load_capture(kdir: Path, marker_file: Path) -> dict:
    rec = json.loads(marker_file.read_text())
    w = rec["window"]
    name = marker_file.name
    if rec.get("missed"):
        raise TraceError(f"{name}: capture missed its window")
    if rec.get("error"):
        raise TraceError(f"{name}: capture failed: {rec['error']}")
    ms, me = rec["markers_start"], rec["markers_end"]
    if not ms or not me or ms[-1][1] / 1e9 > w["t0"] or me[0][0] / 1e9 < w["t1"]:
        raise TraceError(f"{name}: capture does not cover its window")
    tr = kdir / rec["trace"]
    if not tr.exists() and (kdir / (rec["trace"] + ".gz")).exists():
        tr = kdir / (rec["trace"] + ".gz")  # compressed after the run
    if not tr.exists():
        raise TraceError(f"{name}: trace file {tr.name} missing")
    events = [e for e in json.load(gzip.open(tr) if tr.suffix == ".gz" else open(tr))["traceEvents"] if e.get("ph") == "X"]
    spins = sorted((e["ts"], e["dur"]) for e in events if e.get("cat") == "kernel" and MARKER in e["name"])
    to_host, align = alignment(ms, me, spins)
    launches, waits = {}, []
    for e in events:
        if e.get("cat") in LAUNCH and "correlation" in e.get("args", {}):
            launches[e["args"]["correlation"]] = (to_host(e["ts"]), e.get("tid"))
        if e.get("cat") in LAUNCH and any(b in e["name"] for b in BLOCKING):
            s0 = to_host(e["ts"])
            waits.append({"launch": s0, "s": s0, "e": s0 + e["dur"] / 1e6, "tid": e.get("tid"), "name": e["name"]})
    acts = []
    for e in events:
        if e.get("cat") in ACTIVITY and MARKER not in e["name"]:
            corr = e.get("args", {}).get("correlation")
            lt, tid = launches.get(corr, (None, None))
            s = to_host(e["ts"])
            acts.append({"s": s, "e": s + e["dur"] / 1e6, "cat": e["cat"], "name": e["name"], "launch": lt, "tid": tid, "corr": corr})
    return {"proc": rec["proc"], "pid": rec["pid"], "window": w, "align": align, "acts": acts, "waits": waits}


def load_stage_files(d: Path) -> dict:
    """{file stem: {"pid", "ident", "rows", "hb"}} of every stage file."""
    out = {}
    for p in sorted(d.glob("*.jsonl")):
        if p.name in ("requests.jsonl", "gpm.jsonl"):
            continue
        rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
        starts = [r for r in rows if r["s"] == "_start"]
        if not starts:  # a stage file of an older format: no process identity
            raise TraceError(f"{p.name}: no _start row")
        out[p.stem] = {"pid": starts[-1]["pid"], "ident": starts[-1].get("ident"), "rows": [r for r in rows if not r["s"].startswith("_")],
                       "hb": max((r["c0"] for r in rows if r["s"] == "_hb"), default=None)}
    return out


def _assign(spans: list, kernels: list) -> list:
    """For kernels (sorted by launch) of one thread: the stack of spans open at each launch (outermost first)."""
    ev = []
    for i, r in enumerate(spans):
        ev.append((r["c0"], 0, -(r["c1"] - r["c0"]), i))
        ev.append((r["c1"], 2, 0, i))
    for j, k in enumerate(kernels):
        ev.append((k["launch"], 1, 0, j))
    ev.sort()
    stack: list = []
    out: list = [None] * len(kernels)
    for _, kind, _, i in ev:
        if kind == 0:
            stack.append(i)
        elif kind == 2:
            if i in stack:
                stack.remove(i)
        else:
            out[i] = list(stack)
    return out


def attribute_window(tag: str, w0: float, w1: float, captures: list, stage_files: dict, gpm: list | None = None) -> dict:
    span_s = w1 - w0
    res = {"tag": tag, "t0": w0, "t1": w1, "window_s": span_s, "alignment": {}, "processes": {}, "stages": {}}
    all_iv = []
    stages: dict = defaultdict(lambda: {"spans": 0, "rows": 0, "host_s": 0.0, "elapsed_s": 0.0, "self_iv": [], "total_iv": [], "launches": {}, "kernels": 0, "wait_s": 0.0, "waits": 0})
    for cap in captures:
        key = f"{cap['proc']}-{cap['pid']}"
        res["alignment"][key] = cap["align"]
        acts = [a for a in cap["acts"] if a["e"] > w0 and a["s"] < w1]
        iv = [(a["s"], a["e"]) for a in acts]
        all_iv += iv
        by_cat = defaultdict(list)
        for a in acts:
            by_cat[a["cat"]].append((a["s"], a["e"]))
        res["processes"][key] = {"active_frac": union(iv, w0, w1) / span_s, "busy_sum_frac": _clip_sum(iv, w0, w1) / span_s,
                                 "kernels": len(by_cat["kernel"]), **{f"{c}_frac": union(v, w0, w1) / span_s for c, v in by_cat.items()}}
        files = {stem: f for stem, f in stage_files.items() if f["pid"] == cap["pid"]}
        threads = {thread_key(f["ident"]): (stem, f) for stem, f in files.items() if f["ident"] is not None}
        by_tid = defaultdict(list)
        unattributed = []
        for a in acts:
            if a["launch"] is not None and a["tid"] in threads:
                by_tid[a["tid"]].append(a)
            else:
                unattributed.append(a)
        for tid, ks in by_tid.items():
            stem, f = threads[tid]
            spans = [r for r in f["rows"] if r["c1"] >= w0 - 1.0 and r["c0"] <= w1 + 1.0]
            ks.sort(key=lambda a: a["launch"])
            for a, stack in zip(ks, _assign(spans, ks)):
                if not stack:
                    unattributed.append(a)
                    continue
                names = [f"{cap['proc']}.{spans[i]['s']}" for i in stack]
                inner = stages[names[-1]]
                inner["self_iv"].append((a["s"], a["e"]))
                inner["kernels"] += 1
                first_start = inner["launches"].get(a["corr"])
                if first_start is None or a["s"] < first_start[0]:
                    inner["launches"][a["corr"]] = (a["s"], a["launch"])
                for nm in set(names):
                    stages[nm]["total_iv"].append((a["s"], a["e"]))
        # blocking calls (synchronizations, copies): their host time, in the innermost stage open when they began
        wby = defaultdict(list)
        for wt in cap.get("waits", []):
            if wt["e"] > w0 and wt["s"] < w1 and wt["tid"] in threads:
                wby[wt["tid"]].append(wt)
        for tid, ws in wby.items():
            stem, f = threads[tid]
            spans = [r for r in f["rows"] if r["c1"] >= w0 - 1.0 and r["c0"] <= w1 + 1.0]
            ws.sort(key=lambda a: a["launch"])
            for wt, stack in zip(ws, _assign(spans, ws)):
                if stack:
                    st = stages[f"{cap['proc']}.{spans[stack[-1]]['s']}"]
                    st["wait_s"] += max(0.0, min(wt["e"], w1) - max(wt["s"], w0))
                    st["waits"] += 1
        if unattributed:
            u = stages[f"{cap['proc']}.(outside stages)"]
            u["self_iv"] += [(a["s"], a["e"]) for a in unattributed]
            u["total_iv"] += [(a["s"], a["e"]) for a in unattributed]
            u["kernels"] += len(unattributed)
        for f in files.values():
            for r in f["rows"]:
                if r["c1"] > w0 and r["c0"] < w1:
                    st = stages[f"{cap['proc']}.{r['s']}"]
                    st["spans"] += 1
                    st["rows"] += r["n"]
                    st["host_s"] += max(0.0, min(r["c1"], w1) - max(r["c0"], w0))
                    st["elapsed_s"] += max(0.0, min(r["g1"], w1) - max(r["g0"], w0))
    for nm, st in sorted(stages.items()):
        n = st["spans"]
        st["delay"] = [s0 - lt for s0, lt in st["launches"].values()]  # per launch (a graph launch: its first kernel)
        res["stages"][nm] = {
            "spans_per_s": n / span_s, "rows_mean": st["rows"] / n if n else None,
            "host_ms": 1000 * st["host_s"] / n if n else None, "elapsed_ms": 1000 * st["elapsed_s"] / n if n else None,
            "kernel_self_ms": 1000 * union(st["self_iv"], w0, w1) / n if n else None,
            "kernel_total_ms": 1000 * union(st["total_iv"], w0, w1) / n if n else None,
            "host_frac": st["host_s"] / span_s, "elapsed_frac": st["elapsed_s"] / span_s,
            "kernel_self_frac": union(st["self_iv"], w0, w1) / span_s, "kernel_total_frac": union(st["total_iv"], w0, w1) / span_s,
            "kernels_per_s": st["kernels"] / span_s, "blocking_ms": 1000 * st["wait_s"] / n if n else None, "blocking_calls_per_s": st["waits"] / span_s,
            "launch_to_start_ms_p50": None if not st["delay"] else 1000 * _pct(st["delay"], 0.5),
            "launch_to_start_ms_p95": None if not st["delay"] else 1000 * _pct(st["delay"], 0.95)}
    res["device"] = {"active_frac": union(all_iv, w0, w1) / span_s, "busy_sum_frac": _clip_sum(all_iv, w0, w1) / span_s}
    if gpm:
        res["gpm"] = gpm_mean(gpm, w0, w1)
    return res


def gpm_mean(rows: list, w0: float, w1: float) -> dict:
    """Overlap-weighted means of the GPM rows over [w0, w1] (percent), with the covered fraction."""
    acc, wsum = defaultdict(float), 0.0
    for r in rows:
        ov = max(0.0, min(r["t1"], w1) - max(r["t0"], w0))
        if ov <= 0:
            continue
        wsum += ov
        for k, v in r.items():
            if k not in ("t0", "t1") and v is not None:
                acc[k] += v * ov
    return {"covered_frac": wsum / (w1 - w0), **{k: v / wsum for k, v in acc.items()}} if wsum else {"covered_frac": 0.0}


def check_complete(d: Path, tags: list, expected: dict, stage_files: dict) -> list[Path]:
    """The marker files of every registered process for every window; raises when anything required is missing."""
    procs = [json.loads(p.read_text()) for p in sorted((d / "procs").glob("*.json"))]
    counts = defaultdict(int)
    for p in procs:
        counts[p["proc"]] += 1
    for name, n in expected.items():
        if counts[name] != n:
            raise TraceError(f"{counts[name]} registered {name} processes, {n} expected")
    missing, files = [], []
    for p in procs:
        for tag in tags:
            f = d / "kernels" / f"{p['proc']}-{p['pid']}-{tag}.markers.json"
            (files if f.exists() else missing).append(f)
    if missing:
        raise TraceError(f"captures missing: {[m.name for m in missing]}")
    return files


def check_heartbeats(stage_files: dict, t1: float) -> None:
    late = [stem for stem, f in stage_files.items() if f["hb"] is None or f["hb"] < t1]
    if late:
        raise TraceError(f"stage files without a heartbeat after the window (rows may be incomplete): {late}")


def first(events: list, prefix: str):
    """Host time of the first event named ``prefix`` or starting with ``prefix + ':'``."""
    for e, t in events:
        if e == prefix or e.startswith(prefix + ":"):
            return t
    return None


SEGMENTS = [  # (name, from event, to event): the first audio's path
    ("engine_queue", "queued", "admitted"),
    ("features", "w_open", "w_features"),
    ("setup_wait", "w_features", "w_setup"),
    ("setup_call", "w_setup", "w_setup_done"),
    ("prefill", "admitted", "prefilled"),
    ("first_text_chunk", "prefilled", "text_chunk"),
    ("talker_to_window0", "text_chunk", "window_ready:0"),
    ("transport_to_worker", "window_ready:0", "w_recv:0"),
    ("worker_wait", "w_recv:0", "w_decode:0"),
    ("decode_call", "w_decode:0", "w_decoded:0"),
    ("transport_to_engine", "w_decoded:0", "audio_recv:0"),
    ("engine_to_http", "audio_recv:0", "first_yield"),
    ("server_first_audio", "queued", "first_yield"),
]


def critical_path(requests: list, w0: float, w1: float, client: dict | None = None) -> dict:
    """Medians / p95 (ms) of each segment of the first audio's path over the requests whose first audio left the
    engine inside [w0, w1]; ``client``: engine request id -> (client dispatch, client first audio) for the HTTP ends.
    Also every window's transport legs.  Setup and the first window race: ``window0_vs_setup_ms`` is how long the
    first window waited for its stream's setup (negative: the setup was ready first)."""
    segs = defaultdict(list)
    legs = defaultdict(list)
    n = 0
    for r in requests:
        ev = r["events"]
        fy = first(ev, "first_yield")
        if fy is None or not (w0 <= fy <= w1):
            continue
        n += 1
        names = {}
        for e, t in ev:
            names.setdefault(e, t)
        for name, a, b in SEGMENTS:
            ta, tb = (first(ev, a) if ":" not in a else names.get(a)), (first(ev, b) if ":" not in b else names.get(b))
            if ta is not None and tb is not None:
                segs[name].append(1000 * (tb - ta))
        ws, wr = names.get("w_setup_done"), names.get("w_recv:0")
        if ws is not None and wr is not None:
            segs["window0_vs_setup"].append(1000 * (ws - wr))
        if client and r["rid"] in client:
            td, ta = client[r["rid"]]
            if ta is not None:
                segs["client_ttfa"].append(1000 * (ta - td))
                segs["http_to_client"].append(1000 * (ta - fy))
                segs["client_to_engine"].append(1000 * (names["queued"] - td))
        i = 0
        while f"window_ready:{i}" in names:
            for leg, a, b in (("to_worker", "window_ready", "w_recv"), ("worker_wait", "w_recv", "w_decode"), ("decode_call", "w_decode", "w_decoded"),
                              ("to_engine", "w_decoded", "audio_recv")):
                ta, tb = names.get(f"{a}:{i}"), names.get(f"{b}:{i}")
                if ta is not None and tb is not None:
                    legs[leg].append(1000 * (tb - ta))
            i += 1
    summ = lambda xs: {"n": len(xs), "p50_ms": _pct(xs, 0.5), "p95_ms": _pct(xs, 0.95), "mean_ms": sum(xs) / len(xs)}  # noqa: E731
    return {"requests": n, "first_audio": {k: summ(v) for k, v in segs.items()}, "every_window": {k: summ(v) for k, v in legs.items()}}


def analyze_kernels(d: Path, tags: dict, expected: dict) -> dict:
    """``tags``: kernel-window tag -> (t0, t1); ``expected``: process name -> count.  Raises ``TraceError`` on an
    incomplete trace."""
    stage_files = load_stage_files(d)
    files = check_complete(d, list(tags), expected, stage_files)
    gpm_p = d / "gpm.jsonl"
    gpm = [json.loads(x) for x in gpm_p.read_text().splitlines() if x.strip()] if gpm_p.exists() else None
    out = {}
    for tag, (t0, t1) in tags.items():
        check_heartbeats(stage_files, t1)
        caps = [load_capture(d / "kernels", f) for f in files if f.name.endswith(f"-{tag}.markers.json")]
        out[tag] = attribute_window(tag, t0, t1, caps, stage_files, gpm)
    return out
