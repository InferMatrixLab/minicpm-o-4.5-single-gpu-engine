"""Per-stage traces of the baseline stacks under closed-loop load.

    python tools/trace_baselines.py run --stack vllm|sglang --gpu G --port P --concurrency C --out results/l20x/traces/<stack>_c<C>
    python tools/trace_baselines.py extract --trace-dir results/l20x/traces/<stack>_c<C>

``run`` launches a fresh server with each stack's own instrumentation, sends a
warmup of 2C requests, then two windows on the evaluation IDs:

* a stage-timing window of max(4C, 32) requests with the cheap per-request
  instrumentation only (vllm-omni ``return_stage_metrics`` in the request body;
  sglang-omni's JSONL event recorder via ``/start_request_profile``);
* a kernel window of 2C requests under the stack's torch profiler (every stage
  process), used only for GPU time per audio-second and CPU-vs-GPU attribution.
  Latencies inside the profiled window are diagnostic only.

``extract`` turns the saved traces into ``trace_summary.json``: per-request and
aggregate Thinker prefill/decode, time to first Talker output, first
Token2Wav/Code2Wav output, inter-stage transport where the stack records it,
and per-process GPU kernel time, CPU-side launch time and GPU busy seconds per
audio-second.
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import gzip
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.adapters import get_adapter  # noqa: E402
from bench.client import make_session  # noqa: E402
from bench.experiment import Candidate  # noqa: E402
from bench.metrics import request_metrics  # noqa: E402
from bench.protocol import closed_loop  # noqa: E402
from bench.servers import Server  # noqa: E402
from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets  # noqa: E402

VLLM_CONFIG = "minicpmo_4_5_turn.yaml"
PROFILE = {"profiler": "torch", "torch_profiler_with_stack": False, "torch_profiler_record_shapes": False, "torch_profiler_use_gzip": True,
           "torch_profiler_dump_cuda_time_total": True}


class StageMetricsAdapter:
    """The vllm-omni adapter with ``return_stage_metrics`` added to every request body."""

    def __init__(self, inner):
        self.inner = inner

    def __getattr__(self, k):
        return getattr(self.inner, k)

    def build_payload(self, req):
        return {**self.inner.build_payload(req), "return_stage_metrics": True}


def _post(url: str, body: dict | None = None) -> str:
    r = urllib.request.urlopen(urllib.request.Request(url, data=json.dumps(body or {}).encode(), headers={"Content-Type": "application/json"}),
                               timeout=600)
    return r.read().decode()


def server_for(stack: str, gpu: int, port: int, out: Path) -> Server:
    from bench.cli import make_candidate, server_spec

    prof = out / "torch"
    prof.mkdir(parents=True, exist_ok=True)
    if stack == "vllm":
        cand = make_candidate({"adapter": "vllm-omni", "config": VLLM_CONFIG, "label": "trace", "mps": "off"})
        p = {**PROFILE, "torch_profiler_dir": str(prof)}
        extra = ["--stage-overrides", json.dumps({str(s): {"profiler_config": p} for s in range(3)})]
        spec = server_spec(Candidate(cand.adapter, cand.config, tuple(extra), cand.env, cand.label, cand.deploy_files_sha256, cand.mps), port, gpu)
        spec.env["VLLM_CUSTOM_SCOPES_FOR_PROFILING"] = "1"
    else:
        cand = make_candidate({"adapter": "sglang-omni", "config": "speech", "label": "trace", "mps": "off"})
        spec = server_spec(cand, port, gpu)
        spec.env["SGLANG_TORCH_PROFILER_DIR"] = str(prof)
    return Server(spec, out / "server.log")


async def _window(url, adapter, reqs, c, record_raw):
    async with make_session(connections=max(64, 4 * c)) as s:
        t0 = time.perf_counter()
        trs = await closed_loop(s, url, adapter, reqs, c, 180.0, record_raw=record_raw)
        return trs, time.perf_counter() - t0


def _save(path: Path, trs, wall: float) -> dict:
    recs = []
    for t in trs:
        m = request_metrics(t)
        body = b"".join(b for _, b in (getattr(t, "raw_chunks", None) or [])).decode("utf-8", "replace")
        recs.append({"request_id": t.request_id, "ok": m.ok, "duration_s": m.duration_s, "ttfa_s": m.ttfa_s, "e2e_s": m.e2e_s, "rtf": m.rtf,
                     "raw": [ln for ln in body.splitlines() if "stage_metrics" in ln][-3:]})
    doc = {"wall_s": wall, "requests": recs}
    path.write_text(json.dumps(doc))
    return doc


def cmd_run(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    c = a.concurrency
    warm = canonical_requests(rows, sets.warmup(c))
    ids = sets.eval(c)
    timing = canonical_requests(rows, ids[: max(4 * c, 32)])
    kernel = canonical_requests(rows, ids[max(4 * c, 32): max(4 * c, 32) + 2 * c])
    inner = get_adapter("vllm-omni" if a.stack == "vllm" else "sglang-omni")
    adapter = StageMetricsAdapter(inner) if a.stack == "vllm" else inner
    meta = {"stack": a.stack, "concurrency": c, "gpu": a.gpu, "timing_ids": [r.request_id for r in timing], "kernel_ids": [r.request_id for r in kernel]}
    with server_for(a.stack, a.gpu, a.port, out) as srv:
        meta["server_evidence"] = srv.evidence()
        asyncio.run(_window(srv.url, adapter, warm, c, False))
        if a.stack == "sglang":
            _post(srv.url + "/start_request_profile", {"run_id": "timing", "event_dir": str(out / "events")})
        trs, wall = asyncio.run(_window(srv.url, adapter, timing, c, True))
        if a.stack == "sglang":
            _post(srv.url + "/stop_request_profile", {"run_id": "timing"})
        _save(out / "timing_window.json", trs, wall)
        if a.stack == "vllm":
            _post(srv.url + "/start_profile", {"stages": [0, 1, 2]})
        else:
            _post(srv.url + "/start_profile", {"run_id": "kernel", "enable_torch": True, "event_dir": str(out / "events_kernel")})
        t_start = time.time()
        trs, wall = asyncio.run(_window(srv.url, adapter, kernel, c, False))
        t_stop = time.time()
        _post(srv.url + "/stop_profile", {"stages": [0, 1, 2]} if a.stack == "vllm" else {"run_id": "kernel"})
        _save(out / "kernel_window.json", trs, wall)
        meta["kernel_window_unix"] = [t_start, t_stop]
        time.sleep(30)  # traces are written asynchronously after stop
    (out / "trace_meta.json").write_text(json.dumps(meta, indent=1))
    return 0


# --- extraction ---------------------------------------------------------------------------------
def _pct(v: list[float]) -> dict:
    v = [x for x in v if x is not None]
    if not v:
        return {"n": 0}
    a = np.asarray(v, dtype=float)
    return {"n": len(v), "mean": float(a.mean()), "p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95))}


def vllm_stage_rows(doc: dict) -> list[dict]:
    """Per-request stage metrics from the final SSE events (``metrics.stage_metrics``)."""
    out = []
    for r in doc["requests"]:
        sm = None
        for line in reversed(r["raw"]):
            line = line.strip()
            if not line.startswith("data:"):
                continue
            try:
                ev = json.loads(line[5:].strip())
            except json.JSONDecodeError:
                continue
            sm = (ev.get("metrics") or {}).get("stage_metrics")
            if sm:
                break
        if not sm:
            continue
        g = lambda s, k: (sm.get(s) or {}).get(k)  # noqa: E731
        out.append({"request_id": r["request_id"], "thinker_ttft_ms": g("0", "vllm_ttft_ms"), "thinker_gen_ms": g("0", "stage_gen_time_ms"),
                    "thinker_tpot_ms": g("0", "vllm_tpot_ms"), "talker_first_output_ms": g("1", "serving_time_to_first_output_ms"),
                    "talker_gen_ms": g("1", "stage_gen_time_ms"), "code2wav_first_output_ms": g("2", "serving_time_to_first_output_ms"),
                    "code2wav_gen_ms": g("2", "stage_gen_time_ms"), "audio_s": g("2", "audio_duration_s"), "client_ttfa_s": r["ttfa_s"]})
    return out


def sglang_stage_rows(event_dir: Path) -> list[dict]:
    ev: dict[str, list[dict]] = {}
    for f in glob.glob(str(event_dir / "**" / "*.jsonl"), recursive=True):
        for line in open(f):
            e = json.loads(line)
            ev.setdefault(e.get("request_id") or "", []).append(e)
    out = []
    for rid, es in ev.items():
        if not rid:
            continue
        t = {}
        for e in sorted(es, key=lambda e: e["timestamp_ns"]):
            t.setdefault((e.get("stage"), e["event_name"]), e["timestamp_ns"] / 1e6)
        adm = t.get((None, "request_admission")) or min(v for v in t.values())

        def d(a, b):
            return (t[b] - t[a]) if a in t and b in t else None

        out.append({"request_id": rid,
                    "thinker_prefill_ms": d(("thinker", "scheduler_prefill_start"), ("thinker", "scheduler_prefill_end")),
                    "thinker_decode_ms": d(("thinker", "scheduler_prefill_end"), ("thinker", "stage_complete")),
                    "thinker_to_talker_ms": d(("thinker", "stage_hop_sent"), ("talker", "stage_input_received")),
                    "talker_prefill_ms": d(("talker", "scheduler_prefill_start"), ("talker", "scheduler_prefill_end")),
                    "talker_decode_ms": d(("talker", "scheduler_prefill_end"), ("talker", "stage_complete")),
                    "talker_to_code2wav_ms": d(("talker", "stage_hop_sent"), ("code2wav", "stage_input_received")),
                    "code2wav_ms": d(("code2wav", "stage_dispatch"), ("code2wav", "stage_complete")),
                    "admission_to_thinker_ms": (t[("thinker", "scheduler_prefill_start")] - adm) if ("thinker", "scheduler_prefill_start") in t else None})
    return out


def kernel_summary(trace_files: list[str], audio_s: float, root: Path | None = None) -> dict:
    """GPU kernel time and CPU launch-side time per traced process from Chrome traces, keyed by each trace's path under
    ``root`` (stage processes write files with the same name, e.g. vllm-omni's per-stage trace_rank0.json.gz)."""
    procs = {}
    for f in trace_files:
        op = gzip.open if f.endswith(".gz") else open
        with op(f, "rt") as fh:
            doc = json.load(fh)
        evs = doc.get("traceEvents", doc if isinstance(doc, list) else [])
        gpu = sum(e.get("dur", 0) for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")) / 1e6
        launch = sum(e.get("dur", 0) for e in evs if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"
                     and any(s in e.get("name", "") for s in ("Launch", "GraphLaunch", "Memcpy"))) / 1e6
        kts = [(e["ts"], e["ts"] + e.get("dur", 0)) for e in evs if e.get("ph") == "X" and e.get("cat") == "kernel"]
        span = (max(b for _, b in kts) - min(a for a, _ in kts)) / 1e6 if kts else 0.0
        key = str(Path(f).relative_to(root)) if root is not None else str(f)
        if key in procs:
            raise ValueError(f"two traces map to {key}")
        procs[key] = {"gpu_kernel_s": gpu, "cpu_launch_s": launch, "kernel_span_s": span,
                               "gpu_s_per_audio_s": gpu / audio_s if audio_s else None, "kernels": len(kts)}
    return {"audio_s": audio_s, "processes": procs, "gpu_s_per_audio_s_total": (sum(p["gpu_kernel_s"] for p in procs.values()) / audio_s) if audio_s else None}


def cmd_extract(a) -> int:
    d = Path(a.trace_dir)
    meta = json.loads((d / "trace_meta.json").read_text())
    timing = json.loads((d / "timing_window.json").read_text())
    kernel = json.loads((d / "kernel_window.json").read_text())
    rows = vllm_stage_rows(timing) if meta["stack"] == "vllm" else sglang_stage_rows(d / "events")
    keys = sorted({k for r in rows for k in r if k != "request_id"})
    client = {k: _pct([r[k] for r in timing["requests"] if r["ok"]]) for k in ("ttfa_s", "e2e_s", "rtf", "duration_s")}
    files = sorted(glob.glob(str(d / "torch" / "**" / "*.json*"), recursive=True))
    files = [f for f in files if "trace" in Path(f).name]
    audio = sum(r["duration_s"] for r in kernel["requests"] if r["ok"])
    summ = {"stack": meta["stack"], "concurrency": meta["concurrency"], "client_timing_window": client,
            "timing_window_frames_s": 25 * sum(r["duration_s"] for r in timing["requests"] if r["ok"]) / timing["wall_s"],
            "stages": {k: _pct([r.get(k) for r in rows]) for k in keys}, "per_request": rows,
            "kernel_window": kernel_summary(files, audio, d / "torch") if files else {"error": "no torch traces found"}}
    (d / "trace_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps({k: v for k, v in summ.items() if k != "per_request"}, indent=1)[:4000])
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--stack", choices=["vllm", "sglang"], required=True)
    r.add_argument("--gpu", type=int, required=True)
    r.add_argument("--port", type=int, required=True)
    r.add_argument("--concurrency", type=int, required=True)
    r.add_argument("--out", required=True)
    e = sub.add_parser("extract")
    e.add_argument("--trace-dir", required=True)
    a = ap.parse_args(argv)
    return cmd_run(a) if a.cmd == "run" else cmd_extract(a)


if __name__ == "__main__":
    raise SystemExit(main())
