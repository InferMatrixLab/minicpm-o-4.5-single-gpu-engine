"""AC-6 robustness and sustained-load checks against a running engine server.

    python tools/robustness.py --url http://127.0.0.1:PORT --gpu-uuid GPU-... --config CONFIG --out DIR [--sustain-min 30]
        [--concurrency 32] [--acceptance]

Every request that is expected to complete goes through the harness client and its validity rules
(``bench.client.send_request`` + ``bench.metrics.request_metrics``: terminal event, no error event, valid PCM with
consistent offsets, no gaps/overlaps/retransmissions).  An SSE error followed by [DONE] is a failure.

Scenarios:
* mixed: concurrent text-only, image, audio and image+audio requests (all with a reference voice);
* cancel: clients that disconnect after their first audio chunk beside normal companions.  Every cancelling client must
  have received audio before it disconnected, the server must count every disconnect (``cancelled_streams``) and drain
  back to the warmed idle baseline;
* eos-boundary: many replies whose codec frame counts (``usage.audio_frames``) must cover the Token2Wav window boundary
  and its neighbours ((frames - first_chunk) mod 25 in {24, 0, 1});
* fallback: a reference voice long enough that its prompt setup exceeds every captured DiT graph bucket: the worker's
  eager-call counter must rise and the reply must be valid;
* overload: more requests than the admission limit at once: each is valid or an HTTP 503, and some are 503;
* sustain: continuous refill at C for N minutes: C persistent clients each start the next request of a cycling row
  sequence as soon as their previous one ends, until the deadline (the in-flight count is sampled); every request valid,
  continuation stall <= 2 s; after the drain the counters equal the warmed baseline and memory (engine RSS, Token2Wav
  worker RSS, each feature process's RSS by PID, GPU memory used on the served GPU) is within 1% of the steady-state
  sample taken 60 s into the run (the predeclared reference point).  A missing reading fails.

Counters: every reading must carry the declared schema (``REQUIRED``): a missing, null or non-numeric counter fails, the
Token2Wav worker's counters must be fresh (``t2w_stats_time`` at most ``STALE_S`` old), and each feature process must
report its RSS.  The idle baseline is the drained state after the warm-up scenario (``mixed``); later drained states
must equal it on every capacity counter (KV pages, cache slots, CNN rows, streams, queue, admission).

Before any load the report binds the server's identity (``/info``: config bytes, engine source hashes, backend) against
the harness's resolution of ``--config`` (``bench.cli.resolve_engine_deploy``) and the served CUDA device (``/info``
``device_uuid``) against ``--gpu-uuid``; GPU memory is read for that UUID and the report records its SKU, UUID and
driver.  A mismatch ends the run before load.  ``--acceptance`` requires 30 minutes at C >= 32; other runs are labelled
development.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import subprocess
import sys
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.adapters import get_adapter  # noqa: E402
from bench.client import make_session, send_request  # noqa: E402
from bench.metrics import request_metrics  # noqa: E402
from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets  # noqa: E402

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
WINDOW = 25  # codec frames per steady Token2Wav window (engine.token2wav_chunker.CHUNK)
IDLE_KEYS = ("admitted", "live", "queued", "thinker_kv_pages_used", "talker_kv_pages_used", "audio_live", "t2w_requests", "t2w_streams",
             "t2w_cache_slots_used")  # zero when drained
BASELINE_KEYS = IDLE_KEYS + ("t2w_cnn_rows_free", "thinker_kv_pages", "talker_kv_pages", "max_live")  # equal to the warmed baseline
REQUIRED = BASELINE_KEYS + ("cancelled_streams", "t2w_dit_eager_calls", "rss_kib", "t2w_rss_kib", "t2w_stats_time", "feature_processes")
MEMORY_KEYS = ("rss_kib", "t2w_rss_kib", "feature_rss_kib", "gpu_memory_used_mib")
STALE_S = 5.0  # the worker publishes its counters once a second


async def metrics(url: str) -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.get(url + "/metrics") as r:
            return await r.json()


async def info(url: str) -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.get(url + "/info") as r:
            return await r.json() if r.status == 200 else {}


def gpu_query(uuid: str | None) -> dict | None:
    """The GPU with this UUID (``nvidia-smi -i <UUID>``): SKU, UUID, driver, index and memory used; None if unresolvable."""
    if not uuid:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-i", uuid, "--query-gpu=name,uuid,driver_version,index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
        name, u, driver, index, used = [x.strip() for x in out.split(",")]
        return {"sku": name, "uuid": u, "driver": driver, "index": int(index), "memory_used_mib": float(used)}
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def gpu_memory_used_mib(uuid: str | None) -> float | None:
    g = gpu_query(uuid)
    return g["memory_used_mib"] if g else None


def gpu_check(served: dict, expected_uuid: str | None) -> dict:
    """The served CUDA device is the declared GPU, and nvidia-smi resolves it (memory readings use this UUID)."""
    uuid = (served or {}).get("device_uuid")
    g = gpu_query(uuid)
    problems = []
    if not uuid:
        problems.append("the server does not publish its device_uuid")
    elif g is None or g["uuid"] != uuid:
        problems.append(f"nvidia-smi does not resolve the served device {uuid}")
    if not expected_uuid:
        problems.append("no --gpu-uuid declared")
    elif uuid != expected_uuid:
        problems.append(f"served device {uuid} != declared {expected_uuid}")
    return {"served_uuid": uuid, "declared_uuid": expected_uuid, "nvidia_smi": g, "problems": problems, "matches": not problems}


async def sample(url: str, gpu_uuid: str | None) -> dict:
    return {"t": time.time(), **await metrics(url), "gpu_memory_used_mib": gpu_memory_used_mib(gpu_uuid)}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def schema_problems(m: dict, now: float | None = None) -> list[str]:
    """Every declared counter present and numeric, the worker's counters fresh, every feature process's RSS read."""
    now = time.time() if now is None else now
    out = [f"{k} missing" if k not in m else f"{k}={m[k]!r} is not a number" for k in REQUIRED if not _num(m.get(k))]
    t = m.get("t2w_stats_time")
    if _num(t) and now - t > STALE_S:
        out.append(f"worker counters are {now - t:.1f} s old")
    n, by_pid = m.get("feature_processes"), m.get("feature_rss_by_pid")
    if _num(n) and n > 0:
        if not isinstance(by_pid, dict) or len(by_pid) != n:
            out.append(f"feature RSS reported for {len(by_pid) if isinstance(by_pid, dict) else 0} of {n} processes")
        else:
            out += [f"feature process {p} RSS unreadable" for p, v in by_pid.items() if not _num(v)]
        if not _num(m.get("feature_rss_kib")):
            out.append("feature_rss_kib is not a number")
    return out


def idle_problems(m: dict, now: float | None = None) -> list[str]:
    return schema_problems(m, now) + [f"{k}={m[k]}" for k in IDLE_KEYS if _num(m.get(k)) and m[k]]


def baseline_problems(base: dict, m: dict) -> list[str]:
    """A drained state differs from the warmed idle baseline on a capacity counter (a leaked row, slot, page or stream)."""
    if not base:
        return ["no idle baseline"]
    return [f"{k}={m.get(k)!r} != baseline {base.get(k)!r}" for k in BASELINE_KEYS if m.get(k) != base.get(k)]


async def wait_idle(url: str, timeout_s: float = 120.0) -> dict:
    """The drained state: nothing admitted, queued or live, the Token2Wav worker holding nothing; returned after a further
    3 s (several worker counter periods), or at the timeout whatever it is."""
    t0 = time.time()
    while True:
        m = await metrics(url)
        if not idle_problems(m) or time.time() - t0 > timeout_s:
            await asyncio.sleep(3)
            return await metrics(url)
        await asyncio.sleep(1)


def _data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def media_payloads(n: int, ref_wav: str | None = None) -> list[dict]:
    img = _data_url(MODEL / "assets" / "fossil.png", "image/png")
    wav = _data_url(MODEL / "assets" / "HT_ref_audio.wav", "audio/wav")
    forms = [
        [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": "Describe the picture in one sentence."}],
        [{"type": "audio_url", "audio_url": {"url": wav}}, {"type": "text", "text": "What does the speaker say?"}],
        [{"type": "image_url", "image_url": {"url": img}}, {"type": "audio_url", "audio_url": {"url": wav}}, {"type": "text", "text": "Relate them."}],
        [{"type": "text", "text": "Say hello."}],
    ]
    return [{"model": "openbmb/MiniCPM-o-4_5", "messages": [{"role": "user", "content": forms[i % len(forms)]}], "modalities": ["text", "audio"],
             "ref_audio": ref_wav or wav, "stream": True, "stream_options": {"include_usage": True}, "max_tokens": 64, "seed": i} for i in range(n)]


def long_reference_wav(seconds: float = 48.0) -> str:
    """A reference voice of ``seconds`` (the default voice repeated): its prompt setup exceeds every captured DiT bucket."""
    with wave.open(str(MODEL / "assets" / "HT_ref_audio.wav")) as w:
        params, frames = w.getparams(), w.readframes(w.getnframes())
    reps = int(np.ceil(seconds * params.framerate / max(1, params.nframes)))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as o:
        o.setparams(params)
        o.writeframes(frames * reps)
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


async def harness_stream(session, url: str, adapter, payload: dict, rid: str, timeout_s: float = 180.0):
    """One request through the production client and validity rules; returns (metrics, trace)."""
    tr = await send_request(session, url, adapter, SimpleNamespace(request_id=rid), timeout_s=timeout_s, payload=payload)
    return request_metrics(tr), tr


async def cancel_stream(url: str, payload: dict) -> dict:
    """Disconnect after the first audio chunk; records whether audio arrived before the disconnect."""
    audio = 0
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url + "/v1/chat/completions", json=payload) as r:
                if r.status != 200:
                    return {"status": r.status, "audio_events": 0}
                async for line in r.content:
                    if b'"modality": "audio"' in line:
                        audio += 1
                        break
    except (aiohttp.ClientError, asyncio.TimeoutError):
        pass
    return {"status": 200, "audio_events": audio}


def _summ(ms) -> dict:
    bad = [m for m in ms if not m.ok]
    return {"n": len(ms), "ok": len(ms) - len(bad), "invalid_reasons": sorted({r for m in bad for r in m.invalid_reasons})}


async def scenario_mixed(url, adapter, n: int = 16) -> dict:
    async with make_session(connections=4 * n) as s:
        res = await asyncio.gather(*[harness_stream(s, url, adapter, p, f"mixed-{i}") for i, p in enumerate(media_payloads(n))])
    out = _summ([m for m, _ in res])
    out["passed"] = out["ok"] == out["n"] > 0
    return out


async def scenario_cancel(url, adapter, payloads: list[dict], baseline: dict | None = None) -> dict:
    before = await metrics(url)
    half = len(payloads) // 2
    async with make_session(connections=4 * len(payloads)) as s:
        comp = asyncio.gather(*[harness_stream(s, url, adapter, p, f"companion-{i}") for i, p in enumerate(payloads[half:])])
        canc = asyncio.gather(*[cancel_stream(url, p) for p in payloads[:half]])
        cres, kres = await asyncio.gather(comp, canc)
    after = await wait_idle(url)
    a, b = after.get("cancelled_streams"), before.get("cancelled_streams")
    observed = a - b if _num(a) and _num(b) else None
    out = {"companions": _summ([m for m, _ in cres]), "cancelled_sent": half, "cancelled_with_audio": sum(k["audio_events"] > 0 for k in kres),
           "cancelled_observed": observed, "idle_problems": idle_problems(after),
           "baseline_problems": baseline_problems(baseline, after) if baseline is not None else []}
    out["passed"] = (out["companions"]["ok"] == out["companions"]["n"] and out["cancelled_with_audio"] == half and observed is not None
                     and observed >= half and not out["idle_problems"] and not out["baseline_problems"])
    return out


async def scenario_eos_boundary(url, adapter, reqs, first_chunk: int, concurrency: int = 32) -> dict:
    payloads = [adapter.build_payload(r) for r in reqs]
    for p in payloads:
        p["stream_options"] = {"include_usage": True}
    sem = asyncio.Semaphore(concurrency)
    async with make_session(connections=4 * concurrency) as s:
        async def one(i, p):
            async with sem:
                return await harness_stream(s, url, adapter, p, reqs[i].request_id)
        res = await asyncio.gather(*[one(i, p) for i, p in enumerate(payloads)])
    frames = [tr.usage.get("audio_frames") for _, tr in res]
    residues = sorted({(f - first_chunk) % WINDOW for f in frames if isinstance(f, int)})
    need = {WINDOW - 1, 0, 1}
    out = {**_summ([m for m, _ in res]), "first_chunk": first_chunk, "frames_reported": sum(isinstance(f, int) for f in frames),
           "boundary_residues_covered": sorted(need & set(residues)), "boundary_frame_counts": sorted({f for f in frames if isinstance(f, int)
                                                                                                       and (f - first_chunk) % WINDOW in need})[:20]}
    out["passed"] = out["ok"] == out["n"] > 0 and need <= set(residues)
    return out


async def scenario_fallback(url, adapter) -> dict:
    before = await metrics(url)
    p = media_payloads(4, ref_wav=long_reference_wav())[3]  # text-only reply in a 48 s reference voice
    async with make_session(connections=4) as s:
        m, _ = await harness_stream(s, url, adapter, p, "fallback-long-voice")
    await asyncio.sleep(2)
    after = await metrics(url)
    rose = (after.get("t2w_dit_eager_calls") or 0) - (before.get("t2w_dit_eager_calls") or 0)
    out = {**_summ([m]), "dit_eager_calls_added": rose, "counter_present": after.get("t2w_dit_eager_calls") is not None}
    out["passed"] = out["ok"] == 1 and out["counter_present"] and rose > 0
    return out


async def scenario_overload(url, adapter, reqs) -> dict:
    async with make_session(connections=4 * len(reqs)) as s:
        res = await asyncio.gather(*[harness_stream(s, url, adapter, adapter.build_payload(r), r.request_id) for r in reqs])
    statuses = sorted({tr.http_status for _, tr in res if tr.http_status is not None})
    ok = sum(m.ok for m, _ in res)
    rejected = sum(tr.http_status == 503 for _, tr in res)
    other = [m.request_id for m, tr in res if not m.ok and tr.http_status != 503]
    out = {"n": len(res), "ok": ok, "rejected_503": rejected, "statuses": statuses, "other_failures": other[:10]}
    out["passed"] = not other and rejected > 0 and ok > 0
    return out


async def continuous_load(url, adapter, reqs, concurrency: int, stop_at: float, state: dict, timeout_s: float = 180.0) -> list:
    """``concurrency`` persistent clients; each starts the next request of the cycling row sequence as soon as its previous
    one ends, until ``stop_at``.  ``state`` carries the in-flight count (read by the sampler) and the requests started."""
    payloads = [adapter.build_payload(r) for r in reqs]
    traces: list = []
    state.update(in_flight=0, started=0)
    async with make_session(connections=4 * concurrency) as s:
        async def client():
            while time.time() < stop_at:
                i = state["started"]
                state["started"] += 1  # no await between the previous request's end and this start: in flight stays C
                state["in_flight"] += 1
                try:
                    r = reqs[i % len(reqs)]
                    traces.append(await send_request(s, url, adapter, SimpleNamespace(request_id=f"{r.request_id}#{i // len(reqs)}"),
                                                     timeout_s=timeout_s, payload=payloads[i % len(reqs)]))
                finally:
                    state["in_flight"] -= 1
        await asyncio.gather(*(client() for _ in range(concurrency)))
    return traces


def memory_readings(m: dict) -> dict:
    """The memory readings of one sample: the engine-wide keys and each feature process's RSS by PID."""
    out = {k: m.get(k) for k in MEMORY_KEYS}
    for pid, v in (m.get("feature_rss_by_pid") or {}).items():
        out[f"feature_rss_kib[{pid}]"] = v
    return out


async def scenario_sustain(url, adapter, reqs, concurrency: int, minutes: float, gpu_uuid: str | None, baseline: dict | None = None,
                           interval_s: float = 10.0, steady_index: int = 6) -> dict:
    samples, state, stop = [], {}, time.time() + 60 * minutes

    async def sampler():
        while time.time() < stop:
            m = await sample(url, gpu_uuid)
            m["in_flight"] = state.get("in_flight")
            m["schema_problems"] = schema_problems(m)
            samples.append(m)
            await asyncio.sleep(interval_s)

    smp = asyncio.create_task(sampler())
    traces = await continuous_load(url, adapter, reqs, concurrency, stop, state)
    await smp
    ms = [request_metrics(t) for t in traces]
    await wait_idle(url)
    idle = await sample(url, gpu_uuid)
    steady = samples[min(steady_index, len(samples) - 1)] if samples else {}
    ref, end = memory_readings(steady), memory_readings(idle)
    drift = {k: (end[k] - a) / a if _num(a) and _num(end.get(k)) and a else None for k, a in ref.items()}
    drift.update({k: None for k in end if k not in ref})  # a reading that appeared later has no reference
    in_flight = [m["in_flight"] for m in samples if m["t"] < stop - 1.0]
    out = {"minutes": minutes, "concurrency": concurrency, "requests": len(ms), "requests_started": state.get("started", 0),
           "row_cycles": state.get("started", 0) / max(1, len(reqs)), "in_flight_min": min(in_flight, default=None),
           "in_flight_max": max(in_flight, default=None), "invalid": sum(not m.ok for m in ms),
           "invalid_reasons": sorted({r for m in ms if not m.ok for r in m.invalid_reasons}),
           "stall_max_s": max((m.stall_s for m in ms if m.ok), default=None), "steady_sample_t": steady.get("t"),
           "sample_schema_problems": sorted({p for m in samples for p in m["schema_problems"]}),
           "idle_problems": idle_problems(idle), "baseline_problems": baseline_problems(baseline, idle) if baseline is not None else [],
           "memory_drift": drift, "missing_memory": [k for k, v in drift.items() if v is None]}
    out["passed"] = (len(ms) > 0 and out["invalid"] == 0 and (out["stall_max_s"] or 0) <= 2.0 and not out["idle_problems"]
                     and not out["baseline_problems"] and not out["sample_schema_problems"] and not out["missing_memory"]
                     and bool(in_flight) and out["in_flight_min"] == concurrency and all(abs(v) <= 0.01 for v in drift.values()))
    return out, samples, idle


def identity_check(served: dict, config: str | None) -> dict:
    """The served engine is the declared configuration: the same config bytes, engine source and backend as the harness
    resolves (``bench.cli.served_identity_expected``)."""
    from bench.cli import resolve_engine_deploy, served_identity_expected
    from bench.fingerprint import repo_fingerprint

    want = served_identity_expected(config) if config else {}
    ok = bool(served) and bool(config) and all(served.get(k) == v for k, v in want.items())
    declared = resolve_engine_deploy(config) if config else {}
    return {"served": served, "declared": {**{k: declared.get(k) for k in ("config", "config_path")}, **want}, "repo": repo_fingerprint(ROOT),
            "matches": ok}


async def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    adapter = get_adapter("engine")
    if a.acceptance and (a.sustain_min < 30 or a.concurrency < 32):
        raise SystemExit("acceptance mode requires --sustain-min >= 30 at --concurrency >= 32")
    first = await metrics(a.url)
    served = await info(a.url)
    report: dict = {"url": a.url, "mode": "acceptance" if a.acceptance else "development", "identity": identity_check(served, a.config),
                    "gpu": gpu_check(served, a.gpu_uuid), "idle_before": first, "idle_before_problems": idle_problems(first)}
    if not (report["identity"]["matches"] and report["gpu"]["matches"]):  # before any load
        report["overall"] = False
        (out / "robustness.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: report[k] for k in ("identity", "gpu")}, indent=1))
        return report
    report["mixed"] = await scenario_mixed(a.url, adapter)
    baseline = report["baseline"] = await wait_idle(a.url)  # the warmed idle state every later drain must return to
    report["baseline_problems"] = idle_problems(baseline)
    tts = canonical_requests(rows, sets.warmup(8))
    report["cancel"] = await scenario_cancel(a.url, adapter, [adapter.build_payload(r) for r in tts], baseline)
    report["eos_boundary"] = await scenario_eos_boundary(a.url, adapter, canonical_requests(rows, sets.tuning(32)[:a.eos_n]),
                                                         int(first.get("first_chunk") or WINDOW))
    report["fallback"] = await scenario_fallback(a.url, adapter)
    report["overload"] = await scenario_overload(a.url, adapter, canonical_requests(rows, sets.tuning(32)[:a.overload]))
    await wait_idle(a.url)
    sus, samples, idle = await scenario_sustain(a.url, adapter, canonical_requests(rows, sets.eval(a.concurrency)), a.concurrency, a.sustain_min,
                                                report["gpu"]["served_uuid"], baseline)
    report["sustain"] = sus
    report["passed"] = {k: report[k]["passed"] for k in ("mixed", "cancel", "eos_boundary", "fallback", "overload", "sustain")}
    report["passed"].update(identity=report["identity"]["matches"], gpu=report["gpu"]["matches"], baseline=not report["baseline_problems"])
    report["overall"] = all(report["passed"].values())
    (out / "robustness.json").write_text(json.dumps({**report, "samples": samples, "idle_after": idle}, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "idle_before")}, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="the engine config the server was launched with (identity check)")
    ap.add_argument("--gpu-uuid", default=None, help="the GPU the server was launched on (CUDA device UUID, GPU-...)")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--sustain-min", type=float, default=30.0)
    ap.add_argument("--overload", type=int, default=128)
    ap.add_argument("--eos-n", type=int, default=192)
    ap.add_argument("--acceptance", action="store_true")
    a = ap.parse_args(argv)
    rep = asyncio.run(run(a))
    return 0 if rep["overall"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
