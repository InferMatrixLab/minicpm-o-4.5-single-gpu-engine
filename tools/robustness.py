"""AC-6 robustness and sustained-load checks against a running engine server.

    python tools/robustness.py --url http://127.0.0.1:PORT --gpu-uuid GPU-... --config CONFIG --out DIR [--sustain-min 30]
        [--concurrency 32] [--acceptance]
    python tools/robustness_replay.py DIR      # rebuilds the verdict from the saved per-request artifacts

Every request that is expected to complete goes through the harness client and its validity rules
(``bench.client.send_request`` + ``bench.metrics.request_metrics``: terminal event, no error event, valid PCM with
consistent offsets, no gaps/overlaps/retransmissions).  An SSE error followed by [DONE] is a failure.

Scenarios (in this order):
* mixed: concurrent text-only, image, audio and image+audio requests (all with a reference voice);
* cancel: clients that disconnect after their first audio chunk beside normal companions.  Every cancelling client must
  have received audio before it disconnected, the server must count every disconnect (``cancelled_streams``) and drain
  back to the warmed idle baseline;
* eos-boundary: many replies whose codec frame counts (``usage.audio_frames``) must cover the Token2Wav window boundary
  and its neighbours ((frames - first_chunk) mod 25 in {24, 0, 1});
* fallback: a reference voice long enough that its prompt setup exceeds every captured DiT graph bucket: the worker's
  eager-call counter must rise and the reply must be valid;
* overload: more requests than the admission limit at once: each is valid or an HTTP 503, and some are 503;
* sustain: continuous refill at C for N minutes (monotonic deadline): C persistent clients each start the next request
  of a cycling row sequence as soon as their previous one ends; every request valid, continuation stall <= 2 s; every
  client busy until the deadline (its refill gap is recorded).  After the drain the counters equal the warmed baseline.

Memory (predeclared): the reference is the first sample at or after ``REF_S`` = 60 s (monotonic) into the sustain
phase; every later sample and the drained endpoint must lie within +-1% of it for engine RSS, worker RSS, each feature
process's RSS and their total, GPU memory used on the served UUID, and the engine's and worker's CUDA reserved bytes.
After the drain the engine's and worker's CUDA allocated bytes must lie within ``ALLOC_TOL`` of the warmed baseline.

Counters: every reading must carry the declared schema: finite numbers; nonnegative integers for counts, capacities and
bytes; positive RSS; the worker's producer time within [-``STALE_S``, +``FUTURE_S``] of now; the feature-process count
the config declares; the feature PID inventory frozen at the baseline (no absent child, no replacement) and the total
equal to the per-PID sum.  The idle baseline is the drained state after ``mixed``; later drains must equal it on every
capacity counter (KV pages, cache slots, CNN rows, streams, queue, admission).

Evidence: every request of every scenario is appended, as it finishes, to ``requests.jsonl`` (scenario, client, row,
cycle, dispatch/completion times, the client trace without PCM and the request metrics) and its raw SSE chunks with
receipt times to ``wire.jsonl.gz``; cancelling clients' observations go to ``requests.jsonl`` too.  The report carries
the workload declaration, phase boundaries (monotonic), the checker's source hashes and the artifacts' hashes.

Before any load the report binds the server's identity (``/info``: config bytes, engine source hashes, backend,
interpreter) against the harness's resolution of ``--config`` and the served CUDA device (``/info`` ``device_uuid``)
against ``--gpu-uuid``; GPU memory is read for that UUID and the report records its SKU, UUID and driver.  A mismatch
ends the run before load.  ``--acceptance`` requires 30 minutes at C >= 32; other runs are labelled development.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import gzip
import hashlib
import io
import json
import math
import queue
import subprocess
import sys
import threading
import time
import wave
from dataclasses import asdict, dataclass
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
ALLOC_KEYS = ("gpu_memory_allocated_bytes", "t2w_gpu_memory_allocated_bytes")  # live CUDA tensors: back to the baseline
RESERVED_KEYS = ("gpu_memory_reserved_bytes", "t2w_gpu_memory_reserved_bytes")  # CUDA caching-allocator capacity: in the envelope
INT_KEYS = BASELINE_KEYS + ALLOC_KEYS + RESERVED_KEYS + ("cancelled_streams", "t2w_dit_eager_calls", "feature_processes")
POSITIVE_KEYS = ("rss_kib", "t2w_rss_kib")
REQUIRED = INT_KEYS + POSITIVE_KEYS + ("t2w_stats_time",)
MEMORY_KEYS = ("rss_kib", "t2w_rss_kib", "feature_rss_kib", "gpu_memory_used_mib") + RESERVED_KEYS
STALE_S = 5.0  # the worker publishes its counters once a second
FUTURE_S = 1.0  # a producer time this far ahead of the checker's clock is invalid
REF_S = 60.0  # the memory reference: the first sustain sample at or after this many seconds
ENVELOPE = 0.01
ALLOC_TOL = 16 << 20  # bytes
CHECKER_FILES = ("tools/robustness.py", "tools/robustness_replay.py", "bench/client.py", "bench/metrics.py", "bench/audio.py")


@dataclass
class Expect:
    """What the declared configuration implies for every reading: the feature-process count, and (once the baseline is
    taken) the feature PID inventory."""

    feature_processes: int = 0
    pids: frozenset | None = None


def expect_from_config(config: str | None) -> Expect:
    if not config:
        return Expect()
    from bench.cli import engine_config_path

    cfg = json.loads(Path(engine_config_path(config)).read_text())
    return Expect(feature_processes=int(cfg.get("feature_processes", 0)) if cfg.get("t2w_process") else 0)


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
    """One reading: wall time (``t``, for producer freshness), monotonic time (``tm``), counters, GPU memory used."""
    return {"t": time.time(), "tm": time.perf_counter(), **await metrics(url), "gpu_memory_used_mib": gpu_memory_used_mib(gpu_uuid)}


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _count(v) -> bool:
    return _num(v) and float(v).is_integer() and v >= 0


def schema_problems(m: dict, now: float | None = None, expect: Expect | None = None, sampled: bool = False) -> list[str]:
    """The declared schema of one reading (``sampled``: it also carries the GPU memory reading)."""
    now = time.time() if now is None else now
    declared = expect is not None  # without a declaration only the reading's own consistency is checked
    expect = expect or Expect()
    out = []
    for k in REQUIRED:
        v = m.get(k)
        if k not in m:
            out.append(f"{k} missing")
        elif not _num(v):
            out.append(f"{k}={v!r} is not a finite number")
        elif k in INT_KEYS and not _count(v):
            out.append(f"{k}={v!r} is not a nonnegative integer")
        elif k in POSITIVE_KEYS and v <= 0:
            out.append(f"{k}={v!r} is not positive")
    t = m.get("t2w_stats_time")
    if _num(t) and not (now - STALE_S <= t <= now + FUTURE_S):
        out.append(f"worker counters are {now - t:.1f} s old" if t < now else f"worker counter time is {t - now:.1f} s in the future")
    if sampled and not (_num(m.get("gpu_memory_used_mib")) and m["gpu_memory_used_mib"] > 0):
        out.append(f"gpu_memory_used_mib={m.get('gpu_memory_used_mib')!r} is not a positive number")
    n, by_pid = m.get("feature_processes"), m.get("feature_rss_by_pid")
    if declared and _count(n) and n != expect.feature_processes:
        out.append(f"{n} feature processes, the config declares {expect.feature_processes}")
    if _count(n) and n > 0:
        if not isinstance(by_pid, dict) or len(by_pid) != n:
            out.append(f"feature RSS reported for {len(by_pid) if isinstance(by_pid, dict) else 0} of {n} processes")
        else:
            bad = [p for p, v in by_pid.items() if not (_num(v) and v > 0)]
            out += [f"feature process {p} RSS unreadable" for p in bad]
            if expect.pids is not None and frozenset(by_pid) != expect.pids:
                out.append(f"feature PIDs changed: {sorted(by_pid)} != {sorted(expect.pids)}")
            tot = m.get("feature_rss_kib")
            if not (_num(tot) and tot > 0):
                out.append("feature_rss_kib is not a positive number")
            elif not bad and tot != sum(by_pid.values()):
                out.append(f"feature_rss_kib {tot} != the per-PID sum {sum(by_pid.values())}")
    return out


def idle_problems(m: dict, now: float | None = None, expect: Expect | None = None) -> list[str]:
    return schema_problems(m, now, expect) + [f"{k}={m[k]}" for k in IDLE_KEYS if _num(m.get(k)) and m[k]]


def baseline_problems(base: dict, m: dict) -> list[str]:
    """A drained state differs from the warmed idle baseline: a capacity counter (a leaked row, slot, page or stream)
    differs, or live CUDA allocations did not return (within ``ALLOC_TOL``)."""
    if not base:
        return ["no idle baseline"]
    out = [f"{k}={m.get(k)!r} != baseline {base.get(k)!r}" for k in BASELINE_KEYS if m.get(k) != base.get(k)]
    for k in ALLOC_KEYS:
        a, b = base.get(k), m.get(k)
        if not (_num(a) and _num(b)):
            out.append(f"{k} unreadable (baseline {a!r}, now {b!r})")
        elif abs(b - a) > ALLOC_TOL:
            out.append(f"{k} {b - a:+d} bytes against the baseline (tolerance {ALLOC_TOL})")
    return out


async def wait_idle(url: str, timeout_s: float = 120.0, expect: Expect | None = None) -> dict:
    """The drained state: nothing admitted, queued or live, the Token2Wav worker holding nothing; returned after a further
    3 s (several worker counter periods), or at the timeout whatever it is."""
    t0 = time.perf_counter()
    while True:
        m = await metrics(url)
        if not idle_problems(m, expect=expect) or time.perf_counter() - t0 > timeout_s:
            await asyncio.sleep(3)
            return await metrics(url)
        await asyncio.sleep(1)


# --- per-request evidence ---------------------------------------------------------------------------------------------
def _sha_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def checker_identity() -> dict:
    return {f: _sha_file(ROOT / f) for f in CHECKER_FILES}


class Recorder:
    """Append-only per-request artifacts, written by one thread as requests finish: ``requests.jsonl`` (one record per
    request or cancelling client) and ``wire.jsonl.gz`` (its raw SSE chunks, receipt times relative to dispatch)."""

    def __init__(self, out: Path):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue = queue.Queue()
        self.n = 0
        self._req = (self.out / "requests.jsonl").open("w")
        self._wire = gzip.open(self.out / "wire.jsonl.gz", "wt", compresslevel=1)
        self._t = threading.Thread(target=self._write, daemon=True, name="robustness-recorder")
        self._t.start()

    def _write(self) -> None:
        while True:
            item = self.q.get()
            if item is None:
                return
            rec, wire = item
            self._req.write(json.dumps(rec) + "\n")
            if wire is not None:
                self._wire.write(json.dumps(wire) + "\n")

    def request(self, scenario: str, tr, m, client: int | None = None, row: str | None = None, cycle: int | None = None,
                seq: int | None = None) -> None:
        """Persist one finished request (its PCM is not kept: the wire record holds the bytes it arrived as)."""
        self.n += 1
        rec = {"record": "request", "scenario": scenario, "request_id": tr.request_id, "client": client, "row": row, "cycle": cycle, "seq": seq, "t_dispatch": tr.t_dispatch,
               "t_complete": tr.t_complete, "http_status": tr.http_status, "trace": tr.to_json(), "metrics": asdict(m)}
        chunks = getattr(tr, "raw_chunks", None)
        wire = None if chunks is None else {"request_id": tr.request_id, "scenario": scenario, "t_dispatch": tr.t_dispatch, "t_complete": tr.t_complete,
                                            "chunks": [[t - tr.t_dispatch, base64.b64encode(b).decode()] for t, b in chunks]}
        self.q.put((rec, wire))

    def observation(self, scenario: str, obs: dict) -> None:
        self.n += 1
        self.q.put(({**obs, "scenario": scenario, "record": "observation"}, None))

    def close(self) -> dict:
        self.q.put(None)
        self._t.join()
        self._req.close()
        self._wire.close()
        return {"records": self.n, "requests.jsonl": _sha_file(self.out / "requests.jsonl"), "wire.jsonl.gz": _sha_file(self.out / "wire.jsonl.gz")}


# --- requests ---------------------------------------------------------------------------------------------------------
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


async def harness_stream(session, url: str, adapter, payload: dict, rid: str, timeout_s: float = 180.0, rec: Recorder | None = None,
                         scenario: str = "", **ids):
    """One request through the production client and validity rules; persisted when ``rec`` is given; returns
    (metrics, trace)."""
    tr = await send_request(session, url, adapter, SimpleNamespace(request_id=rid), timeout_s=timeout_s, payload=payload, record_raw=rec is not None)
    m = request_metrics(tr)
    if rec is not None:
        rec.request(scenario, tr, m, **ids)
    return m, tr


async def cancel_stream(url: str, payload: dict, rid: str = "", rec: Recorder | None = None) -> dict:
    """Disconnect after the first audio chunk; records whether audio arrived before the disconnect."""
    audio, status, t0 = 0, None, time.perf_counter()
    t_audio = None
    try:
        async with aiohttp.ClientSession() as s:
            async with s.post(url + "/v1/chat/completions", json=payload) as r:
                status = r.status
                if r.status == 200:
                    async for line in r.content:
                        if b'"modality": "audio"' in line:
                            audio += 1
                            t_audio = time.perf_counter()
                            break
    except (aiohttp.ClientError, asyncio.TimeoutError):
        pass
    obs = {"request_id": rid, "kind": "cancel_client", "status": status, "audio_events": audio, "t_dispatch": t0, "t_first_audio": t_audio,
           "t_disconnect": time.perf_counter()}
    if rec is not None:
        rec.observation("cancel", obs)
    return obs


def _summ(ms) -> dict:
    bad = [m for m in ms if not m.ok]
    return {"n": len(ms), "ok": len(ms) - len(bad), "invalid_reasons": sorted({r for m in bad for r in m.invalid_reasons})}


async def scenario_mixed(url, adapter, n: int = 16, rec: Recorder | None = None) -> dict:
    async with make_session(connections=4 * n) as s:
        res = await asyncio.gather(*[harness_stream(s, url, adapter, p, f"mixed-{i}", rec=rec, scenario="mixed") for i, p in enumerate(media_payloads(n))])
    out = _summ([m for m, _ in res])
    out["passed"] = out["ok"] == out["n"] > 0
    return out


async def scenario_cancel(url, adapter, payloads: list[dict], baseline: dict | None = None, rec: Recorder | None = None,
                          expect: Expect | None = None) -> dict:
    before = await metrics(url)
    half = len(payloads) // 2
    async with make_session(connections=4 * len(payloads)) as s:
        comp = asyncio.gather(*[harness_stream(s, url, adapter, p, f"companion-{i}", rec=rec, scenario="cancel") for i, p in enumerate(payloads[half:])])
        canc = asyncio.gather(*[cancel_stream(url, p, f"cancel-{i}", rec) for i, p in enumerate(payloads[:half])])
        cres, kres = await asyncio.gather(comp, canc)
    after = await wait_idle(url, expect=expect)
    a, b = after.get("cancelled_streams"), before.get("cancelled_streams")
    observed = a - b if _num(a) and _num(b) else None
    out = {"companions": _summ([m for m, _ in cres]), "cancelled_sent": half, "cancelled_with_audio": sum(k["audio_events"] > 0 for k in kres),
           "cancelled_observed": observed, "idle_problems": idle_problems(after, expect=expect),
           "baseline_problems": baseline_problems(baseline, after) if baseline is not None else []}
    out["passed"] = (out["companions"]["ok"] == out["companions"]["n"] and out["cancelled_with_audio"] == half and observed is not None
                     and observed >= half and not out["idle_problems"] and not out["baseline_problems"])
    return out


def boundary_coverage(frames: list, first_chunk: int) -> tuple[list, list]:
    residues = sorted({(f - first_chunk) % WINDOW for f in frames if isinstance(f, int)})
    need = {WINDOW - 1, 0, 1}
    return sorted(need & set(residues)), sorted({f for f in frames if isinstance(f, int) and (f - first_chunk) % WINDOW in need})[:20]


async def scenario_eos_boundary(url, adapter, reqs, first_chunk: int, concurrency: int = 32, rec: Recorder | None = None) -> dict:
    payloads = [adapter.build_payload(r) for r in reqs]
    for p in payloads:
        p["stream_options"] = {"include_usage": True}
    sem = asyncio.Semaphore(concurrency)
    async with make_session(connections=4 * concurrency) as s:
        async def one(i, p):
            async with sem:
                return await harness_stream(s, url, adapter, p, reqs[i].request_id, rec=rec, scenario="eos_boundary", row=reqs[i].request_id)
        res = await asyncio.gather(*[one(i, p) for i, p in enumerate(payloads)])
    frames = [tr.usage.get("audio_frames") for _, tr in res]
    covered, counts = boundary_coverage(frames, first_chunk)
    out = {**_summ([m for m, _ in res]), "first_chunk": first_chunk, "frames_reported": sum(isinstance(f, int) for f in frames),
           "boundary_residues_covered": covered, "boundary_frame_counts": counts}
    out["passed"] = out["ok"] == out["n"] > 0 and len(covered) == 3
    return out


async def scenario_fallback(url, adapter, rec: Recorder | None = None) -> dict:
    before = await metrics(url)
    p = media_payloads(4, ref_wav=long_reference_wav())[3]  # text-only reply in a 48 s reference voice
    async with make_session(connections=4) as s:
        m, _ = await harness_stream(s, url, adapter, p, "fallback-long-voice", rec=rec, scenario="fallback")
    await asyncio.sleep(2)
    after = await metrics(url)
    rose = (after.get("t2w_dit_eager_calls") or 0) - (before.get("t2w_dit_eager_calls") or 0)
    out = {**_summ([m]), "dit_eager_calls_added": rose, "counter_present": _num(after.get("t2w_dit_eager_calls"))}
    out["passed"] = out["ok"] == 1 and out["counter_present"] and rose > 0
    return out


async def scenario_overload(url, adapter, reqs, rec: Recorder | None = None) -> dict:
    async with make_session(connections=4 * len(reqs)) as s:
        res = await asyncio.gather(*[harness_stream(s, url, adapter, adapter.build_payload(r), r.request_id, rec=rec, scenario="overload", row=r.request_id)
                                     for r in reqs])
    statuses = sorted({tr.http_status for _, tr in res if tr.http_status is not None})
    ok = sum(m.ok for m, _ in res)
    rejected = sum(tr.http_status == 503 for _, tr in res)
    other = [m.request_id for m, tr in res if not m.ok and tr.http_status != 503]
    out = {"n": len(res), "ok": ok, "rejected_503": rejected, "statuses": statuses, "other_failures": other[:10]}
    out["passed"] = not other and rejected > 0 and ok > 0
    return out


# --- sustained load ---------------------------------------------------------------------------------------------------
async def continuous_load(url, adapter, reqs, concurrency: int, stop_at: float, state: dict, timeout_s: float = 180.0,
                          rec: Recorder | None = None) -> list:
    """``concurrency`` persistent clients; each starts the next request of the cycling row sequence as soon as its previous
    one ends, until ``stop_at`` (``time.perf_counter``).  ``state`` carries the in-flight count (read by the sampler) and
    the requests started.  Returns, per request, (client, metrics, t_dispatch, t_complete); traces are persisted to
    ``rec`` as they finish and not kept."""
    payloads = [adapter.build_payload(r) for r in reqs]
    done: list = []
    state.update(in_flight=0, started=0)
    async with make_session(connections=4 * concurrency) as s:
        async def client(c: int):
            while time.perf_counter() < stop_at:
                i = state["started"]
                state["started"] += 1  # no await between the previous request's end and this start: in flight stays C
                state["in_flight"] += 1
                try:
                    r = reqs[i % len(reqs)]
                    m, tr = await harness_stream(s, url, adapter, payloads[i % len(reqs)], f"{r.request_id}#{i // len(reqs)}", timeout_s=timeout_s,
                                                 rec=rec, scenario="sustain", client=c, row=r.request_id, cycle=i // len(reqs), seq=i)
                    done.append((c, m, tr.t_dispatch, tr.t_complete))
                finally:
                    state["in_flight"] -= 1
        await asyncio.gather(*(client(c) for c in range(concurrency)))
    return done


def refill_gaps(done: list, stop_at: float) -> dict:
    """Per client: the largest gap between one request's completion and its next dispatch, and whether it stayed busy up
    to the deadline (its last request started before the deadline and ended at most 0.5 s before it)."""
    by: dict = {}
    for c, _, t0, t1 in done:
        by.setdefault(c, []).append((t0, t1))
    gaps, busy = [], []
    for c, xs in by.items():
        xs.sort()
        gaps += [b[0] - a[1] for a, b in zip(xs, xs[1:]) if a[1] is not None]
        busy.append(xs[-1][0] < stop_at and (xs[-1][1] or 0) >= stop_at - 0.5)  # (the deadline check follows a completion at once)
    return {"clients": len(by), "max_refill_gap_s": max(gaps, default=None), "clients_busy_to_deadline": sum(busy)}


def memory_readings(m: dict) -> dict:
    """The memory readings of one sample: the engine-wide keys and each feature process's RSS by PID."""
    out = {k: m.get(k) for k in MEMORY_KEYS}
    for pid, v in (m.get("feature_rss_by_pid") or {}).items():
        out[f"feature_rss_kib[{pid}]"] = v
    return out


def memory_envelope(ref: dict, later: list[dict]) -> dict:
    """Per memory reading: the reference value, the largest relative deviation over ``later`` (every later sample and the
    drained endpoint) and where it occurred; a reading missing or unreadable anywhere is reported as ``None``."""
    r = memory_readings(ref)
    out = {}
    for k, a in r.items():
        devs, worst = [], None
        for s in later:
            b = memory_readings(s).get(k)
            if not (_num(a) and _num(b) and a > 0):
                devs = None
                break
            d = abs(b - a) / a
            if not devs or d > max(devs):
                worst = s.get("tm")
            devs.append(d)
        out[k] = {"ref": a, "max_dev": max(devs) if devs else None, "at": worst}
    for s in later:
        for k in memory_readings(s):
            if k not in out:
                out[k] = {"ref": None, "max_dev": None, "at": s.get("tm")}  # a reading with no reference
    return out


async def scenario_sustain(url, adapter, reqs, concurrency: int, minutes: float, gpu_uuid: str | None, baseline: dict | None = None,
                           interval_s: float = 10.0, ref_s: float = REF_S, expect: Expect | None = None, rec: Recorder | None = None) -> tuple:
    samples, state = [], {}
    t_start = time.perf_counter()
    stop = t_start + 60 * minutes

    async def sampler():
        while time.perf_counter() < stop:
            m = await sample(url, gpu_uuid)
            m["in_flight"] = state.get("in_flight")
            m["schema_problems"] = schema_problems(m, expect=expect, sampled=True)
            samples.append(m)
            await asyncio.sleep(interval_s)

    smp = asyncio.create_task(sampler())
    done = await continuous_load(url, adapter, reqs, concurrency, stop, state, rec=rec)
    await smp
    t_load_end = time.perf_counter()
    ms = [m for _, m, _, _ in done]
    await wait_idle(url, expect=expect)
    idle = await sample(url, gpu_uuid)
    idle["schema_problems"] = schema_problems(idle, expect=expect, sampled=True)
    ref = next((s for s in samples if s["tm"] - t_start >= ref_s), None)
    later = ([s for s in samples if s["tm"] > ref["tm"]] + [idle]) if ref is not None else []
    env = memory_envelope(ref, later) if ref is not None else {}
    in_flight = [m["in_flight"] for m in samples if m["tm"] < stop - 1.0]
    out = {"minutes": minutes, "concurrency": concurrency, "t_start": t_start, "deadline": stop, "t_load_end": t_load_end,
           "requests": len(ms), "requests_started": state.get("started", 0), "row_cycles": state.get("started", 0) / max(1, len(reqs)),
           "in_flight_min": min(in_flight, default=None), "in_flight_max": max(in_flight, default=None), **refill_gaps(done, stop),
           "invalid": sum(not m.ok for m in ms), "invalid_reasons": sorted({r for m in ms if not m.ok for r in m.invalid_reasons}),
           "stall_max_s": max((m.stall_s for m in ms if m.ok), default=None), "reference_sample_tm": ref["tm"] if ref else None,
           "reference_elapsed_s": ref["tm"] - t_start if ref else None,
           "sample_schema_problems": sorted({p for m in samples + [idle] for p in m["schema_problems"]}),
           "idle_problems": idle_problems(idle, expect=expect), "baseline_problems": baseline_problems(baseline, idle) if baseline is not None else [],
           "memory_envelope": env, "missing_memory": sorted(k for k, v in env.items() if v["max_dev"] is None) if ref else ["no reference sample"]}
    out["passed"] = (len(ms) > 0 and out["invalid"] == 0 and (out["stall_max_s"] or 0) <= 2.0 and not out["idle_problems"]
                     and not out["baseline_problems"] and not out["sample_schema_problems"] and not out["missing_memory"]
                     and bool(in_flight) and out["in_flight_min"] == concurrency and out["clients"] == concurrency
                     and out["clients_busy_to_deadline"] == concurrency and all(v["max_dev"] <= ENVELOPE for v in env.values()))
    return out, samples, idle


def identity_check(served: dict, config: str | None) -> dict:
    """The served engine is the declared configuration: the same config bytes, engine source, backend and interpreter as
    the harness resolves (``bench.cli.served_identity_expected``)."""
    from bench.cli import resolve_engine_deploy, served_identity_expected
    from bench.fingerprint import repo_fingerprint

    want = served_identity_expected(config) if config else {}
    ok = bool(served) and bool(config) and all(served.get(k) == v for k, v in want.items())
    declared = resolve_engine_deploy(config) if config else {}
    return {"served": served, "declared": {**{k: declared.get(k) for k in ("config", "config_path")}, **want}, "repo": repo_fingerprint(ROOT),
            "matches": ok}


def workload_declaration(sets, rows, concurrency: int, eos_n: int, overload: int) -> dict:
    ids = {"cancel": sets.warmup(8), "eos_boundary": sets.tuning(32)[:eos_n], "overload": sets.tuning(32)[:overload], "sustain": sets.eval(concurrency)}
    return {"mixed": {"n": 16, "seeds": list(range(16))}, **{k: {"row_ids": list(v)} for k, v in ids.items()}}


async def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    adapter = get_adapter("engine")
    if a.acceptance and (a.sustain_min < 30 or a.concurrency < 32):
        raise SystemExit("acceptance mode requires --sustain-min >= 30 at --concurrency >= 32")
    expect = expect_from_config(a.config)
    first = await metrics(a.url)
    served = await info(a.url)
    report: dict = {"url": a.url, "mode": "acceptance" if a.acceptance else "development", "checker": checker_identity(),
                    "workload": workload_declaration(sets, rows, a.concurrency, a.eos_n, a.overload), "identity": identity_check(served, a.config),
                    "gpu": gpu_check(served, a.gpu_uuid), "idle_before": first, "idle_before_problems": idle_problems(first, expect=expect), "phases": {}}
    if not (report["identity"]["matches"] and report["gpu"]["matches"]):  # before any load
        report["overall"] = False
        (out / "robustness.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: report[k] for k in ("identity", "gpu")}, indent=1))
        return report
    rec = Recorder(out)

    async def phase(name, coro):
        t0 = time.perf_counter()
        res = await coro
        report["phases"][name] = {"start": t0, "end": time.perf_counter()}
        return res

    report["mixed"] = await phase("mixed", scenario_mixed(a.url, adapter, rec=rec))
    baseline = report["baseline"] = await wait_idle(a.url, expect=expect)  # the warmed idle state every later drain must return to
    if isinstance(baseline.get("feature_rss_by_pid"), dict) and expect.feature_processes:
        expect.pids = frozenset(baseline["feature_rss_by_pid"])  # the PID inventory from here on
    report["feature_pids"] = sorted(expect.pids) if expect.pids else []
    report["baseline_problems"] = idle_problems(baseline, expect=expect)
    tts = canonical_requests(rows, sets.warmup(8))
    report["cancel"] = await phase("cancel", scenario_cancel(a.url, adapter, [adapter.build_payload(r) for r in tts], baseline, rec=rec, expect=expect))
    report["eos_boundary"] = await phase("eos_boundary", scenario_eos_boundary(a.url, adapter, canonical_requests(rows, sets.tuning(32)[:a.eos_n]),
                                                                               int(first.get("first_chunk") or WINDOW), rec=rec))
    report["fallback"] = await phase("fallback", scenario_fallback(a.url, adapter, rec=rec))
    report["overload"] = await phase("overload", scenario_overload(a.url, adapter, canonical_requests(rows, sets.tuning(32)[:a.overload]), rec=rec))
    await wait_idle(a.url, expect=expect)
    sus, samples, idle = await phase("sustain", scenario_sustain(a.url, adapter, canonical_requests(rows, sets.eval(a.concurrency)), a.concurrency,
                                                                 a.sustain_min, report["gpu"]["served_uuid"], baseline, expect=expect, rec=rec))
    report["sustain"] = sus
    report["artifacts"] = rec.close()
    report["passed"] = {k: report[k]["passed"] for k in ("mixed", "cancel", "eos_boundary", "fallback", "overload", "sustain")}
    report["passed"].update(identity=report["identity"]["matches"], gpu=report["gpu"]["matches"], baseline=not report["baseline_problems"])
    report["overall"] = all(report["passed"].values())
    (out / "robustness.json").write_text(json.dumps({**report, "samples": samples, "idle_after": idle}, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "idle_before", "workload", "baseline")}, indent=1))
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
