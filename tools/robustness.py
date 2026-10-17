"""AC-6 robustness and sustained-load checks against a running engine server.

    python tools/robustness.py --url http://127.0.0.1:PORT --gpu-uuid GPU-... --config CONFIG --out DIR [--sustain-min 30]
        [--concurrency 32] [--acceptance] [--no-long-prefill]
    python tools/robustness_replay.py DIR      # rebuilds the whole verdict from the saved artifacts

Every request that is expected to complete goes through the harness client and its validity rules
(``bench.client.send_request`` + ``bench.metrics.request_metrics``: terminal event, no error event, valid PCM with
consistent offsets, no gaps/overlaps/retransmissions).  An SSE error followed by [DONE] is a failure.

Scenarios (in this order):
* mixed: concurrent text-only, image, audio and image+audio requests (all with a reference voice);
* cancel: clients that disconnect mid-stream beside normal companions.  A cancelling client reads its response through
  the harness parser and assembler and disconnects once the first valid audio event has added unique samples; a
  non-200 status, an error event, an unusable or inconsistent audio prefix, no audio, or a stream that ended by itself
  counts against it (``cancel_problems``).  The server must count every disconnect (``cancelled_streams``, from the
  readings before and after) and drain back to the warmed idle baseline;
* eos-boundary: many replies whose codec frame counts (``usage.audio_frames``) must cover the Token2Wav window boundary
  and its neighbours ((frames - first_chunk) mod 25 in {24, 0, 1});
* fallback: a reference voice long enough that its prompt setup exceeds every captured DiT graph bucket: the worker's
  eager-call counter must be present before and after and rise, and the reply must be valid;
* overload: more requests than the admission limit at once: each is valid or an HTTP 503, and some are 503;
* sustain: continuous refill at C for N minutes (monotonic deadline): C persistent clients each start the next request
  of a cycling row sequence as soon as their previous one ends; every request valid, continuation stall <= 2 s.
  Continuity (from the requests' own times): every client's next request within ``REFILL_TOL_S`` of its previous
  completion, its first within ``REFILL_TOL_S`` of the start, busy to the deadline.  After the drain the counters equal
  the warmed baseline;
* long-prefill (after the sustain endpoint, so the memory series is untouched): ``LONG_N`` prompts of about 5400 tokens
  arrive at once beside ``LONG_STREAMS`` clients that keep speaking: every request valid, every stall <= 2 s, the
  engine's largest per-tick prefill work within its declared token budget (the long prompts chunked), and the drain
  back to the baseline.

Memory (predeclared): the reference is the first sample at or after ``REF_S`` = 60 s (monotonic) into the sustain
phase; every later sample and the drained endpoint must lie within +-1% of it for engine RSS, worker RSS, each feature
process's RSS and their total, GPU memory used on the served UUID, and the engine's and worker's CUDA reserved bytes.
After the drain the engine's and worker's CUDA allocated bytes must lie within ``ALLOC_TOL`` of the warmed baseline.

Counters: every reading must carry the declared schema: finite numbers; nonnegative integers for counts, capacities and
bytes; positive RSS; the worker's producer time within [-``STALE_S``, +``FUTURE_S``] of the reading's time; the
feature-process count the config declares; the feature PID inventory frozen at the baseline (no absent child, no
replacement) and the total equal to the per-PID sum.  The idle baseline is the drained state after ``mixed``; later
drains must equal it on every capacity counter (KV pages, cache slots, CNN rows, streams, queue, admission).

Evidence: ``requests.jsonl`` holds, in the order they happen, every request of every scenario (scenario, client, row,
cycle, seq, dispatch/completion times, the client trace without PCM, the request metrics), every cancelling client
(its parsed prefix, disconnect time and cancellation problems) and every raw observation the verdict uses
(``snapshot`` records: identity, GPU, workload and settings, phase boundaries, the warmed baseline and the frozen PID
inventory, the readings before/after cancel, fallback and long-prefill, every sustain sample, the endpoint, the sustain
geometry); ``wire.jsonl.gz`` the raw SSE chunks of every request and cancelling client with receipt times.  Verdicts
are deterministic functions of these records (``verdict_*``), called by the live run and by the replay alike.  The
report carries the checker's source hashes and the artifacts' hashes.

Before any load the report binds the server's identity (``/info``: config bytes, engine source hashes, backend,
interpreter) against the harness's resolution of ``--config`` and the served CUDA device (``/info`` ``device_uuid``)
against ``--gpu-uuid``; GPU memory is read for that UUID and the report records its SKU, UUID and driver.  A mismatch
ends the run before load.  ``--acceptance`` requires exactly 30 minutes at C >= 32 and every scenario; other runs are
labelled development.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import gc
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
from bench.client import ResponseConsumer, make_session, send_request  # noqa: E402
from bench.metrics import RequestMetrics, request_metrics  # noqa: E402
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
SAMPLE_S = 10.0  # sustain sampling interval
ENVELOPE = 0.01
ALLOC_TOL = 16 << 20  # bytes
STALL_MAX_S = 2.0  # AC-6: continuation stalls under sustained arrivals
REFILL_TOL_S = 0.05  # continuity: a sustain client's next request starts within this of its previous completion
BUSY_TOL_S = 0.5  # a client's last request ends at most this long before the deadline
ACCEPT_MINUTES, ACCEPT_C = 30.0, 32  # acceptance geometry
LONG_N, LONG_STREAMS, LONG_PROMPT_WORDS = 8, 24, 4500  # long-prefill scenario: 8 prompts of about 5400 tokens beside 24 speaking clients
SCENARIOS = ("mixed", "cancel", "eos_boundary", "fallback", "overload", "sustain", "long_prefill")
CHECKER_FILES = ("tools/robustness.py", "tools/robustness_replay.py", "bench/client.py", "bench/metrics.py", "bench/audio.py", "bench/sse.py")


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


async def sample(url: str, gpu_uuid: str | None) -> dict:
    """One reading: wall time (``t``, for producer freshness), monotonic time (``tm``), counters, GPU memory used."""
    t, tm, m = time.time(), time.perf_counter(), await metrics(url)
    # nvidia-smi in a thread: run on the event loop it would hold every client for its duration (~10-100 ms)
    return {"t": t, "tm": tm, **m, "gpu_memory_used_mib": await asyncio.to_thread(gpu_memory_used_mib, gpu_uuid)}


class LoopHealth:
    """The load generator's own health while clients run: the event loop's largest lag (a 5 ms tick that overslept) and
    the young-generation collections run in place of the automatic collector (``quiet_gc``)."""

    TICK_S = 0.005

    def __init__(self):
        self.lag_max_s, self.lag_max_tm, self.lags_over_10ms, self.gc_max_s, self.gc_calls = 0.0, None, 0, 0.0, 0
        self.events: list = []  # (tm, lag) of every lag above 10 ms (bounded)

    async def watch(self, stop_at: float) -> None:
        while time.perf_counter() < stop_at:
            t = time.perf_counter()
            await asyncio.sleep(self.TICK_S)
            lag = time.perf_counter() - t - self.TICK_S
            if lag > 0.010:
                self.lags_over_10ms += 1
                if len(self.events) < 20000:
                    self.events.append((t, lag))
            if lag > self.lag_max_s:
                self.lag_max_s, self.lag_max_tm = lag, t

    def collect_young(self) -> None:
        t = time.perf_counter()
        gc.collect(1)
        self.gc_max_s = max(self.gc_max_s, time.perf_counter() - t)
        self.gc_calls += 1

    def snapshot(self) -> dict:
        return {"loop_lag_max_s": self.lag_max_s, "loop_lag_max_tm": self.lag_max_tm, "loop_lags_over_10ms": self.lags_over_10ms,
                "gc_young_max_s": self.gc_max_s, "gc_young_calls": self.gc_calls, "gc_auto": gc.isenabled(), "lag_events": self.events}


@contextlib.contextmanager
def quiet_gc():
    """No automatic full collections while clients run: the checker holds every request record (tens of thousands), and
    one full collection over them takes ~80-110 ms (measured over the 3f467e3 acceptance records), during which no
    client can start its next request (the 3f467e3 run's single 67 ms refill gap).  The records are frozen out of the
    collector and young generations are collected explicitly (``LoopHealth.collect_young``); the collector is restored
    afterwards."""
    gc.collect()
    gc.freeze()
    gc.disable()
    try:
        yield
    finally:
        gc.enable()
        gc.unfreeze()


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


async def wait_idle(url: str, timeout_s: float = 120.0, expect: Expect | None = None, gpu_uuid: str | None = None) -> dict:
    """The drained state: nothing admitted, queued or live, the Token2Wav worker holding nothing; returned (as a timed
    reading, ``sample``) after a further 3 s (several worker counter periods), or at the timeout whatever it is."""
    t0 = time.perf_counter()
    while True:
        m = await metrics(url)
        if not idle_problems(m, expect=expect) or time.perf_counter() - t0 > timeout_s:
            await asyncio.sleep(3)
            return await sample(url, gpu_uuid)
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
    """Append-only artifacts, written by one thread in the order things happen: ``requests.jsonl`` (one record per
    request, cancelling client or raw observation -- every counter reading the verdict uses) and ``wire.jsonl.gz`` (each
    response's raw SSE chunks, receipt times relative to dispatch).  The methods return the record they persist: the
    live verdict is computed from exactly what the replay reads back."""

    def __init__(self, out: Path):
        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.q: queue.Queue = queue.Queue()
        self.n = 0
        self._n_lock = threading.Lock()  # records are built in worker threads (finish_stream)
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
                seq: int | None = None, kind: str = "request", extra: dict | None = None) -> dict:
        """Persist one finished (or, ``kind="cancel"``, deliberately abandoned) request; its PCM is not kept: the wire
        record holds the bytes it arrived as."""
        with self._n_lock:
            self.n += 1
        rec = {"record": kind, "scenario": scenario, "request_id": tr.request_id, "client": client, "row": row, "cycle": cycle, "seq": seq,
               "t_dispatch": tr.t_dispatch, "t_complete": tr.t_complete, "http_status": tr.http_status, "trace": tr.to_json(),
               "metrics": asdict(m), **(extra or {})}
        chunks = getattr(tr, "raw_chunks", None)
        wire = None if chunks is None else {"request_id": tr.request_id, "scenario": scenario, "t_dispatch": tr.t_dispatch, "t_complete": tr.t_complete,
                                            "chunks": [[t - tr.t_dispatch, base64.b64encode(b).decode()] for t, b in chunks]}
        self.q.put((rec, wire))
        return rec

    def snapshot(self, scenario: str, label: str, m: dict) -> dict:
        """A raw observation (a counter reading, the identity, the geometry, ...) the verdict is derived from."""
        with self._n_lock:
            self.n += 1
        rec = {"record": "snapshot", "scenario": scenario, "label": label, "m": m}
        self.q.put((rec, None))
        return rec

    def close(self) -> dict:
        self.q.put(None)
        self._t.join()
        self._req.close()
        self._wire.close()
        return {"records": self.n, "requests.jsonl": _sha_file(self.out / "requests.jsonl"), "wire.jsonl.gz": _sha_file(self.out / "wire.jsonl.gz")}


class _NoRecorder:
    """In-memory stand-in (scenario functions called without artifacts, e.g. in tests)."""

    def request(self, scenario, tr, m, client=None, row=None, cycle=None, seq=None, kind="request", extra=None) -> dict:
        return {"record": kind, "scenario": scenario, "request_id": tr.request_id, "client": client, "row": row, "cycle": cycle, "seq": seq,
                "t_dispatch": tr.t_dispatch, "t_complete": tr.t_complete, "http_status": tr.http_status, "trace": tr.to_json(),
                "metrics": asdict(m), **(extra or {})}

    def snapshot(self, scenario, label, m) -> dict:
        return {"record": "snapshot", "scenario": scenario, "label": label, "m": m}


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


def long_prompt_payloads(rows, n: int, words: int = LONG_PROMPT_WORDS) -> list[dict]:
    """``n`` requests whose prompt is a passage of about ``words`` words (Seed-TTS texts in a fixed order; at the
    default the shortest is 5422 prompt tokens as served), a short spoken reply in the default voice."""
    texts = [r.target_text for r in sorted(rows, key=lambda r: r.row_id)]
    out = []
    for i in range(n):
        passage, k = [], i
        while sum(len(t.split()) for t in passage) < words:
            passage.append(texts[k % len(texts)])
            k += n
        msg = "Here is a passage:\n\n" + " ".join(passage) + "\n\nIn one sentence, what is the passage about?"
        out.append({**media_payloads(4)[3], "messages": [{"role": "user", "content": [{"type": "text", "text": msg}]}], "seed": 1000 + i})
    return out


async def harness_stream(session, url: str, adapter, payload: dict, rid: str, timeout_s: float = 180.0, rec=None,
                         scenario: str = "", **ids):
    """One request through the production client and validity rules; returns (metrics, trace, record) -- the record
    is persisted when ``rec`` is a ``Recorder``."""
    tr = await send_request(session, url, adapter, SimpleNamespace(request_id=rid), timeout_s=timeout_s, payload=payload,
                            record_raw=isinstance(rec, Recorder))
    return await finish_stream(tr, rec, scenario, **ids)


async def finish_stream(tr, rec, scenario: str, **ids):
    """The validity metrics of a finished stream and its persisted record, off the event loop: computed on the loop they
    hold every client (the audio assembly and checks of one response take milliseconds, and completions coincide)."""
    def run():
        m = request_metrics(tr)
        return m, tr, (rec or _NoRecorder()).request(scenario, tr, m, **ids)

    return await asyncio.to_thread(run)


CANCEL_BAD_FLAGS = ("gap", "overlap", "num_samples_mismatch", "sample_rate", "multiple_terminal_waveforms", "malformed_wav", "missing_offsets")


def cancel_problems(tr) -> list[str]:
    """Why an abandoned stream does not count as "cancelled after valid audio": a non-200 status, an error event or a
    client error, an unusable or inconsistent audio prefix (the harness assembler's flags), no positive unique samples,
    or a stream that ended by itself before the client disconnected (not a mid-stream cancellation)."""
    out = []
    if tr.http_status != 200:
        out.append(f"http_{tr.http_status}")
    if tr.error:
        out.append("error")
    out += sorted({f"audio_{p}" for f in tr.audio.flags for p in CANCEL_BAD_FLAGS if f.startswith(p)})
    if tr.audio.retransmissions:
        out.append("retransmission")
    if not tr.audio.unique_samples:
        out.append("no_audio_before_disconnect")
    if tr.terminal:
        out.append("ended_before_disconnect")
    return out


async def cancel_stream(url: str, adapter, payload: dict, rid: str = "", rec=None, timeout_s: float = 180.0) -> dict:
    """A client that disconnects as soon as the first valid audio event has added unique samples (or at the first error
    or end of stream): its bytes go through the harness parser and assembler and are persisted with the disconnect
    time.  Returns the cancellation record."""
    consumer = ResponseConsumer(adapter, rid, time.perf_counter(), streaming=True)
    tr, raw = consumer.trace, []
    t_disc = None
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=timeout_s)) as s:
            tr.t_dispatch = time.perf_counter()
            async with s.post(url.rstrip("/") + adapter.endpoint, json=payload) as r:
                tr.http_status = r.status
                if r.status != 200:
                    tr.error = f"HTTP {r.status}: {(await r.text())[:300]}"
                else:
                    async for chunk in r.content.iter_any():
                        t = time.perf_counter()
                        raw.append((t, bytes(chunk)))
                        consumer.feed(t, chunk)
                        if tr.audio.unique_samples or tr.error or tr.audio.flags or consumer.done:
                            break
                t_disc = time.perf_counter()  # leaving the context closes the connection: the server sees the disconnect
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        tr.error = tr.error or f"{type(e).__name__}: {e}"
    t_disc = t_disc or time.perf_counter()
    consumer.finish(t_disc)
    tr.raw_chunks = raw  # type: ignore[attr-defined]
    return (rec or _NoRecorder()).request("cancel", tr, request_metrics(tr), kind="cancel",
                                          extra={"t_disconnect": t_disc, "unique_samples": tr.audio.unique_samples, "cancel_problems": cancel_problems(tr)})


# --- verdicts: deterministic functions of the persisted records (the live run and the replay call the same ones) ------
def _metrics_of(recs: list[dict]) -> list:
    return [RequestMetrics(**r["metrics"]) for r in recs]


def _summ(ms) -> dict:
    bad = [m for m in ms if not m.ok]
    return {"n": len(ms), "ok": len(ms) - len(bad), "invalid_reasons": sorted({r for m in bad for r in m.invalid_reasons})}


def _delta(after: dict, before: dict, key: str):
    a, b = (after or {}).get(key), (before or {}).get(key)
    return a - b if _count(a) and _count(b) else None


def verdict_mixed(recs: list[dict]) -> dict:
    out = _summ(_metrics_of(recs))
    out["passed"] = out["ok"] == out["n"] > 0
    return out


def verdict_cancel(companions: list[dict], cancels: list[dict], sent: int, before: dict, after: dict, baseline: dict | None, expect: Expect | None) -> dict:
    observed = _delta(after, before, "cancelled_streams")
    probs = sorted({p for c in cancels for p in c["cancel_problems"]})
    out = {"companions": _summ(_metrics_of(companions)), "cancelled_sent": sent, "cancel_records": len(cancels),
           "cancelled_with_audio": sum(not c["cancel_problems"] for c in cancels), "cancel_problems": probs, "cancelled_observed": observed,
           "idle_problems": idle_problems(after, now=after.get("t"), expect=expect),
           "baseline_problems": baseline_problems(baseline, after) if baseline is not None else []}
    out["passed"] = (out["companions"]["ok"] == out["companions"]["n"] > 0 and len(cancels) == sent > 0 and out["cancelled_with_audio"] == sent
                     and observed is not None and observed >= sent and not out["idle_problems"] and not out["baseline_problems"])
    return out


def boundary_coverage(frames: list, first_chunk: int) -> tuple[list, list]:
    residues = sorted({(f - first_chunk) % WINDOW for f in frames if isinstance(f, int)})
    need = {WINDOW - 1, 0, 1}
    return sorted(need & set(residues)), sorted({f for f in frames if isinstance(f, int) and (f - first_chunk) % WINDOW in need})[:20]


def verdict_eos_boundary(recs: list[dict], first_chunk: int) -> dict:
    frames = [(r["trace"].get("usage") or {}).get("audio_frames") for r in recs]
    covered, counts = boundary_coverage(frames, first_chunk)
    out = {**_summ(_metrics_of(recs)), "first_chunk": first_chunk, "frames_reported": sum(isinstance(f, int) for f in frames),
           "boundary_residues_covered": covered, "boundary_frame_counts": counts}
    out["passed"] = out["ok"] == out["n"] > 0 and len(covered) == 3
    return out


def verdict_fallback(recs: list[dict], before: dict, after: dict) -> dict:
    """Both readings must carry the eager-call counter (a missing one is not zero) and it must rise."""
    rose = _delta(after, before, "t2w_dit_eager_calls")
    out = {**_summ(_metrics_of(recs)), "dit_eager_calls_before": (before or {}).get("t2w_dit_eager_calls"),
           "dit_eager_calls_after": (after or {}).get("t2w_dit_eager_calls"), "dit_eager_calls_added": rose}
    out["passed"] = out["n"] == out["ok"] == 1 and rose is not None and rose > 0
    return out


def verdict_overload(recs: list[dict]) -> dict:
    ms = _metrics_of(recs)
    statuses = sorted({r["http_status"] for r in recs if r["http_status"] is not None})
    ok = sum(m.ok for m in ms)
    rejected = sum(r["http_status"] == 503 for r in recs)
    other = [m.request_id for m, r in zip(ms, recs) if not m.ok and r["http_status"] != 503]
    out = {"n": len(recs), "ok": ok, "rejected_503": rejected, "statuses": statuses, "other_failures": other[:10]}
    out["passed"] = not other and rejected > 0 and ok > 0
    return out


def refill_gaps(done: list, stop_at: float, t_start: float | None = None) -> dict:
    """Per client: the largest gap between one request's completion and its next dispatch, its first dispatch after
    the start, and whether it stayed busy up to the deadline (its last request started before the deadline and ended
    at most ``BUSY_TOL_S`` before it)."""
    by: dict = {}
    for c, _, t0, t1 in done:
        by.setdefault(c, []).append((t0, t1))
    gaps, busy, first = [], [], []
    for c, xs in by.items():
        xs.sort()
        gaps += [b[0] - a[1] for a, b in zip(xs, xs[1:]) if a[1] is not None]
        busy.append(xs[-1][0] < stop_at and (xs[-1][1] or 0) >= stop_at - BUSY_TOL_S)  # (the deadline check follows a completion at once)
        if t_start is not None:
            first.append(xs[0][0] - t_start)
    out = {"clients": len(by), "max_refill_gap_s": max(gaps, default=None), "clients_busy_to_deadline": sum(busy)}
    if t_start is not None:
        out["max_first_dispatch_s"] = max(first, default=None)
    return out


def in_flight_timeline(intervals: list, t0: float, t1: float) -> dict:
    """From the requests' [dispatch, completion] intervals: the time-weighted mean and the minimum number in flight over
    [t0, t1] (the minimum counts every refill instant, so it is C - 1 whenever a client refills at all)."""
    ev = sorted([(a, 1) for a, b in intervals] + [(b, -1) for a, b in intervals if b is not None])
    n, last, area, low = 0, t0, 0.0, None
    for t, d in ev:
        if t > t0:
            tt = min(t, t1)
            if tt > last:
                area += n * (tt - last)
                low = n if low is None else min(low, n)
                last = tt
        n += d
        if t >= t1:
            break
    if last < t1:
        area += n * (t1 - last)
        low = n if low is None else min(low, n)
    return {"mean": area / (t1 - t0) if t1 > t0 else None, "min": low}


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


def sequence_problems(recs: list[dict], row_ids: list) -> list[str]:
    """The sustain workload is the declared row sequence: request ``seq`` i is row i mod n of cycle i div n, and every
    seq from 0 to the request count appears once."""
    out = []
    if not row_ids:
        return ["no declared row sequence"]
    bad = [r["request_id"] for r in recs if r.get("seq") is None or r["row"] != row_ids[r["seq"] % len(row_ids)] or r["cycle"] != r["seq"] // len(row_ids)]
    if bad:
        out.append(f"{len(bad)} records outside the declared row sequence (e.g. {bad[:3]})")
    if sorted(r.get("seq") for r in recs if r.get("seq") is not None) != list(range(len(recs))):
        out.append("the request sequence has gaps or duplicates")
    return out


def verdict_sustain(recs: list[dict], samples: list[dict], endpoint: dict, geo: dict, baseline: dict | None, expect: Expect | None) -> dict:
    """``geo``: t_start, deadline, t_load_end, concurrency, minutes, row_ids, ref_s, requests_started.  Samples carry the
    reading's wall time ``t`` (producer freshness), monotonic ``tm`` and the client-side in-flight count."""
    ms = _metrics_of(recs)
    t_start, stop, conc = geo["t_start"], geo["deadline"], geo["concurrency"]
    done = [(r["client"], m, r["t_dispatch"], r["t_complete"]) for r, m in zip(recs, ms)]
    gaps = refill_gaps(done, stop, t_start)
    tl = in_flight_timeline([(r["t_dispatch"], r["t_complete"]) for r in recs], t_start + 1.0, stop - 1.0)
    ref = next((s for s in samples if s["tm"] - t_start >= geo["ref_s"]), None)
    later = ([s for s in samples if s["tm"] > ref["tm"]] + [endpoint]) if ref is not None else []
    env = memory_envelope(ref, later) if ref is not None else {}
    in_flight = [s.get("in_flight") for s in samples if s["tm"] < stop - 1.0]
    schema = sorted({p for s in samples + [endpoint] for p in schema_problems(s, now=s.get("t"), expect=expect, sampled=True)})
    out = {"minutes": geo["minutes"], "concurrency": conc, "t_start": t_start, "deadline": stop, "t_load_end": geo["t_load_end"],
           "requests": len(ms), "requests_started": geo["requests_started"], "row_cycles": len(ms) / max(1, len(geo["row_ids"])),
           "sequence_problems": sequence_problems(recs, geo["row_ids"]),
           "in_flight_min": min(in_flight, default=None), "in_flight_max": max(in_flight, default=None),
           "in_flight_mean_from_requests": tl["mean"], **gaps,
           "invalid": sum(not m.ok for m in ms), "invalid_reasons": sorted({x for m in ms if not m.ok for x in m.invalid_reasons}),
           "stall_max_s": max((m.stall_s for m in ms if m.ok), default=None), "samples": len(samples),
           "reference_sample_tm": ref["tm"] if ref else None, "reference_elapsed_s": ref["tm"] - t_start if ref else None,
           "sample_schema_problems": schema, "idle_problems": idle_problems(endpoint, now=endpoint.get("t"), expect=expect),
           "baseline_problems": baseline_problems(baseline, endpoint) if baseline is not None else [],
           "memory_envelope": env, "missing_memory": sorted(k for k, v in env.items() if v["max_dev"] is None) if ref else ["no reference sample"],
           "loop_health": geo.get("loop_health")}  # reported, not part of the rule
    out["continuity_problems"] = [p for p, bad in (
        (f"a client waited {gaps['max_refill_gap_s']} s between requests (> {REFILL_TOL_S})", (gaps["max_refill_gap_s"] or 0) > REFILL_TOL_S),
        (f"a client's first request started {gaps.get('max_first_dispatch_s')} s after the start (> {REFILL_TOL_S})",
         (gaps.get("max_first_dispatch_s") or 0) > REFILL_TOL_S),
        (f"{gaps['clients_busy_to_deadline']} of {conc} clients busy to the deadline", gaps["clients_busy_to_deadline"] != conc),
        (f"{gaps['clients']} clients, {conc} declared", gaps["clients"] != conc),
        (f"the sampled in-flight count fell to {out['in_flight_min']}", not in_flight or out["in_flight_min"] != conc)) if bad]
    out["passed"] = (len(ms) > 0 and out["invalid"] == 0 and (out["stall_max_s"] or 0) <= STALL_MAX_S and not out["idle_problems"]
                     and not out["baseline_problems"] and not out["sample_schema_problems"] and not out["missing_memory"]
                     and not out["sequence_problems"] and not out["continuity_problems"] and all(v["max_dev"] <= ENVELOPE for v in env.values()))
    return out


def verdict_long_prefill(longs: list[dict], streams: list[dict], before: dict, after: dict, baseline: dict | None, expect: Expect | None) -> dict:
    """Long prompts beside continuing speech: every request valid, every stall <= ``STALL_MAX_S``, the engine's largest
    per-tick prefill work within its declared token budget, the long prompts chunked (they exceed the budget) and the
    drain back to the warmed baseline."""
    ml, mc = _metrics_of(longs), _metrics_of(streams)
    budget, tick_max = (after or {}).get("prefill_token_budget"), (after or {}).get("prefill_tokens_tick_max")
    prompts = [(r["trace"].get("usage") or {}).get("prompt_tokens") for r in longs]
    out = {"long": _summ(ml), "streams": _summ(mc), "stall_max_s": max((m.stall_s for m in ml + mc if m.ok), default=None),
           "prefill_token_budget": budget, "prefill_tokens_tick_max": tick_max, "long_prompt_tokens_min": min((p for p in prompts if _count(p)), default=None),
           "prefill_chunked_ticks_added": _delta(after, before, "prefill_chunked_ticks"),
           "idle_problems": idle_problems(after, now=after.get("t"), expect=expect),
           "baseline_problems": baseline_problems(baseline, after) if baseline is not None else []}
    out["passed"] = (out["long"]["n"] == out["long"]["ok"] > 0 and out["streams"]["n"] == out["streams"]["ok"] > 0
                     and (out["stall_max_s"] or 0) <= STALL_MAX_S and _count(budget) and _count(tick_max) and tick_max <= budget
                     and _count(out["long_prompt_tokens_min"]) and out["long_prompt_tokens_min"] > budget
                     and (out["prefill_chunked_ticks_added"] or 0) > 0 and not out["idle_problems"] and not out["baseline_problems"])
    return out


def identity_matches(served: dict, expected: dict) -> bool:
    return bool(served) and bool(expected) and all(served.get(k) == v for k, v in expected.items())


def gpu_problems(served_uuid: str | None, smi: dict | None, declared_uuid: str | None) -> list[str]:
    problems = []
    if not served_uuid:
        problems.append("the server does not publish its device_uuid")
    elif smi is None or smi.get("uuid") != served_uuid:
        problems.append(f"nvidia-smi does not resolve the served device {served_uuid}")
    if not declared_uuid:
        problems.append("no --gpu-uuid declared")
    elif served_uuid != declared_uuid:
        problems.append(f"served device {served_uuid} != declared {declared_uuid}")
    return problems


def verdict_overall(rep: dict) -> dict:
    """Every scenario's verdict, identity, GPU and baseline; an acceptance run must hold every scenario and the fixed
    geometry (``ACCEPT_MINUTES`` at C >= ``ACCEPT_C``, deadline exactly that long after the start)."""
    passed = {k: bool((rep.get(k) or {}).get("passed")) for k in SCENARIOS if k in rep}
    passed.update(identity=bool((rep.get("identity") or {}).get("matches")), gpu=bool((rep.get("gpu") or {}).get("matches")),
                  baseline=not rep.get("baseline_problems", ["no baseline"]))
    problems = []
    if rep.get("mode") == "acceptance":
        problems += [f"acceptance run without {k}" for k in SCENARIOS if k not in rep]
        sus = rep.get("sustain") or {}
        if not (sus.get("minutes") == ACCEPT_MINUTES and (sus.get("concurrency") or 0) >= ACCEPT_C
                and _num(sus.get("deadline")) and _num(sus.get("t_start")) and abs(sus["deadline"] - sus["t_start"] - 60 * ACCEPT_MINUTES) < 1e-6):
            problems.append(f"acceptance geometry: {sus.get('minutes')} min at C={sus.get('concurrency')}")
    return {"passed": passed, "problems": problems, "overall": all(passed.values()) and bool(passed) and not problems}


# --- scenarios: collect the raw observations, then call the verdict ---------------------------------------------------
async def scenario_mixed(url, adapter, n: int = 16, rec=None) -> dict:
    async with make_session(connections=4 * n) as s:
        res = await asyncio.gather(*[harness_stream(s, url, adapter, p, f"mixed-{i}", rec=rec, scenario="mixed") for i, p in enumerate(media_payloads(n))])
    return verdict_mixed([r for _, _, r in res])


async def scenario_cancel(url, adapter, payloads: list[dict], baseline: dict | None = None, rec=None, expect: Expect | None = None,
                          gpu_uuid: str | None = None) -> dict:
    rec = rec or _NoRecorder()
    before = rec.snapshot("cancel", "before", await sample(url, gpu_uuid))["m"]
    half = len(payloads) // 2
    async with make_session(connections=4 * len(payloads)) as s:
        comp = asyncio.gather(*[harness_stream(s, url, adapter, p, f"companion-{i}", rec=rec, scenario="cancel") for i, p in enumerate(payloads[half:])])
        canc = asyncio.gather(*[cancel_stream(url, adapter, p, f"cancel-{i}", rec) for i, p in enumerate(payloads[:half])])
        cres, kres = await asyncio.gather(comp, canc)
    after = rec.snapshot("cancel", "after", await wait_idle(url, expect=expect, gpu_uuid=gpu_uuid))["m"]
    return verdict_cancel([r for _, _, r in cres], list(kres), half, before, after, baseline, expect)


async def scenario_eos_boundary(url, adapter, reqs, first_chunk: int, concurrency: int = 32, rec=None) -> dict:
    payloads = [adapter.build_payload(r) for r in reqs]
    for p in payloads:
        p["stream_options"] = {"include_usage": True}
    sem = asyncio.Semaphore(concurrency)
    async with make_session(connections=4 * concurrency) as s:
        async def one(i, p):
            async with sem:
                return await harness_stream(s, url, adapter, p, reqs[i].request_id, rec=rec, scenario="eos_boundary", row=reqs[i].request_id)
        res = await asyncio.gather(*[one(i, p) for i, p in enumerate(payloads)])
    return verdict_eos_boundary([r for _, _, r in res], first_chunk)


async def scenario_fallback(url, adapter, rec=None, gpu_uuid: str | None = None) -> dict:
    rec = rec or _NoRecorder()
    before = rec.snapshot("fallback", "before", await sample(url, gpu_uuid))["m"]
    p = media_payloads(4, ref_wav=long_reference_wav())[3]  # text-only reply in a 48 s reference voice
    async with make_session(connections=4) as s:
        _, _, r = await harness_stream(s, url, adapter, p, "fallback-long-voice", rec=rec, scenario="fallback")
    await asyncio.sleep(2)
    after = rec.snapshot("fallback", "after", await sample(url, gpu_uuid))["m"]
    return verdict_fallback([r], before, after)


async def scenario_overload(url, adapter, reqs, rec=None) -> dict:
    async with make_session(connections=4 * len(reqs)) as s:
        res = await asyncio.gather(*[harness_stream(s, url, adapter, adapter.build_payload(r), r.request_id, rec=rec, scenario="overload", row=r.request_id)
                                     for r in reqs])
    return verdict_overload([r for _, _, r in res])


async def continuous_load(url, adapter, reqs, concurrency: int, stop_at: float, state: dict, timeout_s: float = 180.0,
                          rec=None, scenario: str = "sustain", prefix: str = "", payloads: list | None = None) -> list:
    """``concurrency`` persistent clients; each starts the next request of the cycling row sequence as soon as its previous
    one ends, until ``stop_at`` (``time.perf_counter``).  ``state`` carries the in-flight count (read by the sampler) and
    the requests started.  A finished stream's metrics and record are completed in the background (``finish_stream``)
    after the client has started its next request.  Returns the persisted request records (traces are not kept)."""
    payloads = payloads or [adapter.build_payload(r) for r in reqs]
    finishing: list = []
    state.update(in_flight=0, started=0)
    async with make_session(connections=4 * concurrency) as s:
        async def client(c: int):
            while time.perf_counter() < stop_at:
                i = state["started"]
                state["started"] += 1  # no await between the previous request's end and this start: in flight stays C
                state["in_flight"] += 1
                try:
                    r = reqs[i % len(reqs)]
                    tr = await send_request(s, url, adapter, SimpleNamespace(request_id=f"{prefix}{r.request_id}#{i // len(reqs)}"), timeout_s=timeout_s,
                                            payload=payloads[i % len(reqs)], record_raw=isinstance(rec, Recorder))
                    finishing.append(asyncio.ensure_future(finish_stream(tr, rec, scenario, client=c, row=r.request_id, cycle=i // len(reqs), seq=i)))
                finally:
                    state["in_flight"] -= 1
        await asyncio.gather(*(client(c) for c in range(concurrency)))
    return [record for _, _, record in await asyncio.gather(*finishing)]


async def scenario_sustain(url, adapter, reqs, concurrency: int, minutes: float, gpu_uuid: str | None, baseline: dict | None = None,
                           interval_s: float = 10.0, ref_s: float = REF_S, expect: Expect | None = None, rec=None) -> tuple:
    rec = rec or _NoRecorder()
    samples, state = [], {}
    payloads = [adapter.build_payload(r) for r in reqs]  # before the start: the clients' first requests leave at once
    health = LoopHealth()

    async def sampler():
        while time.perf_counter() < stop:
            m = await sample(url, gpu_uuid)
            m["in_flight"] = state.get("in_flight")
            samples.append(rec.snapshot("sustain", "sample", m)["m"])
            health.collect_young()
            await asyncio.sleep(interval_s)

    with quiet_gc():  # entered (its full collection done) before the start
        t_start = time.perf_counter()
        stop = t_start + 60 * minutes
        smp, watch = asyncio.create_task(sampler()), asyncio.create_task(health.watch(stop))
        recs = await continuous_load(url, adapter, reqs, concurrency, stop, state, rec=rec, payloads=payloads)
        await smp
        await watch
        loop_health = health.snapshot()
    t_load_end = time.perf_counter()
    endpoint = rec.snapshot("sustain", "endpoint", await wait_idle(url, expect=expect, gpu_uuid=gpu_uuid))["m"]
    geo = rec.snapshot("sustain", "geometry", {"t_start": t_start, "deadline": stop, "t_load_end": t_load_end, "concurrency": concurrency,
                                               "minutes": minutes, "row_ids": [r.request_id for r in reqs], "ref_s": ref_s,
                                               "requests_started": state.get("started", 0), "loop_health": loop_health})["m"]
    return verdict_sustain(recs, samples, endpoint, geo, baseline, expect), samples, endpoint


async def scenario_long_prefill(url, adapter, rows, stream_reqs, n_long: int = 8, streams: int = 24, rec=None, baseline: dict | None = None,
                                expect: Expect | None = None, gpu_uuid: str | None = None, lead_s: float = 3.0) -> dict:
    """``streams`` clients keep speaking (continuous refill) while ``n_long`` long-prompt requests arrive at once after
    ``lead_s``; the continuing clients stop once every long request has finished."""
    rec = rec or _NoRecorder()
    before = rec.snapshot("long_prefill", "before", await sample(url, gpu_uuid))["m"]
    stop_box = {"t": time.perf_counter() + 600.0}

    async def streams_load():
        payloads = [adapter.build_payload(r) for r in stream_reqs]
        out, i = [], [0]
        async with make_session(connections=4 * streams) as s:
            async def client(c):
                while time.perf_counter() < stop_box["t"]:
                    k = i[0]
                    i[0] += 1
                    r = stream_reqs[k % len(stream_reqs)]
                    tr = await send_request(s, url, adapter, SimpleNamespace(request_id=f"lp-{r.request_id}#{k}"), payload=payloads[k % len(stream_reqs)],
                                            record_raw=isinstance(rec, Recorder))
                    out.append(asyncio.ensure_future(finish_stream(tr, rec, "long_prefill_streams", client=c, row=r.request_id, seq=k)))
            await asyncio.gather(*(client(c) for c in range(streams)))
        return [record for _, _, record in await asyncio.gather(*out)]

    longs = long_prompt_payloads(rows, n_long)
    with quiet_gc():
        load = asyncio.create_task(streams_load())
        await asyncio.sleep(lead_s)
        async with make_session(connections=4 * n_long) as s:
            res = await asyncio.gather(*[harness_stream(s, url, adapter, p, f"long-prefill-{i}", rec=rec, scenario="long_prefill")
                                         for i, p in enumerate(longs)])
        stop_box["t"] = time.perf_counter()  # the continuing clients finish the request they are on and stop
        srecs = await load
    after = rec.snapshot("long_prefill", "after", await wait_idle(url, expect=expect, gpu_uuid=gpu_uuid))["m"]
    return verdict_long_prefill([r for _, _, r in res], srecs, before, after, baseline, expect)


def identity_check(served: dict, config: str | None) -> dict:
    """The served engine is the declared configuration: the same config bytes, engine source, backend and interpreter as
    the harness resolves (``bench.cli.served_identity_expected``)."""
    from bench.cli import resolve_engine_deploy, served_identity_expected
    from bench.fingerprint import repo_fingerprint

    want = served_identity_expected(config) if config else {}
    declared = resolve_engine_deploy(config) if config else {}
    return {"served": served, "expected": want, "declared": {k: declared.get(k) for k in ("config", "config_path")}, "repo": repo_fingerprint(ROOT),
            "matches": bool(config) and identity_matches(served, want)}


def gpu_check(served: dict, expected_uuid: str | None) -> dict:
    """The served CUDA device is the declared GPU, and nvidia-smi resolves it (memory readings use this UUID)."""
    uuid = (served or {}).get("device_uuid")
    g = gpu_query(uuid)
    problems = gpu_problems(uuid, g, expected_uuid)
    return {"served_uuid": uuid, "declared_uuid": expected_uuid, "nvidia_smi": g, "problems": problems, "matches": not problems}


def workload_declaration(sets, rows, concurrency: int, eos_n: int, overload: int) -> dict:
    ids = {"cancel": sets.warmup(8), "eos_boundary": sets.tuning(32)[:eos_n], "overload": sets.tuning(32)[:overload], "sustain": sets.eval(concurrency),
           "long_prefill_streams": sets.tuning(32)[:LONG_STREAMS]}
    return {"mixed": {"n": 16, "seeds": list(range(16))}, **{k: {"row_ids": list(v)} for k, v in ids.items()},
            "long_prefill": {"n": LONG_N, "words": LONG_PROMPT_WORDS, "streams": LONG_STREAMS}}


async def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    adapter = get_adapter("engine")
    if a.acceptance and (a.sustain_min != ACCEPT_MINUTES or a.concurrency < ACCEPT_C):
        raise SystemExit(f"acceptance mode requires --sustain-min {ACCEPT_MINUTES} at --concurrency >= {ACCEPT_C}")
    expect = expect_from_config(a.config)
    first = await metrics(a.url)
    served = await info(a.url)
    report: dict = {"url": a.url, "mode": "acceptance" if a.acceptance else "development", "checker": checker_identity(),
                    "workload": workload_declaration(sets, rows, a.concurrency, a.eos_n, a.overload), "identity": identity_check(served, a.config),
                    "gpu": gpu_check(served, a.gpu_uuid), "idle_before": first, "idle_before_problems": idle_problems(first, expect=expect), "phases": {}}
    if not (report["identity"]["matches"] and report["gpu"]["matches"]):  # before any load
        report.update(verdict_overall(report))
        (out / "robustness.json").write_text(json.dumps(report, indent=1))
        print(json.dumps({k: report[k] for k in ("identity", "gpu")}, indent=1))
        return report
    rec = Recorder(out)
    uuid = report["gpu"]["served_uuid"]
    rec.snapshot("run", "identity", {k: report["identity"][k] for k in ("served", "expected", "declared")})
    rec.snapshot("run", "gpu", {k: report["gpu"][k] for k in ("served_uuid", "declared_uuid", "nvidia_smi")})
    rec.snapshot("run", "workload", {"mode": report["mode"], "workload": report["workload"], "config": a.config,
                                     "settings": {k: getattr(a, k) for k in ("concurrency", "sustain_min", "overload", "eos_n")}})

    async def phase(name, coro):
        t0 = time.perf_counter()
        res = await coro
        report["phases"][name] = {"start": t0, "end": time.perf_counter()}
        rec.snapshot("run", "phase", {"name": name, **report["phases"][name]})
        return res

    report["mixed"] = await phase("mixed", scenario_mixed(a.url, adapter, rec=rec))
    baseline = report["baseline"] = rec.snapshot("run", "baseline", await wait_idle(a.url, expect=expect, gpu_uuid=uuid))["m"]
    if isinstance(baseline.get("feature_rss_by_pid"), dict) and expect.feature_processes:
        expect.pids = frozenset(baseline["feature_rss_by_pid"])  # the PID inventory from here on
    rec.snapshot("run", "expect", {"feature_processes": expect.feature_processes, "pids": sorted(expect.pids) if expect.pids else None})
    report["feature_pids"] = sorted(expect.pids) if expect.pids else []
    report["baseline_problems"] = idle_problems(baseline, now=baseline.get("t"), expect=expect)
    tts = canonical_requests(rows, sets.warmup(8))
    report["cancel"] = await phase("cancel", scenario_cancel(a.url, adapter, [adapter.build_payload(r) for r in tts], baseline, rec=rec, expect=expect,
                                                             gpu_uuid=uuid))
    report["eos_boundary"] = await phase("eos_boundary", scenario_eos_boundary(a.url, adapter, canonical_requests(rows, sets.tuning(32)[:a.eos_n]),
                                                                               int(first.get("first_chunk") or WINDOW), rec=rec))
    rec.snapshot("eos_boundary", "first_chunk", {"first_chunk": int(first.get("first_chunk") or WINDOW)})
    report["fallback"] = await phase("fallback", scenario_fallback(a.url, adapter, rec=rec, gpu_uuid=uuid))
    report["overload"] = await phase("overload", scenario_overload(a.url, adapter, canonical_requests(rows, sets.tuning(32)[:a.overload]), rec=rec))
    await wait_idle(a.url, expect=expect)
    sus, samples, idle = await phase("sustain", scenario_sustain(a.url, adapter, canonical_requests(rows, sets.eval(a.concurrency)), a.concurrency,
                                                                 a.sustain_min, uuid, baseline, interval_s=SAMPLE_S, ref_s=REF_S, expect=expect, rec=rec))
    report["sustain"] = sus
    if not a.no_long_prefill:  # after the sustain endpoint: the memory reference and envelope are untouched
        report["long_prefill"] = await phase("long_prefill", scenario_long_prefill(a.url, adapter, rows, canonical_requests(rows, sets.tuning(32)[:LONG_STREAMS]),
                                                                                   LONG_N, LONG_STREAMS, rec=rec, baseline=baseline, expect=expect, gpu_uuid=uuid))
    report["artifacts"] = rec.close()
    report.update(verdict_overall(report))
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
    ap.add_argument("--no-long-prefill", action="store_true", help="development runs only: skip the long-prefill scenario")
    a = ap.parse_args(argv)
    if a.acceptance and a.no_long_prefill:
        raise SystemExit("acceptance runs include the long-prefill scenario")
    rep = asyncio.run(run(a))
    return 0 if rep["overall"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
