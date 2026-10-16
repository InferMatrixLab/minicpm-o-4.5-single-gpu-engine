"""AC-6 robustness and sustained-load checks against a running engine server.

    python tools/robustness.py --url http://127.0.0.1:PORT --gpu G --config CONFIG --out DIR [--sustain-min 30] [--concurrency 32]
        [--acceptance]

Every request that is expected to complete goes through the harness client and its validity rules
(``bench.client.send_request`` + ``bench.metrics.request_metrics``: terminal event, no error event, valid PCM with
consistent offsets, no gaps/overlaps/retransmissions).  An SSE error followed by [DONE] is a failure.

Scenarios:
* mixed: concurrent text-only, image, audio and image+audio requests (all with a reference voice);
* cancel: clients that disconnect after their first audio chunk beside normal companions.  The server must count every
  disconnect (``cancelled_streams``) and drain: no admitted/live request, no Thinker/Talker KV page, no Token2Wav request
  or cache slot left;
* eos-boundary: many replies whose codec frame counts (``usage.audio_frames``) must cover the Token2Wav window boundary
  and its neighbours ((frames - first_chunk) mod 25 in {24, 0, 1});
* fallback: a reference voice long enough that its prompt setup exceeds every captured DiT graph bucket: the worker's
  eager-call counter must rise and the reply must be valid;
* overload: more requests than the admission limit at once: each is valid or an HTTP 503, and some are 503;
* sustain: closed loop at C for N minutes on the evaluation rows; every request valid, continuation stall <= 2 s; after
  the drain the counters are idle and memory (engine RSS, Token2Wav worker RSS, feature-process RSS, GPU memory used on
  the server's GPU) is within 1% of the first steady-state sample.  A missing reading fails.

The report binds the server's identity (``/info``: config bytes and engine source hashes) and checks it against the
harness's resolution of ``--config`` (``bench.cli.resolve_engine_deploy``).  ``--acceptance`` requires 30 minutes at
C >= 32; other runs are labelled development.
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
from bench.protocol import closed_loop  # noqa: E402
from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets  # noqa: E402

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
WINDOW = 25  # codec frames per steady Token2Wav window (engine.token2wav_chunker.CHUNK)
IDLE_KEYS = ("admitted", "live", "thinker_kv_pages_used", "talker_kv_pages_used", "audio_live", "t2w_requests", "t2w_cache_slots_used")
MEMORY_KEYS = ("rss_kib", "t2w_rss_kib", "feature_rss_kib", "gpu_memory_used_mib")


async def metrics(url: str) -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.get(url + "/metrics") as r:
            return await r.json()


async def info(url: str) -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.get(url + "/info") as r:
            return await r.json() if r.status == 200 else {}


def gpu_memory_used_mib(gpu: int | None) -> float | None:
    if gpu is None:
        return None
    try:
        out = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=30).stdout.strip()
        return float(out) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


async def sample(url: str, gpu: int | None) -> dict:
    return {"t": time.time(), **await metrics(url), "gpu_memory_used_mib": gpu_memory_used_mib(gpu)}


async def wait_idle(url: str, timeout_s: float = 120.0) -> dict:
    """The drained state: nothing admitted or live, the Token2Wav worker holding nothing (its counters arrive once a second)."""
    t0 = time.time()
    while True:
        m = await metrics(url)
        if all(not m.get(k) for k in IDLE_KEYS if k in m) or time.time() - t0 > timeout_s:
            await asyncio.sleep(3)
            return await metrics(url)
        await asyncio.sleep(1)


def idle_problems(m: dict) -> list[str]:
    return [f"{k}={m[k]}" for k in IDLE_KEYS if m.get(k)]


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


async def scenario_cancel(url, adapter, payloads: list[dict]) -> dict:
    before = await metrics(url)
    half = len(payloads) // 2
    async with make_session(connections=4 * len(payloads)) as s:
        comp = asyncio.gather(*[harness_stream(s, url, adapter, p, f"companion-{i}") for i, p in enumerate(payloads[half:])])
        canc = asyncio.gather(*[cancel_stream(url, p) for p in payloads[:half]])
        cres, kres = await asyncio.gather(comp, canc)
    after = await wait_idle(url)
    observed = after.get("cancelled_streams", 0) - before.get("cancelled_streams", 0)
    out = {"companions": _summ([m for m, _ in cres]), "cancelled_sent": half, "cancelled_with_audio": sum(k["audio_events"] > 0 for k in kres),
           "cancelled_observed": observed, "idle_problems": idle_problems(after)}
    out["passed"] = (out["companions"]["ok"] == out["companions"]["n"] and observed >= half and not out["idle_problems"])
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


async def scenario_sustain(url, adapter, reqs, concurrency: int, minutes: float, gpu: int | None) -> dict:
    samples, stop = [], time.time() + 60 * minutes

    async def sampler():
        while time.time() < stop:
            samples.append(await sample(url, gpu))
            await asyncio.sleep(10)

    async def load():
        traces = []
        async with make_session(connections=4 * concurrency) as s:
            while time.time() < stop:
                traces += await closed_loop(s, url, adapter, reqs, concurrency, 180.0)
        return traces

    smp = asyncio.create_task(sampler())
    traces = await load()
    await smp
    ms = [request_metrics(t) for t in traces]
    await wait_idle(url)
    idle = await sample(url, gpu)
    steady = samples[min(6, len(samples) - 1)] if samples else {}
    drift = {}
    for k in MEMORY_KEYS:
        a, b = steady.get(k), idle.get(k)
        drift[k] = (b - a) / a if isinstance(a, (int, float)) and isinstance(b, (int, float)) and a else None
    out = {"minutes": minutes, "concurrency": concurrency, "requests": len(ms), "invalid": sum(not m.ok for m in ms),
           "invalid_reasons": sorted({r for m in ms if not m.ok for r in m.invalid_reasons}),
           "stall_max_s": max((m.stall_s for m in ms if m.ok), default=None), "idle_problems": idle_problems(idle), "memory_drift": drift,
           "missing_memory": [k for k, v in drift.items() if v is None]}
    out["passed"] = (len(ms) > 0 and out["invalid"] == 0 and (out["stall_max_s"] or 0) <= 2.0 and not out["idle_problems"]
                     and not out["missing_memory"] and all(abs(v) <= 0.01 for v in drift.values() if v is not None))
    return out, samples, idle


def identity_check(served: dict, config: str | None) -> dict:
    """The served engine is the declared configuration: the same config bytes and engine source as the harness resolves."""
    from bench.cli import resolve_engine_deploy
    from bench.fingerprint import repo_fingerprint

    want = resolve_engine_deploy(config) if config else {}
    ok = bool(served) and bool(config) and served.get("config_sha256") == want.get("config_sha256") and \
        served.get("engine_tree_sha256") == want.get("engine_tree_sha256")
    return {"served": served, "declared": {k: want.get(k) for k in ("config", "config_path", "config_sha256", "engine_tree_sha256")},
            "repo": repo_fingerprint(ROOT), "matches": ok}


async def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    adapter = get_adapter("engine")
    first = await metrics(a.url)
    report: dict = {"url": a.url, "mode": "acceptance" if a.acceptance else "development", "identity": identity_check(await info(a.url), a.config),
                    "idle_before": first}
    if a.acceptance and (a.sustain_min < 30 or a.concurrency < 32):
        raise SystemExit("acceptance mode requires --sustain-min >= 30 at --concurrency >= 32")
    report["mixed"] = await scenario_mixed(a.url, adapter)
    tts = canonical_requests(rows, sets.warmup(8))
    report["cancel"] = await scenario_cancel(a.url, adapter, [adapter.build_payload(r) for r in tts])
    report["eos_boundary"] = await scenario_eos_boundary(a.url, adapter, canonical_requests(rows, sets.tuning(32)[:a.eos_n]),
                                                         int(first.get("first_chunk") or WINDOW))
    report["fallback"] = await scenario_fallback(a.url, adapter)
    report["overload"] = await scenario_overload(a.url, adapter, canonical_requests(rows, sets.tuning(32)[:a.overload]))
    await wait_idle(a.url)
    sus, samples, idle = await scenario_sustain(a.url, adapter, canonical_requests(rows, sets.eval(a.concurrency)), a.concurrency, a.sustain_min, a.gpu)
    report["sustain"] = sus
    report["passed"] = {k: report[k]["passed"] for k in ("mixed", "cancel", "eos_boundary", "fallback", "overload", "sustain")}
    report["passed"]["identity"] = report["identity"]["matches"]
    report["overall"] = all(report["passed"].values())
    (out / "robustness.json").write_text(json.dumps({**report, "samples": samples, "idle_after": idle}, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "idle_before")}, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None, help="the engine config the server was launched with (identity check)")
    ap.add_argument("--gpu", type=int, default=None, help="the server's GPU index (memory readings)")
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
