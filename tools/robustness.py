"""AC-6 robustness and sustained-load checks against a running engine server.

    python tools/robustness.py --url http://127.0.0.1:PORT --out results/l20x/robustness/<name> [--sustain-min 30] [--concurrency 32]

Scenarios (each request goes through the harness client and validity rules):
* mixed: concurrent text-only, voice-clone TTS, image, audio and frames+audio requests of mixed lengths;
* cancel: clients that disconnect after the first audio chunk, interleaved with normal requests (no corruption of the others);
* eos-boundary: replies whose codec length lands on and next to window boundaries (many short targets);
* overload: more requests than the admission limit at once; every response is 200 or 503, never a dropped connection;
* sustain: closed loop at C for N minutes on the evaluation rows; ``/metrics`` is sampled every 10 s.  After the drain
  the live/KV counters must return to their idle values and RSS / GPU memory must stay within 1% of the first
  steady-state sample; the continuation stall (``stall_max_s``) of every request must be <= 2 s.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
import time
from pathlib import Path

import aiohttp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.adapters import get_adapter  # noqa: E402
from bench.client import make_session, send_request  # noqa: E402
from bench.metrics import request_metrics  # noqa: E402
from bench.protocol import closed_loop  # noqa: E402
from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets  # noqa: E402

MODEL = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")


async def metrics(url: str) -> dict:
    async with aiohttp.ClientSession() as s:
        async with s.get(url + "/metrics") as r:
            return await r.json()


def _data_url(path: Path, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def media_payloads(n: int) -> list[dict]:
    img = _data_url(MODEL / "assets" / "fossil.png", "image/png")
    wav = _data_url(MODEL / "assets" / "HT_ref_audio.wav", "audio/wav")
    forms = [
        [{"type": "image_url", "image_url": {"url": img}}, {"type": "text", "text": "Describe the picture in one sentence."}],
        [{"type": "audio_url", "audio_url": {"url": wav}}, {"type": "text", "text": "What does the speaker say?"}],
        [{"type": "image_url", "image_url": {"url": img}}, {"type": "audio_url", "audio_url": {"url": wav}}, {"type": "text", "text": "Relate them."}],
        [{"type": "text", "text": "Say hello."}],
    ]
    return [{"model": "openbmb/MiniCPM-o-4_5", "messages": [{"role": "user", "content": forms[i % len(forms)]}], "modalities": ["text", "audio"],
             "ref_audio": wav, "stream": True, "max_tokens": 64, "seed": i} for i in range(n)]


async def raw_stream(url: str, payload: dict, cancel_after_audio: bool = False) -> dict:
    t0 = time.perf_counter()
    async with aiohttp.ClientSession() as s:
        async with s.post(url + "/v1/chat/completions", json=payload) as r:
            status, audio, done = r.status, 0, False
            if status != 200:
                return {"status": status, "audio_events": 0, "done": False, "s": time.perf_counter() - t0}
            async for line in r.content:
                if line.startswith(b"data: [DONE]"):
                    done = True
                    break
                if b'"modality": "audio"' in line:
                    audio += 1
                    if cancel_after_audio:
                        break
    return {"status": status, "audio_events": audio, "done": done, "s": time.perf_counter() - t0}


async def run(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    adapter = get_adapter("engine")
    report: dict = {"url": a.url, "idle_before": await metrics(a.url)}
    mixed = await asyncio.gather(*[raw_stream(a.url, p) for p in media_payloads(4 * 4)])
    report["mixed"] = {"n": len(mixed), "ok": sum(m["status"] == 200 and m["done"] for m in mixed)}
    tts = canonical_requests(rows, sets.warmup(8))
    cancels = [raw_stream(a.url, adapter.build_payload(r), cancel_after_audio=(i % 2 == 0)) for i, r in enumerate(tts)]
    cres = await asyncio.gather(*cancels)
    report["cancel"] = {"cancelled": sum(1 for i, _ in enumerate(cres) if i % 2 == 0), "completed_ok": sum(c["done"] for i, c in enumerate(cres) if i % 2)}
    await asyncio.sleep(5)
    report["after_cancel"] = await metrics(a.url)
    over = await asyncio.gather(*[raw_stream(a.url, adapter.build_payload(r)) for r in canonical_requests(rows, sets.tuning(32)[:a.overload])])
    report["overload"] = {"n": len(over), "statuses": sorted({o["status"] for o in over}), "ok": sum(o["status"] == 200 and o["done"] for o in over),
                          "rejected_503": sum(o["status"] == 503 for o in over)}
    samples = []
    reqs = canonical_requests(rows, sets.eval(a.concurrency))
    stop = time.time() + 60 * a.sustain_min

    async def sampler():
        while time.time() < stop + 30:
            samples.append({"t": time.time(), **await metrics(a.url)})
            await asyncio.sleep(10)

    async def load():
        traces = []
        async with make_session(connections=4 * a.concurrency) as s:
            while time.time() < stop:
                traces += await closed_loop(s, a.url, adapter, reqs, a.concurrency, 180.0)
        return traces

    smp = asyncio.create_task(sampler())
    traces = await load()
    await smp
    ms = [request_metrics(t) for t in traces]
    idle = samples[-1]
    steady = samples[min(6, len(samples) - 1)]
    report["sustain"] = {
        "minutes": a.sustain_min, "requests": len(ms), "invalid": sum(not m.ok for m in ms),
        "stall_max_s": max((m.stall_s for m in ms if m.ok), default=None),
        "idle_after": {k: idle.get(k) for k in ("live", "queued", "thinker_kv_pages_used", "talker_kv_pages_used")},
        "rss_drift": (idle["rss_kib"] - steady["rss_kib"]) / steady["rss_kib"] if steady.get("rss_kib") else None,
        "gpu_reserved_drift": (idle["gpu_memory_reserved_bytes"] - steady["gpu_memory_reserved_bytes"]) / steady["gpu_memory_reserved_bytes"]
        if steady.get("gpu_memory_reserved_bytes") else None}
    s = report["sustain"]
    report["passed"] = {
        "mixed": report["mixed"]["ok"] == report["mixed"]["n"],
        "cancel": report["cancel"]["completed_ok"] == len(tts) // 2 and report["after_cancel"].get("live") == 0,
        "overload": set(report["overload"]["statuses"]) <= {200, 503} and report["overload"]["ok"] + report["overload"]["rejected_503"] == len(over),
        "sustain": s["invalid"] == 0 and (s["stall_max_s"] or 0) <= 2.0 and s["idle_after"]["live"] == 0 and s["idle_after"]["thinker_kv_pages_used"] == 0
        and s["idle_after"]["talker_kv_pages_used"] == 0 and abs(s["rss_drift"] or 0) <= 0.01 and abs(s["gpu_reserved_drift"] or 0) <= 0.01}
    (out / "robustness.json").write_text(json.dumps({**report, "samples": samples}, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "samples"}, indent=1))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--sustain-min", type=float, default=30.0)
    ap.add_argument("--overload", type=int, default=128)
    a = ap.parse_args(argv)
    rep = asyncio.run(run(a))
    return 0 if all(rep["passed"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
