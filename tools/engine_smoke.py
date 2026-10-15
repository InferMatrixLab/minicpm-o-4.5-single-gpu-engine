"""Drive an engine backend in-process on canonical speech requests and report per-request timings (GPU, diagnostic).

    CUDA_VISIBLE_DEVICES=G .venv-engine/bin/python tools/engine_smoke.py --config configs/engine/track_s.json [--n 4] [--c 1]
        [--save-dir DIR]

Per request: time to first text delta and first audio sample, audio seconds, generation seconds, frames; plus the
engine's tick profile when the backend keeps one.  Not the benchmark (that is bench.cli through the HTTP server).
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


async def one(backend, cr, results, t0_all):
    from engine.server import AudioChunk, Done, EngineRequest, TextDelta

    req = EngineRequest(request_id=cr.request_id, messages=cr.messages(), modalities=["text", "audio"], max_tokens=cr.thinker.max_tokens,
                        temperature=0.0, seed=cr.seed, chat_template_kwargs={"use_tts_template": True, "enable_thinking": False},
                        ref_audio=cr.ref_audio_bytes(),
                        talker_sampling={"temperature": cr.talker.temperature, "top_p": cr.talker.top_p, "top_k": cr.talker.top_k,
                                         "repetition_penalty": cr.talker.repetition_penalty, "repetition_window": cr.talker.repetition_window,
                                         "min_new_tokens": cr.talker.min_new_tokens, "max_new_tokens": cr.talker.max_new_tokens}, stream=True)
    t0 = time.perf_counter()
    first_text = first_audio = None
    text, audio, usage = [], [], {}
    async for ev in backend.generate(req):
        now = time.perf_counter() - t0
        if isinstance(ev, TextDelta):
            first_text = first_text if first_text is not None else now
            text.append(ev.text)
        elif isinstance(ev, AudioChunk):
            first_audio = first_audio if first_audio is not None else now
            audio.append(ev.samples)
        elif isinstance(ev, Done):
            usage = ev.usage
    wall = time.perf_counter() - t0
    wav = np.concatenate(audio) if audio else np.zeros(0, np.float32)
    results.append({"id": cr.request_id, "first_text_s": first_text, "ttfa_s": first_audio, "wall_s": wall, "audio_s": len(wav) / 24000,
                    "rtf": wall / max(len(wav) / 24000, 1e-9), "text": "".join(text), "usage": usage, "wav": wav})


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--c", type=int, default=1)
    ap.add_argument("--save-dir", default=None)
    ap.add_argument("--warmup", type=int, default=1)
    a = ap.parse_args()
    from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets
    from engine.server import EngineServer  # noqa: F401  (import check)

    cfg = json.loads(Path(a.config).read_text())
    mod, _, cls = cfg.get("backend", "engine.runtime.backend:EngineBackend").partition(":")
    t = time.perf_counter()
    backend = getattr(__import__(mod, fromlist=[cls]), cls).from_config(a.config)
    if hasattr(backend, "timelines"):
        backend.profile = True
    print(f"backend up in {time.perf_counter() - t:.1f}s", flush=True)
    rows = load_seed_tts_rows()
    ids = make_id_sets(rows).tuning(1)
    reqs = canonical_requests(rows, ids[: a.warmup + a.n])
    warm = []
    for cr in reqs[: a.warmup]:
        await one(backend, cr, warm, 0)
    res = []
    sem = asyncio.Semaphore(a.c)

    async def gated(cr):
        async with sem:
            await one(backend, cr, res, 0)

    t = time.perf_counter()
    await asyncio.gather(*(gated(cr) for cr in reqs[a.warmup:]))
    wall = time.perf_counter() - t
    for r in res:
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("wav", "text")}), r["text"][:60], flush=True)
    for rid, tl in getattr(backend, "timelines", {}).items():
        if tl and rid in {r["id"] for r in res}:
            t0 = tl[0][1]
            print(rid[:28], " ".join(f"{e}@{(t - t0) * 1000:.0f}" for e, t in tl[:14]), flush=True)
    tt = [r["ttfa_s"] for r in res if r["ttfa_s"] is not None]
    print(json.dumps({"n": len(res), "c": a.c, "wall_s": wall, "frames_per_s": 25 * sum(r["audio_s"] for r in res) / wall,
                      "ttfa_p50": float(np.percentile(tt, 50)) if tt else None, "rtf_mean": float(np.mean([r["rtf"] for r in res]))}), flush=True)
    if a.save_dir:
        import soundfile as sf

        d = Path(a.save_dir)
        d.mkdir(parents=True, exist_ok=True)
        for r in res:
            sf.write(d / f"{r['id']}.wav", r["wav"], 24000)
            (d / f"{r['id']}.txt").write_text(r["text"])


if __name__ == "__main__":
    asyncio.run(main())
