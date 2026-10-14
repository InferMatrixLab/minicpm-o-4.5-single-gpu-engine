"""MMMU / MMSU worker: runs in the locked evaluation environment.

    <eval-env>/python evals/multimodal/sglang_mm_worker.py <job.json> <out.json>

Working directory: the pinned sglang-omni checkout.  Samples, prompts, answer
extraction and accuracy come from its ``benchmarks`` package unchanged; only
the wire format of the request is chosen per engine:

* ``sglang`` - sglang-omni's native body (top-level ``images`` / ``audios``);
* ``openai`` - OpenAI content parts: the media first (``image_url`` or
  ``input_audio``), then the prompt text, which is where sglang-omni places
  its media placeholders, so the rendered prompts agree.

Job: ``bench`` (mmmu|mmsu), ``repo_id`` (None = full set), ``max_samples``,
``url``, ``model``, ``format``, ``concurrency``, ``max_tokens``,
``extra_body``, ``timeout_s``.
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
import time
from pathlib import Path

import aiohttp


def _openai_mmmu(sample) -> list:
    parts = [{"type": "image_url", "image_url": {"url": uri}} for uri in sample.image_data_uris]
    return parts + [{"type": "text", "text": sample.prompt}]


def _openai_mmsu(sample, text: str) -> list:
    path = Path(sample.audio_path)
    fmt = path.suffix.lstrip(".").lower() or "wav"
    data = base64.b64encode(path.read_bytes()).decode()
    return [{"type": "input_audio", "input_audio": {"data": data, "format": fmt}}, {"type": "text", "text": text}]


def build_payload(job: dict, sample, mmsu_text: str | None) -> dict:
    body = {"model": job["model"], "max_tokens": job["max_tokens"], "temperature": 0.0, "stream": False, "modalities": ["text"]}
    body.update(job.get("extra_body") or {})
    if job["bench"] == "mmmu":
        if job["format"] == "sglang":
            body.update({"messages": [{"role": "user", "content": sample.prompt}], "images": list(sample.image_data_uris)})
        else:
            body["messages"] = [{"role": "user", "content": _openai_mmmu(sample)}]
    else:
        if job["format"] == "sglang":
            body.update({"messages": [{"role": "user", "content": mmsu_text}], "audios": [sample.audio_path]})
        else:
            body["messages"] = [{"role": "user", "content": _openai_mmsu(sample, mmsu_text)}]
    return body


async def run(job: dict) -> dict:
    from benchmarks.benchmarker.data import RequestResult

    if job["bench"] == "mmmu":
        from benchmarks.dataset.mmmu import load_mmmu_samples

        samples = load_mmmu_samples(job.get("max_samples"), repo_id=job.get("repo_id"))
        texts = [None] * len(samples)
    else:
        from benchmarks.dataset.mmsu import load_mmsu_samples
        from benchmarks.tasks.audio_understanding import DEFAULT_PROMPT, _build_question_text

        samples = load_mmsu_samples(job.get("max_samples"), repo_id=job.get("repo_id"))
        texts = [_build_question_text(s, DEFAULT_PROMPT) for s in samples]
    url = job["url"].rstrip("/") + "/v1/chat/completions"
    results: list = [None] * len(samples)
    sem = asyncio.Semaphore(int(job["concurrency"]))
    timeout = aiohttp.ClientTimeout(total=float(job.get("timeout_s", 600)))

    async def one(i, session):
        s = samples[i]
        r = RequestResult(request_id=s.sample_id)
        payload = build_payload(job, s, texts[i])
        async with sem:
            t0 = time.perf_counter()
            try:
                async with session.post(url, json=payload) as resp:
                    body = await resp.text()
                    r.latency_s = time.perf_counter() - t0
                    if resp.status != 200:
                        r.error = f"HTTP {resp.status}: {body[:300]}"
                    else:
                        obj = json.loads(body)
                        msg = (obj.get("choices") or [{}])[0].get("message") or {}
                        content = msg.get("content")
                        r.text = content if isinstance(content, str) else ""
                        usage = obj.get("usage") or {}
                        r.prompt_tokens = usage.get("prompt_tokens", 0)
                        r.completion_tokens = usage.get("completion_tokens", 0)
                        r.is_success = bool(r.text)
                        if not r.is_success:
                            r.error = "Empty response"
            except Exception as e:  # recorded as a failed sample, never dropped
                r.error = f"{type(e).__name__}: {e}"
        results[i] = r

    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(one(i, session) for i in range(len(samples))))
    wall = time.time() - t0
    if job["bench"] == "mmmu":
        from benchmarks.metrics.mmmu import compute_mmmu_metrics
        from benchmarks.tasks.visual_understand import build_mmmu_result_records

        per = build_mmmu_result_records(samples, results)
        summ = compute_mmmu_metrics(per)
        per_out = [{"id": p["sample_id"], "expected": p["expected"], "predicted": p["predicted"], "is_correct": p["is_correct"],
                    "is_success": p["is_success"], "error": p["error"], "raw": (r.text or "")[:2000]} for p, r in zip(per, results)]
        accuracy = summ["accuracy"]
    else:
        from benchmarks.metrics.mmsu import compute_mmsu_metrics
        from benchmarks.tasks.audio_understanding import build_mmsu_results

        per = build_mmsu_results(results, samples)
        summ = compute_mmsu_metrics(per)
        per_out = [{"id": p.sample_id, "expected": p.correct_choice, "predicted": p.predicted_choice, "is_correct": p.is_correct,
                    "is_success": p.is_success, "error": p.error, "raw": p.raw_response[:2000]} for p in per]
        accuracy = summ["overall_accuracy"]
    return {"bench": job["bench"], "ids": [s.sample_id for s in samples], "per_sample": per_out, "summary": summ, "accuracy": accuracy,
            "wall_s": wall, "example_payload_keys": sorted(build_payload(job, samples[0], texts[0]).keys()) if samples else []}


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, os.getcwd())
    out = asyncio.run(run(job))
    Path(sys.argv[2]).write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
