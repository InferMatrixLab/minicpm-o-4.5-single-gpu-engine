"""Daily-Omni worker: runs in the vllm-omni environment.

    <vllm-omni venv>/python evals/multimodal/daily_omni_worker.py <job.json> <out.json>

Requests come from vllm-omni's ``DailyOmniDataset`` unchanged: ``input_mode=all``
and ``pack_mode=minicpm-interleave`` (1 fps frames interleaved with matching
1 s audio segments as ``image_url``/``audio_url`` parts, inlined as data URLs),
plus the same client body as vllm-omni's MiniCPM-o accuracy test
(``modalities=["text"]``, ``enable_thinking=false``, temperature 0, 512
tokens).  Scoring is ``compute_daily_omni_accuracy_metrics``.

Job: ``url``, ``model``, ``concurrency``, ``max_samples`` (None = all 1197),
``timeout_s``, ``video_dir``, ``qa_json``.  ``dump_payloads`` returns the first N
request bodies (for the frames-plus-audio fixture comparison).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

import aiohttp

EXTRA_BODY = {"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": False}}
MAX_TOKENS = 512


def load_requests(job: dict) -> list:
    from transformers import AutoTokenizer

    from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniDataset

    ds = DailyOmniDataset(qa_json_path=job["qa_json"], video_dir=job["video_dir"], input_mode="all", pack_mode="minicpm-interleave",
                          inline_local_video=True, random_seed=0)
    tok = AutoTokenizer.from_pretrained(job["tokenizer"], trust_remote_code=True)
    n = job.get("max_samples") or len(ds.data)
    return ds.sample(tok, n, output_len=MAX_TOKENS, no_oversample=True)


def payload(job: dict, req) -> dict:
    body = {"model": job["model"], "messages": req.omni_chat_messages, "temperature": 0.0, "max_tokens": MAX_TOKENS, "stream": False}
    body.update(EXTRA_BODY)
    if req.omni_extra_body:
        body.update(req.omni_extra_body)
    return body


async def run(job: dict) -> dict:
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

    from vllm_omni.benchmarks.data_modules.daily_omni_eval import (
        compute_daily_omni_accuracy_metrics,
        evaluate_answer_official,
        extract_predicted_choice,
        normalize_gold_answer,
    )

    reqs = load_requests(job)
    if job.get("dump_payloads"):
        return {"payloads": [payload(job, r) for r in reqs[: int(job["dump_payloads"])]], "ids": [r.request_id for r in reqs]}
    url = job["url"].rstrip("/") + "/v1/chat/completions"
    outs = [None] * len(reqs)
    sem = asyncio.Semaphore(int(job["concurrency"]))
    timeout = aiohttp.ClientTimeout(total=float(job.get("timeout_s", 900)))

    async def one(i, session):
        o = RequestFuncOutput()
        async with sem:
            t0 = time.perf_counter()
            try:
                async with session.post(url, json=payload(job, reqs[i])) as resp:
                    body = await resp.text()
                    o.latency = time.perf_counter() - t0
                    if resp.status != 200:
                        o.error = f"HTTP {resp.status}: {body[:300]}"
                    else:
                        msg = (json.loads(body).get("choices") or [{}])[0].get("message") or {}
                        o.generated_text = msg.get("content") or ""
                        o.success = bool(o.generated_text)
                        if not o.success:
                            o.error = "Empty response"
            except Exception as e:
                o.error = f"{type(e).__name__}: {e}"
        outs[i] = o

    t0 = time.time()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(one(i, session) for i in range(len(reqs))))
    metrics = compute_daily_omni_accuracy_metrics(reqs, outs, include_per_item=True) or {}
    per = []
    for r, o in zip(reqs, outs):
        gold = normalize_gold_answer(r.daily_omni_gold_answer or "")
        pred = extract_predicted_choice(o.generated_text) if o.success else None
        per.append({"id": f"{r.daily_omni_video_id}#{r.request_id}", "video_id": r.daily_omni_video_id, "task_type": r.daily_omni_task_type,
                    "expected": gold, "predicted": pred, "is_correct": bool(o.success and evaluate_answer_official(o.generated_text, r.daily_omni_gold_answer)), "is_success": bool(o.success),
                    "error": o.error, "raw": (o.generated_text or "")[:2000]})
    return {"bench": "daily_omni", "ids": [p["id"] for p in per], "per_sample": per, "summary": {k: v for k, v in metrics.items() if k != "per_item"},
            "per_item": metrics.get("per_item"), "wall_s": time.time() - t0}


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    Path(sys.argv[2]).write_text(json.dumps(asyncio.run(run(job)), default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
