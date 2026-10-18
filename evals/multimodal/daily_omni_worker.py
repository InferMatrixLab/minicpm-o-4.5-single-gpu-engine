"""Daily-Omni worker: runs in the vllm-omni environment.

    <vllm-omni venv>/python evals/multimodal/daily_omni_worker.py <job.json> <out.json>

Requests come from vllm-omni's ``DailyOmniDataset`` unchanged: ``input_mode=all``
and ``pack_mode=minicpm-interleave`` (1 fps frames interleaved with matching
1 s audio segments as ``image_url``/``audio_url`` parts, inlined as data URLs),
plus the same client body as vllm-omni's MiniCPM-o accuracy test
(``modalities=["text"]``, ``enable_thinking=false``, temperature 0, 512
tokens).  Scoring is ``evaluate_answer_official`` /
``compute_daily_omni_accuracy_metrics``.  The QA file and media come from the
pinned snapshot given in the job.

Modes: ``manifest`` (per-item content sha256: QA fields plus the video and audio
file bytes), ``run`` (``format`` ``sglang`` sends sglang-omni's native ``video_url`` form, see
``sglang_payload``; the job's ``generation_digest`` is required before anything is sent and comes back on the result
and every response row), ``rescore`` (saved raw responses -> official correctness),
and ``dump_payloads`` (first N request bodies, for the native-form fixture test).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path

import aiohttp

EXTRA_BODY = {"modalities": ["text"], "chat_template_kwargs": {"enable_thinking": False}}
MAX_TOKENS = 512


def _dataset(job: dict):
    from vllm_omni.benchmarks.data_modules.daily_omni_dataset import DailyOmniDataset

    return DailyOmniDataset(qa_json_path=job["qa_json"], video_dir=job["video_dir"], input_mode="all", pack_mode="minicpm-interleave",
                            inline_local_video=True, random_seed=0)


def _items(job: dict) -> list[dict]:
    """The dataset's own (seeded) order and field normalization; request i is item i."""
    ds = _dataset(job)
    rows = [ds._normalize_qa_fields(ds._coerce_row(it)) for it in ds.data]
    n = job.get("max_samples") or len(rows)
    return rows[:n]


def _key(i: int, f: dict) -> str:
    return f"{f['video_id']}#{i}"


def _file_sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def manifest(job: dict) -> dict:
    vdir = Path(job["video_dir"])
    items = []
    for i, f in enumerate(_items(job)):
        h = hashlib.sha256()
        for part in (f["video_id"], f["question"], json.dumps(f["choice"], default=str), f["answer"], f["task_type"]):
            h.update(str(part).encode())
            h.update(b"\0")
        for rel in (f"{f['video_id']}/{f['video_id']}_video.mp4", f"{f['video_id']}/{f['video_id']}_audio.wav"):
            h.update(_file_sha(vdir / rel).encode())
        items.append([_key(i, f), h.hexdigest()])
    return {"bench": "daily_omni", "items": items, "qa_json_sha256": _file_sha(Path(job["qa_json"]))}


def load_requests(job: dict) -> list:
    from transformers import AutoTokenizer

    ds = _dataset(job)
    tok = AutoTokenizer.from_pretrained(job["tokenizer"], trust_remote_code=True)
    n = job.get("max_samples") or len(ds.data)
    return ds.sample(tok, n, output_len=MAX_TOKENS, no_oversample=True)


def _video_path(job: dict, video_id: str) -> str:
    return str(Path(job["video_dir"]) / video_id / f"{video_id}_video.mp4")


def payload(job: dict, req) -> dict:
    if job.get("format") == "sglang":
        return sglang_payload(job, req)
    body = {"model": job["model"], "messages": req.omni_chat_messages, "temperature": 0.0, "max_tokens": MAX_TOKENS, "stream": False}
    body.update(EXTRA_BODY)
    if req.omni_extra_body:
        body.update(req.omni_extra_body)
    return body


def sglang_payload(job: dict, req) -> dict:
    """sglang-omni's native form of the same request: MiniCPM-o's official omni path.

    vllm-omni expands the video by hand into 1 fps ``image_url`` frames interleaved with 1 s
    ``audio_url`` segments and asks for the omni frame policy through ``mm_processor_kwargs``
    (``max_slice_nums=1``, ``use_image_id=False``).  sglang-omni does not read those kwargs; it
    applies that policy, the 1 fps sampling and the per-frame audio interleave itself, only for a
    ``video_url`` part with ``use_audio``.  Sent as frames, each frame would be sliced and labelled as a
    separate picture.  The question text parts are unchanged; the media is the same video file.
    """
    (msg,) = req.omni_chat_messages
    texts = [c for c in msg["content"] if isinstance(c, dict) and c.get("type") == "text"]
    content = [{"type": "video_url", "video_url": {"url": _video_path(job, req.daily_omni_video_id), "use_audio": True}}, *texts]
    body = {"model": job["model"], "messages": [{"role": msg["role"], "content": content}], "temperature": 0.0, "max_tokens": MAX_TOKENS,
            "stream": False}
    body.update(EXTRA_BODY)
    return body


def generation_digest(job: dict) -> str:
    d = job.get("generation_digest")
    if not isinstance(d, str) or not d:
        raise SystemExit("run mode needs the job's generation_digest (the declaration the responses are produced under)")
    return d


def rescore(job: dict) -> dict:
    from vllm_omni.benchmarks.data_modules.daily_omni_eval import evaluate_answer_official, extract_predicted_choice, normalize_gold_answer

    resp = job["responses"]
    per = []
    for i, f in enumerate(_items(job)):
        key = _key(i, f)
        raw = resp.get(key)
        ok = isinstance(raw, str) and bool(raw)
        per.append({"id": key, "video_id": f["video_id"], "task_type": f["task_type"], "question": f["question"], "gold_raw": f["answer"],
                    "expected": normalize_gold_answer(f["answer"]), "predicted": extract_predicted_choice(raw) if ok else None,
                    "is_correct": bool(ok and evaluate_answer_official(raw, f["answer"])), "is_success": ok, "error": "" if ok else "no response",
                    "raw": raw if ok else None})
    acc = sum(p["is_correct"] for p in per) / len(per) if per else float("nan")
    return {"bench": "daily_omni", "ids": [p["id"] for p in per], "per_sample": per, "accuracy": acc}


async def run(job: dict) -> dict:
    from vllm.benchmarks.lib.endpoint_request_func import RequestFuncOutput

    from vllm_omni.benchmarks.data_modules.daily_omni_eval import compute_daily_omni_accuracy_metrics

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
    # the stored per-sample record is the rescore of the raw responses, so evidence and verification agree by construction
    responses = {f"{r.daily_omni_video_id}#{i}": (o.generated_text if o.success else None) for i, (r, o) in enumerate(zip(reqs, outs))}
    res = rescore({**job, "responses": responses})
    for p, o in zip(res["per_sample"], outs):
        if not o.success:
            p["error"] = o.error
    res["summary"] = {k: v for k, v in metrics.items() if k != "per_item"}
    res["wall_s"] = time.time() - t0
    return res


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    mode = job.get("mode", "run")
    if mode == "manifest":
        out = manifest(job)
    elif mode == "rescore":
        out = rescore(job)
    elif job.get("dump_payloads"):
        out = asyncio.run(run(job))
    else:
        digest = generation_digest(job)  # before anything is sent
        out = asyncio.run(run(job))
        out["per_sample"] = [{**p, "generation_digest": digest} for p in out["per_sample"]]
        out["generation_digest"] = digest
    Path(sys.argv[2]).write_text(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
