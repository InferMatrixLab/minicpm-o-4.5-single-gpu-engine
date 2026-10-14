"""MMMU / MMSU worker: runs in the locked evaluation environment.

    <eval-env>/python evals/multimodal/sglang_mm_worker.py <job.json> <out.json>

Working directory: the pinned sglang-omni checkout.  Samples, prompts, answer
extraction and accuracy come from its ``benchmarks`` package unchanged.  The
upstream loaders call ``datasets.load_dataset`` without a revision; this worker
routes them through a loader boundary that injects the locked revision of each
dataset repo (an unpinned repo is refused), so a different cached revision can
never be evaluated under the accepted identity.

Modes (``job["mode"]``):
* ``manifest`` - load the samples and return (id, content sha256) for each;
  content covers the prompt, choices/answer and media bytes;
* ``run``      - send each sample to ``url`` and score; full raw responses are kept;
* ``rescore``  - re-score saved raw responses (``job["responses"]``: id -> text or null).

Wire format per engine: ``sglang`` native (top-level ``images``/``audios``) or
``openai`` content parts with the media first, then the prompt text.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import aiohttp

#: Locked dataset revisions (must equal evals/assets.lock.json).
REVISIONS = {
    "MMMU/MMMU": "876ce5cb130f7f7e290ce4d9984357737d4db5cf",
    "zhaochenyang20/mmmu-ci-50": "ff72fd69cc7e0719e04a0ddb12d160de89c6fefe",
    "ddwang2000/MMSU": "548e2283105825bf908a7db5c09c00dbcf42bd4c",
    "zhaochenyang20/mmsu-ci-2000": "5ae6ed4343a89566be6dd90023d529f1bd97802d",
}
CONSUMED: list[dict] = []


def _pin_loaders() -> None:
    import datasets

    import benchmarks.dataset.mmmu as mmmu_mod
    import benchmarks.dataset.mmsu as mmsu_mod

    real = datasets.load_dataset

    def pinned(path, *args, **kwargs):
        if path not in REVISIONS:
            raise RuntimeError(f"dataset {path} has no locked revision")
        if kwargs.get("revision") not in (None, REVISIONS[path]):
            raise RuntimeError(f"dataset {path} requested at a foreign revision {kwargs['revision']}")
        kwargs["revision"] = REVISIONS[path]
        ds = real(path, *args, **kwargs)
        # offline mode may fall back to "the latest cached version": the arrow files actually
        # consumed must come from the locked revision's cache directory
        parts = ds.values() if isinstance(ds, dict) else [ds]
        files = [f["filename"] for d in parts for f in d.cache_files]
        if not files or not all(REVISIONS[path] in f for f in files):
            raise RuntimeError(f"dataset {path} was not loaded from revision {REVISIONS[path]}: {files[:2]}")
        CONSUMED.append({"repo": path, "revision": REVISIONS[path], "args": [str(a) for a in args], "split": kwargs.get("split"),
                         "files": sorted(set(files))[:4]})
        return ds

    mmmu_mod.load_dataset = pinned
    mmsu_mod.load_dataset = pinned


def load(job: dict):
    _pin_loaders()
    if job["bench"] == "mmmu":
        from benchmarks.dataset.mmmu import load_mmmu_samples

        samples = load_mmmu_samples(job.get("max_samples"), repo_id=job.get("repo_id"))
        return samples, [None] * len(samples)
    from benchmarks.dataset.mmsu import load_mmsu_samples
    from benchmarks.tasks.audio_understanding import DEFAULT_PROMPT, _build_question_text

    samples = load_mmsu_samples(job.get("max_samples"), repo_id=job.get("repo_id"))
    return samples, [_build_question_text(s, DEFAULT_PROMPT) for s in samples]


def content_sha(bench: str, sample, text: str | None) -> str:
    h = hashlib.sha256()
    if bench == "mmmu":
        for part in (sample.sample_id, sample.prompt, json.dumps(sample.answer, default=str), sample.question_type, *sample.image_data_uris):
            h.update(str(part).encode())
            h.update(b"\0")
    else:
        for part in (sample.sample_id, text, sample.answer_text, str(sample.answer_index), *sample.choices):
            h.update(str(part).encode())
            h.update(b"\0")
        h.update(Path(sample.audio_path).read_bytes())
    return h.hexdigest()


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


def score(bench: str, samples, results) -> tuple[list[dict], dict, float]:
    if bench == "mmmu":
        from benchmarks.metrics.mmmu import compute_mmmu_metrics
        from benchmarks.tasks.visual_understand import build_mmmu_result_records

        per = build_mmmu_result_records(samples, results)
        summ = compute_mmmu_metrics(per)
        rows = [{"id": p["sample_id"], "expected": p["expected"], "predicted": p["predicted"], "is_correct": p["is_correct"],
                 "is_success": p["is_success"], "error": p["error"], "raw": r.text if r.is_success else None} for p, r in zip(per, results)]
        return rows, summ, summ["accuracy"]
    from benchmarks.metrics.mmsu import compute_mmsu_metrics
    from benchmarks.tasks.audio_understanding import build_mmsu_results

    per = build_mmsu_results(results, samples)
    summ = compute_mmsu_metrics(per)
    rows = [{"id": p.sample_id, "expected": p.correct_choice, "predicted": p.predicted_choice, "is_correct": p.is_correct,
             "is_success": p.is_success, "error": p.error, "raw": p.raw_response if p.is_success else None} for p in per]
    return rows, summ, summ["overall_accuracy"]


async def send_all(job: dict, samples, texts) -> list:
    from benchmarks.benchmarker.data import RequestResult

    url = job["url"].rstrip("/") + "/v1/chat/completions"
    results: list = [None] * len(samples)
    sem = asyncio.Semaphore(int(job["concurrency"]))
    timeout = aiohttp.ClientTimeout(total=float(job.get("timeout_s", 600)))

    async def one(i, session):
        s = samples[i]
        r = RequestResult(request_id=s.sample_id)
        async with sem:
            t0 = time.perf_counter()
            try:
                async with session.post(url, json=build_payload(job, s, texts[i])) as resp:
                    body = await resp.text()
                    r.latency_s = time.perf_counter() - t0
                    if resp.status != 200:
                        r.error = f"HTTP {resp.status}: {body[:300]}"
                    else:
                        obj = json.loads(body)
                        msg = (obj.get("choices") or [{}])[0].get("message") or {}
                        content = msg.get("content")
                        r.text = content if isinstance(content, str) else ""
                        r.is_success = bool(r.text)
                        if not r.is_success:
                            r.error = "Empty response"
            except Exception as e:  # recorded as a failed sample, never dropped
                r.error = f"{type(e).__name__}: {e}"
        results[i] = r

    async with aiohttp.ClientSession(timeout=timeout) as session:
        await asyncio.gather(*(one(i, session) for i in range(len(samples))))
    return results


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, os.getcwd())
    from benchmarks.benchmarker.data import RequestResult

    samples, texts = load(job)
    if job["mode"] == "manifest":
        out = {"bench": job["bench"], "items": [[s.sample_id, content_sha(job["bench"], s, t)] for s, t in zip(samples, texts)], "consumed": CONSUMED}
    elif job["mode"] == "rescore":
        resp = job["responses"]
        results = []
        for s in samples:
            r = RequestResult(request_id=s.sample_id)
            text = resp.get(s.sample_id)
            if isinstance(text, str) and text:
                r.text, r.is_success = text, True
            else:
                r.error = "no response"
            results.append(r)
        rows, summ, acc = score(job["bench"], samples, results)
        out = {"bench": job["bench"], "ids": [s.sample_id for s in samples], "per_sample": rows, "summary": summ, "accuracy": acc, "consumed": CONSUMED}
    else:
        t0 = time.time()
        results = asyncio.run(send_all(job, samples, texts))
        rows, summ, acc = score(job["bench"], samples, results)
        out = {"bench": job["bench"], "ids": [s.sample_id for s in samples], "per_sample": rows, "summary": summ, "accuracy": acc,
               "wall_s": time.time() - t0, "consumed": CONSUMED}
    Path(sys.argv[2]).write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
