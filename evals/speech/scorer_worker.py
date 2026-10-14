"""Scorer worker: runs in the locked evaluation environment.

Invoked by ``evals.speech.suite`` as

    <eval-env>/python evals/speech/scorer_worker.py <job.json> <out.json>

with the working directory set to the pinned sglang-omni checkout, so the
pinned ``benchmarks`` scorers are imported from there.  Before any model is
loaded, scorer checkpoints are checked against the SHA-256 values in
``evals/assets.lock.json``.

Job fields: ``rows`` [{id, target_text, wav, ref_wav}], ``tasks`` (subset of
asr/whisper/sim/utmos), ``asr_port``, ``whisper_port``, ``asr_model``,
``whisper_model``, ``asr_concurrency``, ``device``, ``lock``.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_assets(lock_path: str) -> dict:
    lock = json.loads(Path(lock_path).read_text())
    checked = {}
    for path, info in lock["files"].items():
        p = Path(path)
        if not p.exists():
            raise SystemExit(f"pinned asset {p} is missing")
        got = _sha256(p)
        if got != info["sha256"]:
            raise SystemExit(f"asset {p} sha256 {got} != pinned {info['sha256']}")
        checked[path] = got
    return checked


def run_asr(rows: list[dict], port: int, model: str, concurrency: int) -> dict:
    from benchmarks.dataset.seedtts import SampleInput
    from benchmarks.metrics.wer import SampleOutput
    from benchmarks.tasks.asr import apply_wer, run_asr_transcription

    samples = [SampleInput(sample_id=r["id"], ref_text=r["target_text"], ref_audio=r["wav"], target_text=r["target_text"]) for r in rows]
    t0 = time.time()
    results, wall = asyncio.run(run_asr_transcription(samples, port=port, model_path=model, lang="en", concurrency=concurrency, warmup=0))
    by_id = {r.request_id: r for r in results}
    out = {}
    for r in rows:
        res = by_id.get(r["id"])
        o = SampleOutput(sample_id=r["id"], target_text=r["target_text"])
        if res is None or not res.is_success:
            out[r["id"]] = {"ok": False, "error": (res.error if res else "") or "no transcription"}
            continue
        o = apply_wer(o, res.text, "en")
        out[r["id"]] = {
            "ok": o.is_success,
            "error": o.error,
            "hyp_text": res.text,
            "ref_norm": o.ref_norm,
            "hyp_norm": o.hyp_norm,
            "wer": o.wer,
            "substitutions": o.substitutions,
            "deletions": o.deletions,
            "insertions": o.insertions,
            "hits": o.hits,
        }
    return {"rows": out, "wall_s": wall, "elapsed_s": time.time() - t0, "model": model, "concurrency": concurrency}


def run_sim(rows: list[dict], device: str, batch: int = 16) -> dict:
    from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity
    from benchmarks.metrics.speaker_similarity_assets import ensure_speaker_similarity_assets

    assets = ensure_speaker_similarity_assets()
    scorer = WavLMSpeakerSimilarity(finetune_checkpoint=assets.finetune_checkpoint, wavlm_base=assets.wavlm_base, device=device)
    out = {}
    for i in range(0, len(rows), batch):
        part = rows[i : i + batch]
        scores = scorer.score_batch([r["ref_wav"] for r in part], [r["wav"] for r in part])
        for r, s in zip(part, scores):
            out[r["id"]] = float(s)
    return {"rows": out, "batch": batch, "scale": "x100"}


def run_utmos(rows: list[dict], device: str) -> dict:
    from benchmarks.metrics.utmos import UTMOSScorer

    scorer = UTMOSScorer(device=device)
    scores = scorer.score_batch([r["wav"] for r in rows])
    return {"rows": {r["id"]: float(s) for r, s in zip(rows, scores)}}


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    out_path = Path(sys.argv[2])
    sys.path.insert(0, os.getcwd())
    result = {"assets": verify_assets(job["lock"]), "tasks": {}}
    rows = job["rows"]
    for task in job["tasks"]:
        if task == "asr":
            result["tasks"]["asr"] = run_asr(rows, job["asr_port"], job["asr_model"], job["asr_concurrency"])
        elif task == "whisper":
            result["tasks"]["whisper"] = run_asr(rows, job["whisper_port"], job["whisper_model"], job["asr_concurrency"])
        elif task == "sim":
            result["tasks"]["sim"] = run_sim(rows, job["device"])
        elif task == "utmos":
            result["tasks"]["utmos"] = run_utmos(rows, job["device"])
        else:
            raise SystemExit(f"unknown task {task}")
    out_path.write_text(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
