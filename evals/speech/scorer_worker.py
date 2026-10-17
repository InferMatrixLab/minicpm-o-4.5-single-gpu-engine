"""Scorer worker: runs in the locked evaluation environment.

    <eval-env>/python evals/speech/scorer_worker.py <job.json> <out.json>

The working directory is the pinned sglang-omni checkout, so its ``benchmarks``
scorers are imported.  The orchestrator verifies the evaluator fingerprint
(checkout, environments, checkpoint hashes) before launching this worker; the
worker loads the scorers from the explicitly verified checkpoint paths.

Job: ``rows`` [{id, target_text, wav, ref_wav, hyp_text?, whisper_text?}],
``tasks`` (subset of wer/whisper_wer/sim/utmos), ``device``,
``wavlm_finetune``, ``wavlm_base``, ``utmos_jit``, ``sim_batch``, ``sim_attention`` (``query_blocked_v1``: the WavLM
attention is evaluated in query blocks, ``evals.speech.wavlm_blocked``, so a long output fits beside its batch).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def run_wer(rows: list[dict], field: str) -> dict:
    from benchmarks.metrics.wer import SampleOutput
    from benchmarks.tasks.asr import apply_wer

    out = {}
    for r in rows:
        o = apply_wer(SampleOutput(sample_id=r["id"], target_text=r["target_text"]), r[field], "en")
        out[r["id"]] = {"ok": o.is_success, "error": o.error, "hyp_text": r[field], "ref_norm": o.ref_norm, "hyp_norm": o.hyp_norm,
                        "wer": o.wer, "substitutions": o.substitutions, "deletions": o.deletions, "insertions": o.insertions, "hits": o.hits}
    return {"rows": out}


def run_sim(rows: list[dict], device: str, finetune: str, base: str, batch: int, attention: str) -> dict:
    from benchmarks.metrics.speaker_similarity import WavLMSpeakerSimilarity

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import evals.speech.wavlm_blocked as wb

    if attention != wb.VERSION:
        raise SystemExit(f"unknown SIM attention {attention!r}")
    wb.install()
    scorer = WavLMSpeakerSimilarity(finetune_checkpoint=finetune, wavlm_base=base, device=device)
    out = {}
    order = sorted(rows, key=lambda r: r["id"])
    for i in range(0, len(order), batch):
        part = order[i : i + batch]
        for r, s in zip(part, scorer.score_batch([r["ref_wav"] for r in part], [r["wav"] for r in part])):
            out[r["id"]] = float(s)
    return {"rows": out, "batch": batch, "scale": "x100", "attention": wb.VERSION, "query_block": wb.QUERY_BLOCK,
            "checkpoints": {"finetune": finetune, "wavlm_base": base}}


def run_utmos(rows: list[dict], device: str, utmos_jit: str) -> dict:
    from benchmarks.metrics.utmos import UTMOSScorer

    scorer = UTMOSScorer(device=device, cache_dir=str(Path(utmos_jit).parent))
    order = sorted(rows, key=lambda r: r["id"])
    return {"rows": {r["id"]: float(s) for r, s in zip(order, scorer.score_batch([r["wav"] for r in order]))}, "checkpoint": utmos_jit}


def main() -> int:
    job = json.loads(Path(sys.argv[1]).read_text())
    sys.path.insert(0, os.getcwd())
    rows = job["rows"]
    out = {}
    for task in job["tasks"]:
        if task == "wer":
            out["asr"] = run_wer(rows, "hyp_text")
        elif task == "whisper_wer":
            out["whisper"] = run_wer(rows, "whisper_text")
        elif task == "sim":
            out["sim"] = run_sim(rows, job["device"], job["wavlm_finetune"], job["wavlm_base"], int(job["sim_batch"]), job["sim_attention"])
        elif task == "utmos":
            out["utmos"] = run_utmos(rows, job["device"], job["utmos_jit"])
        else:
            raise SystemExit(f"unknown task {task}")
    Path(sys.argv[2]).write_text(json.dumps({"tasks": out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
