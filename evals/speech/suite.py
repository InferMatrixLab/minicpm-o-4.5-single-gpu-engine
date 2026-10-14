"""Speech quality suite over harness run directories.

    python -m evals.speech.suite score --run results/.../run1:0 --run results/.../run1:1 \
        --gpu 2 --out results/quality/<name>.json [--d-ref d_ref.json] [--no-whisper]

Each ``--run DIR:SEED`` names one harness run directory (``requests.jsonl``,
``traces.jsonl``, ``wav/``) produced with that seed index.  Every request of
every run must be valid and have audio: coverage is complete or the suite
fails.  Scorers run in the locked evaluation environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.servers import Server, ServerSpec  # noqa: E402
from bench.workload import load_seed_tts_rows  # noqa: E402
from evals.speech.detectors import (  # noqa: E402
    CATASTROPHIC_WER,
    RowQuality,
    corpus_metrics,
    repetition_flag,
    silence_profile,
    truncation_flag,
    word_recall,
)
from evals.speech.gates import absolute_floors  # noqa: E402

EVAL_PYTHON = "/workspace/cgzheng/cgzheng/omni-baselines/eval-env/bin/python"
SGLANG_DIR = "/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main"
ASSETS_LOCK = ROOT / "evals" / "assets.lock.json"
ASR_MODEL = "Qwen/Qwen3-ASR-1.7B"
WHISPER_MODEL = "openai/whisper-large-v3"
ASR_CONCURRENCY = 16  # fixed and recorded


class CoverageError(RuntimeError):
    pass


@dataclass
class GenRow:
    key: str  # "<seed>:<row_id>"
    row_id: str
    seed_index: int
    target_text: str
    ref_wav: str
    wav: str
    thinker_text: str
    duration_s: float


def collect_rows(runs: list[tuple[Path, int]]) -> list[GenRow]:
    meta = {r.row_id: r for r in load_seed_tts_rows()}
    out: list[GenRow] = []
    for run_dir, seed in runs:
        run_dir = Path(run_dir).resolve()  # the scorer worker runs from another working directory
        reqs = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines() if line.strip()]
        texts = {}
        for line in (run_dir / "traces.jsonl").read_text().splitlines():
            if line.strip():
                t = json.loads(line)
                texts[t["request_id"]] = t.get("text", "")
        bad = [r["request_id"] for r in reqs if not r["ok"]]
        if bad:
            raise CoverageError(f"{run_dir}: {len(bad)} invalid requests (e.g. {bad[:3]}); quality needs complete coverage")
        for r in reqs:
            rid = r["request_id"]
            wav = run_dir / "wav" / f"{rid}.wav"
            if not wav.exists():
                raise CoverageError(f"{run_dir}: missing audio for {rid}")
            row = meta[rid]
            out.append(GenRow(f"{seed}:{rid}", rid, seed, row.target_text, str(Path(row.ref_wav_path).resolve()), str(wav), texts.get(rid, ""),
                              r["duration_s"]))
    keys = [g.key for g in out]
    if len(set(keys)) != len(keys):
        raise CoverageError("duplicate (seed, row) pairs across runs")
    return out


def rows_sha256(rows: list[GenRow]) -> str:
    return hashlib.sha256("\n".join(sorted(r.key for r in rows)).encode()).hexdigest()


def _worker(job: dict, gpu: int) -> dict:
    with tempfile.TemporaryDirectory() as td:
        jp, op = Path(td) / "job.json", Path(td) / "out.json"
        jp.write_text(json.dumps(job))
        env = {"CUDA_VISIBLE_DEVICES": str(gpu), "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
               "HF_HUB_CACHE": "/workspace/models/hub", "HF_MODULES_CACHE": "/workspace/models/cgzheng/hf_modules", "HF_HUB_OFFLINE": "1"}
        subprocess.run([EVAL_PYTHON, str(ROOT / "evals" / "speech" / "scorer_worker.py"), str(jp), str(op)], cwd=SGLANG_DIR, env=env, check=True)
        return json.loads(op.read_text())


def _asr_spec(kind: str, port: int, gpu: int) -> ServerSpec:
    return ServerSpec(name=f"asr:{kind}", cmd=["bash", str(ROOT / "scripts" / "serve_sglang_asr.sh"), kind, str(port)], port=port, gpu=gpu)


def score(rows: list[GenRow], gpu: int, out_dir: Path, whisper: bool = True, port: int = 18300) -> dict:
    job_rows = [{"id": r.key, "target_text": r.target_text, "wav": r.wav, "ref_wav": r.ref_wav} for r in rows]
    base = {"rows": job_rows, "lock": str(ASSETS_LOCK), "asr_concurrency": ASR_CONCURRENCY, "device": "cuda:0",
            "asr_model": ASR_MODEL, "whisper_model": WHISPER_MODEL, "asr_port": port, "whisper_port": port}
    results: dict = {"tasks": {}}
    with Server(_asr_spec("qwen3-asr", port, gpu), out_dir / "asr_server.log"):
        r = _worker({**base, "tasks": ["asr"]}, gpu)
        results["assets"] = r["assets"]
        results["tasks"]["asr"] = r["tasks"]["asr"]
    if whisper:
        with Server(_asr_spec("whisper", port, gpu), out_dir / "whisper_server.log"):
            results["tasks"]["whisper"] = _worker({**base, "tasks": ["whisper"]}, gpu)["tasks"]["whisper"]
    r = _worker({**base, "tasks": ["sim", "utmos"]}, gpu)
    results["tasks"].update(r["tasks"])
    return results


def build_quality(rows: list[GenRow], scored: dict, d_ref: dict[str, float] | None) -> list[RowQuality]:
    asr = scored["tasks"]["asr"]["rows"]
    sim = scored["tasks"]["sim"]["rows"]
    utm = scored["tasks"]["utmos"]["rows"]
    wsp = scored["tasks"].get("whisper", {}).get("rows", {})
    failed = [r.key for r in rows if not asr.get(r.key, {}).get("ok") or r.key not in sim or r.key not in utm]
    if failed:
        raise CoverageError(f"{len(failed)} rows were not scored (e.g. {failed[:3]}); coverage must be complete")
    out = []
    for r in rows:
        a = asr[r.key]
        pcm, sr = sf.read(r.wav, dtype="float32", always_2d=False)
        if pcm.ndim > 1:
            pcm = pcm.mean(axis=1)
        prof = silence_profile(np.asarray(pcm, dtype=np.float32), sr)
        recall = word_recall(a["hits"], a["substitutions"], a["deletions"])
        dref = d_ref.get(r.row_id) if d_ref else None
        w = wsp.get(r.key)
        out.append(
            RowQuality(
                row_id=r.row_id, seed_index=r.seed_index, target_text=r.target_text, thinker_text=r.thinker_text, hyp_text=a["hyp_text"],
                ref_norm=a["ref_norm"], hyp_norm=a["hyp_norm"], wer=a["wer"], substitutions=a["substitutions"], deletions=a["deletions"],
                insertions=a["insertions"], hits=a["hits"], sim=sim[r.key], utmos=utm[r.key], duration_s=r.duration_s, d_ref_s=dref,
                leading_silence_s=prof.leading_s, trailing_silence_s=prof.trailing_s, max_internal_silence_s=prof.max_internal_s,
                silent_fraction=prof.silent_fraction, catastrophic=a["wer"] > CATASTROPHIC_WER, silence=prof.flagged,
                repetition=repetition_flag(a["hyp_norm"], a["ref_norm"]), truncation=truncation_flag(recall, r.duration_s, dref),
                thinker_exact_match=r.thinker_text.strip() == r.target_text.strip(),
                whisper_wer=(w["wer"] if w and w.get("ok") else None),
            )
        )
    return out


def summarize(quality: list[RowQuality]) -> dict:
    seeds = sorted({q.seed_index for q in quality})
    per_seed = []
    for s in seeds:
        m = corpus_metrics([q for q in quality if q.seed_index == s])
        m["seed_index"] = s
        m["absolute_floors"] = absolute_floors(m)
        per_seed.append(m)
    pooled = corpus_metrics(quality)
    return {"seed_indices": seeds, "per_seed": per_seed, "pooled": pooled, "absolute_floors_pooled": absolute_floors(pooled)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.speech.suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("score")
    sp.add_argument("--run", action="append", required=True, help="RUN_DIR:SEED_INDEX")
    sp.add_argument("--gpu", type=int, required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--d-ref", default=None, help="JSON {row_id: D_ref seconds} from the cell's throughput winner")
    sp.add_argument("--no-whisper", action="store_true")
    sp.add_argument("--port", type=int, default=18300)
    a = ap.parse_args(argv)
    runs = []
    for spec in a.run:
        d, _, s = spec.rpartition(":")
        runs.append((Path(d), int(s)))
    rows = collect_rows(runs)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scored = score(rows, a.gpu, out.parent, whisper=not a.no_whisper, port=a.port)
    d_ref = json.loads(Path(a.d_ref).read_text()) if a.d_ref else None
    quality = build_quality(rows, scored, d_ref)
    rep = {
        "runs": [{"dir": str(d), "seed_index": s} for d, s in runs],
        "rows_sha256": rows_sha256(rows),
        "n_rows": len(rows),
        "asr": {"model": ASR_MODEL, "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5", "concurrency": ASR_CONCURRENCY, "decoding": "greedy server default"},
        "whisper": None if a.no_whisper else {"model": WHISPER_MODEL, "revision": "06f233fe06e710322aca913c1bc4249a0d71fce1"},
        "assets_verified": scored.get("assets", {}),
        "summary": summarize(quality),
        "wall_s": time.time() - t0,
    }
    out.write_text(json.dumps(rep, indent=1))
    with out.with_suffix(".rows.jsonl").open("w") as f:
        for q in quality:
            f.write(json.dumps(q.to_json()) + "\n")
    s = rep["summary"]["pooled"]
    print(json.dumps({k: s[k] for k in ("rows", "wer_corpus", "wer_filtered_corpus", "catastrophic_count", "sim_mean", "utmos_mean",
                                         "silence_rate", "repetition_rate", "thinker_exact_match_rate")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
