"""Speech quality suite over verified harness cells.

    python -m evals.speech.suite score --cell CELL_DIR [--cell ...] --scope development|final \
        --gpu 2 --out results/quality/<name>.json [--d-ref d_ref.json] [--no-whisper]

Each ``--cell`` is a measured cell directory; it must pass ``bench.verify``.
The seed index comes from the cell's frozen experiment spec, never from the
command line.  All cells must share one candidate, GPU, concurrency and row set,
with distinct seeds.  Every declared row needs its request record, trace text
and a WAV whose length matches the recorded duration: coverage is complete or
the suite fails.  ASR is offline and fixed-batch (``asr_worker.py``); WER
normalization, SIM and UTMOS run in the locked evaluation environment.  The
evaluator fingerprint is verified before any model is loaded.
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

from bench import paths  # noqa: E402
from bench.verify import verify_cell  # noqa: E402
from bench.workload import load_seed_tts_rows  # noqa: E402
from evals.evaluator import ASR_PYTHON, ASSETS_LOCK, EVAL_PYTHON, SETTINGS, SGLANG_DIR, verify_evaluator  # noqa: E402
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

SCOPES = ("development", "final")
FULL_ROWS = 1088


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


@dataclass
class Collection:
    rows: list[GenRow]
    generation: dict  # shared identity of all seeds
    seeds: dict  # seed_index -> source binding


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def collect(cells: list[Path], run_label: str | None = None) -> Collection:
    cells = [Path(c).resolve() for c in cells]
    if len(set(cells)) != len(cells):
        raise CoverageError("the same source cell was given more than once")
    meta = {r.row_id: r for r in load_seed_tts_rows()}
    rows: list[GenRow] = []
    seeds: dict[int, dict] = {}
    gen = None
    for cell in cells:
        v = verify_cell(cell)
        if not v.ok:
            raise CoverageError(f"{cell} failed verification: {v.problems[:3]}")
        spec = v.spec
        k = int(spec["k"])
        label = run_label or ("run1" if k == 1 else None)
        if label is None or label == "run0" or label not in {f"run{i}" for i in range(1, k + 1)}:
            raise CoverageError(f"{cell}: choose one measured run (run1..run{k}); run0 is discarded")
        ident = {
            "candidate_id": spec["candidate"]["candidate_id"],
            "candidate_digest": spec["candidate"]["digest"],
            "candidate": dict(spec["candidate"]),  # complete resolved launch identity (config, args, env, deploy hashes, MPS)
            "gpu_sku": spec["hardware"].get("gpu_sku"),
            # the hardware class (SKU, driver, memory) must agree; seeds may run on different devices of that class,
            # each seed's own device UUID is bound below
            "gpu_driver": spec["hardware"].get("driver"),
            "gpu_memory_mib": spec["hardware"].get("memory_mib"),
            "concurrency": spec["concurrency"],
            "id_set": spec["id_set"],
            "measured_ids_sha256": hashlib.sha256("\n".join(spec["measured_ids"]).encode()).hexdigest(),
            "dataset": spec["dataset"],
            "model": spec["model"],
            "seed_policy": spec["sampling"].get("seed_policy"),
        }
        if gen is None:
            gen = ident
        elif ident != gen:
            diff = {k2: (gen[k2], ident[k2]) for k2 in gen if gen[k2] != ident[k2]}
            raise CoverageError(f"{cell} differs from the other seeds: {diff}")
        seed = int(spec["seed_index"])
        if seed in seeds:
            raise CoverageError(f"seed index {seed} appears twice ({seeds[seed]['cell']} and {cell})")
        run_dir = cell / label
        reqs = {r["request_id"]: r for r in _jsonl(run_dir / "requests.jsonl")}
        traces = {t["request_id"]: t for t in _jsonl(run_dir / "traces.jsonl")}
        res = json.loads((cell / "result.json").read_text())
        for rid in spec["measured_ids"]:
            r = reqs.get(rid)
            if r is None or not r["ok"]:
                raise CoverageError(f"{cell}/{label}: request {rid} missing or invalid")
            t = traces.get(rid)
            if t is None or "text" not in t:
                raise CoverageError(f"{cell}/{label}: trace text for {rid} missing")
            wav = run_dir / "wav" / f"{rid}.wav"
            try:
                info = sf.info(str(wav))
            except Exception:
                raise CoverageError(f"{cell}/{label}: audio for {rid} missing") from None
            if info.samplerate != 24000 or abs(info.frames / info.samplerate - r["duration_s"]) > 1.0 / 24000 + 1e-9:
                raise CoverageError(f"{cell}/{label}: audio for {rid} does not match its recorded duration/rate")
            row = meta[rid]
            rows.append(GenRow(f"{seed}:{rid}", rid, seed, row.target_text, str(Path(row.ref_wav_path).resolve()), str(wav), t["text"], r["duration_s"]))
        seeds[seed] = {"cell": str(cell), "run": label, "experiment_digest": res["experiment_digest"],
                       "artifacts": res["artifacts"][label], "rows": len(spec["measured_ids"]),
                       "gpu_uuid": spec["hardware"].get("gpu_uuid")}
    if gen is None:
        raise CoverageError("no source cells")
    return Collection(rows=rows, generation=gen, seeds={str(s): seeds[s] for s in sorted(seeds)})


def rows_sha256(rows: list[GenRow]) -> str:
    return hashlib.sha256("\n".join(sorted(r.key for r in rows)).encode()).hexdigest()


def _env() -> dict:
    return {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "HF_HUB_CACHE": str(paths.HF_HUB_CACHE), "HF_HOME": str(paths.HF_HOME), "HF_DATASETS_CACHE": str(paths.HF_DATASETS_CACHE),
            "HF_MODULES_CACHE": str(paths.HF_MODULES_CACHE), "HF_HUB_OFFLINE": "1"}


def _run_worker(python: str, script: str, job: dict, gpu: int, cwd: Path) -> dict:
    with tempfile.TemporaryDirectory() as td:
        jp, op = Path(td) / "job.json", Path(td) / "out.json"
        jp.write_text(json.dumps(job))
        subprocess.run([python, str(ROOT / "evals" / "speech" / script), str(jp), str(op)], cwd=str(cwd), env={**_env(), "CUDA_VISIBLE_DEVICES": str(gpu)},
                       check=True)
        return json.loads(op.read_text())


def score(rows: list[GenRow], gpu: int, whisper: bool = True) -> dict:
    """Verify the evaluator, then transcribe (fixed batches) and score every row."""
    fp = verify_evaluator()
    assets = json.loads(ASSETS_LOCK.read_text())
    files = list(assets["files"])
    finetune = next(f for f in files if f.endswith("wavlm_large_finetune.pth"))
    base = next(f for f in files if f.endswith("wavlm_large.pt"))
    utmos = next(f for f in files if f.endswith("utmos.jit"))
    asr_rows = [{"id": r.key, "wav": r.wav} for r in rows]
    asr = _run_worker(ASR_PYTHON, "asr_worker.py", {"rows": asr_rows, "model": "qwen3-asr", "model_path": assets["models"]["qwen3-asr"]["dir"],
                                                    "batch_size": SETTINGS["asr"]["batch_size"]}, gpu, ROOT)
    wsp = None
    if whisper:
        wsp = _run_worker(ASR_PYTHON, "asr_worker.py", {"rows": asr_rows, "model": "whisper", "model_path": assets["models"]["whisper"]["dir"],
                                                        "batch_size": SETTINGS["whisper"]["batch_size"]}, gpu, ROOT)
    job_rows = [{"id": r.key, "target_text": r.target_text, "wav": r.wav, "ref_wav": r.ref_wav, "hyp_text": asr["texts"][r.key],
                 "whisper_text": (wsp["texts"][r.key] if wsp else "")} for r in rows]
    tasks = ["wer", "sim", "utmos"] + (["whisper_wer"] if whisper else [])
    scored = _run_worker(EVAL_PYTHON, "scorer_worker.py", {"rows": job_rows, "tasks": tasks, "device": "cuda:0", "wavlm_finetune": finetune,
                                                          "wavlm_base": base, "utmos_jit": utmos, "sim_batch": SETTINGS["sim"]["batch"],
                                                          "sim_attention": SETTINGS["sim"]["attention"]}, gpu, SGLANG_DIR)
    sim = scored["tasks"]["sim"]
    if (sim.get("attention"), sim.get("query_block"), sim.get("batch")) != (SETTINGS["sim"]["attention"], SETTINGS["sim"]["query_block"], SETTINGS["sim"]["batch"]):
        raise RuntimeError(f"SIM execution evidence contradicts the declared evaluator settings: {sim.get('attention')}/{sim.get('query_block')}")
    evidence = {"asr": asr["evidence"], "whisper": wsp["evidence"] if wsp else None}
    for name, ev in evidence.items():
        if ev is None:
            continue
        b = SETTINGS[name]["batch_size"]
        if any(s != b for s in ev["batch_sizes"]) or "bfloat16" not in ev["settings"]["dtype"] or ev["settings"]["do_sample"] is not False:
            raise RuntimeError(f"{name} execution evidence contradicts the declared evaluator settings: {ev}")
    return {"evaluator": fp, "execution_evidence": evidence, "tasks": scored["tasks"]}


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


def check_scope(scope: str, col: Collection) -> None:
    if scope not in SCOPES:
        raise ValueError(f"scope must be one of {SCOPES}")
    if scope == "final":
        if col.generation["id_set"] != "full":
            raise CoverageError("final-scope quality needs the full 1088-row set")
        if len(col.seeds) != 3 or any(s["rows"] != FULL_ROWS for s in col.seeds.values()):
            raise CoverageError("final-scope quality needs exactly three seeds of all 1088 rows")


def write_report(out: Path, scope: str, col: Collection, scored: dict, quality: list[RowQuality], d_ref_path: str | None, t0: float) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    sidecar = out.with_suffix(".rows.jsonl")
    with sidecar.open("w") as f:
        for q in quality:
            f.write(json.dumps(q.to_json()) + "\n")
    rep = {
        "scope": scope,
        "generation": col.generation,
        "seeds": col.seeds,
        "rows_per_seed": {str(s): sum(1 for q in quality if q.seed_index == s) for s in sorted({q.seed_index for q in quality})},
        "rows_sha256": rows_sha256([GenRow(f"{q.seed_index}:{q.row_id}", q.row_id, q.seed_index, "", "", "", "", 0.0) for q in quality]),
        "n_rows": len(quality),
        "evaluator": scored["evaluator"],
        "execution_evidence": scored["execution_evidence"],
        "d_ref": {"path": d_ref_path, "sha256": hashlib.sha256(Path(d_ref_path).read_bytes()).hexdigest()} if d_ref_path else None,
        "truncation_evaluated": all(q.truncation is not None for q in quality),
        "row_sidecar_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(),
        "summary": summarize(quality),
        "wall_s": time.time() - t0,
    }
    out.write_text(json.dumps(rep, indent=1))
    return rep


def load_dref(path: str, col: Collection) -> dict[str, float]:
    """The verified per-row map of a d_ref/v1 truncation reference that matches this collection."""
    from evals.speech.verify_report import verify_dref

    g = col.generation
    expect = {"role": "truncation", "concurrency": g["concurrency"], "gpu_class": {k: g[k] for k in ("gpu_sku", "gpu_driver", "gpu_memory_mib")},
              "model": g["model"], "dataset": g["dataset"], "seed_policy": g["seed_policy"], "rows": sorted({r.row_id for r in col.rows})}
    problems, rows = verify_dref(path, expect)
    if problems:
        raise CoverageError(f"duration reference {path} does not verify: {problems[:4]}")
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.speech.suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("score")
    sp.add_argument("--cell", action="append", required=True, help="verified measured cell directory (one per seed)")
    sp.add_argument("--run", default=None, help="measured run label to score (default run1 for K=1 cells)")
    sp.add_argument("--scope", choices=SCOPES, required=True)
    sp.add_argument("--gpu", type=int, required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--d-ref", default=None, help="d_ref/v1 truncation reference built from the frozen throughput winner at this C")
    sp.add_argument("--no-whisper", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    col = collect([Path(c) for c in a.cell], a.run)
    check_scope(a.scope, col)
    d_ref = load_dref(a.d_ref, col) if a.d_ref else None
    scored = score(col.rows, a.gpu, whisper=not a.no_whisper)
    quality = build_quality(col.rows, scored, d_ref)
    rep = write_report(Path(a.out), a.scope, col, scored, quality, a.d_ref, t0)
    s = rep["summary"]["pooled"]
    print(json.dumps({k: s[k] for k in ("rows", "wer_corpus", "wer_filtered_corpus", "catastrophic_count", "sim_mean", "utmos_mean",
                                         "silence_rate", "repetition_rate", "thinker_exact_match_rate")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
