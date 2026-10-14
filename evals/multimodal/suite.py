"""Multimodal accuracy suite: MMMU (val), MMSU, Daily-Omni.

    python -m evals.multimodal.suite run --adapter vllm-omni --config minicpmo_4_5_turn_mrv2.yaml --gpu 1 --port 18101 \
        --bench mmmu --bench mmsu --bench daily_omni --subset full|dev --scope development|final --out results/multimodal/<name>
    python -m evals.multimodal.suite freeze --out qref_mm.json --reference REPORT.json [...]
    python -m evals.multimodal.suite calibrate --report REPORT.json --out shuffled.json

``run`` verifies the evaluator, launches a fresh server for the candidate
(through the harness, with its runtime evidence), runs each benchmark worker
with the pinned dataset loaders and scorers, and requires full coverage: every
declared sample answered without a request failure.  The report binds the
candidate, GPU, server evidence, dataset revisions, ID sets and evaluator.

Floors: MMMU >= 0.64, MMSU >= 0.533, Daily-Omni >= 0.77, and each >= Q_ref - 1 pp.
"""

from __future__ import annotations

import argparse
import os
import hashlib
import json
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.evaluator import EVAL_PYTHON, SGLANG_DIR, EvaluatorMismatch, verify_evaluator  # noqa: E402

VLLM_PYTHON = "/workspace/cgzheng/cgzheng/vllm-omni/.venv/bin/python"
VLLM_DIR = Path("/workspace/cgzheng/cgzheng/vllm-omni")
VLLM_COMMIT = "2ec5f265c69520cdd5e5f0a19db485d38885843e"
MODEL_PATH = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"
DAILY_ROOT = Path("/workspace/models/hub/datasets--liarliar--Daily-Omni/snapshots/bf5a6ee4c829510b7a14869e3104efeafc330b07")
FLOORS = {"mmmu": 0.64, "mmsu": 0.533, "daily_omni": 0.77}
RELATIVE_MARGIN = 0.01
#: Full deliverable sets and the development subsets used for quick checks.
SETS = {
    "full": {"mmmu": {"repo_id": None, "max_samples": None, "dataset": "MMMU/MMMU validation (900)"},
             "mmsu": {"repo_id": None, "max_samples": None, "dataset": "ddwang2000/MMSU (5000)"},
             "daily_omni": {"max_samples": None, "dataset": "liarliar/Daily-Omni qa.json (1197)"}},
    "dev": {"mmmu": {"repo_id": "zhaochenyang20/mmmu-ci-50", "max_samples": None, "dataset": "zhaochenyang20/mmmu-ci-50"},
            "mmsu": {"repo_id": "zhaochenyang20/mmsu-ci-2000", "max_samples": 400, "dataset": "zhaochenyang20/mmsu-ci-2000 first 400"},
            "daily_omni": {"max_samples": 150, "dataset": "liarliar/Daily-Omni first 150 (seeded order)"}},
}
MODEL_NAME = {"vllm-omni": "openbmb/MiniCPM-o-4_5", "sglang-omni": "minicpmo", "engine": "openbmb/MiniCPM-o-4_5"}
MAX_TOKENS = {"mmmu": 2048, "mmsu": 32}
WORKERS = ("evals/multimodal/sglang_mm_worker.py", "evals/multimodal/daily_omni_worker.py", "evals/multimodal/suite.py")


class CoverageError(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def verify_mm_evaluator() -> dict:
    base = verify_evaluator(check_weights=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=VLLM_DIR, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=VLLM_DIR, capture_output=True, text=True).stdout.strip()
    if commit != VLLM_COMMIT or dirty:
        raise EvaluatorMismatch(f"vllm-omni scorer checkout {commit} dirty={bool(dirty)} != pinned {VLLM_COMMIT}")
    vllm_env = hashlib.sha256(subprocess.run(["uv", "pip", "freeze", "--python", VLLM_PYTHON], capture_output=True, text=True).stdout.encode()).hexdigest()
    assets = json.loads((ROOT / "evals" / "assets.lock.json").read_text())
    datasets = {r: v["snapshots"] for r, v in assets["repos"].items() if v["kind"] == "dataset"}
    rec = {"speech_evaluator_digest": base["digest"], "sglang": base["scorer_checkout"], "eval_env": base["envs"]["eval-env"],
           "vllm_omni": {"commit": commit, "env_freeze_sha256": vllm_env}, "datasets": datasets,
           "workers": {w: _sha(ROOT / w) for w in WORKERS}, "floors": FLOORS, "max_tokens": MAX_TOKENS}
    rec["digest"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    return rec


def _worker(python: str, script: str, job: dict, cwd: Path, gpu: int | None = None) -> dict:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "HF_HUB_CACHE": "/workspace/models/hub", "HF_HOME": os.environ.get("HF_HOME", "/workspace/models"), "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", "/workspace/models/cgzheng/datasets"),
           "HF_MODULES_CACHE": "/workspace/models/cgzheng/hf_modules", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with tempfile.TemporaryDirectory() as td:
        jp, op = Path(td) / "job.json", Path(td) / "out.json"
        jp.write_text(json.dumps(job))
        subprocess.run([python, str(ROOT / script), str(jp), str(op)], cwd=str(cwd), env=env, check=True)
        return json.loads(op.read_text())


def run_bench(bench: str, adapter: str, url: str, subset: str, concurrency: int) -> dict:
    cfg = SETS[subset][bench]
    if bench == "daily_omni":
        job = {"url": url, "model": MODEL_NAME[adapter], "concurrency": concurrency, "max_samples": cfg["max_samples"],
               "video_dir": str(DAILY_ROOT / "Videos"), "qa_json": str(DAILY_ROOT / "qa.json"), "tokenizer": MODEL_PATH}
        out = _worker(VLLM_PYTHON, "evals/multimodal/daily_omni_worker.py", job, ROOT)
        out["accuracy"] = out["summary"].get("daily_omni_accuracy_incl_http_fail")
    else:
        # sglang-omni (like the HF reference chat()) renders audio inputs with use_tts_template=True;
        # other engines get the same template choice explicitly
        kwargs = {"enable_thinking": False, **({"use_tts_template": True} if bench == "mmsu" else {})}
        extra = {} if adapter == "sglang-omni" else {"chat_template_kwargs": kwargs}
        job = {"bench": bench, "repo_id": cfg["repo_id"], "max_samples": cfg["max_samples"], "url": url, "model": MODEL_NAME[adapter],
               "format": "sglang" if adapter == "sglang-omni" else "openai", "concurrency": concurrency, "max_tokens": MAX_TOKENS[bench],
               "extra_body": extra}
        out = _worker(EVAL_PYTHON, "evals/multimodal/sglang_mm_worker.py", job, SGLANG_DIR)
    out["dataset"] = cfg["dataset"]
    return out


def coverage(out: dict) -> dict:
    ids = out["ids"]
    got = [p["id"] for p in out["per_sample"]]
    failed = [p["id"] for p in out["per_sample"] if not p["is_success"]]
    return {"declared": len(ids), "answered": len(got), "failed": len(failed), "failed_examples": failed[:5],
            "complete": got == ids and not failed and len(set(ids)) == len(ids), "ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest()}


def accuracy_from(per: list[dict]) -> float:
    return sum(1 for p in per if p["is_correct"]) / len(per) if per else float("nan")


def gates(report: dict, qref: dict | None) -> dict:
    out = {}
    for bench, r in report["benches"].items():
        acc = r["accuracy"]
        g = {"accuracy": acc, "floor": FLOORS[bench], "floor_passed": acc is not None and acc >= FLOORS[bench], "coverage_complete": r["coverage"]["complete"]}
        if qref is not None:
            ref = qref["values"].get(bench)
            same = ref is not None and qref["identity"][bench] == r["identity"]
            g["q_ref"] = ref
            g["q_ref_comparable"] = same
            g["relative_passed"] = bool(same and acc >= ref - RELATIVE_MARGIN)
        g["passed"] = bool(g["floor_passed"] and g["coverage_complete"] and (qref is None or g.get("relative_passed")))
        out[bench] = g
    return out


def cmd_run(a) -> int:
    from bench.cli import make_candidate, server_spec
    from bench.fingerprint import gpu_identity
    from bench.servers import Server

    evaluator = verify_mm_evaluator()
    cand = make_candidate({"adapter": a.adapter, "config": a.config, "extra": a.extra or [], "label": a.label, "mps": a.mps})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    spec = server_spec(cand, a.port, a.gpu)
    report = {"scope": a.scope, "subset": a.subset, "candidate": {**cand.identity(), "candidate_id": cand.candidate_id, "digest": cand.digest},
              "gpu": gpu_identity(a.gpu), "evaluator": evaluator, "benches": {}}
    t0 = time.time()
    with Server(spec, out / "server.log") as srv:
        report["server_evidence"] = srv.evidence()
        for bench in a.bench:
            res = run_bench(bench, a.adapter, srv.url, a.subset, a.concurrency)
            cov = coverage(res)
            side = out / f"{bench}.per_sample.jsonl"
            side.write_text("".join(json.dumps(p) + "\n" for p in res["per_sample"]))
            report["benches"][bench] = {"accuracy": res["accuracy"], "summary": res["summary"], "dataset": res["dataset"], "coverage": cov,
                                        "identity": {"dataset": res["dataset"], "ids_sha256": cov["ids_sha256"], "evaluator": evaluator["digest"],
                                                     "gpu_sku": report["gpu"].get("gpu_sku"), "subset": a.subset},
                                        "per_sample_sha256": _sha(side), "wall_s": res.get("wall_s")}
            print(json.dumps({bench: {"accuracy": res["accuracy"], "coverage": cov["complete"], "failed": cov["failed"]}}), flush=True)
    report["gates"] = gates(report, json.loads(Path(a.qref).read_text()) if a.qref else None)
    report["wall_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return 0 if all(g["passed"] for g in report["gates"].values()) else 1


def cmd_freeze(a) -> int:
    """Multimodal Q_ref: best accuracy per benchmark among qualified baselines with identical identity."""
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a frozen Q_ref is never rewritten")
    values, identity, sources = {}, {}, {}
    for spec in a.reference:
        path, _, qual_path = spec.partition("=")
        rep = json.loads(Path(path).read_text())
        qual = json.loads(Path(qual_path).read_text())
        if not qual.get("qualified") or qual.get("candidate_digest") != rep["candidate"]["digest"]:
            raise SystemExit(f"{path}: candidate is not a qualified baseline")
        if rep["scope"] != a.scope:
            raise SystemExit(f"{path}: scope {rep['scope']} != {a.scope}")
        for bench, r in rep["benches"].items():
            if not r["coverage"]["complete"]:
                continue  # a baseline that does not support the benchmark does not set its reference
            if bench in identity and identity[bench] != r["identity"]:
                raise SystemExit(f"{bench}: references differ in identity")
            identity[bench] = r["identity"]
            if r["accuracy"] > values.get(bench, -1):
                values[bench], sources[bench] = r["accuracy"], rep["candidate"]["candidate_id"]
    out.write_text(json.dumps({"scope": a.scope, "values": values, "identity": identity, "sources": sources, "frozen_unix": time.time()}, indent=1))
    print(json.dumps(values))
    return 0


def cmd_calibrate(a) -> int:
    """Shuffled-answer negative fixture: derange predictions across samples; the gate must reject it."""
    rep = json.loads(Path(a.report).read_text())
    out = {}
    for bench in rep["benches"]:
        per = [json.loads(line) for line in (Path(a.report).parent / f"{bench}.per_sample.jsonl").read_text().splitlines() if line.strip()]
        rng = random.Random(0)
        preds = [p.get("predicted") for p in per]
        perm = list(range(len(per)))
        while len(per) > 1 and any(i == j for i, j in enumerate(perm)):
            rng.shuffle(perm)
        shuffled = [{**p, "predicted": preds[j], "is_correct": bool(p.get("expected") and preds[j] == p.get("expected"))} for p, j in zip(per, perm)]
        acc = accuracy_from(shuffled)
        out[bench] = {"clean_accuracy": rep["benches"][bench]["accuracy"], "shuffled_accuracy": acc, "floor": FLOORS[bench],
                      "rejected": acc < FLOORS[bench] and acc < rep["benches"][bench]["accuracy"] - RELATIVE_MARGIN}
    out["all_rejected"] = all(v["rejected"] for v in out.values() if isinstance(v, dict))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0 if out["all_rejected"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.multimodal.suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapter", required=True, choices=["vllm-omni", "sglang-omni", "engine"])
    r.add_argument("--config", required=True)
    r.add_argument("--label", default="")
    r.add_argument("--mps", choices=["off", "on"], default="off")
    r.add_argument("--gpu", type=int, required=True)
    r.add_argument("--port", type=int, default=18101)
    r.add_argument("--bench", action="append", choices=list(FLOORS), required=True)
    r.add_argument("--subset", choices=list(SETS), required=True)
    r.add_argument("--scope", choices=["development", "final"], required=True)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--qref", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("extra", nargs="*")
    f = sub.add_parser("freeze")
    f.add_argument("--scope", choices=["development", "final"], required=True)
    f.add_argument("--reference", action="append", required=True, help="REPORT.json=QUALIFICATION.json")
    f.add_argument("--out", required=True)
    c = sub.add_parser("calibrate")
    c.add_argument("--report", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "run":
        if a.scope == "final" and a.subset != "full":
            raise SystemExit("final scope needs the full benchmark sets")
        return cmd_run(a)
    if a.cmd == "freeze":
        return cmd_freeze(a)
    return cmd_calibrate(a)


if __name__ == "__main__":
    raise SystemExit(main())
