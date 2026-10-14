"""Multimodal accuracy suite: MMMU (val), MMSU, Daily-Omni.

    python -m evals.multimodal.suite freeze-expected --subset full|dev
    python -m evals.multimodal.suite run --adapter vllm-omni --config minicpmo_4_5_turn_mrv2.yaml --gpu 1 --port 18101 \
        --subset full|dev --scope development|final --out results/multimodal/<name> [--bench ...]
    python -m evals.multimodal.suite freeze --scope final --out qref_mm.json --reference REPORT_DIR=QUALIFICATION.json [...]
    python -m evals.multimodal.suite certify --report REPORT_DIR --qref qref_mm.json --headline-digest D
    python -m evals.multimodal.suite calibrate --report REPORT_DIR --out results/multimodal/calibration/<name>

Evidence boundaries:
* the expected sample IDs and content hashes of every benchmark are frozen
  (``evals/multimodal/expected/``) before any request is sent; ``run`` checks
  the consumed data against them before launching a server;
* dataset revisions are pinned at load time (see the workers);
* ``verify_mm_report`` re-derives coverage and accuracy from the hash-checked
  sidecars by re-scoring the saved responses with the pinned scorers, and is
  used by ``freeze``, ``certify`` and ``calibrate``;
* ``certify`` is the only final AC-4.5 verdict: final scope, a final Q_ref,
  all three benchmarks, the declared headline candidate and its observed
  runtime MPS/GPU.

Floors: MMMU >= 0.64, MMSU >= 0.533, Daily-Omni >= 0.77, and each >= Q_ref - 1 pp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.evaluator import EVAL_PYTHON, SGLANG_DIR, EvaluatorMismatch, verify_evaluator  # noqa: E402

VLLM_PYTHON = "/workspace/cgzheng/cgzheng/vllm-omni/.venv/bin/python"
VLLM_DIR = Path("/workspace/cgzheng/cgzheng/vllm-omni")
VLLM_COMMIT = "2ec5f265c69520cdd5e5f0a19db485d38885843e"
VLLM_LOCK = ROOT / "evals" / "vllm-omni-env.lock"
MODEL_PATH = "/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582"
DAILY_ROOT = Path("/workspace/models/hub/datasets--liarliar--Daily-Omni/snapshots/bf5a6ee4c829510b7a14869e3104efeafc330b07")
EXPECTED_DIR = ROOT / "evals" / "multimodal" / "expected"
BENCHES = ("mmmu", "mmsu", "daily_omni")
FLOORS = {"mmmu": 0.64, "mmsu": 0.533, "daily_omni": 0.77}
FULL_COUNTS = {"mmmu": 900, "mmsu": 5000, "daily_omni": 1197}
RELATIVE_MARGIN = 0.01
SETS = {
    "full": {"mmmu": {"repo_id": None, "max_samples": None, "dataset": "MMMU/MMMU validation"},
             "mmsu": {"repo_id": None, "max_samples": None, "dataset": "ddwang2000/MMSU"},
             "daily_omni": {"max_samples": None, "dataset": "liarliar/Daily-Omni qa.json"}},
    "dev": {"mmmu": {"repo_id": "zhaochenyang20/mmmu-ci-50", "max_samples": None, "dataset": "zhaochenyang20/mmmu-ci-50"},
            "mmsu": {"repo_id": "zhaochenyang20/mmsu-ci-2000", "max_samples": 400, "dataset": "zhaochenyang20/mmsu-ci-2000 first 400"},
            "daily_omni": {"max_samples": 150, "dataset": "liarliar/Daily-Omni first 150 (seeded order)"}},
}
MODEL_NAME = {"vllm-omni": "openbmb/MiniCPM-o-4_5", "sglang-omni": "minicpmo", "engine": "openbmb/MiniCPM-o-4_5"}
MAX_TOKENS = {"mmmu": 2048, "mmsu": 32}
WORKERS = ("evals/multimodal/sglang_mm_worker.py", "evals/multimodal/daily_omni_worker.py", "evals/multimodal/suite.py")
SHUFFLE_LADDER = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80)


class EvidenceError(RuntimeError):
    pass


def _sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(line) for line in Path(p).read_text().splitlines() if line.strip()]


def verify_mm_evaluator() -> dict:
    base = verify_evaluator(check_weights=False)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=VLLM_DIR, capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=VLLM_DIR, capture_output=True, text=True).stdout.strip()
    if commit != VLLM_COMMIT or dirty:
        raise EvaluatorMismatch(f"vllm-omni scorer checkout {commit} dirty={bool(dirty)} != pinned {VLLM_COMMIT}")
    frozen = subprocess.run(["uv", "pip", "freeze", "--python", VLLM_PYTHON], capture_output=True, text=True).stdout
    if frozen != VLLM_LOCK.read_text():
        raise EvaluatorMismatch("vllm-omni evaluator environment differs from evals/vllm-omni-env.lock")
    rec = {"speech_evaluator_digest": base["digest"], "sglang": base["scorer_checkout"], "eval_env": base["envs"]["eval-env"],
           "vllm_omni": {"commit": commit, "env_freeze_sha256": hashlib.sha256(frozen.encode()).hexdigest()},
           "expected": {p.name: _sha(p) for p in sorted(EXPECTED_DIR.glob("*.json"))},
           "workers": {w: _sha(ROOT / w) for w in WORKERS}, "floors": FLOORS, "max_tokens": MAX_TOKENS}
    rec["digest"] = hashlib.sha256(json.dumps(rec, sort_keys=True).encode()).hexdigest()
    return rec


def _env(gpu: int | None = None) -> dict:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "HF_HUB_CACHE": "/workspace/models/hub", "HF_HOME": os.environ.get("HF_HOME", "/workspace/models"),
           "HF_DATASETS_CACHE": os.environ.get("HF_DATASETS_CACHE", "/workspace/models/cgzheng/datasets"),
           "HF_MODULES_CACHE": "/workspace/models/cgzheng/hf_modules", "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    return env


def _worker(bench: str, job: dict) -> dict:
    python, script, cwd = ((VLLM_PYTHON, "evals/multimodal/daily_omni_worker.py", ROOT) if bench == "daily_omni"
                           else (EVAL_PYTHON, "evals/multimodal/sglang_mm_worker.py", SGLANG_DIR))
    with tempfile.TemporaryDirectory() as td:
        jp, op = Path(td) / "job.json", Path(td) / "out.json"
        jp.write_text(json.dumps(job))
        subprocess.run([python, str(ROOT / script), str(jp), str(op)], cwd=str(cwd), env=_env(), check=True)
        return json.loads(op.read_text())


def _job(bench: str, subset: str, mode: str, **kw) -> dict:
    cfg = SETS[subset][bench]
    if bench == "daily_omni":
        job = {"mode": mode, "max_samples": cfg["max_samples"], "video_dir": str(DAILY_ROOT / "Videos"), "qa_json": str(DAILY_ROOT / "qa.json"),
               "tokenizer": MODEL_PATH}
    else:
        job = {"mode": mode, "bench": bench, "repo_id": cfg["repo_id"], "max_samples": cfg["max_samples"]}
    job.update(kw)
    return job


def expected_path(subset: str, bench: str) -> Path:
    return EXPECTED_DIR / f"{subset}_{bench}.json"


def load_expected(subset: str, bench: str) -> dict:
    p = expected_path(subset, bench)
    if not p.exists():
        raise EvidenceError(f"no frozen expected manifest {p}; run freeze-expected first")
    doc = json.loads(p.read_text())
    ids = [i for i, _ in doc["items"]]
    if not ids or len(set(ids)) != len(ids):
        raise EvidenceError(f"{p} is empty or has duplicate IDs")
    if subset == "full" and len(ids) != FULL_COUNTS[bench]:
        raise EvidenceError(f"{p}: {len(ids)} items, the full {bench} set has {FULL_COUNTS[bench]}")
    return doc


def check_consumed(subset: str, bench: str) -> dict:
    """Load the data now and require it to equal the frozen manifest (IDs and content), before any server starts."""
    exp = load_expected(subset, bench)
    now = _worker(bench, _job(bench, subset, "manifest"))
    if now["items"] != exp["items"]:
        diff = sum(1 for a, b in zip(now["items"], exp["items"]) if a != b) + abs(len(now["items"]) - len(exp["items"]))
        raise EvidenceError(f"{bench}/{subset}: consumed data differs from the frozen manifest in {diff} items")
    return exp


def coverage(ids: list[str], per: list[dict], expected_ids: list[str]) -> dict:
    got = [p["id"] for p in per]
    failed = [p["id"] for p in per if not p["is_success"]]
    complete = bool(expected_ids) and got == expected_ids == ids and not failed and len(set(got)) == len(got)
    return {"declared": len(expected_ids), "answered": len(got), "failed": len(failed), "failed_examples": failed[:5], "complete": complete,
            "ids_sha256": hashlib.sha256("\n".join(expected_ids).encode()).hexdigest()}


def verify_mm_report(report_dir: str | Path, rescore: bool = True) -> tuple[dict, dict, list[str]]:
    """Returns (report, derived per-bench {accuracy, coverage, identity}, problems)."""
    d = Path(report_dir)
    rep = json.loads((d / "report.json").read_text())
    problems, derived = [], {}
    for bench, r in rep.get("benches", {}).items():
        side = d / f"{bench}.per_sample.jsonl"
        if not side.exists() or _sha(side) != r.get("per_sample_sha256"):
            problems.append(f"{bench}: sidecar missing or does not match its recorded hash")
            continue
        try:
            exp = load_expected(rep["subset"], bench)
        except EvidenceError as e:
            problems.append(f"{bench}: {e}")
            continue
        if _sha(expected_path(rep["subset"], bench)) != r.get("expected_sha256"):
            problems.append(f"{bench}: the frozen expected manifest changed since the run")
        per = _jsonl(side)
        exp_ids = [i for i, _ in exp["items"]]
        cov = coverage([p["id"] for p in per], per, exp_ids)
        acc = sum(1 for p in per if p["is_correct"]) / len(per) if per else float("nan")
        if rescore:
            rescored = _worker(bench, _job(bench, rep["subset"], "rescore", responses={p["id"]: p.get("raw") for p in per}))
            if {p["id"]: p["is_correct"] for p in rescored["per_sample"]} != {p["id"]: p["is_correct"] for p in per}:
                problems.append(f"{bench}: per-sample correctness differs from the pinned scorer's re-scoring")
            acc = rescored["accuracy"]
        if r.get("accuracy") != acc:
            problems.append(f"{bench}: recorded accuracy {r.get('accuracy')} != derived {acc}")
        derived[bench] = {"accuracy": acc, "coverage": cov,
                          "identity": {"dataset": SETS[rep["subset"]][bench]["dataset"], "expected_sha256": _sha(expected_path(rep["subset"], bench)),
                                       "ids_sha256": cov["ids_sha256"], "gpu_sku": rep["gpu"].get("gpu_sku"), "subset": rep["subset"],
                                       "scope": rep["scope"], "evaluator": rep["evaluator"]["digest"]}}
    ev = rep.get("server_evidence") or {}
    observed = (ev.get("mps") or {}).get("observed")
    if observed != rep["candidate"]["mps"]:
        problems.append(f"declared MPS {rep['candidate']['mps']} but observed {observed}")
    if (ev.get("gpu") or {}).get("gpu_uuid") != rep["gpu"].get("gpu_uuid"):
        problems.append("server ran on a different GPU than the report declares")
    if not isinstance(rep.get("concurrency"), int):
        problems.append("report does not record its concurrency")
    return rep, derived, problems


def cmd_freeze_expected(a) -> int:
    EXPECTED_DIR.mkdir(parents=True, exist_ok=True)
    for bench in BENCHES:
        p = expected_path(a.subset, bench)
        if p.exists():
            print(f"{p} exists (frozen); not rewritten")
            continue
        m = _worker(bench, _job(bench, a.subset, "manifest"))
        doc = {"subset": a.subset, "bench": bench, "dataset": SETS[a.subset][bench]["dataset"], "items": m["items"],
               "consumed": m.get("consumed"), "qa_json_sha256": m.get("qa_json_sha256"), "frozen_unix": time.time()}
        p.write_text(json.dumps(doc))
        print(json.dumps({bench: len(m["items"])}), flush=True)
    return 0


def cmd_run(a) -> int:
    from bench.cli import make_candidate, server_spec
    from bench.fingerprint import gpu_identity
    from bench.servers import Server

    evaluator = verify_mm_evaluator()
    benches = a.bench or list(BENCHES)
    expected = {b: check_consumed(a.subset, b) for b in benches}  # before any server starts
    cand = make_candidate({"adapter": a.adapter, "config": a.config, "extra": a.extra or [], "label": a.label, "mps": a.mps,
                           "env": dict(kv.split("=", 1) for kv in a.env)})
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"scope": a.scope, "subset": a.subset, "concurrency": a.concurrency,
              "candidate": {**cand.identity(), "candidate_id": cand.candidate_id, "digest": cand.digest},
              "gpu": gpu_identity(a.gpu), "evaluator": evaluator, "benches": {}}
    t0 = time.time()
    with Server(server_spec(cand, a.port, a.gpu), out / "server.log") as srv:
        report["server_evidence"] = srv.evidence()
        for bench in benches:
            extra = {} if a.adapter == "sglang-omni" else {
                "chat_template_kwargs": {"enable_thinking": False, **({"use_tts_template": True} if bench == "mmsu" else {})}}
            job = _job(bench, a.subset, "run", url=srv.url, model=MODEL_NAME[a.adapter], concurrency=a.concurrency,
                       format="sglang" if a.adapter == "sglang-omni" else "openai", max_tokens=MAX_TOKENS.get(bench, 512), extra_body=extra)
            res = _worker(bench, job)
            side = out / f"{bench}.per_sample.jsonl"
            side.write_text("".join(json.dumps(p) + "\n" for p in res["per_sample"]))
            cov = coverage(res["ids"], res["per_sample"], [i for i, _ in expected[bench]["items"]])
            report["benches"][bench] = {"accuracy": res["accuracy"], "summary": res.get("summary"), "coverage": cov, "per_sample_sha256": _sha(side),
                                        "expected_sha256": _sha(expected_path(a.subset, bench)), "consumed": res.get("consumed"), "wall_s": res.get("wall_s")}
            print(json.dumps({bench: {"accuracy": res["accuracy"], "coverage": cov["complete"], "failed": cov["failed"]}}), flush=True)
    report["wall_s"] = time.time() - t0
    (out / "report.json").write_text(json.dumps(report, indent=1))
    _, derived, problems = verify_mm_report(out)
    floors = {b: bool(d["accuracy"] >= FLOORS[b] and d["coverage"]["complete"]) for b, d in derived.items()}
    (out / "verification.json").write_text(json.dumps({"problems": problems, "diagnostic_floors": floors}, indent=1))
    print(json.dumps({"diagnostic_floors": floors, "problems": problems[:5]}))
    return 0 if not problems else 1


def cmd_freeze(a) -> int:
    """Multimodal Q_ref from verified reports of verified, qualified baselines."""
    from evals.speech.qref import QRefError, verify_qualification

    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a frozen Q_ref is never rewritten")
    values, identity, sources, inputs = {}, {}, {}, {}
    for spec in a.reference:
        rdir, _, qual_path = spec.partition("=")
        rep, derived, problems = verify_mm_report(rdir)
        if problems:
            raise SystemExit(f"{rdir} fails verification: {problems[:3]}")
        if rep["scope"] != a.scope:
            raise SystemExit(f"{rdir}: scope {rep['scope']} != {a.scope}")
        try:
            qual = verify_qualification(qual_path)
        except QRefError as e:
            raise SystemExit(str(e)) from e
        if not qual["qualified"] or qual["candidate_digest"] != rep["candidate"]["digest"] or qual["gpu_uuid"] != rep["gpu"].get("gpu_uuid"):
            raise SystemExit(f"{rdir}: not a qualified baseline on this GPU")
        inputs[rep["candidate"]["candidate_id"]] = {"report": str(Path(rdir) / "report.json"), "report_sha256": _sha(Path(rdir) / "report.json"),
                                                    "qualification": qual_path, "qualification_sha256": _sha(qual_path)}
        for bench, d in derived.items():
            if not d["coverage"]["complete"]:
                continue  # a baseline that does not support the benchmark does not set its reference
            if bench in identity and identity[bench] != d["identity"]:
                raise SystemExit(f"{bench}: references differ in identity")
            identity[bench] = d["identity"]
            if d["accuracy"] > values.get(bench, -1):
                values[bench], sources[bench] = d["accuracy"], rep["candidate"]["candidate_id"]
    if a.scope == "final" and set(values) != set(BENCHES):
        raise SystemExit(f"final multimodal Q_ref needs a supporting qualified baseline for every benchmark; have {sorted(values)}")
    if not values:
        raise SystemExit("no supported benchmark: nothing to freeze")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scope": a.scope, "values": values, "identity": identity, "sources": sources, "inputs": inputs,
                               "frozen_unix": time.time()}, indent=1))
    print(json.dumps(values))
    return 0


def certify(report_dir: str, qref_path: str, headline_digest: str | None) -> dict:
    rep, derived, problems = verify_mm_report(report_dir)
    q = json.loads(Path(qref_path).read_text())
    for cid, inp in (q.get("inputs") or {}).items():
        if _sha(inp["report"]) != inp["report_sha256"] or _sha(inp["qualification"]) != inp["qualification_sha256"]:
            problems.append(f"Q_ref source {cid} changed since freezing")
    if rep["scope"] != "final" or q.get("scope") != "final" or rep["subset"] != "full":
        problems.append("certification needs a final full-set report and a final Q_ref")
    if set(derived) != set(BENCHES):
        problems.append(f"certification needs all benchmarks; report has {sorted(derived)}")
    if headline_digest is None or rep["candidate"]["digest"] != headline_digest:
        problems.append("report is not the declared headline configuration")
    verdicts = {}
    for bench in BENCHES:
        d = derived.get(bench)
        ref = (q.get("values") or {}).get(bench)
        if d is None or ref is None:
            verdicts[bench] = {"passed": False, "reason": "missing evidence or reference"}
            continue
        same = (q.get("identity") or {}).get(bench) == d["identity"]
        ok = d["coverage"]["complete"] and same and d["accuracy"] >= FLOORS[bench] and d["accuracy"] >= ref - RELATIVE_MARGIN
        verdicts[bench] = {"passed": bool(ok), "accuracy": d["accuracy"], "floor": FLOORS[bench], "q_ref": ref, "identity_matches": same,
                           "coverage_complete": d["coverage"]["complete"]}
    certified = not problems and all(v["passed"] for v in verdicts.values())
    return {"certified": certified, "problems": problems, "benches": verdicts}


def cmd_certify(a) -> int:
    res = certify(a.report, a.qref, a.headline_digest)
    print(json.dumps(res, indent=1))
    return 0 if res["certified"] else 1


def cmd_calibrate(a) -> int:
    """Shuffled-answer negative fixture, calibrated against the measured noise of the clean outputs.

    Noise: bootstrap standard deviation of each benchmark's accuracy over its
    fixed clean per-sample correctness.  For each fraction on the ladder, that
    fraction of samples (fixed seed) has its responses deranged among
    themselves; the fixture is re-scored with the pinned scorer.  The smallest
    fraction whose accuracy drop exceeds the 1 pp margin by at least twice the
    noise is selected, and the production comparison (Q_ref = clean accuracy)
    must reject it.
    """
    rep, derived, problems = verify_mm_report(a.report)
    if problems:
        raise SystemExit(f"clean report fails verification: {problems[:3]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"source": {"report": str(Path(a.report) / "report.json"), "report_sha256": _sha(Path(a.report) / "report.json"),
                         "evaluator": rep["evaluator"]["digest"]}, "benches": {}}
    for bench, d in derived.items():
        per = _jsonl(Path(a.report) / f"{bench}.per_sample.jsonl")
        correct = np.array([1.0 if p["is_correct"] else 0.0 for p in per])
        rng = np.random.default_rng(0)
        noise = float(np.std([correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(1000)]))
        need = RELATIVE_MARGIN + 2 * noise
        ladder, selected = [], None
        for frac in SHUFFLE_LADDER:
            seed = 1000 + int(frac * 100)
            r = random.Random(seed)
            idx = sorted(r.sample(range(len(per)), max(2, int(round(frac * len(per))))))
            raws = [per[i].get("raw") for i in idx]
            shuffled = {p["id"]: p.get("raw") for p in per}
            for j, i in enumerate(idx):
                shuffled[per[i]["id"]] = raws[(j + 1) % len(raws)]  # cyclic derangement within the chosen subset
            res = _worker(bench, _job(bench, rep["subset"], "rescore", responses=shuffled))
            drop = d["accuracy"] - res["accuracy"]
            entry = {"fraction": frac, "seed": seed, "accuracy": res["accuracy"], "drop": drop, "required_drop": need,
                     "gate_rejects": bool(res["accuracy"] < d["accuracy"] - RELATIVE_MARGIN)}  # production relative gate, Q_ref = clean
            ladder.append(entry)
            if drop >= need:
                selected = entry
                side = out / f"{bench}.shuffled.per_sample.jsonl"
                side.write_text("".join(json.dumps(p) + "\n" for p in res["per_sample"]))
                entry["fixture"] = str(side)
                entry["fixture_sha256"] = _sha(side)
                break
        result["benches"][bench] = {"clean_accuracy": d["accuracy"], "noise": noise, "required_drop": need, "ladder": ladder, "selected": selected,
                                    "rejected": bool(selected and selected["gate_rejects"])}
    result["all_rejected"] = bool(result["benches"]) and all(v["rejected"] for v in result["benches"].values())
    (out / "calibration.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({b: {"selected": v["selected"] and v["selected"]["fraction"], "noise": v["noise"], "rejected": v["rejected"]}
                      for b, v in result["benches"].items()}, indent=1))
    return 0 if result["all_rejected"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.multimodal.suite")
    sub = ap.add_subparsers(dest="cmd", required=True)
    fe = sub.add_parser("freeze-expected")
    fe.add_argument("--subset", choices=list(SETS), required=True)
    r = sub.add_parser("run")
    r.add_argument("--adapter", required=True, choices=["vllm-omni", "sglang-omni", "engine"])
    r.add_argument("--config", required=True)
    r.add_argument("--label", default="")
    r.add_argument("--mps", choices=["off", "on"], default="off")
    r.add_argument("--env", action="append", default=[])
    r.add_argument("--gpu", type=int, required=True)
    r.add_argument("--port", type=int, default=18101)
    r.add_argument("--bench", action="append", choices=list(BENCHES), default=None)
    r.add_argument("--subset", choices=list(SETS), required=True)
    r.add_argument("--scope", choices=["development", "final"], required=True)
    r.add_argument("--concurrency", type=int, default=16)
    r.add_argument("--out", required=True)
    r.add_argument("extra", nargs="*")
    f = sub.add_parser("freeze")
    f.add_argument("--scope", choices=["development", "final"], required=True)
    f.add_argument("--reference", action="append", required=True, help="REPORT_DIR=QUALIFICATION.json")
    f.add_argument("--out", required=True)
    c = sub.add_parser("certify")
    c.add_argument("--report", required=True)
    c.add_argument("--qref", required=True)
    c.add_argument("--headline-digest", default=None)
    cal = sub.add_parser("calibrate")
    cal.add_argument("--report", required=True)
    cal.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "freeze-expected":
        return cmd_freeze_expected(a)
    if a.cmd == "run":
        if a.scope == "final" and (a.subset != "full" or (a.bench and set(a.bench) != set(BENCHES))):
            raise SystemExit("final scope runs every benchmark on the full sets")
        return cmd_run(a)
    if a.cmd == "freeze":
        return cmd_freeze(a)
    if a.cmd == "certify":
        return cmd_certify(a)
    return cmd_calibrate(a)


if __name__ == "__main__":
    raise SystemExit(main())
