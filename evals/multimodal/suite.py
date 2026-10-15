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
* ``run`` freezes a generation declaration (candidate, model, code and
  environment fingerprints, request policy, hardware, evaluator) before the
  server starts;
* ``verify_mm_report`` recomputes that declaration's digests and reconciles it
  with the report and the observed launch, GPU (SKU, driver, memory, UUID) and
  MPS state; it requires the current pinned evaluator to be the recorded one,
  checks the consumed data content against the frozen manifest, and re-derives
  coverage and accuracy by re-scoring the saved responses with the pinned
  scorers.  It is used by ``freeze``, ``certify`` and ``calibrate``;
* ``verify_mm_qref`` rebuilds a frozen Q_ref from its recorded references; a
  final Q_ref needs full-set coverage of every benchmark;
* ``certify`` is the only final multimodal verdict: a verified final Q_ref frozen
  before the report was generated, a final full-set report of all three
  benchmarks, the declared headline candidate, and ``compare_bench`` (the
  production comparison, also used by ``calibrate``) for every benchmark.

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
#: Scoring code: the workers load data, send requests and grade answers; floors and request settings are recorded separately.
WORKERS = ("evals/multimodal/sglang_mm_worker.py", "evals/multimodal/daily_omni_worker.py")
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


def evaluator_consistent(ev: dict) -> bool:
    body = {k: v for k, v in ev.items() if k != "digest"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() == ev.get("digest")


def request_policy(adapter: str, bench: str, concurrency: int) -> dict:
    """Everything a multimodal request depends on besides the sample: model name, request form, sampling, extra body."""
    extra = {} if adapter == "sglang-omni" else {"chat_template_kwargs": {"enable_thinking": False, **({"use_tts_template": True} if bench == "mmsu" else {})}}
    return {"model": MODEL_NAME[adapter], "concurrency": concurrency, "format": "sglang" if adapter == "sglang-omni" else "openai",
            "max_tokens": MAX_TOKENS.get(bench, 512), "extra_body": extra}


def generation_identity(cand, gpu: dict, benches: list[str], concurrency: int, port: int, evaluator_digest: str) -> dict:
    """The frozen multimodal generation declaration: candidate, model, code/environment fingerprints, request policy, hardware."""
    from bench.cli import ADAPTER_PYTHON, DEFAULT_REPOS, execution_fingerprints
    from bench.workload import MODEL_REPO, MODEL_REVISION

    doc = {"candidate": {**cand.identity(), "candidate_id": cand.candidate_id, "digest": cand.digest, "label": cand.label},
           "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION}, "fingerprints": execution_fingerprints(cand.adapter),
           "adapter_python": ADAPTER_PYTHON[cand.adapter], "repos_checked": sorted(DEFAULT_REPOS), "port": port,
           "hardware": {k: gpu.get(k) for k in ("gpu_sku", "gpu_uuid", "driver", "memory_mib", "gpu_index")},
           "requests": {b: request_policy(cand.adapter, b, concurrency) for b in benches}, "evaluator_digest": evaluator_digest}
    doc["digest"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc


def generation_problems(rep: dict) -> list[str]:
    """Recompute the generation identity's digests and reconcile it with the report and the observed server."""
    from bench.cli import server_spec
    from bench.experiment import Candidate
    from evals.speech.verify_report import candidate_problems

    gen = rep.get("generation")
    if not gen:
        return ["report has no frozen generation declaration"]
    out = []
    body = {k: v for k, v in gen.items() if k != "digest"}
    if hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest() != gen.get("digest"):
        out.append("generation declaration does not match its digest")
    c = gen["candidate"]
    out += [f"candidate: {p}" for p in candidate_problems(c)]
    if rep.get("candidate") != c:
        out.append("report candidate differs from the generation declaration")
    cand = Candidate(adapter=c["adapter"], config=c["config"], extra=tuple(c["extra"]), env=tuple(tuple(x) for x in c["env"]), label=c.get("label", ""),
                     deploy_files_sha256=tuple(c["deploy_files_sha256"]), mps=c["mps"])
    ev = rep.get("server_evidence") or {}
    hw = gen.get("hardware") or {}
    if ev.get("launch_argv") != server_spec(cand, gen["port"], hw.get("gpu_index")).cmd:
        out.append("observed launch arguments are not the declared candidate's")
    if any((ev.get("launch_env") or {}).get(k) != v for k, v in c["env"]):
        out.append("observed launch environment lacks the declared candidate environment")
    for k in ("gpu_sku", "gpu_uuid", "driver", "memory_mib"):
        if (ev.get("gpu") or {}).get(k) != hw.get(k) or (rep.get("gpu") or {}).get(k) != hw.get(k):
            out.append(f"GPU {k}: declared {hw.get(k)!r}, report {(rep.get('gpu') or {}).get(k)!r}, observed {(ev.get('gpu') or {}).get(k)!r}")
    for bench in rep.get("benches", {}):
        if (gen.get("requests") or {}).get(bench) != request_policy(c["adapter"], bench, rep.get("concurrency")):
            out.append(f"{bench}: recorded request policy is not the adapter's policy at C={rep.get('concurrency')}")
    if gen.get("evaluator_digest") != rep["evaluator"].get("digest"):
        out.append("generation evaluator differs from the report evaluator")
    return out


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
    if not evaluator_consistent(rep.get("evaluator") or {}):
        problems.append("evaluator record does not match its digest")
    problems += generation_problems(rep)
    if rescore:
        try:  # the rescorer must be the pinned evaluator the report recorded (sources, environments, manifests, workers)
            if verify_mm_evaluator()["digest"] != rep["evaluator"].get("digest"):
                problems.append("the current pinned evaluator is not the evaluator the report recorded")
        except EvaluatorMismatch as e:
            problems.append(f"evaluator: {e}")
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
            try:
                check_consumed(rep["subset"], bench)  # the rescorer's data is the frozen content, not just the same IDs
            except EvidenceError as e:
                problems.append(f"{bench}: {e}")
            rescored = _worker(bench, _job(bench, rep["subset"], "rescore", responses={p["id"]: p.get("raw") for p in per}))
            if {p["id"]: p["is_correct"] for p in rescored["per_sample"]} != {p["id"]: p["is_correct"] for p in per}:
                problems.append(f"{bench}: per-sample correctness differs from the pinned scorer's re-scoring")
            acc = rescored["accuracy"]
        if r.get("accuracy") != acc:
            problems.append(f"{bench}: recorded accuracy {r.get('accuracy')} != derived {acc}")
        derived[bench] = {"accuracy": acc, "coverage": cov,
                          "identity": {"dataset": SETS[rep["subset"]][bench]["dataset"], "expected_sha256": _sha(expected_path(rep["subset"], bench)),
                                       "ids_sha256": cov["ids_sha256"], "gpu_sku": rep["gpu"].get("gpu_sku"), "gpu_driver": rep["gpu"].get("driver"),
                                       "gpu_memory_mib": rep["gpu"].get("memory_mib"), "subset": rep["subset"],
                                       "scope": rep["scope"], "evaluator": rep["evaluator"]["digest"], "model": (rep.get("generation") or {}).get("model")}}
    ev = rep.get("server_evidence") or {}
    observed = (ev.get("mps") or {}).get("observed")
    if observed != rep["candidate"]["mps"]:
        problems.append(f"declared MPS {rep['candidate']['mps']} but observed {observed}")
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
    gpu = gpu_identity(a.gpu)
    gen = generation_identity(cand, gpu, benches, a.concurrency, a.port, evaluator["digest"])
    (out / "generation.json").write_text(json.dumps(gen, indent=1))  # frozen before the server starts
    report = {"scope": a.scope, "subset": a.subset, "concurrency": a.concurrency, "candidate": gen["candidate"], "generation": gen,
              "gpu": gpu, "evaluator": evaluator, "started_unix": time.time(), "benches": {}}
    t0 = time.time()
    with Server(server_spec(cand, a.port, a.gpu), out / "server.log") as srv:
        report["server_evidence"] = srv.evidence()
        for bench in benches:
            pol = gen["requests"][bench]
            job = _job(bench, a.subset, "run", url=srv.url, model=pol["model"], concurrency=pol["concurrency"], format=pol["format"],
                       max_tokens=pol["max_tokens"], extra_body=pol["extra_body"])
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


def compare_bench(bench: str, d: dict, ref_value: float, ref_identity: dict) -> dict:
    """The production multimodal comparison of one benchmark; every check keeps its own verdict."""
    checks = {"coverage": bool(d["coverage"]["complete"]), "identity": d["identity"] == ref_identity, "floor": bool(d["accuracy"] >= FLOORS[bench]),
              "relative": bool(d["accuracy"] >= ref_value - RELATIVE_MARGIN)}
    return {"passed": all(checks.values()), "checks": checks, "failed": [k for k, v in checks.items() if not v], "accuracy": d["accuracy"],
            "q_ref": ref_value, "floor": FLOORS[bench], "margin": RELATIVE_MARGIN}


MM_QREF_DECISIVE = ("scope", "values", "identity", "sources", "references")


def build_mm_qref(scope: str, refs: list[tuple[str, str]]) -> dict:
    """Multimodal Q_ref from verified reports of verified, qualified baselines (every benchmark's best supporting baseline)."""
    from evals.speech.qref import QRefError, verify_qualification

    if not refs:
        raise EvidenceError("no references")
    values, identity, sources, inputs = {}, {}, {}, {}
    for rdir, qual_path in refs:
        rep, derived, problems = verify_mm_report(rdir)
        if problems:
            raise EvidenceError(f"{rdir} fails verification: {problems[:3]}")
        if rep["scope"] != scope:
            raise EvidenceError(f"{rdir}: scope {rep['scope']} != {scope}")
        if scope == "final" and rep["subset"] != "full":
            raise EvidenceError(f"{rdir}: a final multimodal Q_ref is frozen from the full sets, not {rep['subset']}")
        try:
            qual = verify_qualification(qual_path)
        except QRefError as e:
            raise EvidenceError(str(e)) from e
        g = rep["gpu"]
        if not qual["qualified"] or qual["candidate_digest"] != rep["candidate"]["digest"] or \
                (qual["gpu_sku"], qual["gpu_driver"], qual.get("gpu_memory_mib")) != (g.get("gpu_sku"), g.get("driver"), g.get("memory_mib")):
            raise EvidenceError(f"{rdir}: not a qualified baseline on this GPU class")
        cid = rep["candidate"]["candidate_id"]
        if cid in inputs:
            raise EvidenceError(f"{cid} is given twice")
        inputs[cid] = {"report": str(Path(rdir) / "report.json"), "report_sha256": _sha(Path(rdir) / "report.json"),
                       "sidecars": {b: _sha(Path(rdir) / f"{b}.per_sample.jsonl") for b in rep["benches"]},
                       "qualification": qual_path, "qualification_sha256": _sha(qual_path)}
        for bench, d in derived.items():
            if not d["coverage"]["complete"]:
                continue  # a baseline that does not support the benchmark does not set its reference
            if scope == "final" and d["coverage"]["declared"] != FULL_COUNTS[bench]:
                raise EvidenceError(f"{rdir}: {bench} covers {d['coverage']['declared']} of {FULL_COUNTS[bench]} samples")
            if bench in identity and identity[bench] != d["identity"]:
                raise EvidenceError(f"{bench}: references differ in identity")
            identity[bench] = d["identity"]
            if d["accuracy"] > values.get(bench, -1) or (d["accuracy"] == values[bench] and cid < sources[bench]):
                values[bench], sources[bench] = d["accuracy"], cid
    if scope == "final" and set(values) != set(BENCHES):
        raise EvidenceError(f"final multimodal Q_ref needs a supporting qualified baseline for every benchmark; have {sorted(values)}")
    if not values:
        raise EvidenceError("no supported benchmark: nothing to freeze")
    return {"scope": scope, "values": values, "identity": identity, "sources": sources, "references": inputs}


def verify_mm_qref(path: str | Path) -> dict:
    """Rebuild the multimodal Q_ref from its recorded references (re-verified and re-scored) and compare every decisive field."""
    q = json.loads(Path(path).read_text())
    refs = q.get("references") or {}
    if not refs:
        raise EvidenceError(f"{path}: Q_ref records no references")
    for cid, inp in refs.items():
        if not Path(inp["report"]).exists() or _sha(inp["report"]) != inp["report_sha256"] or _sha(inp["qualification"]) != inp["qualification_sha256"]:
            raise EvidenceError(f"{path}: reference {cid} changed since freezing")
    again = json.loads(json.dumps(build_mm_qref(q.get("scope"), [(str(Path(v["report"]).parent), v["qualification"]) for v in refs.values()])))
    bad = [k for k in MM_QREF_DECISIVE if again.get(k) != q.get(k)]
    if bad:
        raise EvidenceError(f"{path}: recorded {bad} differ from the Q_ref rebuilt from its references")
    if not isinstance(q.get("frozen_unix"), (int, float)):
        raise EvidenceError(f"{path}: no freeze time")
    return q


def cmd_freeze(a) -> int:
    out = Path(a.out)
    if out.exists():
        raise SystemExit(f"{out} exists: a frozen Q_ref is never rewritten")
    try:
        doc = build_mm_qref(a.scope, [tuple(r.split("=", 1)) for r in a.reference])
    except EvidenceError as e:
        raise SystemExit(str(e)) from e
    doc["frozen_unix"] = time.time()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    print(json.dumps(doc["values"]))
    return 0


def certify(report_dir: str, qref_path: str, headline_digest: str | None) -> dict:
    rep, derived, problems = verify_mm_report(report_dir)
    try:
        q = verify_mm_qref(qref_path)
    except EvidenceError as e:
        return {"certified": False, "problems": problems + [f"Q_ref: {e}"], "benches": {}}
    if rep["scope"] != "final" or q.get("scope") != "final" or rep["subset"] != "full":
        problems.append("certification needs a final full-set report and a final Q_ref")
    if set(derived) != set(BENCHES):
        problems.append(f"certification needs all benchmarks; report has {sorted(derived)}")
    if headline_digest is None or rep["candidate"]["digest"] != headline_digest:
        problems.append("report is not the declared headline configuration")
    if not isinstance(rep.get("started_unix"), (int, float)) or q["frozen_unix"] >= rep["started_unix"]:
        problems.append("the multimodal Q_ref was not frozen before the report was generated")
    verdicts = {}
    for bench in BENCHES:
        d, ref = derived.get(bench), q["values"].get(bench)
        verdicts[bench] = compare_bench(bench, d, ref, q["identity"].get(bench)) if d is not None and ref is not None else \
            {"passed": False, "failed": ["missing evidence or reference"]}
    certified = not problems and all(v["passed"] for v in verdicts.values())
    return {"certified": certified, "problems": problems, "benches": verdicts}


def cmd_certify(a) -> int:
    res = certify(a.report, a.qref, a.headline_digest)
    print(json.dumps(res, indent=1))
    return 0 if res["certified"] else 1


def _write_fixture(clean_dir: Path, rep: dict, bench: str, per: list[dict], out: Path) -> Path:
    """A report directory holding only ``bench`` with the given responses, scored by the pinned scorer."""
    out.mkdir(parents=True, exist_ok=True)
    res = _worker(bench, _job(bench, rep["subset"], "rescore", responses={p["id"]: p.get("raw") for p in per}))
    correct = {p["id"]: p["is_correct"] for p in res["per_sample"]}
    side = out / f"{bench}.per_sample.jsonl"
    side.write_text("".join(json.dumps({**p, "is_correct": correct[p["id"]]}) + "\n" for p in per))
    fixture = json.loads(json.dumps(rep))
    fixture["benches"] = {bench: {**rep["benches"][bench], "accuracy": res["accuracy"], "per_sample_sha256": _sha(side)}}
    fixture["fixture_of"] = {"report": str(clean_dir / "report.json"), "report_sha256": _sha(clean_dir / "report.json")}
    (out / "report.json").write_text(json.dumps(fixture, indent=1))
    return out


def cmd_calibrate(a) -> int:
    """Shuffled-answer negative fixture, calibrated against the measured noise of the clean outputs.

    Noise: bootstrap standard deviation of each benchmark's accuracy over its
    fixed clean per-sample correctness.  For each fraction on the ladder, that
    fraction of samples (fixed seed) has its responses deranged among
    themselves and is re-scored with the pinned scorer.  The smallest fraction
    whose accuracy drop exceeds the 1 pp margin by at least twice the noise is
    selected and saved as a report directory.  That fixture is then reloaded
    through ``verify_mm_report`` (re-scored) and judged by the production
    comparison ``compare_bench`` against the clean accuracy as Q_ref: the clean
    control must pass the relative check and the fixture must fail it, with
    coverage and identity unchanged from the clean control (the injected drop
    may also cross the absolute floor).
    """
    clean_dir = Path(a.report)
    rep, derived, problems = verify_mm_report(clean_dir)
    if problems:
        raise SystemExit(f"clean report fails verification: {problems[:3]}")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    result = {"source": {"report": str(clean_dir / "report.json"), "report_sha256": _sha(clean_dir / "report.json"),
                         "evaluator": rep["evaluator"]["digest"]}, "benches": {}}
    for bench, d in derived.items():
        per = _jsonl(clean_dir / f"{bench}.per_sample.jsonl")
        correct = np.array([1.0 if p["is_correct"] else 0.0 for p in per])
        rng = np.random.default_rng(0)
        noise = float(np.std([correct[rng.integers(0, len(correct), len(correct))].mean() for _ in range(1000)]))
        need = RELATIVE_MARGIN + 2 * noise
        control = compare_bench(bench, d, d["accuracy"], d["identity"])
        ladder, selected = [], None
        for frac in SHUFFLE_LADDER:
            seed = 1000 + int(frac * 100)
            r = random.Random(seed)
            idx = sorted(r.sample(range(len(per)), max(2, int(round(frac * len(per))))))
            raws = [per[i].get("raw") for i in idx]
            shuffled = [dict(p) for p in per]
            for j, i in enumerate(idx):
                shuffled[i]["raw"] = raws[(j + 1) % len(raws)]  # cyclic derangement within the chosen subset
            res = _worker(bench, _job(bench, rep["subset"], "rescore", responses={p["id"]: p.get("raw") for p in shuffled}))
            entry = {"fraction": frac, "seed": seed, "accuracy": res["accuracy"], "drop": d["accuracy"] - res["accuracy"], "required_drop": need}
            ladder.append(entry)
            if entry["drop"] >= need:
                fdir = _write_fixture(clean_dir, rep, bench, shuffled, out / f"{bench}_fixture")
                frep, fderived, fproblems = verify_mm_report(fdir)
                verdict = compare_bench(bench, fderived[bench], d["accuracy"], d["identity"]) if bench in fderived else {"failed": ["unverified"]}
                selected = {**entry, "fixture": str(fdir), "fixture_report_sha256": _sha(fdir / "report.json"), "fixture_problems": fproblems[:3],
                            "verdict": verdict}
                break
        rejected = bool(selected and not selected["fixture_problems"] and control["checks"]["relative"]
                        and selected["verdict"]["failed"] and "relative" in selected["verdict"]["failed"]
                        and all(selected["verdict"]["checks"][k] == control["checks"][k] for k in ("coverage", "identity")))
        result["benches"][bench] = {"clean_accuracy": d["accuracy"], "noise": noise, "required_drop": need, "clean_control": control,
                                    "ladder": ladder, "selected": selected, "rejected": rejected}
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
