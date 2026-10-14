"""Command line entry points.

    python -m bench.cli smoke --adapter vllm-omni --url http://127.0.0.1:18001 --n 2
    python -m bench.cli cell  --adapter vllm-omni --config minicpmo_4_5_turn_mrv2.yaml --gpu 1 --port 18001 \
                              --concurrency 8 --k 5 --out results/l20x/...           # fresh server per run
    python -m bench.cli sweep --spec sweeps/l20x_baselines.json --out results/l20x/sweep1
    python -m bench.cli validate-manifest results/.../result.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import yaml

from bench.adapters import get_adapter
from bench.adapters.vllm_omni import stage_sampling
from bench.client import make_session, send_request
from bench.manifest import build_manifest, validate_manifest
from bench.metrics import request_metrics
from bench.experiment import Candidate, CellSpec, asdict_candidate, build_spec, digest, spec_base_digest, spec_digest, unique_candidates
from bench.fingerprint import env_fingerprint, file_sha256, gpu_identity, repo_fingerprint, tree_sha256
from bench.protocol import RunMeta, evaluate_run, ids_sha256, plan_sweep, run_cell, save_run
from bench.verify import verify_cell
from bench.servers import Server, ServerSpec, sglang_omni_spec, vllm_omni_spec
from bench.workload import (
    MODEL_REPO,
    MODEL_REVISION,
    SEED_TTS_REPO,
    SEED_TTS_REVISION,
    TalkerSampling,
    ThinkerSampling,
    canonical_requests,
    load_seed_tts_rows,
    make_id_sets,
)

ROOT = Path(__file__).resolve().parents[1]
VLLM_OMNI = Path("/workspace/cgzheng/cgzheng/vllm-omni")
SGLANG_OMNI = Path("/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main")
DEFAULT_REPOS = {"qwen3_omni": ROOT, "vllm-omni": VLLM_OMNI, "sglang-omni": SGLANG_OMNI}
DEFAULT_PYTHONS = {
    "bench": sys.executable,
    "vllm-omni": str(VLLM_OMNI / ".venv/bin/python"),
    "sglang-omni": str(SGLANG_OMNI / ".venv/bin/python"),
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def resolve_vllm_deploy(config: str) -> dict:
    """The YAML inheritance chain (``base_config``), verbatim and hashed."""
    deploy_dir = VLLM_OMNI / "vllm_omni" / "deploy"
    chain = []
    path = Path(config) if config.startswith("/") else deploy_dir / config
    seen = set()
    while path is not None and path not in seen:
        seen.add(path)
        text = path.read_text()
        chain.append({"file": str(path), "sha256": _sha(text), "text": text})
        base = (yaml.safe_load(text) or {}).get("base_config")
        path = (path.parent / base) if base else None  # vllm-omni resolves base_config relative to the overlay
    return {"engine": "vllm-omni", "config": config, "chain": chain}


def resolve_sglang_deploy(extra: list[str]) -> dict:
    script = (ROOT / "scripts" / "serve_sglang_omni.sh").read_text()
    cfg = (SGLANG_OMNI / "sglang_omni" / "models" / "minicpm_o" / "config.py").read_text()
    return {
        "engine": "sglang-omni",
        "config": "speech",
        "serve_script": script,
        "serve_script_sha256": _sha(script),
        "extra_args": extra,
        "model_config_py_sha256": _sha(cfg),
    }


def server_spec(cand: Candidate, port: int, gpu: int) -> ServerSpec:
    extra = list(cand.extra)
    if cand.adapter == "vllm-omni":
        spec = vllm_omni_spec(cand.config, port, gpu, extra)
    elif cand.adapter == "sglang-omni":
        spec = sglang_omni_spec(port, gpu, extra)
    elif cand.adapter == "engine":
        spec = ServerSpec(name="engine", cmd=["bash", str(ROOT / "scripts" / "serve_engine.sh"), cand.config, str(port), *extra], port=port, gpu=gpu)
    else:
        raise ValueError(cand.adapter)
    spec.env.update(cand.env_dict())
    spec.name = cand.candidate_id
    return spec


def resolved_deploy(adapter: str, config: str, extra: list[str], env: dict) -> dict:
    if adapter == "vllm-omni":
        d = resolve_vllm_deploy(config)
    elif adapter == "sglang-omni":
        d = resolve_sglang_deploy(extra)
    else:
        script = ROOT / "scripts" / "serve_engine.sh"
        d = {"engine": adapter, "config": config, "serve_script_sha256": _sha(script.read_text()) if script.exists() else None}
    d["extra_args"] = list(extra)
    d["env_overlay"] = dict(env)
    return d


def deploy_files_sha256(resolved: dict) -> tuple:
    if "chain" in resolved:
        return tuple(c["sha256"] for c in resolved["chain"])
    return tuple(v for k, v in sorted(resolved.items()) if k.endswith("sha256") and v)


def make_candidate(d: dict) -> Candidate:
    base = Candidate.from_spec(d)
    resolved = resolved_deploy(base.adapter, base.config, list(base.extra), base.env_dict())
    return Candidate.from_spec(d, deploy_files_sha256(resolved))


def make_cell(cand: Candidate, concurrency: int, k: int, id_set: str, seed_index: int, timeout_s: float, gpu: int) -> CellSpec:
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    workload = {
        "dataset": SEED_TTS_REPO,
        "revision": SEED_TTS_REVISION,
        "warmup_ids_sha256": ids_sha256(sets.warmup(concurrency)),
        "measured_ids_sha256": ids_sha256(sets.measured(id_set, concurrency)),
    }
    return CellSpec(cand, concurrency, k, id_set, seed_index, timeout_s, gpu, workload)


ADAPTER_PYTHON = {"vllm-omni": DEFAULT_PYTHONS["vllm-omni"], "sglang-omni": DEFAULT_PYTHONS["sglang-omni"], "engine": sys.executable}
SERVE_SCRIPTS = {"vllm-omni": "serve_vllm_omni.sh", "sglang-omni": "serve_sglang_omni.sh", "engine": "serve_engine.sh"}


def execution_fingerprints(adapter_name: str) -> dict:
    """Everything outside the candidate that the measurement depends on."""
    scripts = ROOT / "scripts"
    return {
        "harness_tree_sha256": tree_sha256(ROOT, ("bench", "scripts")),
        "repos": {"vllm-omni": repo_fingerprint(VLLM_OMNI), "sglang-omni": repo_fingerprint(SGLANG_OMNI)},
        "envs": {"bench": env_fingerprint(sys.executable)["freeze_sha256"], adapter_name: env_fingerprint(ADAPTER_PYTHON[adapter_name])["freeze_sha256"]},
        "launch_scripts": {name: file_sha256(scripts / name) for name in ("env.sh", SERVE_SCRIPTS[adapter_name])},
    }


def build_experiment(cell: CellSpec, kind: str, cache_policy: str, strict: bool, save_audio: bool, record_raw: bool, sglang_stream: bool) -> dict:
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    adapter = _adapter(cell.candidate.adapter, sglang_stream)
    example = canonical_requests(rows, sets.measured(cell.id_set, cell.concurrency)[:1], seed_index=cell.seed_index)[0]
    return build_spec(
        kind=kind,
        cell=cell,
        warmup_ids=sets.warmup(cell.concurrency),
        measured_ids=sets.measured(cell.id_set, cell.concurrency),
        model={"repo": MODEL_REPO, "revision": MODEL_REVISION},
        dataset={"repo": SEED_TTS_REPO, "revision": SEED_TTS_REVISION, "locale": "en"},
        sampling=effective_sampling(cell.candidate.adapter, adapter, example) | {"seed_policy": "sha256(row_id) + seed_index"},
        adapter_options={"name": cell.candidate.adapter, "sglang_stream": bool(sglang_stream), "streaming": bool(adapter.streaming)},
        fingerprints=execution_fingerprints(cell.candidate.adapter),
        hardware=gpu_identity(cell.gpu_index),
        cache_policy=cache_policy,
        strict=strict,
        save_audio=save_audio,
        record_raw=record_raw,
    )


def effective_sampling(adapter_name: str, adapter, example) -> dict:
    out = {"canonical": {"thinker": asdict(ThinkerSampling()), "talker": asdict(TalkerSampling())}}
    payload = adapter.build_payload(example)
    if adapter_name == "vllm-omni":
        out["sent"] = {"sampling_params_list": stage_sampling(example), "control_sampling": adapter.control_sampling}
    else:
        out["sent"] = {k: v for k, v in payload.items() if k not in ("messages", "ref_audio", "audio")}
    if adapter_name == "sglang-omni":
        out["fixed_by_implementation"] = {"min_new_tokens": 50, "repetition_window": 16}
    return out


def _adapter(name: str, sglang_stream: bool = False):
    return get_adapter(name, **({"stream": True} if name == "sglang-omni" and sglang_stream else {}))


async def _smoke(args) -> int:
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    ids = sets.warmup(max(1, (args.n + 1) // 2))[: args.n]
    reqs = canonical_requests(rows, ids)
    adapter = _adapter(args.adapter, args.sglang_stream)
    ok = True
    traces = []
    async with make_session() as s:
        for r in reqs:
            tr = await send_request(s, args.url, adapter, r, timeout_s=args.timeout)
            traces.append(tr)
            m = request_metrics(tr)
            ok &= m.ok
            print(json.dumps({"id": r.request_id, "ok": m.ok, "reasons": m.invalid_reasons, "D_s": round(m.duration_s, 3), "ttfa_s": m.ttfa_s,
                              "rtf": m.rtf, "e2e_s": m.e2e_s, "n_audio_events": m.n_audio_events, "target": r.row.target_text, "text": tr.text,
                              "error": tr.error[:300]}))
    if args.out:
        save_run(evaluate_run("smoke", 1, traces, measured=False), Path(args.out))
    return 0 if ok else 1


async def run_one_cell(
    *, cell: CellSpec, out: Path, port: int, url: str | None, cache_policy: str, run_order_seed: int, save_audio: bool, strict: bool,
    sglang_stream: bool = False, kind: str = "performance", spec: dict | None = None, run_indices: list[int] | None = None,
    recovery: list[dict] | None = None,
) -> dict:
    """Run (or extend) one cell.  The frozen experiment spec is written before any run executes."""
    cand = cell.candidate
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    ids_eval = sets.measured(cell.id_set, cell.concurrency)
    warm = canonical_requests(rows, sets.warmup(cell.concurrency), seed_index=cell.seed_index)
    evaluation = canonical_requests(rows, ids_eval, seed_index=cell.seed_index)
    adapter = _adapter(cand.adapter, sglang_stream)
    out.mkdir(parents=True, exist_ok=True)
    resolved = resolved_deploy(cand.adapter, cand.config, list(cand.extra), cand.env_dict())
    if deploy_files_sha256(resolved) != cand.deploy_files_sha256:
        raise RuntimeError(f"deploy files changed since candidate {cand.candidate_id} was declared")
    if spec is None:
        spec = build_experiment(cell, kind, cache_policy, strict, save_audio, True, sglang_stream)
    if spec["hardware"].get("gpu_uuid") != gpu_identity(cell.gpu_index).get("gpu_uuid"):
        raise RuntimeError("GPU differs from the frozen experiment")
    base_digest = spec_base_digest(spec)
    prev = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else None
    manifest = build_manifest(
        repos=DEFAULT_REPOS,
        env_pythons=DEFAULT_PYTHONS,
        out_dir=out,
        gpu_index=cell.gpu_index,
        deploy={"engine": cand.adapter, "config": cand.config, "resolved": resolved, "candidate": asdict_candidate(cand), "cell": cell.to_json(),
                "experiment": spec, "experiment_digest": spec_digest(spec)},
        sampling=spec["sampling"],
        datasets=[{"repo": SEED_TTS_REPO, "revision": SEED_TTS_REVISION, "locale": "en",
                   "ids": {"warmup": [r.request_id for r in warm], cell.id_set: ids_eval}}],
        run_order_seed=run_order_seed,
        extra={"cache_policy": cache_policy, "port": port},
    )
    header = {"experiment": spec, "experiment_digest": spec_digest(spec), "experiment_base_digest": base_digest, "manifest": manifest}
    prev_decl = json.loads((out / "experiment.json").read_text()) if (out / "experiment.json").exists() else {}
    declare(out, {**header, "recoveries": prev_decl.get("recoveries", []) + list(recovery or [])})  # frozen before execution
    gpu_name = spec["hardware"].get("gpu_sku", "")
    meta = RunMeta(gpu=gpu_name, engine=cand.adapter, config=cand.config, candidate_id=cand.candidate_id, config_digest=cand.digest,
                   cell_digest=cell.digest, experiment_base_digest=base_digest, mps_policy=cand.mps, id_set=cell.id_set, seed_index=cell.seed_index)
    factory = None
    if cache_policy == "restart":
        sspec = server_spec(cand, port, cell.gpu_index)
        factory = lambda i: Server(sspec, out / f"server_run{i}.log")  # noqa: E731
    t0 = time.time()
    results = await run_cell(url, adapter, warm, evaluation, cell.concurrency, spec["k"], out, timeout_s=cell.timeout_s, cache_policy=cache_policy,
                             server_factory=factory, meta=meta, save_audio=save_audio, strict=strict, run_indices=run_indices)
    runs = {r["label"]: r for r in (prev or {}).get("runs", [])} if run_indices is not None else {}
    artifacts = dict((prev or {}).get("artifacts", {})) if run_indices is not None else {}
    for r in results:
        runs[r.label] = r.summary()
        artifacts[r.label] = r.artifacts
    ordered = [runs[f"run{i}"] for i in range(spec["k"] + 1) if f"run{i}" in runs]
    result = {**header, "runs": ordered, "measured_runs": [r for r in ordered if r["label"] != "run0"], "artifacts": artifacts,
              "wall_s": time.time() - t0}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    ver = verify_cell(out)  # compliance is derived from the artifacts, never stored as a verdict to trust
    result["verification"] = {"ok": ver.ok, "problems": ver.problems}
    result["cell_valid"] = all(r["valid"] for r in result["measured_runs"]) and bool(result["measured_runs"])
    result["cell_compliant"] = ver.ok
    result["manifest_errors"] = validate_manifest(manifest)
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


async def _cell(args) -> int:
    url = None if args.cache_policy == "restart" else (args.url or f"http://127.0.0.1:{args.port}")
    cand = make_candidate({"adapter": args.adapter, "config": args.config, "extra": args.extra or [], "label": args.label, "mps": args.mps,
                           "env": dict(kv.split("=", 1) for kv in args.env)})
    cell = make_cell(cand, args.concurrency, args.k, args.id_set, args.seed_index, args.timeout, args.gpu)
    res = await run_one_cell(cell=cell, out=Path(args.out), port=args.port, url=url, cache_policy=args.cache_policy, kind=args.kind,
                             run_order_seed=args.run_order_seed, save_audio=not args.no_audio, strict=not args.loose, sglang_stream=args.sglang_stream)
    print(json.dumps({k: res[k] for k in ("cell_valid", "cell_compliant", "manifest_errors")} | {"problems": res["verification"]["problems"][:5],
                                                                                                  "out": args.out}))
    return 0 if res["cell_compliant"] else 1


def declare(out: Path, header: dict) -> None:
    """Atomically publish the authoritative experiment declaration."""
    tmp = out / ".experiment.json.tmp"
    tmp.write_text(json.dumps(header, indent=1))
    os.replace(tmp, out / "experiment.json")


async def _extend(args) -> int:
    """Development K extension: publish the new K in the declaration first, then run only the new runs.

    A declared extension that did not finish leaves the cell unverifiable; it is
    completed only by an explicit ``--recover``, which re-runs the missing runs
    (setting aside any partial run directories) and records the recovery.
    """
    out = Path(args.cell)
    res = json.loads((out / "result.json").read_text())
    decl = json.loads((out / "experiment.json").read_text())
    recovery = None
    if decl["experiment"] != res.get("experiment"):
        if not args.recover:
            raise SystemExit("cell has an unfinished declared extension; complete it with --recover")
        spec = decl["experiment"]
        done = {r["label"] for r in res.get("runs", [])}
        missing = [i for i in range(int(spec["k"]) + 1) if f"run{i}" not in done]
        for i in missing:
            if (out / f"run{i}").exists():
                _set_aside(out / f"run{i}")
        k0 = min(missing) - 1
        args.k = int(spec["k"])
        recovery = [{"recovered_runs": [f"run{i}" for i in missing], "unix": time.time()}]
    else:
        spec = res["experiment"]
        k0 = int(spec["k"])
        if args.k <= k0:
            raise SystemExit(f"new K {args.k} must exceed the declared K {k0}")
        v = verify_cell(out)
        if not v.ok:
            raise SystemExit(f"cannot extend an unverified cell: {v.problems[:5]}")
        if args.k > 10:
            raise SystemExit("development extensions stop at K=10")
        spec = json.loads(json.dumps(spec))
        spec["k_updates"].append({"from": k0, "to": args.k, "reason": args.reason, "unix": time.time()})
        spec["k"] = args.k
    c = spec["candidate"]
    cand = Candidate.from_spec({"adapter": c["adapter"], "config": c["config"], "extra": c["extra"], "env": dict(map(tuple, c["env"])),
                                "label": c["label"], "mps": c["mps"]}, tuple(c["deploy_files_sha256"]))
    cell = make_cell(cand, spec["concurrency"], args.k, spec["id_set"], spec["seed_index"], spec["timeout_s"], spec["hardware"]["gpu_index"])
    if spec_base_digest(build_experiment(cell, spec["kind"], spec["cache_policy"], spec["strict"], spec["save_audio"], spec["record_raw"],
                                         spec["adapter_options"]["sglang_stream"])) != spec_base_digest(spec):
        raise SystemExit("current execution fingerprints differ from the frozen experiment; start a new experiment instead")
    res2 = await run_one_cell(cell=cell, out=out, port=args.port, url=None, cache_policy=spec["cache_policy"], run_order_seed=0,
                              save_audio=spec["save_audio"], strict=spec["strict"], sglang_stream=spec["adapter_options"]["sglang_stream"],
                              kind=spec["kind"], spec=spec, run_indices=list(range(k0 + 1, args.k + 1)), recovery=recovery)
    print(json.dumps({"cell_compliant": res2["cell_compliant"], "k": args.k, "problems": res2["verification"]["problems"][:5]}))
    return 0 if res2["cell_compliant"] else 1


class SweepError(RuntimeError):
    pass


def resume_state(cell_dir: Path, current_spec: dict) -> str:
    """'new', 'reuse' (verified against the current experiment) or 'retry' (same experiment, not compliant).

    Raises if the directory holds a different experiment.
    """
    rp = cell_dir / "result.json"
    if not rp.exists():
        return "retry" if cell_dir.exists() and any(cell_dir.iterdir()) else "new"
    try:
        res = json.loads(rp.read_text())
    except json.JSONDecodeError:
        return "retry"
    stored = res.get("experiment") or {}
    if not stored or spec_base_digest(stored) != spec_base_digest(current_spec):
        raise SweepError(f"{cell_dir} holds a different experiment than the current spec and fingerprints; refusing to reuse or overwrite it")
    return "reuse" if verify_cell(cell_dir, current_spec=current_spec).ok else "retry"


def _set_aside(cell_dir: Path) -> Path:
    """Keep a failed or incomplete attempt for the audit record."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    dest = cell_dir.parent / "attempts" / f"{cell_dir.name}.{stamp}"
    n = 1
    while dest.exists():
        dest = cell_dir.parent / "attempts" / f"{cell_dir.name}.{stamp}.{n}"
        n += 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    cell_dir.rename(dest)
    return dest


async def _sweep(args) -> int:
    """Execute every (candidate, C) cell of a spec in a seeded random order, a fresh server per run."""
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    spec_d = digest(spec)
    rec_path = out / "sweep.json"
    if rec_path.exists():
        old = json.loads(rec_path.read_text())
        if old.get("spec_digest") != spec_d:
            raise SweepError(f"{out} belongs to a different sweep spec (digest {old.get('spec_digest')}); use a new output directory")
        record = old
    else:
        record = {"spec": spec, "spec_digest": spec_d, "cells": {}, "attempts": []}
    seed = int(spec["run_order_seed"])
    kind = spec.get("kind", "performance")
    cands = unique_candidates([make_candidate(c) for c in spec["configs"]])
    cells = [make_cell(c, conc, int(spec["k"]), spec.get("id_set", "eval"), int(spec.get("seed_index", 0)), float(spec.get("timeout_s", 180.0)),
                       int(spec["gpu"])) for c in cands for conc in spec["concurrencies"]]
    order = plan_sweep(cells, seed)
    record["order"] = [{"candidate_id": c.candidate.candidate_id, "concurrency": c.concurrency, "digest": c.digest} for c in order]
    rec_path.write_text(json.dumps(record, indent=1))
    rc = 0
    for i, cell in enumerate(order):
        safe = cell.candidate.candidate_id.replace(":", "_").replace("/", "_")
        cell_dir = out / f"{i:02d}_{safe}_c{cell.concurrency}"
        current = build_experiment(cell, kind, "restart", True, bool(spec.get("save_audio", True)), True, False)
        state = resume_state(cell_dir, current)
        if state == "retry":
            record["attempts"].append({"cell": cell.digest, "moved_to": str(_set_aside(cell_dir))})
        if state == "reuse":
            res = json.loads((cell_dir / "result.json").read_text())
        else:
            try:
                res = await run_one_cell(cell=cell, out=cell_dir, port=int(spec.get("port", 18100)), url=None, cache_policy="restart",
                                         run_order_seed=seed, save_audio=bool(spec.get("save_audio", True)), strict=True, kind=kind, spec=current)
            except Exception as e:  # a failed cell (e.g. server startup) is recorded and the sweep continues
                record["cells"][cell.digest] = {"index": i, "candidate_id": cell.candidate.candidate_id, "concurrency": cell.concurrency,
                                                "dir": str(cell_dir), "cell_compliant": False, "failure": f"{type(e).__name__}: {e}"[:500]}
                rec_path.write_text(json.dumps(record, indent=1))
                rc = 1
                continue
        compliant = verify_cell(cell_dir, current_spec=current).ok
        record["cells"][cell.digest] = {"index": i, "candidate_id": cell.candidate.candidate_id, "concurrency": cell.concurrency,
                                        "dir": str(cell_dir), "reused": state == "reuse", "cell_compliant": compliant,
                                        "manifest_errors": res.get("manifest_errors", [])}
        rec_path.write_text(json.dumps(record, indent=1))
        if not compliant:
            rc = 1
    return rc


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("smoke")
    sp.add_argument("--adapter", required=True, choices=["engine", "vllm-omni", "sglang-omni"])
    sp.add_argument("--url", required=True)
    sp.add_argument("--n", type=int, default=2)
    sp.add_argument("--timeout", type=float, default=180.0)
    sp.add_argument("--out", default=None)
    sp.add_argument("--sglang-stream", action="store_true")
    cp = sub.add_parser("cell")
    cp.add_argument("--adapter", required=True, choices=["engine", "vllm-omni", "sglang-omni"])
    cp.add_argument("--config", required=True, help="deploy config name (vllm-omni YAML, 'speech' for sglang-omni)")
    cp.add_argument("--gpu", type=int, required=True)
    cp.add_argument("--port", type=int, default=18100)
    cp.add_argument("--url", default=None, help="existing server (cache policy flush/none)")
    cp.add_argument("--cache-policy", choices=["restart", "flush", "none"], default="restart")
    cp.add_argument("--concurrency", type=int, required=True)
    cp.add_argument("--k", type=int, default=5)
    cp.add_argument("--id-set", choices=["eval", "tuning", "full"], default="eval")
    cp.add_argument("--kind", choices=["performance", "quality"], default="performance")
    cp.add_argument("--mps", choices=["off", "on"], default="off", help="declared MPS policy (verified at runtime)")
    cp.add_argument("--seed-index", type=int, default=0)
    cp.add_argument("--run-order-seed", type=int, default=0)
    cp.add_argument("--timeout", type=float, default=180.0)
    cp.add_argument("--out", required=True)
    cp.add_argument("--no-audio", action="store_true")
    cp.add_argument("--loose", action="store_true", help="skip mandatory-geometry checks (development only)")
    cp.add_argument("--sglang-stream", action="store_true")
    cp.add_argument("--label", default="", help="variant label, e.g. mps-on")
    cp.add_argument("--env", action="append", default=[], help="KEY=VALUE added to the server environment")
    cp.add_argument("extra", nargs="*", help="extra server args after --")
    ex = sub.add_parser("extend", help="declared development K extension of a verified cell")
    ex.add_argument("--cell", required=True)
    ex.add_argument("--k", type=int, required=True)
    ex.add_argument("--reason", required=True)
    ex.add_argument("--port", type=int, default=18100)
    ex.add_argument("--recover", action="store_true", help="complete an interrupted declared extension")
    sw = sub.add_parser("sweep")
    sw.add_argument("--spec", required=True)
    sw.add_argument("--out", required=True)
    vm = sub.add_parser("validate-manifest")
    vm.add_argument("path")
    args = p.parse_args(argv)
    if args.cmd == "smoke":
        return asyncio.run(_smoke(args))
    if args.cmd == "cell":
        return asyncio.run(_cell(args))
    if args.cmd == "sweep":
        return asyncio.run(_sweep(args))
    if args.cmd == "extend":
        return asyncio.run(_extend(args))
    data = json.loads(Path(args.path).read_text())
    errs = validate_manifest(data.get("manifest", data))
    print(json.dumps({"errors": errs}))
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
