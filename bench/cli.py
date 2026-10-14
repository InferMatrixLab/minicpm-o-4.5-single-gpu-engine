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
from bench.manifest import build_manifest, gpu_info, validate_manifest
from bench.metrics import request_metrics
from bench.experiment import Candidate, CellSpec, asdict_candidate, digest, unique_candidates
from bench.protocol import RunMeta, cell_compliant, cell_valid, evaluate_run, ids_sha256, measured_runs, plan_sweep, run_cell, save_run
from bench.servers import Server, ServerSpec, sglang_omni_spec, vllm_omni_spec
from bench.workload import (
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
        path = (deploy_dir / base) if base else None
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
    ids_eval = sets.tuning(concurrency) if id_set == "tuning" else sets.eval(concurrency)
    workload = {
        "dataset": SEED_TTS_REPO,
        "revision": SEED_TTS_REVISION,
        "warmup_ids_sha256": ids_sha256(sets.warmup(concurrency)),
        "measured_ids_sha256": ids_sha256(ids_eval),
    }
    return CellSpec(cand, concurrency, k, id_set, seed_index, timeout_s, gpu, workload)


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
    sglang_stream: bool = False,
) -> dict:
    cand = cell.candidate
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    ids_eval = sets.tuning(cell.concurrency) if cell.id_set == "tuning" else sets.eval(cell.concurrency)
    warm = canonical_requests(rows, sets.warmup(cell.concurrency), seed_index=cell.seed_index)
    evaluation = canonical_requests(rows, ids_eval, seed_index=cell.seed_index)
    adapter = _adapter(cand.adapter, sglang_stream)
    out.mkdir(parents=True, exist_ok=True)
    hw = gpu_info(cell.gpu_index)
    resolved = resolved_deploy(cand.adapter, cand.config, list(cand.extra), cand.env_dict())
    if deploy_files_sha256(resolved) != cand.deploy_files_sha256:
        raise RuntimeError(f"deploy files changed since candidate {cand.candidate_id} was declared")
    manifest = build_manifest(
        repos=DEFAULT_REPOS,
        env_pythons=DEFAULT_PYTHONS,
        out_dir=out,
        gpu_index=cell.gpu_index,
        deploy={"engine": cand.adapter, "config": cand.config, "resolved": resolved, "candidate": asdict_candidate(cand), "cell": cell.to_json()},
        sampling={**effective_sampling(cand.adapter, adapter, evaluation[0]), "seed_index": cell.seed_index},
        datasets=[{"repo": SEED_TTS_REPO, "revision": SEED_TTS_REVISION, "locale": "en",
                   "ids": {"warmup": [r.request_id for r in warm], cell.id_set: ids_eval}}],
        run_order_seed=run_order_seed,
        extra={"concurrency": cell.concurrency, "k": cell.k, "id_set": cell.id_set, "timeout_s": cell.timeout_s, "cache_policy": cache_policy,
               "port": port, "strict": strict},
    )
    gpu_name = str(hw.get("gpu_sku", "")).replace("NVIDIA ", "")
    meta = RunMeta(gpu=gpu_name, engine=cand.adapter, config=cand.config, candidate_id=cand.candidate_id, config_digest=cand.digest,
                   cell_digest=cell.digest, id_set=cell.id_set, seed_index=cell.seed_index)
    factory = None
    if cache_policy == "restart":
        spec = server_spec(cand, port, cell.gpu_index)
        factory = lambda i: Server(spec, out / f"server_run{i}.log")  # noqa: E731
    t0 = time.time()
    results = await run_cell(url, adapter, warm, evaluation, cell.concurrency, cell.k, out, timeout_s=cell.timeout_s, cache_policy=cache_policy,
                             server_factory=factory, meta=meta, save_audio=save_audio, strict=strict)
    errs = validate_manifest(manifest)
    result = {
        "cell_spec": cell.to_json(),
        "manifest": manifest,
        "manifest_errors": errs,
        "cell_valid": cell_valid(results),
        "cell_compliant": cell_compliant(results) and not errs,
        "runs": [r.summary() for r in results],
        "measured_runs": [r.summary() for r in measured_runs(results)],
        "wall_s": time.time() - t0,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    return result


async def _cell(args) -> int:
    url = None if args.cache_policy == "restart" else (args.url or f"http://127.0.0.1:{args.port}")
    cand = make_candidate({"adapter": args.adapter, "config": args.config, "extra": args.extra or [], "label": args.label,
                           "env": dict(kv.split("=", 1) for kv in args.env)})
    cell = make_cell(cand, args.concurrency, args.k, args.id_set, args.seed_index, args.timeout, args.gpu)
    res = await run_one_cell(cell=cell, out=Path(args.out), port=args.port, url=url, cache_policy=args.cache_policy,
                             run_order_seed=args.run_order_seed, save_audio=not args.no_audio, strict=not args.loose, sglang_stream=args.sglang_stream)
    print(json.dumps({k: res[k] for k in ("cell_valid", "cell_compliant", "manifest_errors")} | {"out": args.out}))
    return 0 if res["cell_compliant"] else 1


class SweepError(RuntimeError):
    pass


def _cell_artifacts_complete(cell_dir: Path, k: int) -> bool:
    return all((cell_dir / f"run{i}" / "summary.json").exists() and (cell_dir / f"run{i}" / "traces.jsonl").exists() for i in range(k + 1))


def resume_state(cell_dir: Path, cell: CellSpec) -> str:
    """'new' (no result), 'reuse' (matching, compliant, complete) or 'retry' (matching but not compliant).

    Raises if the directory holds a different experiment.
    """
    rp = cell_dir / "result.json"
    if not rp.exists():
        if cell_dir.exists() and any(cell_dir.iterdir()):
            return "retry"
        return "new"
    res = json.loads(rp.read_text())
    got = (res.get("cell_spec") or {}).get("digest")
    if got != cell.digest:
        raise SweepError(f"{cell_dir} holds a different experiment (digest {got} != {cell.digest}); refusing to reuse or overwrite it")
    m = res.get("manifest") or {}
    if res.get("cell_compliant") and not validate_manifest(m) and _cell_artifacts_complete(cell_dir, cell.k):
        return "reuse"
    return "retry"


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
    spec_digest = digest(spec)
    rec_path = out / "sweep.json"
    if rec_path.exists():
        old = json.loads(rec_path.read_text())
        if old.get("spec_digest") != spec_digest:
            raise SweepError(f"{out} belongs to a different sweep spec (digest {old.get('spec_digest')}); use a new output directory")
        record = old
    else:
        record = {"spec": spec, "spec_digest": spec_digest, "cells": {}, "attempts": []}
    seed = int(spec["run_order_seed"])
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
        state = resume_state(cell_dir, cell)
        if state == "retry":
            record["attempts"].append({"cell": cell.digest, "moved_to": str(_set_aside(cell_dir))})
        if state == "reuse":
            res = json.loads((cell_dir / "result.json").read_text())
        else:
            res = await run_one_cell(cell=cell, out=cell_dir, port=int(spec.get("port", 18100)), url=None, cache_policy="restart",
                                     run_order_seed=seed, save_audio=bool(spec.get("save_audio", True)), strict=True)
        record["cells"][cell.digest] = {"index": i, "candidate_id": cell.candidate.candidate_id, "concurrency": cell.concurrency,
                                        "dir": str(cell_dir), "reused": state == "reuse", "cell_valid": res["cell_valid"],
                                        "cell_compliant": res["cell_compliant"], "manifest_errors": res["manifest_errors"]}
        rec_path.write_text(json.dumps(record, indent=1))
        if not res["cell_compliant"] or res["manifest_errors"]:
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
    cp.add_argument("--id-set", choices=["eval", "tuning"], default="eval")
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
    data = json.loads(Path(args.path).read_text())
    errs = validate_manifest(data.get("manifest", data))
    print(json.dumps({"errors": errs}))
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
