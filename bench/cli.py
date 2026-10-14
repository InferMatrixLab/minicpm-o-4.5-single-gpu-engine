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
from bench.protocol import CellPlan, RunMeta, cell_compliant, cell_valid, evaluate_run, measured_runs, plan_sweep, run_cell, save_run
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


def server_spec(adapter: str, config: str, port: int, gpu: int, extra: list[str]) -> ServerSpec:
    if adapter == "vllm-omni":
        return vllm_omni_spec(config, port, gpu, extra)
    if adapter == "sglang-omni":
        return sglang_omni_spec(port, gpu, extra)
    if adapter == "engine":
        return ServerSpec(name="engine", cmd=["bash", str(ROOT / "scripts" / "serve_engine.sh"), config, str(port), *extra], port=port, gpu=gpu)
    raise ValueError(adapter)


def resolved_deploy(adapter: str, config: str, extra: list[str]) -> dict:
    if adapter == "vllm-omni":
        return resolve_vllm_deploy(config)
    if adapter == "sglang-omni":
        return resolve_sglang_deploy(extra)
    return {"engine": adapter, "config": config, "extra_args": extra}


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
    *, adapter_name: str, config: str, concurrency: int, k: int, out: Path, gpu: int, port: int, url: str | None, cache_policy: str,
    id_set: str, seed_index: int, run_order_seed: int, timeout: float, extra: list[str], save_audio: bool, strict: bool, sglang_stream: bool = False,
) -> dict:
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    ids_eval = sets.tuning(concurrency) if id_set == "tuning" else sets.eval(concurrency)
    warm = canonical_requests(rows, sets.warmup(concurrency), seed_index=seed_index)
    evaluation = canonical_requests(rows, ids_eval, seed_index=seed_index)
    adapter = _adapter(adapter_name, sglang_stream)
    out.mkdir(parents=True, exist_ok=True)
    hw = gpu_info(gpu)
    manifest = build_manifest(
        repos=DEFAULT_REPOS,
        env_pythons=DEFAULT_PYTHONS,
        out_dir=out,
        gpu_index=gpu,
        deploy={"engine": adapter_name, "config": config, "resolved": resolved_deploy(adapter_name, config, extra)},
        sampling={**effective_sampling(adapter_name, adapter, evaluation[0]), "seed_index": seed_index},
        datasets=[{"repo": SEED_TTS_REPO, "revision": SEED_TTS_REVISION, "locale": "en", "ids": {"warmup": [r.request_id for r in warm], id_set: ids_eval}}],
        run_order_seed=run_order_seed,
        extra={"concurrency": concurrency, "k": k, "id_set": id_set, "timeout_s": timeout, "cache_policy": cache_policy, "port": port},
    )
    gpu_name = str(hw.get("gpu_sku", "")).replace("NVIDIA ", "")
    meta = RunMeta(gpu=gpu_name, engine=adapter_name, config=config, id_set=id_set, seed_index=seed_index)
    factory = None
    if cache_policy == "restart":
        spec = server_spec(adapter_name, config, port, gpu, extra)
        factory = lambda i: Server(spec, out / f"server_run{i}.log")  # noqa: E731
    t0 = time.time()
    results = await run_cell(url, adapter, warm, evaluation, concurrency, k, out, timeout_s=timeout, cache_policy=cache_policy,
                             server_factory=factory, meta=meta, save_audio=save_audio, strict=strict)
    errs = validate_manifest(manifest)
    result = {
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
    res = await run_one_cell(
        adapter_name=args.adapter, config=args.config, concurrency=args.concurrency, k=args.k, out=Path(args.out), gpu=args.gpu, port=args.port,
        url=url, cache_policy=args.cache_policy, id_set=args.id_set, seed_index=args.seed_index, run_order_seed=args.run_order_seed,
        timeout=args.timeout, extra=args.extra or [], save_audio=not args.no_audio, strict=not args.loose, sglang_stream=args.sglang_stream,
    )
    print(json.dumps({k: res[k] for k in ("cell_valid", "cell_compliant", "manifest_errors")} | {"out": args.out}))
    return 0 if res["cell_valid"] and not res["manifest_errors"] else 1


async def _sweep(args) -> int:
    """Execute every (engine, config, C) cell of a spec in a seeded random order, a fresh server per run."""
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seed = int(spec["run_order_seed"])
    cells = [CellPlan(c["adapter"], c["config"], conc) for c in spec["configs"] for conc in spec["concurrencies"]]
    extra_by = {(c["adapter"], c["config"]): c.get("extra", []) for c in spec["configs"]}
    order = plan_sweep(cells, seed)
    record = {"spec": spec, "order": [asdict(c) for c in order], "cells": []}
    (out / "sweep.json").write_text(json.dumps(record, indent=1))
    rc = 0
    for i, c in enumerate(order):
        cell_dir = out / f"{i:02d}_{c.engine}_{Path(c.config).stem}_c{c.concurrency}"
        if (cell_dir / "result.json").exists():
            res = json.loads((cell_dir / "result.json").read_text())
        else:
            res = await run_one_cell(
                adapter_name=c.engine, config=c.config, concurrency=c.concurrency, k=int(spec["k"]), out=cell_dir, gpu=int(spec["gpu"]),
                port=int(spec.get("port", 18100)), url=None, cache_policy="restart", id_set=spec.get("id_set", "eval"),
                seed_index=int(spec.get("seed_index", 0)), run_order_seed=seed, timeout=float(spec.get("timeout_s", 180.0)),
                extra=extra_by[(c.engine, c.config)], save_audio=bool(spec.get("save_audio", True)), strict=True,
            )
        record["cells"].append({"index": i, **asdict(c), "dir": str(cell_dir), "cell_valid": res["cell_valid"], "cell_compliant": res["cell_compliant"]})
        (out / "sweep.json").write_text(json.dumps(record, indent=1))
        rc |= 0 if res["cell_valid"] else 1
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
