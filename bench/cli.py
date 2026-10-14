"""Command line entry points.

    python -m bench.cli smoke --adapter vllm-omni --url http://127.0.0.1:18001 --n 2
    python -m bench.cli cell --adapter engine --url ... --concurrency 8 --k 5 --out results/...
    python -m bench.cli validate-manifest results/.../result.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

from bench.adapters import get_adapter
from bench.client import make_session, send_request
from bench.manifest import build_manifest, validate_manifest
from bench.metrics import request_metrics
from bench.protocol import cell_valid, measured_runs, run_cell, save_run, evaluate_run
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
DEFAULT_REPOS = {
    "qwen3_omni": ROOT,
    "vllm-omni": Path("/workspace/cgzheng/cgzheng/vllm-omni"),
    "sglang-omni": Path("/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main"),
}
DEFAULT_PYTHONS = {
    "bench": sys.executable,
    "vllm-omni": "/workspace/cgzheng/cgzheng/vllm-omni/.venv/bin/python",
    "sglang-omni": "/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main/.venv/bin/python",
}


def _adapter(args):
    kw = {}
    if args.adapter == "sglang-omni" and getattr(args, "sglang_stream", False):
        kw["stream"] = True
    return get_adapter(args.adapter, **kw)


async def _smoke(args) -> int:
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    ids = sets.warmup(max(1, (args.n + 1) // 2))[: args.n]
    reqs = canonical_requests(rows, ids)
    adapter = _adapter(args)
    out = Path(args.out) if args.out else None
    ok = True
    async with make_session() as s:
        traces = []
        for r in reqs:
            tr = await send_request(s, args.url, adapter, r, timeout_s=args.timeout)
            traces.append(tr)
            m = request_metrics(tr)
            ok &= m.ok
            print(
                json.dumps(
                    {
                        "id": r.request_id,
                        "ok": m.ok,
                        "reasons": m.invalid_reasons,
                        "D_s": round(m.duration_s, 3),
                        "ttfa_s": m.ttfa_s,
                        "rtf": m.rtf,
                        "e2e_s": m.e2e_s,
                        "n_audio_events": m.n_audio_events,
                        "first_chunk_s": m.first_chunk_s,
                        "target": r.row.target_text,
                        "text": tr.text,
                        "error": tr.error[:300],
                    }
                )
            )
    if out:
        save_run(evaluate_run("smoke", 1, traces, measured=False), out)
    return 0 if ok else 1


async def _cell(args) -> int:
    rows = load_seed_tts_rows()
    sets = make_id_sets(rows)
    c = args.concurrency
    ids_eval = sets.tuning(c) if args.id_set == "tuning" else sets.eval(c)
    warm = canonical_requests(rows, sets.warmup(c), seed_index=args.seed_index)
    evaluation = canonical_requests(rows, ids_eval, seed_index=args.seed_index)
    adapter = _adapter(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(
        repos=DEFAULT_REPOS,
        env_pythons=DEFAULT_PYTHONS,
        out_dir=out,
        gpu_index=args.gpu_index,
        deploy={"engine": args.adapter, "config": args.config_name, "resolved": json.loads(args.resolved_config) if args.resolved_config else {"name": args.config_name}},
        sampling={"thinker": asdict(ThinkerSampling()), "talker": asdict(TalkerSampling()), "seed_index": args.seed_index, "adapter_payload_example": _redact(adapter.build_payload(evaluation[0]))},
        datasets=[{"repo": SEED_TTS_REPO, "revision": SEED_TTS_REVISION, "locale": "en", "ids": {"warmup": [r.request_id for r in warm], args.id_set: ids_eval}}],
        run_order_seed=args.run_order_seed,
        extra={"concurrency": c, "k": args.k, "id_set": args.id_set, "timeout_s": args.timeout, "url": args.url},
    )
    t0 = time.time()
    results = await run_cell(args.url, adapter, warm, evaluation, c, args.k, out, timeout_s=args.timeout, save_audio=not args.no_audio)
    result = {
        "manifest": manifest,
        "manifest_errors": validate_manifest(manifest),
        "cell_valid": cell_valid(results),
        "runs": [r.summary() for r in results],
        "measured_runs": [r.summary() for r in measured_runs(results)],
        "wall_s": time.time() - t0,
    }
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(json.dumps({"cell_valid": result["cell_valid"], "manifest_errors": result["manifest_errors"], "out": str(out)}))
    return 0 if result["cell_valid"] else 1


def _redact(payload: dict) -> dict:
    out = json.loads(json.dumps(payload))
    for key in ("ref_audio",):
        if isinstance(out.get(key), str) and len(out[key]) > 80:
            out[key] = out[key][:40] + f"...<{len(out[key])} chars>"
    audio = out.get("audio")
    if isinstance(audio, dict) and isinstance(audio.get("ref_audio"), str):
        audio["ref_audio"] = audio["ref_audio"][:40] + "..."
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="bench")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("smoke", "cell"):
        sp = sub.add_parser(name)
        sp.add_argument("--adapter", required=True, choices=["engine", "vllm-omni", "sglang-omni"])
        sp.add_argument("--url", required=True)
        sp.add_argument("--timeout", type=float, default=180.0)
        sp.add_argument("--out", default=None)
        sp.add_argument("--sglang-stream", action="store_true")
        if name == "smoke":
            sp.add_argument("--n", type=int, default=2)
        else:
            sp.add_argument("--concurrency", type=int, required=True)
            sp.add_argument("--k", type=int, default=5)
            sp.add_argument("--id-set", choices=["eval", "tuning"], default="eval")
            sp.add_argument("--seed-index", type=int, default=0)
            sp.add_argument("--config-name", required=True)
            sp.add_argument("--resolved-config", default=None, help="JSON of the resolved deploy config")
            sp.add_argument("--gpu-index", type=int, required=True)
            sp.add_argument("--run-order-seed", type=int, default=0)
            sp.add_argument("--no-audio", action="store_true")
    vm = sub.add_parser("validate-manifest")
    vm.add_argument("path")
    args = p.parse_args(argv)
    if args.cmd == "smoke":
        return asyncio.run(_smoke(args))
    if args.cmd == "cell":
        if not args.out:
            p.error("--out is required for cell")
        return asyncio.run(_cell(args))
    data = json.loads(Path(args.path).read_text())
    errs = validate_manifest(data.get("manifest", data))
    print(json.dumps({"errors": errs}))
    return 0 if not errs else 1


if __name__ == "__main__":
    raise SystemExit(main())
