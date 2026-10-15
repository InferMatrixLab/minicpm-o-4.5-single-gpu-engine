"""Compare each repo's native benchmark with the harness client on the same configuration and concurrency.

    python tools/xcheck_compare.py --dir results/l20x/campaign/xcheck --out results/l20x/campaign/xcheck/xcheck.json

vllm-omni (``vllm bench serve --omni``, C=1/4/8) and sglang-omni
(``benchmarks.eval.benchmark_omni_seedtts``, C=16) each ran on a fresh server of
the class-R config next to a fresh-server K=1 harness cell on the evaluation
IDs.  Compared quantities, per side: request throughput (req/s), audio
throughput (audio-s/s), median and mean TTFA (vLLM ``audio_ttfp``; sglang's
non-streaming latency), and mean RTF.  A difference above 5% must be
explained; the workloads differ by construction (native prompts and sampling
defaults, row order, warmup), recorded per side as the mean output duration and
the request count.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TOL = 0.05


def ours(cell: Path) -> dict:
    from bench.verify import verify_cell

    v = verify_cell(cell)
    if not v.ok:
        raise SystemExit(f"{cell} fails verification: {v.problems[:3]}")
    run = cell / "run1"
    reqs = [json.loads(line) for line in (run / "requests.jsonl").read_text().splitlines() if line.strip()]
    s = v.measured_summaries[0]
    ttfa = [r["ttfa_s"] for r in reqs if r.get("ttfa_s") is not None]
    return {"requests": s["n_requests"], "req_s": s["req_s"], "audio_s_per_s": s["audio_s_per_s"], "ttfa_median_s": float(np.median(ttfa)),
            "ttfa_mean_s": float(np.mean(ttfa)), "rtf_mean": s["rtf_mean"], "mean_duration_s": s["total_audio_s"] / s["n_requests"],
            "e2e_p50_s": s["e2e_p50_s"]}


def native_vllm(path: Path) -> dict:
    d = json.loads(path.read_text())
    n = d["completed"]
    return {"requests": n, "req_s": d["request_throughput"], "audio_s_per_s": d["audio_throughput"], "ttfa_median_s": d["median_audio_ttfp_ms"] / 1e3,
            "ttfa_mean_s": d["mean_audio_ttfp_ms"] / 1e3, "rtf_mean": d["mean_audio_rtf"], "mean_duration_s": d["total_audio_duration_s"] / n,
            "failed": d.get("failed")}


def native_sglang(d: Path) -> dict:
    doc = json.loads((d / "speed_results.json").read_text())
    s = doc["summary"]
    lat = [r["latency_s"] for r in doc["per_request"] if r.get("latency_s") is not None]
    return {"requests": len(lat), "req_s": s.get("throughput_qps"), "audio_s_per_s": s.get("audio_throughput_s_per_s"),
            "ttfa_median_s": float(np.median(lat)), "ttfa_mean_s": float(np.mean(lat)), "rtf_mean": s.get("rtf_mean"),
            "mean_duration_s": s.get("audio_duration_mean_s"), "failed": s.get("failed_requests")}


def compare(a: dict, b: dict) -> dict:
    out = {}
    for k in ("req_s", "audio_s_per_s", "ttfa_median_s", "ttfa_mean_s", "rtf_mean"):
        if a.get(k) and b.get(k):
            out[k] = {"harness": a[k], "native": b[k], "rel_diff": (a[k] - b[k]) / b[k], "within_5pct": abs(a[k] - b[k]) <= TOL * abs(b[k])}
    # the throughput/latency gap expected from different output lengths alone
    out["duration_ratio"] = a["mean_duration_s"] / b["mean_duration_s"] if b.get("mean_duration_s") else None
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    d = Path(a.dir)
    rows = []
    for c in (1, 4, 8):
        if (d / f"native_vllm_c{c}.json").exists() and (d / f"ours_vllm_c{c}" / "result.json").exists():
            rows.append({"repo": "vllm-omni", "concurrency": c, "harness": (h := ours(d / f"ours_vllm_c{c}")),
                         "native": (n := native_vllm(d / f"native_vllm_c{c}.json")), "comparison": compare(h, n)})
    if (d / "native_sglang_c16" / "speed_results.json").exists() and (d / "ours_sglang_c16" / "result.json").exists():
        rows.append({"repo": "sglang-omni", "concurrency": 16, "harness": (h := ours(d / "ours_sglang_c16")),
                     "native": (n := native_sglang(d / "native_sglang_c16")), "comparison": compare(h, n)})
    Path(a.out).write_text(json.dumps({"tolerance": TOL, "rows": rows}, indent=1))
    for r in rows:
        print(r["repo"], r["concurrency"], {k: (round(v["rel_diff"], 3), v["within_5pct"]) for k, v in r["comparison"].items() if isinstance(v, dict)},
              "duration_ratio", r["comparison"]["duration_ratio"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
