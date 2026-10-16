"""Provisional AC-5 table: each engine evaluation cell against the envelope's winner cell, per C and metric.

    .venv/bin/python tools/engine_vs_envelope.py results/engine_sweeps/<eval sweep> [results/l20x/campaign/envelope.json]

Both cells are verified from their artifacts and compared with ``bench.stats.compare_cells`` (run-level bootstrap of the
oriented ratio, development mode: their declared K).  Point ratio, one-sided 95% lower bound and verdict per cell.  Not
the final H200 claim (fixed K=10, quality gates, duration guards).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.stats import compare_cells  # noqa: E402

METRICS = {"throughput": "throughput_frames_s", "ttfa_p50": "ttfa_p50_s", "ttfa_p95": "ttfa_p95_s", "rtf_mean": "rtf_mean"}


def main():
    sweep = Path(sys.argv[1])
    env = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "results/l20x/campaign/envelope.json").read_text())
    rows = []
    for cell in sorted(p for p in sweep.iterdir() if (p / "result.json").exists()):
        c = str(json.loads((cell / "experiment.json").read_text())["experiment"]["concurrency"])
        for name, metric in METRICS.items():
            base = env["cells"][c][name]["cell"]
            e = compare_cells(cell, base, metric, final=False)
            rows.append({"C": int(c), "metric": name, "point": round(e.point, 3), "lcb95": round(e.lcb95, 3), "verdict": e.verdict,
                         "k_engine": e.k_new, "k_baseline": e.k_base, "baseline": env["cells"][c][name]["candidate_id"]})
    rows.sort(key=lambda r: (r["C"], r["metric"]))
    for r in rows:
        print(f"C={r['C']:<3} {r['metric']:<10} point {r['point']:<6} lcb95 {r['lcb95']:<6} {r['verdict']:<12} vs {r['baseline']}")
    out = sweep / "vs_envelope.json"
    out.write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
