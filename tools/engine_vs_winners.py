"""Development comparison: an engine sweep's measured runs against the frozen tuning winners (point ratios only).

    .venv/bin/python tools/engine_vs_winners.py results/engine_sweeps/<sweep-dir> [results/l20x/campaign/winners.json]

For every cell of the sweep (one concurrency each) and every headline metric: the engine's mean over measured runs,
the winner's tuning value at that C, and the speedup oriented so that > 1 is better (throughput new/base, latencies
base/new).  Not the AC-5 estimate: that needs evaluation-set cells at K >= 5 and bench.stats.compare_cells.
"""

import json
import sys
from pathlib import Path

FIELDS = {"throughput": ("throughput_frames_s", True), "ttfa_p50": ("ttfa_p50_s", False), "ttfa_p95": ("ttfa_p95_s", False),
          "rtf_mean": ("rtf_mean", False)}


def main():
    sweep = Path(sys.argv[1])
    winners = json.loads(Path(sys.argv[2] if len(sys.argv) > 2 else "results/l20x/campaign/winners.json").read_text())
    rows = []
    for cell in sorted(p for p in sweep.iterdir() if (p / "result.json").exists()):
        res = json.loads((cell / "result.json").read_text())
        runs = [r for r in res["runs"] if r.get("measured") and r.get("valid")]
        if not runs:
            rows.append((cell.name, "no valid measured runs", res.get("cell_valid")))
            continue
        c = str(runs[0]["concurrency"])
        for metric, (field, higher) in FIELDS.items():
            new = sum(r[field] for r in runs) / len(runs)
            base = winners["cells"].get(c, {}).get(metric, {}).get("value")
            ratio = (new / base if higher else base / new) if base else float("nan")
            rows.append((f"C={c}", metric, round(new, 4), round(base, 4) if base else None, round(ratio, 3), len(runs)))
    for r in rows:
        print(*r, sep="\t")


if __name__ == "__main__":
    main()
