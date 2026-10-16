"""Two engine configurations of one scheduler-comparison sweep against each other, per C and metric (diagnostic).

    .venv/bin/python tools/sched_compare.py NEW_LABEL BASE_LABEL SWEEP_DIR [SWEEP_DIR ...] [--envelope results/l20x/campaign/envelope.json]

Each pair of cells (same C, labels NEW_LABEL and BASE_LABEL) is verified from its artifacts and compared with
``bench.stats.compare_cells`` (run-level bootstrap of the oriented ratio NEW over BASE, >1 is better for NEW; development
mode, the declared K).  Also the means over the measured runs of the stall and inter-chunk gap metrics, and, when an
envelope is given, both cells' point and lower bound against the envelope's winner cell.  Writes ``sched_compare.json``
into the first sweep directory.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.stats import compare_cells  # noqa: E402

METRICS = {"throughput": "throughput_frames_s", "ttfa_p50": "ttfa_p50_s", "ttfa_p95": "ttfa_p95_s", "rtf_mean": "rtf_mean"}
EXTRA = ("stall_mean_s", "stall_max_s", "inter_chunk_gap_p95_s", "inter_chunk_gap_max_s", "e2e_p95_s")


def _cells(sweeps):
    out = {}
    for sweep in sweeps:
        for cell in sorted(p for p in Path(sweep).iterdir() if (p / "result.json").exists()):
            exp = json.loads((cell / "experiment.json").read_text())["experiment"]
            out[(exp["candidate"]["label"], int(exp["concurrency"]))] = cell
    return out


def _mean(cell, key):
    res = json.loads((cell / "result.json").read_text())
    return statistics.fmean(r[key] for r in res["measured_runs"])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("new")
    ap.add_argument("base")
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("--envelope")
    a = ap.parse_args(argv)
    cells = _cells(a.sweeps)
    env = json.loads(Path(a.envelope).read_text()) if a.envelope else None
    rows = []
    for c in sorted({c for _, c in cells}):
        new, base = cells.get((a.new, c)), cells.get((a.base, c))
        if new is None or base is None:
            print(f"C={c}: missing {'new' if new is None else 'base'} cell")
            continue
        row = {"C": c, "new_cell": str(new), "base_cell": str(base), "new_vs_base": {}, "means": {}, "vs_envelope": {}}
        for name, metric in METRICS.items():
            e = compare_cells(new, base, metric, final=False)
            row["new_vs_base"][name] = {"point": round(e.point, 4), "lcb95": round(e.lcb95, 4), "k": [e.k_new, e.k_base]}
            row["means"][name] = {a.new: _mean(new, metric), a.base: _mean(base, metric)}
            if env:
                ref = env["cells"][str(c)][name]["cell"]
                row["vs_envelope"][name] = {lab: {"point": round(x.point, 3), "lcb95": round(x.lcb95, 3)}
                                            for lab, x in ((a.new, compare_cells(new, ref, metric, final=False)),
                                                           (a.base, compare_cells(base, ref, metric, final=False)))}
        for k in EXTRA:
            row["means"][k] = {a.new: _mean(new, k), a.base: _mean(base, k)}
        rows.append(row)
        print(f"C={c}")
        for name in METRICS:
            r, m = row["new_vs_base"][name], row["means"][name]
            env_s = ""
            if env:
                v = row["vs_envelope"][name]
                env_s = "  vs envelope " + ", ".join(f"{lab} {v[lab]['point']}/{v[lab]['lcb95']}" for lab in (a.new, a.base))
            print(f"  {name:<10} {a.new}/{a.base} point {r['point']:<7} lcb95 {r['lcb95']:<7} means {m[a.new]:.4g} vs {m[a.base]:.4g}{env_s}")
        print("  " + "  ".join(f"{k} {row['means'][k][a.new]:.3f}/{row['means'][k][a.base]:.3f}" for k in EXTRA))
    out = Path(a.sweeps[0]) / "sched_compare.json"
    out.write_text(json.dumps({"new": a.new, "base": a.base, "sweeps": a.sweeps, "envelope": a.envelope, "rows": rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
