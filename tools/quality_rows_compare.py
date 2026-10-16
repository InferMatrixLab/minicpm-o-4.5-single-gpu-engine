"""Per-row comparison of speech quality reports on their common rows (development diagnostics).

    .venv/bin/python tools/quality_rows_compare.py NAME=REPORT.rows.jsonl [NAME=...] [--seed 0]

Prints mean UTMOS / SIM / WER of every report on the rows all of them share (same row_id and seed), and the paired
mean difference of each report against the first, with a bootstrap 90% interval.
"""

import json
import sys

import numpy as np


def load(path, seed):
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r.get("seed_index") == seed:
            out[r["row_id"]] = r
    return out


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seed = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 0
    reps = {a.split("=", 1)[0]: load(a.split("=", 1)[1], seed) for a in args if "=" in a}
    common = sorted(set.intersection(*(set(r) for r in reps.values())))
    names = list(reps)
    print(f"{len(common)} common rows (seed {seed})")
    rng = np.random.default_rng(0)
    for m in ("utmos", "sim", "wer"):
        vals = {n: np.array([float(reps[n][i][m]) for i in common]) for n in names}
        line = "  ".join(f"{n} {vals[n].mean():.4f}" for n in names)
        print(f"{m:6s} {line}")
        base = vals[names[0]]
        for n in names[1:]:
            d = vals[n] - base
            boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            print(f"       {n} - {names[0]}: {d.mean():+.4f} [{np.quantile(boots, 0.05):+.4f}, {np.quantile(boots, 0.95):+.4f}]")


if __name__ == "__main__":
    main()
