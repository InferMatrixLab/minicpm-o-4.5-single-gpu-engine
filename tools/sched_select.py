"""The round-8 scheduler ablation's predeclared selection rule (round-8-contract.md), applied to its tuning-ID cells.

    .venv/bin/python tools/sched_select.py SWEEP_DIR [SWEEP_DIR ...] [--ref A-legacy] [--candidates C-...,D-...,E-...,F-...]

Every cell is verified from its artifacts (``bench.verify.verify_cell``).  For each candidate label and C in the sweeps,
the oriented ratio candidate / reference (``bench.stats.estimate``, run-level bootstrap, >1 better for the candidate) of
TTFA p50, TTFA p95, throughput and RTF mean.  Rule: one configuration across all C -- maximize the minimum point ratio
over C and the four metrics, among candidates whose every request is valid and whose C=32 mean run-max stall is at most
the reference's; ties within 0.01 go to the lower C=32 mean run-max stall; if no candidate qualifies, the lowest C=32
mean run-max stall.  Writes ``sched_select.json`` into the first sweep directory.

The contract predeclares K=3 per cell, below ``bench.stats``' development minimum (K=5), so each side is checked with
``validate_cell_runs`` (exactly run1..runK, valid, one configuration and workload) and the two sides' workloads are
compared here; the rule uses only the point ratios, and the K=3 lower bounds are printed as indicative only.
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench.stats import WORKLOAD_FIELDS, StatisticsInputError, _estimate_unchecked, validate_cell_runs  # noqa: E402
from bench.verify import verify_cell  # noqa: E402

METRICS = {"ttfa_p50": "ttfa_p50_s", "ttfa_p95": "ttfa_p95_s", "throughput": "throughput_frames_s", "rtf_mean": "rtf_mean"}
TIE = 0.01


def ratio(new: dict, ref: dict, metric: str):
    ni = validate_cell_runs(new["runs"], new["k"], "candidate")
    bi = validate_cell_runs(ref["runs"], ref["k"], "reference")
    for f in WORKLOAD_FIELDS:
        if ni[f] != bi[f]:
            raise StatisticsInputError(f"candidate and reference differ on {f}: {ni[f]} vs {bi[f]}")
    return _estimate_unchecked(new["runs"], ref["runs"], metric)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("sweeps", nargs="+")
    ap.add_argument("--ref", default="A-legacy")
    ap.add_argument("--candidates", default="C-df-floor300,D-df-nofloor,E-lane100-floor100,F-lane100-nofloor")
    ap.add_argument("--stall-c", type=int, default=32)
    a = ap.parse_args(argv)
    cands = a.candidates.split(",")
    cells, problems = {}, []
    for sweep in a.sweeps:
        for cell in sorted(p for p in Path(sweep).iterdir() if (p / "result.json").exists()):
            v = verify_cell(cell)
            if not v.ok:
                problems.append(f"{cell}: {v.problems[:3]}")
                continue
            exp = json.loads((cell / "experiment.json").read_text())["experiment"]
            runs = v.measured_summaries
            cells[(exp["candidate"]["label"], int(exp["concurrency"]))] = {
                "dir": str(cell), "k": v.spec["k"], "runs": runs, "valid": all(r.get("valid") and not r.get("n_failed") for r in runs),
                "stall_mean_run_max": statistics.fmean(r["stall_max_s"] for r in runs), "stall_abs_max": max(r["stall_max_s"] for r in runs)}
    cs = sorted({c for _, c in cells})
    table = {}
    for lab in [x for x in {lab for lab, _ in cells} if x != a.ref]:
        row = {"ratios": {}, "valid": True}
        for c in cs:
            new, ref = cells.get((lab, c)), cells.get((a.ref, c))
            if new is None or ref is None:
                problems.append(f"{lab} C={c}: missing cell")
                row["valid"] = False
                continue
            row["valid"] &= new["valid"]
            row["ratios"][c] = {}
            for name, metric in METRICS.items():
                e = ratio(new, ref, metric)
                row["ratios"][c][name] = {"point": e.point, "lcb95": e.lcb95}
        row["min_point"] = min((x["point"] for r in row["ratios"].values() for x in r.values()), default=None)
        sc = cells.get((lab, a.stall_c))
        row["stall_mean_run_max_c"] = sc["stall_mean_run_max"] if sc else None
        row["stall_abs_max_c"] = sc["stall_abs_max"] if sc else None
        table[lab] = row
    ref_stall = cells[(a.ref, a.stall_c)]["stall_mean_run_max"] if (a.ref, a.stall_c) in cells else None
    eligible = [lab for lab in cands if lab in table and table[lab]["valid"] and table[lab]["min_point"] is not None
                and ref_stall is not None and table[lab]["stall_mean_run_max_c"] <= ref_stall]
    if eligible:
        best = max(table[lab]["min_point"] for lab in eligible)
        tied = [lab for lab in eligible if table[lab]["min_point"] >= best - TIE]
        chosen = min(tied, key=lambda lab: table[lab]["stall_mean_run_max_c"])
        rule = "max-min ratio among eligible" + (f" (tie within {TIE} among {tied}: lower C={a.stall_c} mean run-max stall)" if len(tied) > 1 else "")
    else:
        chosen = min((lab for lab in cands if lab in table), key=lambda lab: table[lab]["stall_mean_run_max_c"])
        rule = f"no candidate met the stall condition: the lowest C={a.stall_c} mean run-max stall"
    for lab in sorted(table):
        r = table[lab]
        print(f"{lab:<22} valid {r['valid']!s:<5} min ratio {r['min_point']:.4f}  C={a.stall_c} stall mean run-max {r['stall_mean_run_max_c']:.3f} "
              f"(abs {r['stall_abs_max_c']:.3f}){'  <- eligible' if lab in eligible else ''}")
        for c, m in r["ratios"].items():
            print(f"    C={c:<3} " + "  ".join(f"{k} {v['point']:.3f}/{v['lcb95']:.3f}" for k, v in m.items()))
    print(f"reference {a.ref}: C={a.stall_c} stall mean run-max {ref_stall}")
    print(f"chosen: {chosen} ({rule})")
    out = {"ref": a.ref, "candidates": cands, "ref_stall_mean_run_max": ref_stall, "table": table, "eligible": eligible, "chosen": chosen,
           "rule": rule, "problems": problems, "cells": {f"{k[0]}@{k[1]}": {x: y for x, y in v.items() if x != "runs"} for k, v in cells.items()}}
    Path(a.sweeps[0], "sched_select.json").write_text(json.dumps(out, indent=1, default=str))
    if problems:
        print("problems:", problems)
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
