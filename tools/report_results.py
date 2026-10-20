"""Build the published results from ``evidence/``: per-cell means, the envelope ratios and the comparison figure.

    python tools/report_results.py evidence/20261019-l20x

Reads every cell's measured run summaries (``run1`` .. ``runK``; ``run0`` is the discarded first run) and the saved
envelope comparison (``campaign/r11c_vs_envelope.json``, produced by ``tools/engine_vs_envelope.py`` on the raw cells).
Writes ``docs/current-results.json``, ``docs/figures/full-comparison.svg`` and prints the README table.
"""

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
METRICS = ("throughput_frames_s", "ttfa_p50_s", "ttfa_p95_s", "rtf_mean")
FRAMEWORKS = ("engine", "vllm-omni", "sglang-omni")
LABEL = {"engine": "minicpm-o-4.5-single-gpu-engine", "vllm-omni": "vllm-omni", "sglang-omni": "sglang-omni"}
COLOR = {"engine": "#d1495b", "vllm-omni": "#2e86ab", "sglang-omni": "#3b9b5a"}


def load_cells(ev: Path) -> list[dict]:
    cells = []
    for d in sorted(p for p in ev.iterdir() if (p / "experiment.json").exists()):
        exp = json.loads((d / "experiment.json").read_text())["experiment"]
        runs = []
        for s in sorted(d.glob("run*/summary.json")):
            r = json.loads(s.read_text())
            if r["measured"]:
                if not r["valid"]:
                    raise SystemExit(f"{s}: invalid measured run")
                runs.append(r)
        cand = runs[0]["candidate_id"]
        cells.append({"dir": d.name, "framework": runs[0]["engine"], "candidate_id": cand, "config": runs[0]["config"],
                      "mps": runs[0]["mps_policy"], "C": exp["concurrency"], "k": len(runs), "n_requests": runs[0]["n_requests"],
                      "failed_requests": sum(r["n_failed"] for r in runs),
                      **{m: statistics.fmean(r[m] for r in runs) for m in METRICS},
                      "runs": {m: [r[m] for r in runs] for m in METRICS}})
    return cells


def best_per_framework(cells: list[dict]) -> dict:
    """Per (framework, C): the evaluated cell with the highest mean throughput (the one row shown per framework)."""
    out = {}
    for c in cells:
        key = (c["framework"], c["C"])
        if key not in out or c["throughput_frames_s"] > out[key]["throughput_frames_s"]:
            out[key] = c
    return out


def svg_chart(rows: dict, cs: list[int]) -> str:
    panels = [("throughput_frames_s", "Throughput (frames/s)", 1, True), ("ttfa_p50_s", "TTFA p50 (ms)", 1000, False),
              ("ttfa_p95_s", "TTFA p95 (ms)", 1000, False), ("rtf_mean", "RTF (mean)", 1, False)]
    W, H, pw, ph, ml, mt = 1040, 330, 210, 200, 58, 52
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="Helvetica, Arial, sans-serif" font-size="11">',
           f'<rect width="{W}" height="{H}" fill="#ffffff"/>',
           '<text x="20" y="24" font-size="15" font-weight="bold" fill="#222">MiniCPM-o 4.5 speech streaming on one L20X '
           '(Seed-TTS en, K=5 runs per point)</text>']
    for i, (m, title, scale, up) in enumerate(panels):
        x0, y0 = ml + i * (pw + 50), mt
        vals = [rows[(f, c)][m] * scale for f in FRAMEWORKS for c in cs if (f, c) in rows]
        hi = max(vals) * 1.08
        out.append(f'<text x="{x0 + pw / 2}" y="{y0 - 8}" text-anchor="middle" font-weight="bold" fill="#222">{title}'
                   f'{" ↑" if up else " ↓"}</text>')
        out.append(f'<rect x="{x0}" y="{y0}" width="{pw}" height="{ph}" fill="none" stroke="#999"/>')
        for t in range(5):
            v = hi * t / 4
            y = y0 + ph - ph * t / 4
            out.append(f'<line x1="{x0}" x2="{x0 + pw}" y1="{y:.1f}" y2="{y:.1f}" stroke="#eee"/>')
            out.append(f'<text x="{x0 - 4}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{v:.0f}</text>' if hi > 10 else
                       f'<text x="{x0 - 4}" y="{y + 4:.1f}" text-anchor="end" fill="#555">{v:.2f}</text>')
        xs = {c: x0 + pw * (j + 0.5) / len(cs) for j, c in enumerate(cs)}
        for c in cs:
            out.append(f'<text x="{xs[c]:.1f}" y="{y0 + ph + 14}" text-anchor="middle" fill="#555">{c}</text>')
        out.append(f'<text x="{x0 + pw / 2}" y="{y0 + ph + 30}" text-anchor="middle" fill="#555">concurrency C</text>')
        for f in FRAMEWORKS:
            pts = [(xs[c], y0 + ph - ph * rows[(f, c)][m] * scale / hi) for c in cs if (f, c) in rows]
            out.append(f'<polyline fill="none" stroke="{COLOR[f]}" stroke-width="2" points="{" ".join(f"{x:.1f},{y:.1f}" for x, y in pts)}"/>')
            out += [f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{COLOR[f]}"/>' for x, y in pts]
    lx = ml
    for f in FRAMEWORKS:
        out.append(f'<rect x="{lx}" y="{H - 22}" width="14" height="4" fill="{COLOR[f]}"/>')
        out.append(f'<text x="{lx + 20}" y="{H - 17}" fill="#222">{LABEL[f]}</text>')
        lx += 250
    out.append("</svg>")
    return "\n".join(out) + "\n"


def main():
    ev = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "evidence" / "20261019-l20x").resolve()
    cells = load_cells(ev)
    ratios = json.loads((ev / "campaign" / "r11c_vs_envelope.json").read_text())
    rows = best_per_framework(cells)
    cs = sorted({c["C"] for c in cells})
    (ROOT / "docs" / "current-results.json").write_text(json.dumps({"evidence": str(ev.relative_to(ROOT)), "cells": cells,
                                                                    "engine_vs_envelope": ratios}, indent=1) + "\n")
    (ROOT / "docs" / "figures").mkdir(parents=True, exist_ok=True)
    (ROOT / "docs" / "figures" / "full-comparison.svg").write_text(svg_chart(rows, cs))
    print("| C | Framework | Config | Throughput (frames/s) | Throughput / SGLang | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |")
    print("|---:|---|---|---:|---:|---:|---:|---:|")
    for c in cs:
        sg = rows.get(("sglang-omni", c))
        for f in FRAMEWORKS:
            r = rows.get((f, c))
            if r is None:
                print(f"| {c} | {LABEL[f]} | not evaluated at this C | — | — | — | — | — |")
                continue
            cfg = r["candidate_id"].split(":")[1].removesuffix("-r11c")
            print(f"| {c} | {LABEL[f]} | `{cfg}` | {r['throughput_frames_s']:.2f} | "
                  f"{r['throughput_frames_s'] / sg['throughput_frames_s']:.2f}× | {r['ttfa_p50_s'] * 1e3:.1f} | "
                  f"{r['ttfa_p95_s'] * 1e3:.1f} | {r['rtf_mean']:.4f} |")
    print()
    print("| C | Throughput | TTFA p50 | TTFA p95 | RTF (mean) |")
    print("|---:|---:|---:|---:|---:|")
    by = {(r["C"], r["metric"]): r for r in ratios}
    for c in cs:
        cell = [by[(c, m)] for m in ("throughput", "ttfa_p50", "ttfa_p95", "rtf_mean")]
        print(f"| {c} | " + " | ".join((f"**{r['point']:.2f}× / {r['lcb95']:.2f}**" if r["verdict"] == "pass" else f"{r['point']:.2f}× / {r['lcb95']:.2f}")
                                       for r in cell) + " |")


if __name__ == "__main__":
    main()
