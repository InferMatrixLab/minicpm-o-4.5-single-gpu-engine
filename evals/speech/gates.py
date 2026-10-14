"""Speech quality gates.

Absolute floors (baseline qualification and engine sanity) come from
sglang-omni ``tests/test_model/omni_ci_config.py`` at the pinned commit:
filtered WER <= 0.0215 * 1.25, SIM >= 42.867 * 0.97, UTMOS >= 4.2922 * 0.97.

Comparative gates hold the exact headline configuration against a Q_ref that
was frozen before any engine measurement, on identical rows, seed policy and
denominators.  Each metric is averaged over the seeds.  If the seed-to-seed
noise (max - min over seeds, on either side) exceeds a metric's margin, that
metric is "inconclusive", not "pass".
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FLOOR_FILTERED_WER = 0.0215 * 1.25
FLOOR_SIM = 42.867 * 0.97
FLOOR_UTMOS = 4.2922 * 0.97

# metric -> (direction, margin, unit). "max": engine <= ref + margin; "min": engine >= ref - margin.
COMPARATIVE = {
    "wer_corpus": ("max", 0.005, "fraction"),  # +0.5 pp
    "catastrophic_count": ("max", 2, "rows"),
    "sim_mean": ("min", 1.0, "x100"),
    "utmos_mean": ("min", 0.05, "MOS"),
    "silence_rate": ("max", 0.005, "fraction"),
    "repetition_rate": ("max", 0.005, "fraction"),
    "truncation_rate": ("max", 0.005, "fraction"),
}


def absolute_floors(m: dict) -> dict:
    checks = {
        "wer_filtered_corpus": (m["wer_filtered_corpus"], FLOOR_FILTERED_WER, m["wer_filtered_corpus"] <= FLOOR_FILTERED_WER),
        "sim_mean": (m["sim_mean"], FLOOR_SIM, m["sim_mean"] >= FLOOR_SIM),
        "utmos_mean": (m["utmos_mean"], FLOOR_UTMOS, m["utmos_mean"] >= FLOOR_UTMOS),
    }
    return {
        "passed": all(ok for _, _, ok in checks.values()),
        "checks": {k: {"value": v, "floor": f, "passed": ok} for k, (v, f, ok) in checks.items()},
    }


@dataclass
class QRef:
    """Frozen speech reference: per metric, the best value among qualified class-R baselines on one GPU."""

    gpu: str
    rows_sha256: str
    seed_indices: tuple
    values: dict  # metric -> best seed-mean value
    per_seed: dict  # metric -> list of seed values of the config that supplied the best value
    sources: dict  # metric -> candidate_id
    frozen_unix: float

    def to_json(self) -> dict:
        d = asdict(self)
        d["seed_indices"] = list(self.seed_indices)
        return d

    @classmethod
    def load(cls, path: Path) -> "QRef":
        d = json.loads(Path(path).read_text())
        d["seed_indices"] = tuple(d["seed_indices"])
        return cls(**d)


def seed_mean(per_seed: list[dict], metric: str) -> tuple[float | None, list]:
    vals = [s.get(metric) for s in per_seed]
    if any(v is None or (isinstance(v, float) and math.isnan(v)) for v in vals):
        return None, vals
    return float(np.mean(vals)), vals


def freeze_qref(candidates: dict[str, dict], gpu: str, now: float) -> QRef:
    """``candidates``: candidate_id -> {"qualified": bool, "rows_sha256", "seed_indices", "per_seed": [corpus_metrics per seed]}."""
    qualified = {k: v for k, v in candidates.items() if v["qualified"]}
    if not qualified:
        raise ValueError("no qualified baseline to freeze Q_ref from")
    rows = {v["rows_sha256"] for v in qualified.values()}
    seeds = {tuple(v["seed_indices"]) for v in qualified.values()}
    if len(rows) != 1 or len(seeds) != 1:
        raise ValueError("Q_ref candidates were scored on different rows or seed policies")
    values, per_seed, sources = {}, {}, {}
    for metric, (direction, _, _) in COMPARATIVE.items():
        best = None
        for cid, v in qualified.items():
            mean, vals = seed_mean(v["per_seed"], metric)
            if mean is None:
                continue
            better = best is None or (mean < best[0] if direction == "max" else mean > best[0])
            if better:
                best = (mean, vals, cid)
        if best is not None:
            values[metric], per_seed[metric], sources[metric] = best
    return QRef(gpu=gpu, rows_sha256=rows.pop(), seed_indices=seeds.pop(), values=values, per_seed=per_seed, sources=sources, frozen_unix=now)


def comparative_gates(engine_per_seed: list[dict], rows_sha256: str, seed_indices: tuple, qref: QRef) -> dict:
    if rows_sha256 != qref.rows_sha256:
        raise ValueError("engine and Q_ref were scored on different rows")
    if tuple(seed_indices) != tuple(qref.seed_indices):
        raise ValueError("engine and Q_ref use different seed policies")
    out = {}
    for metric, (direction, margin, unit) in COMPARATIVE.items():
        ref = qref.values.get(metric)
        eng, vals = seed_mean(engine_per_seed, metric)
        if ref is None or eng is None:
            out[metric] = {"verdict": "not_evaluated", "engine": eng, "q_ref": ref}
            continue
        noise = max(float(np.ptp(vals)) if len(vals) > 1 else 0.0, float(np.ptp(qref.per_seed[metric])) if len(qref.per_seed[metric]) > 1 else 0.0)
        excess = (eng - ref) if direction == "max" else (ref - eng)  # how much worse than Q_ref
        if noise > margin:
            # too noisy to certify a pass; a regression beyond margin + noise is still a failure
            verdict = "fail" if excess > margin + noise else "inconclusive"
        else:
            verdict = "pass" if excess <= margin else "fail"
        out[metric] = {"verdict": verdict, "engine": eng, "engine_per_seed": vals, "q_ref": ref, "margin": margin, "unit": unit,
                       "direction": direction, "seed_noise": noise}
    verdicts = [v["verdict"] for v in out.values()]
    overall = "fail" if "fail" in verdicts else ("inconclusive" if "inconclusive" in verdicts or "not_evaluated" in verdicts else "pass")
    return {"overall": overall, "metrics": out}
