"""Speedup estimator with complete-run bootstrap.

Each statistic is computed per measured run (throughput from the run's
makespan, TTFA p50/p95 as the run's percentiles, RTF mean as the run's mean).
The point estimate is the mean over K runs and the ratio is formed from the
point estimates.  Uncertainty comes from resampling complete runs,
independently for the engine and for the baseline; individual requests are
never resampled because requests within a run share queueing and batching.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TARGET_SPEEDUP = 2.0
BOOTSTRAP_REPLICATES = 10_000
FINAL_K = 10

#: metric -> True when larger is better
HEADLINE_METRICS = {
    "throughput_frames_s": True,
    "ttfa_p50_s": False,
    "ttfa_p95_s": False,
    "rtf_mean": False,
}


class StatisticsInputError(ValueError):
    pass


def per_run_values(run_summaries: list[dict], metric: str) -> np.ndarray:
    """Extract one value per *measured run* from run summary dicts.

    Rejects request-level inputs: every element must be a run summary (has a
    ``label`` and ``measured`` flag) and must be a measured, valid run.
    """
    if not run_summaries:
        raise StatisticsInputError("no runs")
    vals = []
    labels = set()
    for s in run_summaries:
        if not isinstance(s, dict) or "label" not in s or "measured" not in s or "n_requests" not in s:
            raise StatisticsInputError("statistics take complete-run summaries, not request-level records")
        if "request_id" in s:
            raise StatisticsInputError("request-level record passed to the run-level estimator")
        if not s["measured"]:
            raise StatisticsInputError(f"run {s['label']} is not a measured run")
        if not s.get("valid", False):
            raise StatisticsInputError(f"run {s['label']} is invalid and cannot be used")
        if s["label"] in labels:
            raise StatisticsInputError(f"duplicate run label {s['label']}")
        labels.add(s["label"])
        vals.append(float(s[metric]))
    return np.asarray(vals, dtype=np.float64)


def oriented_ratio(new_mean: float, base_mean: float, higher_is_better: bool) -> float:
    """Speedup >1 means the new engine is better."""
    return new_mean / base_mean if higher_is_better else base_mean / new_mean


def bootstrap_ratio(
    new_vals: np.ndarray,
    base_vals: np.ndarray,
    higher_is_better: bool,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n, b = len(new_vals), len(base_vals)
    new_idx = rng.integers(0, n, size=(replicates, n))
    base_idx = rng.integers(0, b, size=(replicates, b))
    new_m = new_vals[new_idx].mean(axis=1)
    base_m = base_vals[base_idx].mean(axis=1)
    return new_m / base_m if higher_is_better else base_m / new_m


@dataclass
class RatioEstimate:
    metric: str
    point: float
    lcb95: float  # one-sided 95% lower confidence bound
    ci90: tuple[float, float]  # two-sided 90% interval
    k_new: int
    k_base: int
    verdict: str  # "pass" | "inconclusive" | "fail"

    @property
    def interval_too_wide(self) -> bool:
        """Development rule: add runs (up to K=10) if the 90% interval is wider than 10% of the estimate."""
        lo, hi = self.ci90
        return (hi - lo) > 0.10 * self.point


def verdict(point: float, lcb: float, target: float = TARGET_SPEEDUP) -> str:
    if point >= target and lcb >= target:
        return "pass"
    if point >= target:
        return "inconclusive"
    return "fail"


def estimate(
    new_runs: list[dict],
    base_runs: list[dict],
    metric: str,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
    target: float = TARGET_SPEEDUP,
) -> RatioEstimate:
    if metric not in HEADLINE_METRICS:
        raise StatisticsInputError(f"unknown headline metric {metric}")
    hib = HEADLINE_METRICS[metric]
    nv = per_run_values(new_runs, metric)
    bv = per_run_values(base_runs, metric)
    point = oriented_ratio(float(nv.mean()), float(bv.mean()), hib)
    boot = bootstrap_ratio(nv, bv, hib, replicates=replicates, seed=seed)
    lcb = float(np.quantile(boot, 0.05))
    ci = (float(np.quantile(boot, 0.05)), float(np.quantile(boot, 0.95)))
    return RatioEstimate(metric=metric, point=point, lcb95=lcb, ci90=ci, k_new=len(nv), k_base=len(bv), verdict=verdict(point, lcb, target))


@dataclass
class DurationGuard:
    corpus_ratio: float
    tail_fraction: float
    silence_excess_pp: float
    passed: bool
    reasons: list[str]


def duration_guard(
    d_new: dict[str, float],
    d_ref: dict[str, float],
    silence_new: dict[str, float] | None = None,
    silence_ref: dict[str, float] | None = None,
) -> DurationGuard:
    """Paired output-duration guard against D_ref (the cell's throughput winner).

    sum(D_new)/sum(D_ref) must lie in [0.95, 1.05]; at most 2% of rows may have
    D_new,i/D_ref,i outside [0.67, 1.5]; the mean silence fraction may exceed
    the reference's by at most 2 percentage points.
    """
    ids = sorted(set(d_new) & set(d_ref))
    reasons = []
    if set(d_new) != set(d_ref):
        reasons.append("row sets differ")
    if not ids:
        return DurationGuard(float("nan"), float("nan"), float("nan"), False, ["no paired rows"])
    sn = sum(d_new[i] for i in ids)
    sr = sum(d_ref[i] for i in ids)
    corpus = sn / sr if sr > 0 else float("inf")
    if not (0.95 <= corpus <= 1.05):
        reasons.append(f"corpus duration ratio {corpus:.3f} outside [0.95, 1.05]")
    outside = 0
    for i in ids:
        r = d_new[i] / d_ref[i] if d_ref[i] > 0 else float("inf")
        if not (0.67 <= r <= 1.5):
            outside += 1
    tail = outside / len(ids)
    if tail > 0.02:
        reasons.append(f"{tail:.1%} of rows outside [0.67, 1.5] (max 2%)")
    excess = 0.0
    if silence_new is not None and silence_ref is not None:
        excess = 100.0 * (float(np.mean([silence_new[i] for i in ids])) - float(np.mean([silence_ref[i] for i in ids])))
        if excess > 2.0:
            reasons.append(f"silence fraction exceeds reference by {excess:.2f} pp (max 2)")
    return DurationGuard(corpus, tail, excess, not reasons, reasons)


def cell_estimates(new_runs: list[dict], base_runs_by_metric: dict[str, list[dict]], seed: int = 0) -> dict[str, RatioEstimate]:
    """All four headline metrics for one (GPU, C) cell; each metric has its own winning baseline."""
    return {m: estimate(new_runs, base_runs_by_metric[m], m, seed=seed) for m in HEADLINE_METRICS}
