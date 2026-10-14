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
DEV_MIN_K = 5

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
    ``label`` and ``measured`` flag) and must be a measured, valid run with a
    finite, positive value.
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
        v = float(s[metric])
        if not np.isfinite(v) or v <= 0:
            raise StatisticsInputError(f"run {s['label']} has non-finite or non-positive {metric}={v}")
        vals.append(v)
    return np.asarray(vals, dtype=np.float64)


#: Workload identity: must agree within a side and between engine and baseline.
WORKLOAD_FIELDS = ("gpu", "concurrency", "id_set", "ids_sha256", "seed_index", "n_requests", "timeout_s")
#: Configuration identity: must be constant within a side (one declared configuration).
CONFIG_FIELDS = ("engine", "config", "candidate_id", "config_digest", "cell_digest", "experiment_base_digest", "mps_policy")
REQUIRED_TIMEOUT_S = 180.0


def _check_run(r: dict, side: str) -> None:
    from bench.workload import n_eval

    lab = r.get("label")
    if not r.get("strict", False):
        raise StatisticsInputError(f"{side}: run {lab} was not measured with mandatory geometry (strict)")
    if not r.get("ids_sha256") or r.get("ids_sha256") != r.get("expected_ids_sha256"):
        raise StatisticsInputError(f"{side}: run {lab} measured IDs differ from the declared ID set")
    if r.get("id_set") == "eval" and r.get("n_requests") != n_eval(int(r.get("concurrency", 0))):
        raise StatisticsInputError(f"{side}: run {lab} has {r.get('n_requests')} requests, N_eval is {n_eval(int(r.get('concurrency', 0)))}")
    if float(r.get("timeout_s", 0)) != REQUIRED_TIMEOUT_S:
        raise StatisticsInputError(f"{side}: run {lab} used timeout {r.get('timeout_s')} s (required {REQUIRED_TIMEOUT_S})")
    policy = r.get("cache_policy")
    ev = r.get("cache_evidence") or {}
    if policy == "restart":
        if ev.get("fresh_server") is not True or not ev.get("server_pid") or not (r.get("server_startup_s") or 0) > 0:
            raise StatisticsInputError(f"{side}: run {lab} claims a fresh server without launch evidence")
    elif policy == "flush":
        if ev.get("empty") is not True:
            raise StatisticsInputError(f"{side}: run {lab} has no verified cache flush")
    else:
        raise StatisticsInputError(f"{side}: run {lab} did not start with verified-cold caches")
    mps = (r.get("runtime_evidence") or {}).get("mps") or {}
    if mps.get("observed") != r.get("mps_policy"):
        raise StatisticsInputError(
            f"{side}: run {lab} declared MPS {r.get('mps_policy')!r} but observed {mps.get('observed', 'nothing')!r}: {mps.get('reason', '')}"
        )


def validate_cell_runs(runs: list[dict], expected_k: int, side: str) -> dict:
    """One complete, comparable experiment: exactly run1..runK, valid, one configuration, one workload."""
    labels = sorted((r.get("label") for r in runs), key=str)
    want = sorted((f"run{i}" for i in range(1, expected_k + 1)), key=str)
    if labels != want:
        raise StatisticsInputError(f"{side}: expected measured runs {want}, got {labels} (failed runs cannot be omitted)")
    ident = {}
    for f in WORKLOAD_FIELDS + CONFIG_FIELDS:
        vals = {json_key(r.get(f)) for r in runs}
        if len(vals) != 1 or None in vals or "" in vals:
            raise StatisticsInputError(f"{side}: runs disagree on or lack {f}: {sorted(map(str, vals))}")
        ident[f] = runs[0].get(f)
    for r in runs:
        if not r.get("measured") or not r.get("valid"):
            raise StatisticsInputError(f"{side}: run {r.get('label')} is not a valid measured run")
        _check_run(r, side)
    pids = [(r.get("cache_evidence") or {}).get("server_pid") for r in runs if r.get("cache_policy") == "restart"]
    if len(pids) != len(set(pids)):
        raise StatisticsInputError(f"{side}: restart runs share a server process")
    return ident


def json_key(v):
    return v if isinstance(v, (str, int, float, type(None))) else str(v)


def validate_comparison(new_runs: list[dict], base_runs: list[dict], final: bool, new_k: int, base_k: int) -> None:
    """Final confirmation: predeclared K=10 on both sides.  Development: K=5..10 complete runs.

    ``new_k`` / ``base_k`` are the run counts declared by each side's frozen
    experiment spec, so a failed tail run cannot be dropped from the list.
    """
    for side, runs, k in (("engine", new_runs, new_k), ("baseline", base_runs, base_k)):
        if final and k != FINAL_K:
            raise StatisticsInputError(f"{side}: final confirmation requires a declared K={FINAL_K}, got {k}")
        if not final and not (DEV_MIN_K <= k <= FINAL_K):
            raise StatisticsInputError(f"{side}: development comparisons declare {DEV_MIN_K}..{FINAL_K} runs, got {k}")
    ni = validate_cell_runs(new_runs, new_k, "engine")
    bi = validate_cell_runs(base_runs, base_k, "baseline")
    for f in WORKLOAD_FIELDS:
        if ni[f] != bi[f]:
            raise StatisticsInputError(f"engine and baseline differ on {f}: {ni[f]} vs {bi[f]}")


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
    *,
    final: bool,
    new_k: int,
    base_k: int,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = 0,
    target: float = TARGET_SPEEDUP,
) -> RatioEstimate:
    """Validated estimate: ``final=True`` applies the final-confirmation rules, ``False`` the development rules."""
    validate_comparison(new_runs, base_runs, final=final, new_k=new_k, base_k=base_k)
    return _estimate_unchecked(new_runs, base_runs, metric, replicates=replicates, seed=seed, target=target)


def _estimate_unchecked(
    new_runs: list[dict], base_runs: list[dict], metric: str, replicates: int = BOOTSTRAP_REPLICATES, seed: int = 0, target: float = TARGET_SPEEDUP
) -> RatioEstimate:
    """Numerical core only; never use it for reported results."""
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
    silence_new: dict[str, float],
    silence_ref: dict[str, float],
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
    missing = [i for i in ids if i not in silence_new or i not in silence_ref]
    if missing:
        reasons.append(f"silence fractions missing for {len(missing)} rows")
        excess = float("nan")
    else:
        excess = 100.0 * (float(np.mean([silence_new[i] for i in ids])) - float(np.mean([silence_ref[i] for i in ids])))
        if not np.isfinite(excess) or excess > 2.0:
            reasons.append(f"silence fraction exceeds reference by {excess:.2f} pp (max 2)")
    return DurationGuard(corpus, tail, excess, not reasons, reasons)


def compare_cells(new_cell_dir, base_cell_dir, metric: str, final: bool, seed: int = 0) -> RatioEstimate:
    """The reporting entry point: verify both cells from their artifacts, then estimate with their declared K."""
    from bench.verify import verify_cell

    vn, vb = verify_cell(new_cell_dir), verify_cell(base_cell_dir)
    for side, v in (("engine", vn), ("baseline", vb)):
        if not v.ok:
            raise StatisticsInputError(f"{side} cell {v.cell_dir} failed verification: {v.problems[:5]}")
        if v.spec.get("kind") != "performance" or v.spec.get("id_set") != "eval":
            raise StatisticsInputError(f"{side} cell {v.cell_dir} is a {v.spec.get('kind')}/{v.spec.get('id_set')} experiment, not a performance evaluation")
        if final and (v.spec.get("k_updates") or int(v.spec.get("k", 0)) != FINAL_K):
            raise StatisticsInputError(f"{side} cell {v.cell_dir}: final confirmation needs a predeclared fixed K={FINAL_K} without extensions")
    return estimate(vn.measured_summaries, vb.measured_summaries, metric, final=final, new_k=vn.spec["k"], base_k=vb.spec["k"], seed=seed)


def needs_more_runs(e: RatioEstimate) -> bool:
    """Development policy: start at K=5; add runs up to K=10 while the 90% interval is wider than 10%."""
    return e.interval_too_wide and min(e.k_new, e.k_base) < FINAL_K
