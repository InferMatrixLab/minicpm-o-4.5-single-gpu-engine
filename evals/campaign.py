"""L20X baseline campaign: qualification, tuning winners, evaluation and full-set quality.

    python -m evals.campaign specs [--check]
    python -m evals.campaign classify --out results/l20x/campaign/classification.json
    python -m evals.campaign qualify --sweep DIR [--sweep ...] --gpu 2 [--gpu 3] --out results/l20x/campaign/qualification
    python -m evals.campaign select --qualification Q.json --sweep TUNE_DIR [--sweep ...] --out results/l20x/campaign/winners.json
    python -m evals.campaign stage-specs --winners results/l20x/campaign/winners.json [--check]
    python -m evals.campaign envelope --winners W.json --sweep EVAL_DIR [--sweep ...] --out results/l20x/campaign/envelope.json
    python -m evals.campaign dref --winners W.json --role guard|truncation --sweep DIR [...] --out-dir results/l20x/campaign/dref
    python -m evals.campaign fullq-score --sweep FULLQ_DIR [...] --concurrency 32 --d-ref DREF.json --gpu 2 --out-dir results/l20x/campaign/fullq_scored

``specs`` generates the sweeps that do not depend on measured results: one
tuning-quality qualification sweep per seed (every baseline candidate in
MPS-off and MPS-on form plus the sglang-omni multimodal launch), the
tuning-performance sweeps of the qualified candidates (each concurrency on one
device), and the full-set C=32 quality sweeps (one per seed).  ``stage-specs``
generates the sweeps that follow from the frozen winners: the evaluation
sweeps of every winner (K=5, disjoint evaluation IDs) and the full-set C=1
quality sweeps of the C=1 throughput winner.

The campaign inventory is what the executed sweep declarations say must exist
(``sweep.json``, digest-checked); every attempt below a sweep, including
archived ones under ``attempts/``, is verified and either selected (exactly one
verified cell per candidate, seed and C) or recorded as a failed attempt with
its reasons.  ``qualify`` scores each complete 3-seed group, and a campaign row
is only built from a report whose generation identity and source cells are
the group's, and from a qualification whose input report is that report.
``select`` freezes the per-C, per-metric winners from verified tuning cells of
qualified class-R candidates; ``verify_winners`` recomputes them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "sweeps" / "campaign"
EFFECTIVE_DIR = ROOT / "results" / "effective_inputs"
VLLM_MPS_DIR = ROOT / "configs" / "vllm"

VLLM_CONFIGS = ("minicpmo_4_5_turn", "minicpmo_4_5_turn_mrv2", "minicpmo_4_5_turn_mrv2_h200")
DUPLEX_CONFIG = "minicpmo_4_5.yaml"
SEEDS = ({"seed_index": 0, "gpu": 1, "port": 18410}, {"seed_index": 1, "gpu": 2, "port": 18420}, {"seed_index": 2, "gpu": 3, "port": 18430})
RUN_ORDER_SEED = 20261015
SEEDS_REQUIRED = 3
QUALIFY_C = 8
#: Reruns of cells whose server failed (each is recorded; the failed attempt stays in the campaign record).
RERUNS = ({"seed_index": 1, "adapter": "sglang-omni", "label": "mps-on",
           "reason": "run1 startup refused: sglang MPS state left dirty by the run0 teardown (group SIGTERM raced the launcher's ordered retire)"},)
#: Candidates that failed tuning qualification (UTMOS floor, campaign_qualification.json); they are measured no further.
NOT_QUALIFIED = (("vllm-omni", "minicpmo_4_5_turn_mrv2", "mps-on"), ("vllm-omni", "minicpmo_4_5_turn_mrv2_h200", "mps-off"))
TUNE_K = 2
EVAL_K = 5
#: Each concurrency is measured on one device so the per-C winner comparison never crosses devices.
TUNE_LANES = ({"gpu": 1, "port": 18510, "concurrencies": [1, 32]}, {"gpu": 2, "port": 18520, "concurrencies": [2, 16]},
              {"gpu": 3, "port": 18530, "concurrencies": [4, 8]})
CONCURRENCIES = (1, 2, 4, 8, 16, 32)
#: metric -> (run-summary field, higher is better)
METRICS = {"throughput": ("throughput_frames_s", True), "ttfa_p50": ("ttfa_p50_s", False), "ttfa_p95": ("ttfa_p95_s", False),
           "rtf_mean": ("rtf_mean", False)}


class CampaignError(RuntimeError):
    pass


class WinnerError(ValueError):
    pass


# --- candidates and generated specs --------------------------------------------------------------
def candidates() -> list[dict]:
    out = []
    for name in VLLM_CONFIGS:
        out.append({"adapter": "vllm-omni", "config": f"{name}.yaml", "label": "mps-off", "mps": "off"})
        out.append({"adapter": "vllm-omni", "config": str(VLLM_MPS_DIR / f"{name}_mps.yaml"), "label": "mps-on", "mps": "on"})
    out.append({"adapter": "sglang-omni", "config": "speech", "label": "mps-off", "mps": "off"})
    out.append({"adapter": "sglang-omni", "config": "speech", "label": "mps-on", "mps": "on", "env": {"OMNI_SGLANG_MPS": "1"}})
    # Daily-Omni prompts reach ~13k tokens; the multimodal launch needs the longer context
    out.append({"adapter": "sglang-omni", "config": "speech", "label": "mm32k", "mps": "off", "env": {"OMNI_SGLANG_MAX_SEQ": "32768"}})
    return out


def candidate_key(c: dict) -> tuple:
    return (c["adapter"], Path(c["config"]).stem.removesuffix("_mps"), c["label"])


def tuning_candidates() -> list[dict]:
    return [c for c in candidates() if candidate_key(c) not in NOT_QUALIFIED]


def sweep_spec(seed: dict) -> dict:
    return {"description": f"L20X tuning qualification, seed {seed['seed_index']}", "kind": "quality", "id_set": "tuning",
            "seed_index": seed["seed_index"], "gpu": seed["gpu"], "port": seed["port"], "k": 1,
            "run_order_seed": RUN_ORDER_SEED + seed["seed_index"], "timeout_s": 180, "save_audio": True, "concurrencies": [QUALIFY_C],
            "configs": candidates()}


def rerun_spec(i: int, rerun: dict) -> dict:
    seed = next(s for s in SEEDS if s["seed_index"] == rerun["seed_index"])
    spec = sweep_spec(seed)
    spec["description"] = f"L20X tuning qualification, seed {seed['seed_index']}, rerun {i}: {rerun['reason']}"
    spec["configs"] = [c for c in spec["configs"] if c["adapter"] == rerun["adapter"] and c["label"] == rerun["label"]]
    return spec


def tune_spec(lane: dict) -> dict:
    return {"description": f"L20X tuning performance (winner selection), GPU {lane['gpu']}", "kind": "performance", "id_set": "tuning",
            "seed_index": 0, "gpu": lane["gpu"], "port": lane["port"], "k": TUNE_K, "run_order_seed": RUN_ORDER_SEED + 100 + lane["gpu"],
            "timeout_s": 180, "save_audio": True, "concurrencies": lane["concurrencies"], "configs": tuning_candidates()}


def fullq_spec(seed: dict, c: int, configs: list[dict]) -> dict:
    return {"description": f"L20X full-set speech quality at C={c}, seed {seed['seed_index']}", "kind": "quality", "id_set": "full",
            "seed_index": seed["seed_index"], "gpu": seed["gpu"], "port": seed["port"] + 50, "k": 1,
            "run_order_seed": RUN_ORDER_SEED + 200 + 10 * c + seed["seed_index"], "timeout_s": 180, "save_audio": True, "concurrencies": [c],
            "configs": configs}


def all_specs() -> dict[str, dict]:
    out = {f"qualify_seed{s['seed_index']}.json": sweep_spec(s) for s in SEEDS}
    for i, r in enumerate(RERUNS, 1):
        out[f"qualify_seed{r['seed_index']}_rerun{i}.json"] = rerun_spec(i, r)
    for lane in TUNE_LANES:
        out[f"tune_gpu{lane['gpu']}.json"] = tune_spec(lane)
    for s in SEEDS:
        out[f"fullq_c32_seed{s['seed_index']}.json"] = fullq_spec(s, 32, tuning_candidates())
    return out


def _lane_of(c: int) -> dict:
    return next(lane for lane in TUNE_LANES if c in lane["concurrencies"])


def _digest(c: dict) -> str:
    from bench.cli import make_candidate

    return make_candidate(c).digest


def stage_specs(winners: dict) -> dict[str, dict]:
    """Sweeps that follow from frozen winners: evaluation of every winner, and C=1 full-set quality of the C=1 throughput winner."""
    by_digest = {_digest(c): c for c in tuning_candidates()}
    out = {}
    for c in CONCURRENCIES:
        digests = sorted({w["candidate_digest"] for w in winners["cells"][str(c)].values()})
        lane = _lane_of(c)
        out[f"eval_c{c}.json"] = {"description": f"L20X evaluation of the C={c} winners", "kind": "performance", "id_set": "eval", "seed_index": 0,
                                  "gpu": lane["gpu"], "port": lane["port"] + 5, "k": EVAL_K, "run_order_seed": RUN_ORDER_SEED + 300 + c,
                                  "timeout_s": 180, "save_audio": True, "concurrencies": [c], "configs": [by_digest[d] for d in digests]}
    top = by_digest[winners["cells"]["1"]["throughput"]["candidate_digest"]]
    for s in SEEDS:
        out[f"fullq_c1_seed{s['seed_index']}.json"] = fullq_spec(s, 1, [top])
    return out


def _write_or_check(specs: dict[str, dict], check: bool) -> int:
    bad = []
    for name, want in specs.items():
        p = SPEC_DIR / name
        if check:
            if not p.exists() or json.loads(p.read_text()) != want:
                bad.append(str(p))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(want, indent=1))  # byte-identical to the executed specs
    if bad:
        print(f"specs differ from the generator: {bad}")
        return 1
    return 0


def cmd_specs(a) -> int:
    return _write_or_check(all_specs(), a.check)


def cmd_stage_specs(a) -> int:
    return _write_or_check(stage_specs(verify_winners(a.winners)), a.check)


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# --- inventory and attempts ---------------------------------------------------------------------
def sweep_record(sw: str | Path) -> dict:
    from bench.experiment import digest

    p = Path(sw) / "sweep.json"
    if not p.exists():
        raise CampaignError(f"{sw}: no sweep declaration (sweep.json)")
    rec = json.loads(p.read_text())
    if digest(rec.get("spec")) != rec.get("spec_digest"):
        raise CampaignError(f"{sw}: sweep declaration does not match its digest")
    return rec


def inventory(sweeps: list[str]) -> dict[tuple, dict]:
    """(candidate digest, seed, C) -> declaring sweeps and candidate, from the digest-checked sweep declarations."""
    from bench.cli import make_candidate

    if not sweeps:
        raise CampaignError("no sweeps given")
    inv: dict[tuple, dict] = {}
    for sw in sweeps:
        spec = sweep_record(sw)["spec"]
        for c in spec["configs"]:
            cand = make_candidate(c)
            for conc in spec["concurrencies"]:
                key = (cand.digest, int(spec.get("seed_index", 0)), int(conc))
                inv.setdefault(key, {"candidate_id": cand.candidate_id, "sweeps": [], "kind": spec.get("kind", "performance"),
                                     "id_set": spec.get("id_set", "eval"), "gpu": int(spec["gpu"]), "k": int(spec["k"])})["sweeps"].append(str(sw))
    return inv


def attempts(sweeps: list[str]) -> list[dict]:
    """Every attempt below the sweeps (current and archived), each verified from its artifacts."""
    from bench.verify import verify_cell

    out = []
    for sw in sweeps:
        exps = sorted(Path(sw).glob("*/experiment.json")) + sorted(Path(sw).glob("attempts/*/experiment.json"))
        for exp in exps:
            d = exp.parent
            spec = json.loads(exp.read_text()).get("experiment") or {}
            v = verify_cell(d)
            out.append({"cell": str(d), "sweep": str(sw), "archived": d.parent.name == "attempts", "digest": (spec.get("candidate") or {}).get("digest"),
                        "candidate_id": (spec.get("candidate") or {}).get("candidate_id"), "seed_index": spec.get("seed_index"),
                        "concurrency": spec.get("concurrency"), "kind": spec.get("kind"), "id_set": spec.get("id_set"), "ok": v.ok,
                        "gpu": (spec.get("hardware") or {}).get("gpu_index"), "k": spec.get("k"), "k_updates": spec.get("k_updates") or [],
                        "problems": v.problems[:3], "spec": v.spec if v.ok else None})
    return out


def select_cells(sweeps: list[str]) -> tuple[dict, dict[tuple, dict], dict]:
    """(inventory, {(digest, seed, C): verified attempt}, {digest: [failed attempts]})."""
    inv = inventory(sweeps)
    chosen: dict[tuple, list[dict]] = {}
    failed: dict[str, list[dict]] = {}
    for a in attempts(sweeps):
        key = (a["digest"], a["seed_index"], a["concurrency"])
        declared = inv.get(key)
        schedule_ok = declared is not None and (a["k"] == declared["k"] or (a["k_updates"] and declared["k"] < a["k"] <= 10
                                                                            and a["k_updates"][0].get("from") == declared["k"]))
        if a["ok"] and declared is not None and a["sweep"] in declared["sweeps"] and a["kind"] == declared["kind"] and a["id_set"] == declared["id_set"] \
                and a["gpu"] == declared["gpu"] and schedule_ok:
            chosen.setdefault(key, []).append(a)
            continue
        reason = a["problems"] if not a["ok"] else ["cell does not match its sweep's declaration (candidate, seed, C, kind, ID set, GPU, K)"]
        failed.setdefault(a["digest"], []).append({k: a[k] for k in ("cell", "archived", "seed_index", "concurrency")} | {"reasons": reason})
    for key, cs in chosen.items():
        if len(cs) > 1:
            raise CampaignError(f"candidate {inv[key]['candidate_id']} seed {key[1]} C={key[2]} has several verified cells: {[c['cell'] for c in cs]}")
    return inv, {k: v[0] for k, v in chosen.items()}, failed


def group_cells(sweeps: list[str], concurrency: int = QUALIFY_C) -> dict[str, dict]:
    """digest -> {"candidate", "cells": {seed: cell_dir}, "failed_attempts", "expected_seeds"} over the declared inventory at one C."""
    inv, chosen, failed = select_cells(sweeps)
    groups: dict[str, dict] = {}
    for (dg, seed, c), meta in sorted(inv.items()):
        if c != concurrency:
            continue
        g = groups.setdefault(dg, {"candidate": {"digest": dg, "candidate_id": meta["candidate_id"]}, "cells": {}, "expected_seeds": [],
                                   "failed_attempts": failed.get(dg, [])})
        g["expected_seeds"].append(seed)
        if (dg, seed, c) in chosen:
            g["cells"][seed] = chosen[(dg, seed, c)]["cell"]
            g["candidate"] = chosen[(dg, seed, c)]["spec"]["candidate"]
    return groups


# --- qualification ------------------------------------------------------------------------------
def _slug(cand: dict) -> str:
    return f"{cand['adapter']}_{Path(cand['config']).stem}_{cand.get('label') or 'x'}_{cand['digest'][:12]}"


def _run(cmd: list[str], log: Path) -> int:
    with log.open("a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode


def report_join_problems(vr, cand: dict, cells: dict) -> list[str]:
    """A verified report belongs to ``cand`` and was scored from exactly ``cells`` ({seed: dir})."""
    gen = vr.report.get("generation") or {}
    out = []
    if gen.get("candidate_digest") != cand["digest"] or (gen.get("candidate") or {}).get("digest") != cand["digest"]:
        out.append(f"report generation candidate {gen.get('candidate_id')} is not {cand.get('candidate_id')}")
    got = {int(s): str(Path(v["cell"]).resolve()) for s, v in (vr.report.get("seeds") or {}).items()}
    want = {int(s): str(Path(c).resolve()) for s, c in cells.items()}
    if got != want:
        out.append("report source cells are not the selected cells")
    return out


def _qualify_one(cand: dict, cells: dict, gpu: int, out: Path, log: Path) -> None:
    from evals.speech.verify_report import ReportError, verify_report

    d = out / _slug(cand)
    d.mkdir(parents=True, exist_ok=True)
    quality = d / "speech_quality.json"
    reuse = False
    if quality.exists():
        try:
            vr = verify_report(quality)
            reuse = vr.ok and not report_join_problems(vr, cand, cells)
        except ReportError:
            reuse = False
    if not reuse:  # resumable: only a verified report of exactly these cells is reused; anything else is rescored
        quality.unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "evals.speech.suite", "score", "--scope", "development", "--gpu", str(gpu), "--out", str(quality)]
        for s in sorted(cells):
            cmd += ["--cell", cells[s]]
        if _run(cmd, log) != 0:
            return
    _run([sys.executable, "-m", "evals.speech.qref", "qualify", "--quality", str(quality), "--effective-dir", str(EFFECTIVE_DIR),
          "--out", str(d / "qualification.json")], log)


def record(cand: dict, cells: dict, out: Path, expected_seeds: list[int] | None = None) -> dict:
    """Campaign row for one candidate, derived from re-verified, joined evidence."""
    from evals.speech.qref import QRefError, verify_qualification
    from evals.speech.verify_report import ReportError, verify_report

    row = {"candidate_id": cand["candidate_id"], "digest": cand["digest"], "adapter": cand.get("adapter"), "config": cand.get("config"),
           "label": cand.get("label"), "mps": cand.get("mps"), "cells": {str(s): c for s, c in sorted(cells.items())}}
    expected = sorted(expected_seeds) if expected_seeds is not None else list(range(SEEDS_REQUIRED))
    if expected != list(range(SEEDS_REQUIRED)) or sorted(cells) != expected:
        return {**row, "status": "incomplete", "reason": f"verified seeds {sorted(cells)} of declared {expected}"}
    d = out / _slug(cand)
    quality, qual = d / "speech_quality.json", d / "qualification.json"
    if not quality.exists():
        return {**row, "status": "unscored", "reason": "speech scoring failed (see log)"}
    try:
        vr = verify_report(quality)
    except ReportError as e:
        return {**row, "status": "unverified", "reason": [str(e)]}
    problems = vr.problems + report_join_problems(vr, cand, cells)
    if problems:
        return {**row, "status": "unverified", "reason": problems[:3]}
    pooled = vr.summary["pooled"]
    row["quality"] = {"path": str(quality), "sha256": _sha(quality),
                      **{k: pooled[k] for k in ("rows", "wer_corpus", "wer_filtered_corpus", "sim_mean", "utmos_mean", "catastrophic_count")}}
    try:
        q = verify_qualification(str(qual))
    except (QRefError, FileNotFoundError, KeyError) as e:
        return {**row, "status": "unqualified", "reason": f"qualification does not verify: {e}"}
    recorded = json.loads(qual.read_text())
    if q["candidate_digest"] != cand["digest"] or recorded["inputs"]["quality"]["sha256"] != _sha(quality) or \
            Path(recorded["inputs"]["quality"]["path"]).resolve() != quality.resolve():
        return {**row, "status": "unqualified", "reason": "qualification was not computed from this candidate's report"}
    row["qualification"] = {"path": str(qual), "sha256": _sha(qual), "floors_passed": q["floors"]["passed"],
                            "effective_inputs_eligible": q["effective_inputs_eligible"], "class_r": (q["class_r"] or {}).get("class_r"),
                            "class_r_unknown_reason": q["class_r_unknown_reason"], "per_seed_floors": q["per_seed_floors"]}
    return {**row, "status": "qualified" if q["qualified"] else "not-qualified"}


def campaign_rows(sweeps: list[str], out: Path) -> list[dict]:
    groups = group_cells(sweeps)
    if not groups:
        raise CampaignError("the sweeps declare no candidates")
    return [{**record(g["candidate"], g["cells"], out, g["expected_seeds"]), "failed_attempts": g["failed_attempts"]}
            for g in sorted(groups.values(), key=lambda g: g["candidate"]["candidate_id"])]


ROW_DECISIVE = ("candidate_id", "digest", "cells", "status", "quality", "qualification", "failed_attempts")
_VERIFIED: dict[tuple, tuple] = {}


def _stamp(paths) -> str:
    """Content-change stamp of every file below ``paths`` (path, size, mtime): a cached verification is reused only if it is unchanged."""
    import os

    h = hashlib.sha256()
    for root in sorted(str(Path(p).resolve()) for p in paths):
        if os.path.isfile(root):
            st = os.stat(root)
            h.update(f"{root}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
            continue
        for d, dirs, files in os.walk(root):
            dirs.sort()
            for f in sorted(files):
                fp = os.path.join(d, f)
                try:
                    st = os.stat(fp)
                except FileNotFoundError:
                    continue
                h.update(f"{fp}\0{st.st_size}\0{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def verify_campaign(path: str | Path) -> dict:
    """Recompute every campaign row from the recorded sweeps; decisive fields must agree."""
    path = Path(path)
    doc = json.loads(path.read_text())
    deps = list(doc["sweeps"]) + [path.parent, EFFECTIVE_DIR]
    key = (str(path.resolve()), _sha(path))
    if key in _VERIFIED and _VERIFIED[key][0] == _stamp(deps):
        return _VERIFIED[key][1]
    stamp = _stamp(deps)
    rows = json.loads(json.dumps(campaign_rows(doc["sweeps"], path.parent)))
    want = {r["digest"]: {k: r.get(k) for k in ROW_DECISIVE} for r in rows}
    got = {r["digest"]: {k: r.get(k) for k in ROW_DECISIVE} for r in doc.get("candidates", [])}
    if want != got:
        bad = sorted(d for d in set(want) | set(got) if want.get(d) != got.get(d))
        raise CampaignError(f"{path}: recorded rows differ from the recomputed campaign for {bad[:3]}")
    _VERIFIED[key] = (stamp, doc)
    return doc


def cmd_qualify(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "campaign.log"
    groups = group_cells(a.sweep)
    if not groups:
        raise CampaignError("the sweeps declare no candidates")
    work = [g for g in groups.values() if sorted(g["cells"]) == list(range(SEEDS_REQUIRED)) == sorted(g["expected_seeds"])]
    lanes = {gpu: work[i::len(a.gpu)] for i, gpu in enumerate(a.gpu)}  # one scorer per GPU at a time

    def lane(gpu, items):
        for g in items:
            _qualify_one(g["candidate"], g["cells"], gpu, out, log)

    threads = [threading.Thread(target=lane, args=(gpu, items)) for gpu, items in lanes.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = campaign_rows(a.sweep, out)
    doc = {"scope": "development", "id_set": "tuning", "sweeps": [str(s) for s in a.sweep], "effective_dir": str(EFFECTIVE_DIR), "candidates": rows}
    (out / "campaign_qualification.json").write_text(json.dumps(doc, indent=1))
    for r in rows:
        q = r.get("quality") or {}
        print(f"{r['status']:>13}  {r['candidate_id']:<70} WERf={q.get('wer_filtered_corpus')} SIM={q.get('sim_mean')} UTMOS={q.get('utmos_mean')}"
              f"  {r.get('reason') or ''}")
    return 0 if rows and all(r["status"] in ("qualified", "not-qualified") for r in rows) else 1


# --- classification (including the default duplex profile) -------------------------------------
def classification_candidates() -> list[dict]:
    return candidates() + [{"adapter": "vllm-omni", "config": DUPLEX_CONFIG, "label": "duplex-default", "mps": "off"}]


def cmd_classify(a) -> int:
    from bench.cli import make_candidate
    from bench.semantics import SemanticsUnknown, class_r, resolve_vllm_configs

    resolve_vllm_configs([c["config"] for c in classification_candidates() if c["adapter"] == "vllm-omni"])
    rows = []
    for c in classification_candidates():
        cand = make_candidate(c)
        ident = {**cand.identity(), "candidate_id": cand.candidate_id, "digest": cand.digest}
        try:
            cr = class_r(ident, canonical_inputs=True).to_json()  # input equality is decided at qualification
            rows.append({"candidate_id": cand.candidate_id, "digest": cand.digest, "semantics": cr, "class_r_semantics": cr["class_r"]})
        except SemanticsUnknown as e:
            rows.append({"candidate_id": cand.candidate_id, "digest": cand.digest, "semantics": None, "unknown": str(e), "class_r_semantics": False})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"candidates": rows}, indent=1))
    for r in rows:
        print(f"{r['candidate_id']:<70} class_r={r['class_r_semantics']} {((r.get('semantics') or {}).get('class_r_violations')) or r.get('unknown') or ''}")
    return 0


# --- winners ------------------------------------------------------------------------------------
def compute_winners(qualification: str | Path, sweeps: list[str]) -> dict:
    """Per-C, per-metric winners over verified tuning cells of qualified class-R candidates."""
    from bench.stats import StatisticsInputError, per_run_values
    from bench.verify import verify_cell
    from evals.speech.verify_report import gpu_class

    qual = verify_campaign(qualification)
    rows = {r["digest"]: r for r in qual["candidates"]}
    inv, chosen, failed = select_cells(sweeps)
    eligible = {d for d, r in rows.items() if r["status"] == "qualified" and (r.get("qualification") or {}).get("class_r") is True}
    excluded = {rows[d]["candidate_id"]: f"{r['status']}; class_r={(r.get('qualification') or {}).get('class_r')}" for d, r in rows.items() if d not in eligible}
    values: dict[str, dict] = {str(c): {} for c in CONCURRENCIES}
    classes, failures = set(), []
    for (dg, seed, c), meta in sorted(inv.items()):
        if meta["kind"] != "performance" or meta["id_set"] != "tuning" or seed != 0:
            raise WinnerError(f"{meta['candidate_id']}: winners are selected from seed-0 tuning performance cells")
        if dg not in rows:
            raise WinnerError(f"tuning candidate {meta['candidate_id']} has no qualification row")
        if (dg, seed, c) not in chosen:
            failures.append({"candidate_id": meta["candidate_id"], "concurrency": c, "reason": "no verified tuning cell",
                             "attempts": [f for f in failed.get(dg, []) if f["concurrency"] == c]})
            continue
        a = chosen[(dg, seed, c)]
        v = verify_cell(a["cell"])
        if int(v.spec["k"]) != TUNE_K:
            failures.append({"candidate_id": meta["candidate_id"], "concurrency": c, "reason": f"tuning cell K={v.spec['k']}, the tuning schedule is K={TUNE_K}"})
            continue
        classes.add(json.dumps(gpu_class(v.spec["hardware"]), sort_keys=True))
        entry = {"candidate_id": meta["candidate_id"], "cell": a["cell"], "k": int(v.spec["k"]),
                 "experiment_digest": json.loads((Path(a["cell"]) / "experiment.json").read_text())["experiment_digest"]}
        try:
            for m, (field, _) in METRICS.items():
                entry[m] = float(np.mean(per_run_values(v.measured_summaries, field)))
        except StatisticsInputError as e:
            failures.append({"candidate_id": meta["candidate_id"], "concurrency": c, "reason": str(e)})
            continue
        values[str(c)][dg] = entry
    for dg, fs in sorted(failed.items()):  # failed attempts that were later rerun stay on record
        for f in fs:
            if not any(x.get("attempts") and f in x["attempts"] for x in failures):
                failures.append({"candidate_id": (rows.get(dg) or {}).get("candidate_id", dg), "concurrency": f["concurrency"],
                                 "reason": "failed attempt", "attempts": [f]})
    if len(classes) != 1:
        raise WinnerError(f"tuning cells span {len(classes)} hardware classes")
    cells = {}
    for c in CONCURRENCIES:
        pool = {d: e for d, e in values[str(c)].items() if d in eligible}
        if not pool:
            raise WinnerError(f"C={c}: no qualified class-R candidate has a verified tuning cell")
        missing = sorted(rows[d]["candidate_id"] for d in eligible if d not in pool)
        if missing:
            raise WinnerError(f"C={c}: eligible candidates without a verified tuning cell: {missing}")
        cells[str(c)] = {}
        for m, (_, higher) in METRICS.items():
            best = (max if higher else min)(sorted(pool), key=lambda d: pool[d][m])
            cells[str(c)][m] = {"candidate_digest": best, "candidate_id": pool[best]["candidate_id"], "value": pool[best][m], "cell": pool[best]["cell"]}
    return {"schema": "winners/v1", "gpu_class": json.loads(classes.pop()), "selected_on": "tuning",
            "metrics": {m: ("max" if h else "min") for m, (_, h) in METRICS.items()},
            "qualification": {"path": str(qualification), "sha256": _sha(qualification)}, "sweeps": [str(s) for s in sweeps],
            "eligible": sorted(rows[d]["candidate_id"] for d in eligible), "excluded": excluded, "values": values, "cells": cells,
            "failures": failures}


_WINNERS: dict[tuple, tuple] = {}


def verify_winners(path: str | Path, check_sources: bool = True) -> dict:
    """Recompute the winner declaration from its recorded qualification and tuning sweeps."""
    path = Path(path)
    doc = json.loads(path.read_text())
    key = (str(path.resolve()), _sha(path))
    q0 = (doc.get("qualification") or {}).get("path")
    deps = list(doc.get("sweeps") or []) + ([q0, Path(q0).parent] if q0 else []) + [EFFECTIVE_DIR]
    if key in _WINNERS and _WINNERS[key][0] == _stamp(deps):
        return _WINNERS[key][1]
    stamp = _stamp(deps)
    if doc.get("schema") != "winners/v1":
        raise WinnerError(f"{path} is not a winners/v1 declaration")
    q = doc.get("qualification") or {}
    if not Path(q.get("path", "")).exists() or _sha(q["path"]) != q.get("sha256"):
        raise WinnerError("the qualification record changed since the winners were frozen")
    try:
        again = json.loads(json.dumps(compute_winners(q["path"], doc["sweeps"])))
    except CampaignError as e:
        raise WinnerError(str(e)) from e
    bad = [k for k in ("gpu_class", "eligible", "excluded", "values", "cells", "failures") if again[k] != doc.get(k)]
    if bad:
        raise WinnerError(f"{path}: recorded {bad} differ from the recomputed winners")
    _WINNERS[key] = (stamp, doc)
    return doc


def cmd_select(a) -> int:
    out = Path(a.out)
    if out.exists():
        raise WinnerError(f"{out} exists: frozen winners are never rewritten")
    doc = compute_winners(a.qualification, a.sweep)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    for c in CONCURRENCIES:
        print(f"C={c:<3}", "  ".join(f"{m}={w['candidate_id']}({w['value']:.4g})" for m, w in doc["cells"][str(c)].items()))
    for f in doc["failures"]:
        print("failure:", f)
    return 0


# --- evaluation envelope and duration references --------------------------------------------------
def _run_mean_ci(vals: np.ndarray, replicates: int = 10000, seed: int = 0) -> tuple[float, float]:
    """Two-sided 90% interval of the mean over complete runs (runs are resampled, never requests)."""
    rng = np.random.default_rng(seed)
    means = vals[rng.integers(0, len(vals), size=(replicates, len(vals)))].mean(axis=1)
    return float(np.quantile(means, 0.05)), float(np.quantile(means, 0.95))


def compute_envelope(winners_path: str | Path, sweeps: list[str]) -> dict:
    """The provisional class-R envelope: every (C, metric) winner measured on the evaluation IDs, from verified cells only."""
    from bench.eligibility import ConfigClass, EnvelopeEntry, validate_envelope
    from bench.semantics import SemanticsUnknown, class_r
    from bench.stats import StatisticsInputError, per_run_values
    from bench.verify import verify_cell
    from evals.speech.verify_report import gpu_class

    w = verify_winners(winners_path)
    rows = {r["digest"]: r for r in verify_campaign(w["qualification"]["path"])["candidates"]}
    inv, chosen, failed = select_cells(sweeps)
    cells, extend, entries = {}, set(), []
    for c in CONCURRENCIES:
        cells[str(c)] = {}
        for m, (field, _) in METRICS.items():
            win = w["cells"][str(c)][m]
            key = (win["candidate_digest"], 0, c)
            if key not in chosen:
                raise WinnerError(f"C={c} {m}: winner {win['candidate_id']} has no verified evaluation cell")
            v = verify_cell(chosen[key]["cell"])
            spec = v.spec
            if spec["kind"] != "performance" or spec["id_set"] != "eval" or not EVAL_K <= int(spec["k"]) <= 10:
                raise WinnerError(f"C={c} {m}: {chosen[key]['cell']} is not a K=5..10 evaluation performance cell")
            if gpu_class(spec["hardware"]) != w["gpu_class"]:
                raise WinnerError(f"C={c} {m}: evaluation hardware class differs from the tuning class")
            try:
                cr = class_r(spec["candidate"], bool((rows[win["candidate_digest"]].get("qualification") or {}).get("effective_inputs_eligible")))
            except SemanticsUnknown as e:
                raise WinnerError(f"C={c} {m}: {e}") from e
            try:
                vals = per_run_values(v.measured_summaries, field)
            except StatisticsInputError as e:
                raise WinnerError(f"C={c} {m}: {e}") from e
            mean, (lo, hi) = float(vals.mean()), _run_mean_ci(vals)
            wide = (hi - lo) > 0.10 * mean
            if wide and int(spec["k"]) < 10:
                extend.add((c, win["candidate_digest"], chosen[key]["cell"]))
            entries.append(EnvelopeEntry(gpu=spec["hardware"]["gpu_sku"], concurrency=c, metric=m, config=cr, value=mean,
                                         qualified=rows[win["candidate_digest"]]["status"] == "qualified", provisional=True))
            cells[str(c)][m] = {"candidate_id": win["candidate_id"], "candidate_digest": win["candidate_digest"], "cell": chosen[key]["cell"],
                                "experiment_digest": json.loads((Path(chosen[key]["cell"]) / "experiment.json").read_text())["experiment_digest"],
                                "k": int(spec["k"]), "runs": [float(x) for x in vals], "mean": mean, "ci90": [lo, hi], "interval_too_wide": wide,
                                "tuning_value": win["value"]}
    validate_envelope(entries, w["gpu_class"]["gpu_sku"])
    fails = [{"candidate_digest": d, **f} for d, fs in sorted(failed.items()) for f in fs]
    return {"schema": "envelope/v1", "provisional": True, "gpu_class": w["gpu_class"], "winners": {"path": str(winners_path), "sha256": _sha(winners_path)},
            "sweeps": [str(s) for s in sweeps], "cells": cells, "extend": sorted([list(e) for e in extend]), "failures": fails}


def envelope_markdown(doc: dict) -> str:
    units = {"throughput": "frames/s", "ttfa_p50": "s", "ttfa_p95": "s", "rtf_mean": ""}
    out = [f"# Provisional class-R envelope ({doc['gpu_class']['gpu_sku']}, driver {doc['gpu_class']['gpu_driver']})", "",
           "Winners were selected on the tuning IDs and measured on the evaluation IDs (fresh server per run, run0 discarded). "
           "Each value is the mean over K measured runs with its 90% run-bootstrap interval.", "",
           "| C | " + " | ".join(f"{m} ({units[m]})" if units[m] else m for m in METRICS) + " |", "|---|" + "---|" * len(METRICS)]
    for c in CONCURRENCIES:
        row = []
        for m in METRICS:
            e = doc["cells"][str(c)][m]
            row.append(f"{e['mean']:.4g} [{e['ci90'][0]:.4g}, {e['ci90'][1]:.4g}] K={e['k']}<br>{e['candidate_id']}")
        out.append(f"| {c} | " + " | ".join(row) + " |")
    out += ["", f"Cells whose interval exceeds 10% of the estimate (extend to K<=10): {doc['extend'] or 'none'}", "",
            f"Failed or superseded attempts: {len(doc['failures'])}"] + [f"- {f['cell']}: {f['reasons'][:1]}" for f in doc["failures"]]
    return "\n".join(out) + "\n"


def cmd_envelope(a) -> int:
    doc = compute_envelope(a.winners, a.sweep)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1))
    out.with_suffix(".md").write_text(envelope_markdown(doc))
    print(envelope_markdown(doc))
    return 0


def cmd_dref(a) -> int:
    """Duration references of each throughput winner: the guard from its evaluation cell, truncation from its full-set cells."""
    from evals.speech.verify_report import build_dref

    w = verify_winners(a.winners)
    inv, chosen, _ = select_cells(a.sweep)
    out = Path(a.out_dir)
    rc = 0
    for c in (a.concurrency or (CONCURRENCIES if a.role == "guard" else (32, 1))):
        dg = w["cells"][str(c)]["throughput"]["candidate_digest"]
        want = DREF_SEEDS[a.role]
        cells = [chosen[(dg, s, c)]["cell"] for s in want if (dg, s, c) in chosen]
        if len(cells) != len(want):
            print(f"C={c}: throughput winner {w['cells'][str(c)]['throughput']['candidate_id']} lacks verified cells for seeds {want}")
            rc = 1
            continue
        doc = build_dref(cells, out / f"dref_{a.role}_c{c}.json", a.role, a.winners)
        print(f"C={c}: {a.role} D_ref of {doc['candidate_id']} over {len(doc['rows'])} rows")
    return rc


DREF_SEEDS = {"guard": [0], "truncation": [0, 1, 2]}


def cmd_fullq_score(a) -> int:
    """Score every candidate's three full-set seeds at one C with that C's verified truncation D_ref (final scope)."""
    inv, chosen, failed = select_cells(a.sweep)
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "fullq_score.log"
    digests = sorted({dg for (dg, s, c) in inv if c == a.concurrency})
    rc = 0
    for dg in digests:
        cells = [chosen.get((dg, s, a.concurrency)) for s in range(SEEDS_REQUIRED)]
        cid = inv[(dg, 0, a.concurrency)]["candidate_id"] if (dg, 0, a.concurrency) in inv else dg
        if any(c is None for c in cells):
            print(f"{cid}: incomplete full-set seeds {[s for s, c in enumerate(cells) if c is None]}; failed attempts {failed.get(dg, [])[:2]}")
            rc = 1
            continue
        cand = cells[0]["spec"]["candidate"]
        dst = out / f"{_slug(cand)}_c{a.concurrency}.json"
        if dst.exists():
            from evals.speech.verify_report import ReportError, verify_report

            try:
                vr = verify_report(dst)
                same_ref = (vr.report.get("d_ref") or {}).get("sha256") == _sha(a.d_ref) if Path(a.d_ref).exists() else False
                if vr.ok and vr.report.get("scope") == "final" and same_ref and not report_join_problems(vr, cand, {s: c["cell"] for s, c in enumerate(cells)}):
                    print(f"{cid}: verified report exists, reused")
                    continue
            except ReportError:
                pass
            dst.unlink()
        cmd = [sys.executable, "-m", "evals.speech.suite", "score", "--scope", "final", "--gpu", str(a.gpu), "--out", str(dst), "--d-ref", a.d_ref]
        for c in cells:
            cmd += ["--cell", c["cell"]]
        if _run(cmd, log) != 0:
            print(f"{cid}: scoring failed (see {log})")
            rc = 1
        else:
            print(f"{cid}: scored -> {dst}")
    return rc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("specs")
    sp.add_argument("--check", action="store_true", help="fail if the committed specs differ from the generator")
    cl = sub.add_parser("classify")
    cl.add_argument("--out", required=True)
    q = sub.add_parser("qualify")
    q.add_argument("--sweep", action="append", required=True)
    q.add_argument("--gpu", type=int, action="append", required=True)
    q.add_argument("--out", required=True)
    se = sub.add_parser("select")
    se.add_argument("--qualification", required=True)
    se.add_argument("--sweep", action="append", required=True)
    se.add_argument("--out", required=True)
    ss = sub.add_parser("stage-specs")
    ss.add_argument("--winners", required=True)
    ss.add_argument("--check", action="store_true")
    ev = sub.add_parser("envelope")
    ev.add_argument("--winners", required=True)
    ev.add_argument("--sweep", action="append", required=True)
    ev.add_argument("--out", required=True)
    dr = sub.add_parser("dref")
    dr.add_argument("--winners", required=True)
    dr.add_argument("--role", choices=sorted(DREF_SEEDS), required=True)
    dr.add_argument("--sweep", action="append", required=True)
    dr.add_argument("--out-dir", required=True)
    dr.add_argument("--concurrency", type=int, action="append", default=None)
    fq = sub.add_parser("fullq-score")
    fq.add_argument("--sweep", action="append", required=True)
    fq.add_argument("--concurrency", type=int, required=True)
    fq.add_argument("--d-ref", required=True)
    fq.add_argument("--gpu", type=int, required=True)
    fq.add_argument("--out-dir", required=True)
    a = ap.parse_args(argv)
    return {"fullq-score": cmd_fullq_score, "specs": cmd_specs, "classify": cmd_classify, "qualify": cmd_qualify, "select": cmd_select, "stage-specs": cmd_stage_specs,
            "envelope": cmd_envelope, "dref": cmd_dref}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
