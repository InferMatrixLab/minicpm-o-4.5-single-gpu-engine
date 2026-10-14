"""L20X baseline qualification campaign (task7, stage 1).

    python -m evals.campaign specs [--check]
    python -m evals.campaign qualify --sweep DIR [--sweep ...] --gpu 2 [--gpu 3] --out results/l20x/campaign/qualification

``specs`` generates one quality sweep per seed from the candidate list below
(every baseline candidate in MPS-off and MPS-on form, plus the sglang-omni
multimodal launch); seeds are sharded across GPUs of one hardware class.

``qualify`` groups the verified cells of the per-seed sweeps by candidate
digest, scores each complete 3-seed group with the speech suite (development
scope) and qualifies it against the bound effective-input dumps.  The campaign
record is built from re-verified evidence only: every report goes through
``verify_report`` and every qualification through ``verify_qualification``;
recorded verdicts are never copied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "sweeps" / "campaign"
EFFECTIVE_DIR = ROOT / "results" / "effective_inputs"
VLLM_MPS_DIR = ROOT / "configs" / "vllm"

VLLM_CONFIGS = ("minicpmo_4_5_turn", "minicpmo_4_5_turn_mrv2", "minicpmo_4_5_turn_mrv2_h200")
SEEDS = ({"seed_index": 0, "gpu": 1, "port": 18410}, {"seed_index": 1, "gpu": 2, "port": 18420}, {"seed_index": 2, "gpu": 3, "port": 18430})
RUN_ORDER_SEED = 20261015
SEEDS_REQUIRED = 3


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


def sweep_spec(seed: dict) -> dict:
    return {"description": f"L20X tuning qualification, seed {seed['seed_index']}", "kind": "quality", "id_set": "tuning",
            "seed_index": seed["seed_index"], "gpu": seed["gpu"], "port": seed["port"], "k": 1,
            "run_order_seed": RUN_ORDER_SEED + seed["seed_index"], "timeout_s": 180, "save_audio": True, "concurrencies": [8],
            "configs": candidates()}


def cmd_specs(a) -> int:
    bad = []
    for seed in SEEDS:
        p = SPEC_DIR / f"qualify_seed{seed['seed_index']}.json"
        want = sweep_spec(seed)
        if a.check:
            if not p.exists() or json.loads(p.read_text()) != want:
                bad.append(str(p))
        else:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(json.dumps(want, indent=1) + "\n")
    if bad:
        print(f"specs differ from the generator: {bad}")
        return 1
    return 0


def _sha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def group_cells(sweeps: list[str]) -> dict[str, dict]:
    """digest -> {"candidate": identity, "cells": {seed_index: cell_dir}}; a seed seen twice for one digest is an error."""
    groups: dict[str, dict] = {}
    for sw in sweeps:
        for exp in sorted(Path(sw).glob("*/experiment.json")):
            spec = json.loads(exp.read_text())["experiment"]
            cand = spec["candidate"]
            g = groups.setdefault(cand["digest"], {"candidate": cand, "cells": {}})
            seed = int(spec["seed_index"])
            if seed in g["cells"]:
                raise SystemExit(f"candidate {cand['candidate_id']}: seed {seed} appears in {g['cells'][seed]} and {exp.parent}")
            g["cells"][seed] = str(exp.parent)
    return groups


def _slug(cand: dict) -> str:
    return f"{cand['adapter']}_{Path(cand['config']).stem}_{cand.get('label') or 'x'}_{cand['digest'][:12]}"


def _run(cmd: list[str], log: Path) -> int:
    with log.open("a") as fh:
        fh.write(f"$ {' '.join(cmd)}\n")
        fh.flush()
        return subprocess.run(cmd, cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT).returncode


def _qualify_one(cand: dict, cells: dict, gpu: int, out: Path, log: Path) -> None:
    from evals.speech.verify_report import verify_report

    d = out / _slug(cand)
    d.mkdir(parents=True, exist_ok=True)
    quality = d / "speech_quality.json"
    if not (quality.exists() and verify_report(quality).ok):  # resumable: a verified report is reused, anything else is rescored
        quality.unlink(missing_ok=True)
        cmd = [sys.executable, "-m", "evals.speech.suite", "score", "--scope", "development", "--gpu", str(gpu), "--out", str(quality)]
        for s in sorted(cells):
            cmd += ["--cell", cells[s]]
        if _run(cmd, log) != 0:
            return
    _run([sys.executable, "-m", "evals.speech.qref", "qualify", "--quality", str(quality), "--effective-dir", str(EFFECTIVE_DIR),
          "--out", str(d / "qualification.json")], log)


def record(cand: dict, cells: dict, out: Path) -> dict:
    """Campaign row for one candidate, derived from re-verified evidence."""
    from evals.speech.qref import QRefError, verify_qualification
    from evals.speech.verify_report import verify_report

    row = {"candidate_id": cand["candidate_id"], "digest": cand["digest"], "adapter": cand["adapter"], "config": cand["config"],
           "label": cand.get("label"), "mps": cand.get("mps"), "cells": {str(s): c for s, c in sorted(cells.items())}}
    if sorted(cells) != list(range(SEEDS_REQUIRED)):
        return {**row, "status": "incomplete", "reason": f"seeds {sorted(cells)}"}
    d = out / _slug(cand)
    quality, qual = d / "speech_quality.json", d / "qualification.json"
    if not quality.exists():
        return {**row, "status": "unscored", "reason": "speech scoring failed (see log)"}
    vr = verify_report(quality)
    if not vr.ok:
        return {**row, "status": "unverified", "reason": vr.problems[:3]}
    pooled = vr.summary["pooled"]
    row["quality"] = {"path": str(quality), "sha256": _sha(quality),
                      **{k: pooled[k] for k in ("rows", "wer_corpus", "wer_filtered_corpus", "sim_mean", "utmos_mean", "catastrophic_count")}}
    try:
        q = verify_qualification(str(qual))
    except (QRefError, FileNotFoundError, KeyError) as e:
        return {**row, "status": "unqualified", "reason": f"qualification does not verify: {e}"}
    row["qualification"] = {"path": str(qual), "sha256": _sha(qual), "floors_passed": q["floors"]["passed"],
                            "effective_inputs_eligible": q["effective_inputs_eligible"], "class_r": (q["class_r"] or {}).get("class_r"),
                            "class_r_unknown_reason": q["class_r_unknown_reason"], "per_seed_floors": q["per_seed_floors"]}
    return {**row, "status": "qualified" if q["qualified"] else "not-qualified"}


def cmd_qualify(a) -> int:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "campaign.log"
    groups = group_cells(a.sweep)
    work = [g for g in groups.values() if sorted(g["cells"]) == list(range(SEEDS_REQUIRED))]
    lanes = {gpu: work[i::len(a.gpu)] for i, gpu in enumerate(a.gpu)}  # one scorer per GPU at a time

    def lane(gpu, items):
        for g in items:
            _qualify_one(g["candidate"], g["cells"], gpu, out, log)

    threads = [threading.Thread(target=lane, args=(gpu, items)) for gpu, items in lanes.items()]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    rows = [record(g["candidate"], g["cells"], out) for g in sorted(groups.values(), key=lambda g: g["candidate"]["candidate_id"])]
    doc = {"scope": "development", "id_set": "tuning", "sweeps": [str(s) for s in a.sweep], "effective_dir": str(EFFECTIVE_DIR), "candidates": rows}
    (out / "campaign_qualification.json").write_text(json.dumps(doc, indent=1))
    for r in rows:
        q = r.get("quality") or {}
        print(f"{r['status']:>13}  {r['candidate_id']:<70} WERf={q.get('wer_filtered_corpus')} SIM={q.get('sim_mean')} UTMOS={q.get('utmos_mean')}"
              f"  {r.get('reason') or ''}")
    return 0 if all(r["status"] in ("qualified", "not-qualified") for r in rows) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.campaign")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("specs")
    sp.add_argument("--check", action="store_true", help="fail if the committed specs differ from the generator")
    q = sub.add_parser("qualify")
    q.add_argument("--sweep", action="append", required=True)
    q.add_argument("--gpu", type=int, action="append", required=True)
    q.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return cmd_specs(a) if a.cmd == "specs" else cmd_qualify(a)


if __name__ == "__main__":
    raise SystemExit(main())
