"""Calibrate injected quality violations on fixed baseline outputs.

    python -m evals.speech.calibrate --cell SEED0_CELL --cell SEED1_CELL --cell SEED2_CELL --gpu 2 --out results/quality/calibration

The clean baseline outputs (three seeds) are scored once; Q_ref is frozen from
them.  Each injection is applied at a ladder of magnitudes to the same rows and
seeds, all fixtures are scored in one pass, and for each injection the smallest
magnitude whose effect on its target metric exceeds the gate margin by at least
twice the measured seed-to-seed noise is selected.  That fixture must fail its
comparative gate; the selected magnitudes are recorded.

Injections:
* silence substitution - a fraction of rows get 2.5 s of silence inserted mid-clip
  (target metric: silence_rate);
* speaker swap - a fraction of rows are scored against another speaker's reference
  (target: sim_mean);
* additive noise - white noise at a given SNR on every row (target: utmos_mean).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evals.speech.gates import COMPARATIVE, comparative_gates, freeze_qref  # noqa: E402
from evals.speech.suite import GenRow, build_quality, collect, rows_sha256, score, summarize  # noqa: E402

LADDERS = {
    "silence_substitution": ("silence_rate", [0.01, 0.02, 0.05, 0.10]),
    "speaker_swap": ("sim_mean", [0.05, 0.10, 0.20, 0.40]),
    "additive_noise": ("utmos_mean", [30.0, 20.0, 15.0, 10.0]),  # SNR dB, decreasing = stronger
}
INSERT_SILENCE_S = 2.5


def _pick(rows: list[GenRow], frac: float, rng: np.random.Generator) -> set[str]:
    ids = sorted({r.row_id for r in rows})
    n = max(1, int(round(frac * len(ids))))
    return set(rng.choice(ids, size=n, replace=False).tolist())


def inject(kind: str, mag: float, rows: list[GenRow], work: Path, seed: int = 0) -> list[GenRow]:
    rng = np.random.default_rng(seed)
    out_dir = work / kind / f"{mag:g}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    if kind == "speaker_swap":
        chosen = _pick(rows, mag, rng)
        refs = sorted({r.ref_wav for r in rows})
        for r in rows:
            ref = r.ref_wav
            if r.row_id in chosen:
                others = [x for x in refs if x != r.ref_wav]
                ref = others[int(rng.integers(len(others)))]
            out.append(replace(r, key=f"{kind}@{mag:g}|{r.key}", ref_wav=ref))
        return out
    chosen = _pick(rows, mag, rng) if kind == "silence_substitution" else None
    for r in rows:
        pcm, sr = sf.read(r.wav, dtype="float32", always_2d=False)
        dur = r.duration_s
        if kind == "silence_substitution" and r.row_id in chosen:
            mid = len(pcm) // 2
            pcm = np.concatenate([pcm[:mid], np.zeros(int(INSERT_SILENCE_S * sr), dtype=np.float32), pcm[mid:]])
            dur = len(pcm) / sr
        elif kind == "additive_noise":
            power = float(np.mean(pcm.astype(np.float64) ** 2)) or 1e-12
            noise = rng.standard_normal(len(pcm)) * np.sqrt(power / (10 ** (mag / 10)))
            pcm = np.clip(pcm + noise.astype(np.float32), -1, 1)
        path = out_dir / f"{r.seed_index}_{r.row_id}.wav"
        sf.write(path, pcm, sr, subtype="PCM_16")
        out.append(replace(r, key=f"{kind}@{mag:g}|{r.key}", wav=str(path), duration_s=dur))
    return out


def _restore_keys(rows: list[GenRow]) -> list[GenRow]:
    return [replace(r, key=r.key.split("|", 1)[1]) for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="evals.speech.calibrate")
    ap.add_argument("--cell", action="append", required=True, help="verified cell directory (three seeds)")
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    col = collect([Path(c) for c in a.cell])
    clean = col.rows
    seeds = tuple(sorted({r.seed_index for r in clean}))
    if len(seeds) < 3:
        raise SystemExit("calibration needs three seeds for the noise measurement")
    fixtures = {(k, m): inject(k, m, clean, out / "fixtures") for k, (_, ladder) in LADDERS.items() for m in ladder}
    all_rows = clean + [r for rows in fixtures.values() for r in rows]
    t0 = time.time()
    scored = score(all_rows, a.gpu, whisper=False)

    def summary_for(rows: list[GenRow]) -> dict:
        """Score summary of one fixture: its rows re-keyed to the clean keys."""
        plain = _restore_keys(rows) if "|" in rows[0].key else rows
        sub = {"tasks": {}}
        for task, d in scored["tasks"].items():
            sub["tasks"][task] = {"rows": {p.key: d["rows"][r.key] for r, p in zip(rows, plain) if r.key in d["rows"]}}
        return summarize(build_quality(plain, sub, None))

    base = summary_for(clean)
    rsha = rows_sha256(clean)
    qref = freeze_qref({"clean-baseline": {"qualified": True, "rows_sha256": rsha, "seed_indices": seeds, "per_seed": base["per_seed"]}}, "L20X", time.time())
    report = {"generation": col.generation, "seeds": col.seeds, "evaluator": scored["evaluator"], "rows_sha256": rsha,
              "q_ref_scope": "calibration only", "q_ref": qref.to_json(), "injections": {}}
    ok_all = True
    for kind, (metric, ladder) in LADDERS.items():
        direction, margin, _ = COMPARATIVE[metric]
        noise = float(np.ptp([s[metric] for s in base["per_seed"]]))
        need = margin + 2 * noise
        tried = []
        selected = None
        for mag in ladder:
            summ = summary_for(fixtures[(kind, mag)])
            eff_vals = [s[metric] for s in summ["per_seed"]]
            effect = (np.mean(eff_vals) - qref.values[metric]) if direction == "max" else (qref.values[metric] - np.mean(eff_vals))
            gate = comparative_gates(summ["per_seed"], rsha, seeds, qref)
            tried.append({"magnitude": mag, "effect": float(effect), "required_effect": need, "gate_verdict": gate["metrics"][metric]["verdict"],
                          "overall": gate["overall"]})
            if selected is None and effect >= need:
                selected = tried[-1]
        rejected = selected is not None and selected["gate_verdict"] == "fail"
        ok_all &= rejected
        report["injections"][kind] = {"metric": metric, "margin": margin, "seed_noise": noise, "required_effect": need, "ladder": tried,
                                      "selected": selected, "rejected_by_gate": rejected}
    report["all_rejected"] = ok_all
    report["wall_s"] = time.time() - t0
    (out / "calibration.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {"selected": v["selected"], "rejected": v["rejected_by_gate"], "noise": v["seed_noise"]} for k, v in report["injections"].items()}, indent=1))
    return 0 if ok_all else 1


if __name__ == "__main__":
    raise SystemExit(main())
