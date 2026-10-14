"""Compare effective-input dumps against the canonical dump and write an eligibility report.

    python -m bench.effective_report results/effective_inputs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from bench.effective_inputs import compare_sets, load_dumps, rel_l2, summarize


def stack_report(cand: dict, canon: dict, strict: bool) -> dict:
    mm = compare_sets(cand, canon, strict_reference=strict)
    rep = summarize(mm, len(canon))
    keys = [k for k in canon if k in cand]
    spk = [rel_l2(cand[k].speaker_embedding, canon[k].speaker_embedding) for k in keys if cand[k].speaker_embedding and canon[k].speaker_embedding]
    agree = []
    for k in keys:
        a, b = cand[k].speech_tokens or [], canon[k].speech_tokens or []
        n = min(len(a), len(b))
        if n:
            agree.append(float(np.mean(np.asarray(a[:n]) == np.asarray(b[:n]))))
    rep["reference_conditioning"] = {
        "speech_token_agreement_mean": float(np.mean(agree)) if agree else None,
        "speech_token_length_equal_rows": sum(len(cand[k].speech_tokens or []) == len(canon[k].speech_tokens or []) for k in keys),
        "speaker_embedding_rel_l2_median": float(np.median(spk)) if spk else None,
        "speaker_embedding_rel_l2_max": float(np.max(spk)) if spk else None,
    }
    rep["strict_reference"] = strict
    return rep


#: Stacks held to the HF reference path exactly (the new engine); baselines are not.
STRICT_STACKS = {"engine"}
BASELINE_STACKS = {"vllm-omni", "sglang-omni"}


def main(argv: list[str]) -> int:
    d = Path(argv[0] if argv else "results/effective_inputs")
    canon = load_dumps(d / "canonical.jsonl")
    report = {}
    for path in sorted(d.glob("*.jsonl")):
        name = path.stem
        if name == "canonical":
            continue
        if name not in STRICT_STACKS | BASELINE_STACKS:
            raise SystemExit(f"unknown stack dump {path.name}; expected one of {sorted(STRICT_STACKS | BASELINE_STACKS)}")
        report[name] = stack_report(load_dumps(path), canon, strict=name in STRICT_STACKS)
    (d / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: {"eligible": v["eligible"], **v["reference_conditioning"]} for k, v in report.items()}, indent=1))
    return 0 if all(v["eligible"] for v in report.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
