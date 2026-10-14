"""Effective-input equality: the eligibility rule for class R.

Each stack's preprocessing is dumped for the same rows and compared with the
canonical inputs produced by the HF reference path.

Two tiers:

* Hard rules for every configuration: identical prompt token IDs (including
  special tokens) and identical reference-audio file bytes in the request.
  A violation removes the configuration from class R.
* Reference conditioning (resampled waveform hash, reference speech tokens,
  speaker embedding).  A baseline's shipped reference preprocessing is
  reported as a documented semantic difference and does not remove class R.
  The new engine is held to the HF path exactly: equal waveform hash and
  speech tokens, and speaker embedding within relative L2 1e-3.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REL_L2_TOL = 1e-3


def array_hash(x) -> str:
    a = np.ascontiguousarray(np.asarray(x))
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def rel_l2(a, b) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return float("inf")
    denom = np.linalg.norm(b)
    if denom == 0:
        return 0.0 if np.linalg.norm(a) == 0 else float("inf")
    return float(np.linalg.norm(a - b) / denom)


@dataclass
class EffectiveInput:
    row_id: str
    prompt_token_ids: list[int]
    ref_source_sha256: str | None = None  # bytes of the reference file as sent in the request
    ref_waveform_hash: str | None = None  # waveform the reference conditioning was computed from
    ref_waveform_sample_rate: int | None = None
    speech_tokens: list[int] | None = None
    speaker_embedding: list[float] | None = None
    extra: dict = field(default_factory=dict)

    def to_json(self) -> dict:
        return self.__dict__.copy()

    @classmethod
    def from_json(cls, d: dict) -> "EffectiveInput":
        return cls(**d)


@dataclass
class Mismatch:
    row_id: str
    field: str
    detail: str
    hard: bool  # True: removes class R; False: documented semantic difference


def compare(candidate: EffectiveInput, canonical: EffectiveInput, strict_reference: bool = False, tol: float = REL_L2_TOL) -> list[Mismatch]:
    """Compare one row.  ``strict_reference`` makes reference-conditioning differences hard (new engine)."""
    out: list[Mismatch] = []
    rid = canonical.row_id
    soft = strict_reference
    if candidate.row_id != canonical.row_id:
        out.append(Mismatch(rid, "row_id", f"{candidate.row_id} != {canonical.row_id}", True))
    if list(candidate.prompt_token_ids) != list(canonical.prompt_token_ids):
        a, b = candidate.prompt_token_ids, canonical.prompt_token_ids
        first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        out.append(Mismatch(rid, "prompt_token_ids", f"len {len(a)} vs {len(b)}, first diff at {first}", True))
    if canonical.ref_source_sha256 is not None and candidate.ref_source_sha256 != canonical.ref_source_sha256:
        out.append(Mismatch(rid, "ref_source_sha256", "reference file bytes differ", True))
    if canonical.ref_waveform_hash is not None and candidate.ref_waveform_hash != canonical.ref_waveform_hash:
        out.append(Mismatch(rid, "ref_waveform_hash", "conditioning waveform differs", soft))
    if canonical.ref_waveform_sample_rate is not None and candidate.ref_waveform_sample_rate != canonical.ref_waveform_sample_rate:
        out.append(
            Mismatch(rid, "ref_waveform_sample_rate", f"{candidate.ref_waveform_sample_rate} != {canonical.ref_waveform_sample_rate}", soft)
        )
    if canonical.speech_tokens is not None:
        a = list(candidate.speech_tokens or [])
        b = list(canonical.speech_tokens)
        if a != b:
            n = min(len(a), len(b))
            agree = float(np.mean(np.asarray(a[:n]) == np.asarray(b[:n]))) if n else 0.0
            out.append(Mismatch(rid, "speech_tokens", f"len {len(a)} vs {len(b)}, agreement {agree:.3f}", soft))
    if canonical.speaker_embedding is not None:
        if candidate.speaker_embedding is None:
            out.append(Mismatch(rid, "speaker_embedding", "missing", soft))
        else:
            e = rel_l2(candidate.speaker_embedding, canonical.speaker_embedding)
            if e > tol:
                out.append(Mismatch(rid, "speaker_embedding", f"rel L2 {e:.2e} > {tol:.0e}", soft))
    return out


def compare_sets(candidates: dict[str, EffectiveInput], canonical: dict[str, EffectiveInput], strict_reference: bool = False) -> list[Mismatch]:
    out: list[Mismatch] = []
    for rid, canon in canonical.items():
        cand = candidates.get(rid)
        if cand is None:
            out.append(Mismatch(rid, "missing", "no dump for row", True))
            continue
        out.extend(compare(cand, canon, strict_reference=strict_reference))
    return out


def eligible(mismatches: list[Mismatch]) -> bool:
    return not any(m.hard for m in mismatches)


def summarize(mismatches: list[Mismatch], n_rows: int) -> dict:
    by_field: dict[str, dict] = {}
    for m in mismatches:
        f = by_field.setdefault(m.field, {"rows": 0, "hard": m.hard, "examples": []})
        f["rows"] += 1
        if len(f["examples"]) < 3:
            f["examples"].append(f"{m.row_id}: {m.detail}")
    return {"rows": n_rows, "eligible": eligible(mismatches), "fields": by_field}


def load_dumps(path: Path) -> dict[str, EffectiveInput]:
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            e = EffectiveInput.from_json(json.loads(line))
            out[e.row_id] = e
    return out


def save_dumps(path: Path, dumps: list[EffectiveInput]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in dumps:
            f.write(json.dumps(d.to_json()) + "\n")
