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
            if not np.isfinite(np.asarray(candidate.speaker_embedding, dtype=np.float64)).all():
                out.append(Mismatch(rid, "speaker_embedding", "non-finite values", True))
            elif not (e <= tol):
                out.append(Mismatch(rid, "speaker_embedding", f"rel L2 {e:.2e} > {tol:.0e}", soft))
    return out


MIN_ROWS = 32
_HEX = set("0123456789abcdef")


def _is_sha(h) -> bool:
    return isinstance(h, str) and len(h) == 64 and set(h) <= _HEX


def evidence_problems(e: EffectiveInput) -> list[str]:
    """Structural completeness of one dump: every field present, well-formed and finite."""
    p = []
    ids = e.prompt_token_ids
    if not isinstance(ids, list) or not ids or not all(isinstance(i, int) and i >= 0 for i in ids):
        p.append("prompt_token_ids missing or malformed")
    for f in ("ref_source_sha256", "ref_waveform_hash"):
        if not _is_sha(getattr(e, f)):
            p.append(f"{f} missing or not a sha256")
    if not isinstance(e.ref_waveform_sample_rate, int) or e.ref_waveform_sample_rate <= 0:
        p.append("ref_waveform_sample_rate missing or invalid")
    toks = e.speech_tokens
    if not isinstance(toks, list) or not toks or not all(isinstance(t, int) and t >= 0 for t in toks):
        p.append("speech_tokens missing or malformed")
    emb = e.speaker_embedding
    if not isinstance(emb, list) or not emb:
        p.append("speaker_embedding missing")
    elif not np.isfinite(np.asarray(emb, dtype=np.float64)).all():
        p.append("speaker_embedding non-finite")
    return p


def compare_sets(
    candidates: dict[str, EffectiveInput], canonical: dict[str, EffectiveInput], strict_reference: bool = False, min_rows: int = MIN_ROWS
) -> list[Mismatch]:
    """Compare a stack's dumps with the canonical dumps.

    Both sets are validated structurally first: identical row sets of at least
    ``min_rows`` rows, and complete, finite evidence in every dump.  Missing
    evidence is always a hard failure; only measured, finite reference-conditioning
    differences of a baseline are soft (documented).
    """
    out: list[Mismatch] = []
    if len(canonical) < min_rows:
        out.append(Mismatch("*", "evidence", f"canonical set has {len(canonical)} rows (need {min_rows})", True))
    if set(candidates) != set(canonical):
        extra = sorted(set(candidates) - set(canonical))
        missing = sorted(set(canonical) - set(candidates))
        out.append(Mismatch("*", "row_set", f"row sets differ: missing {missing[:3]}, extra {extra[:3]}", True))
    for side, dumps in (("canonical", canonical), ("candidate", candidates)):
        for rid, e in dumps.items():
            for prob in evidence_problems(e):
                out.append(Mismatch(rid, "evidence", f"{side}: {prob}", True))
    if any(m.hard for m in out):
        return out
    for rid, canon in canonical.items():
        out.extend(compare(candidates[rid], canon, strict_reference=strict_reference))
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
            if e.row_id in out:
                raise ValueError(f"duplicate dump for row {e.row_id} in {path}")
            out[e.row_id] = e
    return out


def save_dumps(path: Path, dumps: list[EffectiveInput]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for d in dumps:
            f.write(json.dumps(d.to_json()) + "\n")


# --- provenance: which execution produced a dump -------------------------------------------------
PROVENANCE_SCHEMA = "effective_dump/v1"
STACK_REPOS = {"vllm-omni": "/workspace/cgzheng/cgzheng/vllm-omni", "sglang-omni": "/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main"}
_ADAPTER_MODULE = {"vllm-omni": "vllm_omni", "sglang-omni": "sglang_omni", "engine": "engine"}


def provenance_path(dump: Path) -> Path:
    return Path(dump).with_suffix(".provenance.json")


PRODUCER_FILES = ["bench/effective_dump.py", "bench/effective_inputs.py"]


def _code_files(stack: str) -> list[str]:
    """Code that determines what a request sends (compared against today's files when a dump is used)."""
    files = ["bench/workload.py", "bench/adapters/base.py"]
    if stack in _ADAPTER_MODULE:
        files.append(f"bench/adapters/{_ADAPTER_MODULE[stack]}.py")
    if stack == "engine":
        files += ["engine/models/token2wav.py", "engine/runtime/backend.py"]
    return files


def code_hashes(stack: str, files: list[str] | None = None) -> dict:
    root = Path(__file__).resolve().parents[1]
    return {f: hashlib.sha256((root / f).read_bytes()).hexdigest() for f in (files if files is not None else _code_files(stack))}


def write_provenance(dump: Path, stack: str, model: dict, server: dict | None = None, preprocessing: dict | None = None) -> dict:
    """Record the execution that produced ``dump``: stack commit, model revision, request code, preprocessing settings."""
    from bench.fingerprint import repo_fingerprint

    doc = {"schema": PROVENANCE_SCHEMA, "stack": stack, "dump_sha256": hashlib.sha256(Path(dump).read_bytes()).hexdigest(),
           "rows": len(load_dumps(Path(dump))), "model": model,
           "stack_repo": repo_fingerprint(STACK_REPOS[stack]) if stack in STACK_REPOS else None, "code_sha256": code_hashes(stack),
           "producer_sha256": code_hashes(stack, PRODUCER_FILES),
           "server": server, "preprocessing": preprocessing}
    provenance_path(dump).write_text(json.dumps(doc, indent=1))
    return doc


def verify_provenance(dump: Path, stack: str, model: dict, stack_repo: dict | None, preprocessing: dict | None = None) -> list[str]:
    """Problems binding ``dump`` to the measured execution (model revision, stack commit, request code, preprocessing settings)."""
    p = provenance_path(dump)
    if not p.exists():
        return [f"{dump}: no provenance record"]
    doc = json.loads(p.read_text())
    out = []
    if doc.get("schema") != PROVENANCE_SCHEMA or doc.get("stack") != stack:
        out.append(f"{dump}: provenance is for {doc.get('stack')!r} ({doc.get('schema')}), not {stack!r}")
    if doc.get("dump_sha256") != hashlib.sha256(Path(dump).read_bytes()).hexdigest():
        out.append(f"{dump}: dump bytes differ from the recorded execution")
    if doc.get("model") != model:
        out.append(f"{dump}: produced with model {doc.get('model')}, measured with {model}")
    if stack in STACK_REPOS:
        rec = doc.get("stack_repo") or {}
        if stack_repo is None or rec.get("commit") != stack_repo.get("commit") or rec.get("dirty") or stack_repo.get("dirty"):
            out.append(f"{dump}: produced at {stack} {rec.get('commit')} (dirty={rec.get('dirty')}), measured at {stack_repo}")
    if preprocessing is not None and doc.get("preprocessing") != preprocessing:
        out.append(f"{dump}: produced under preprocessing {doc.get('preprocessing')}, the measured launch has {preprocessing}")
    rec = doc.get("code_sha256") or {}
    changed = [f for f, h in code_hashes(stack).items() if rec.get(f) != h]
    if changed:
        out.append(f"{dump}: request code changed since the dump was produced: {changed}")
    return out
