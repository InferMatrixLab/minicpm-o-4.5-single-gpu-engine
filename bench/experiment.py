"""Experiment identity: candidates, cell specifications and their digests.

A *candidate* is one launchable configuration of one engine: adapter, deploy
config, server arguments and environment overlay.  Its ``candidate_id`` carries a
digest of all of these plus the deploy files' contents, so two launch variants
of the same config (for example MPS off/on) can never be confused.

A *cell spec* is a candidate measured at one concurrency with one workload
(ID set, seed index, K, timeout).  Its digest is stored with every result and
checked before a result is reused or compared.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class Candidate:
    adapter: str
    config: str
    extra: tuple = ()
    env: tuple = ()  # sorted (key, value) pairs added to the server environment
    label: str = ""  # human-readable variant name, e.g. "mps-on"
    deploy_files_sha256: tuple = ()  # sha256 of every deploy file the launch reads
    mps: str = "off"  # declared MPS policy, verified at runtime: "off" | "on"

    @classmethod
    def from_spec(cls, d: dict, deploy_files_sha256: tuple = ()) -> "Candidate":
        return cls(
            adapter=d["adapter"],
            config=d["config"],
            extra=tuple(str(x) for x in d.get("extra", [])),
            env=tuple(sorted((str(k), str(v)) for k, v in d.get("env", {}).items())),
            label=str(d.get("label", "")),
            deploy_files_sha256=tuple(deploy_files_sha256),
            mps=str(d.get("mps", "off")),
        )

    def identity(self) -> dict:
        if self.mps not in ("off", "on"):
            raise ValueError(f"mps policy must be 'off' or 'on', got {self.mps!r}")
        return {"adapter": self.adapter, "config": self.config, "extra": list(self.extra), "env": [list(p) for p in self.env],
                "deploy_files_sha256": list(self.deploy_files_sha256), "mps": self.mps}

    @property
    def digest(self) -> str:
        return digest(self.identity())

    @property
    def candidate_id(self) -> str:
        stem = self.config.rsplit("/", 1)[-1].removesuffix(".yaml")
        tag = f"-{self.label}" if self.label else ""
        return f"{self.adapter}:{stem}{tag}:{self.digest[:12]}"

    def env_dict(self) -> dict:
        return dict(self.env)


@dataclass(frozen=True)
class CellSpec:
    candidate: Candidate
    concurrency: int
    k: int
    id_set: str
    seed_index: int
    timeout_s: float
    gpu_index: int
    workload: dict = field(default_factory=dict)  # dataset repo/revision and ID-list hashes

    def identity(self) -> dict:
        return {
            "candidate": self.candidate.identity(),
            "concurrency": self.concurrency,
            "k": self.k,
            "id_set": self.id_set,
            "seed_index": self.seed_index,
            "timeout_s": self.timeout_s,
            "gpu_index": self.gpu_index,
            "workload": self.workload,
        }

    @property
    def digest(self) -> str:
        """Excludes K: a declared K extension stays the same cell."""
        return digest({k: v for k, v in self.identity().items() if k != "k"})

    def to_json(self) -> dict:
        return {**self.identity(), "candidate_id": self.candidate.candidate_id, "digest": self.digest}


def unique_candidates(cands: list[Candidate]) -> list[Candidate]:
    ids: set[str] = set()
    digests: set[str] = set()
    for c in cands:
        if c.candidate_id in ids or c.digest in digests:
            raise ValueError(f"duplicate candidate {c.candidate_id}")
        ids.add(c.candidate_id)
        digests.add(c.digest)
    return cands


def asdict_candidate(c: Candidate) -> dict:
    d = asdict(c)
    d["candidate_id"] = c.candidate_id
    d["digest"] = c.digest
    return d


#: Spec fields that a deliberate K extension may change; the base digest excludes them.
SCHEDULE_FIELDS = ("k", "k_updates")


def spec_digest(spec: dict) -> str:
    return digest(spec)


def spec_base_digest(spec: dict) -> str:
    return digest({k: v for k, v in spec.items() if k not in SCHEDULE_FIELDS})


def build_spec(
    *,
    kind: str,
    cell: CellSpec,
    warmup_ids: list[str],
    measured_ids: list[str],
    model: dict,
    dataset: dict,
    sampling: dict,
    adapter_options: dict,
    fingerprints: dict,
    hardware: dict,
    cache_policy: str,
    strict: bool,
    save_audio: bool,
    record_raw: bool,
) -> dict:
    """The frozen experiment specification stored with, and checked against, every result."""
    if kind not in ("performance", "quality"):
        raise ValueError(f"unknown experiment kind {kind}")
    return {
        "kind": kind,
        "candidate": {**cell.candidate.identity(), "candidate_id": cell.candidate.candidate_id, "digest": cell.candidate.digest,
                      "label": cell.candidate.label},
        "concurrency": cell.concurrency,
        "k": cell.k,
        "k_updates": [],
        "id_set": cell.id_set,
        "warmup_ids": list(warmup_ids),
        "measured_ids": list(measured_ids),
        "seed_index": cell.seed_index,
        "timeout_s": cell.timeout_s,
        "model": model,
        "dataset": dataset,
        "sampling": sampling,
        "adapter_options": adapter_options,
        "fingerprints": fingerprints,
        "hardware": hardware,
        "cache_policy": cache_policy,
        "strict": strict,
        "save_audio": save_audio,
        "record_raw": record_raw,
    }
