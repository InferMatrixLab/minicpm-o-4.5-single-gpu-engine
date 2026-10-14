"""Configuration classification and the class-R baseline envelope.

Class R (the reference class) is: Track T conditioning, 10 Euler steps, the
repo's shipped default precision, the official 25-token chunk geometry or a
full waveform, turn session mode, and canonical effective inputs.  The baseline
envelope is formed over class-R configurations only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

#: Shipped default flow precision per repo (class-R precision rule).
SHIPPED_FLOW_PRECISION = {"vllm-omni": "tf32", "sglang-omni": "fp16"}
OFFICIAL_CHUNK_TOKENS = 25
REQUIRED_EULER_STEPS = 10
DEV_GPUS = {"L20X"}
TARGET_GPUS = {"H200", "B200"}


@dataclass(frozen=True)
class ConfigClass:
    engine: str
    config: str
    track: str  # "T" (Talker after the Thinker reply ends) | "S" (incremental)
    audio_output: str  # "stream" | "full"
    chunk_tokens: int | None  # None for a full waveform
    flow_precision: str
    thinker_precision: str
    euler_steps: int
    session_mode: str  # "turn" | "duplex"
    canonical_inputs: bool
    mps: bool = False

    def class_r_violations(self) -> list[str]:
        v = []
        if self.track != "T":
            v.append(f"track {self.track} (class R requires T)")
        if self.euler_steps != REQUIRED_EULER_STEPS:
            v.append(f"{self.euler_steps} Euler steps (class R requires {REQUIRED_EULER_STEPS})")
        shipped = SHIPPED_FLOW_PRECISION.get(self.engine)
        if shipped is not None and self.flow_precision != shipped:
            v.append(f"flow precision {self.flow_precision} (shipped default is {shipped})")
        if self.thinker_precision != "bf16":
            v.append(f"thinker precision {self.thinker_precision} (class R requires bf16)")
        if self.audio_output == "stream" and self.chunk_tokens != OFFICIAL_CHUNK_TOKENS:
            v.append(f"chunk geometry {self.chunk_tokens} (class R requires {OFFICIAL_CHUNK_TOKENS} or full waveform)")
        if self.audio_output not in ("stream", "full"):
            v.append(f"unknown audio output {self.audio_output}")
        if self.session_mode != "turn":
            v.append(f"session mode {self.session_mode} (class R requires turn)")
        if not self.canonical_inputs:
            v.append("non-canonical effective inputs")
        return v

    @property
    def is_class_r(self) -> bool:
        return not self.class_r_violations()

    def to_json(self) -> dict:
        d = asdict(self)
        d["class_r"] = self.is_class_r
        d["class_r_violations"] = self.class_r_violations()
        return d


@dataclass(frozen=True)
class EnvelopeEntry:
    gpu: str
    concurrency: int
    metric: str
    config: ConfigClass
    value: float
    qualified: bool
    provisional: bool = False


class EnvelopeError(ValueError):
    pass


def validate_envelope(entries: list[EnvelopeEntry], gpu: str, require_final: bool = False) -> None:
    """Reject non-class-R, unqualified or wrong-hardware entries.

    ``require_final`` rejects provisional (development-GPU) evidence offered as
    target-GPU evidence.
    """
    for e in entries:
        if not e.config.is_class_r:
            raise EnvelopeError(f"{e.config.engine}/{e.config.config} is not class R: {e.config.class_r_violations()}")
        if not e.qualified:
            raise EnvelopeError(f"{e.config.engine}/{e.config.config} did not qualify")
        if e.gpu != gpu:
            raise EnvelopeError(f"entry measured on {e.gpu} offered as {gpu} evidence")
        if require_final and (e.provisional or e.gpu in DEV_GPUS):
            raise EnvelopeError(f"provisional {e.gpu} result offered as final evidence")
    if require_final and gpu not in TARGET_GPUS:
        raise EnvelopeError(f"{gpu} is not a target GPU")


def envelope_winner(entries: list[EnvelopeEntry], higher_is_better: bool) -> EnvelopeEntry:
    if not entries:
        raise EnvelopeError("no qualified class-R configuration for this cell")
    return max(entries, key=lambda e: e.value) if higher_is_better else min(entries, key=lambda e: e.value)
