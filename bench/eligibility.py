"""Configuration classification and the class-R baseline envelope.

Class R (the reference class) is: Track T conditioning, 10 Euler steps, the
repo's shipped default precision for every component, the official 25-token
chunk geometry or a full waveform, turn session mode, and canonical effective
inputs (identical prompt token IDs and reference file bytes).  The baseline
envelope is formed over class-R configurations only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

COMPONENTS = ("thinker", "talker", "flow", "hift")
#: Shipped default precision per repo and component (class-R precision rule),
#: read from the pinned commits: vllm-omni Code2Wav enables TF32 for the CFM DiT
#: and keeps HiFT in FP32; sglang-omni's code2wav stage casts the flow to FP16
#: and keeps HiFT in FP32.
SHIPPED_PRECISION = {
    "vllm-omni": {"thinker": "bf16", "talker": "bf16", "flow": "tf32", "hift": "fp32"},
    "sglang-omni": {"thinker": "bf16", "talker": "bf16", "flow": "fp16", "hift": "fp32"},
}
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
    precision: dict  # component -> precision
    euler_steps: int
    session_mode: str  # "turn" | "duplex"
    canonical_inputs: bool  # prompt token IDs and reference bytes match the canonical dump
    mps: bool = False
    #: Documented semantic differences (e.g. reference preprocessing); never disqualifying.
    documented_differences: tuple = field(default_factory=tuple)

    def class_r_violations(self) -> list[str]:
        v = []
        shipped = SHIPPED_PRECISION.get(self.engine)
        if shipped is None:
            v.append(f"engine {self.engine!r} is not a baseline with a known shipped configuration")
        else:
            for comp in COMPONENTS:
                got = self.precision.get(comp)
                if got != shipped[comp]:
                    v.append(f"{comp} precision {got} (shipped default is {shipped[comp]})")
        if self.track != "T":
            v.append(f"track {self.track} (class R requires T)")
        if self.euler_steps != REQUIRED_EULER_STEPS:
            v.append(f"{self.euler_steps} Euler steps (class R requires {REQUIRED_EULER_STEPS})")
        if self.audio_output == "stream":
            if self.chunk_tokens != OFFICIAL_CHUNK_TOKENS:
                v.append(f"chunk geometry {self.chunk_tokens} (class R requires {OFFICIAL_CHUNK_TOKENS} or full waveform)")
        elif self.audio_output != "full":
            v.append(f"unknown audio output {self.audio_output}")
        if self.session_mode != "turn":
            v.append(f"session mode {self.session_mode} (class R requires turn)")
        if not self.canonical_inputs:
            v.append("non-canonical effective inputs (prompt token IDs or reference bytes)")
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
    selected_on: str = "tuning"  # winners are chosen on the tuning ID set


class EnvelopeError(ValueError):
    pass


def validate_envelope(entries: list[EnvelopeEntry], gpu: str, require_final: bool = False) -> None:
    """Reject non-class-R, unqualified or wrong-hardware entries.

    ``require_final`` rejects provisional (development-GPU) evidence offered as
    target-GPU evidence.
    """
    if require_final and gpu not in TARGET_GPUS:
        raise EnvelopeError(f"{gpu} is not a target GPU")
    for e in entries:
        if not e.config.is_class_r:
            raise EnvelopeError(f"{e.config.engine}/{e.config.config} is not class R: {e.config.class_r_violations()}")
        if not e.qualified:
            raise EnvelopeError(f"{e.config.engine}/{e.config.config} did not qualify")
        if e.gpu != gpu:
            raise EnvelopeError(f"entry measured on {e.gpu} offered as {gpu} evidence")
        if require_final and (e.provisional or e.gpu in DEV_GPUS):
            raise EnvelopeError(f"provisional {e.gpu} result offered as final evidence")


def envelope_winner(entries: list[EnvelopeEntry], higher_is_better: bool, gpu: str, require_final: bool = False) -> EnvelopeEntry:
    """Winner of one (GPU, C, metric) cell, selected on the tuning set from validated entries."""
    if not entries:
        raise EnvelopeError("no qualified class-R configuration for this cell")
    validate_envelope(entries, gpu, require_final=require_final)
    cells = {(e.gpu, e.concurrency, e.metric) for e in entries}
    if len(cells) != 1:
        raise EnvelopeError(f"entries span several cells: {sorted(cells)}")
    if any(e.selected_on != "tuning" for e in entries):
        raise EnvelopeError("winners must be selected on the tuning ID set")
    return max(entries, key=lambda e: e.value) if higher_is_better else min(entries, key=lambda e: e.value)
