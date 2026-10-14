"""Seed-TTS-eval EN voice-clone workload, canonical requests and ID sets."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

SEED_TTS_REPO = "zhaochenyang20/seed-tts-eval"
SEED_TTS_REVISION = "8f5e1aa2a35d42f42e940074c1983358b9491f89"
HF_HUB_CACHE = os.environ.get("HF_HUB_CACHE_BENCH", "/workspace/models/hub")
MODEL_REPO = "openbmb/MiniCPM-o-4_5"
MODEL_REVISION = "503e754207c94da6bb26850b4469f367c9ea3582"

# The Thinker reads the target text back verbatim.  This is the form the
# sglang-omni MiniCPM-o runbook sends; every adapter sends the same messages so
# that prompt token IDs agree across engines.
CANONICAL_USER_PROMPT_EN = "Please read the following text out loud in English: {text}"

CONCURRENCIES = (1, 2, 4, 8, 16, 32)
WARMUP_POOL = 64  # 2 * max(C)
EVAL_POOL = 640  # max(320, 20 * max(C))
ID_SPLIT_SEED = "minicpmo45-seedtts-en-v1"


@dataclass(frozen=True)
class ThinkerSampling:
    temperature: float = 0.0  # greedy
    max_tokens: int = 256


@dataclass(frozen=True)
class TalkerSampling:
    temperature: float = 0.8
    top_p: float = 0.85
    top_k: int = 25
    repetition_penalty: float = 1.05
    repetition_window: int = 16
    min_new_tokens: int = 50
    max_new_tokens: int = 2048


@dataclass(frozen=True)
class SeedTTSRow:
    row_id: str
    ref_text: str
    ref_wav_path: str
    target_text: str


@dataclass
class CanonicalRequest:
    row: SeedTTSRow
    seed: int
    thinker: ThinkerSampling = field(default_factory=ThinkerSampling)
    talker: TalkerSampling = field(default_factory=TalkerSampling)
    enable_thinking: bool = False
    use_tts_template: bool = True
    stream: bool = True

    @property
    def request_id(self) -> str:
        return self.row.row_id

    @property
    def user_prompt(self) -> str:
        return CANONICAL_USER_PROMPT_EN.format(text=self.row.target_text)

    def messages(self) -> list[dict]:
        return [{"role": "user", "content": self.user_prompt}]

    def ref_audio_bytes(self) -> bytes:
        return Path(self.row.ref_wav_path).read_bytes()

    def ref_audio_data_url(self) -> str:
        return "data:audio/wav;base64," + base64.b64encode(self.ref_audio_bytes()).decode("ascii")

    def describe(self) -> dict:
        return {
            "row": asdict(self.row),
            "seed": self.seed,
            "thinker": asdict(self.thinker),
            "talker": asdict(self.talker),
            "enable_thinking": self.enable_thinking,
            "use_tts_template": self.use_tts_template,
            "stream": self.stream,
            "messages": self.messages(),
        }


def seed_tts_root() -> Path:
    env = os.environ.get("SEED_TTS_ROOT")
    if env:
        return Path(env)
    return Path(HF_HUB_CACHE) / "datasets--zhaochenyang20--seed-tts-eval" / "snapshots" / SEED_TTS_REVISION


def load_seed_tts_rows(root: Path | None = None, locale: str = "en") -> list[SeedTTSRow]:
    base = (root or seed_tts_root()) / locale
    rows: list[SeedTTSRow] = []
    for line in (base / "meta.lst").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|")
        if len(parts) < 4 or not parts[3].strip():
            continue
        rows.append(
            SeedTTSRow(
                row_id=parts[0].strip(),
                ref_text=parts[1].strip(),
                ref_wav_path=str(base / parts[2].strip()),
                target_text=parts[3].strip(),
            )
        )
    return rows


def _rank(row_id: str) -> str:
    return hashlib.sha256(f"{ID_SPLIT_SEED}:{row_id}".encode()).hexdigest()


@dataclass(frozen=True)
class IdSets:
    warmup_pool: tuple[str, ...]
    tuning_pool: tuple[str, ...]
    eval_pool: tuple[str, ...]

    def warmup(self, c: int) -> list[str]:
        n = 2 * c
        if n > len(self.warmup_pool):
            raise ValueError(f"warmup pool too small for C={c}")
        return list(self.warmup_pool[:n])

    def eval(self, c: int) -> list[str]:
        n = n_eval(c)
        if n > len(self.eval_pool):
            raise ValueError(f"eval pool too small for C={c}")
        return list(self.eval_pool[:n])

    def full(self) -> list[str]:
        """Every row, for full-set quality generation (AC-4.4); never a performance ID set."""
        return list(self.warmup_pool + self.eval_pool + self.tuning_pool)

    def measured(self, id_set: str, c: int) -> list[str]:
        if id_set == "eval":
            return self.eval(c)
        if id_set == "tuning":
            return self.tuning(c)
        if id_set == "full":
            return self.full()
        raise ValueError(f"unknown id set {id_set}")

    def tuning(self, c: int) -> list[str]:
        n = min(len(self.tuning_pool), n_eval(c))
        return list(self.tuning_pool[:n])

    def to_json(self) -> dict:
        return {
            "split_seed": ID_SPLIT_SEED,
            "warmup_pool": list(self.warmup_pool),
            "tuning_pool": list(self.tuning_pool),
            "eval_pool": list(self.eval_pool),
        }


def n_eval(c: int) -> int:
    return max(320, 20 * c)


def make_id_sets(rows: list[SeedTTSRow]) -> IdSets:
    ids = sorted((r.row_id for r in rows), key=_rank)
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate row ids")
    if len(ids) < WARMUP_POOL + EVAL_POOL + 1:
        raise ValueError("dataset too small for the ID split")
    warm = tuple(ids[:WARMUP_POOL])
    evl = tuple(ids[WARMUP_POOL : WARMUP_POOL + EVAL_POOL])
    tune = tuple(ids[WARMUP_POOL + EVAL_POOL :])
    return IdSets(warmup_pool=warm, tuning_pool=tune, eval_pool=evl)


def request_seed(row_id: str, seed_index: int = 0) -> int:
    """Fixed per-request Talker seed; ``seed_index`` selects one of the quality seeds."""
    base = int(hashlib.sha256(row_id.encode()).hexdigest()[:8], 16) % (2**31 - 1000)
    return base + seed_index


def canonical_requests(rows: list[SeedTTSRow], ids: list[str], seed_index: int = 0, stream: bool = True) -> list[CanonicalRequest]:
    by_id = {r.row_id: r for r in rows}
    return [CanonicalRequest(row=by_id[i], seed=request_seed(i, seed_index), stream=stream) for i in ids]


def dump_id_sets(path: Path, sets: IdSets) -> None:
    path.write_text(json.dumps(sets.to_json(), indent=1))
