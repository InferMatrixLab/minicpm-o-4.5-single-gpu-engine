"""Per-row speech defect detectors and corpus metrics.

Every rate uses the number of evaluated rows as its denominator.

* catastrophic: per-clip WER > 50%
* silence: leading or trailing silence > 1.0 s, or any internal silence run > 2.0 s
  (20 ms frames below -50 dBFS)
* repetition: a 3-word n-gram repeated 3 or more times in a row in the ASR
  transcript, where the target text does not itself contain that repetition
* truncation: ASR word recall < 0.8 together with D_i < 0.7 * D_ref,i
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from bench.audio import SAMPLE_RATE, SILENCE_FRAME_S, silent_frame_mask

CATASTROPHIC_WER = 0.5
EDGE_SILENCE_S = 1.0
INTERNAL_SILENCE_S = 2.0
REPEAT_N = 3
REPEAT_TIMES = 3
TRUNC_RECALL = 0.8
TRUNC_DURATION_FRAC = 0.7


@dataclass
class SilenceProfile:
    leading_s: float
    trailing_s: float
    max_internal_s: float
    silent_fraction: float

    @property
    def flagged(self) -> bool:
        return self.leading_s > EDGE_SILENCE_S or self.trailing_s > EDGE_SILENCE_S or self.max_internal_s > INTERNAL_SILENCE_S


def silence_profile(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> SilenceProfile:
    mask = silent_frame_mask(samples, sample_rate)
    if mask.size == 0:
        return SilenceProfile(0.0, 0.0, 0.0, 1.0)
    if mask.all():
        total = mask.size * SILENCE_FRAME_S
        return SilenceProfile(total, total, 0.0, 1.0)
    voiced = np.flatnonzero(~mask)
    first, last = int(voiced[0]), int(voiced[-1])
    leading = first * SILENCE_FRAME_S
    trailing = (mask.size - 1 - last) * SILENCE_FRAME_S
    inner = mask[first : last + 1]
    longest = run = 0
    for v in inner:
        run = run + 1 if v else 0
        longest = max(longest, run)
    return SilenceProfile(leading, trailing, longest * SILENCE_FRAME_S, float(mask.mean()))


def _consecutive_ngram_repeat(words: list[str], n: int = REPEAT_N, times: int = REPEAT_TIMES) -> set[tuple[str, ...]]:
    """n-grams that occur ``times`` or more times back to back."""
    found = set()
    for i in range(0, max(0, len(words) - n * times + 1)):
        gram = tuple(words[i : i + n])
        if all(tuple(words[i + k * n : i + (k + 1) * n]) == gram for k in range(1, times)):
            found.add(gram)
    return found


def repetition_flag(hyp_norm: str, ref_norm: str) -> bool:
    hyp = _consecutive_ngram_repeat(hyp_norm.split())
    if not hyp:
        return False
    ref = _consecutive_ngram_repeat(ref_norm.split())
    return bool(hyp - ref)


def word_recall(hits: int, substitutions: int, deletions: int) -> float:
    n = hits + substitutions + deletions
    return hits / n if n else 0.0


def truncation_flag(recall: float, duration_s: float, d_ref_s: float | None) -> bool | None:
    if d_ref_s is None or d_ref_s <= 0:
        return None
    return recall < TRUNC_RECALL and duration_s < TRUNC_DURATION_FRAC * d_ref_s


@dataclass
class RowQuality:
    row_id: str
    seed_index: int
    target_text: str
    thinker_text: str
    hyp_text: str
    ref_norm: str
    hyp_norm: str
    wer: float
    substitutions: int
    deletions: int
    insertions: int
    hits: int
    sim: float
    utmos: float
    duration_s: float
    d_ref_s: float | None
    leading_silence_s: float
    trailing_silence_s: float
    max_internal_silence_s: float
    silent_fraction: float
    catastrophic: bool
    silence: bool
    repetition: bool
    truncation: bool | None
    thinker_exact_match: bool
    whisper_wer: float | None = None

    def to_json(self) -> dict:
        return asdict(self)


def corpus_metrics(rows: list[RowQuality]) -> dict:
    """Corpus metrics over evaluated rows (all rows must have been scored)."""
    n = len(rows)
    if n == 0:
        raise ValueError("no evaluated rows")
    err = sum(r.substitutions + r.deletions + r.insertions for r in rows)
    ref = sum(r.substitutions + r.deletions + r.hits for r in rows)
    ok = [r for r in rows if r.wer <= CATASTROPHIC_WER]
    ok_err = sum(r.substitutions + r.deletions + r.insertions for r in ok)
    ok_ref = sum(r.substitutions + r.deletions + r.hits for r in ok)
    trunc = [r.truncation for r in rows]
    trunc_known = [t for t in trunc if t is not None]
    whisper = [r.whisper_wer for r in rows if r.whisper_wer is not None]
    return {
        "rows": n,
        "wer_corpus": err / ref if ref else float("nan"),
        "wer_filtered_corpus": ok_err / ok_ref if ok_ref else float("nan"),
        "catastrophic_count": sum(r.catastrophic for r in rows),
        "sim_mean": float(np.mean([r.sim for r in rows])),
        "utmos_mean": float(np.mean([r.utmos for r in rows])),
        "silence_rate": sum(r.silence for r in rows) / n,
        "repetition_rate": sum(r.repetition for r in rows) / n,
        "truncation_rate": (sum(trunc_known) / n) if len(trunc_known) == n else None,
        "silent_fraction_mean": float(np.mean([r.silent_fraction for r in rows])),
        "thinker_exact_match_rate": sum(r.thinker_exact_match for r in rows) / n,
        "duration_total_s": float(sum(r.duration_s for r in rows)),
        "whisper_wer_mean": float(np.mean(whisper)) if len(whisper) == n else None,
    }
