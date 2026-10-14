"""Client-side metric definitions shared by every engine.

All timestamps are ``time.perf_counter()`` values taken by the client:
``t_dispatch`` immediately before the request is sent, each event on receipt
(before decoding), completion at the terminal event or end of body.

Per request:
    D_i    = unique delivered PCM samples / sample_rate          (seconds)
    TTFA_i = t(first event adding >= 1 unique sample) - t_dispatch
    RTF_i  = (t(last event adding unique samples) - t_dispatch) / D_i

Per run:
    Throughput (frames/s) = 25 * sum(D_i) / W, W = first measured dispatch to
    last measured completion.  TTFA p50/p95 use numpy linear interpolation.
    RTF mean is the unweighted mean of RTF_i.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field

import numpy as np

from bench.audio import CODEC_FRAME_RATE, SAMPLE_RATE, AssembledAudio, AudioContribution


@dataclass
class EventRecord:
    t: float
    kind: str  # "text" | "audio" | "usage" | "error" | "done" | "body" | "other"
    nbytes: int = 0
    samples: int = 0
    unique_samples: int = 0
    note: str = ""


@dataclass
class RequestTrace:
    request_id: str
    t_dispatch: float
    t_complete: float | None = None
    http_status: int | None = None
    terminal: bool = False
    timed_out: bool = False
    error: str = ""
    text: str = ""
    usage: dict = field(default_factory=dict)
    audio: AssembledAudio = field(default_factory=AssembledAudio)
    events: list[EventRecord] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "request_id": self.request_id,
            "t_dispatch": self.t_dispatch,
            "t_complete": self.t_complete,
            "http_status": self.http_status,
            "terminal": self.terminal,
            "timed_out": self.timed_out,
            "error": self.error,
            "text": self.text,
            "usage": self.usage,
            "audio_flags": list(self.audio.flags),
            "audio_sample_rate": self.audio.sample_rate,
            "contributions": [asdict(c) for c in self.audio.contributions],
            "events": [asdict(e) for e in self.events],
        }


@dataclass
class RequestMetrics:
    request_id: str
    ok: bool
    invalid_reasons: list[str]
    duration_s: float
    ttfa_s: float | None
    rtf: float | None
    e2e_s: float | None
    stall_s: float
    first_chunk_s: float
    inter_chunk_gaps_s: list[float]
    n_audio_events: int
    retransmissions: int
    t_dispatch: float
    t_complete: float | None


def _contributing(contribs: list[AudioContribution]) -> list[AudioContribution]:
    return [c for c in contribs if c.unique_samples > 0]


def playback_stall_s(times: list[float], cumulative_audio_s: list[float]) -> float:
    """Total playback stall for audio arriving at ``times``.

    Playback starts at the first arrival and advances in real time while
    buffered audio exists; it pauses when the buffer empties.  With
    ``t_exhaust = t_resume + (S_{k-1} - p_resume)``: if ``t_k > t_exhaust`` the
    stall grows by ``t_k - t_exhaust`` and playback resumes at ``t_k`` from
    position ``S_{k-1}``.  Accounting ends at the last audio event.
    """
    if len(times) != len(cumulative_audio_s):
        raise ValueError("times and cumulative_audio_s must have equal length")
    if len(times) < 2:
        return 0.0
    stall = 0.0
    t_resume = times[0]
    p_resume = 0.0
    for k in range(1, len(times)):
        t_exhaust = t_resume + (cumulative_audio_s[k - 1] - p_resume)
        if times[k] > t_exhaust:
            stall += times[k] - t_exhaust
            t_resume = times[k]
            p_resume = cumulative_audio_s[k - 1]
    return stall


def request_metrics(trace: RequestTrace) -> RequestMetrics:
    reasons: list[str] = []
    if trace.timed_out:
        reasons.append("timeout")
    if trace.error:
        reasons.append("error")
    if trace.http_status is not None and trace.http_status != 200:
        reasons.append(f"http_{trace.http_status}")
    if not trace.terminal:
        reasons.append("truncated")
    contribs = _contributing(trace.audio.contributions)
    rate = trace.audio.sample_rate or SAMPLE_RATE
    unique = sum(c.unique_samples for c in contribs)
    duration = unique / float(rate)
    if unique == 0:
        reasons.append("empty_audio")
    if trace.audio.retransmissions:
        reasons.append("retransmission")
    for flag in trace.audio.flags:
        for prefix in (
            "gap",
            "overlap",
            "num_samples_mismatch",
            "sample_rate",
            "multiple_terminal_waveforms",
            "malformed_wav",
            "missing_offsets",
        ):
            if flag.startswith(prefix):
                reasons.append(f"audio_{prefix}")
    ttfa = rtf = None
    stall = 0.0
    gaps: list[float] = []
    first_chunk = 0.0
    if contribs:
        ttfa = contribs[0].t - trace.t_dispatch
        rtf = (contribs[-1].t - trace.t_dispatch) / duration
        first_chunk = contribs[0].unique_samples / float(rate)
        times = [c.t for c in contribs]
        cum = np.cumsum([c.unique_samples / float(rate) for c in contribs]).tolist()
        stall = playback_stall_s(times, cum)
        gaps = [b - a for a, b in zip(times, times[1:])]
    e2e = None if trace.t_complete is None else trace.t_complete - trace.t_dispatch
    return RequestMetrics(
        request_id=trace.request_id,
        ok=not reasons,
        invalid_reasons=sorted(set(reasons)),
        duration_s=duration,
        ttfa_s=ttfa,
        rtf=rtf,
        e2e_s=e2e,
        stall_s=stall,
        first_chunk_s=first_chunk,
        inter_chunk_gaps_s=gaps,
        n_audio_events=len(contribs),
        retransmissions=trace.audio.retransmissions,
        t_dispatch=trace.t_dispatch,
        t_complete=trace.t_complete,
    )


def _pct(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q, method="linear"))


@dataclass
class RunMetrics:
    n_requests: int
    n_failed: int
    valid: bool
    invalid_reasons: dict[str, int]
    makespan_s: float
    throughput_frames_s: float
    ttfa_p50_s: float
    ttfa_p95_s: float
    rtf_mean: float
    rtf_duration_weighted: float
    req_s: float
    audio_s_per_s: float
    e2e_p50_s: float
    e2e_p95_s: float
    total_audio_s: float
    first_chunk_mean_s: float
    stall_mean_s: float
    stall_max_s: float
    inter_chunk_gap_p50_s: float
    inter_chunk_gap_p95_s: float
    inter_chunk_gap_max_s: float

    def to_json(self) -> dict:
        return asdict(self)


def run_metrics(per_request: list[RequestMetrics]) -> RunMetrics:
    """Aggregate one measured run.  Any invalid request makes the run invalid."""
    if not per_request:
        raise ValueError("run has no measured requests")
    reasons: dict[str, int] = {}
    for m in per_request:
        for r in m.invalid_reasons:
            reasons[r] = reasons.get(r, 0) + 1
    n_failed = sum(1 for m in per_request if not m.ok)
    first_dispatch = min(m.t_dispatch for m in per_request)
    completes = [m.t_complete for m in per_request if m.t_complete is not None]
    last_complete = max(completes) if completes else first_dispatch
    makespan = last_complete - first_dispatch
    total_d = sum(m.duration_s for m in per_request)
    ttfas = [m.ttfa_s for m in per_request if m.ttfa_s is not None]
    rtfs = [m.rtf for m in per_request if m.rtf is not None]
    e2es = [m.e2e_s for m in per_request if m.e2e_s is not None]
    weighted_num = sum(m.rtf * m.duration_s for m in per_request if m.rtf is not None)
    weighted_den = sum(m.duration_s for m in per_request if m.rtf is not None)
    gaps = [g for m in per_request for g in m.inter_chunk_gaps_s]
    stalls = [m.stall_s for m in per_request]
    firsts = [m.first_chunk_s for m in per_request if m.n_audio_events]
    safe = makespan if makespan > 0 else float("nan")
    return RunMetrics(
        n_requests=len(per_request),
        n_failed=n_failed,
        valid=n_failed == 0,
        invalid_reasons=reasons,
        makespan_s=makespan,
        throughput_frames_s=CODEC_FRAME_RATE * total_d / safe,
        ttfa_p50_s=_pct(ttfas, 50),
        ttfa_p95_s=_pct(ttfas, 95),
        rtf_mean=float(np.mean(rtfs)) if rtfs else float("nan"),
        rtf_duration_weighted=weighted_num / weighted_den if weighted_den > 0 else float("nan"),
        req_s=len(per_request) / safe,
        audio_s_per_s=total_d / safe,
        e2e_p50_s=_pct(e2es, 50),
        e2e_p95_s=_pct(e2es, 95),
        total_audio_s=total_d,
        first_chunk_mean_s=float(np.mean(firsts)) if firsts else float("nan"),
        stall_mean_s=float(np.mean(stalls)) if stalls else 0.0,
        stall_max_s=float(np.max(stalls)) if stalls else 0.0,
        inter_chunk_gap_p50_s=_pct(gaps, 50),
        inter_chunk_gap_p95_s=_pct(gaps, 95),
        inter_chunk_gap_max_s=float(np.max(gaps)) if gaps else float("nan"),
    )


def frames_from_samples(num_samples: int, sample_rate: int = SAMPLE_RATE) -> float:
    """Codec-frame-equivalents (25 Hz) for a PCM sample count: 960 samples per frame at 24 kHz."""
    return num_samples / float(sample_rate) * CODEC_FRAME_RATE


def is_finite(x: float | None) -> bool:
    return x is not None and math.isfinite(x)
