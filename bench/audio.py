"""PCM decoding, loudness and per-engine audio assembly.

Every adapter turns the audio payloads of one response into a sequence of
``AudioContribution`` records.  A contribution says how many *unique* PCM
samples an event added to the request's output, so TTFA, RTF, duration and the
stall model are computed the same way for every engine.
"""

from __future__ import annotations

import hashlib
import io
import struct
from dataclasses import dataclass, field

import numpy as np

SAMPLE_RATE = 24000
CODEC_FRAME_RATE = 25
SAMPLES_PER_CODEC_FRAME = SAMPLE_RATE // CODEC_FRAME_RATE  # 960
SILENCE_DBFS = -50.0
SILENCE_FRAME_S = 0.020


class WavFormatError(ValueError):
    pass


@dataclass(frozen=True)
class Pcm:
    samples: np.ndarray  # float32 in [-1, 1], mono
    sample_rate: int

    @property
    def num_samples(self) -> int:
        return int(self.samples.shape[0])


#: Data-chunk sizes that streaming writers use when the length is unknown.
_PLACEHOLDER_SIZES = (0, 0xFFFFFFFF)


def decode_wav(data: bytes) -> Pcm:
    """Decode a self-contained RIFF/WAVE payload (PCM16 or IEEE float32) into mono float32.

    A header-only payload decodes to zero samples.  A data chunk whose size is
    the streaming placeholder (0 or 0xFFFFFFFF) takes the bytes present; any
    other declared size must be fully present.  Partial sample frames and
    non-finite float samples are rejected.
    """
    if len(data) < 12 or data[0:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WavFormatError("not a RIFF/WAVE payload")
    pos = 12
    fmt = None
    pcm_bytes = None
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        (chunk_size,) = struct.unpack_from("<I", data, pos + 4)
        body_start = pos + 8
        available = len(data) - body_start
        if chunk_id == b"fmt ":
            if chunk_size < 16 or available < chunk_size:
                raise WavFormatError("fmt chunk truncated")
            audio_format, channels, sample_rate, _, _, bits = struct.unpack_from("<HHIIHH", data, body_start)
            if audio_format == 0xFFFE:
                if chunk_size < 40:
                    raise WavFormatError("extensible fmt chunk truncated")
                (audio_format,) = struct.unpack_from("<H", data, body_start + 24)
            fmt = (audio_format, channels, sample_rate, bits)
        elif chunk_id == b"data":
            if chunk_size in _PLACEHOLDER_SIZES:
                chunk_size = available
            elif chunk_size > available:
                raise WavFormatError(f"data chunk truncated: declared {chunk_size} bytes, have {available}")
            pcm_bytes = data[body_start : body_start + chunk_size]
            break
        elif chunk_size > available:
            raise WavFormatError("chunk truncated")
        pos = body_start + chunk_size + (chunk_size & 1)
    if fmt is None:
        raise WavFormatError("missing fmt chunk")
    audio_format, channels, sample_rate, bits = fmt
    if pcm_bytes is None:
        pcm_bytes = b""
    if channels < 1 or sample_rate < 1:
        raise WavFormatError("invalid channel count or sample rate")
    if audio_format == 1 and bits == 16:
        frame_bytes = 2 * channels
        if len(pcm_bytes) % frame_bytes:
            raise WavFormatError("partial sample frame")
        raw = np.frombuffer(pcm_bytes, dtype="<i2").astype(np.float32) / 32768.0
    elif audio_format == 3 and bits == 32:
        frame_bytes = 4 * channels
        if len(pcm_bytes) % frame_bytes:
            raise WavFormatError("partial sample frame")
        raw = np.frombuffer(pcm_bytes, dtype="<f4").astype(np.float32)
        if not np.isfinite(raw).all():
            raise WavFormatError("non-finite float samples")
    else:
        raise WavFormatError(f"unsupported WAV encoding format={audio_format} bits={bits}")
    if channels > 1:
        raw = raw.reshape(-1, channels).mean(axis=1)
    return Pcm(samples=np.ascontiguousarray(raw, dtype=np.float32), sample_rate=int(sample_rate))


def encode_wav_pcm16(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 32767.0 / 32768.0)
    pcm = np.round(clipped * 32768.0).astype("<i2").tobytes()
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + len(pcm)))
    buf.write(b"WAVEfmt ")
    buf.write(struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, sample_rate * 2, 2, 16))
    buf.write(b"data")
    buf.write(struct.pack("<I", len(pcm)))
    buf.write(pcm)
    return buf.getvalue()


def rms_dbfs(samples: np.ndarray) -> float:
    if samples.size == 0:
        return float("-inf")
    rms = float(np.sqrt(np.mean(np.square(samples.astype(np.float64)))))
    if rms <= 0.0:
        return float("-inf")
    return 20.0 * float(np.log10(rms))


def silent_frame_mask(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Boolean mask over consecutive 20 ms frames: True where RMS < -50 dBFS.

    A trailing partial frame is evaluated on the samples it has.
    """
    frame = max(1, int(round(SILENCE_FRAME_S * sample_rate)))
    n = samples.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    n_frames = (n + frame - 1) // frame
    out = np.empty(n_frames, dtype=bool)
    for i in range(n_frames):
        out[i] = rms_dbfs(samples[i * frame : (i + 1) * frame]) < SILENCE_DBFS
    return out


def silence_fraction(samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> float:
    mask = silent_frame_mask(samples, sample_rate)
    return float(mask.mean()) if mask.size else 0.0


def pcm_hash(samples: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(samples, dtype=np.float32).tobytes()).hexdigest()


@dataclass
class AudioContribution:
    """One audio-bearing event after assembly."""

    t: float  # receipt timestamp (perf_counter) of the event
    received_samples: int  # samples carried by the event
    unique_samples: int  # samples this event added to the output
    sample_rate: int
    retransmission: bool = False
    gap_samples: int = 0  # missing samples before this event (offset adapters)
    overlap_samples: int = 0  # partially re-sent samples (offset adapters)


@dataclass
class AssembledAudio:
    contributions: list[AudioContribution] = field(default_factory=list)
    chunks: list[np.ndarray] = field(default_factory=list)
    sample_rate: int | None = None
    flags: list[str] = field(default_factory=list)

    @property
    def unique_samples(self) -> int:
        return sum(c.unique_samples for c in self.contributions)

    @property
    def retransmissions(self) -> int:
        return sum(1 for c in self.contributions if c.retransmission)

    def pcm(self) -> np.ndarray:
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(self.chunks).astype(np.float32, copy=False)

    def _note_rate(self, sample_rate: int) -> None:
        if self.sample_rate is None:
            self.sample_rate = sample_rate
            if sample_rate != SAMPLE_RATE:
                self.flags.append(f"sample_rate_{sample_rate}")
        elif sample_rate != self.sample_rate:
            self.flags.append(f"sample_rate_changed_{self.sample_rate}_to_{sample_rate}")


class WavChunkAssembler:
    """vllm-omni: each audio event is a self-contained WAV, appended in order.

    A chunk whose PCM hash equals an earlier chunk of the same request is a
    retransmission and is not counted.  Chunks quieter than -50 dBFS are
    excluded from duplicate detection because silence legitimately repeats.
    """

    def __init__(self) -> None:
        self.out = AssembledAudio()
        self._seen: set[str] = set()

    def add(self, t: float, wav_bytes: bytes) -> AudioContribution:
        pcm = decode_wav(wav_bytes)
        self.out._note_rate(pcm.sample_rate)
        n = pcm.num_samples
        retrans = False
        if n > 0 and rms_dbfs(pcm.samples) >= SILENCE_DBFS:
            h = pcm_hash(pcm.samples)
            if h in self._seen:
                retrans = True
            else:
                self._seen.add(h)
        unique = 0 if retrans else n
        if unique:
            self.out.chunks.append(pcm.samples)
        c = AudioContribution(
            t=t,
            received_samples=n,
            unique_samples=unique,
            sample_rate=pcm.sample_rate,
            retransmission=retrans,
        )
        self.out.contributions.append(c)
        return c


class OffsetAssembler:
    """New engine: every delta carries ``audio_offset_samples`` and ``num_samples``.

    Deltas must be contiguous.  A fully re-sent range counts once and is
    flagged; gaps and partial overlaps are flagged and make the run invalid.
    """

    def __init__(self) -> None:
        self.out = AssembledAudio()
        self._end = 0

    def add(self, t: float, wav_bytes: bytes, offset: int, num_samples: int) -> AudioContribution:
        pcm = decode_wav(wav_bytes)
        self.out._note_rate(pcm.sample_rate)
        n = pcm.num_samples
        if n != num_samples:
            self.out.flags.append(f"num_samples_mismatch_{num_samples}_vs_{n}")
        gap = overlap = 0
        retrans = False
        if offset > self._end:
            gap = offset - self._end
            self.out.flags.append(f"gap_{self._end}_{offset}")
            keep = pcm.samples
        elif offset + n <= self._end:
            retrans = n > 0
            keep = pcm.samples[:0]
        elif offset < self._end:
            overlap = self._end - offset
            self.out.flags.append(f"overlap_{offset}_{self._end}")
            keep = pcm.samples[overlap:]
        else:
            keep = pcm.samples
        unique = int(keep.shape[0])
        if unique:
            self.out.chunks.append(keep)
            self._end = max(self._end, offset + n)
        c = AudioContribution(
            t=t,
            received_samples=n,
            unique_samples=unique,
            sample_rate=pcm.sample_rate,
            retransmission=retrans,
            gap_samples=gap,
            overlap_samples=overlap,
        )
        self.out.contributions.append(c)
        return c


class TerminalWaveformAssembler:
    """sglang-omni non-streaming speech: one terminal waveform is the only audio event."""

    def __init__(self) -> None:
        self.out = AssembledAudio()

    def add(self, t: float, wav_bytes: bytes) -> AudioContribution:
        if self.out.contributions:
            self.out.flags.append("multiple_terminal_waveforms")
        pcm = decode_wav(wav_bytes)
        self.out._note_rate(pcm.sample_rate)
        if pcm.num_samples:
            self.out.chunks.append(pcm.samples)
        c = AudioContribution(
            t=t,
            received_samples=pcm.num_samples,
            unique_samples=pcm.num_samples,
            sample_rate=pcm.sample_rate,
        )
        self.out.contributions.append(c)
        return c
