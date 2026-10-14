import math

import numpy as np
import pytest
from helpers import engine_audio, engine_text, header_only_wav_b64, sglang_body, sse, tone, vllm_audio, vllm_text, wav_b64

from bench.adapters import EngineAdapter, SglangOmniAdapter, VllmOmniAdapter
from bench.audio import SAMPLES_PER_CODEC_FRAME, decode_wav, encode_wav_pcm16, silence_fraction, silent_frame_mask
from bench.client import replay
from bench.metrics import RequestTrace, frames_from_samples, playback_stall_s, request_metrics, run_metrics

SR = 24000


def engine_stream(t0, pieces):
    """pieces: list of (t, seconds) audio chunks with distinct tones, then DONE at last t + 0.1."""
    chunks = [(t0 + 0.05, engine_text("Get the trust fund"))]
    off = 0
    for i, (t, sec) in enumerate(pieces):
        n = int(round(sec * SR))
        chunks.append((t, engine_audio(tone(n, 200 + 50 * i), off)))
        off += n
    chunks.append((pieces[-1][0] + 0.1, sse("[DONE]")))
    return chunks


def test_engine_fixture_exact_metrics():
    t0 = 100.0
    tr = replay(EngineAdapter(), "r1", t0, engine_stream(t0, [(100.5, 1.0), (101.2, 1.0), (103.0, 0.5)]))
    m = request_metrics(tr)
    assert m.ok, m.invalid_reasons
    assert m.duration_s == pytest.approx(2.5, abs=1e-12)
    assert m.ttfa_s == pytest.approx(0.5, abs=1e-12)
    assert m.rtf == pytest.approx(3.0 / 2.5, abs=1e-12)
    # exhaust after chunk 2 at 100.5 + 2.0 = 102.5; chunk 3 arrives at 103.0
    assert m.stall_s == pytest.approx(0.5, abs=1e-12)
    assert m.first_chunk_s == pytest.approx(1.0)
    assert m.inter_chunk_gaps_s == pytest.approx([0.7, 1.8])
    assert m.e2e_s == pytest.approx(3.1)
    assert tr.text == "Get the trust fund"


def test_back_to_back_stalls():
    # 1.0 s at 0.5; 0.5 s at 2.0 (stall 0.5); 1.0 s at 3.0 (exhaust 2.0 + 0.5 = 2.5 -> stall 0.5)
    assert playback_stall_s([0.5, 2.0, 3.0], [1.0, 1.5, 2.5]) == pytest.approx(1.0)
    tr = replay(EngineAdapter(), "r", 0.0, engine_stream(0.0, [(0.5, 1.0), (2.0, 0.5), (3.0, 1.0)]))
    assert request_metrics(tr).stall_s == pytest.approx(1.0)


def test_no_stall_when_audio_arrives_ahead():
    assert playback_stall_s([0.0, 0.5, 1.0, 1.5], [1.0, 2.0, 3.0, 4.0]) == 0.0
    assert playback_stall_s([1.0], [1.0]) == 0.0


def test_run_throughput_is_codec_frames_per_second():
    t0 = 10.0
    a = replay(EngineAdapter(), "a", t0, engine_stream(t0, [(10.4, 1.0), (11.0, 1.0)]))  # D=2, done 11.1
    b = replay(EngineAdapter(), "b", 10.5, engine_stream(10.5, [(11.5, 2.0)]))  # D=2, done 11.6
    rm = run_metrics([request_metrics(a), request_metrics(b)])
    assert rm.valid
    w = 11.6 - 10.0
    assert rm.makespan_s == pytest.approx(w)
    assert rm.throughput_frames_s == pytest.approx(25 * 4.0 / w)
    assert rm.ttfa_p50_s == pytest.approx(np.percentile([0.4, 1.0], 50))
    assert rm.ttfa_p95_s == pytest.approx(np.percentile([0.4, 1.0], 95))
    assert rm.rtf_mean == pytest.approx(np.mean([1.0 / 2.0, 1.0 / 2.0]))
    assert rm.audio_s_per_s == pytest.approx(4.0 / w)
    assert rm.req_s == pytest.approx(2 / w)


def test_counting_pcm_samples_as_frames_fails():
    # One codec frame is 960 samples at 24 kHz; 1 s of audio is 25 frames, not 24000.
    assert SAMPLES_PER_CODEC_FRAME == 960
    assert frames_from_samples(960) == pytest.approx(1.0)
    assert frames_from_samples(SR) == pytest.approx(25.0)
    tr = replay(EngineAdapter(), "a", 0.0, engine_stream(0.0, [(0.5, 1.0)]))
    rm = run_metrics([request_metrics(tr)])
    assert rm.throughput_frames_s == pytest.approx(25 / 0.6)
    assert not math.isclose(rm.throughput_frames_s, SR / 0.6)


def test_empty_first_audio_event_does_not_set_ttfa_vllm():
    chunks = [
        (0.10, vllm_text("Get")),
        (0.20, vllm_audio(header_only_wav_b64())),
        (0.25, vllm_audio(None)),
        (0.60, vllm_audio(wav_b64(tone(SR)))),
        (0.70, sse("[DONE]")),
    ]
    tr = replay(VllmOmniAdapter(), "v", 0.0, chunks)
    m = request_metrics(tr)
    assert m.ok, m.invalid_reasons
    assert m.ttfa_s == pytest.approx(0.60)
    assert m.duration_s == pytest.approx(1.0)
    assert m.rtf == pytest.approx(0.60)


def test_vllm_retransmission_flagged_and_excluded():
    a = tone(12000, 300)
    b = tone(12000, 500)
    chunks = [
        (0.5, vllm_audio(wav_b64(a))),
        (0.9, vllm_audio(wav_b64(b))),
        (1.0, vllm_audio(wav_b64(b))),  # resent
        (1.1, sse("[DONE]")),
    ]
    tr = replay(VllmOmniAdapter(), "v", 0.0, chunks)
    m = request_metrics(tr)
    assert tr.audio.retransmissions == 1
    assert m.duration_s == pytest.approx(1.0)
    assert m.rtf == pytest.approx(0.9 / 1.0)  # the resent chunk adds no unique samples
    assert "retransmission" in m.invalid_reasons
    assert not run_metrics([m]).valid


def test_vllm_silent_chunks_excluded_from_duplicate_detection():
    silence = np.zeros(4800, dtype=np.float32)
    chunks = [
        (0.5, vllm_audio(wav_b64(silence))),
        (0.6, vllm_audio(wav_b64(silence))),
        (0.7, vllm_audio(wav_b64(tone(4800)))),
        (0.8, sse("[DONE]")),
    ]
    tr = replay(VllmOmniAdapter(), "v", 0.0, chunks)
    m = request_metrics(tr)
    assert m.ok
    assert tr.audio.retransmissions == 0
    assert m.duration_s == pytest.approx(3 * 4800 / SR)


def test_engine_retransmitted_range_counts_once_and_is_flagged():
    a, b = tone(SR, 200), tone(SR, 300)
    chunks = [
        (0.5, engine_audio(a, 0)),
        (1.5, engine_audio(b, SR)),
        (1.6, engine_audio(b, SR)),
        (1.7, sse("[DONE]")),
    ]
    tr = replay(EngineAdapter(), "e", 0.0, chunks)
    m = request_metrics(tr)
    assert m.duration_s == pytest.approx(2.0)
    assert tr.audio.retransmissions == 1
    assert not m.ok and "retransmission" in m.invalid_reasons


def test_engine_gap_makes_request_invalid():
    chunks = [(0.5, engine_audio(tone(SR), 0)), (1.5, engine_audio(tone(SR), SR + 10)), (1.6, sse("[DONE]"))]
    m = request_metrics(replay(EngineAdapter(), "e", 0.0, chunks))
    assert "audio_gap" in m.invalid_reasons


def test_engine_num_samples_mismatch_invalid():
    chunks = [(0.5, engine_audio(tone(SR), 0, num=SR - 1)), (0.6, sse("[DONE]"))]
    m = request_metrics(replay(EngineAdapter(), "e", 0.0, chunks))
    assert "audio_num_samples_mismatch" in m.invalid_reasons


def test_missing_done_is_truncation():
    chunks = [(0.5, engine_audio(tone(SR), 0))]
    m = request_metrics(replay(EngineAdapter(), "e", 0.0, chunks))
    assert "truncated" in m.invalid_reasons


def test_empty_audio_invalid():
    chunks = [(0.5, engine_text("hi")), (0.6, sse("[DONE]"))]
    m = request_metrics(replay(EngineAdapter(), "e", 0.0, chunks))
    assert "empty_audio" in m.invalid_reasons and m.ttfa_s is None


def test_single_timeout_makes_run_invalid():
    good = request_metrics(replay(EngineAdapter(), "g", 0.0, engine_stream(0.0, [(0.5, 1.0)])))
    bad_trace = RequestTrace(request_id="t", t_dispatch=0.0, timed_out=True, error="timeout after 180s")
    bad = request_metrics(bad_trace)
    rm = run_metrics([good] * 50 + [bad])
    assert not rm.valid
    assert rm.n_failed == 1
    assert rm.invalid_reasons["timeout"] == 1


def test_stream_error_event_invalid():
    chunks = [(0.5, engine_audio(tone(SR), 0)), (0.6, sse({"error": {"message": "boom"}})), (0.7, sse("[DONE]"))]
    m = request_metrics(replay(EngineAdapter(), "e", 0.0, chunks))
    assert "error" in m.invalid_reasons


def test_sse_fragmentation_byte_by_byte_matches():
    t0 = 0.0
    chunks = engine_stream(t0, [(0.5, 0.4), (0.9, 0.4)])
    whole = request_metrics(replay(EngineAdapter(), "e", t0, chunks))
    frag = []
    for t, b in chunks:
        for i in range(len(b)):
            frag.append((t, b[i : i + 1]))
    split = request_metrics(replay(EngineAdapter(), "e", t0, frag))
    assert split.ttfa_s == whole.ttfa_s and split.rtf == whole.rtf and split.duration_s == whole.duration_s
    # events coalesced into one TCP read share that read's receipt time
    joined = [(0.9, b"".join(b for _, b in chunks))]
    m = request_metrics(replay(EngineAdapter(), "e", t0, joined))
    assert m.ttfa_s == pytest.approx(0.9) and m.rtf == pytest.approx(0.9 / 0.8)


def test_sglang_terminal_waveform_is_only_audio_event():
    body = sglang_body(tone(2 * SR))
    half = len(body) // 2
    tr = replay(SglangOmniAdapter(), "s", 0.0, [(1.0, body[:half]), (2.5, body[half:])], t_end=2.5)
    m = request_metrics(tr)
    assert m.ok, m.invalid_reasons
    assert m.ttfa_s == pytest.approx(2.5)  # end of body
    assert m.e2e_s == pytest.approx(2.5)
    assert m.rtf == pytest.approx(2.5 / 2.0)
    assert m.n_audio_events == 1
    assert tr.text == "hello"


def test_wav_roundtrip_and_float_wav():
    x = tone(1234)
    pcm = decode_wav(encode_wav_pcm16(x))
    assert pcm.num_samples == 1234 and pcm.sample_rate == SR
    assert np.max(np.abs(pcm.samples - x)) < 1e-4
    import struct

    data = x.astype("<f4").tobytes()
    hdr = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 3, 1, SR, SR * 4, 4, 32)
    fw = hdr + b"data" + struct.pack("<I", len(data)) + data
    assert np.allclose(decode_wav(fw).samples, x)
    # streaming placeholder data size
    fw2 = hdr + b"data" + struct.pack("<I", 0xFFFFFFFF) + data
    assert decode_wav(fw2).num_samples == 1234
    from bench.audio import WavFormatError

    # a declared 4800-sample chunk that carries only 480 samples is rejected
    short = encode_wav_pcm16(tone(4800))[: 44 + 960]
    with pytest.raises(WavFormatError):
        decode_wav(short)
    with pytest.raises(WavFormatError):  # partial sample frame
        decode_wav(hdr + b"data" + struct.pack("<I", 0) + data + b"\x00")
    nan = np.array([0.1, np.nan], dtype="<f4").tobytes()
    with pytest.raises(WavFormatError):
        decode_wav(hdr + b"data" + struct.pack("<I", len(nan)) + nan)
    with pytest.raises(WavFormatError):  # truncated fmt chunk
        decode_wav(b"RIFF" + struct.pack("<I", 20) + b"WAVEfmt " + struct.pack("<I", 16) + b"\x01\x00")


def test_truncated_wav_chunk_invalidates_request():
    short = encode_wav_pcm16(tone(4800))[: 44 + 960]
    import base64 as b64

    chunks = [(0.5, vllm_audio(b64.b64encode(short).decode())), (0.6, vllm_audio(wav_b64(tone(SR)))), (0.7, sse("[DONE]"))]
    m = request_metrics(replay(VllmOmniAdapter(), "v", 0.0, chunks))
    assert "audio_malformed_wav" in m.invalid_reasons


def test_events_after_done_are_ignored():
    chunks = [(0.5, engine_audio(tone(SR), 0)), (0.6, sse("[DONE]")), (0.9, engine_audio(tone(SR, 900), SR))]
    tr = replay(EngineAdapter(), "e", 0.0, chunks)
    m = request_metrics(tr)
    assert m.ok and m.duration_s == pytest.approx(1.0) and tr.t_complete == pytest.approx(0.6)


def test_silence_frames():
    x = np.concatenate([np.zeros(SR // 2, dtype=np.float32), tone(SR // 2)])
    mask = silent_frame_mask(x)
    assert mask.shape[0] == 50
    assert mask[:25].all() and not mask[25:].any()
    assert silence_fraction(x) == pytest.approx(0.5)
    # -50 dBFS boundary: RMS of 0.003 is about -50.5 dBFS
    quiet = tone(480, amp=0.003 * np.sqrt(2))
    assert silent_frame_mask(quiet)[0]
