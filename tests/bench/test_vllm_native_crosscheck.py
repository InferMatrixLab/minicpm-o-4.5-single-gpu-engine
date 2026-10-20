"""Replay a recorded vllm-omni stream through vllm-omni's own benchmark client.

Run with the vllm-omni environment (the test skips elsewhere):

    "$OMNI_VLLM_REPO"/.venv/bin/python -m pytest tests/bench/test_vllm_native_crosscheck.py

The native client stamps ``audio_ttfp`` on the first ``modality == "audio"``
event even when it carries no samples, and ``audio_rtf`` on the last audio
event.  Our TTFA uses the first event that adds unique samples, so the native
value is compared with our first-audio-event time, and our TTFA must not be
earlier than it.
"""

import asyncio
import base64
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from bench.adapters import VllmOmniAdapter
from bench.client import replay
from bench.metrics import request_metrics

FIXTURE = Path(__file__).parent / "fixtures" / "vllm_omni_stream.json"


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Content:
    def __init__(self, chunks, clock) -> None:
        self._chunks = chunks
        self._clock = clock

    async def iter_any(self):
        for t, b in self._chunks:
            self._clock.now = t
            yield b


class _Response:
    status = 200
    reason = "OK"

    def __init__(self, chunks, clock) -> None:
        self.content = _Content(chunks, clock)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


class _Session:
    def __init__(self, chunks, clock) -> None:
        self._chunks = chunks
        self._clock = clock

    def post(self, url, json=None, headers=None):
        return _Response(self._chunks, self._clock)


def _load():
    fx = json.loads(FIXTURE.read_text())
    chunks = [(t, base64.b64decode(b)) for t, b in fx["chunks"]]
    return fx, chunks


def _first_audio_event_time(trace):
    return min(e.t for e in trace.events if e.kind == "audio")


def test_recorded_stream_matches_native_vllm_omni_client(monkeypatch):
    patch = pytest.importorskip("vllm_omni.benchmarks.patch.patch")
    fx, chunks = _load()
    clock = _Clock()
    clock.now = fx["t_dispatch"]
    monkeypatch.setattr(patch.time, "perf_counter", clock)
    rfi = SimpleNamespace(
        api_url="http://fixture/v1/chat/completions",
        model="openbmb/MiniCPM-o-4_5",
        model_name=None,
        prompt="x",
        prompt_len=1,
        output_len=256,
        extra_body={"modalities": ["text", "audio"]},
        multi_modal_content=None,
        ignore_eos=False,
        language=None,
        request_id=None,
        extra_headers=None,
    )
    out = asyncio.run(patch.async_request_openai_chat_omni_completions(rfi, _Session(chunks, clock)))
    assert out.success, out.error

    trace = replay(VllmOmniAdapter(), fx["request_id"], fx["t_dispatch"], chunks)
    ours = request_metrics(trace)
    assert ours.ok, ours.invalid_reasons

    first_audio = _first_audio_event_time(trace) - fx["t_dispatch"]
    assert abs(out.audio_ttfp - first_audio) <= 1e-3
    assert ours.ttfa_s >= out.audio_ttfp - 1e-9
    assert abs(out.audio_duration - ours.duration_s) <= 1e-6
    assert abs(out.audio_rtf - ours.rtf) <= 1e-3
