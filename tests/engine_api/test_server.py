"""Engine API contract with a scripted backend, driven by the harness's own engine adapter and client."""

import asyncio
import base64
import io
import json
import wave
import sys
from pathlib import Path

import aiohttp
import numpy as np
import pytest
from aiohttp.test_utils import TestServer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from bench.adapters import get_adapter  # noqa: E402
from bench.client import make_session, send_request  # noqa: E402
from bench.metrics import request_metrics  # noqa: E402
from bench.workload import canonical_requests, load_seed_tts_rows, make_id_sets  # noqa: E402
from engine.server import AudioChunk, Done, EngineServer, Overloaded, TextDelta  # noqa: E402

SR = 24000


class Scripted:
    def __init__(self, chunks=3, delay=0.0, overloaded=False):
        self.chunks, self.delay, self.overloaded = chunks, delay, overloaded
        self.cancelled, self.seen = [], []

    async def generate(self, req):
        self.seen.append(req)
        if self.overloaded:
            raise Overloaded("at capacity")
        try:
            yield TextDelta("Hello ")
            yield TextDelta("world.")
            off = 0
            for i in range(self.chunks):
                await asyncio.sleep(self.delay)
                x = (0.2 * np.sin(np.arange(SR // 2) * 0.05 * (i + 1))).astype(np.float32)
                yield AudioChunk(x, off)
                off += len(x)
            yield Done("stop", {"prompt_tokens": 10, "completion_tokens": 3})
        except (GeneratorExit, asyncio.CancelledError):  # closed by the server, or cancelled with the handler
            self.cancelled.append(req.request_id)
            raise


def _req(stream=True):
    rows = load_seed_tts_rows()
    return canonical_requests(rows, make_id_sets(rows).warmup(1)[:1], stream=stream)[0]


async def _with_server(backend, fn):
    srv = TestServer(EngineServer(backend).app)
    await srv.start_server()
    try:
        return await fn(f"http://127.0.0.1:{srv.port}")
    finally:
        await srv.close()


def test_streaming_contract_through_the_harness_adapter():
    b = Scripted()

    async def go(url):
        async with make_session() as s:
            return await send_request(s, url, get_adapter("engine"), _req())

    tr = asyncio.run(_with_server(b, go))
    m = request_metrics(tr)
    assert m.ok and tr.terminal and tr.text == "Hello world." and abs(m.duration_s - 1.5) < 1e-6 and m.ttfa_s is not None
    r = b.seen[0]
    assert r.ref_audio and r.chat_template_kwargs["use_tts_template"] is True and r.talker_sampling["top_k"] == 25 and r.seed is not None


def test_non_streaming_body():
    async def go(url):
        async with make_session() as s:
            return await send_request(s, url, get_adapter("engine"), _req(stream=False))

    tr = asyncio.run(_with_server(Scripted(), go))
    m = request_metrics(tr)
    assert m.ok and tr.text == "Hello world." and abs(m.duration_s - 1.5) < 1e-6


def _empty_wav_data_url() -> str:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


@pytest.mark.parametrize("body", [
    "not json",
    {},
    {"messages": []},
    {"messages": [{"role": "robot", "content": "x"}]},
    {"messages": [{"role": "user", "content": [{"type": "hologram"}]}]},
    {"messages": [{"role": "user", "content": "x"}], "modalities": ["smell"]},
    {"messages": [{"role": "user", "content": "x"}], "modalities": ["audio"]},  # audio without a voice prompt
    {"messages": [{"role": "user", "content": "x"}], "modalities": ["text"], "max_tokens": -1},
    {"messages": [{"role": "user", "content": "x"}], "modalities": ["text"], "talker_sampling": {"top_k": 0}},
    {"messages": [{"role": "user", "content": "x"}], "ref_audio": "data:audio/wav;base64,@@@"},
    {"messages": [{"role": "user", "content": "x"}], "modalities": ["text"], "stream": "yes"},
    # nested media payloads of the wrong type (the image case used to reach the media decoder and return 500)
    {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "invalid"}]}], "modalities": ["text"]},
    {"messages": [{"role": "user", "content": [{"type": "audio_url", "audio_url": {"url": 3}}]}], "modalities": ["text"]},
    {"messages": [{"role": "user", "content": [{"type": "video_url"}]}], "modalities": ["text"]},
    {"messages": [{"role": "user", "content": [{"type": "input_audio", "input_audio": "AAAA"}]}], "modalities": ["text"]},
    # voice prompts that are valid base64 but not decodable audio, or hold no samples
    {"messages": [{"role": "user", "content": "x"}], "ref_audio": "data:audio/wav;base64," + base64.b64encode(b"\x13" * 4000).decode()},
    {"messages": [{"role": "user", "content": "x"}], "ref_audio": _empty_wav_data_url()},
])
def test_malformed_requests_get_400(body):
    async def go(url):
        async with aiohttp.ClientSession() as s:
            data = body if isinstance(body, str) else json.dumps(body)
            async with s.post(url + "/v1/chat/completions", data=data, headers={"Content-Type": "application/json"}) as r:
                return r.status, await r.json()

    status, doc = asyncio.run(_with_server(Scripted(), go))
    assert status == 400 and doc["error"]["type"] == "invalid_request_error"


def test_overload_returns_503():
    async def go(url):
        async with aiohttp.ClientSession() as s:
            p = get_adapter("engine").build_payload(_req())
            async with s.post(url + "/v1/chat/completions", json=p) as r:
                return r.status

    assert asyncio.run(_with_server(Scripted(overloaded=True), go)) == 503


def test_disconnect_cancels_the_request():
    b = Scripted(chunks=50, delay=0.05)

    async def go(url):
        async with aiohttp.ClientSession() as s:
            p = get_adapter("engine").build_payload(_req())
            async with s.post(url + "/v1/chat/completions", json=p) as r:
                async for _ in r.content.iter_any():
                    break  # read one chunk, then drop the connection
        for _ in range(100):
            if b.cancelled:
                break
            await asyncio.sleep(0.05)

    asyncio.run(_with_server(b, go))
    assert b.cancelled


class FailsMidStream(Scripted):
    async def generate(self, req):
        yield TextDelta("Hello ")
        yield AudioChunk(np.zeros(SR // 4, np.float32), 0)
        raise RuntimeError("engine error: boom")


def test_engine_failure_mid_stream_ends_with_an_error_event():
    """An engine failure after the stream began is an OpenAI-style error event followed by [DONE] on a cleanly ended
    chunked body (not a cut connection); the harness client records the error and the request as invalid."""
    adapter = get_adapter("engine")

    async def go(url):
        async with aiohttp.ClientSession() as s:
            async with s.post(url + "/v1/chat/completions", json=adapter.build_payload(_req())) as r:
                body = (await r.read()).decode()
        async with make_session(connections=2) as sess:
            trace = await send_request(sess, url, adapter, _req(), timeout_s=30)
        return body, trace

    body, trace = asyncio.run(_with_server(FailsMidStream(), go))
    lines = [ln for ln in body.split("\n\n") if ln.strip()]
    assert lines[-1] == "data: [DONE]"
    err = json.loads(lines[-2][len("data: "):])
    assert err["error"]["type"] == "server_error" and "boom" in err["error"]["message"]
    assert not request_metrics(trace).ok


def test_bad_inputs_are_400_in_both_modes_and_the_next_request_is_served():
    """Malformed media and an undecodable voice are rejected before any response header in stream and non-stream modes;
    the admission state is untouched (the next request is served)."""
    import base64

    adapter = get_adapter("engine")
    bad_voice = "data:audio/wav;base64," + base64.b64encode(b"\x13" * 4000).decode()

    async def go(url):
        statuses = []
        async with aiohttp.ClientSession() as s:
            for stream in (True, False):
                for body in ({"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": "invalid"}]}], "modalities": ["text"]},
                             {"messages": [{"role": "user", "content": "x"}], "ref_audio": bad_voice}):
                    async with s.post(url + "/v1/chat/completions", json={**body, "stream": stream}) as r:
                        statuses.append((r.status, (await r.json())["error"]["type"]))
        async with make_session(connections=2) as sess:
            trace = await send_request(sess, url, adapter, _req(), timeout_s=30)
        return statuses, trace

    statuses, trace = asyncio.run(_with_server(Scripted(), go))
    assert statuses == [(400, "invalid_request_error")] * 4
    assert request_metrics(trace).ok


def test_worker_reference_audio_failure_stays_an_input_error():
    """A reference-audio failure reported by the Token2Wav worker reaches the request as BadRequest (other worker
    failures stay runtime errors)."""
    import queue as _q

    from engine.runtime.backend_s import TrackSBackend
    from engine.server import BadRequest

    class LV:
        def __init__(self):
            self.cancelled, self.out = False, _q.Queue()

        def emit(self, ev):
            self.out.put(ev)

    b = TrackSBackend.__new__(TrackSBackend)
    for kind, want in (("bad_request", BadRequest), (None, RuntimeError)):
        lv = LV()
        b.audio_live = {7: lv}
        b._on_t2w(("error", 7, "reference audio: LibsndfileError: x") + ((kind,) if kind else ()))
        ev = lv.out.get_nowait()
        assert type(ev) is want, (kind, type(ev))
