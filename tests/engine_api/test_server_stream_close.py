"""The server closes the backend's event stream on every path (CPU): a client that disconnects early, one that leaves
right at the end (while the last events are written) and a completed request all give their admission slot back at
once, with garbage collection disabled (it must not depend on a collector pass)."""

import asyncio
import gc
import sys
from pathlib import Path

import aiohttp
import numpy as np
from aiohttp.test_utils import TestServer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from engine.server import AudioChunk, Done, EngineServer, TextDelta  # noqa: E402

PAYLOAD = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "modalities": ["text", "audio"], "stream": True,
           "stream_options": {"include_usage": True}, "ref_audio": None}


class Slots:
    def __init__(self, n_audio=40, done_delay=0.0):
        self.admitted, self.n_audio, self.done_delay = 0, n_audio, done_delay

    async def generate(self, req):
        self.admitted += 1
        try:
            yield TextDelta("hi")
            for i in range(self.n_audio):
                yield AudioChunk(np.zeros(2400, np.float32), 2400 * i)
            await asyncio.sleep(self.done_delay)  # the client is gone by now: the final writes after Done fail
            yield Done("stop", {"audio_frames": 1})
        finally:
            self.admitted -= 1


async def _client(url, stop_after_lines):
    async with aiohttp.ClientSession() as s:
        async with s.post(url + "/v1/chat/completions", json=PAYLOAD) as r:
            n = 0
            async for line in r.content:
                n += line.startswith(b"data:")
                if stop_after_lines is not None and n >= stop_after_lines:
                    return  # disconnect


def test_every_path_gives_the_slot_back_without_gc():
    async def go():
        be = Slots(done_delay=0.2)
        srv = TestServer(EngineServer(be).app)
        await srv.start_server()
        url = f"http://127.0.0.1:{srv.port}"
        gc.disable()
        try:
            for stop in (2, 41, None):  # early, after the last audio (the writes after Done fail), completed
                await _client(url, stop)
                for _ in range(50):
                    if be.admitted == 0:
                        break
                    await asyncio.sleep(0.02)
                assert be.admitted == 0, stop
        finally:
            gc.enable()
            await srv.close()
    asyncio.run(go())
