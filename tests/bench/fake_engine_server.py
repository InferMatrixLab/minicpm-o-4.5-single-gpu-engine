"""Tiny engine-format SSE server for end-to-end harness tests.

    python tests/bench/fake_engine_server.py <port> [--delay S]

Streams 0.5 s of audio per request in two offset-tagged chunks, then [DONE].
"""

import asyncio
import json
import sys
from pathlib import Path

from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from helpers import engine_audio, engine_text, sse, tone  # noqa: E402

SR = 24000


async def health(_):
    return web.Response(text="ok")


async def chat(request: web.Request) -> web.StreamResponse:
    await request.json()
    resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    await resp.write(engine_text("hi"))
    await asyncio.sleep(DELAY)
    await resp.write(engine_audio(tone(SR // 4, 300), 0))
    await resp.write(engine_audio(tone(SR // 4, 400), SR // 4))
    await resp.write(sse("[DONE]"))
    return resp


async def info(_):
    """The served identity a real engine reports, for the test candidates' config name ("fake"); FAKE_INFO_OVERRIDE (a
    JSON object) replaces fields, to serve something other than the declaration."""
    import os

    from bench.cli import served_identity_expected

    return web.json_response({**served_identity_expected("fake"), **json.loads(os.environ.get("FAKE_INFO_OVERRIDE", "{}"))})


if __name__ == "__main__":
    port = int(sys.argv[1])
    DELAY = float(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--delay" else 0.001
    app = web.Application()
    app.router.add_get("/health", health)
    app.router.add_get("/info", info)
    app.router.add_post("/v1/chat/completions", chat)
    web.run_app(app, host="127.0.0.1", port=port, print=None)
    print(json.dumps({"stopped": True}))
