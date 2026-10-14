"""Async HTTP client: one canonical request in, one timestamped ``RequestTrace`` out.

``ResponseConsumer`` holds all response handling.  The live client feeds it
bytes as they arrive; ``replay`` feeds it recorded ``(t, bytes)`` chunks, so
fixtures and recordings run through exactly the same code as live traffic.

A streaming request completes at its terminal event (``[DONE]``): the client
stops reading there and releases the closed-loop slot, and nothing received
afterwards can change the result.
"""

from __future__ import annotations

import asyncio
import json
import time

import aiohttp

from bench.adapters.base import Adapter
from bench.metrics import EventRecord, RequestTrace
from bench.sse import SSEParser
from bench.workload import CanonicalRequest

DEFAULT_TIMEOUT_S = 180.0


class ResponseConsumer:
    def __init__(self, adapter: Adapter, request_id: str, t_dispatch: float, streaming: bool | None = None) -> None:
        self.adapter = adapter
        self.streaming = adapter.streaming if streaming is None else streaming
        self.asm = adapter.new_assembler()
        self.trace = RequestTrace(request_id=request_id, t_dispatch=t_dispatch)
        self.trace.audio = self.asm.out
        self._parser = SSEParser()
        self._body = bytearray()

    @property
    def done(self) -> bool:
        return self.trace.terminal

    def _handle(self, t: float, data: str) -> None:
        try:
            self.adapter.handle_sse_data(t, data, self.trace, self.asm)
        except Exception as e:  # malformed payloads invalidate the request, never the run
            self.asm.out.flags.append("malformed_wav")
            self.trace.error = self.trace.error or f"parse error: {type(e).__name__}: {e}"
            self.trace.events.append(EventRecord(t=t, kind="other", nbytes=len(data), note="parse error"))
        if self.trace.terminal and self.trace.t_complete is None:
            self.trace.t_complete = t

    def feed(self, t: float, chunk: bytes) -> bool:
        """``t`` is the receipt time of ``chunk``, taken before decoding.  Returns True once terminal."""
        if self.done:
            return True
        if not self.streaming:
            self._body.extend(chunk)
            return False
        for data in self._parser.feed(chunk):
            self._handle(t, data)
            if self.done:
                break
        return self.done

    def finish(self, t_end: float) -> RequestTrace:
        """End of response at ``t_end`` (end of body for non-streaming responses)."""
        tr = self.trace
        if self.streaming:
            if not self.done:
                for data in self._parser.flush():
                    self._handle(t_end, data)
                    if self.done:
                        break
            if tr.t_complete is None:
                tr.t_complete = t_end
            return tr
        tr.t_complete = t_end
        try:
            obj = json.loads(bytes(self._body))
        except json.JSONDecodeError:
            tr.error = tr.error or "non-json response body"
            tr.events.append(EventRecord(t=t_end, kind="other", nbytes=len(self._body), note="non-json body"))
            return tr
        try:
            self.adapter.handle_body(t_end, obj, tr, self.asm)
        except Exception as e:
            self.asm.out.flags.append("malformed_wav")
            tr.error = tr.error or f"parse error: {type(e).__name__}: {e}"
        return tr


def replay(
    adapter: Adapter,
    request_id: str,
    t_dispatch: float,
    chunks: list[tuple[float, bytes]],
    t_end: float | None = None,
    streaming: bool | None = None,
) -> RequestTrace:
    """Run recorded or synthetic ``(t, bytes)`` chunks through the live parsing path."""
    c = ResponseConsumer(adapter, request_id, t_dispatch, streaming=streaming)
    c.trace.http_status = 200
    for t, b in chunks:
        if c.feed(t, b):
            break
    end = t_end if t_end is not None else (chunks[-1][0] if chunks else t_dispatch)
    return c.finish(end)


def make_session(connections: int = 256) -> aiohttp.ClientSession:
    # One connection per request: a pooled keep-alive connection that the
    # server closed while idle (between runs) would fail the next request.
    connector = aiohttp.TCPConnector(limit=connections, force_close=True)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=30)
    return aiohttp.ClientSession(connector=connector, timeout=timeout, auto_decompress=False)


async def _run(session, url: str, payload: dict, consumer: ResponseConsumer, raw: list | None) -> None:
    tr = consumer.trace
    tr.t_dispatch = time.perf_counter()  # immediately before the request is sent
    async with session.post(url, json=payload) as resp:
        tr.http_status = resp.status
        if resp.status != 200:
            text = await resp.text()
            tr.t_complete = time.perf_counter()
            tr.error = f"HTTP {resp.status}: {text[:500]}"
            return
        async for chunk in resp.content.iter_any():
            t = time.perf_counter()
            if raw is not None:
                raw.append((t, bytes(chunk)))
            if consumer.feed(t, chunk):
                resp.close()  # terminal event seen: release the slot now
                return
        consumer.finish(time.perf_counter())


async def send_request(
    session: aiohttp.ClientSession,
    base_url: str,
    adapter: Adapter,
    req: CanonicalRequest,
    timeout_s: float = DEFAULT_TIMEOUT_S,
    payload: dict | None = None,
    record_raw: bool = False,
) -> RequestTrace:
    """Send one request.  The payload is built before ``t_dispatch`` is taken."""
    if payload is None:
        payload = adapter.build_payload(req)
    url = base_url.rstrip("/") + adapter.endpoint
    consumer = ResponseConsumer(adapter, req.request_id, time.perf_counter(), streaming=bool(payload.get("stream", adapter.streaming)))
    raw: list | None = [] if record_raw else None
    tr = consumer.trace
    try:
        await asyncio.wait_for(_run(session, url, payload, consumer, raw), timeout=timeout_s)
    except asyncio.TimeoutError:
        if not tr.terminal:
            tr.timed_out = True
            tr.error = tr.error or f"timeout after {timeout_s}s"
    except (aiohttp.ClientError, ConnectionError) as e:
        if not tr.terminal:
            tr.error = tr.error or f"{type(e).__name__}: {e}"
    except Exception as e:  # never let one request abort the closed loop
        tr.error = tr.error or f"client error: {type(e).__name__}: {e}"
    if tr.t_complete is None and not tr.timed_out and not tr.error:
        tr.t_complete = time.perf_counter()
    if raw is not None:
        tr.raw_chunks = raw  # type: ignore[attr-defined]
    return tr
