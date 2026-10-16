"""OpenAI-compatible ``/v1/chat/completions`` server for the MiniCPM-o 4.5 engine.

Request: ``messages`` (text, ``image_url``, ``input_audio`` / ``audio_url`` and
``video_url`` parts), ``modalities`` ⊆ {text, audio}, ``stream``, ``max_tokens``,
``temperature``, ``seed``, ``chat_template_kwargs``, the voice prompt
``ref_audio`` (data URL, base64 or path; also accepted under ``audio.ref_audio``
and ``extra_body``) and optional ``talker_sampling``.  Malformed requests get
HTTP 400 with an OpenAI-style error; an engine at its admission limit answers 503.

Streaming responses are SSE ``chat.completion.chunk`` events:

* text: ``{"modality": "text", "choices": [{"delta": {"content": ...}}]}``;
* audio: ``{"modality": "audio", "audio_offset_samples": o, "num_samples": n,
  "choices": [{"delta": {"content": <base64 WAV, 24 kHz mono PCM16>}}]}`` with
  contiguous, non-overlapping sample ranges;
* a final chunk with ``finish_reason``, a usage chunk when
  ``stream_options.include_usage``, then ``data: [DONE]``.

``stream=false`` returns the text and the whole waveform in one body.  A client
disconnect closes the backend's event stream, which cancels the request.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import struct
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Protocol

import numpy as np
from aiohttp import web

SAMPLE_RATE = 24000
MAX_TOKENS_LIMIT = 4096
PART_TYPES = {"text", "image_url", "input_audio", "audio_url", "video_url"}
PROCESSOR_FIELDS = ("max_slice_nums", "use_image_id")  # the checkpoint processor's image policy (native Daily-Omni frames: 1, False)
TALKER_FIELDS = {"temperature": (0.0, 5.0), "top_p": (0.0, 1.0), "top_k": (1, 6562), "repetition_penalty": (0.5, 5.0),
                 "repetition_window": (0, 512), "min_new_tokens": (0, 4096), "max_new_tokens": (1, 4096)}


class BadRequest(ValueError):
    pass


class Overloaded(RuntimeError):
    """The engine is at its admission limit and the request was not queued."""


@dataclass
class EngineRequest:
    request_id: str
    messages: list[dict]
    modalities: list[str]
    max_tokens: int
    temperature: float
    seed: int | None
    chat_template_kwargs: dict
    ref_audio: bytes | None
    talker_sampling: dict
    stream: bool
    processor_kwargs: dict = field(default_factory=dict)  # validated mm_processor_kwargs (image slicing policy)


@dataclass
class TextDelta:
    text: str


@dataclass
class AudioChunk:
    samples: np.ndarray  # float32 [-1, 1], 24 kHz mono
    offset: int  # first sample index within the utterance


@dataclass
class Done:
    finish_reason: str = "stop"
    usage: dict = field(default_factory=dict)


class Backend(Protocol):
    def generate(self, req: EngineRequest) -> AsyncIterator[TextDelta | AudioChunk | Done]:
        """Events of one request; closing the iterator early cancels the request."""


def wav_pcm16(samples: np.ndarray, sr: int = SAMPLE_RATE) -> bytes:
    pcm = (np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVEfmt " + struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) + b"data" + \
        struct.pack("<I", len(pcm)) + pcm


def _decode_audio_ref(v) -> bytes:
    if isinstance(v, dict):
        v = v.get("data") or v.get("url")
    if not isinstance(v, str) or not v:
        raise BadRequest("ref_audio must be a data URL, base64 audio or a file path")
    try:
        if v.startswith("data:"):
            return base64.b64decode(v.split(",", 1)[1], validate=True)
        if v.startswith("file://") or v.startswith("/"):
            with open(v.removeprefix("file://"), "rb") as f:
                return f.read()
        return base64.b64decode(v, validate=True)
    except (ValueError, IndexError, OSError) as e:
        raise BadRequest(f"ref_audio is not readable: {e}") from None


def _check_decodable_audio(data: bytes) -> None:
    """The voice prompt must decode (as the Token2Wav feature path reads it, librosa at 16 kHz) and hold samples: checked
    before any response header, so an unreadable voice is an HTTP 400 rather than a failure mid-stream."""
    import io

    import soundfile as sf

    try:
        n = sf.info(io.BytesIO(data)).frames
    except Exception:
        try:
            import librosa

            n = len(librosa.load(io.BytesIO(data), sr=16000, mono=True, duration=1.0)[0])
        except Exception as e:
            raise BadRequest(f"ref_audio is not decodable audio: {type(e).__name__}") from None
    if not n:
        raise BadRequest("ref_audio holds no samples")


def _number(d: dict, k: str, kind, lo, hi, default):
    v = d.get(k, default)
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind is int and not float(v).is_integer()):
        raise BadRequest(f"{k} must be a {kind.__name__}")
    v = kind(v)
    if not lo <= v <= hi:
        raise BadRequest(f"{k}={v} outside [{lo}, {hi}]")
    return v


def parse_request(body) -> EngineRequest:
    if not isinstance(body, dict):
        raise BadRequest("request body must be a JSON object")
    extra = body.get("extra_body") if isinstance(body.get("extra_body"), dict) else {}
    body = {**extra, **body}
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs:
        raise BadRequest("messages must be a non-empty list")
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant"):
            raise BadRequest("every message needs a role of system, user or assistant")
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                if not isinstance(part, dict) or part.get("type") not in PART_TYPES:
                    raise BadRequest(f"unsupported content part {part if not isinstance(part, dict) else part.get('type')!r}")
                t = part["type"]
                if t == "text" and not isinstance(part.get("text"), str):
                    raise BadRequest("text parts need a string 'text'")
                if t in ("image_url", "audio_url", "video_url") and not (isinstance(part.get(t), dict) and isinstance(part[t].get("url"), str)):
                    raise BadRequest(f"{t} parts need an object {{'url': string}} in '{t}'")
                if t == "input_audio" and not (isinstance(part.get(t), dict) and isinstance(part[t].get("data"), str)):
                    raise BadRequest("input_audio parts need an object {'data': base64 string} in 'input_audio'")
        elif not isinstance(c, str):
            raise BadRequest("message content must be a string or a list of parts")
    mods = body.get("modalities", ["text", "audio"])
    if not isinstance(mods, list) or not mods or any(x not in ("text", "audio") for x in mods):
        raise BadRequest("modalities must be a non-empty subset of ['text', 'audio']")
    ref = body.get("ref_audio")
    if ref is None and isinstance(body.get("audio"), dict):
        ref = body["audio"].get("ref_audio")
    ref_bytes = _decode_audio_ref(ref) if ref else None
    if ref_bytes is not None:
        _check_decodable_audio(ref_bytes)
    if "audio" in mods and ref_bytes is None:
        raise BadRequest("audio output needs a voice prompt (ref_audio)")
    ts = body.get("talker_sampling") or {}
    if not isinstance(ts, dict) or set(ts) - set(TALKER_FIELDS):
        raise BadRequest(f"talker_sampling accepts {sorted(TALKER_FIELDS)}")
    talker = {k: _number(ts, k, int if isinstance(lo, int) and isinstance(hi, int) else float, lo, hi, None) for k, (lo, hi) in TALKER_FIELDS.items()
              if k in ts}
    ctk = body.get("chat_template_kwargs") or {}
    if not isinstance(ctk, dict):
        raise BadRequest("chat_template_kwargs must be an object")
    stream = body.get("stream", False)
    if not isinstance(stream, bool):
        raise BadRequest("stream must be a boolean")
    seed = body.get("seed")
    if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
        raise BadRequest("seed must be an integer")
    pk = body.get("mm_processor_kwargs") or {}
    if not isinstance(pk, dict) or set(pk) - set(PROCESSOR_FIELDS):
        raise BadRequest(f"mm_processor_kwargs accepts {sorted(PROCESSOR_FIELDS)}")
    if pk.get("max_slice_nums") is not None and (isinstance(pk["max_slice_nums"], bool) or not isinstance(pk["max_slice_nums"], int)
                                                 or pk["max_slice_nums"] < 1):
        raise BadRequest("mm_processor_kwargs.max_slice_nums must be a positive integer")
    if pk.get("use_image_id") is not None and not isinstance(pk["use_image_id"], bool):
        raise BadRequest("mm_processor_kwargs.use_image_id must be a boolean")
    return EngineRequest(request_id=f"chatcmpl-{uuid.uuid4().hex}", messages=msgs, modalities=mods,
                         max_tokens=_number(body, "max_tokens", int, 1, MAX_TOKENS_LIMIT, 256),
                         temperature=_number(body, "temperature", float, 0.0, 2.0, 0.0), seed=seed, chat_template_kwargs=ctk,
                         ref_audio=ref_bytes, talker_sampling=talker, stream=stream, processor_kwargs=dict(pk))


def _err(status: int, message: str, kind: str) -> web.Response:
    return web.json_response({"error": {"message": message, "type": kind, "code": status}}, status=status)


class EngineServer:
    def __init__(self, backend: Backend, model_name: str = "openbmb/MiniCPM-o-4_5"):
        self.backend, self.model_name = backend, model_name
        self.app = web.Application(client_max_size=256 << 20)
        self.app.router.add_get("/health", self.health)
        self.app.router.add_get("/v1/models", self.models)
        self.app.router.add_post("/v1/chat/completions", self.chat)
        self.app.router.add_get("/metrics", self.metrics)
        self.app.router.add_get("/info", self.info_route)
        self.cancelled = 0
        self.info: dict = {}  # the served identity (config bytes, engine source), set by main()

    async def health(self, _):
        return web.Response(text="ok")

    async def metrics(self, _):
        """Engine counters for the robustness suite: live requests, KV pages, memory; {} for backends without counters."""
        stats = getattr(self.backend, "stats", None)
        return web.json_response({**(stats() if callable(stats) else {}), "cancelled_streams": self.cancelled, "failed_streams": getattr(self, "failed", 0)})

    async def info_route(self, _):
        return web.json_response(self.info)

    async def models(self, _):
        return web.json_response({"object": "list", "data": [{"id": self.model_name, "object": "model"}]})

    def _chunk(self, req: EngineRequest, created: int, **kw) -> bytes:
        obj = {"id": req.request_id, "object": "chat.completion.chunk", "created": created, "model": self.model_name, **kw}
        return b"data: " + json.dumps(obj).encode() + b"\n\n"

    async def chat(self, request: web.Request):
        try:
            body = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError):
            return _err(400, "request body is not valid JSON", "invalid_request_error")
        try:
            req = parse_request(body)
        except BadRequest as e:
            return _err(400, str(e), "invalid_request_error")
        include_usage = bool((body.get("stream_options") or {}).get("include_usage")) if isinstance(body.get("stream_options"), dict) else False
        events = self.backend.generate(req)
        created = int(time.time())
        if not req.stream:
            return await self._full(req, events, created)
        resp = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
        try:
            first = await events.__anext__()
        except Overloaded as e:
            return _err(503, str(e) or "engine at its admission limit", "server_overloaded")
        except BadRequest as e:  # media or prompt problems found while preparing the request
            return _err(400, str(e), "invalid_request_error")
        except StopAsyncIteration:
            first = Done()
        await resp.prepare(request)
        done, finished, failure = None, False, None
        try:
            ev = first
            while True:
                if isinstance(ev, TextDelta):
                    await resp.write(self._chunk(req, created, modality="text", choices=[{"index": 0, "delta": {"content": ev.text}, "finish_reason": None}]))
                elif isinstance(ev, AudioChunk):
                    data = base64.b64encode(wav_pcm16(ev.samples)).decode()
                    await resp.write(self._chunk(req, created, modality="audio", audio_offset_samples=int(ev.offset), num_samples=int(len(ev.samples)),
                                                 choices=[{"index": 0, "delta": {"content": data}, "finish_reason": None}]))
                elif isinstance(ev, Done):
                    done = ev
                    break
                ev = await events.__anext__()
            finished = True
        except StopAsyncIteration:
            done, finished = done or Done(), True
        except ConnectionResetError:  # the client went away while we wrote (handled below as a cancellation)
            raise
        except Exception as e:  # the engine failed this request after the stream began: an error event, then [DONE]
            failure, finished = e, True
        finally:
            if not finished:  # the client went away (write failed or the handler was cancelled): cancel in the engine
                self.cancelled += 1
                await asyncio.shield(asyncio.ensure_future(events.aclose()))
        if failure is not None:
            self.failed = getattr(self, "failed", 0) + 1
            try:
                kind, code = ("invalid_request_error", 400) if isinstance(failure, BadRequest) else ("server_error", 500)
                await resp.write(f"data: {json.dumps({'error': {'message': str(failure)[:500], 'type': kind, 'code': code}})}\n\n".encode())
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
            except ConnectionResetError:
                pass
            return resp
        await resp.write(self._chunk(req, created, choices=[{"index": 0, "delta": {}, "finish_reason": done.finish_reason}]))
        if include_usage:
            await resp.write(self._chunk(req, created, choices=[], usage=done.usage))
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp

    async def _full(self, req: EngineRequest, events, created: int):
        text, audio, done = [], [], Done()
        try:
            async for ev in events:
                if isinstance(ev, TextDelta):
                    text.append(ev.text)
                elif isinstance(ev, AudioChunk):
                    audio.append(ev.samples)
                elif isinstance(ev, Done):
                    done = ev
        except Overloaded as e:
            return _err(503, str(e) or "engine at its admission limit", "server_overloaded")
        except BadRequest as e:
            return _err(400, str(e), "invalid_request_error")
        msg = {"role": "assistant", "content": "".join(text)}
        if "audio" in req.modalities:
            wav = np.concatenate(audio) if audio else np.zeros(0, dtype=np.float32)
            msg["audio"] = {"data": base64.b64encode(wav_pcm16(wav)).decode(), "format": "wav", "sample_rate": SAMPLE_RATE, "num_samples": int(len(wav))}
        return web.json_response({"id": req.request_id, "object": "chat.completion", "created": created, "model": self.model_name,
                                  "choices": [{"index": 0, "message": msg, "finish_reason": done.finish_reason}], "usage": done.usage})


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="engine.server")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--backend", default=None, help="module:Class (default: the config's \"backend\" key, else the Track T backend)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args(argv)
    spec = a.backend
    if spec is None and a.config:
        spec = json.loads(Path(a.config).read_text()).get("backend")
    mod, _, cls = (spec or "engine.runtime.backend:EngineBackend").partition(":")
    backend = getattr(__import__(mod, fromlist=[cls]), cls).from_config(a.config)
    srv = EngineServer(backend)
    srv.info = served_identity(a.config, spec or "engine.runtime.backend:EngineBackend", backend)
    web.run_app(srv.app, host=a.host, port=a.port, print=None)


def served_identity(config: str | None, backend_spec: str | None = None, backend=None) -> dict:
    """What this process serves, in the harness's terms (``bench.cli.served_identity_expected``): the config path and
    bytes hash, the backend actually instantiated, the engine source tree hash, and where it runs (CUDA device UUID and
    name, interpreter, engine and Token2Wav worker PIDs), computed at startup from what was actually loaded."""
    import hashlib
    import sys

    from bench.fingerprint import tree_sha256

    root = Path(__file__).resolve().parents[1]
    raw = Path(config).read_bytes() if config else None
    device = {}
    try:
        import torch

        if torch.cuda.is_initialized():  # the loaded backend's device (never opens a context of its own on device 0)
            props = torch.cuda.get_device_properties(0)
            device = {"device_uuid": f"GPU-{props.uuid}", "device_name": props.name}
    except Exception:  # noqa: BLE001 - CPU-only test servers report no device
        pass
    t2wp = getattr(backend, "t2wp", None)
    return {"config_path": str(Path(config).resolve()) if config else None, "config_sha256": hashlib.sha256(raw).hexdigest() if raw is not None else None,
            "backend": backend_spec, "engine_tree_sha256": tree_sha256(root, ("engine",)), "python": sys.executable, "pid": os.getpid(),
            "t2w_worker_pid": getattr(getattr(t2wp, "proc", None), "pid", None), **device}


if __name__ == "__main__":
    # ``python -m engine.server`` runs this file as ``__main__``; backends import ``engine.server``, whose event classes
    # would then differ from these (isinstance checks fail and every event is dropped).  Run the imported module.
    from engine.server import main as _main

    raise SystemExit(_main())
