"""Adapter interface."""

from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod

from bench.metrics import EventRecord, RequestTrace
from bench.workload import CanonicalRequest


class Adapter(ABC):
    name: str = "base"
    #: True when the engine streams SSE events; False when one JSON body holds the result.
    streaming: bool = True
    endpoint: str = "/v1/chat/completions"

    def __init__(self, model: str = "minicpmo") -> None:
        self.model = model

    @abstractmethod
    def build_payload(self, req: CanonicalRequest) -> dict:
        """Native JSON body for one canonical request."""

    @abstractmethod
    def new_assembler(self):
        """Per-request audio assembler."""

    def handle_sse_data(self, t: float, data: str, trace: RequestTrace, asm) -> None:
        """Handle one SSE ``data:`` payload received at ``t``."""
        if data.strip() == "[DONE]":
            trace.terminal = True
            trace.events.append(EventRecord(t=t, kind="done"))
            return
        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            trace.events.append(EventRecord(t=t, kind="other", nbytes=len(data), note="non-json"))
            return
        self.handle_event(t, obj, trace, asm, nbytes=len(data))

    def handle_event(self, t: float, obj: dict, trace: RequestTrace, asm, nbytes: int = 0) -> None:
        raise NotImplementedError

    def handle_body(self, t: float, obj: dict, trace: RequestTrace, asm) -> None:
        raise NotImplementedError

    # -- shared helpers -------------------------------------------------
    @staticmethod
    def _record_error(t: float, obj: dict, trace: RequestTrace) -> bool:
        err = obj.get("error")
        if err is None:
            return False
        msg = err.get("message") if isinstance(err, dict) else str(err)
        trace.error = trace.error or str(msg or err)
        trace.events.append(EventRecord(t=t, kind="error", note=str(msg)[:200]))
        return True

    @staticmethod
    def _record_usage(obj: dict, trace: RequestTrace) -> None:
        usage = obj.get("usage")
        if isinstance(usage, dict):
            trace.usage = usage

    @staticmethod
    def _b64(data: str) -> bytes:
        if data.startswith("data:"):
            data = data.split(",", 1)[1]
        return base64.b64decode(data)
