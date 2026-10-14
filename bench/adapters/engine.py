"""Adapter for the new engine's ``/v1/chat/completions`` API.

Audio deltas carry ``modality: "audio"``, a base64 WAV chunk and explicit
``audio_offset_samples`` / ``num_samples`` so contiguity can be checked.
"""

from __future__ import annotations

from bench.adapters.base import Adapter
from bench.audio import OffsetAssembler, WavFormatError
from bench.metrics import EventRecord, RequestTrace
from bench.workload import CanonicalRequest


class EngineAdapter(Adapter):
    name = "engine"
    streaming = True

    def __init__(self, model: str = "openbmb/MiniCPM-o-4_5") -> None:
        super().__init__(model=model)

    def build_payload(self, req: CanonicalRequest) -> dict:
        payload = {
            "model": self.model,
            "messages": req.messages(),
            "modalities": ["text", "audio"],
            "stream": req.stream,
            "temperature": req.thinker.temperature,
            "max_tokens": req.thinker.max_tokens,
            "seed": req.seed,
            "chat_template_kwargs": {
                "enable_thinking": req.enable_thinking,
                "use_tts_template": req.use_tts_template,
            },
            "ref_audio": req.ref_audio_data_url(),
            "talker_sampling": {
                "temperature": req.talker.temperature,
                "top_p": req.talker.top_p,
                "top_k": req.talker.top_k,
                "repetition_penalty": req.talker.repetition_penalty,
                "repetition_window": req.talker.repetition_window,
                "min_new_tokens": req.talker.min_new_tokens,
                "max_new_tokens": req.talker.max_new_tokens,
            },
        }
        if req.stream:
            payload["stream_options"] = {"include_usage": True}
        return payload

    def new_assembler(self) -> OffsetAssembler:
        return OffsetAssembler()

    def _add_audio(self, t: float, audio_b64: str, offset, num, trace: RequestTrace, asm: OffsetAssembler, nbytes: int) -> None:
        if offset is None or num is None:
            asm.out.flags.append("missing_offsets")
            trace.events.append(EventRecord(t=t, kind="audio", nbytes=nbytes, note="missing_offsets"))
            return
        try:
            c = asm.add(t, self._b64(audio_b64), int(offset), int(num))
        except (WavFormatError, ValueError) as e:
            asm.out.flags.append("malformed_wav")
            trace.events.append(EventRecord(t=t, kind="audio", nbytes=nbytes, note=f"malformed:{e}"[:120]))
            return
        trace.events.append(
            EventRecord(
                t=t,
                kind="audio",
                nbytes=nbytes,
                samples=c.received_samples,
                unique_samples=c.unique_samples,
                note="retransmission" if c.retransmission else "",
            )
        )

    def handle_event(self, t: float, obj: dict, trace: RequestTrace, asm: OffsetAssembler, nbytes: int = 0) -> None:
        if self._record_error(t, obj, trace):
            return
        self._record_usage(obj, trace)
        choices = obj.get("choices") or []
        if not choices:
            trace.events.append(EventRecord(t=t, kind="usage" if "usage" in obj else "other", nbytes=nbytes))
            return
        delta = choices[0].get("delta") or {}
        if obj.get("modality") == "audio":
            content = delta.get("content")
            if not content:
                trace.events.append(EventRecord(t=t, kind="audio", nbytes=nbytes, note="empty"))
                return
            self._add_audio(t, content, obj.get("audio_offset_samples"), obj.get("num_samples"), trace, asm, nbytes)
        else:
            content = delta.get("content")
            if isinstance(content, str):
                trace.text += content
            trace.events.append(EventRecord(t=t, kind="text", nbytes=nbytes))

    def handle_body(self, t: float, obj: dict, trace: RequestTrace, asm: OffsetAssembler) -> None:
        if self._record_error(t, obj, trace):
            return
        self._record_usage(obj, trace)
        trace.terminal = True
        choices = obj.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            if isinstance(msg.get("content"), str):
                trace.text = msg["content"]
            audio = msg.get("audio")
            if isinstance(audio, dict) and audio.get("data"):
                n = audio.get("num_samples")
                self._add_audio(t, audio["data"], 0, n, trace, asm, 0)
        trace.events.append(EventRecord(t=t, kind="body"))
