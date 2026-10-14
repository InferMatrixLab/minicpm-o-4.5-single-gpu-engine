"""sglang-omni MiniCPM-o adapter.

Turn-mode MiniCPM-o speech in sglang-omni does not stream audio: the single
terminal waveform in ``choices[0].message.audio.data`` is the only audio event,
timestamped at the end of the response body.  Its TTFA is therefore close to
its end-to-end latency, and its results are labelled "non-streaming".

Talker controls: the chat API forwards ``talker_temperature``, ``talker_top_p``,
``talker_top_k``, ``talker_repetition_penalty``, ``talker_max_new_tokens`` and
the top-level ``seed``.  ``min_new_tokens`` (50) and the 16-frame penalty window
are fixed implementation defaults at the pinned commit
(``talker_request.py`` / ``talker_model_runner.py``) and are not request fields.
"""

from __future__ import annotations

from bench.adapters.base import Adapter
from bench.audio import TerminalWaveformAssembler, WavChunkAssembler, WavFormatError
from bench.metrics import EventRecord, RequestTrace
from bench.workload import CanonicalRequest


class SglangOmniAdapter(Adapter):
    name = "sglang-omni"

    def __init__(self, model: str = "minicpmo", stream: bool = False, pass_talker_params: bool = True) -> None:
        super().__init__(model=model)
        self.streaming = stream
        self.pass_talker_params = pass_talker_params

    def build_payload(self, req: CanonicalRequest) -> dict:
        payload = {
            "model": self.model,
            "messages": req.messages(),
            "modalities": ["text", "audio"],
            "audio": {"format": "wav", "ref_audio": req.ref_audio_data_url()},
            "max_tokens": req.thinker.max_tokens,
            "temperature": req.thinker.temperature,
            "stream": self.streaming,
            "seed": req.seed,
        }
        if self.pass_talker_params:
            payload.update(
                {
                    "talker_temperature": req.talker.temperature,
                    "talker_top_p": req.talker.top_p,
                    "talker_top_k": req.talker.top_k,
                    "talker_repetition_penalty": req.talker.repetition_penalty,
                    "talker_max_new_tokens": req.talker.max_new_tokens,
                }
            )
        return payload

    def new_assembler(self):
        return WavChunkAssembler() if self.streaming else TerminalWaveformAssembler()

    def handle_event(self, t: float, obj: dict, trace: RequestTrace, asm, nbytes: int = 0) -> None:
        """Streaming chat SSE: audio in ``delta.audio.data``, text in ``delta.content``."""
        if self._record_error(t, obj, trace):
            return
        self._record_usage(obj, trace)
        handled = False
        for choice in obj.get("choices") or []:
            delta = choice.get("delta") or {}
            audio = delta.get("audio")
            if isinstance(audio, dict) and audio.get("data"):
                try:
                    c = asm.add(t, self._b64(audio["data"]))
                except (WavFormatError, ValueError) as e:
                    asm.out.flags.append("malformed_wav")
                    trace.events.append(EventRecord(t=t, kind="audio", nbytes=nbytes, note=f"malformed:{e}"[:120]))
                    handled = True
                    continue
                trace.events.append(
                    EventRecord(t=t, kind="audio", nbytes=nbytes, samples=c.received_samples, unique_samples=c.unique_samples)
                )
                handled = True
            content = delta.get("content")
            if isinstance(content, str) and content:
                trace.text += content
                trace.events.append(EventRecord(t=t, kind="text", nbytes=nbytes))
                handled = True
        if not handled:
            trace.events.append(EventRecord(t=t, kind="usage" if "usage" in obj else "other", nbytes=nbytes))

    def handle_body(self, t: float, obj: dict, trace: RequestTrace, asm) -> None:
        if self._record_error(t, obj, trace):
            return
        self._record_usage(obj, trace)
        trace.terminal = True
        choices = obj.get("choices") or []
        if not choices:
            trace.error = trace.error or "no choices in response"
            return
        msg = choices[0].get("message") or {}
        if isinstance(msg.get("content"), str):
            trace.text = msg["content"]
        audio = msg.get("audio")
        if isinstance(audio, dict) and audio.get("data"):
            try:
                c = asm.add(t, self._b64(audio["data"]))
            except (WavFormatError, ValueError) as e:
                asm.out.flags.append("malformed_wav")
                trace.error = trace.error or f"malformed audio: {e}"
                return
            trace.events.append(EventRecord(t=t, kind="audio", samples=c.received_samples, unique_samples=c.unique_samples))
        trace.events.append(EventRecord(t=t, kind="body"))
