"""vllm-omni ``/v1/chat/completions`` streaming adapter.

Audio events carry ``modality: "audio"`` and a self-contained base64 WAV in
``choices[0].delta.content`` (or ``delta.audio.data``).  The WAV header is
stripped and PCM appended in event order; repeated PCM is a retransmission.
"""

from __future__ import annotations

from bench.adapters.base import Adapter
from bench.audio import WavChunkAssembler, WavFormatError
from bench.metrics import EventRecord, RequestTrace
from bench.workload import CanonicalRequest


class VllmOmniAdapter(Adapter):
    name = "vllm-omni"
    streaming = True

    def __init__(self, model: str = "openbmb/MiniCPM-o-4_5", stage_sampling_override: list[dict] | None = None) -> None:
        super().__init__(model=model)
        #: Optional full per-stage sampling list (``sampling_params_list``); when
        #: None the deploy YAML defaults apply and are recorded in the manifest.
        self.stage_sampling_override = stage_sampling_override

    def build_payload(self, req: CanonicalRequest) -> dict:
        payload = {
            "model": self.model,
            "messages": req.messages(),
            "modalities": ["text", "audio"],
            "stream": req.stream,
            "temperature": req.thinker.temperature,
            "max_tokens": req.thinker.max_tokens,
            "chat_template_kwargs": {
                "enable_thinking": req.enable_thinking,
                "use_tts_template": req.use_tts_template,
            },
            "ref_audio": req.ref_audio_data_url(),
        }
        if req.stream:
            payload["stream_options"] = {"include_usage": True}
        if self.stage_sampling_override is not None:
            payload["sampling_params_list"] = self.stage_sampling_override
        return payload

    def new_assembler(self) -> WavChunkAssembler:
        return WavChunkAssembler()

    def handle_event(self, t: float, obj: dict, trace: RequestTrace, asm: WavChunkAssembler, nbytes: int = 0) -> None:
        if self._record_error(t, obj, trace):
            return
        self._record_usage(obj, trace)
        choices = obj.get("choices") or []
        if not choices:
            trace.events.append(EventRecord(t=t, kind="usage" if "usage" in obj else "other", nbytes=nbytes))
            return
        modality = obj.get("modality")
        delta = choices[0].get("delta") or {}
        content = delta.get("content")
        if not content and isinstance(delta.get("audio"), dict):
            content = delta["audio"].get("data")
        if modality == "audio":
            if not content:
                trace.events.append(EventRecord(t=t, kind="audio", nbytes=nbytes, note="empty"))
                return
            try:
                c = asm.add(t, self._b64(content))
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
        else:
            if isinstance(content, str):
                trace.text += content
            trace.events.append(EventRecord(t=t, kind="text", nbytes=nbytes))

    def handle_body(self, t: float, obj: dict, trace: RequestTrace, asm: WavChunkAssembler) -> None:
        """``stream=false``: choices may hold a text and an audio message."""
        if self._record_error(t, obj, trace):
            return
        self._record_usage(obj, trace)
        trace.terminal = True
        for choice in obj.get("choices") or []:
            msg = choice.get("message") or {}
            audio = msg.get("audio")
            if isinstance(audio, dict) and audio.get("data"):
                c = asm.add(t, self._b64(audio["data"]))
                trace.events.append(EventRecord(t=t, kind="audio", samples=c.received_samples, unique_samples=c.unique_samples))
            elif isinstance(msg.get("content"), str):
                trace.text += msg["content"]
        trace.events.append(EventRecord(t=t, kind="body"))
