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


def stage_sampling(req: CanonicalRequest) -> list[dict]:
    """Full per-stage ``sampling_params_list`` for one canonical request.

    vllm-omni builds each supplied stage dict into ``SamplingParams`` as is
    (no merge with deploy defaults), and top-level OpenAI fields only reach the
    Thinker when no list is supplied.  So every stage is spelled out: the
    non-canonical fields mirror ``vllm_omni/deploy/minicpmo_4_5.yaml`` (which
    every candidate config inherits), and the canonical fields come from the
    request.  The Talker's ``repetition_penalty`` is consumed by the Talker as
    the multiplicative 16-frame window penalty, not vLLM's sampler penalty.
    """
    t = req.talker
    return [
        {
            "temperature": req.thinker.temperature,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": req.thinker.max_tokens,
            "seed": 42,
            "detokenize": True,
            "repetition_penalty": 1.0,
        },
        {
            "temperature": t.temperature,
            "top_p": t.top_p,
            "top_k": t.top_k,
            "repetition_penalty": t.repetition_penalty,
            "min_tokens": t.min_new_tokens,
            "max_tokens": t.max_new_tokens,
            "seed": req.seed,
            "detokenize": False,
        },
        {"temperature": 0.0, "top_p": 1.0, "top_k": -1, "max_tokens": 65536, "detokenize": True},
    ]


class VllmOmniAdapter(Adapter):
    name = "vllm-omni"
    streaming = True

    def __init__(self, model: str = "openbmb/MiniCPM-o-4_5", control_sampling: bool = True) -> None:
        super().__init__(model=model)
        #: Send a full per-request ``sampling_params_list`` so the canonical
        #: Talker sampling and per-request seed take effect.
        self.control_sampling = control_sampling

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
        if self.control_sampling:
            payload["sampling_params_list"] = stage_sampling(req)
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
