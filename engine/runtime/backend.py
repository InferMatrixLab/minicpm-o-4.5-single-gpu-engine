"""Single-process engine backend for the Track T text+speech path (reference-correct loop).

One engine thread owns the GPU and advances every live request each tick:

1. **Thinker**: new requests are prefilled and live ones decoded in one ragged
   FlashInfer batch (greedy, ``max_tokens``); the final-norm hidden state of
   every fed reply token is kept.  A reply ends at ``<|tts_eos|>``
   (or ``<|im_end|>`` / ``max_tokens``).
2. **Talker** (Track T): once the reply has ended, its condition
   ``emb_text(ids) + normalize(projector_semantic(h))``, ``text_eos``,
   ``audio_bos`` is prefilled; every tick decodes one codec frame per live row
   with the HF-order sampler (per-request counter RNG, 16-frame penalty,
   ``min_new_tokens``/``max_new_tokens``).
3. **Token2Wav**: frames feed each request's stream chunker; every ready
   28-token window is decoded with the request's own flow/HiFT state and sent
   as audio with its sample offset; EOS triggers the final flush.

Requests are cancelled when their event stream is closed (client disconnect);
admission is bounded by ``max_live`` and by free KV pages (``Overloaded``).
This loop is the correctness baseline; graphs, K-step decode, batched
Token2Wav and the scheduler policy are layered on it.
"""

from __future__ import annotations

import asyncio
import hashlib
import queue
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from engine.server import AudioChunk, BadRequest, Done, EngineRequest, Overloaded, TextDelta

MODEL_DIR = Path("/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582")
TTS_EOS, IM_END = 151704, 151645


def _rss_kib() -> int:
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0


def request_seed(req: EngineRequest) -> int:
    if req.seed is not None:
        return int(req.seed) & ((1 << 63) - 1)
    return int(hashlib.sha256(req.request_id.encode()).hexdigest()[:15], 16)


@dataclass
class Live:
    req: EngineRequest
    out: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    prompt_ids: list[int]
    seq: int
    generated: list[int] = field(default_factory=list)
    hidden: list[torch.Tensor] = field(default_factory=list)
    thinker_done: bool = False
    talker_slot: int | None = None
    talker_steps: int = 0
    last_code: int | None = None
    stream: object = None
    chunker: object = None
    sample_offset: int = 0
    cancelled: bool = False
    finished: bool = False
    pending_prefill: bool = True
    prompt_embeds: torch.Tensor | None = None

    def emit(self, ev) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, ev)


class EngineBackend:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "cuda", max_live: int = 64, thinker_pages: int = 8192, talker_pages: int = 8192,
                 page_size: int = 16, float16_flow: bool = False):
        from transformers import AutoTokenizer

        from engine.models.talker import Talker
        from engine.models.thinker import Thinker
        from engine.models.token2wav import Token2Wav
        from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
        from engine.talker_sampler import DecodeState, TalkerSampling

        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.thinker = Thinker.load(model_dir, device=device)
        self.talker = Talker.load(model_dir, device=device)
        c, t = self.thinker.cfg, self.talker.cfg
        self.tspec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
        self.aspec = DecoderSpec(t.num_layers, t.hidden_size, t.num_heads, t.num_kv_heads, t.head_dim, t.rope_theta, t.rms_eps, False)
        self.tkv = PagedKV(self.tspec, thinker_pages, page_size, device=device)
        self.akv = PagedKV(self.aspec, talker_pages, page_size, device=device)
        self.trun = DecoderRunner(self.thinker.w, self.tspec, self.tkv)
        self.arun = DecoderRunner(self.talker.w, self.aspec, self.akv)
        self.t2w = Token2Wav(model_dir / "assets" / "token2wav", float16=float16_flow)
        from engine.models.multimodal import MultimodalFrontend

        self.frontend = MultimodalFrontend(model_dir, device=device)
        self.device, self.max_live = device, max_live
        self._DecodeState, self._TalkerSampling = DecodeState, TalkerSampling
        self.inbox: queue.Queue = queue.Queue()
        self.live: dict[int, Live] = {}
        self._next = 0
        self._lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @classmethod
    def from_config(cls, path: str | None) -> "EngineBackend":
        import json

        cfg = json.loads(Path(path).read_text()) if path else {}
        cfg.pop("backend", None)
        return cls(**cfg)

    # --- request side ----------------------------------------------------------------------------
    def _context_limit(self) -> int:
        """Most tokens (prompt plus reply) one Thinker sequence holds in this engine's graph decoder."""
        dec = getattr(self, "tdec", None)
        return int(getattr(dec, "max_len", 8192))

    def _check_length(self, n_prompt: int, req: EngineRequest) -> None:
        from engine.server import BadRequest

        if n_prompt + req.max_tokens > self._context_limit():
            raise BadRequest(f"prompt of {n_prompt} tokens plus max_tokens {req.max_tokens} exceeds the engine's context of {self._context_limit()}")

    def _prompt(self, req: EngineRequest) -> tuple[list[int], torch.Tensor | None]:
        """Prompt token IDs, and the Thinker input embeddings when the request carries media (None for text-only prompts).

        Media prompts use the request's processor policy (``mm_processor_kwargs``: image slicing, image IDs) and are never
        truncated: a prompt that does not fit the engine's context is rejected (HTTP 400) before inference."""
        kw = {"use_tts_template": True, "enable_thinking": False, **req.chat_template_kwargs}
        if not any(isinstance(m["content"], list) and any(p["type"] != "text" for p in m["content"]) for m in req.messages):
            msgs = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else "".join(p["text"] for p in m["content"])}
                    for m in req.messages]
            ids = self.tok(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))["input_ids"]
            self._check_length(len(ids), req)
            return ids, None
        from engine.media import parts_to_native

        msgs = [{"role": m["role"], "content": m["content"] if isinstance(m["content"], str) else parts_to_native(m["content"])} for m in req.messages]
        pk = getattr(req, "processor_kwargs", None) or {}
        with self._lock:
            inputs = self.frontend.build_prompt(msgs, use_tts_template=kw["use_tts_template"], enable_thinking=kw["enable_thinking"],
                                                max_slice_nums=pk.get("max_slice_nums"), use_image_id=pk.get("use_image_id"), max_inp_length=None)
            self._check_length(int(inputs["input_ids"].shape[1]), req)
            emb = self.frontend.embed(inputs)
        return inputs["input_ids"][0].tolist(), emb

    async def generate(self, req: EngineRequest):
        prompt, embeds = await asyncio.to_thread(self._prompt, req)
        with self._lock:
            if len(self.live) + self.inbox.qsize() >= self.max_live:
                raise Overloaded(f"{self.max_live} live requests")
            seq = self._next
            self._next += 1
        lv = Live(req=req, out=asyncio.Queue(), loop=asyncio.get_running_loop(), prompt_ids=prompt, seq=seq, prompt_embeds=embeds)
        self.inbox.put(lv)
        try:
            while True:
                ev = await lv.out.get()
                if isinstance(ev, Exception):
                    raise ev
                yield ev
                if isinstance(ev, Done):
                    return
        finally:
            lv.cancelled = True  # closed early (disconnect) or finished: the engine thread releases it

    def stats(self) -> dict:
        import resource

        return {"live": len(self.live), "queued": self.inbox.qsize(), "thinker_kv_pages_used": self.tkv.used_pages,
                "talker_kv_pages_used": self.akv.used_pages, "thinker_kv_pages": self.tkv.num_pages, "talker_kv_pages": self.akv.num_pages,
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()), "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
                "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "rss_kib": _rss_kib()}

    # --- engine thread ---------------------------------------------------------------------------
    def _run(self) -> None:
        torch.cuda.set_device(torch.device(self.device).index or 0)  # torch >= 2.13 rejects a bare "cuda"
        while True:
            try:
                while True:
                    lv = self.inbox.get(block=not self.live, timeout=None if not self.live else 0)
                    try:
                        self._admit(lv)
                    except Exception as e:  # a request that cannot be admitted fails alone
                        lv.emit(RuntimeError(f"engine admission error: {type(e).__name__}: {e}"))
            except queue.Empty:
                pass
            try:
                with torch.inference_mode():
                    self._tick()
            except Exception as e:  # a failing tick fails its requests, never the engine thread
                for lv in list(self.live.values()):
                    lv.emit(RuntimeError(f"engine error: {type(e).__name__}: {e}"))
                    self._release(lv)

    def _admit(self, lv: Live) -> None:
        if lv.req.ref_audio is not None and "audio" in lv.req.modalities:
            lv.stream = self.t2w.new_stream(self.t2w.prepare(lv.req.ref_audio))
            from engine.token2wav_chunker import StreamChunker

            lv.chunker = StreamChunker()
        self.live[lv.seq] = lv

    def _release(self, lv: Live) -> None:
        self.tkv.release(lv.seq)
        self.akv.release(lv.seq)
        self.live.pop(lv.seq, None)
        lv.finished = True

    def _tick(self) -> None:
        for lv in [x for x in self.live.values() if x.cancelled and not x.finished]:
            self._release(lv)
        self._thinker_step([x for x in self.live.values() if not x.thinker_done])
        self._talker_step([x for x in self.live.values() if x.thinker_done and x.stream is not None and not x.finished])

    def _thinker_step(self, rows: list[Live]) -> None:
        if not rows:
            return
        feeds = [lv.prompt_ids if lv.pending_prefill else [lv.generated[-1]] for lv in rows]
        lay = self.tkv.layout([lv.seq for lv in rows], [len(f) for f in feeds])
        embs = []
        for lv, f in zip(rows, feeds):
            if lv.pending_prefill and lv.prompt_embeds is not None:
                embs.append(lv.prompt_embeds.to(self.thinker.dtype))  # media states already scattered into the placeholders
            else:
                embs.append(self.thinker.embed(torch.tensor(f, device=self.device)))
        hidden = self.trun.forward(torch.cat(embs), lay)
        last = hidden[lay.last_token_index]
        nxt = torch.nn.functional.linear(last, self.thinker.w["lm_head.weight"]).float().argmax(-1).tolist()
        for lv, h, tok in zip(rows, last, nxt):
            if not lv.pending_prefill:
                lv.hidden.append(h)  # hidden state of the reply token just fed
            lv.pending_prefill = False
            if tok in (TTS_EOS, IM_END) or len(lv.generated) + 1 >= lv.req.max_tokens:
                if tok not in (TTS_EOS, IM_END):
                    lv.generated.append(tok)
                    lv.emit(TextDelta(self.tok.decode([tok])))
                lv.thinker_done = True
                self.tkv.release(lv.seq)
                if lv.stream is None:
                    lv.emit(Done("stop", {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated)}))
                    self._release(lv)
                continue
            lv.generated.append(tok)
            lv.emit(TextDelta(self.tok.decode([tok])))

    def _talker_step(self, rows: list[Live]) -> None:
        if not rows:
            return
        feeds, new = [], []
        for lv in rows:
            if lv.talker_slot is None:
                n = len(lv.hidden)
                ids = torch.tensor(lv.generated[:n], device=self.device)
                cond = self.talker.condition(ids, torch.stack(lv.hidden)) if n else self.talker.condition(ids, torch.zeros(0, 4096, device=self.device))
                feeds.append(cond)
                lv.talker_slot = lv.seq
                ts = dict(lv.req.talker_sampling)
                if "repetition_window" in ts:
                    ts["window"] = ts.pop("repetition_window")
                lv.cfg = self._TalkerSampling(**ts)
                lv.decode = self._DecodeState.new([request_seed(lv.req)], lv.cfg, device=self.device)
                new.append(lv)
            else:
                feeds.append(self.talker.code_embed(torch.tensor([lv.last_code], device=self.device)))
        lay = self.akv.layout([lv.seq for lv in rows], [f.shape[0] for f in feeds])
        hidden = self.arun.forward(torch.cat(feeds), lay)
        logits = torch.nn.functional.linear(hidden[lay.last_token_index], self.talker.head).float()
        from engine.talker_sampler import step

        eos = self.talker.cfg.eos
        for i, lv in enumerate(rows):
            tok = int(step(lv.decode, logits[i:i + 1], eos, lv.cfg)[0])
            if tok == eos:
                w = lv.chunker.finish()
                if w is not None:
                    self._emit_audio(lv, w)
                lv.emit(Done("stop", {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated), "audio_frames": lv.talker_steps}))
                self._release(lv)
                continue
            lv.last_code = tok
            lv.talker_steps += 1
            for w in lv.chunker.push([tok]):
                self._emit_audio(lv, w)

    def _emit_audio(self, lv: Live, w) -> None:
        wav = self.t2w.decode(lv.stream, w.tokens, w.last).astype(np.float32)
        if len(wav):
            lv.emit(AudioChunk(wav, lv.sample_offset))
            lv.sample_offset += len(wav)
