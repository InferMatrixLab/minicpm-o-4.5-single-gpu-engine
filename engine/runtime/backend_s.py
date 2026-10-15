"""Track S engine backend: Thinker, Talker and Token2Wav advance together, the Talker conditioned chunk by chunk.

Per-request semantics are those of ``engine.runtime.track_s`` (the HF ``streaming_generate`` port, checked against the
reference yield rules), run for many requests at once:

1. **Thinker** (greedy): new prompts are prefilled in one ragged FlashInfer batch; live rows decode one token per tick
   in a CUDA-graph bucket (``ThinkerDecoder``), which also returns the final-norm hidden state of each fed token.  Text
   chunks follow the Track S rule: the first after 11 generated tokens (the 10 with hidden states), then every 10
   (the previous chunk's last token plus 9), and the final one at ``<|tts_eos|>`` / ``<|im_end|>`` / the token limit.
2. **Talker**: a row with a pending text chunk prefills ``emb_text(ids) + normalize(projector(h))`` (+ ``text_eos`` on
   the final chunk) + ``audio_bos`` after everything it has seen, samples the chunk's first frame from the prefill
   logits, then decodes K frames per tick in a CUDA-graph bucket with the HF-order sampler (per-request counter RNG,
   16-frame penalty spanning chunks, frame-0 rule on the utterance's first frame only, no minimum length).  EOS ends
   the chunk: the KV length is rolled back to the last fed frame, and the row waits for the next chunk; EOS of the
   final chunk ends the utterance.
3. **Token2Wav**: frames feed each request's stream chunker (3 silence tokens, 25 + 3 lookahead windows, final
   flush); every tick decodes all ready windows of all requests in one batched call (``BatchedToken2Wav``).

The reference voice is prepared (S3 tokens, speaker embedding, prompt mel, flow caches) in a worker thread while the
Thinker already runs; a request's windows wait for it.  Requests are cancelled when their event stream closes.
"""

from __future__ import annotations

import asyncio
import queue
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from engine.runtime.backend import IM_END, MODEL_DIR, TTS_EOS, _rss_kib, request_seed
from engine.server import AudioChunk, Done, EngineRequest, Overloaded, TextDelta

THINKER_CHUNK = 10
TALKER_MAX_PER_CHUNK = 500


@dataclass
class TextChunk:
    ids: list[int]
    hidden: torch.Tensor  # [n, H]
    final: bool


@dataclass
class LiveS:
    req: EngineRequest
    out: asyncio.Queue
    loop: asyncio.AbstractEventLoop
    prompt_ids: list[int]
    seq: int
    prompt_embeds: torch.Tensor | None = None
    # thinker
    pending_prefill: bool = True
    generated: list[int] = field(default_factory=list)
    hidden: list[torch.Tensor] = field(default_factory=list)
    passed: int = 0
    chunk_index: int = 0
    thinker_done: bool = False
    # talker
    chunks: list[TextChunk] = field(default_factory=list)
    talking: bool = False  # decoding frames of a chunk
    chunk_frames: int = 0
    feed_last: bool = False  # the chunk ended at the per-chunk limit: the last frame still has to be fed
    ts: dict | None = None  # codes, seeds, steps, history, live (device tensors, one row)
    frames: int = 0
    utterance_done: bool = False
    # token2wav
    prompt_future: Future | None = None
    stream: object = None
    chunker: object = None
    windows: list = field(default_factory=list)
    sample_offset: int = 0
    cancelled: bool = False
    finished: bool = False
    timeline: list | None = None

    def emit(self, ev) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, ev)

    def mark(self, event: str) -> None:
        if self.timeline is not None:
            import time

            self.timeline.append((event, time.perf_counter()))

    @property
    def wants_audio(self) -> bool:
        return self.req.ref_audio is not None and "audio" in self.req.modalities


class TrackSBackend:
    def __init__(self, model_dir: Path = MODEL_DIR, device: str = "cuda", max_live: int = 64, thinker_pages: int = 8192, talker_pages: int = 8192,
                 page_size: int = 16, float16_flow: bool = False, talker_k: int = 4, use_graphs: bool = True, prepare_workers: int = 4,
                 profile: bool = False):
        from transformers import AutoTokenizer

        from engine.models.multimodal import MultimodalFrontend
        from engine.models.talker import Talker
        from engine.models.thinker import Thinker
        from engine.models.token2wav import Token2Wav
        from engine.runtime.paged import DecoderRunner, DecoderSpec, PagedKV
        from engine.runtime.talker_decode import TalkerDecoder
        from engine.runtime.thinker_decode import ThinkerDecoder
        from engine.runtime.token2wav_batch import BatchedToken2Wav
        from engine.talker_sampler import TalkerSampling

        torch.backends.cudnn.allow_tf32 = False  # AC-3.4: Token2Wav against FP32 (see engine.models.token2wav.strict_fp32)
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
        # Track S sampling: the canonical warpers, no minimum length, no global maximum (HF streaming path)
        self.cfg = TalkerSampling(min_new_tokens=0, max_new_tokens=1 << 30)
        self.tdec = ThinkerDecoder(self.thinker, self.tkv, self.tspec, use_graphs=use_graphs)
        self.adec = TalkerDecoder(self.talker, self.akv, self.aspec, self.cfg, use_graphs=use_graphs)
        self.tdec.capture_all()
        self.adec.capture_all()
        self.t2w = Token2Wav(model_dir / "assets" / "token2wav", float16=float16_flow)
        self.t2w_batch = BatchedToken2Wav(self.t2w)
        self.frontend = MultimodalFrontend(model_dir, device=device)
        self.device, self.max_live, self.talker_k = device, max_live, talker_k
        self.profile = profile
        self.timelines: dict[str, list[tuple[str, float]]] = {}  # request id -> (event, perf_counter), when profiling
        self.pool = ThreadPoolExecutor(prepare_workers, thread_name_prefix="t2w-prepare")
        self.inbox: queue.Queue = queue.Queue()
        self.live: dict[int, LiveS] = {}
        self._next = 0
        self._lock = threading.Lock()
        self._prepare_lock = threading.Lock()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @classmethod
    def from_config(cls, path: str | None) -> "TrackSBackend":
        import json

        cfg = json.loads(Path(path).read_text()) if path else {}
        cfg.pop("backend", None)
        return cls(**cfg)

    # --- request side ----------------------------------------------------------------------------
    def _prompt(self, req: EngineRequest):
        from engine.runtime.backend import EngineBackend

        return EngineBackend._prompt(self, req)  # same prompt construction (text template or multimodal frontend)

    def _check_sampling(self, req: EngineRequest) -> None:
        from engine.server import BadRequest

        ts = {k: v for k, v in dict(req.talker_sampling).items() if v is not None}
        for k in ("temperature", "top_p", "top_k", "repetition_penalty"):
            if k in ts and float(ts[k]) != float(getattr(self.cfg, k)):
                raise BadRequest(f"talker {k}={ts[k]} differs from the engine's Track S sampling ({getattr(self.cfg, k)})")
        if "repetition_window" in ts and int(ts["repetition_window"]) != self.cfg.window:
            raise BadRequest(f"talker repetition_window={ts['repetition_window']} differs from the engine's ({self.cfg.window})")

    def _prepare(self, ref: bytes):
        with self._prepare_lock, torch.inference_mode():  # the reference prompt path uses the flow's shared scratch buffers
            return self.t2w.prepare(ref)

    async def generate(self, req: EngineRequest):
        self._check_sampling(req)
        with self._lock:
            if len(self.live) + self.inbox.qsize() >= self.max_live:
                raise Overloaded(f"{self.max_live} live requests")
        fut = self.pool.submit(self._prepare, req.ref_audio) if req.ref_audio is not None and "audio" in req.modalities else None
        prompt, embeds = await asyncio.to_thread(self._prompt, req)
        with self._lock:
            seq = self._next
            self._next += 1
        lv = LiveS(req=req, out=asyncio.Queue(), loop=asyncio.get_running_loop(), prompt_ids=prompt, seq=seq, prompt_embeds=embeds,
                   prompt_future=fut)
        if self.profile:
            lv.timeline = self.timelines.setdefault(req.request_id, [])
            lv.mark("queued")
            if fut is not None:
                fut.add_done_callback(lambda _: lv.mark("prompt_ready"))
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
            lv.cancelled = True

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
                import traceback

                traceback.print_exc()
                for lv in list(self.live.values()):
                    lv.emit(RuntimeError(f"engine error: {type(e).__name__}: {e}"))
                    self._release(lv)

    def _admit(self, lv: LiveS) -> None:
        if lv.wants_audio:
            from engine.token2wav_chunker import StreamChunker

            lv.chunker = StreamChunker()
        self.live[lv.seq] = lv

    def _release(self, lv: LiveS) -> None:
        self.tkv.release(lv.seq)
        self.akv.release(lv.seq)
        self.live.pop(lv.seq, None)
        lv.finished = True

    def _finish(self, lv: LiveS) -> None:
        lv.emit(Done("stop", {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated), "audio_frames": lv.frames}))
        self._release(lv)

    def _tick(self) -> None:
        for lv in [x for x in self.live.values() if x.cancelled and not x.finished]:
            self._release(lv)
        self._thinker_step([x for x in self.live.values() if not x.thinker_done])
        self._talker_step([x for x in self.live.values() if x.wants_audio and not x.utterance_done])
        self._token2wav_step()
        for lv in [x for x in list(self.live.values()) if x.utterance_done and not x.windows and not x.finished]:
            self._finish(lv)

    # thinker -----------------------------------------------------------------------------------
    def _thinker_step(self, rows: list[LiveS]) -> None:
        if not rows:
            return
        pre = [lv for lv in rows if lv.pending_prefill]
        dec = [lv for lv in rows if not lv.pending_prefill]
        results = []  # (lv, next token, hidden of the fed token or None)
        if pre:
            lay = self.tkv.layout([lv.seq for lv in pre], [len(lv.prompt_ids) for lv in pre])
            embs = [lv.prompt_embeds.to(self.thinker.dtype) if lv.prompt_embeds is not None
                    else self.thinker.embed(torch.tensor(lv.prompt_ids, device=self.device)) for lv in pre]
            hidden = self.trun.forward(torch.cat(embs), lay)
            nxt = torch.nn.functional.linear(hidden[lay.last_token_index], self.thinker.w["lm_head.weight"]).float().argmax(-1).tolist()
            for lv, tok in zip(pre, nxt):
                lv.pending_prefill = False
                lv.mark("prefilled")
                results.append((lv, tok, None))
        if dec:
            nxt, hid = self.tdec.step([lv.seq for lv in dec], [lv.generated[-1] for lv in dec])
            results += [(lv, tok, hid[i]) for i, (lv, tok) in enumerate(zip(dec, nxt))]
        for lv, tok, h in results:
            if h is not None:
                lv.hidden.append(h)  # hidden state of the reply token just fed
            if tok in (TTS_EOS, IM_END):
                self._close_thinker(lv)
                continue
            lv.generated.append(tok)
            lv.emit(TextDelta(self.tok.decode([tok])))
            n = len(lv.generated)
            boundary = THINKER_CHUNK + 1 + THINKER_CHUNK * lv.chunk_index
            if n == boundary:
                last = lv.chunk_index == -(-lv.req.max_tokens // THINKER_CHUNK) - 1
                self._pass_chunk(lv, final=last)
                lv.chunk_index += 1
                if last:
                    self._close_thinker(lv, passed_final=True)

    def _close_thinker(self, lv: LiveS, passed_final: bool = False) -> None:
        if not passed_final:
            self._pass_chunk(lv, final=True)
        lv.thinker_done = True
        self.tkv.release(lv.seq)
        if not lv.wants_audio:
            self._finish(lv)

    def _pass_chunk(self, lv: LiveS, final: bool) -> None:
        """The tokens fed since the last chunk (those with hidden states), as ``track_s.thinker_chunks`` yields them."""
        ids, hs = lv.generated[lv.passed:len(lv.hidden)], lv.hidden[lv.passed:]
        lv.passed = len(lv.hidden)
        if lv.wants_audio:
            h = torch.stack(hs) if hs else torch.zeros(0, self.tspec.hidden_size, device=self.device, dtype=self.thinker.dtype)
            lv.chunks.append(TextChunk(ids, h, final))
            lv.mark(f"text_chunk{'_final' if final else ''}:{len(ids)}")

    # talker ------------------------------------------------------------------------------------
    def _talker_step(self, rows: list[LiveS]) -> None:
        from engine.talker_sampler import PAD, DecodeState, step

        eos = self.talker.cfg.eos
        starts = [lv for lv in rows if not lv.talking and lv.chunks]
        if starts:
            feeds = []
            for lv in starts:
                ch = lv.chunks[0]
                cond = self.talker.condition(torch.tensor(ch.ids, dtype=torch.long, device=self.device), ch.hidden)
                if not ch.final:
                    cond = torch.cat([cond[:-2], cond[-1:]], dim=0)  # text_eos only on the final chunk; audio_bos always
                if lv.feed_last:  # the previous chunk stopped at the per-chunk limit before its last frame was fed
                    cond = torch.cat([self.talker.code_embed(lv.ts["codes"]), cond], dim=0)
                    lv.feed_last = False
                feeds.append(cond)
            lay = self.akv.layout([lv.seq for lv in starts], [f.shape[0] for f in feeds])
            hidden = self.arun.forward(torch.cat(feeds), lay)
            logits = torch.nn.functional.linear(hidden[lay.last_token_index], self.talker.head).float()
            for i, lv in enumerate(starts):
                if lv.ts is None:
                    seed = request_seed(lv.req)
                    lv.ts = {"codes": torch.zeros(1, dtype=torch.int64, device=self.device), "seeds": torch.tensor([seed], device=self.device),
                             "steps": torch.zeros(1, dtype=torch.int64, device=self.device),
                             "history": torch.full((1, self.cfg.window), PAD, dtype=torch.int64, device=self.device),
                             "live": torch.ones(1, dtype=torch.bool, device=self.device)}
                st = DecodeState(lv.ts["seeds"], lv.ts["steps"], lv.ts["history"], torch.zeros(1, dtype=torch.bool, device=self.device),
                                 torch.ones(1, dtype=torch.bool, device=self.device))
                tok = int(step(st, logits[i:i + 1], eos, self.cfg)[0])
                lv.ts.update(steps=st.steps, history=st.history)
                lv.chunk_frames = 0
                self._take_frame(lv, tok, fed_upto=None)
        active = [lv for lv in rows if lv.talking]
        if not active:
            return
        starts_len = [self.akv.lengths[lv.seq] for lv in active]
        state = {name: torch.cat([lv.ts[name] for lv in active]) for name in ("codes", "seeds", "steps", "history", "live")}
        codes = self.adec.decode_k([lv.seq for lv in active], state, self.talker_k)  # [K, n] on the host
        for i, lv in enumerate(active):
            for name in ("codes", "seeds", "steps", "history", "live"):
                lv.ts[name] = state[name][i:i + 1].clone()
            for j in range(codes.shape[0]):
                tok = int(codes[j, i])
                if tok == PAD:
                    break
                self._take_frame(lv, tok, fed_upto=starts_len[i] + j + 1)
                if not lv.talking:
                    break

    def _take_frame(self, lv: LiveS, tok: int, fed_upto: int | None) -> None:
        """One sampled frame.  ``fed_upto`` is the Talker KV length once the frames before ``tok`` are fed (decode path)."""
        eos = self.talker.cfg.eos
        ch = lv.chunks[0]
        if tok == eos:
            if fed_upto is not None:
                self.akv.truncate(lv.seq, fed_upto)  # later K-block replays fed EOS/PAD; the reference never feeds them
            self._end_chunk(lv)
            return
        lv.frames += 1
        lv.chunk_frames += 1
        for w in lv.chunker.push([tok]):
            lv.windows.append(w)
            lv.mark(f"window_ready:{w.index}")
        lv.ts["codes"] = torch.tensor([tok], dtype=torch.int64, device=self.device)
        if lv.chunk_frames >= TALKER_MAX_PER_CHUNK:
            # the reference feeds this frame and moves on to the next chunk without an EOS
            if fed_upto is not None:
                self.akv.truncate(lv.seq, fed_upto)
            lv.feed_last = True
            self._end_chunk(lv)
            return
        lv.talking = True
        lv.ts["live"] = torch.ones(1, dtype=torch.bool, device=self.device)

    def _end_chunk(self, lv: LiveS) -> None:
        ch = lv.chunks.pop(0)
        lv.talking = False
        if ch.final:
            lv.utterance_done = True
            w = lv.chunker.finish()
            if w is not None:
                lv.windows.append(w)

    # token2wav ---------------------------------------------------------------------------------
    def _token2wav_step(self) -> None:
        items, who = [], []
        for lv in list(self.live.values()):
            if not lv.windows:
                continue
            if lv.stream is None:
                if not lv.prompt_future.done():
                    continue
                try:
                    lv.stream = self.t2w.new_stream(lv.prompt_future.result(), noise_seed=request_seed(lv.req))
                except Exception as e:  # a bad reference voice fails its own request only
                    from engine.server import BadRequest

                    lv.emit(BadRequest(f"reference audio: {type(e).__name__}: {e}"))
                    self._release(lv)
                    continue
            w = lv.windows.pop(0)
            items.append((lv.stream, w.tokens, w.last))
            who.append(lv)
        if not items:
            return
        for lv, wav in zip(who, self.t2w_batch.decode(items)):
            lv.mark(f"window_decoded:{len(items)}")
            wav = wav.astype(np.float32)
            if len(wav):
                lv.emit(AudioChunk(wav, lv.sample_offset))
                lv.sample_offset += len(wav)
