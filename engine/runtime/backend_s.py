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
    codes: list[int] = field(default_factory=list)  # every non-EOS codec frame of the utterance
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
                 t2w_graphs: bool = False, t2w_compile: bool = False, spk_threads: int = 8, t2w_min_batch: int = 8, t2w_max_wait_ms: float = 40.0,
                 switch_interval_s: float = 0.0002, first_chunk: int = 25, t2w_process: bool = False, feature_workers: int = 4,
                 t2w_profile_path: str | None = None, thinker_page_size: int | None = None, talker_page_size: int | None = None,
                 feature_processes: int = 0, thinker_fp8: bool = False, talker_k_max: int = 4, prefill_graphs: bool = False,
                 t2w_euler_graphs: bool = False,
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
        import sys

        # two launch-bound host threads (Thinker/Talker and Token2Wav) share the GIL; hand it over often
        sys.setswitchinterval(switch_interval_s)
        self.tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.thinker = Thinker.load(model_dir, device=device)
        self.talker = Talker.load(model_dir, device=device)
        c, t = self.thinker.cfg, self.talker.cfg
        self.tspec = DecoderSpec(c.num_layers, c.hidden_size, c.num_heads, c.num_kv_heads, c.head_dim, c.rope_theta, c.rms_eps, True)
        self.aspec = DecoderSpec(t.num_layers, t.hidden_size, t.num_heads, t.num_kv_heads, t.head_dim, t.rope_theta, t.rms_eps, False)
        # larger pages mean fewer FlashInfer re-plans in the graph decoders (a plan is redone when a row starts a page)
        self.tkv = PagedKV(self.tspec, thinker_pages, thinker_page_size or page_size, device=device)
        self.akv = PagedKV(self.aspec, talker_pages, talker_page_size or page_size, device=device)
        from engine.runtime.fp8 import quantize_layers
        from engine.runtime.paged import fuse_weights

        tlayers = fuse_weights(self.thinker.w, self.tspec)  # one fused copy shared by prefill and graph decode
        if thinker_fp8:  # approved approximation, qualified by AC-3.7
            tlayers = quantize_layers(tlayers)
        # the per-projection layer weights are only read by the reference forward, which the engine never runs
        for k in [k for k in self.thinker.w if ".layers." in k and ("_proj." in k)]:
            del self.thinker.w[k]
        torch.cuda.empty_cache()
        self.trun = DecoderRunner(self.thinker.w, self.tspec, self.tkv, layers=tlayers)
        self.arun = DecoderRunner(self.talker.w, self.aspec, self.akv)
        # Track S sampling: the canonical warpers, no minimum length, no global maximum (HF streaming path)
        self.cfg = TalkerSampling(min_new_tokens=0, max_new_tokens=1 << 30)
        self.tdec = ThinkerDecoder(self.thinker, self.tkv, self.tspec, use_graphs=use_graphs, layers=tlayers)
        self.adec = TalkerDecoder(self.talker, self.akv, self.aspec, self.cfg, use_graphs=use_graphs)
        self.tprefill = None
        if prefill_graphs and use_graphs:
            from engine.runtime.prefill_graph import PrefillGraphs

            self.tprefill = PrefillGraphs(self.trun)
            self.tprefill.capture_all()
        self.tdec.capture_all()
        self.adec.capture_all()
        self.t2wp = None
        self.audio_live: dict[int, LiveS] = {}  # process mode: requests whose audio the worker still owes
        if t2w_process:  # Token2Wav (features, setups, windows) in its own process: no GIL shared with the engine loop
            from engine.runtime.t2w_worker import Token2WavProcess

            self.t2w = self.t2w_batch = None
            self.t2wp = Token2WavProcess({"model_dir": str(model_dir), "float16_flow": float16_flow, "t2w_graphs": t2w_graphs,
                                          "t2w_compile": t2w_compile, "spk_threads": spk_threads, "t2w_min_batch": t2w_min_batch,
                                          "t2w_max_wait_ms": t2w_max_wait_ms, "feature_workers": feature_workers,
                                          "profile_path": t2w_profile_path, "feature_processes": feature_processes,
                                          "t2w_euler_graphs": t2w_euler_graphs}, self._on_t2w)
        else:
            self.t2w = Token2Wav(model_dir / "assets" / "token2wav", float16=float16_flow, spk_threads=spk_threads)
            self.t2w_batch = BatchedToken2Wav(self.t2w, use_graphs=t2w_graphs, compile_estimator=t2w_compile, euler_graphs=t2w_euler_graphs)
            if t2w_graphs:  # a prepared default voice supplies the cache dtypes
                self.t2w_batch.capture_graphs(self.t2w_batch.setup(self.t2w.features(model_dir / "assets" / "HT_ref_audio.wav")))
        self.frontend = MultimodalFrontend(model_dir, device=device)
        self.device, self.max_live, self.talker_k, self.talker_k_max = device, max_live, talker_k, talker_k_max
        self.profile = profile
        self.timelines: dict[str, list[tuple[str, float]]] = {}  # request id -> (event, perf_counter), when profiling
        self.tick_profile: dict[str, list] = {}  # stage -> [calls, seconds], when profiling
        self.pool = ThreadPoolExecutor(prepare_workers, thread_name_prefix="t2w-prepare")
        self.inbox: queue.Queue = queue.Queue()
        self.live: dict[int, LiveS] = {}
        self._next = 0
        self._lock = threading.Lock()
        self.t2w_min_batch, self.t2w_max_wait = t2w_min_batch, t2w_max_wait_ms / 1000
        self.first_chunk = first_chunk  # < 25: the first-chunk ramp (engine.token2wav_chunker.StreamChunker)
        self.t2w_inbox: queue.Queue = queue.Queue()
        self.t2w_live: dict[int, LiveS] = {}  # owned by the Token2Wav thread
        self.t2w_wake = threading.Event()
        self.t2w_stream = torch.cuda.Stream(device=device)
        self.thread = threading.Thread(target=self._run, daemon=True, name="engine")
        self.thread.start()
        if self.t2wp is None:
            self.t2w_thread = threading.Thread(target=self._run_t2w, daemon=True, name="token2wav")
            self.t2w_thread.start()

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
        """Worker thread: the reference voice's features (no flow work; the Token2Wav thread runs the flow setup)."""
        feats = self.t2w.features(ref)
        torch.cuda.current_stream().synchronize()  # the Token2Wav thread reads them on its own stream
        self.t2w_wake.set()
        return feats

    async def generate(self, req: EngineRequest):
        self._check_sampling(req)
        with self._lock:
            if len(self.live) + self.inbox.qsize() >= self.max_live:
                raise Overloaded(f"{self.max_live} live requests")
        wants = req.ref_audio is not None and "audio" in req.modalities
        fut = self.pool.submit(self._prepare, req.ref_audio) if wants and self.t2wp is None else None
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

            lv.chunker = StreamChunker(first_chunk=self.first_chunk)
            if self.t2wp is not None:
                self.audio_live[lv.seq] = lv
                self.t2wp.send("open", lv.seq, lv.req.ref_audio, request_seed(lv.req))
            else:
                self.t2w_inbox.put(lv)
                self.t2w_wake.set()
        self.live[lv.seq] = lv

    def _release(self, lv: LiveS) -> None:
        self.tkv.release(lv.seq)
        self.akv.release(lv.seq)
        self.live.pop(lv.seq, None)
        lv.finished = True
        if self.t2wp is not None and self.audio_live.pop(lv.seq, None) is not None:
            self.t2wp.send("close", lv.seq)

    def _on_t2w(self, m) -> None:
        """Receiver thread (process mode): the worker's audio, completion and errors for each request."""
        kind, seq = m[0], m[1] if len(m) > 1 else None
        lv = self.audio_live.get(seq)
        if lv is None:
            return
        if lv.cancelled and kind != "done":
            self.audio_live.pop(seq, None)
            self.t2wp.send("close", seq)
            return
        if kind == "audio":
            if lv.sample_offset == 0:
                lv.mark("first_audio")
            lv.emit(AudioChunk(m[3], m[2]))
            lv.sample_offset = m[2] + len(m[3])
        elif kind == "done":
            self.audio_live.pop(seq, None)
            lv.emit(Done("stop", {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated), "audio_frames": lv.frames}))
            lv.finished = True
        elif kind == "error":
            self.audio_live.pop(seq, None)
            lv.emit(RuntimeError(m[2]))

    def _queue_window(self, lv: LiveS, w) -> None:
        lv.mark(f"window_ready:{w.index}")
        if self.t2wp is not None:
            self.t2wp.send("window", lv.seq, w.tokens, w.last, w.index)
            return
        w.t_ready = _now()
        lv.windows.append(w)
        self.t2w_wake.set()

    def _finish(self, lv: LiveS) -> None:
        lv.emit(Done("stop", {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated), "audio_frames": lv.frames}))
        self._release(lv)

    def _timed(self, name: str, fn, *a) -> None:
        if not self.profile:
            fn(*a)
            return
        import time

        torch.cuda.current_stream().synchronize()
        t = time.perf_counter()
        fn(*a)
        torch.cuda.current_stream().synchronize()
        acc = self.tick_profile.setdefault(name, [0, 0.0])
        acc[0] += 1
        acc[1] += time.perf_counter() - t

    def _tick(self) -> None:
        for lv in [x for x in self.live.values() if x.cancelled and not x.finished]:
            self._release(lv)
        self._timed("thinker", self._thinker_step, [x for x in self.live.values() if not x.thinker_done])
        self._timed("talker", self._talker_step, [x for x in self.live.values() if x.wants_audio and not x.utterance_done])
        for lv in [x for x in list(self.live.values()) if x.utterance_done]:
            # the Thinker and Talker are done: free their KV; the Token2Wav thread emits the rest and Done
            self.tkv.release(lv.seq)
            self.akv.release(lv.seq)
            self.live.pop(lv.seq, None)
            if self.t2wp is None:
                self.t2w_wake.set()

    # thinker -----------------------------------------------------------------------------------
    def _thinker_step(self, rows: list[LiveS]) -> None:
        if not rows:
            return
        pre = [lv for lv in rows if lv.pending_prefill]
        dec = [lv for lv in rows if not lv.pending_prefill]
        results = []  # (lv, next token, hidden of the fed token or None)
        if pre:
            embs = [lv.prompt_embeds.to(self.thinker.dtype) if lv.prompt_embeds is not None
                    else self.thinker.embed(torch.tensor(lv.prompt_ids, device=self.device)) for lv in pre]
            if len(pre) == 1 and self.tprefill is not None and self.tprefill.supports(embs[0].shape[0]):
                last = self.tprefill.forward(embs[0], pre[0].seq)[-1:]  # graph-captured single-request prefill
            else:
                lay = self.tkv.layout([lv.seq for lv in pre], [len(lv.prompt_ids) for lv in pre])
                last = self.trun.forward(torch.cat(embs), lay)[lay.last_token_index]
            nxt = torch.nn.functional.linear(last, self.thinker.w["lm_head.weight"]).float().argmax(-1).tolist()
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
        # adaptive K: short blocks while work is waiting to join (a chunk start, a Thinker prefill) or any Thinker is
        # still generating (one text token per tick must stay ahead of ~4 codec frames per token); long blocks once all
        # text is done (fewer ticks per frame).  A row that hits EOS inside a block is rolled back exactly either way.
        waiting = any((not lv.talking and lv.chunks) or not lv.thinker_done for lv in self.live.values()) or not self.inbox.empty()
        k = self.talker_k if waiting else max(self.talker_k, self.talker_k_max)
        codes = self.adec.decode_k([lv.seq for lv in active], state, k)  # [K, n] on the host
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
            # the reference never adds a chunk's EOS to its token history: the penalty window holds real frames only
            from engine.runtime.track_s import history_from

            lv.ts["history"] = history_from(lv.codes, self.cfg.window, self.device)
            self._end_chunk(lv)
            return
        lv.frames += 1
        lv.codes.append(tok)
        lv.chunk_frames += 1
        for w in lv.chunker.push([tok]):
            self._queue_window(lv, w)
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
            w = lv.chunker.finish()
            if w is not None:
                self._queue_window(lv, w)
            lv.utterance_done = True  # after the final window is queued (the Token2Wav thread reads both)

    # token2wav thread ---------------------------------------------------------------------------
    def _run_t2w(self) -> None:
        """All flow work (prompt setup, window decodes) on its own thread and CUDA stream, overlapping the Thinker/Talker.

        Batching: first windows are decoded at once (TTFA); continuation windows wait until ``t2w_min_batch`` of them
        (or every live stream's) are pending, or the oldest has waited ``t2w_max_wait_ms``.
        """
        torch.cuda.set_device(torch.device(self.device).index or 0)
        while True:
            try:
                while True:
                    lv = self.t2w_inbox.get_nowait()
                    self.t2w_live[lv.seq] = lv
            except queue.Empty:
                pass
            try:
                with torch.cuda.stream(self.t2w_stream), torch.inference_mode():
                    worked = self._t2w_tick()
            except Exception as e:
                import traceback

                traceback.print_exc()
                for lv in list(self.t2w_live.values()):
                    lv.emit(RuntimeError(f"token2wav error: {type(e).__name__}: {e}"))
                    lv.cancelled = True
                self.t2w_live.clear()
                worked = False
            if not worked:
                self.t2w_wake.wait(0.002)
                self.t2w_wake.clear()

    def _t2w_tick(self) -> bool:
        worked = self._setups()
        for lv in list(self.t2w_live.values()):
            if lv.cancelled:
                self.t2w_live.pop(lv.seq, None)
        ready = [lv for lv in self.t2w_live.values() if lv.stream is not None and lv.windows]
        if ready:
            now = _now()
            first = any(lv.windows[0].index == 0 for lv in ready)
            oldest = min(lv.windows[0].t_ready for lv in ready)
            enough = len(ready) >= min(self.t2w_min_batch, len(self.t2w_live))
            if first or enough or now - oldest >= self.t2w_max_wait or any(lv.windows[0].last for lv in ready):
                self._timed("token2wav", self._decode_windows, ready)
                worked = True
        for lv in list(self.t2w_live.values()):
            if lv.utterance_done and not lv.windows and lv.seq not in self.live:
                lv.emit(Done("stop", {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated), "audio_frames": lv.frames}))
                lv.finished = True
                self.t2w_live.pop(lv.seq, None)
        return worked

    def _setups(self) -> bool:
        """Prompt setups of every request whose voice features are ready, batched per frame bucket (row limits)."""
        from engine.runtime.dit_graph import ROW_LIMIT, SETUP_T

        todo = []
        for lv in list(self.t2w_live.values()):
            if lv.cancelled or lv.stream is not None or lv.prompt_future is None or not lv.prompt_future.done():
                continue
            try:
                todo.append((lv, lv.prompt_future.result()))
            except Exception as e:  # a bad reference voice fails its own request only
                from engine.server import BadRequest

                lv.emit(BadRequest(f"reference audio: {type(e).__name__}: {e}"))
                lv.cancelled = True
                self.t2w_live.pop(lv.seq, None)
        if not todo:
            return False
        groups: dict[int, list] = {}
        for lv, feats in todo:
            frames = feats[3].shape[1]
            groups.setdefault(next((b for b in SETUP_T if b >= frames), -frames), []).append((lv, feats))
        for b, items in groups.items():
            per = max(1, ROW_LIMIT.get(b, 2) // 2)
            for k in range(0, len(items), per):
                part = items[k:k + per]
                try:
                    prompts = self._timed_ret("setup", self.t2w_batch.setup_many, [f for _, f in part])
                except Exception as e:
                    for lv, _ in part:
                        lv.emit(RuntimeError(f"prompt setup: {type(e).__name__}: {e}"))
                        lv.cancelled = True
                        self.t2w_live.pop(lv.seq, None)
                    continue
                for (lv, _), p in zip(part, prompts):
                    lv.stream = self.t2w.new_stream(p, noise_seed=request_seed(lv.req))
                    lv.mark(f"prompt_setup:{len(part)}")
        return True

    def _timed_ret(self, name: str, fn, *a):
        out = []
        self._timed(name, lambda *x: out.append(fn(*x)), *a)
        return out[0]

    def _decode_windows(self, ready: list[LiveS]) -> None:
        items = []
        for lv in ready:
            w = lv.windows.pop(0)
            items.append((lv.stream, w.tokens, w.last))
        for lv, wav in zip(ready, self.t2w_batch.decode(items)):
            lv.mark(f"window_decoded:{len(items)}")
            wav = wav.astype(np.float32)
            if len(wav):
                lv.emit(AudioChunk(wav, lv.sample_offset))
                lv.sample_offset += len(wav)


def _now() -> float:
    import time

    return time.perf_counter()
