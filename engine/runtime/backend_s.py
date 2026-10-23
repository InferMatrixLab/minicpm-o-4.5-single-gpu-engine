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
from engine.runtime.hostcopy import to_device
from engine.runtime.stage_trace import span
from engine.server import AudioChunk, Done, EngineRequest, Overloaded, TextDelta

THINKER_CHUNK = 10
TALKER_MAX_PER_CHUNK = 500
TALKER_FRAMES_DEFAULT = 2048  # a request without talker max_new_tokens: the canonical value (bench.workload)


def thinker_bound(max_tokens: int) -> int:
    """Reply tokens the Thinker may generate: Track S closes at the first chunk boundary (11 + 10k tokens) at or after
    ``max_tokens``, so up to 10 * ceil(max_tokens / 10) + 1."""
    return THINKER_CHUNK * -(-max_tokens // THINKER_CHUNK) + 1


def talker_lifetime(max_tokens: int, frames: int, k_max: int) -> int:
    """Upper bound on one utterance's Talker KV tokens: every text chunk's condition (its reply tokens, audio_bos,
    text_eos on the final chunk, the carried last frame of a chunk cut at the per-chunk limit), every fed frame (at
    most ``frames``, the request's cap) and the K-frame block a decode reserves before an EOS rolls it back."""
    chunks = -(-max_tokens // THINKER_CHUNK)
    return thinker_bound(max_tokens) + 2 * chunks + 1 + frames + k_max


@dataclass
class TextChunk:
    ids: list[int]
    hidden: torch.Tensor  # [n, H]
    final: bool
    ready: object = None  # CUDA event recorded on the Thinker stream after ``hidden`` (split-thread mode)


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
    prefilled: int = 0  # prompt tokens already in the Thinker KV (a long prompt is prefilled in budget-sized chunks)
    prefill_now: int = 0  # prompt tokens the lane grants this tick
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
    frame_cap: int = TALKER_FRAMES_DEFAULT  # the request's talker max_new_tokens: the utterance ends at this many frames
    codes: list[int] = field(default_factory=list)  # every non-EOS codec frame of the utterance
    utterance_done: bool = False
    finish_reason: str = "stop"  # "length" once the frame cap ended the utterance
    # token2wav
    prompt_future: Future | None = None
    stream: object = None
    chunker: object = None
    windows: list = field(default_factory=list)
    sample_offset: int = 0
    cancelled: bool = False
    finished: bool = False
    timeline: list | None = None
    t_admit: float = 0.0  # when the engine loop admitted it (the prefill lane ages from here)
    audio: object = None  # t2w_sched.AudioState: the playback clock of the audio handed to the client

    def emit(self, ev) -> None:
        self.loop.call_soon_threadsafe(self.out.put_nowait, ev)

    def mark(self, event: str, t: float | None = None) -> None:
        """A timeline event at ``t`` (the shared host clock; now when not given)."""
        if self.timeline is not None:
            import time

            self.timeline.append((event, time.perf_counter() if t is None else t))

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
                 t2w_euler_graphs: bool = False, split_threads: bool = False, t2w_fast_dit: bool = False, t2w_fast_slots: int = 49152,
                 t2w_mps_sm_percent: int = 0, thinker_lazy: bool = False, profile: bool = False, t2w_policy: str = "edf",
                 t2w_setup_calls_when_busy: int = 1, t2w_max_decode_rows: int | None = None, t2w_first_slack_ms: float = 250.0,
                 t2w_first_lane_slack_ms: float | None = None, prefill_token_budget: int | None = 2048, prefill_max_age_ms: float = 250.0,
                 prefill_slack_floor_ms: float | None = 300.0, talker_start_budget: int | None = 1024, feature_gpu_tokens: bool = True,
                 feature_allocator: str = "jemalloc", stage_trace_dir: str | None = None, t2w_hift_decode_tf32: bool = False,
                 t2w_batch_setup_encoder: bool = False, talker_prefill_graphs: bool = False, t2w_async_results: bool = False,
                 t2w_hift_rows: list | None = None, talker_whole_k: bool = False, tick_overlap: bool = False,
                 tick_overlap_first_chunks: bool = True, tick_overlap_max_live: int | None = None, thinker_max_len: int = 32768,
                 t2w_merge_solve: bool = False, audio_resampler: str = "librosa", t2w_fuse_setup_frames: int = 0,
                 talker_lean_frame: bool = False, t2w_hold_slack_ms: float | None = None, t2w_hold_max_ms: float = 100.0,
                 t2w_hold_rows: int = 16, t2w_hold_setups: bool = False, t2w_hift_lean: bool = False,
                 t2w_dit_fused_residual: bool = False, thinker_chunk_prefill_graphs: bool = False,
                 prefill_hf_kernels: bool = False, t2w_dit_fused_conv: bool = False, t2w_hift_fused_bias: bool = False,
                 t2w_setup_prefetch: bool = False, prefill_rope_hf: bool = False, mps_client_priority: int | None = None,
                 t2w_mps_client_priority: int | None = None, t2w_encoder_batched_io: bool = False):
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

        if mps_client_priority is not None:  # under MPS: this process's kernels at normal (0) or below-normal (1) priority
            import os

            assert mps_client_priority in (0, 1) and not torch.cuda.is_initialized(), "set before the engine's CUDA context"
            os.environ["CUDA_MPS_CLIENT_PRIORITY"] = str(mps_client_priority)

        from engine.runtime.t2w_sched import PrefillPolicy

        # scheduler knobs (task17), validated before any model is loaded
        if feature_allocator not in ("jemalloc", "glibc"):
            raise ValueError(f"feature_allocator must be 'jemalloc' or 'glibc', not {feature_allocator!r}")
        from engine.runtime.t2w_worker import MODES

        if t2w_policy not in MODES:
            raise ValueError(f"t2w_policy must be one of {MODES}, not {t2w_policy!r}")
        if int(t2w_setup_calls_when_busy) < 1 or (t2w_max_decode_rows is not None and int(t2w_max_decode_rows) < 1) or t2w_first_slack_ms < 0 \
                or (t2w_first_lane_slack_ms is not None and t2w_first_lane_slack_ms < 0):
            raise ValueError("t2w_setup_calls_when_busy and t2w_max_decode_rows must be >= 1, t2w_first_slack_ms and t2w_first_lane_slack_ms >= 0")
        if t2w_fuse_setup_frames and (not isinstance(t2w_fuse_setup_frames, int) or t2w_fuse_setup_frames < 0 or not t2w_fast_dit
                                      or not t2w_process or t2w_policy != "edf"):
            raise ValueError("t2w_fuse_setup_frames must be an integer >= 0 and needs t2w_fast_dit, t2w_process and t2w_policy edf")
        if (t2w_hold_slack_ms is not None and (t2w_hold_slack_ms < 0 or t2w_policy != "edf" or not t2w_process)) or t2w_hold_max_ms < 0 \
                or int(t2w_hold_rows) < 1 or (t2w_hold_setups and not t2w_fuse_setup_frames):
            raise ValueError("t2w_hold_slack_ms >= 0 needs t2w_policy edf and t2w_process; t2w_hold_max_ms >= 0, t2w_hold_rows >= 1; "
                             "t2w_hold_setups needs t2w_fuse_setup_frames")
        if (prefill_token_budget is not None and int(prefill_token_budget) < 1) or prefill_max_age_ms < 0 or \
                (prefill_slack_floor_ms is not None and prefill_slack_floor_ms < 0) or (talker_start_budget is not None and int(talker_start_budget) < 1):
            raise ValueError("prefill_token_budget and talker_start_budget must be >= 1 (or null), prefill_max_age_ms and prefill_slack_floor_ms >= 0")
        from engine.media import RESAMPLERS

        if audio_resampler not in RESAMPLERS:
            raise ValueError(f"audio_resampler must be one of {RESAMPLERS}, not {audio_resampler!r}")
        self.audio_resampler = audio_resampler  # 16 kHz resampling of understanding inputs (engine.media)
        pool = thinker_pages * (thinker_page_size or page_size)
        if not 0 < thinker_max_len <= pool:
            raise ValueError(f"thinker_max_len={thinker_max_len} must be in 1..{pool} (the Thinker KV pool)")
        self.prefill_policy = PrefillPolicy(token_budget=prefill_token_budget, max_age_s=prefill_max_age_ms / 1000,
                                            slack_floor_s=None if prefill_slack_floor_ms is None else prefill_slack_floor_ms / 1000,
                                            talker_start_budget=talker_start_budget)
        self.sched = {"prefill_token_budget": prefill_token_budget, "prefill_tokens_tick_max": 0, "prefill_chunked_ticks": 0,
                      "prefill_kv_wait_ticks": 0, "talker_starts_deferred_ticks": 0, "talker_kv_wait_ticks": 0}  # counters for /metrics

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
        # prefill_hf_kernels: the prefill layer stacks (eager and graphs) run the fused reference-rounding kernels
        self.trun = DecoderRunner(self.thinker.w, self.tspec, self.tkv, layers=tlayers, hf_kernels=prefill_hf_kernels)
        self.arun = DecoderRunner(self.talker.w, self.aspec, self.akv, hf_kernels=prefill_hf_kernels, rope_kernel=prefill_rope_hf)
        # Track S sampling: the canonical warpers, no minimum length, no global maximum (HF streaming path)
        self.cfg = TalkerSampling(min_new_tokens=0, max_new_tokens=1 << 30)
        # the served context (prompt plus reply) of one Thinker sequence: Daily-Omni's interleaved frames and audio reach ~13k
        # tokens, so 32768 as the multimodal reference served (SGLang-Omni at 32k); a longer prompt is an HTTP 400
        self.tdec = ThinkerDecoder(self.thinker, self.tkv, self.tspec, use_graphs=use_graphs, layers=tlayers, max_len=thinker_max_len)
        self.adec = TalkerDecoder(self.talker, self.akv, self.aspec, self.cfg, use_graphs=use_graphs, whole_k=talker_whole_k,
                                  lean_frame=talker_lean_frame)
        self.tprefill = None
        if prefill_graphs and use_graphs:
            from engine.runtime.prefill_graph import PrefillGraphs

            self.tprefill = PrefillGraphs(self.trun)
            self.tprefill.capture_all()
        # multi-row Thinker prompt prefills (and continued chunks): graphs per row count and token bucket instead of the
        # host-bound eager forward; their scratch sequence's pages are kept out of the prefill lane's grants
        self.tchunk = None
        if thinker_chunk_prefill_graphs and use_graphs:
            from engine.runtime.prefill_graph import ChunkPrefillGraphs

            buckets = (64, 128, 256, 512, 1024)
            self.tchunk = ChunkPrefillGraphs(self.trun, rows=(1, 2, 3, 4), buckets=buckets, lane="thinker",
                                             pad_pages=-(-max(buckets) // self.tkv.page_size))
            self.tchunk.capture_all()
        # Talker chunk prefills: CUDA graphs per row count and token bucket (eager beyond them), the first frames of
        # one call sampled together (one host transfer instead of one per row)
        self.aprefill = None
        if talker_prefill_graphs and use_graphs:
            from engine.runtime.prefill_graph import ChunkPrefillGraphs

            self.aprefill = ChunkPrefillGraphs(self.arun)
            self.aprefill.capture_all()
        self.tdec.capture_all()
        self.adec.capture_all(ks=sorted({talker_k, max(talker_k, talker_k_max)}))  # whole-K graphs too (talker_whole_k)
        self.t2wp = None
        self.audio_live: dict[int, LiveS] = {}  # process mode: requests whose audio the worker still owes
        if t2w_process:  # Token2Wav (features, setups, windows) in its own process: no GIL shared with the engine loop
            from engine.runtime.t2w_worker import Token2WavProcess

            self.t2w = self.t2w_batch = None
            self.t2wp = Token2WavProcess({"model_dir": str(model_dir), "float16_flow": float16_flow, "t2w_graphs": t2w_graphs,
                                          "t2w_compile": t2w_compile, "spk_threads": spk_threads, "t2w_min_batch": t2w_min_batch,
                                          "t2w_max_wait_ms": t2w_max_wait_ms, "feature_workers": feature_workers,
                                          "profile_path": t2w_profile_path, "feature_processes": feature_processes,
                                          "t2w_euler_graphs": t2w_euler_graphs, "t2w_fast_dit": t2w_fast_dit,
                                          "t2w_fast_slots": t2w_fast_slots, "mps_sm_percent": t2w_mps_sm_percent, "t2w_policy": t2w_policy,
                                          "t2w_setup_calls_when_busy": t2w_setup_calls_when_busy, "t2w_max_decode_rows": t2w_max_decode_rows,
                                          "t2w_first_slack_ms": t2w_first_slack_ms, "t2w_first_lane_slack_ms": t2w_first_lane_slack_ms,
                                          "feature_gpu_tokens": feature_gpu_tokens,
                                          "feature_allocator": feature_allocator, "stage_trace_dir": stage_trace_dir,
                                          "t2w_hift_decode_tf32": t2w_hift_decode_tf32,
                                          "t2w_batch_setup_encoder": t2w_batch_setup_encoder, "t2w_async_results": t2w_async_results,
                                          "t2w_hift_rows": t2w_hift_rows, "t2w_merge_solve": t2w_merge_solve,
                                          "t2w_fuse_setup_frames": t2w_fuse_setup_frames, "t2w_hold_slack_ms": t2w_hold_slack_ms,
                                          "t2w_hold_max_ms": t2w_hold_max_ms, "t2w_hold_rows": t2w_hold_rows,
                                          "t2w_hold_setups": t2w_hold_setups, "t2w_hift_lean": t2w_hift_lean,
                                          "t2w_dit_fused_residual": t2w_dit_fused_residual, "t2w_dit_fused_conv": t2w_dit_fused_conv,
                                          "t2w_hift_fused_bias": t2w_hift_fused_bias, "t2w_setup_prefetch": t2w_setup_prefetch, "t2w_encoder_batched_io": t2w_encoder_batched_io,
                                          # the worker inherits the engine's environment: normal priority unless set
                                          "mps_client_priority": t2w_mps_client_priority if t2w_mps_client_priority is not None
                                          else (0 if mps_client_priority is not None else None)}, self._on_t2w)
        else:
            assert not t2w_fast_dit, "the fast DiT runs in the Token2Wav process (t2w_process)"
            assert not t2w_async_results, "asynchronous results are a Token2Wav-process option (t2w_process)"
            assert not t2w_merge_solve, "the merged solve is a fast-DiT option (t2w_process)"
            self.t2w = Token2Wav(model_dir / "assets" / "token2wav", float16=float16_flow, spk_threads=spk_threads)
            self.t2w_batch = BatchedToken2Wav(self.t2w, use_graphs=t2w_graphs, compile_estimator=t2w_compile, euler_graphs=t2w_euler_graphs,
                                              hift_decode_tf32=t2w_hift_decode_tf32, batch_setup_encoder=t2w_batch_setup_encoder,
                                              hift_rows=t2w_hift_rows, hift_lean=t2w_hift_lean)
            if t2w_graphs:  # a prepared default voice supplies the cache dtypes
                self.t2w_batch.capture_graphs(self.t2w_batch.setup(self.t2w.features(model_dir / "assets" / "HT_ref_audio.wav")))
        self.frontend = MultimodalFrontend(model_dir, device=device)
        self.device, self.max_live, self.talker_k, self.talker_k_max = device, max_live, talker_k, talker_k_max
        # Talker KV is committed for an utterance's whole lifetime before its first chunk starts (``_grant_talker``):
        # seq -> committed pages.  The decoder's bucket padding rows take one K-frame block each beside them.
        self.talker_commit: dict[int, int] = {}
        self.talker_pad_pages = (max(self.adec.buckets) - 1) * -(-max(talker_k, talker_k_max) // self.akv.page_size)
        if self.aprefill is not None:  # the chunk-prefill graphs' scratch sequence
            self.talker_pad_pages += self.aprefill.PAD_PAGES
        self.admitted = 0
        self.thinker_lazy = thinker_lazy
        self.profile = profile
        self.stage_trace_dir = stage_trace_dir  # diagnostic stage trace (engine.runtime.stage_trace): one per engine thread
        self.timelines: dict[str, list[tuple[str, float]]] = {}  # request id -> (event, perf_counter), when profiling or tracing
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
        self.split_threads = split_threads
        # tick_overlap: one engine thread, the Thinker step queued on its own stream before the Talker step and collected
        # after it, so the two GPU waits of a tick overlap (text chunks reach the Talker one tick later)
        if tick_overlap and split_threads:
            raise ValueError("tick_overlap and split_threads are alternatives")
        self.tick_overlap = tick_overlap
        # False: a tick whose Thinker step completes some request's first text chunk runs serially, so that chunk starts
        # in the same tick's Talker (time to first audio); the other ticks overlap
        self.tick_overlap_first_chunks = tick_overlap_first_chunks
        # a tick with more live requests runs serially: at C=32 the Token2Wav worker is GPU bound and the overlapped
        # ticks' smaller Talker batches cost it GPU time (round 15: -3% throughput at C=32, +2-7% at C<=16)
        self.tick_overlap_max_live = tick_overlap_max_live
        if tick_overlap:
            self.thinker_stream, self.talker_stream = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
        if split_threads:  # Thinker and Talker loops on their own threads and CUDA streams (graph replays overlap)
            self.talker_wake = threading.Event()
            self.thinker_stream, self.talker_stream = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
            self.thread = threading.Thread(target=self._run_thinker, daemon=True, name="thinker")
            self.talker_thread = threading.Thread(target=self._run_talker, daemon=True, name="talker")
            self.thread.start()
            self.talker_thread.start()
        else:
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

    @staticmethod
    def _frame_cap(req: EngineRequest) -> int:
        return int(dict(req.talker_sampling).get("max_new_tokens") or TALKER_FRAMES_DEFAULT)

    def _talker_need(self, req: EngineRequest) -> int:
        """Talker KV pages one audio request may use over its whole utterance (``talker_lifetime``)."""
        return -(-talker_lifetime(req.max_tokens, self._frame_cap(req), max(self.talker_k, self.talker_k_max)) // self.akv.page_size)

    def _check_capacity(self, req: EngineRequest) -> None:
        """An audio request whose lifetime bound exceeds the Talker context or the whole pool can never be granted."""
        from engine.server import BadRequest

        if req.ref_audio is None or "audio" not in req.modalities:
            return
        tokens = talker_lifetime(req.max_tokens, self._frame_cap(req), max(self.talker_k, self.talker_k_max))
        if tokens > self.adec.max_len or self._talker_need(req) > self.akv.num_pages - self.talker_pad_pages:
            raise BadRequest(f"max_tokens={req.max_tokens} with {self._frame_cap(req)} speech frames may need {tokens} Talker KV tokens, "
                             f"beyond this engine's Talker capacity ({self.adec.max_len} positions, "
                             f"{(self.akv.num_pages - self.talker_pad_pages) * self.akv.page_size} pooled tokens)")

    def _prepare(self, ref: bytes):
        """Worker thread: the reference voice's features (no flow work; the Token2Wav thread runs the flow setup)."""
        feats = self.t2w.features(ref)
        torch.cuda.current_stream().synchronize()  # the Token2Wav thread reads them on its own stream
        self.t2w_wake.set()
        return feats

    async def generate(self, req: EngineRequest):
        self._check_sampling(req)
        self._check_capacity(req)
        with self._lock:  # an admission slot is taken here, before any await, and given back when the request ends
            if self.admitted >= self.max_live:
                raise Overloaded(f"{self.max_live} live requests")
            self.admitted += 1
        try:
            async for ev in self._generate(req):
                yield ev
        finally:
            with self._lock:
                self.admitted -= 1

    async def _generate(self, req: EngineRequest):
        wants = req.ref_audio is not None and "audio" in req.modalities
        fut = self.pool.submit(self._prepare, req.ref_audio) if wants and self.t2wp is None else None
        prompt, embeds = await asyncio.to_thread(self._prompt, req)
        with self._lock:
            seq = self._next
            self._next += 1
        lv = LiveS(req=req, out=asyncio.Queue(), loop=asyncio.get_running_loop(), prompt_ids=prompt, seq=seq, prompt_embeds=embeds,
                   prompt_future=fut, frame_cap=self._frame_cap(req))
        if self.profile or self.stage_trace_dir:
            lv.timeline = self.timelines.setdefault(req.request_id, [])
            lv.mark("queued")
            if fut is not None:
                fut.add_done_callback(lambda _: lv.mark("prompt_ready"))
        self.inbox.put(lv)
        try:
            first = True
            while True:
                ev = await lv.out.get()
                if isinstance(ev, Exception):
                    raise ev
                if first and isinstance(ev, AudioChunk):
                    first = False
                    lv.mark("first_yield")  # handed to the HTTP response
                yield ev
                if isinstance(ev, Done):
                    return
        finally:
            lv.cancelled = True
            if self.stage_trace_dir and lv.timeline is not None:
                self._write_timeline(req.request_id, lv)

    def _write_timeline(self, rid: str, lv: LiveS) -> None:
        """Tracing: the request's timeline to ``<stage_trace_dir>/requests.jsonl`` (``tools/engine_profile.py``)."""
        import json

        lv.mark("end")
        row = json.dumps({"rid": rid, "seq": lv.seq, "prompt_tokens": len(lv.prompt_ids), "frames": lv.frames,
                          "events": [[e, round(t, 6)] for e, t in lv.timeline]})
        with self._lock:
            with open(Path(self.stage_trace_dir) / "requests.jsonl", "a") as f:
                f.write(row + "\n")
        self.timelines.pop(rid, None)

    def stats(self) -> dict:
        import resource

        return {"live": len(self.live), "queued": self.inbox.qsize(), "admitted": self.admitted, "max_live": self.max_live, "first_chunk": self.first_chunk, "thinker_kv_pages_used": self.tkv.used_pages,
                "talker_kv_pages_used": self.akv.used_pages, "thinker_kv_pages": self.tkv.num_pages, "talker_kv_pages": self.akv.num_pages,
                "talker_kv_committed_pages": sum(self.talker_commit.values()), "talker_kv_pad_pages": self.talker_pad_pages,
                "gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()), "gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
                "rss_max_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss, "rss_kib": _rss_kib(), "audio_live": len(self.audio_live),
                **self.sched, **getattr(self, "t2w_stats", {})}

    # --- engine thread ---------------------------------------------------------------------------
    @property
    def trace(self):
        """The calling thread's stage trace (made on first use: ``engine`` for the tick thread, ``engine-<thread>`` for
        the split-thread and in-process Token2Wav threads), or None when tracing is off."""
        if not getattr(self, "stage_trace_dir", None):
            return None
        tl = self.__dict__.get("_trace_tl")
        if tl is None:
            tl = self.__dict__.setdefault("_trace_tl", threading.local())
        if getattr(tl, "trace", None) is None:
            from engine.runtime.stage_trace import StageTrace

            name = threading.current_thread().name
            tl.trace = StageTrace(self.stage_trace_dir, "engine" if name == "engine" else f"engine-{name}", proc="engine")
        return tl.trace

    def _run(self) -> None:
        torch.cuda.set_device(torch.device(self.device).index or 0)  # torch >= 2.13 rejects a bare "cuda"
        _ = self.trace  # made before the first request (its base event synchronizes)
        while True:
            try:
                while True:
                    if not self.live:
                        self.tdec.release_pads()  # idle: the decoders' scratch rows give their pages back
                        if self.trace is not None:
                            self.trace.flush(block=True)
                    lv = self.inbox.get(block=not self.live, timeout=None if not self.live else 0)
                    try:
                        self._admit(lv)
                    except Exception as e:  # a request that cannot be admitted fails alone
                        lv.emit(RuntimeError(f"engine admission error: {type(e).__name__}: {e}"))
            except queue.Empty:
                pass
            try:
                with torch.inference_mode(), torch.cuda.stream(self.talker_stream if self.tick_overlap else torch.cuda.current_stream()):
                    self._tick()
            except Exception as e:  # a failing tick fails its requests, never the engine thread
                import traceback

                traceback.print_exc()
                for lv in list(self.live.values()):
                    lv.emit(RuntimeError(f"engine error: {type(e).__name__}: {e}"))
                    self._release(lv)

    def _run_thinker(self) -> None:
        """Split-thread mode: admission, prefill and Thinker decode on the Thinker stream; text chunks to the Talker."""
        torch.cuda.set_device(torch.device(self.device).index or 0)
        while True:
            busy = any(not lv.thinker_done for lv in list(self.live.values()))
            if not busy:
                self.tdec.release_pads()
            try:
                while True:
                    lv = self.inbox.get(block=not busy, timeout=None if not busy else 0)
                    try:
                        self._admit(lv)
                        busy = True
                    except Exception as e:
                        lv.emit(RuntimeError(f"engine admission error: {type(e).__name__}: {e}"))
            except queue.Empty:
                pass
            try:
                with torch.cuda.stream(self.thinker_stream), torch.inference_mode():
                    for lv in list(self.live.values()):
                        if lv.cancelled and not lv.thinker_done:
                            lv.thinker_done = True
                            self.tkv.release(lv.seq)
                            if not lv.wants_audio:
                                self.live.pop(lv.seq, None)
                                lv.finished = True
                    rows, _ = self._prefill_lane([lv for lv in list(self.live.values()) if not lv.thinker_done])
                    if rows:
                        self._timed("thinker", self._thinker_step, rows)
                self.talker_wake.set()
            except Exception as e:
                import traceback

                traceback.print_exc()
                for lv in list(self.live.values()):
                    if not lv.thinker_done:
                        lv.emit(RuntimeError(f"engine error: {type(e).__name__}: {e}"))
                        lv.cancelled = True

    def _run_talker(self) -> None:
        """Split-thread mode: Talker chunk prefills and K-frame decode on the Talker stream; windows to Token2Wav."""
        torch.cuda.set_device(torch.device(self.device).index or 0)
        while True:
            try:
                with torch.cuda.stream(self.talker_stream), torch.inference_mode():
                    for lv in list(self.live.values()):
                        if lv.cancelled and not lv.utterance_done and lv.wants_audio:
                            lv.utterance_done = True
                            self._akv_release(lv.seq)
                            if self.t2wp is not None and self.audio_live.pop(lv.seq, None) is not None:
                                self.t2wp.send("close", lv.seq)
                            if lv.thinker_done:
                                self.live.pop(lv.seq, None)
                                lv.finished = True
                    rows = [lv for lv in list(self.live.values()) if lv.wants_audio and not lv.utterance_done]
                    work = [lv for lv in rows if lv.talking or lv.chunks]
                    if work:
                        self._timed("talker", self._talker_step, rows)
                    for lv in [x for x in list(self.live.values()) if x.utterance_done and x.thinker_done and not x.cancelled]:
                        self._akv_release(lv.seq)
                        self.live.pop(lv.seq, None)
                if not work:
                    self.talker_wake.wait(0.002)
                    self.talker_wake.clear()
            except Exception as e:
                import traceback

                traceback.print_exc()
                for lv in list(self.live.values()):
                    if lv.wants_audio and not lv.utterance_done:
                        lv.emit(RuntimeError(f"engine error: {type(e).__name__}: {e}"))
                        lv.cancelled = True

    def _admit(self, lv: LiveS) -> None:
        from engine.runtime.t2w_sched import AudioState

        lv.t_admit, lv.audio = _now(), AudioState()
        lv.mark("admitted")
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

    def _akv_release(self, seq: int) -> None:
        self.akv.release(seq)
        self.talker_commit.pop(seq, None)

    def _release(self, lv: LiveS) -> None:
        self.tkv.release(lv.seq)
        self._akv_release(lv.seq)
        self.live.pop(lv.seq, None)
        lv.finished = True
        if self.t2wp is not None and self.audio_live.pop(lv.seq, None) is not None:
            self.t2wp.send("close", lv.seq)

    def _on_t2w(self, m) -> None:
        """Receiver thread (process mode): the worker's audio, completion and errors for each request."""
        kind, seq = m[0], m[1] if len(m) > 1 else None
        if kind == "stats":
            self.t2w_stats = m[2]
            return
        if kind == "mark":  # tracing: a worker-side timeline event
            lv = self.audio_live.get(seq)
            if lv is not None:
                lv.mark(m[2], m[3])
            return
        if kind == "fatal":  # the Token2Wav worker died: every request still owed audio fails now
            for lv in list(self.audio_live.values()):
                lv.emit(RuntimeError(f"token2wav worker failed: {m[2].strip().splitlines()[-1] if m[2] else ''}"))
            self.audio_live.clear()
            return
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
            if len(m) > 4:  # tracing: the window's receipt and decode call in the worker, and its receipt here
                index, t_recv, t_dec0, t_dec1 = m[4]
                lv.mark(f"w_recv:{index}", t_recv)
                lv.mark(f"w_decode:{index}", t_dec0)
                lv.mark(f"w_decoded:{index}", t_dec1)
                lv.mark(f"audio_recv:{index}")
            lv.audio.sent(len(m[3]), _now())
            lv.emit(AudioChunk(m[3], m[2]))
            lv.sample_offset = m[2] + len(m[3])
        elif kind == "done":
            self.audio_live.pop(seq, None)
            lv.emit(Done(lv.finish_reason, {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated), "audio_frames": lv.frames}))
            lv.finished = True
        elif kind == "error":
            self.audio_live.pop(seq, None)
            from engine.server import BadRequest

            lv.emit(BadRequest(m[2]) if len(m) > 3 and m[3] == "bad_request" else RuntimeError(m[2]))

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

    def _timed(self, name: str, fn, *a, live: bool = True) -> None:
        """``fn(*a)`` as stage ``name``; ``a[0]`` is the stage's rows (``live``: LiveS rows, whose Talker frames the
        trace counts; otherwise only their number)."""
        tr = self.trace
        if tr is not None and a[0]:  # stage trace: no synchronization; rows and (Talker) frames produced
            frames = sum(lv.frames for lv in a[0]) if live else 0
            tok = tr.begin(name)
            try:
                fn(*a)
            finally:  # a failing stage still ends its span (an open one holds the completeness watermark)
                tr.end(tok, len(a[0]), sum(lv.frames for lv in a[0]) - frames if live else 0)
            return
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
        rows, urgent = self._prefill_lane([x for x in self.live.values() if not x.thinker_done])
        if self.thinker_lazy and rows and not any(self._thinker_urgent(x) for x in rows):
            rows = []  # every Talker already holds a chunk beyond the one it decodes: give this tick to the Talker
        if self.tick_overlap:
            self._tick_overlapped(rows, urgent)
        elif urgent:  # a continuing stream is short of playback: its next frames first, then the aged prefill slice
            self._timed("thinker", self._thinker_step, [x for x in rows if not x.pending_prefill])
            self._timed("talker", self._talker_step, [x for x in self.live.values() if x.wants_audio and not x.utterance_done])
            self._timed("prefill", self._thinker_step, [x for x in rows if x.pending_prefill])
        else:
            self._timed("thinker", self._thinker_step, rows)
            self._timed("talker", self._talker_step, [x for x in self.live.values() if x.wants_audio and not x.utterance_done])
        for lv in [x for x in list(self.live.values()) if x.utterance_done]:
            # the Thinker and Talker are done: free their KV; the Token2Wav thread emits the rest and Done
            self.tkv.release(lv.seq)
            self._akv_release(lv.seq)
            self.live.pop(lv.seq, None)
            if self.t2wp is None:
                self.t2w_wake.set()

    def _tick_overlapped(self, rows: list, urgent: bool) -> None:
        """``_tick``'s Thinker and Talker steps with their GPU work overlapped: the Thinker step (prefill included unless
        a continuing stream is short of playback) is queued on the Thinker stream, the Talker step runs on the Talker
        stream meanwhile, then the Thinker's tokens are collected.  Chunks passed now start in the next tick's Talker.
        Without ``tick_overlap_first_chunks`` a tick that completes a first chunk runs serially (same streams), and so
        does a tick with more than ``tick_overlap_max_live`` live requests."""
        first = [x for x in rows if not x.pending_prefill] if urgent else rows
        if ((not self.tick_overlap_first_chunks and self._first_chunk_due(first))
                or (self.tick_overlap_max_live is not None and len(self.live) > self.tick_overlap_max_live)):
            with torch.cuda.stream(self.thinker_stream):
                self._timed("thinker", self._thinker_step, first)
            self._timed("talker", self._talker_step, [x for x in self.live.values() if x.wants_audio and not x.utterance_done])
            if urgent:
                with torch.cuda.stream(self.thinker_stream):
                    self._timed("prefill", self._thinker_step, [x for x in rows if x.pending_prefill])
            return
        with torch.cuda.stream(self.thinker_stream):
            pending = self._timed_ret("thinker", self._thinker_launch, first)
        self._timed("talker", self._talker_step, [x for x in self.live.values() if x.wants_audio and not x.utterance_done])
        with torch.cuda.stream(self.thinker_stream):
            self._timed("thinker.wait", self._thinker_finish, first, pending)
            if urgent:
                self._timed("prefill", self._thinker_step, [x for x in rows if x.pending_prefill])

    @staticmethod
    def _first_chunk_due(rows: list) -> bool:
        """Does this Thinker step complete some audio request's first text chunk (its 11th reply token)?"""
        return any(lv.wants_audio and not lv.pending_prefill and lv.chunk_index == 0 and len(lv.generated) == THINKER_CHUNK for lv in rows)

    def _min_slack(self, now: float) -> float | None:
        """The least buffered playback among the continuing audio streams (None if none)."""
        return min((lv.audio.slack(now) for lv in list(self.live.values()) + list(self.audio_live.values())
                    if lv.audio is not None and lv.audio.started and not lv.finished and not lv.cancelled), default=None)

    @staticmethod
    def _growth(lv: LiveS) -> int:
        """Thinker KV tokens a request may still add: the rest of its prompt and of its chunk-rounded reply."""
        return len(lv.prompt_ids) - lv.prefilled + max(0, thinker_bound(lv.req.max_tokens) - len(lv.generated)) + 1

    def _prefill_lane(self, rows: list) -> tuple[list, bool]:
        """The Thinker rows of this tick and whether a continuing stream is short of playback (``t2w_sched``): every
        decoding row, and the prefill work the bounded first-chunk lane grants (``choose_prefills``: oldest first within
        the hard token budget, chunking a prompt that alone exceeds it; only while the Thinker KV pool holds the
        request's lifetime beside what running requests may still need; only aged ones below the slack floor).
        ``lv.prefill_now`` is each granted request's share."""
        from engine.runtime.t2w_sched import choose_prefills, low_slack

        pending = [lv for lv in rows if lv.pending_prefill]
        for lv in pending:
            lv.prefill_now = 0
        if not pending:
            return rows, False
        now = _now()
        min_slack = self._min_slack(now)
        free = len(self.tkv.free) - self.tdec.pad_pages_outstanding() - getattr(getattr(self, "tchunk", None), "pad_pages", 0) - sum(
            self.tkv.pages_needed(lv.seq, self._growth(lv)) for lv in rows if not lv.pending_prefill or lv.prefilled)
        take = dict(choose_prefills([(lv.t_admit, len(lv.prompt_ids) - lv.prefilled, lv.seq,
                                      0 if lv.prefilled else self.tkv.pages_needed(lv.seq, self._growth(lv))) for lv in pending],
                                    now, min_slack, self.prefill_policy, free_pages=free))
        urgent = low_slack(min_slack, self.prefill_policy)
        for lv in pending:
            lv.prefill_now = take.get(lv.seq, 0)
        n = sum(take.values())
        sc = self.sched
        sc["prefill_tokens_tick_max"] = max(sc["prefill_tokens_tick_max"], n)
        if any(0 < lv.prefill_now < len(lv.prompt_ids) - lv.prefilled for lv in pending):
            sc["prefill_chunked_ticks"] += 1
        oldest = min(pending, key=lambda lv: lv.t_admit)
        if not oldest.prefill_now and not (urgent and now - oldest.t_admit < self.prefill_policy.max_age_s):
            sc["prefill_kv_wait_ticks"] += 1  # the oldest waits for KV pages, not for the slack floor
        return [lv for lv in rows if not lv.pending_prefill or lv.prefill_now], urgent and n > 0

    # thinker -----------------------------------------------------------------------------------
    @staticmethod
    def _thinker_urgent(lv: LiveS) -> bool:
        """A Thinker row that must step now: before its first text chunk (time to first audio), text-only, or its
        Talker holds at most the chunk it is decoding.  Lazy mode steps the Thinker only when some row is urgent (and
        then steps every active row, a step costs the same at any batch); greedy text makes this timing-only."""
        return lv.pending_prefill or not lv.wants_audio or lv.chunk_index == 0 or len(lv.chunks) <= 1

    def _thinker_step(self, rows: list[LiveS]) -> None:
        if rows:
            self._thinker_finish(rows, self._thinker_launch(rows))

    def _thinker_launch(self, rows: list[LiveS]) -> list:
        """Queue the Thinker step of ``rows`` on the current stream: [(rows, next tokens in a host tensor filled
        asynchronously, hidden states of the fed tokens or None)], valid once the event recorded last is reached."""
        if not rows:
            return []
        pre = [lv for lv in rows if lv.pending_prefill]
        dec = [lv for lv in rows if not lv.pending_prefill]
        out = []
        if pre:
            with span(self.trace, "thinker.prefill", len(pre)):
                done, toks = self._prefill_forward(pre)
            if done:
                out.append(([pre[i] for i in done], toks, None))
        if dec:
            with span(self.trace, "thinker.decode", len(dec)):
                nxt, hid = self.tdec.step_async([lv.seq for lv in dec], [lv.generated[-1] for lv in dec])
            out.append((dec, nxt, hid))
        if out:
            ev = None
            if torch.device(self.device).type == "cuda":
                ev = torch.cuda.Event()
                ev.record()
            out.append(ev)
        return out

    def _thinker_finish(self, rows: list[LiveS], pending: list) -> None:
        """The tokens of a launched Thinker step: text deltas, Track S chunks and closes, as in the reference."""
        if not pending:
            return
        ev = pending.pop()
        if ev is not None:
            ev.synchronize()
        results = []  # (lv, next token, hidden of the fed token or None)
        for lvs, toks, hid in pending:
            for i, (lv, tok) in enumerate(zip(lvs, toks.tolist())):
                if hid is None:  # its prompt is complete: the first reply token
                    lv.pending_prefill = False
                    lv.mark("prefilled")
                results.append((lv, tok, None if hid is None else hid[i]))
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

    def _prompt_embeds(self, lv: LiveS, a: int, b: int) -> torch.Tensor:
        if lv.prompt_embeds is not None:
            return lv.prompt_embeds[a:b].to(self.thinker.dtype)
        return self.thinker.embed(to_device(lv.prompt_ids[a:b], torch.int64, self.device))

    def _prefill_forward(self, pre: list[LiveS]) -> tuple[list[int], torch.Tensor | None]:
        """Prefill each row's granted share of its prompt (``prefill_now``; the whole rest when not set) in one ragged
        batch; returns the indices of the rows whose prompt is now complete and their first reply tokens (greedy), in a
        host tensor filled once the current stream reaches this point."""
        ns = [lv.prefill_now or len(lv.prompt_ids) - lv.prefilled for lv in pre]
        whole = all(lv.prefilled == 0 and n == len(lv.prompt_ids) for lv, n in zip(pre, ns))
        if len(pre) == 1 and whole and self.tprefill is not None and self.tprefill.supports(ns[0]):
            last = self.tprefill.forward(self._prompt_embeds(pre[0], 0, ns[0]), pre[0].seq)[-1:]  # graph-captured single-request prefill
        elif getattr(self, "tchunk", None) is not None and self.tchunk.supports(len(pre), sum(ns)):
            hid, idx = self.tchunk.forward(torch.cat([self._prompt_embeds(lv, lv.prefilled, lv.prefilled + n) for lv, n in zip(pre, ns)]),
                                           [lv.seq for lv in pre], ns)
            last = hid[idx]  # a copy: the graph's output buffer is reused by the next call
        else:
            lay = self.tkv.layout([lv.seq for lv in pre], ns)
            last = self.trun.forward(torch.cat([self._prompt_embeds(lv, lv.prefilled, lv.prefilled + n) for lv, n in zip(pre, ns)]), lay)
            last = last[lay.last_token_index]
        for lv, n in zip(pre, ns):
            lv.prefilled += n
            lv.prefill_now = 0
        done = [i for i, lv in enumerate(pre) if lv.prefilled >= len(lv.prompt_ids)]
        if not done:
            return done, None
        sel = last if len(done) == len(pre) else last[to_device(done, torch.int64, last.device)]
        ids = torch.nn.functional.linear(sel, self.thinker.w["lm_head.weight"]).float().argmax(-1)
        host = torch.empty(ids.shape, dtype=ids.dtype, pin_memory=True)
        host.copy_(ids, non_blocking=True)
        return done, host

    def _close_thinker(self, lv: LiveS, passed_final: bool = False) -> None:
        if not passed_final:
            self._pass_chunk(lv, final=True)
        lv.thinker_done = True
        self.tkv.release(lv.seq)
        if not lv.wants_audio:
            if self.split_threads:  # text only: the Talker never saw it
                lv.emit(Done("stop", {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated), "audio_frames": 0}))
                self.live.pop(lv.seq, None)
                lv.finished = True
            else:
                self._finish(lv)

    def _pass_chunk(self, lv: LiveS, final: bool) -> None:
        """The tokens fed since the last chunk (those with hidden states), as ``track_s.thinker_chunks`` yields them."""
        ids, hs = lv.generated[lv.passed:len(lv.hidden)], lv.hidden[lv.passed:]
        lv.passed = len(lv.hidden)
        if lv.wants_audio:
            h = torch.stack(hs) if hs else torch.zeros(0, self.tspec.hidden_size, device=self.device, dtype=self.thinker.dtype)
            ev = None
            if self.split_threads or self.tick_overlap:
                ev = torch.cuda.Event()
                ev.record(torch.cuda.current_stream())
                h.record_stream(self.talker_stream)  # read on the Talker stream: not reused before that work is done
            lv.chunks.append(TextChunk(ids, h, final, ev))
            lv.mark(f"text_chunk{'_final' if final else ''}:{len(ids)}")
            if self.split_threads:
                self.talker_wake.set()

    # talker ------------------------------------------------------------------------------------
    def _grant_talker(self, starts: list[LiveS]) -> list[LiveS]:
        """The starts whose utterance holds a Talker KV commitment.  A first start is granted its whole lifetime bound
        (``_talker_need``) in admission order while it fits beside every commitment and the decoder's padding; the
        first that does not fit holds the later ones (no starvation).  Every committed utterance can run to its end, so
        continuation chunks never wait for KV and a pool full of unfinished utterances cannot deadlock."""
        first = sorted((lv for lv in starts if lv.seq not in self.talker_commit), key=lambda lv: (lv.t_admit, lv.seq))
        room = self.akv.num_pages - self.talker_pad_pages - sum(self.talker_commit.values())
        for lv in first:
            need = self._talker_need(lv.req)
            if need > room:
                self.sched["talker_kv_wait_ticks"] += 1
                break
            self.talker_commit[lv.seq] = need
            room -= need
        return [lv for lv in starts if lv.seq in self.talker_commit]

    def _beyond_commit(self, lv: LiveS, extra: int) -> bool:
        """Would ``extra`` more Talker KV tokens take ``lv`` past its committed pages?  (A broken lifetime bound.)"""
        held = len(self.akv.pages.get(lv.seq, [])) + self.akv.pages_needed(lv.seq, extra)
        if held <= self.talker_commit.get(lv.seq, 0):
            return False
        lv.emit(RuntimeError(f"engine error: Talker KV beyond the request's committed {self.talker_commit.get(lv.seq, 0)} pages"))
        lv.cancelled = True  # fails alone; released next tick
        return True

    def _talker_starts(self, starts: list[LiveS]) -> list[LiveS]:
        """The chunk starts (Talker prefills) of this tick: those holding a KV commitment (``_grant_talker``), earliest
        playback deadline first (a request without audio yet: admission + ``max_age_s``, so first chunks lead unless a
        continuing stream is behind), within the per-tick token budget (``t2w_sched.choose_starts``; at least one)."""
        starts = self._grant_talker(starts)
        if not starts:
            return starts
        from engine.runtime.t2w_sched import choose_starts

        pol = self.prefill_policy
        keep = set(choose_starts([(lv.audio.t_exhaust if lv.audio is not None and lv.audio.started else lv.t_admit + pol.max_age_s,
                                   len(lv.chunks[0].ids) + 2, lv.seq) for lv in starts], pol))
        if len(keep) < len(starts):
            self.sched["talker_starts_deferred_ticks"] += 1
        return [lv for lv in starts if lv.seq in keep]

    def _talker_step(self, rows: list[LiveS]) -> None:
        from engine.talker_sampler import PAD, DecodeState, step

        eos = self.talker.cfg.eos
        starts = self._talker_starts([lv for lv in rows if not lv.talking and lv.chunks])
        if starts:
            with span(self.trace, "talker.condition", len(starts)):
                feeds = []
                for lv in starts:
                    ch = lv.chunks[0]
                    if ch.ready is not None:  # the hidden states come from the Thinker stream
                        torch.cuda.current_stream().wait_event(ch.ready)
                    cond = self.talker.condition(to_device(ch.ids, torch.long, self.device), ch.hidden)
                    if not ch.final:
                        cond = torch.cat([cond[:-2], cond[-1:]], dim=0)  # text_eos only on the final chunk; audio_bos always
                    if lv.feed_last:  # the previous chunk stopped at the per-chunk limit before its last frame was fed
                        cond = torch.cat([self.talker.code_embed(lv.ts["codes"]), cond], dim=0)
                        lv.feed_last = False
                    feeds.append(cond)
                ok = [i for i, (lv, f) in enumerate(zip(starts, feeds)) if not self._beyond_commit(lv, f.shape[0])]
                starts, feeds = [starts[i] for i in ok], [feeds[i] for i in ok]
        if starts:
            with span(self.trace, "talker.prefill", len(starts)):
                seqs, ns = [lv.seq for lv in starts], [f.shape[0] for f in feeds]
                if self.aprefill is not None and self.aprefill.supports(len(starts), sum(ns)):
                    hidden, last = self.aprefill.forward(torch.cat(feeds), seqs, ns)
                else:
                    lay = self.akv.layout(seqs, ns)
                    hidden, last = self.arun.forward(torch.cat(feeds), lay), lay.last_token_index
                logits = torch.nn.functional.linear(hidden[last], self.talker.head).float()
                for lv in starts:
                    if lv.ts is None:
                        seed = request_seed(lv.req)
                        lv.ts = {"codes": torch.zeros(1, dtype=torch.int64, device=self.device), "seeds": to_device([seed], torch.int64, self.device),
                                 "steps": torch.zeros(1, dtype=torch.int64, device=self.device),
                                 "history": torch.full((1, self.cfg.window), PAD, dtype=torch.int64, device=self.device),
                                 "live": torch.ones(1, dtype=torch.bool, device=self.device)}
                n = len(starts)
                if self.aprefill is not None:  # one sampling call and one host transfer for every row (rows are independent)
                    st = DecodeState(torch.cat([lv.ts["seeds"] for lv in starts]), torch.cat([lv.ts["steps"] for lv in starts]),
                                     torch.cat([lv.ts["history"] for lv in starts]), torch.zeros(n, dtype=torch.bool, device=self.device),
                                     torch.ones(n, dtype=torch.bool, device=self.device))
                    toks = step(st, logits, eos, self.cfg).tolist()
                    for i, lv in enumerate(starts):
                        lv.ts.update(steps=st.steps[i:i + 1], history=st.history[i:i + 1])
                else:
                    toks = []
                    for i, lv in enumerate(starts):
                        st = DecodeState(lv.ts["seeds"], lv.ts["steps"], lv.ts["history"], torch.zeros(1, dtype=torch.bool, device=self.device),
                                         torch.ones(1, dtype=torch.bool, device=self.device))
                        toks.append(int(step(st, logits[i:i + 1], eos, self.cfg)[0]))
                        lv.ts.update(steps=st.steps, history=st.history)
                for lv, tok in zip(starts, toks):
                    lv.chunk_frames = 0
                    self._take_frame(lv, tok, fed_upto=None)
        active = [lv for lv in rows if lv.talking and not lv.cancelled]
        if not active:
            return
        # adaptive K: short blocks while work is waiting to join (a chunk start, a Thinker prefill) or any Thinker is
        # still generating (one text token per tick must stay ahead of ~4 codec frames per token); long blocks once all
        # text is done (fewer ticks per frame).  A row that hits EOS inside a block is rolled back exactly either way.
        # (in split-thread mode the Thinker runs beside the Talker, so only waiting chunk starts keep blocks short)
        waiting = any((not lv.talking and lv.chunks) or (not lv.thinker_done and not self.split_threads)
                      for lv in list(self.live.values())) or not self.inbox.empty()
        k = self.talker_k if waiting else max(self.talker_k, self.talker_k_max)
        active = [lv for lv in active if not self._beyond_commit(lv, k)]
        if not active:
            return
        starts_len = [self.akv.lengths[lv.seq] for lv in active]
        state = {name: torch.cat([lv.ts[name] for lv in active]) for name in ("codes", "seeds", "steps", "history", "live")}
        with span(self.trace, "talker.decode", len(active)):
            codes = self.adec.decode_k([lv.seq for lv in active], state, k)  # [K, n] on the host
        for i, lv in enumerate(active):
            for name in ("codes", "seeds", "steps", "history", "live"):
                lv.ts[name] = state[name][i:i + 1]  # a view: the batch is built anew by the next tick's cat, never written in place
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
        lv.ts["codes"] = to_device([tok], torch.int64, self.device)
        if lv.frames >= lv.frame_cap:
            # the request's frame cap (talker max_new_tokens): the utterance ends with this frame, unfed; text the
            # Thinker has not yet passed gets no speech
            if fed_upto is not None:
                self.akv.truncate(lv.seq, fed_upto)
            lv.finish_reason = "length"
            self._end_chunk(lv, last=True)
            return
        if lv.chunk_frames >= TALKER_MAX_PER_CHUNK:
            # the reference feeds this frame and moves on to the next chunk without an EOS
            if fed_upto is not None:
                self.akv.truncate(lv.seq, fed_upto)
            lv.feed_last = True
            self._end_chunk(lv)
            return
        lv.talking = True
        lv.ts["live"] = torch.ones(1, dtype=torch.bool, device=self.device)

    def _end_chunk(self, lv: LiveS, last: bool = False) -> None:
        """The current chunk ends; the utterance too at its final chunk, or (``last``) at the frame cap."""
        ch = lv.chunks.pop(0)
        lv.talking = False
        if ch.final or last:
            lv.chunks.clear()
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
                lv.emit(Done(lv.finish_reason, {"prompt_tokens": len(lv.prompt_ids), "completion_tokens": len(lv.generated), "audio_frames": lv.frames}))
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
                    prompts = self._timed_ret("setup", self.t2w_batch.setup_many, [f for _, f in part], live=False)
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

    def _timed_ret(self, name: str, fn, *a, live: bool = True):
        out = []
        self._timed(name, lambda *x: out.append(fn(*x)), *a, live=live)
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
                lv.audio.sent(len(wav), _now())
                lv.emit(AudioChunk(wav, lv.sample_offset))
                lv.sample_offset += len(wav)


def _now() -> float:
    import time

    return time.perf_counter()
