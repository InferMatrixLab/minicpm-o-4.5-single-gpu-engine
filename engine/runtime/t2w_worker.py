"""Token2Wav worker process: reference-voice features, prompt setups and batched window decodes off the engine's GIL.

The engine process (Thinker, Talker, stream chunkers) sends small messages; this process owns every Token2Wav state:

    ("open", seq, ref_audio_bytes, noise_seed)      a request with audio output was admitted
    ("window", seq, tokens, last, index)            a codec window is ready (in stream order)
    ("close", seq)                                  the request was cancelled
    ("stop",)

and answers with ``("audio", seq, offset, float32 samples)``, ``("done", seq)`` after a stream's last window, and
``("error", seq, message)``.  Batching: first windows are decoded at once, continuation windows wait for
``min_batch`` streams or ``max_wait_ms``; setups of all ready voices run batched before the windows that need them.
Token2Wav itself is ``engine.runtime.token2wav_batch.BatchedToken2Wav`` (same per-request semantics as in-process).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from engine.runtime.t2w_sched import AudioState, Policy, choose_windows, plan_tick


_FEAT = None  # per feature process: an object with what ``Token2wav._prepare_prompt`` reads


def _rss_kib(pid: int) -> int | None:
    try:
        for line in open(f"/proc/{pid}/status"):
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _feat_init(model_dir: str, spk_threads: int, gpu_tokens: bool = True) -> None:
    """Feature-process initializer: CAMPPlus (CPU ONNX) and the fbank/mel front ends, plus the S3 tokenizer on the GPU
    when ``gpu_tokens`` (one CUDA context per feature process; the S3 encoder is launch-heavy Python, and in a thread of
    the worker it contends for the worker's GIL: measured slower).  Without it the process is CPU only and the worker
    finishes the prompt (``finish_prompt``)."""
    import types

    import onnxruntime

    import torch

    torch.set_num_threads(2)  # torch defaults to one thread per core: a dozen such processes oversubscribe the host
    global _FEAT
    h = types.SimpleNamespace()
    if gpu_tokens:
        import s3tokenizer

        h.audio_tokenizer = s3tokenizer.load_model(f"{model_dir}/assets/token2wav/speech_tokenizer_v2_25hz.onnx").cuda().eval()
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = spk_threads
    # every voice has its own length: ONNX Runtime's per-shape memory patterns would grow the process with each new one
    # (the per-shape FFT plans of the front end are bounded and filled before load: _warm_shape_caches).  The CPU arena
    # stays on: it holds a job's working set as a fixed pool (without it every job returns ~40 MB to the OS and takes it
    # back, and RSS swings +-3% with the sampling instant: results/engine_dev/r7_memory); it keeps the peak of the
    # longest voice seen, so it is filled before load with a voice of ARENA_WARM_S (_warm_arena)
    opt.enable_mem_pattern = False
    h.spk_model = onnxruntime.InferenceSession(f"{model_dir}/assets/token2wav/campplus.onnx", sess_options=opt, providers=["CPUExecutionProvider"])
    _FEAT = h


_RESAMPLE: dict = {}


def prompt_front_end(h, prompt_wav: str) -> tuple:
    """The CPU part of ``stepaudio2.Token2wav._prepare_prompt``, step for step (same inputs, same ops): the S3 log-mel
    (with its length), the CAMPPlus embedding (a future of the numpy array ONNX returns) and the 24 kHz prompt mel before the
    final replicate padding.  ``finish_prompt`` completes it on the GPU.  The 16 -> 24 kHz ``Resample`` transform is
    built once per rate (the reference rebuilds the same kernel every call)."""
    import s3tokenizer
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

    from concurrent.futures import ThreadPoolExecutor

    global _SPK_POOL
    if "_SPK_POOL" not in globals():
        _SPK_POOL = ThreadPoolExecutor(1, thread_name_prefix="campplus")
    audio = s3tokenizer.load_audio(prompt_wav, sr=16000)
    spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
    # CAMPPlus (ONNX Runtime releases the GIL) overlaps the S3 mel and tokens and the 24 kHz mel; resolved last
    spk = _SPK_POOL.submit(lambda: h.spk_model.run(None, {h.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()})[0])
    mels = s3tokenizer.log_mel_spectrogram(audio)
    mels, mels_lens = s3tokenizer.padding([mels])
    audio, sample_rate = torchaudio.load(prompt_wav, backend="soundfile")
    audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 24000:
        if sample_rate not in _RESAMPLE:
            _RESAMPLE[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
        audio = _RESAMPLE[sample_rate](audio)
    prompt_mel = mel_spectrogram(audio).transpose(1, 2).squeeze(0)
    return mels, mels_lens, spk, prompt_mel


def finish_prompt(audio_tokenizer, front: tuple, up_rate: int = 2) -> tuple:
    """The GPU part: S3 tokens of the log-mel (batch 1, as the reference), the CAMPPlus embedding and the prompt mel on
    the GPU, the mel replicate-padded to ``up_rate`` x the token count.  Equal to ``Token2wav._prepare_prompt``."""
    import torch

    mels, mels_lens, spk, prompt_mel = front
    tokens, tokens_lens = audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())
    spk_emb = torch.tensor(spk.result() if hasattr(spk, "result") else spk, device="cuda")
    prompt_mels = prompt_mel.unsqueeze(0).cuda()
    prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device="cuda")
    prompt_mels = torch.nn.functional.pad(prompt_mels, (0, 0, 0, tokens.shape[1] * up_rate - prompt_mels.shape[1]), mode="replicate")
    return tokens, tokens_lens, spk_emb, prompt_mels, prompt_mels_lens


def _feat_job(ref: bytes):
    """The CPU front end of ``Token2Wav.features`` in a feature process (the HF 16 kHz PCM16 path)."""
    import io
    import tempfile

    import librosa
    import soundfile as sf
    import torch

    def run():
        wav16, _ = librosa.load(io.BytesIO(ref), sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, torch.inference_mode():
            sf.write(f.name, wav16, 16000)  # PCM_16, as in init_token2wav_cache
            front = prompt_front_end(_FEAT, f.name)
            if getattr(_FEAT, "audio_tokenizer", None) is None:  # CPU only: resolve the embedding before it crosses processes
                return front[:2] + (front[2].result(),) + front[3:]
            return ("done",) + tuple(x.cpu() for x in finish_prompt(_FEAT.audio_tokenizer, front))  # the whole prompt, on the CPU

    try:
        out = run()  # every temporary of the job (audio, mels, futures, GPU tensors) is released when run() returns
    finally:
        _trim_heap()  # so the trim sees the job's freed memory, not memory its locals still hold
    return out


def _trim_heap() -> None:
    """Give freed heap back to the OS after a job: jobs of varied audio sizes otherwise leave glibc holding the peak
    (feature-process RSS crept up ~3% over a 30-minute soak, slowing but not flat)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


JEMALLOC = "/lib/x86_64-linux-gnu/libjemalloc.so.2"  # the feature processes' allocator (system package libjemalloc2)
SETUP_FRAME_BUDGET = 16 * 512  # rows x frames per eager batched setup call (its new caches: ~2.7 GB in FP16)


SHAPE_WARM_LENGTHS = 1200  # distinct input lengths: above the per-shape FFT caches' capacity (~750 entries measured)


def _warm_shape_caches(n: int = SHAPE_WARM_LENGTHS, seed: int = 0) -> None:
    """Run the front end over ``n`` distinct input lengths of 1.5-25 s before load, so per-shape state is created up
    front.  Round 6 attributed the per-length growth (~36 MB per process, plateauing after ~750 lengths) to FFT plans;
    round 7 found raw FFTs do not grow and the owner is oneDNN's primitive cache (1024 shape-keyed entries; the
    resampler's conv1d), now disabled in the feature processes (ONEDNN_PRIMITIVE_CACHE_CAPACITY=0).  The warm-up is kept
    because the measured flat configuration includes it.  Numerics are unaffected (the same ops)."""
    import numpy as np
    import s3tokenizer
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

    rng = np.random.default_rng(seed)
    if 16000 not in _RESAMPLE:
        _RESAMPLE[16000] = torchaudio.transforms.Resample(orig_freq=16000, new_freq=24000)
    for n16 in np.linspace(int(1.5 * 16000), int(25 * 16000), n).astype(int):
        audio = torch.from_numpy(rng.standard_normal(int(n16)).astype(np.float32) * 0.1)
        with torch.inference_mode():
            kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
            s3tokenizer.log_mel_spectrogram(audio)
            mel_spectrogram(_RESAMPLE[16000](audio.unsqueeze(0)))
    _trim_heap()


ARENA_WARM_S = 60.0  # the reference length the CAMPPlus arena is sized for before load (longer voices still extend it)


def _warm_arena(seconds: float = ARENA_WARM_S, seed: int = 1) -> None:
    """Run the speaker model once on a ``seconds`` voice so its CPU arena reaches that working set before load."""
    import numpy as np
    import torch
    import torchaudio.compliance.kaldi as kaldi

    audio = torch.from_numpy(np.random.default_rng(seed).standard_normal(int(seconds * 16000)).astype(np.float32) * 0.1)
    feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    feat = feat - feat.mean(dim=0, keepdim=True)
    _FEAT.spk_model.run(None, {_FEAT.spk_model.get_inputs()[0].name: feat.unsqueeze(0).numpy()})
    del audio, feat
    _trim_heap()


def _warm_jobs(n: int, seed: int = 2) -> None:
    """``n`` whole feature jobs on synthetic voices (noise, 1.5-25 s, 24 kHz WAV) before load, so the heap reaches the
    layout steady load leaves it in before the first measurement."""
    import io

    import numpy as np
    import soundfile as sf

    rng = np.random.default_rng(seed)
    for n24 in rng.integers(int(1.5 * 24000), 25 * 24000, n):
        b = io.BytesIO()
        sf.write(b, (rng.standard_normal(int(n24)) * 0.05).astype(np.float32), 24000, format="WAV", subtype="PCM_16")
        _feat_job(b.getvalue())


def _feat_warm(ref: bytes):
    """Warm a feature process before load (the per-shape FFT caches, the speaker model's arena at ARENA_WARM_S, then two
    real jobs); the job also holds the process for a while, so the pool starts every process (each runs one)."""
    _warm_shape_caches()
    _warm_arena()
    _warm_jobs(int(os.environ.get("FEATURE_WARM_JOBS", "0")))
    _feat_job(ref)
    time.sleep(2.0)
    return _feat_job(ref)


@dataclass
class _Req:
    seq: int
    noise_seed: int
    features: object = None  # Future
    stream: object = None
    windows: deque = field(default_factory=deque)  # (tokens, last, index, t_ready)
    offset: int = 0
    t_open: float = 0.0
    feat: tuple | None = None  # the resolved prompt features (on this process's stream), once ready
    t_feat: float = 0.0  # when the loop saw them ready (setups are taken oldest first)
    audio: AudioState = field(default_factory=AudioState)  # first-audio state and playback clock (t2w_sched)


def setup_calls(ready: list, setup_t, row_limit: dict, frame_budget: int = SETUP_FRAME_BUDGET) -> list:
    """Setup calls for the requests whose features are ready (oldest first), batched per frame bucket: small groups
    ride the setup graphs (row limits); larger ones run as fewer, bigger eager calls bounded by a frame budget (they are
    compute-bound, and one call beats many for every request's first audio)."""
    groups: dict[int, list] = {}
    for r in ready:
        groups.setdefault(next((b for b in setup_t if b >= r.feat[3].shape[1]), -r.feat[3].shape[1]), []).append(r)
    calls = []
    for b, items in groups.items():
        per = max(1, row_limit.get(b, 2) // 2)
        if len(items) > per:
            per = max(per, frame_budget // (2 * max(abs(b), 1)))
        calls += [(b, items[k:k + per]) for k in range(0, len(items), per)]
    return calls


class WorkerLoop:
    """The worker's per-request state and scheduling over injected parts, so the production loop runs in CPU tests.

    ``bt``: setup_many / new_stream / release_prompt / release_stream / decode (``BatchedToken2Wav``); ``emit``: one
    message to the engine; ``resolve``: a done feature future -> prompt features (raises for unusable audio);
    ``calls_for``: ready requests, oldest first -> [(bucket, requests)] setup calls; ``clock``: the worker clock.

    mode "edf" (task17, the default): work-conserving -- every ready window is decoded in the tick it is ready, earliest
    playback deadline first (``Policy.max_decode_rows`` may cap a call); a setup never runs ahead of a waiting
    continuation window (decode first, then at most ``Policy.setup_calls_when_busy`` setup calls); setups oldest first.
    mode "legacy" (rounds 4-5, kept for the scheduler comparison): every pending setup first, then a decode held until
    ``min_batch`` windows wait, a first or last window is among them, or the oldest waited ``max_wait``."""

    def __init__(self, bt, emit, resolve, calls_for, policy: Policy, mode: str = "edf", min_batch: int = 8, max_wait: float = 0.04,
                 clock=time.perf_counter, log=None, note=None):
        if mode not in ("edf", "legacy"):
            raise ValueError(f"t2w_policy must be 'edf' or 'legacy', not {mode!r}")
        self.bt, self.emit, self.resolve, self.calls_for, self.policy, self.mode = bt, emit, resolve, calls_for, policy, mode
        self.min_batch, self.max_wait, self.clock = min_batch, max_wait, clock
        self.log = log or (lambda kind, n, t0: None)
        self.note = note
        self.reqs: dict[int, _Req] = {}

    # messages from the engine
    def open(self, seq: int, seed: int, fut) -> None:
        self.reqs[seq] = _Req(seq, seed, features=fut, t_open=self.clock())

    def window(self, seq: int, tokens, last: bool, index: int) -> None:
        if seq in self.reqs:
            self.reqs[seq].windows.append((tokens, last, index, self.clock()))

    def close(self, seq: int) -> None:
        """A request leaves the worker: its stream's cache slots go back (fast DiT)."""
        r = self.reqs.pop(seq, None)
        if r is not None and r.stream is not None:
            self.bt.release_stream(r.stream)

    # one scheduling tick
    def tick(self) -> bool:
        self._resolve_ready()
        ready = sorted((r for r in self.reqs.values() if r.stream is None and r.feat is not None), key=lambda r: r.t_feat)
        calls = [(part[0].t_feat, (b, part)) for b, part in self.calls_for(ready)]
        if self.mode == "legacy":
            worked = self._setups([c for _, c in calls])
            return self._decode(gate=True) or worked
        waiting = any(r.stream is not None and r.windows and r.audio.started for r in self.reqs.values())
        decode_first, run = plan_tick(waiting, calls, self.policy)
        worked = self._decode() if decode_first else False
        worked = self._setups(run) or worked
        if not decode_first:
            worked = self._decode() or worked
        return worked

    def _resolve_ready(self) -> None:
        for r in [r for r in self.reqs.values() if r.stream is None and r.feat is None and r.features.done()]:
            try:
                f = self.resolve(r.features)
            except Exception as e:
                self.emit(("error", r.seq, f"reference audio: {type(e).__name__}: {e}", "bad_request"))  # an input error, typed across IPC
                self.close(r.seq)
                continue
            r.feat, r.t_feat, r.features = f, self.clock(), None  # the future is not kept once resolved
            if self.note is not None:
                self.note(f"features_ready 1 {(r.t_feat - r.t_open) * 1000:.1f}")

    def _setups(self, calls: list) -> bool:
        for b, part in calls:
            try:
                t0 = time.perf_counter()
                prompts = self.bt.setup_many([r.feat for r in part])
                self.log(f"setup@{b}", len(part), t0)
            except Exception as e:
                for r in part:
                    self.emit(("error", r.seq, f"prompt setup: {type(e).__name__}: {e}"))
                    self.close(r.seq)
                continue
            for r, p in zip(part, prompts):
                r.feat = None
                if self.reqs.get(r.seq) is r:
                    r.stream = self.bt.new_stream(p, noise_seed=r.noise_seed, adopt=True)
                else:  # closed while its setup was pending
                    self.bt.release_prompt(p)
        return bool(calls)

    def _decode(self, gate: bool = False) -> bool:
        """One decode call over the ready head windows (earliest playback deadline first); False if nothing ran."""
        ready = [r for r in self.reqs.values() if r.stream is not None and r.windows]
        if not ready:
            return False
        now = self.clock()
        if gate:
            heads = [r.windows[0] for r in ready]
            if not (any(h[2] == 0 for h in heads) or any(h[1] for h in heads) or len(ready) >= min(self.min_batch, len(self.reqs))
                    or now - min(h[3] for h in heads) >= self.max_wait):
                return False
        dl = {r.seq: r.audio.deadline(r.windows[0][3], self.policy) for r in ready}
        sel = [self.reqs[k] for k in choose_windows([(d, k) for k, d in dl.items()], self.policy)]
        if self.note is not None:
            self.note(f"slack {len(sel)} {(min(dl[r.seq] for r in sel) - now) * 1000:.1f}")
        items = []
        for r in sel:
            tokens, last, _, _ = r.windows.popleft()
            items.append((r.stream, tokens, last))
        try:
            t0 = time.perf_counter()
            wavs = self.bt.decode(items)
            self.log("decode", len(items), t0)
        except Exception as e:
            for r in sel:
                self.emit(("error", r.seq, f"token2wav: {type(e).__name__}: {e}"))
                self.close(r.seq)
            return True
        sent = self.clock()
        for r, (_, _, last), wav in zip(sel, items, wavs):
            if len(wav):
                self.emit(("audio", r.seq, r.offset, wav))
                r.offset += len(wav)
                r.audio.sent(len(wav), sent)
            if last:
                self.emit(("done", r.seq))
                self.close(r.seq)
        return True


def worker_main(inq, outq, cfg: dict) -> None:
    """The worker process: a fatal error is written out (flushed), reported to the engine (``fatal``: its audio requests
    fail instead of waiting) and ends the process without waiting on the feature processes."""
    import sys
    import traceback

    try:
        _worker_main(inq, outq, cfg)
    except BaseException:
        tb = traceback.format_exc()
        sys.stderr.write("token2wav worker failed:\n" + tb)
        sys.stderr.flush()
        try:
            outq.put(("fatal", None, tb[-2000:]))
            time.sleep(1.0)  # let the queue's feeder thread deliver it
        finally:
            os._exit(1)


def _worker_main(inq, outq, cfg: dict) -> None:
    import torch

    from engine.models.token2wav import Token2Wav
    from engine.runtime.dit_graph import ROW_LIMIT, SETUP_T
    from engine.runtime.token2wav_batch import BatchedToken2Wav

    torch.backends.cudnn.allow_tf32 = False  # AC-3.4: Token2Wav against FP32
    model_dir = cfg["model_dir"]
    t2w = Token2Wav(f"{model_dir}/assets/token2wav", float16=cfg.get("float16_flow", False), spk_threads=cfg.get("spk_threads", 8))
    bt = BatchedToken2Wav(t2w, use_graphs=cfg.get("t2w_graphs", False), compile_estimator=cfg.get("t2w_compile", False),
                          euler_graphs=cfg.get("t2w_euler_graphs", False), fast_dit=cfg.get("t2w_fast_dit", False),
                          fast_slots=cfg.get("t2w_fast_slots", 49152))
    if cfg.get("t2w_graphs"):
        with torch.inference_mode():
            p0 = bt.setup(t2w.features(f"{model_dir}/assets/HT_ref_audio.wav"))
            bt.capture_graphs(p0)
            bt.release_prompt(p0)
    if cfg.get("feature_processes", 0):  # features in their own processes (own GILs): parallel under load
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor

        # CPU-only processes (no CUDA context); a few malloc arenas each, so varied audio sizes do not keep growing the heap.
        # oneDNN's primitive cache (1024 shape-keyed entries; the resampler's conv1d is planned per input length) is off:
        # its content followed the recent mix of voice lengths, so RSS rose and then decayed by ~1-2% whenever the mix
        # changed; off, it costs nothing measurable and the features are bit-identical.  A fixed mmap threshold keeps
        # large job buffers out of the heap (glibc's dynamic threshold otherwise moves them in, and the heap creeps).
        # Measured: results/engine_dev/r7_memory (tools/spike/feature_scenario_rss.py over the robustness sequence).
        # The allocator is jemalloc: with glibc (even with the fixed threshold) per-process RSS still crept ~+0.3 MB per
        # 1000 jobs for the whole run (1.07-1.15% at the soak's length), with jemalloc it plateaus (0.58-0.90% over 32,000
        # jobs, bit-identical features, the same job time).  A missing library fails the start rather than silently
        # running another allocator.
        os.environ.setdefault("MALLOC_ARENA_MAX", str(cfg.get("feature_malloc_arenas", 2)))
        os.environ.setdefault("ONEDNN_PRIMITIVE_CACHE_CAPACITY", str(cfg.get("feature_onednn_cache", 0)))
        os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", str(cfg.get("feature_mmap_threshold", 131072)))
        alloc = cfg.get("feature_allocator", "jemalloc")
        if alloc == "jemalloc":
            lib = cfg.get("feature_jemalloc_path", JEMALLOC)
            if not os.path.isfile(lib):
                raise RuntimeError(f"feature_allocator jemalloc: {lib} not found")
            os.environ["LD_PRELOAD"] = lib  # the spawned feature processes only; this process keeps its allocator
        elif alloc != "glibc":
            raise ValueError(f"feature_allocator must be 'jemalloc' or 'glibc', not {alloc!r}")
        pool = ProcessPoolExecutor(cfg["feature_processes"], mp_context=mp.get_context("spawn"), initializer=_feat_init,
                                   initargs=(model_dir, cfg.get("spk_threads", 8), cfg.get("feature_gpu_tokens", True)))
        ref = open(f"{model_dir}/assets/HT_ref_audio.wav", "rb").read()
        for f in [pool.submit(_feat_warm, ref) for _ in range(cfg["feature_processes"])]:
            f.result()  # every process started, its models loaded and its kernels warm
    else:
        pool = ThreadPoolExecutor(cfg.get("feature_workers", 4), thread_name_prefix="t2w-features")
    # S3 tokens of the reference voices: one thread with its own CUDA stream, so they overlap the decode loop
    from concurrent.futures import Future

    tok_pool = ThreadPoolExecutor(1, thread_name_prefix="s3-tokens")
    tok_stream = torch.cuda.Stream()

    def _tokens(front):
        with torch.cuda.stream(tok_stream), torch.inference_mode():
            out = finish_prompt(t2w.ref.audio_tokenizer, front)
        tok_stream.synchronize()
        return out

    def open_features(ref: bytes) -> Future:
        """CPU front end in a feature process, then S3 tokens on the GPU in this process (one future for both)."""
        out: Future = Future()

        def gpu_done(g):
            e = g.exception()
            out.set_exception(e) if e is not None else out.set_result(g.result())
            wake.set()

        def cpu_done(f):
            e = f.exception()
            if e is not None:
                out.set_exception(e)
                wake.set()
                return
            res = f.result()
            if res[0] == "done":  # the feature process ran the S3 tokens itself
                out.set_result(tuple(x.cuda(non_blocking=True) for x in res[1:]))
                wake.set()
                return
            tok_pool.submit(_tokens, res).add_done_callback(gpu_done)

        pool.submit(_feat_job, ref).add_done_callback(cpu_done)
        return out

    policy = Policy(setup_calls_when_busy=cfg.get("t2w_setup_calls_when_busy", 1), max_decode_rows=cfg.get("t2w_max_decode_rows"),
                    first_slack_s=cfg.get("t2w_first_slack_ms", 250.0) / 1000)

    def resolve(fut):
        """A done feature future -> the prompt features on this process's stream (raises for unusable audio)."""
        if cfg.get("feature_processes", 0):  # finished on the S3 thread's stream: record them on this one
            f = fut.result()
            for x in f:
                x.record_stream(torch.cuda.current_stream())
            return f
        return tuple(x.cuda(non_blocking=True) if x.device.type == "cpu" else x for x in fut.result())

    wake = threading.Event()
    msgs: deque = deque()
    prof_path = cfg.get("profile_path")
    prof = open(prof_path, "a") if prof_path else None

    def log(kind: str, n: int, t0: float) -> None:
        if prof is not None:
            torch.cuda.current_stream().synchronize()
            prof.write(f"{time.time():.4f} {kind} {n} {(time.perf_counter() - t0) * 1000:.1f}\n")
            prof.flush()

    def feats(ref: bytes):
        t0 = time.perf_counter()
        f = t2w.features(ref)
        torch.cuda.current_stream().synchronize()
        log("features", 1, t0)
        wake.set()
        return f

    def note(line: str) -> None:
        if prof is not None:
            prof.write(f"{time.time():.4f} {line}\n")

    loop = WorkerLoop(bt, outq.put, resolve, lambda ready: setup_calls(ready, SETUP_T, ROW_LIMIT), policy, mode=cfg.get("t2w_policy", "edf"),
                      min_batch=cfg.get("t2w_min_batch", 8), max_wait=cfg.get("t2w_max_wait_ms", 40.0) / 1000, log=log, note=note)
    reqs = loop.reqs

    def reader():
        while True:
            m = inq.get()
            msgs.append(m)
            wake.set()
            if m[0] == "stop":
                return

    threading.Thread(target=reader, daemon=True).start()

    def parent_watch(ppid: int):  # the engine died without a stop message (SIGKILL / SIGTERM to its PID only)
        import os

        while True:
            time.sleep(1.0)
            if os.getppid() != ppid:
                msgs.append(("stop",))
                wake.set()
                time.sleep(5)
                os._exit(0)

    threading.Thread(target=parent_watch, args=(os.getppid(),), daemon=True).start()
    outq.put(("ready",))
    torch.cuda.set_device(0)
    last_stats = 0.0

    def feature_rss() -> dict:
        """RSS of each feature process by PID (None when unreadable); the total is None unless every process was read."""
        n = cfg.get("feature_processes", 0)
        if not n:
            return {"feature_processes": 0, "feature_rss_by_pid": {}, "feature_rss_kib": None}
        by_pid = {str(p): _rss_kib(p) for p in (getattr(pool, "_processes", None) or {})}
        whole = len(by_pid) == n and all(v is not None for v in by_pid.values())
        return {"feature_processes": n, "feature_rss_by_pid": by_pid, "feature_rss_kib": sum(by_pid.values()) if whole else None,
                "feature_allocator": cfg.get("feature_allocator", "jemalloc")}

    while True:
        now_s = time.monotonic()
        if now_s - last_stats >= 1.0:  # counters for the engine's /metrics (leak checks after a drain)
            last_stats = now_s
            fast = bt.fast
            outq.put(("stats", None, {"t2w_requests": len(reqs), "t2w_streams": sum(r.stream is not None for r in reqs.values()),
                                      "t2w_cache_slots_used": (fast.pool.in_use() - fast.scratch) if fast is not None else None,
                                      "t2w_cnn_rows_free": len(fast.cnn_free) if fast is not None else None,
                                      "t2w_gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
                                      "t2w_gpu_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
                                      "t2w_dit_graph_calls": fast.counts["graph"] if fast is not None else None,
                                      "t2w_dit_eager_calls": fast.counts["eager"] if fast is not None else None,
                                      "t2w_rss_kib": _rss_kib(os.getpid()), **feature_rss(),
                                      "t2w_stats_time": time.time()}))  # producer time: the checker rejects stale counters
        while msgs:
            m = msgs.popleft()
            kind = m[0]
            if kind == "stop":
                if hasattr(pool, "shutdown"):
                    pool.shutdown(wait=False, cancel_futures=True)
                return
            if kind == "open":
                _, seq, ref, seed = m
                loop.open(seq, seed, open_features(ref) if cfg.get("feature_processes", 0) else pool.submit(feats, ref))
            elif kind == "window":
                loop.window(*m[1:])
            elif kind == "close":
                loop.close(m[1])
        with torch.inference_mode():
            worked = loop.tick()
        if not worked and not msgs:
            wake.wait(0.002)
            wake.clear()


class Token2WavProcess:
    """The engine-side handle: start the worker, send messages, deliver its answers to callbacks."""

    def __init__(self, cfg: dict, on_message):
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        self.inq, self.outq = ctx.Queue(), ctx.Queue()
        # not daemonic: the worker starts its own feature processes; an exit hook stops it with the engine
        self.proc = ctx.Process(target=worker_main, args=(self.inq, self.outq, cfg), daemon=False, name="token2wav-worker")
        import os

        pct = cfg.get("mps_sm_percent")  # under MPS: cap the worker's SM share so the Thinker / Talker kernels are not queued
        old = os.environ.get("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
        if pct:
            os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(int(pct))  # the spawned worker inherits the environment
        try:
            self.proc.start()
        finally:
            if pct:
                if old is None:
                    os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)
                else:
                    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = old
        import atexit

        atexit.register(self.close)
        deadline = time.time() + 1800
        while True:  # wait for "ready", but fail at once if the worker dies during start-up
            try:
                first = self.outq.get(timeout=5)
                break
            except queue.Empty:
                if not self.proc.is_alive():
                    raise RuntimeError(f"token2wav worker exited during start-up (code {self.proc.exitcode})")
                if time.time() > deadline:
                    self.proc.kill()
                    raise RuntimeError("token2wav worker start-up timed out")
        if first != ("ready",):
            raise RuntimeError(f"token2wav worker failed to start: {first}")
        self.on_message = on_message
        threading.Thread(target=self._recv, daemon=True, name="t2w-recv").start()

    def send(self, *msg) -> None:
        self.inq.put(msg)

    def _recv(self) -> None:
        while True:
            try:
                m = self.outq.get()
            except (EOFError, OSError):
                return
            self.on_message(m)

    def close(self) -> None:
        try:
            self.inq.put(("stop",))
        except Exception:
            pass
        self.proc.join(timeout=10)
        if self.proc.is_alive():
            self.proc.kill()
