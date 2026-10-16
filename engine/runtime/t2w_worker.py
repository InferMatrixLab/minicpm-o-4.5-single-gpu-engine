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

    try:
        wav16, _ = librosa.load(io.BytesIO(ref), sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(suffix=".wav") as f, torch.inference_mode():
            sf.write(f.name, wav16, 16000)  # PCM_16, as in init_token2wav_cache
            front = prompt_front_end(_FEAT, f.name)
            if getattr(_FEAT, "audio_tokenizer", None) is None:  # CPU only: resolve the embedding before it crosses processes
                return front[:2] + (front[2].result(),) + front[3:]
            return ("done",) + tuple(x.cpu() for x in finish_prompt(_FEAT.audio_tokenizer, front))  # the whole prompt, on the CPU
    finally:
        _trim_heap()


def _trim_heap() -> None:
    """Give freed heap back to the OS after a job: jobs of varied audio sizes otherwise leave glibc holding the peak
    (feature-process RSS crept up ~3% over a 30-minute soak, slowing but not flat)."""
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


SETUP_FRAME_BUDGET = 16 * 512  # rows x frames per eager batched setup call (its new caches: ~2.7 GB in FP16)


def _feat_warm(ref: bytes):
    """A warm-up job that holds its process for a while, so the pool starts every process (each runs one).  (A process's
    native memory still grows by ~35 MB over its first ~200 real voices and then stays flat; warming with a 30 s voice or
    150 synthetic voices of 1.5-20 s did not prevent it, so neither is done.)"""
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

        # CPU-only processes (no CUDA context); a few malloc arenas each, so varied audio sizes do not keep growing the heap
        os.environ.setdefault("MALLOC_ARENA_MAX", str(cfg.get("feature_malloc_arenas", 2)))
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

    min_batch, max_wait = cfg.get("t2w_min_batch", 8), cfg.get("t2w_max_wait_ms", 40.0) / 1000
    reqs: dict[int, _Req] = {}

    def drop(seq: int) -> None:  # a request leaves the worker: its stream's cache slots go back (fast DiT)
        r = reqs.pop(seq, None)
        if r is not None and r.stream is not None:
            bt.release_stream(r.stream)
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
    while True:
        now_s = time.monotonic()
        if now_s - last_stats >= 1.0:  # counters for the engine's /metrics (leak checks after a drain)
            last_stats = now_s
            fast = bt.fast
            outq.put(("stats", None, {"t2w_requests": len(reqs), "t2w_streams": sum(r.stream is not None for r in reqs.values()),
                                      "t2w_cache_slots_used": (fast.pool.in_use() - fast.scratch) if fast is not None else None,
                                      "t2w_cnn_rows_free": len(fast.cnn_free) if fast is not None else None,
                                      "t2w_gpu_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
                                      "t2w_dit_graph_calls": fast.counts["graph"] if fast is not None else None,
                                      "t2w_dit_eager_calls": fast.counts["eager"] if fast is not None else None,
                                      "t2w_rss_kib": _rss_kib(os.getpid()),
                                      "feature_rss_kib": sum(_rss_kib(p) or 0 for p in getattr(pool, "_processes", {}) or {}) or None}))
        while msgs:
            m = msgs.popleft()
            kind = m[0]
            if kind == "stop":
                if hasattr(pool, "shutdown"):
                    pool.shutdown(wait=False, cancel_futures=True)
                return
            if kind == "open":
                _, seq, ref, seed = m
                if cfg.get("feature_processes", 0):
                    fut = open_features(ref)
                else:
                    fut = pool.submit(feats, ref)
                reqs[seq] = _Req(seq, seed, features=fut, t_open=time.perf_counter())
            elif kind == "window":
                _, seq, tokens, last, index = m
                if seq in reqs:
                    reqs[seq].windows.append((tokens, last, index, time.perf_counter()))
            elif kind == "close":
                drop(m[1])
        worked = False
        with torch.inference_mode():
            # setups of every voice whose features are ready, batched per frame bucket
            todo = [r for r in reqs.values() if r.stream is None and r.features.done()]
            groups: dict[int, list] = {}
            for r in todo:
                try:
                    if cfg.get("feature_processes", 0):  # finished on the S3 thread's stream: record them on this one
                        f = r.features.result()
                        for x in f:
                            x.record_stream(torch.cuda.current_stream())
                    else:
                        f = tuple(x.cuda(non_blocking=True) if x.device.type == "cpu" else x for x in r.features.result())
                except Exception as e:
                    outq.put(("error", r.seq, f"reference audio: {type(e).__name__}: {e}", "bad_request"))  # an input error, typed across IPC
                    drop(r.seq)
                    continue
                groups.setdefault(next((b for b in SETUP_T if b >= f[3].shape[1]), -f[3].shape[1]), []).append((r, f))
                if prof is not None:
                    prof.write(f"{time.time():.4f} features_ready 1 {(time.perf_counter() - r.t_open) * 1000:.1f}\n")
            for b, items in groups.items():
                # small groups ride the setup graphs (row limits); larger ones run as fewer, bigger eager calls bounded
                # by a frame budget (they are compute-bound, and one call beats many for every request's first audio)
                per = max(1, ROW_LIMIT.get(b, 2) // 2)
                if len(items) > per:
                    per = max(per, SETUP_FRAME_BUDGET // (2 * max(abs(b), 1)))
                for k in range(0, len(items), per):
                    part = items[k:k + per]
                    try:
                        t0 = time.perf_counter()
                        prompts = bt.setup_many([f for _, f in part])
                        log(f"setup@{b}", len(part), t0)
                    except Exception as e:
                        for r, _ in part:
                            outq.put(("error", r.seq, f"prompt setup: {type(e).__name__}: {e}"))
                            drop(r.seq)
                        continue
                    for (r, _), p in zip(part, prompts):
                        if r.seq in reqs:
                            r.stream = bt.new_stream(p, noise_seed=r.noise_seed, adopt=True)
                        else:  # closed while its setup ran
                            bt.release_prompt(p)
                    worked = True
            ready = [r for r in reqs.values() if r.stream is not None and r.windows]
            if ready:
                now = time.perf_counter()
                heads = [r.windows[0] for r in ready]
                go = (any(h[2] == 0 for h in heads) or any(h[1] for h in heads) or len(ready) >= min(min_batch, len(reqs))
                      or now - min(h[3] for h in heads) >= max_wait)
                if go:
                    items = []
                    for r in ready:
                        tokens, last, _, _ = r.windows.popleft()
                        items.append((r.stream, tokens, last))
                    try:
                        t0 = time.perf_counter()
                        wavs = bt.decode(items)
                        log("decode", len(items), t0)
                    except Exception as e:
                        for r in ready:
                            outq.put(("error", r.seq, f"token2wav: {type(e).__name__}: {e}"))
                            drop(r.seq)
                        wavs = []
                    for r, (_, _, last), wav in zip(ready, items, wavs):
                        if len(wav):
                            outq.put(("audio", r.seq, r.offset, wav))
                            r.offset += len(wav)
                        if last:
                            outq.put(("done", r.seq))
                            drop(r.seq)
                    worked = True
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
