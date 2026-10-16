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

import queue
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field


_FEAT = None  # per feature process: an object with what ``Token2wav._prepare_prompt`` reads


def _feat_init(model_dir: str, spk_threads: int) -> None:
    """Feature-process initializer: the S3 tokenizer (GPU) and CAMPPlus (CPU ONNX) only, no flow."""
    import types

    import onnxruntime
    import s3tokenizer

    import torch

    torch.set_num_threads(2)  # torch defaults to one thread per core: a dozen such processes oversubscribe the host
    global _FEAT
    h = types.SimpleNamespace()
    h.audio_tokenizer = s3tokenizer.load_model(f"{model_dir}/assets/token2wav/speech_tokenizer_v2_25hz.onnx").cuda().eval()
    opt = onnxruntime.SessionOptions()
    opt.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    opt.intra_op_num_threads = spk_threads
    h.spk_model = onnxruntime.InferenceSession(f"{model_dir}/assets/token2wav/campplus.onnx", sess_options=opt, providers=["CPUExecutionProvider"])
    h.flow = types.SimpleNamespace(up_rate=2)  # the flow's token-to-mel rate (flow.yaml), all _prepare_prompt reads of it
    _FEAT = h


_RESAMPLE: dict = {}


def prepare_prompt_overlapped(h, prompt_wav: str):
    """``stepaudio2.Token2wav._prepare_prompt`` step for step (same inputs, same ops, identical outputs), with the CPU
    CAMPPlus run (ONNX Runtime releases the GIL) overlapping the S3 tokenizer and the 24 kHz mel, and the 16 -> 24 kHz
    ``Resample`` transform built once per rate (the reference rebuilds the same kernel every call)."""
    from concurrent.futures import ThreadPoolExecutor

    import s3tokenizer
    import torch
    import torchaudio
    import torchaudio.compliance.kaldi as kaldi
    from stepaudio2.flashcosyvoice.utils.audio import mel_spectrogram

    global _SPK_POOL
    if "_SPK_POOL" not in globals():
        _SPK_POOL = ThreadPoolExecutor(1, thread_name_prefix="campplus")
    audio = s3tokenizer.load_audio(prompt_wav, sr=16000)
    spk_feat = kaldi.fbank(audio.unsqueeze(0), num_mel_bins=80, dither=0, sample_frequency=16000)
    spk_feat = spk_feat - spk_feat.mean(dim=0, keepdim=True)
    spk = _SPK_POOL.submit(h.spk_model.run, None, {h.spk_model.get_inputs()[0].name: spk_feat.unsqueeze(dim=0).cpu().numpy()})
    mels = s3tokenizer.log_mel_spectrogram(audio)
    mels, mels_lens = s3tokenizer.padding([mels])
    tokens, tokens_lens = h.audio_tokenizer.quantize(mels.cuda(), mels_lens.cuda())
    audio, sample_rate = torchaudio.load(prompt_wav, backend="soundfile")
    audio = audio.mean(dim=0, keepdim=True)
    if sample_rate != 24000:
        if sample_rate not in _RESAMPLE:
            _RESAMPLE[sample_rate] = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=24000)
        audio = _RESAMPLE[sample_rate](audio)
    prompt_mels = mel_spectrogram(audio).transpose(1, 2).squeeze(0).unsqueeze(0).cuda()
    prompt_mels_lens = torch.tensor([prompt_mels.shape[1]], dtype=torch.int32, device="cuda")
    prompt_mels = torch.nn.functional.pad(prompt_mels, (0, 0, 0, tokens.shape[1] * h.flow.up_rate - prompt_mels.shape[1]), mode="replicate")
    spk_emb = torch.tensor(spk.result()[0], device="cuda")
    return tokens, tokens_lens, spk_emb, prompt_mels, prompt_mels_lens


def _feat_job(ref: bytes):
    """``Token2Wav.features`` in a feature process; tensors come back on the CPU."""
    import io
    import tempfile

    import librosa
    import soundfile as sf
    import torch

    wav16, _ = librosa.load(io.BytesIO(ref), sr=16000, mono=True)
    with tempfile.NamedTemporaryFile(suffix=".wav") as f, torch.inference_mode():
        sf.write(f.name, wav16, 16000)  # PCM_16, as in init_token2wav_cache
        out = prepare_prompt_overlapped(_FEAT, f.name)
    return tuple(x.cpu() for x in out)


SETUP_FRAME_BUDGET = 16 * 512  # rows x frames per eager batched setup call (its new caches: ~2.7 GB in FP16)


def _feat_warm(ref: bytes):
    """A warm-up job that holds its process for a while, so the pool starts every process (each runs one)."""
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

        pool = ProcessPoolExecutor(cfg["feature_processes"], mp_context=mp.get_context("spawn"), initializer=_feat_init,
                                   initargs=(model_dir, cfg.get("spk_threads", 8)))
        ref = open(f"{model_dir}/assets/HT_ref_audio.wav", "rb").read()
        for f in [pool.submit(_feat_warm, ref) for _ in range(cfg["feature_processes"])]:
            f.result()  # every process started, its models loaded and its kernels warm
    else:
        pool = ThreadPoolExecutor(cfg.get("feature_workers", 4), thread_name_prefix="t2w-features")
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

    import os

    threading.Thread(target=parent_watch, args=(os.getppid(),), daemon=True).start()
    outq.put(("ready",))
    torch.cuda.set_device(0)
    while True:
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
                    fut = pool.submit(_feat_job, ref)
                    fut.add_done_callback(lambda _f: wake.set())
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
                    f = tuple(x.cuda(non_blocking=True) if x.device.type == "cpu" else x for x in r.features.result())
                except Exception as e:
                    outq.put(("error", r.seq, f"reference audio: {type(e).__name__}: {e}"))
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
