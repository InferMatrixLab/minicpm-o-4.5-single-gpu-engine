# Runtime architecture

This describes the Python runtime at the 2026-10-19 snapshot. Source features, defaults and measured profiles are
distinct, and the all-concurrency target has not passed. The per-component parity evidence is in [engine.md](engine.md);
the process-layout measurements are in [architecture_decision.md](architecture_decision.md).

![MiniCPM-o 4.5 single-GPU architecture](figures/architecture.svg)

## Module ownership

| Module | Responsibility |
|---|---|
| [server.py](../engine/server.py) | aiohttp `/v1/chat/completions` (streamed and whole responses), request validation, disconnect cancellation, `/health`, `/info` (served identity), `/metrics`; listens on loopback |
| [backend_s.py](../engine/runtime/backend_s.py) | Track S engine loop: the single GPU-owner thread, admission, KV lifetime commitments, the scheduler tick, Thinker chunks, Talker conditioning and the hand-off to the worker |
| [t2w_sched.py](../engine/runtime/t2w_sched.py) | Playback clock (the harness stall model), first-chunk lane, prefill and Talker-start selection |
| [paged.py](../engine/runtime/paged.py) | Page pools per model, ragged batches and FlashInfer prefill/decode wrappers |
| [thinker_decode.py](../engine/runtime/thinker_decode.py) / [talker_decode.py](../engine/runtime/talker_decode.py) | CUDA-graph decoders per batch bucket; the Talker graph runs K frames per call with device-side sampling |
| [prefill_graph.py](../engine/runtime/prefill_graph.py) | Graph-captured Thinker prefills and Talker chunk prefills (per row count and token bucket) |
| [kernels.py](../engine/runtime/kernels.py) | Triton fused residual add + RMSNorm, q/k norm + RoPE + v copy and SiLU-multiply, reproducing the reference rounding |
| [hostcopy.py](../engine/runtime/hostcopy.py) | Queued host-to-device copies without stream synchronization, and ordered FlashInfer `plan` calls per wrapper |
| [models/thinker.py](../engine/models/thinker.py), [models/talker.py](../engine/models/talker.py) | Qwen3-8B and the 20-layer Llama Talker on the checkpoint's `llm.*` and `tts.*` weights |
| [models/multimodal.py](../engine/models/multimodal.py), [media.py](../engine/media.py) | The checkpoint processor, SigLIP NaViT + resampler, Whisper + projection; image, audio and video decoding |
| [talker_sampler.py](../engine/talker_sampler.py) | HF-order Talker sampling (repetition penalty, temperature, top-k, top-p), counter-based per-request RNG, min/max length, EOS freeze |
| [t2w_worker.py](../engine/runtime/t2w_worker.py) | Token2Wav worker process: window scheduler, prompt setups, feature-process pool, audio return |
| [token2wav_batch.py](../engine/runtime/token2wav_batch.py), [dit_fast.py](../engine/runtime/dit_fast.py), [dit_graph.py](../engine/runtime/dit_graph.py), [hift_graph.py](../engine/runtime/hift_graph.py) | Batched streaming Token2Wav: flow encoder, fast DiT solve with slot caches, HiFT, and their CUDA graphs |
| [models/token2wav.py](../engine/models/token2wav.py), [token2wav_chunker.py](../engine/token2wav_chunker.py) | Per-stream reference decode and the window chunker of `streaming_generate` |
| [stage_trace.py](../engine/runtime/stage_trace.py) | Stage spans on CUDA events with a completeness watermark, read by [tools/engine_profile.py](../tools/engine_profile.py) |
| [bench](../bench) | Streaming client, adapters for the three servers, metrics, protocol, artifact verification and statistics ([bench/README.md](../bench/README.md)) |
| [evals](../evals) | Speech quality (WER, speaker similarity, UTMOS, detectors) and multimodal accuracy suites with pinned evaluators |
| [tools](../tools) | Profiling, robustness, comparison reports, HF reference generation (`tools/hfref`) and component probes (`tools/spike`) |

## Request and state lifecycle

1. The API validates the messages, the modalities, the voice prompt (`ref_audio`: a data URL, base64 audio or a path)
   and the sampling fields. Audio output without a voice prompt is rejected with HTTP 400, as is a prompt longer than the
   served context (8192 tokens), which is never truncated.
2. The engine builds the prompt with the checkpoint processor (token IDs equal to the HF path) and admits the request
   only when the Thinker pool can hold its whole lifetime (prompt plus the chunk-rounded reply) beside the running
   requests. Otherwise it waits in admission order. The worker starts the voice prompt's features at once.
3. Each tick runs every decoding row. Pending Thinker prefills run oldest first within 2048 tokens; a longer prompt is
   prefilled in budget-sized chunks. Talker starts are granted in admission order, each after its whole Talker KV
   lifetime is committed, so a continuing utterance never waits for KV and a full pool cannot deadlock.
4. Track S: every 10 Thinker tokens the reply chunk's hidden states condition the Talker in place, without a copy
   across processes. The Talker decodes K=4 frames per graph replay. Sampling, repetition history and EOS stay on the
   GPU. The Talker KV is rolled back at a chunk's EOS, as in the reference `streaming_generate`.
5. Codec tokens go to the worker as 25-token windows (the first after 8 frames). The worker decodes ready windows
   earliest playback deadline first, batched across streams: flow encoder, DiT solve (10 Euler steps), HiFT.
   New voices' prompt setups run first while every continuing stream keeps 100 ms of playback slack.
6. Audio returns to the engine as 24 kHz PCM16 and is streamed with its sample offset. A request completes after its
   last window; cancellation frees its KV pages, Talker commitment, worker state and feature job.

Readiness follows weight loading, CUDA graph capture, Token2Wav compilation and the feature-process warmup (2000
synthetic jobs each, so their memory is flat during load). Startup is outside the measured requests.

## Processes

| Process | GPU memory (L20X) | Why it exists |
|---|---|---|
| Engine: API, Thinker, Talker | ~57 GB | weights, paged KV and graphs, one GPU-owner thread |
| Token2Wav worker | ~33 GB | flow and HiFT with a 65k-slot cache pool. In the engine process its launch-heavy thread contends for the GIL with the tick; as its own process under MPS, C=8 rose from 374–419 to 525–699 frames/s |
| 4 feature processes | ~1.3 GB each | CAMPPlus (CPU ONNX) and the S3 tokenizer per voice, outside the worker's GIL. 4 match 12 and 24 is no faster; S3 in a worker thread was slower |
| CUDA MPS server | <0.1 GB | lets the engine and the worker run kernels concurrently; without it the two contexts time-slice and the tick slows 2–4× |

## Defaults versus measured profiles

`scripts/serve_engine.sh <config>` loads `configs/engine/<config>.json`; nothing else changes the engine's behaviour.
Options that are off in `track_s_fast` are not part of the qualified path.

| Setting | `track_s_fast` | `track_s_fast_r11` (measured profile) |
|---|---|---|
| Engine loop | Track S, `max_live` 64, Thinker 1024 × 64-token pages, Talker 1280 × 128-token pages | same |
| Thinker / Talker decode | CUDA graphs, fused Triton kernels, Talker K=4 | same |
| Talker chunk prefill | eager | CUDA graphs with batched first-frame sampling (`talker_prefill_graphs`) |
| First chunk | 8 frames (reference 25) | same |
| Flow | FP16, fast DiT with slot caches and graphs | same |
| HiFT | strict FP32, graphs | upsampling stages in TF32 (`t2w_hift_decode_tf32`) |
| Prompt setup encoder | one prompt at a time | batched per call (`t2w_batch_setup_encoder`) |
| Scheduling | first-chunk lane 100 ms, prefill budget 2048, no slack floor | same |
| Worker / features | worker process under MPS, 4 feature processes | same |

Neutral or negative options remain in the source, off: asynchronous Token2Wav results (`t2w_async_results`), HiFT
buckets above 16 rows (`t2w_hift_rows`) and whole-K Talker capture (`talker_whole_k`, profile `track_s_fast_r14`).
See [optimizations.md](optimizations.md).

`track_s_fast_r11` combines TF32 HiFT with the batched setup encoder. Together they exceed the HiFT chunk error bound
on 1 of 190 chunks (1.06e-2 against 1e-2; `tests/engine/test_hift_tf32.py[rows_to_16_batched_setup]` is a strict
expected failure). Each option alone stays within its bound. The profile's performance figures stand; its numerical
qualification does not.

## External code and weights

The checkpoint (`openbmb/MiniCPM-o-4_5@503e754`) provides the weights, the processor and the encoder code, all
Apache-2.0. The engine imports them; one method is adapted for transformers 5.x (`MiniCPMWhisperEncoderLayer.forward`,
in `engine/models/multimodal.py`). `minicpmo-utils` 1.0.6 (`stepaudio2`, Apache-2.0) provides the reference Token2Wav,
CosyVoice2 flow and HiFT modules. The engine's batched and graph-captured paths are tested against them. Track S is a
port of `MiniCPMO.streaming_generate`.

FlashInfer, Triton and PyTorch provide the GPU operators. No vLLM or SGLang engine, scheduler or serving pipeline runs
inside the engine; those two run only as the baselines. Parity references come from the model card's stack in a
separate environment (`engine/env/hfref.lock`, `tools/hfref/`).
