# MiniCPM-o 4.5 engine: components and reference points

The engine is a single-GPU server (`scripts/serve_engine.sh`, `engine/server.py`). It is built from the checkpoint
`openbmb/MiniCPM-o-4_5@503e754` and serves the OpenAI-compatible `/v1/chat/completions` API with streamed text and
audio.

## Components

| Component | Module | Reference | Parity evidence |
|---|---|---|---|
| Thinker (Qwen3-8B), reference precision | `engine/models/thinker.py` | transformers `Qwen3ForCausalLM` on the `llm.*` weights | `tests/engine/test_thinker_parity.py`: exact in fp32 on CPU (min cosine at least 0.99999, all argmax equal). GPU bf16: `tests/engine/test_thinker_gpu_parity.py` passes the user-approved noise-relative AC-3.1 rule (no further from HF bf16 eager than HF's own SDPA path), not the literal thresholds |
| Talker (20-layer Llama TTS), conditioning | `engine/models/talker.py` | `MiniCPMTTS`, `_generate_speech_non_streaming` | `tests/engine/test_talker_parity.py`: conditioning formula, teacher-forced codec logits (relative L2 below 1e-4), hidden/token misalignment negative |
| Talker sampler | `engine/talker_sampler.py` | `gen_logits`, `MiniCPMTTS.generate` (HF order) | `tests/engine/test_talker_sampler.py`: transformers warpers, frame-0 rule, min/max EOS, counter RNG, K-step, EOS freeze, padding isolation, swapped-order negative |
| Token2Wav stream chunker | `engine/token2wav_chunker.py` | `streaming_generate` stream driver | `tests/engine/test_token2wav_chunker.py`: windows identical to the reference loop; missing silence prefix or lookahead are detected |
| Token2Wav flow + HiFT stream decode | `engine/models/token2wav.py` | `stepaudio2.Token2wav` (minicpmo-utils 1.0.6) | `tests/engine/test_token2wav_stream.py` (GPU): against the FP32 reference (cuDNN TF32 off) with the HiFT source noise seeded per request and chunk, <= 1e-4 |
| Multimodal prompt assembly and encoders | `engine/models/multimodal.py`, `engine/media.py` | checkpoint processor, SigLIP and resampler, Whisper and projection | `tests/engine/test_multimodal_frontend.py`: 32 mixed samples, token IDs identical to the HF environment, embeddings with cosine at least 0.999 in fp32 on CPU, dropped-placeholder negative. GPU bf16 against HF bf16: images identical, audio min cosine 0.9969 (below 0.999; open). The request's `mm_processor_kwargs` (the Daily-Omni frame/slice policy) reach the processor in both backends, and a prompt longer than the context (`tdec.max_len`, 8192 tokens) is rejected with HTTP 400 rather than truncated; long Daily-Omni prompts (~13k tokens) therefore still need a larger served context (open) |
| Paged KV and ragged batch forward | `engine/runtime/paged.py` | the reference ports | `tests/engine/test_paged_runtime.py` (GPU) |
| Talker K-frame graph decode | `engine/runtime/talker_decode.py` | eager per-frame path | `tests/engine/test_talker_decode.py` (GPU) |
| Track T engine loop | `engine/runtime/backend.py` | HF `chat(generate_audio=True)` conditioning path | engine effective-input dump (`bench/effective_dump.py engine`) and the speech quality gates |
| API | `engine/server.py` | AC-7 | `tests/engine_api/`: harness adapter, unchanged vLLM native client, 400/503, disconnect cancellation |
| Track S engine loop | `engine/runtime/backend_s.py` | `engine/runtime/track_s.py` (HF `streaming_generate` port) | Thinker chunks per the reference yield rules (`tests/engine/test_track_s_chunks.py`), Talker KV rolled back at chunk EOS; smoke `tools/engine_smoke.py` |
| Batched Token2Wav | `engine/runtime/token2wav_batch.py` | per-stream `Token2Wav.decode` | batched == per-stream (ragged prompts, mixed stages, merged final windows), batched setups == reference `setup_cache` (`tests/engine/test_token2wav_stream.py`) |
| Token2Wav graphs | `engine/runtime/dit_graph.py`, `engine/runtime/hift_graph.py` | eager batched path | mel 1e-4 (FP32), AC-3.4 FP16 bound for FP16/compiled; HiFT graphs == eager on equal mel |
| Token2Wav worker process | `engine/runtime/t2w_worker.py` | in-process Token2Wav thread | same `BatchedToken2Wav`; requires MPS (see `docs/architecture_decision.md`) |
| Graph decoders | `engine/runtime/thinker_decode.py`, `engine/runtime/talker_decode.py` | eager per-step paths | graph == eager exactly, with plans skipped between page crossings (`tests/engine/test_{thinker,talker}_decode.py`) |
| Fused Thinker decode kernels | `engine/runtime/kernels.py` (Triton: residual add + norm, q/k norm + RoPE + v copy, SiLU-and-multiply) | the torch reference-rounding ops (`hf_rms_norm`, `hf_rope`) | every reference rounding reproduced with integer round-to-nearest-even (`tests/engine/test_kernels.py`); fused vs torch decode: same greedy tokens (`tests/engine/test_thinker_decode.py`); AC-3.1 greedy runs through this decoder (`tests/engine/test_thinker_gpu_parity.py`) |
| Fast DiT solve (headline `track_s_fast`) | `engine/runtime/dit_fast.py`, `engine/runtime/dit_kernels.py` | `BatchedToken2Wav._solve` (reference DiT module) | attention caches as page-size-1 slots with the reference trimming as slot bookkeeping, packed ragged rows, CUDA graphs per (rows, tokens) bucket: FP16 mels within the AC-3.4 FP16 bound for setups, staggered streams, ramped first windows and merged finals; every slot returned (`tests/engine/test_dit_fast.py`, `tests/engine/test_token2wav_stream.py::test_fast_dit_matches_eager`) |
| TF32 HiFT upsampling (option `t2w_hift_decode_tf32`, candidate `track_s_fast_hift`) | `engine/models/token2wav.py::decoder_tf32`, `_hift_decode(tf32=True)` | strict-FP32 HiFT, same seeded source noise | only the upsampling stages (`ups`, `source_downs`/`source_resblocks`, `resblocks`) in TF32; the f0 predictor, sine source and input/output convolutions stay strict FP32 (TF32 there moves the waveform by ~34%). Served graph path, 32 utterances of real speech tokens, 190 chunks: max chunk rel-L2 5.3e-3, max utterance 1.75e-3, strict-FP32 run-to-run 3.9e-6; AC-3.4 HiFT bound 1e-2 (`tests/engine/test_hift_tf32.py`) |
| Batched prompt-setup encoder (option `t2w_batch_setup_encoder`, candidate `track_s_fast_r11`) | `engine/runtime/token2wav_batch.py::_setup_encoder_batched` | per-prompt reference `forward_chunk` (`_setup_encoder_one`) | the prompts of one `setup_many` call in one right-padded flow-encoder batch: attention keys past each prompt masked, conv caches gathered at each prompt's real end, attention caches cut to its length (the conformer layers have no conv module; the pre-lookahead conv reads only each prompt's own 3 silence tokens). 16 voices of 80-158 tokens in calls of 2-8: encoder output and caches within rel-L2 3.2e-4 of the per-prompt forward (per-prompt run to run bitwise 0; `tests/engine/test_setup_batched.py`); served chunks of 32 batched setups against the FP32 reference: max mel rel-L2 3.71e-3 (per-prompt setups 4.23e-3), AC-3.4 FP16 bound 5e-3 (`tests/engine/test_ac34_fp32.py::...[setup_batched]`) |
| Talker chunk-prefill graphs (option `talker_prefill_graphs`, candidate `track_s_fast_r11`) | `engine/runtime/prefill_graph.py::ChunkPrefillGraphs`, `backend_s._talker_step` | eager `DecoderRunner.forward` of the same rows; per-row first-frame sampling | CUDA graphs per row count (1-4) and token bucket (32/64/128) of a prefill appending to existing sequences (a scratch sequence holds the padding; one reserved page), eager beyond them; the first frames of one call sampled in one `step` with one host transfer. Real Talker weights, prior contexts of 1-300 tokens, 14 calls: most calls bitwise equal to eager, the rest within the noise of prefilling the same rows together vs one at a time (hidden/K-V/logits); batched first-frame sampling bitwise equal to per-row (`tests/engine/test_talker_prefill_graph.py`) |
| Host-to-device copies (every config, 3c6aebe + plan ordering e73aced) | `engine/runtime/hostcopy.py` and its call sites (fast DiT plan, flow, setup encoder, paged layout, prefill graphs, graph decoders, backend) | `torch.tensor(..., device="cuda")` and device-side FlashInfer plan inputs | queued copies and host plan inputs carry the same values; FlashInfer plans of one wrapper wait for the previous plan's upload (`hostcopy.plan`; without it, a plan-buffer race at 3c6aebe). Setups bitwise equal to 1fab3f0, decoded windows within the decode's own run-to-run spread (max rel-L2 2.7e-5; `tools/spike/t2w_decode_digest.py`); graph decoders, prefill graphs, fast DiT, stream and AC-3.4 suites pass (`tests/engine/`) |
| Prompt features (worker) | `engine/runtime/t2w_worker.py::prompt_front_end` + `finish_prompt` | `stepaudio2.Token2wav._prepare_prompt` | identical tensors (`tests/engine/test_token2wav_stream.py::test_split_prompt_features_equal_reference`). WAVs are decoded with soundfile (`_load_wav`) where the reference calls `torchaudio.load`: torchaudio 2.11 always decodes through TorchCodec, which grows the process by ~0.6 KB per call; the tensors are identical (`tests/engine_api/test_t2w_sched.py::test_load_wav_equals_torchaudio_load`) |

## Scheduling (task17) and robustness evidence (AC-6)

- **Playback clock** (`engine/runtime/t2w_sched.py AudioState`): the harness stall model. A stream's buffer runs out at `max(previous exhaustion, delivery) + seconds delivered`; the worker and the engine keep one per request, from the audio they hand on.
- **Token2Wav worker** (`engine/runtime/t2w_worker.py WorkerLoop`, `t2w_policy`):
  - `"edf"` is work-conserving: every ready window is decoded in the tick it becomes ready, earliest playback deadline first (`t2w_max_decode_rows` may cap a call).
  - First-chunk lane (`t2w_first_lane_slack_ms`): pending prompt setups (new requests' first audio) run before the decode while every waiting continuation window keeps that much playback slack, rechecked before each setup call. Otherwise the tick decodes first and then runs at most `t2w_setup_calls_when_busy` setups. `null` gives round 7's decode-first rule.
  - Setups run oldest first.
  - `"r6"` is the named round-6 control: decode first, then bounded setups, with the batching gate and round 6's clock (first audio + seconds sent). `tests/engine_api/test_t2w_worker_loop.py::test_r6_control_makes_the_round6_decisions` checks it against the round-6 policy module recovered from git.
  - `"legacy"` is rounds 4–5: every setup first, then a decode held for `t2w_min_batch` windows or `t2w_max_wait_ms`.
- **Engine tick** (`backend_s.TrackSBackend._tick` / `_prefill_lane`, `t2w_sched.choose_prefills` / `choose_starts`): the bounded first-chunk lane.
  - Every decoding row runs every tick.
  - Pending Thinker prefills run oldest first within a hard `prefill_token_budget`. Aged requests count against it too. The oldest always progresses: a prompt that alone exceeds the budget is prefilled in budget-sized chunks, with its progress kept (`LiveS.prefilled`).
  - A request is granted Thinker KV only while the pool holds its whole lifetime (prompt + chunk-rounded reply, `thinker_bound`) beside what running requests and the decoder's padding rows may still take. Requests are granted in admission order, so the others wait (backpressure) rather than exhaust the pool.
  - Talker KV (B22) is committed for an utterance's whole lifetime before its first chunk starts. The bound is `talker_lifetime`: every chunk's condition tokens and carried frame, the request's frame cap (`talker_sampling.max_new_tokens`, default 2048, the canonical request's value) and one K-frame block.
    - First starts are granted in admission order beside every commitment and the decoder's bucket padding. The first one that does not fit holds the later ones, so continuation chunks never wait for KV and a full pool cannot deadlock.
    - A request whose bound exceeds the Talker's 4096 positions or the whole pool is rejected at admission (HTTP 400). A layout beyond a request's commitment fails that request alone.
    - At the frame cap the utterance ends (`finish_reason` "length").
    - `track_s_fast` has 1280 Talker pages of 128 tokens: 64 canonical commitments (19 pages each) plus 63 padding pages.
  - While a continuing stream has less than `prefill_slack_floor_ms` of buffered playback, only prefills that have waited `prefill_max_age_ms` run, and they run after that tick's Thinker decode and Talker step.
  - Talker chunk starts run earliest playback deadline first within `talker_start_budget` tokens per tick, at least one. A first chunk's deadline is its admission plus `prefill_max_age_ms`.
  - `null` budget and floor give the rounds 4–6 behaviour (every pending prefill each tick); the KV check stays.
  - `/metrics` publishes `prefill_token_budget`, `prefill_tokens_tick_max`, `prefill_chunked_ticks`, `prefill_kv_wait_ticks`, `talker_starts_deferred_ticks`, `talker_kv_wait_ticks`, `talker_kv_committed_pages` and `talker_kv_pad_pages`.
- **Tests** (CPU):
  - `tests/engine_api/test_t2w_sched.py`, `test_t2w_worker_loop.py` (the production loop with a fake model and clock) and `test_prefill_lane.py`;
  - `test_engine_tick.py`: the actual `_tick` with timed fake stage runners and the real KV accounting. It covers aged long prefills beside continuing speech (and the same workload unbounded, which stalls), KV backpressure, repeated arrivals, cancellation during a chunked prefill, and Talker start order;
  - `test_talker_capacity.py`: the real `_tick` and `_talker_step` on the real `PagedKV`. It covers the review's counterexample: 64 utterances growing past 1024 tokens on the 512-page pool exhaust it without commitments, and complete with them. It also covers 32 utterances past 2048 tokens, admission order, the per-chunk carry, the frame cap, cancellation and reuse, rejection at admission, and a broken bound failing alone.
- **Measured, round 7 (diagnostic; L20X, K=5, 7be6f4a; `tools/sched_compare.py` over `results/engine_sweeps/r7_sched_cmp_gpu{2,3}`)**: round 7's policy against `track_s_fast_legacy` (rounds 4–5).
  - Throughput and RTF are within ±2.5% at every C.
  - The mean run-max stall is lower: at C=32, 0.16 s against 0.27 s (absolute 0.20 against 0.34 s).
  - TTFA latency is higher from C=4 up: p95 +6.5% at C=4, +9.5% at C=8, +19.6% at C=16 and +8.7% at C=32; p50 +11% at C=32.
- **Round 8 ablation (tuning IDs, K=3, C=4/16/32, 55af971; `tools/sched_select.py` over `results/engine_sweeps/r8_sched_abl_gpu{2,3}`)**: oriented point ratios against `track_s_fast_legacy` (>1 better). The rule fixed in advance chose `r8_abl_f`, now frozen as `track_s_fast`: the first-chunk lane at 100 ms of slack and no prefill slack floor.
  - F (lane 100 ms, no floor): minimum ratio 0.997 over the 12 cells. At C=32 the TTFA p50/p95 ratios are 1.021/1.005. C=32 mean run-max stall 0.100 s (absolute 0.134 s), against the legacy policy's 0.212 s (0.243 s).
  - D (decode first, no floor): minimum 0.880, at C=32 TTFA p95 (+13.6% latency); stall 0.103 s.
  - C (decode first, floor 300 ms): minimum 0.857, at C=16 TTFA p95 (+16.7%); stall 0.114 s.
  - E (lane 100 ms, floor 100 ms): minimum 0.980; stall 0.147 s.
  - B (round 6 control): minimum 0.846, at C=32 TTFA p95 (+18.2%); stall 0.134 s.
  - Attribution: the first-chunk lane recovers the C=32 TTFA (D to F: p95 ratio 0.880 to 1.005). The slack floor costs TTFA at C=16 (D to C: p95 1.007 to 0.857) and adds stall with the lane on (F to E: 0.100 to 0.147 s). Throughput and RTF are within ±2.5% for every variant.
  - K=3 is below the development minimum for reported comparisons; the frozen policy is measured against the round 6 control at K=5 on the evaluation IDs (`sweeps/engine/r8_sched_cmp_gpu{2,3}.json`).
- **Round 8 frozen policy against the round 6 control (evaluation IDs, K=5, 5db059a; `tools/sched_compare.py r8-frozen r6-control` over `results/engine_sweeps/r8_sched_cmp_gpu{2,3}`, all 12 cells verify)**:
  - C=32: TTFA p95 -11.4% (LCB 1.089), p50 -5.1% (LCB 1.045), throughput +1.4%, RTF -1.9%; mean run-max stall 0.109 s against 0.142 s.
  - C=16: TTFA p95 -6.2% (LCB 1.035).
  - C=1/2/4/8: TTFA changes from -4.0% (C=2 p50) to +1.6% (C=8 p50); throughput from -1.7% (C=4, LCB 0.971) to +0.9%.
- **Round 8 headline measurement (evaluation IDs, K=5, 028998b = 9ed19a1 plus the sweep specs; `sweeps/engine/r8_head_gpu{2,3}.json`, `tools/engine_vs_envelope.py`, `results/engine_sweeps/r8_head_gpu{2,3}/vs_envelope.json`)**: point ratio / one-sided 95% LCB against the best baseline cell at each C (>1 better).

  | C | Throughput | TTFA p50 | TTFA p95 | RTF mean |
  |---|---|---|---|---|
  | 1 | 1.342 / 1.335 | **2.224 / 2.212** | **2.480 / 2.429** | 1.380 / 1.373 |
  | 2 | 1.083 / 1.073 | 1.905 / 1.861 | 1.946 / 1.899 | 1.106 / 1.095 |
  | 4 | 0.984 / 0.976 | **2.134 / 2.115** | **2.529 / 2.484** | 1.003 / 0.996 |
  | 8 | 1.020 / 1.010 | **2.213 / 2.186** | **2.595 / 2.522** | 1.041 / 1.030 |
  | 16 | 1.015 / 0.996 | 2.010 / 1.958 | **2.754 / 2.655** | 1.033 / 1.014 |
  | 32 | 1.034 / 1.026 | 1.349 / 1.313 | **2.616 / 2.546** | 1.050 / 1.042 |

  - 8 of the 24 ratios have LCB >= 2.0, all of them TTFA; C=16 p50 is inconclusive (point above 2, LCB below). No throughput or RTF ratio comes near 2. This is development evidence on L20X, not the AC-5 result (H200, K=10).
  - Against the 5db059a frozen cells (`tools/sched_compare.py r8-head r8-frozen`), the memory repairs leave performance unchanged: every point ratio is within 1% except C=32 TTFA p95 (-4.5% latency, LCB 1.009). The C=32 mean run-max stall is 0.140 s against 0.109 s.
- **Round 11 attribution: kernel activity and hardware counters** (`tools/engine_profile.py` with `tools/kernel_attribution.py` and `tools/gpm_sampler.py`; the round-10 bullet below is corrected by it).
  - Method:
    - Every GPU process of the engine (engine, Token2Wav worker, 4 feature processes) captures its CUDA activity (CUPTI) over a 1.5 s kernel window per C, aligned to the host clock by marker kernels (measured offset bound at most +-9 us).
    - NVML GPM counters are sampled over the same windows.
    - Each kernel is joined to the innermost stage span of its launching thread. The CUPTI runtime records also give the host time spent in blocking calls.
    - The first-audio path of every request is traced (engine queue, features, setup, prefill, first text chunk, Talker to window 0, transport each way, worker wait, decode).
    - Run identity (commit, dirty files, config digest, interpreter, GPU UUID), the GPU's process inventory per window (no foreign process in any run) and incomplete-trace rejection (`tests/engine_api/test_kernel_attribution.py`).
  - Measurement cost (one server at a time, `track_s_fast_hift`, tuning IDs; `results/engine_profile/r11_ctrl_*`, `r11_bisect_*`):

    | C | untraced fps | spans + request timelines | + idle window threads | + CUPTI captures |
    |---|---|---|---|---|
    | 1 | 384 | 374 | 373 | 350 |
    | 16 | 1544 | 1518 | 1528 | 1231 |
    | 32 | 2272 | 2225 | 2199 | 1906 |

    The spans cost 2-3%; CUPTI costs 9-20% (a ~2 s start, then slower launches for the rest of the process). Host and elapsed times are therefore taken from spans-only runs, kernel times, kernel activity and GPM from the CUPTI runs.
  - Finding (`track_s_fast_hift`, `results/engine_profile/r11_base`, CUPTI run): the GPU is not saturated. Device-wide some kernel runs 71% / 82% / 81% / 79% of the time at C = 1 / 8 / 16 / 32, but SM utilization is only 35% / 50% / 52% / 51%, occupancy 6-15%, DRAM bandwidth 13-16% and tensor pipes 1.5-5%. The kernels are small and latency-bound (the engine launches 77-140k kernels per second, ~400 per Talker frame).
  - The engine thread is host-bound. Spans-only at C=32: a 34.8 ms tick (28.6/s) holds ~18 ms of kernels; the largest gaps per tick are the Talker chunk prefill (13.75 ms host vs 3.2 ms of kernels, ~1000 eagerly launched kernels a call), the Talker's per-row bookkeeping (~4 ms), the K-step decode (9.8 ms host vs 6.6 ms kernels) and the Thinker prefill (22 ms host vs 12 ms kernels). Launch-to-start delays are ~10 us: kernels do not queue on the GPU.
  - The Token2Wav worker is host-bound too: at C=32 decode calls (6.5/s x 113 ms) and setups (9/s x 29 ms) fill its whole second; half of a decode call's host time is waiting in blocking calls (each `torch.tensor(..., device="cuda")` synchronizes the stream: 15 per flow call, 22 ms), the rest Python and launches.
  - The first audio at C=32 (spans-only, p50; client TTFA 686 ms): prefill 23 ms, first text chunk 343 ms (11 Thinker steps at the tick rate), Talker to window 0 57 ms, window 0 waiting for the worker loop 70 ms, worker queue 26 ms, decode call 121 ms; the stream's setup is ready 153 ms before window 0 (features 156 ms, setup call 27 ms), so the setup is off the first-audio path. At C=1 (TTFA 133 ms): prefill 11, first text chunk 60, Talker to window 0 27, decode call 30 ms.
  - Levers implemented by this ranking (candidate `track_s_fast_r11`, see the component table): the batched prompt-setup encoder and the Talker chunk-prefill graphs with batched first-frame sampling. Spans-only, same tool, tuning IDs (`results/engine_profile/r11_cand_spans` vs `r11_bisect_spans`): C=1 374 -> 380 fps, C=16 1518 -> 1603, C=32 2225 -> 2382 (+7.1%); TTFA p50 C=32 0.688 -> 0.662 s. The Talker prefill drops from 13.75 to 7.35 ms host, the tick from 34.8 to 29.6 ms and the first text chunk at C=32 from 343 to 283 ms; the worker is then the limit (decode 5.7/s x 132 ms + setup 8.8/s x 27 ms per second; window 0 waits 88 ms for the worker loop, 48 ms in its queue and 133 ms in the decode call, together +52 ms).
- **Round 11 measurement, `track_s_fast_r11`** (`track_s_fast_hift` plus `t2w_batch_setup_encoder` and `talker_prefill_graphs`; evaluation IDs, K=5, 1fab3f0; `sweeps/engine/r11_gpu{2,3}.json`, `results/engine_sweeps/r11_gpu{2,3}`; all 30 runs valid, no failed request, six cells verified from artifacts in `results/engine_sweeps/r11_verify.log`). 10 of 24 ratios have LCB >= 2.0 against the envelope, all of them TTFA (round 10: 9; C=2 TTFA p95 now passes at 2.046).
  - Against r10-hift (`tools/sched_compare.py r11 r10-hift`, `results/engine_sweeps/r11_vs_r10.log`; separate sessions, diagnostic):
    - Throughput rises at every C with every LCB above 1: C=1 +3.7%, C=2 +4.5%, C=4 +5.1%, C=8 +6.8%, C=16 +6.1%, C=32 +4.0% (2353 vs 2262 fps).
    - RTF mean falls 3.4-6.2% at every C.
    - TTFA improves at C=1, 2 and 32 (C=32 p50 -5.3%, p95 -8.1%), is unchanged at C=4, and is worse at C=8 (p50 +1.9%, p95 +2.8%, LCB 0.97) and in the C=16 p95 (+2.5%): the worker, not the engine tick, now sets those first-audio paths.
  - Envelope ratios, point / LCB (`results/engine_sweeps/r11_envelope.log`):

    | C | throughput | RTF | TTFA p50 | TTFA p95 |
    |---|---|---|---|---|
    | 1 | 1.419/1.411 | 1.459/1.451 | 2.340/2.325 pass | 2.573/2.517 pass |
    | 2 | 1.148/1.138 | 1.171/1.160 | 1.977/1.931 | 2.092/2.046 pass |
    | 4 | 1.049/1.038 | 1.070/1.058 | 2.185/2.163 pass | 2.605/2.562 pass |
    | 8 | 1.108/1.094 | 1.128/1.113 | 2.291/2.257 pass | 2.647/2.574 pass |
    | 16 | 1.117/1.097 | 1.132/1.111 | 2.137/2.089 pass | 2.871/2.764 pass |
    | 32 | 1.179/1.168 | 1.194/1.183 | 1.574/1.527 | 2.924/2.864 pass |

  - Gap to the goal (2.0 / LCB): throughput and RTF need 1.4-1.9x more at every C; TTFA p50 C=32 needs 1.31x, C=2 1.04x.
  - Not yet a headline change: the candidate needs the AC-4 quality qualification.
- **Host-to-device copies without waiting** (3c6aebe, every config). `engine.runtime.hostcopy` replaces `torch.tensor(..., device="cuda")` (a pageable copy, then a stream synchronization) by queued copies. FlashInfer plans get host index pointers and last-page lengths (given device tensors, `plan` copies them back, one synchronization each). Applied to the fast DiT plan, the flow window and setup-encoder lengths, the paged layout, the prefill graphs, the graph decoders and the backend's per-request tensors; the Talker's per-row state after a K-step decode becomes a view instead of five clones.
  - Same values. Setups are bitwise equal to 1fab3f0. Decoded windows stay within the decode's existing run-to-run spread: 1fab3f0 against itself gives 13/36 windows bitwise equal, max rel-L2 2.7e-5; 3c6aebe against 1fab3f0 gives max 2.7e-5 (`tools/spike/t2w_decode_digest.py`, `results/engine_profile/r11_dump_*`).
  - Served, spans-only, parallel runs of the two commits (`results/engine_profile/r11_sync_{before,after}`): C=1 379.5 -> 390.6 fps, C=8 1078 -> 1094, C=16 1612 -> 1648, C=32 2374 -> 2472 (+4.1%); TTFA p50 C=32 669 -> 638 ms.
    - The worker's flow call drops from 30.9 to 7.3 ms of host time and its setup call from 28 to 22.5 ms.
    - The decode call stays at ~134 ms: it now waits once, at its final device-to-host copy, for GPU work that runs beside the engine's under MPS.
  - **Regression found and fixed (e73aced).**
    - The formal six-C sweep of 3c6aebe (`results/engine_sweeps/r11b_gpu{2,3}`, kept as recorded) failed: 13 of 30 measured runs had failed requests, and four of six cells do not verify (`r11b_verify.log`). The cause was CUDA illegal memory accesses in the engine.
    - Mechanism: FlashInfer's `plan` writes the plan into the wrapper's page-locked host buffer and uploads it asynchronously. The device-side plan inputs of the old code made every `plan` synchronize first. With host inputs, a second plan of the same wrapper (a page crossing inside a Talker K-step, two DiT solves in one bucket) could overwrite the buffer before the first upload ran, and the replay read the wrong plan.
    - Fix: `engine.runtime.hostcopy.plan` makes each plan wait for its predecessor's upload only (an event recorded after it); every plan call goes through it.
    - Stress test, untraced, C=16/32 alternating, 2 min windows, the two commits in parallel (`results/engine_profile/r11_stress_*`): 3c6aebe crashes in the first window (illegal memory access; 158,476 of 158,519 requests fail). e73aced serves 8,832 requests with none invalid, at C=16 1693-1706 fps and C=32 2486-2534 fps.
    - The GPU suites pass as at 1fab3f0: 35 passed, and the 2 failures that predate this round fail identically there. The formal six-C sweep of e73aced is `r11c` (below).
    - Formal sweep `r11c` (e73aced code, six C, K=5, evaluation IDs; `results/engine_sweeps/r11c_gpu{2,3}`): 30 runs valid, no failed request, six cells verify (`r11c_verify.log`). Against r11 (`r11c_vs_r11.log`): throughput +0.5% / +0.8% / +2.3% / +2.5% / +3.0% / +3.6% at C = 1/2/4/8/16/32 (2437 vs 2353 fps at C=32; LCB > 1 except C=1, 0.999); TTFA p50 -1.0..-3.0% at C=2..32, unchanged at C=1. Envelope (`r11c_envelope.log`): still 10 of 24 LCB >= 2.0 (all TTFA); throughput LCB 1.061-1.420, RTF 1.083-1.460; failing TTFA p50 C=2 1.997/1.951 and C=32 1.590/1.550.
    - The crashed run also showed two queued operations gaps. The worker's feature processes outlived it (Q1/worker death). The engine server ignored SIGTERM after the error, so the profiler's launcher kill left it running (Q6). Both were stopped by exact PID.
- **Remaining levers, by the round-11 measurements:**
  - The worker idles during that wait. Running setups and the next call's host work while a decode call's GPU work is in flight (asynchronous result copy) is the next worker lever.
  - About half of the engine tick is still host gaps (engine kernel-active 0.46 at C=32 in the CUPTI run): the Talker's per-row bookkeeping (~4 ms a tick at C=32), the K-step decode (3 ms), multi-row Thinker prefills run eagerly (22 ms host vs 12 ms of kernels), and the tick serializes the Thinker's and the Talker's synchronizations.
  - The first text chunk (283 ms of the C=32 first audio) scales with the tick.
  - Kernel efficiency: SM utilization ~50-56% with 6-18% occupancy means small kernels; whole K-step capture and fusion target the ~400 kernels of a Talker frame.
- **Round 10 attribution** (corrected by round 11: its `gpu_busy` was the union of stage CUDA-event spans, which include host submission gaps, synchronizations and waits, so it showed neither kernel activity nor saturation; the "~0.9 s of GPU work per second" summed isolated wall times. The HiFT measurement and its served gain stand.) (`engine/runtime/stage_trace.py`, `tools/engine_profile.py`; 8338bda `track_s_fast`, GPU2 under MPS, closed loop on tuning IDs, 20 s warm-up and a 40 s window per C; `results/engine_profile/r10_base`). The trace (CUDA events, no synchronization) reproduces the sweep: C=1 373 fps, TTFA p50 0.133 s; C=32 2120 fps, 0.759 s.
  - From C=8 up, the engine thread and the Token2Wav worker are both about 99% busy, and some GPU work is in flight 100% of the time. C=32 per second: Talker K=4 step 29.3 x 22.3 ms (65%) and Thinker step 29.3 x 11.7 ms (34%) on the engine thread; worker decode 5.0 x 151 ms (74%, 20.5 rows a call) and prompt setups 7.8 x 32 ms (25%).
  - Alone on the GPU the same work costs far less: Talker frame graph 1.27 ms at bucket 32 (K=4: ~5.1 ms), Thinker step 5.4 ms (buckets 4-16; weight-bandwidth bound), worker decode at 20 rows 70.5 ms (encoder 6.4, DiT solve 25.4, HiFT 37.5), setup 19 / 27 / 46 ms for 1 / 2 / 4 prompts (half host time), S3 tokens 3.4 ms of GPU per reference (`tools/spike/talker_frame_profile.py`, `thinker_step_profile.py`, `t2w_decode_isolated.py`). Summed at the C=32 rates that is ~0.9 s of GPU work per second: the GPU is saturated, and the engine's short-kernel chains wait behind the worker's under MPS (Talker 22 ms served vs 5 ms alone).
  - GPU milliseconds per second of audio at C=32 (alone): setups ~1.6 plus host, HiFT 1.9, Thinker 1.9, Talker 1.8, DiT 1.3, flow encoder 0.3, S3 0.7. 2x throughput needs this halved; no single lever is that large.
  - Scheduling and placement levers measured and rejected (C=8/32 against the same control): `thinker_lazy` (every row stays urgent: unchanged), adaptive K up to 8 (the short-K rule holds while any Thinker row generates: unchanged), MPS SM cap on the worker 75% / 60% (the Talker speeds up, the worker slows: C=32 2210 / 2177 fps against 2288 without a cap).
  - Lever implemented: HiFT's FP32 convolutions were 40% of the decode call's GPU time. With the upsampling stages in TF32 (AC-3.4 bound kept, see the component table) HiFT at 20 rows takes 17.5 ms instead of 37.5 and the decode call 50.5 instead of 70.5. Served (same tool): C=8 995 -> 1060 fps, C=16 1459 -> 1561, C=32 2120 -> 2288 (+7.9%); TTFA p50 C=16 0.525 -> 0.480 s, C=32 0.759 -> 0.679 s; RTF C=32 0.391 -> 0.359; C=1 unchanged (`results/engine_profile/r10_hift_tf32`).
  - Remaining levers by size: the first text chunk (11 Thinker steps at the tick rate: ~360 ms of the 0.68 s C=32 TTFA), prompt setups (host-bound encoder loop and padded setup buckets), Thinker weight bandwidth (FP8 failed AC-3.7), the Talker's launch-bound frame (~240 kernels a frame), DiT solve.
- **Round 10 measurement, `track_s_fast_hift`** (`track_s_fast` plus `t2w_hift_decode_tf32`; evaluation IDs, K=5, 295fad6; `sweeps/engine/r10_hift_gpu{2,3}.json`, `results/engine_sweeps/r10_hift_gpu{2,3}`; all 30 runs valid, no failed request, cells verified from artifacts in `results/engine_sweeps/r10_hift_verify.log`). 9 of 24 ratios have LCB >= 2.0 against the envelope, all of them TTFA (round 9: 8; C=16 TTFA p50 now passes at LCB 2.100).
  - Against r9-b22 (`tools/sched_compare.py r10-hift r9-b22`; separate sessions, diagnostic) every metric improves at every C, every LCB above 1. Throughput: C=1 +2.3%, C=2 +4.3%, C=4 +4.0%, C=8 +2.8%, C=16 +6.2%, C=32 +10.3% (2262 vs 2050 fps). TTFA p50: -1.7% to -9.4% (C=32 0.698 vs 0.771 s). TTFA p95: -2.7% to -8.3%. RTF mean: -2.2% to -9.6%.
  - Envelope ratios, point / LCB. Throughput: C=1 1.369/1.362, C=2 1.099/1.089, C=4 0.999/0.991, C=8 1.038/1.026, C=16 1.053/1.024, C=32 1.133/1.123. RTF follows throughput within 0.04. TTFA p50: C=1 2.252/2.241, C=2 1.927/1.883 (fail), C=4 2.194/2.174, C=8 2.334/2.302, C=16 2.154/2.100, C=32 1.491/1.450 (fail). TTFA p95: C=2 1.971/1.926 (fail), all other C pass (2.447-2.837 LCB).
  - C=16 run 4 is low (1448 fps against 1544-1559 for the other four runs); it is kept, which is why the C=16 LCBs sit further below the point estimates.
  - Gap to the goal (2.0 / LCB): throughput and RTF need 1.4-2.0x more at every C; TTFA p50 C=32 needs 1.38x, C=2 1.06x; TTFA p95 C=2 1.04x.
  - Not yet a headline change: `track_s_fast_hift` needs the AC-4 quality qualification before it replaces `track_s_fast`.
- **Round 9 measurement with the B22 Talker capacity commitments** (evaluation IDs, K=5, 8338bda; `sweeps/engine/r9_b22_gpu{2,3}.json`, `results/engine_sweeps/r9_b22_gpu{2,3}`). Same 8 of 24 ratios at LCB >= 2.0 against the envelope as round 8.
  - Against r8-head (`tools/sched_compare.py r9-b22 r8-head`; separate sessions, diagnostic), every metric is 0.2-2.9% worse: throughput C=2 -2.8%, C=4 -2.4%, C=16 -2.4%, the others -0.3% to -1.0%. TTFA p95 is +0.6% to +2.8%. C=8 and C=32 TTFA p50 are unchanged.
  - The commitments never held a start: `talker_kv_wait_ticks` stays 0 through the 30-minute C=32 acceptance load, at about 342 committed pages of 1280. The difference is per-tick bookkeeping or session drift, not admission waits.
  - Envelope ratios, point / LCB: C=1 throughput 1.338/1.330, TTFA p50 2.214/2.201; C=16 TTFA p50 1.999/1.947; C=32 throughput 1.027/1.019, TTFA p50 1.351/1.312, TTFA p95 2.571/2.520.
- **Robustness** (`tools/robustness.py`, replay with `tools/robustness_replay.py`): the module docstring lists the scenarios, the predeclared memory rule (±1% of the first sample at or after 60 s, over every later sample and the drained endpoint; CUDA allocated bytes back within 16 MiB of the warmed baseline) and the artifacts (`requests.jsonl` with every request, cancellation and raw observation; `wire.jsonl.gz`). The replay rebuilds every scenario's verdict and the overall verdict from them with the same functions the live run uses.
  - Acceptance policy (B17e, `ACCEPTANCE_POLICY` version 1). An acceptance run persists the policy and is audited from its raw records, live and in the replay, by `policy_problems`:
    - the exact settings: C=32, 30 min, 60 s reference, overload 128, EOS 192;
    - the declared workloads and every scenario's record count;
    - every record and timed observation inside its scenario's phase, and the report's phases equal to the persisted ones;
    - sustain sampling: first sample within 5 s of the start, no gap above 15 s, last within 15 s of the deadline, a reference sample within 15 s after 60 s, and the drained endpoint after the deadline and every request;
    - one request at a time per closed-loop client, and 32 sustain clients;
    - long prefill: 8 long requests, each with a served prompt above the budget, arriving within 1 s beside 24 companion clients, each served throughout the long interval (gaps ≤ 50 ms).
  - Re-audit of the saved 9ed19a1 acceptance under this policy, written to `results/l20x/robustness/r9_reaudit_9ed19a1.json`; the run's own report is unchanged. Its only problem is that the run did not persist policy version 1, which it predates. Counts, phases, sampling coverage, the reference and the long-prefill geometry all hold.
  - Round 9 acceptance, 8338bda (B22 commitments, policy v1; `results/l20x/robustness/r9_acceptance_8338bda`): PASS, policy problems none; `replay.json` rebuilds the same verdict and policy (ok).
    - 33,291 requests, 0 invalid, 32 clients in flight throughout.
    - Max stall 0.179 s, refill max 9.1 ms, first dispatch 34 ms.
    - Memory max deviation: engine 0.133%, worker 0.265%, feature total 0.064%, GPU 0.
    - Long prefill: 8 of 8 long requests (≥ 5422 tokens) and 110 of 110 companion streams valid, per-tick prefill max 2048 = budget, stall max 0.396 s.
  - Round 8 acceptance attempts on GPU3, C=32, 30-minute sustain, the predeclared rules unchanged; each replays to its own verdict under its recorded revision (`results/l20x/robustness/r8_acceptance_<rev>`):
    - 5db059a FAIL: one feature process +1.034%. Cause: torchaudio 2.11 decodes every `torchaudio.load` through TorchCodec, +0.6 KB per call; 3f467e3 decodes with soundfile (identical tensors).
    - 3f467e3 FAIL: one client refill gap of 67 ms > 50 ms. Cause: the checker's own full GC over the held records (80-110 ms); d67f834 suspends automatic full collections during the load.
    - d67f834 PASS, at a 0.14 ms margin (refill max 49.86 ms; the load generator's event loop lagged up to 58 ms). 53d1135 completes a stream's metrics and record in a worker thread after the client has started its next request.
    - 53d1135 FAIL: one feature process +1.077%, a rise in the first ~3 minutes of load after the 60 s reference, flat afterwards. 9ed19a1: jemalloc returns freed pages at once and each feature process runs 2000 synthetic jobs before load (startup about 4 minutes longer).
    - 9ed19a1 PASS: 32,736 requests, 0 invalid, max stall 0.181 s, refill max 11.7 ms, first dispatch 30 ms; memory max deviation engine 0.122%, worker 0.276%, feature total 0.076% (each process 0.059-0.108%), GPU 0; long prefill 8 prompts of >= 5422 tokens beside 24 continuing stream clients, all 8 and all 121 stream requests valid, per-tick prefill max 2048 = budget, stall max 0.301 s.

## Quality references (AC-2, AC-4)

- **SIM attention v1** (`evals/speech/wavlm_blocked.py`; evaluator setting `sim.attention = "query_blocked_v1"`, `query_block` 256).
  - The pinned WavLM fast path materialises the relative-position bias, its gated copy and the key-padding sum as `[B*H, T, T]` fp32. One 82 s output padded into its 16 + 16 batch needs 32 GiB for each.
  - v1 evaluates the same elements per block of 256 query rows over the complete keys, with the same `scaled_dot_product_attention` call.
  - Projections, batches, padding, weights and precision are unchanged.
- **Equivalence** (`tools/sim_equivalence.py`, on the batches of the report that could not be scored, `results/l20x/campaign/sim_v2/`):
  - 203 of 204 batches compare, and all 203 give bitwise identical embeddings (largest score difference 0.0).
  - The 82 s batch runs out of memory under the pinned attention and scores under v1, with a 52.8 GiB peak.
- **Rescoring.** The evaluator digest changes, so every dependent is rescored under it, with the unchanged inputs and settings.
  - Qualification v3: 9 candidates × 960 rows. SIM and UTMOS are identical on every row. The bf16 ASR differs run to run on a few rows: one Qwen3-ASR transcript of an unintelligible output, and Whisper WER on 4 rows. Every decisive qualification field and status is unchanged (7 of 9 qualify).
  - Then `winners_v3.json` and `sim_v2/dref_v3`, both identical to their predecessors except for the binding.
  - Full-set references (8 × 3264 rows, `sim_v2/fullq_scored_v3`). Compared row by row with the seven earlier scorings (`sim_v2/compare_fullq_v2_v3.json`), SIM and UTMOS are identical on every row. The ASR differs on 1–3 rows per report. The h200 MPS-on C=32 reference, previously unscorable, now scores all 3264 rows; its 82 s row has SIM 30.5.
- **Final speech Q_ref** (`results/l20x/campaign/qref/speech_qref_final.json`).
  - Frozen from the 8 references, each paired with its `qualification_v3` record, the v3 campaign qualification and `winners_v3.json`.
  - Values: WER 0.01557, catastrophic count 4, SIM 49.56, UTMOS 4.298, and silence, repetition and truncation rates of 0.
  - `qref verify` rebuilt it from its sources and reports it unchanged (`speech_qref_verify.log`).
- **Multimodal evaluator v2** (`evals/multimodal/suite.py`).
  - Its identity covers only what multimodal scoring executes:
    - the scorer checkout and its environment;
    - the vLLM-Omni checkout and its environment;
    - the frozen manifests;
    - the workers;
    - the floors and token limits.
  - A version-1 record maps to the digest of these components.
  - All four multimodal Q_ref sources match the current components. The Q_ref was refrozen from them unchanged as `results/l20x/campaign/qref/multimodal_qref_final_v2.json` (MMMU 0.6556, MMSU 0.5192, Daily-Omni 0.7803) and verified.
- **Engine multimodal generations.**
  - The candidate's own interpreter and served identity are bound (B15-MM).
  - Every response sidecar row carries the generation digest, so a Q_ref attached afterwards fails certification (B12d).
  - Round 10 (B12d worker boundary): both run-mode workers require the job's generation digest before sending and return it on the result and every response row; `cmd_run` stores a result only when the result and every row carry exactly this generation's digest (missing, stale, mixed: nothing certifiable is written, `result_binding_problems`) and keeps the rows as returned. Collection-to-certification tests cover stale result, stale rows, missing, mixed and the actual worker entry points.
- **Multimodal evaluator v3** (round 10): the worker change alters the worker hashes, so `WORKER_LINEAGE` maps exactly the predecessor revisions (whose `manifest`, `rescore`, payload and scoring code are byte-identical) to their successors. The multimodal Q_ref was refrozen from the same four source reports as `results/l20x/campaign/qref/multimodal_qref_final_v3.json`, with identical values and identity except the evaluator digest, and verified (re-scoring every source report under the new workers).

## Environments

See `engine/env/README.md`. The engine runs in `.venv-engine` (torch 2.13, transformers 5.14, flashinfer 0.7). The HF
reference runs in `.venv-hfref` (transformers 4.51, the model card's stack). Parity references are produced in the
reference environment and compared in the engine environment.

## Licenses

The checkpoint's code (processor, encoders, `modeling_minicpmo.py`) is Apache-2.0, as is `minicpmo-utils` (`stepaudio2`
Token2Wav, CosyVoice2 flow, HiFT). The engine imports them and adapts one method for transformers 5.x
(`MiniCPMWhisperEncoderLayer.forward`). The adapted copy is in `engine/models/multimodal.py`.
