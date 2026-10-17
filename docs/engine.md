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
- **Robustness** (`tools/robustness.py`, replay with `tools/robustness_replay.py`): the module docstring lists the scenarios, the predeclared memory rule (±1% of the first sample at or after 60 s, over every later sample and the drained endpoint; CUDA allocated bytes back within 16 MiB of the warmed baseline) and the artifacts (`requests.jsonl` with every request, cancellation and raw observation; `wire.jsonl.gz`). The replay rebuilds every scenario's verdict and the overall verdict from them with the same functions the live run uses.
  - Acceptance policy (B17e, `ACCEPTANCE_POLICY` version 1). An acceptance run persists the policy and is audited from its raw records, live and in the replay, by `policy_problems`:
    - the exact settings: C=32, 30 min, 60 s reference, overload 128, EOS 192;
    - the declared workloads and every scenario's record count;
    - every record and timed observation inside its scenario's phase, and the report's phases equal to the persisted ones;
    - sustain sampling: first sample within 5 s of the start, no gap above 15 s, last within 15 s of the deadline, a reference sample within 15 s after 60 s, and the drained endpoint after the deadline and every request;
    - one request at a time per closed-loop client, and 32 sustain clients;
    - long prefill: 8 long requests, each with a served prompt above the budget, arriving within 1 s beside 24 companion clients, each served throughout the long interval (gaps ≤ 50 ms).
  - Re-audit of the saved 9ed19a1 acceptance under this policy, written to `results/l20x/robustness/r9_reaudit_9ed19a1.json`; the run's own report is unchanged. Its only problem is that the run did not persist policy version 1, which it predates. Counts, phases, sampling coverage, the reference and the long-prefill geometry all hold.
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

## Environments

See `engine/env/README.md`. The engine runs in `.venv-engine` (torch 2.13, transformers 5.14, flashinfer 0.7). The HF
reference runs in `.venv-hfref` (transformers 4.51, the model card's stack). Parity references are produced in the
reference environment and compared in the engine environment.

## Licenses

The checkpoint's code (processor, encoders, `modeling_minicpmo.py`) is Apache-2.0, as is `minicpmo-utils` (`stepaudio2`
Token2Wav, CosyVoice2 flow, HiFT). The engine imports them and adapts one method for transformers 5.x
(`MiniCPMWhisperEncoderLayer.forward`). The adapted copy is in `engine/models/multimodal.py`.
