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
| Prompt features (worker) | `engine/runtime/t2w_worker.py::prompt_front_end` + `finish_prompt` | `stepaudio2.Token2wav._prepare_prompt` | identical tensors (`tests/engine/test_token2wav_stream.py::test_split_prompt_features_equal_reference`) |

## Scheduling (task17) and robustness evidence (AC-6)

- **Playback clock** (`engine/runtime/t2w_sched.py AudioState`): the harness stall model. A stream's buffer runs out at `max(previous exhaustion, delivery) + seconds delivered`; the worker and the engine keep one per request, from the audio they hand on.
- **Token2Wav worker** (`engine/runtime/t2w_worker.py WorkerLoop`, `t2w_policy: "edf"`):
  - work-conserving: every ready window is decoded in the tick it becomes ready, earliest playback deadline first (`t2w_max_decode_rows` may cap a call);
  - a setup never runs ahead of a waiting continuation window: the tick decodes first, then runs at most `t2w_setup_calls_when_busy` setup calls;
  - setups run oldest first.
  - `"legacy"` keeps the rounds 4–5 order (every setup first, then a decode held for `t2w_min_batch` windows or `t2w_max_wait_ms`), for comparison.
- **Engine tick** (`backend_s.TrackSBackend._prefill_lane`, `t2w_sched.choose_prefills`): the bounded first-chunk lane.
  - Every decoding row runs every tick.
  - Pending Thinker prefills run oldest first within `prefill_token_budget` tokens; one that has waited `prefill_max_age_ms` always runs.
  - While a continuing stream has less than `prefill_slack_floor_ms` of buffered playback, only aged prefills run.
  - `null` budget and floor give the rounds 4–6 behaviour (every pending prefill each tick).
- **Tests** (CPU): `tests/engine_api/test_t2w_sched.py`, `test_t2w_worker_loop.py` (the production loop with a fake model and clock) and `test_prefill_lane.py`.
- **Robustness** (`tools/robustness.py`, replay with `tools/robustness_replay.py`): the module docstring lists the scenarios, the predeclared memory rule (±1% of the first sample at or after 60 s, over every later sample and the drained endpoint; CUDA allocated bytes back within 16 MiB of the warmed baseline) and the per-request artifacts (`requests.jsonl`, `wire.jsonl.gz`) the verdict can be rebuilt from.

## Environments

See `engine/env/README.md`. The engine runs in `.venv-engine` (torch 2.13, transformers 5.14, flashinfer 0.7). The HF
reference runs in `.venv-hfref` (transformers 4.51, the model card's stack). Parity references are produced in the
reference environment and compared in the engine environment.

## Licenses

The checkpoint's code (processor, encoders, `modeling_minicpmo.py`) is Apache-2.0, as is `minicpmo-utils` (`stepaudio2`
Token2Wav, CosyVoice2 flow, HiFT). The engine imports them and adapts one method for transformers 5.x
(`MiniCPMWhisperEncoderLayer.forward`). The adapted copy is in `engine/models/multimodal.py`.
