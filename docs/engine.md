# MiniCPM-o 4.5 engine: components and reference points

The engine is a single-GPU server (`scripts/serve_engine.sh`, `engine/server.py`). It is built from the checkpoint
`openbmb/MiniCPM-o-4_5@503e754` and serves the OpenAI-compatible `/v1/chat/completions` API with streamed text and
audio.

## Components

| Component | Module | Reference | Parity evidence |
|---|---|---|---|
| Thinker (Qwen3-8B), reference precision | `engine/models/thinker.py` | transformers `Qwen3ForCausalLM` on the `llm.*` weights | `tests/engine/test_thinker_parity.py`: exact in fp32 on CPU (min cosine at least 0.99999, all argmax equal); bf16 on GPU pending |
| Talker (20-layer Llama TTS), conditioning | `engine/models/talker.py` | `MiniCPMTTS`, `_generate_speech_non_streaming` | `tests/engine/test_talker_parity.py`: conditioning formula, teacher-forced codec logits (relative L2 below 1e-4), hidden/token misalignment negative |
| Talker sampler | `engine/talker_sampler.py` | `gen_logits`, `MiniCPMTTS.generate` (HF order) | `tests/engine/test_talker_sampler.py`: transformers warpers, frame-0 rule, min/max EOS, counter RNG, K-step, EOS freeze, padding isolation, swapped-order negative |
| Token2Wav stream chunker | `engine/token2wav_chunker.py` | `streaming_generate` stream driver | `tests/engine/test_token2wav_chunker.py`: windows identical to the reference loop; missing silence prefix or lookahead are detected |
| Token2Wav flow + HiFT stream decode | `engine/models/token2wav.py` | `stepaudio2.Token2wav` (minicpmo-utils 1.0.6) | `tests/engine/test_token2wav_stream.py` (GPU): against the FP32 reference (cuDNN TF32 off) with the HiFT source noise seeded per request and chunk, <= 1e-4 |
| Multimodal prompt assembly and encoders | `engine/models/multimodal.py`, `engine/media.py` | checkpoint processor, SigLIP and resampler, Whisper and projection | `tests/engine/test_multimodal_frontend.py`: 32 mixed samples, token IDs identical to the HF environment, embeddings with cosine at least 0.999, dropped-placeholder negative |
| Paged KV and ragged batch forward | `engine/runtime/paged.py` | the reference ports | `tests/engine/test_paged_runtime.py` (GPU) |
| Talker K-frame graph decode | `engine/runtime/talker_decode.py` | eager per-frame path | `tests/engine/test_talker_decode.py` (GPU) |
| Track T engine loop | `engine/runtime/backend.py` | HF `chat(generate_audio=True)` conditioning path | engine effective-input dump (`bench/effective_dump.py engine`) and the speech quality gates |
| API | `engine/server.py` | AC-7 | `tests/engine_api/`: harness adapter, unchanged vLLM native client, 400/503, disconnect cancellation |
| Track S engine loop | `engine/runtime/backend_s.py` | `engine/runtime/track_s.py` (HF `streaming_generate` port) | Thinker chunks per the reference yield rules (`tests/engine/test_track_s_chunks.py`), Talker KV rolled back at chunk EOS; smoke `tools/engine_smoke.py` |
| Batched Token2Wav | `engine/runtime/token2wav_batch.py` | per-stream `Token2Wav.decode` | batched == per-stream (ragged prompts, mixed stages, merged final windows), batched setups == reference `setup_cache` (`tests/engine/test_token2wav_stream.py`) |
| Token2Wav graphs | `engine/runtime/dit_graph.py`, `engine/runtime/hift_graph.py` | eager batched path | mel 1e-4 (FP32), AC-3.4 FP16 bound for FP16/compiled; HiFT graphs == eager on equal mel |
| Token2Wav worker process | `engine/runtime/t2w_worker.py` | in-process Token2Wav thread | same `BatchedToken2Wav`; requires MPS (see `docs/architecture_decision.md`) |
| Graph decoders | `engine/runtime/thinker_decode.py`, `engine/runtime/talker_decode.py` | eager per-step paths | graph == eager exactly, with plans skipped between page crossings (`tests/engine/test_{thinker,talker}_decode.py`) |

## Environments

See `engine/env/README.md`. The engine runs in `.venv-engine` (torch 2.13, transformers 5.14, flashinfer 0.7). The HF
reference runs in `.venv-hfref` (transformers 4.51, the model card's stack). Parity references are produced in the
reference environment and compared in the engine environment.

## Licenses

The checkpoint's code (processor, encoders, `modeling_minicpmo.py`) is Apache-2.0, as is `minicpmo-utils` (`stepaudio2`
Token2Wav, CosyVoice2 flow, HiFT). The engine imports them and adapts one method for transformers 5.x
(`MiniCPMWhisperEncoderLayer.forward`). The adapted copy is in `engine/models/multimodal.py`.
