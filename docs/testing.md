# Test results

Run on 2026-10-20 from this repository on the L20X host (CPU suites with `CUDA_VISIBLE_DEVICES=""`; GPU suite on one
idle GPU). Commands are in [reproduce-benchmark.md](reproduce-benchmark.md#tests).

| Suite | Environment | Result |
|---|---|---|
| `tests/bench`, `tests/evals` (harness, statistics, artifact verification, evaluators, campaign) | `.venv` | 148 passed, 19 skipped |
| `tests/engine_api` (API, scheduler tick, KV commitments, worker loop, attribution, robustness checker) | `.venv-engine` | 132 passed, 1 skipped (round 15, df485cf: 135 passed) |
| `tests/engine` with `ENGINE_GPU_TESTS=1` (parity, graphs, kernels, Token2Wav) | `.venv-engine`, 1 GPU | 126 passed, 5 failed, 1 expected failure (round 15 with per-lane capture streams, df485cf: the same) |

Skips: tests that need the vLLM-Omni environment, generated reference artifacts (`results/...`) or specs recorded in
the same checkout skip when those are absent.

## GPU failures

All five are numerical bounds that the engine misses by a small margin and that are recorded as open items; none is
new in this repository (the engine code here differs from the measured commit only in path constants and unused
imports).

| Test | Measured | Bound | Note |
|---|---:|---:|---|
| `test_multimodal_frontend.py::test_gpu_bf16_embeddings_match_hf_bf16` | audio min cosine 0.9969 | 0.999 | images bit-identical; the HF reference environment picks other bf16 SDPA kernels for the Whisper encoder; CPU fp32 parity passes the literal bound |
| `test_talker_gpu_parity.py::test_condition_and_teacher_forced_logits` | top-1 96.7%, condition rel-L2 ≤ 1e-2 | 99% | HF's own bf16 eager vs SDPA agree on only 96.4–96.6% |
| `test_track_s_gpu_parity.py::test_track_s_teacher_forced_and_conditions` | top-1 96.9%, condition rel-L2 1.6e-3 | 99% | same noise floor as above |
| `test_thinker_decode.py::test_fp8_thinker_ac37` | top-1 94.5%, min cosine 0.848 | 98% | FP8 Thinker is off in every profile |
| `test_paged_runtime.py::test_ragged_batch_matches_reference[thinker]` | min cosine 0.9987 | 0.999 | bf16 ragged batch vs per-row reference |

Expected failure: `test_hift_tf32.py[rows_to_16_batched_setup]` (TF32 HiFT with the batched setup encoder, max chunk
1.06e-2 against 1e-2; see [current-results.md](current-results.md#numerical-and-quality-status)).

## End-to-end smoke

`track_s_fast_r11` started from this repository with `scripts/serve_engine.sh` under MPS (ready after 320 s, including
graph capture, Token2Wav compilation and the feature-process warmup):

- `python -m bench.cli smoke --adapter engine --n 4`: 4 of 4 requests valid, transcripts equal to the target text,
  TTFA 110–207 ms, RTF 0.061–0.073.
- The README request (`body.json` with a base64 voice prompt): streamed text deltas, 4 audio deltas, `finish_reason`
  `stop`.

## Robustness acceptance

Recorded on `track_s_fast` (2026-10-18; `tools/robustness.py`, policy version 1): C=32 for 30 minutes with
cancellations, EOS-length requests, a 48 s fallback, 128-request overload and 8 long prompts (≥ 5422 tokens) beside
24 streaming clients.

| Check | Result |
|---|---|
| Requests | 33,291, 0 invalid, 32 clients in flight throughout |
| Playback | max stall 0.179 s, refill max 9.1 ms, first dispatch 34 ms |
| Memory (max deviation from the 60 s reference, rule ±1%) | engine 0.133%, worker 0.265%, feature processes 0.064%, GPU 0 |
| Long prompts | 8 of 8 long requests and 110 of 110 companion streams valid; per-tick prefill max 2048 (the budget) |
| Replay | `tools/robustness_replay.py` rebuilds the same verdict from the raw records |

Not repeated on `track_s_fast_r11` or `track_s_fast_r16`.
