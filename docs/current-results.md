# Current results

Snapshot of 2026-10-21. The engine's six cells (`track_s_fast_r16`, round 16) were measured on 2026-10-21, the
baseline cells on 2026-10-15 and 2026-10-16, all on the same L20X host. Everything below is rebuilt from
[`evidence/20261021-l20x`](../evidence/20261021-l20x) by `python tools/report_results.py evidence/20261021-l20x`
(also [current-results.json](current-results.json)); the unchanged baseline cells, tuning cells and traces are read
from [`evidence/20261019-l20x`](../evidence/20261019-l20x), which also keeps the previous engine snapshot
(`track_s_fast_r11`).

## Setup

- **Hardware:** NVIDIA L20X, 143 GB, driver 570.133.20; one GPU per server.
- **Model:** `openbmb/MiniCPM-o-4_5` at `503e754207c94da6bb26850b4469f367c9ea3582`.
- **Workload:** Seed-TTS eval, English (`zhaochenyang20/seed-tts-eval@8f5e1aa`). Each request is
  "Please read the following text out loud in English: {text}" with the row's reference audio as the voice prompt.
  Thinker greedy, at most 256 tokens. Talker temperature 0.8, top-p 0.85, top-k 25, repetition penalty 1.05 over 16
  frames, 50–2048 frames, a fixed seed per request.
- **Request sets:** a fixed hash split of the 1088 rows: a 64-row warmup pool, a 640-row evaluation pool and the rest
  for tuning. Each run sends max(320, 20 × C) evaluation requests through C closed-loop clients.
- **Protocol per cell:** warmup, a discarded run 0, then K=5 measured runs, each on a freshly started server (caches
  cold). SGLang-Omni at C=16 was extended to K=10 before its envelope was frozen. Every run's requests, wire events and
  audio are recorded and every cell is verified from its artifacts before it is used.
- **Metrics** (client side, `bench/metrics.py`): throughput = 25 × Σ delivered audio seconds / wall time (25 Hz codec
  frames per second); TTFA = time to the first audio sample; RTF = time to the last audio sample / audio duration,
  averaged over requests. Stalls are computed from a playback clock and reported separately.
- **Statistics** (`bench/stats.py`): oriented ratio engine / baseline (> 1 is better), with a one-sided 95% lower bound
  from a run-level bootstrap.

## Engine and baselines

| C | Framework | Config | Throughput (frames/s) | Throughput / SGLang | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 406.01 | 1.46× | 125.8 | 137.6 | 0.0623 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 187.52 | 0.67× | 295.3 | 339.9 | 0.1354 |
| 1 | sglang-omni | `speech-mm32k` | 278.72 | 1.00× | 381.2 | 502.9 | 0.0931 |
| 2 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 562.93 | 1.25× | 177.3 | 212.2 | 0.0906 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 318.61 | 0.71× | 380.0 | 465.1 | 0.1596 |
| 2 | sglang-omni | `speech-mps-on` | 450.37 | 1.00× | 471.0 | 637.7 | 0.1152 |
| 4 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 810.38 | 1.14× | 272.7 | 326.7 | 0.1261 |
| 4 | vllm-omni | not evaluated at this C | — | — | — | — | — |
| 4 | sglang-omni | `speech-mps-on` | 709.73 | 1.00× | 597.6 | 825.6 | 0.1463 |
| 8 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 1180.65 | 1.14× | 367.2 | 439.6 | 0.1734 |
| 8 | vllm-omni | not evaluated at this C | — | — | — | — | — |
| 8 | sglang-omni | `speech-mps-on` | 1033.40 | 1.00× | 853.9 | 1150.4 | 0.2001 |
| 16 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 1682.74 | 1.16× | 484.6 | 597.5 | 0.2426 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 411.03 | 0.28× | 1057.2 | 1982.3 | 1.0148 |
| 16 | sglang-omni | `speech-mps-on` | 1455.39 | 1.00× | 1179.2 | 1686.0 | 0.2829 |
| 32 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 2419.94 | 1.21× | 654.1 | 804.2 | 0.3366 |
| 32 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 407.68 | 0.20× | 1041.1 | 1965.6 | 2.0344 |
| 32 | sglang-omni | `speech-mps-off` | 1996.67 | 1.00× | 1613.0 | 2374.3 | 0.4137 |

Baseline configurations:

- `speech-*`: SGLang-Omni's MiniCPM-o server with its Seed-TTS runbook flags (`scripts/serve_sglang_omni.sh`). `mm32k`
  raises the Thinker's maximum sequence length to 32768; `mps-on` enables SGLang-Omni's own pipeline MPS runtime.
- `minicpmo_4_5_turn*`: vLLM-Omni's shipped deploy configs (`minicpmo_4_5_turn.yaml`, `minicpmo_4_5_turn_mrv2_h200.yaml`)
  with vLLM-Omni's private-MPS switch (`configs/vllm/`).
- Each baseline was tuned per concurrency and metric on the tuning requests among the class-R configurations (same
  prompt token IDs and reference audio bytes as the reference path). The winners were then measured on the evaluation
  requests. At C=4 and C=8 every metric was won by SGLang-Omni, so vLLM-Omni has no evaluation cell there.
- SGLang-Omni returns one waveform per request (no audio streaming), so its TTFA equals its end-to-end latency.
- Every qualified baseline configuration at every C (tuning requests), the quality qualification and the vLLM-Omni
  stage trace are in [baselines.md](baselines.md).

## Streaming behaviour

Means over the measured runs; the max stall is the largest over all runs. E2E is the time to the last audio sample. Chunk gaps
are the times between audio events of one request; SGLang-Omni sends one event, so it has none.

| C | Framework | Config | E2E p50 (s) | E2E p95 (s) | Stall mean (ms) | Stall max (s) | Chunk gap p95 (ms) | Audio per run (s) | Requests/s |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 0.28 | 0.37 | 0.0 | 0.00 | 45 | 1461 | 3.56 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 0.55 | 0.80 | 3.6 | 1.57 | 91 | 1409 | 1.70 |
| 1 | sglang-omni | `speech-mm32k` | 0.38 | 0.50 | 0.0 | 0.00 | — | 1389 | 2.57 |
| 2 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 0.40 | 0.53 | 0.0 | 0.00 | 76 | 1461 | 4.93 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 0.68 | 0.97 | 0.0 | 0.00 | 186 | 1411 | 2.89 |
| 2 | sglang-omni | `speech-mps-on` | 0.47 | 0.64 | 0.0 | 0.00 | — | 1388 | 4.15 |
| 4 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 0.56 | 0.73 | 0.0 | 0.00 | 133 | 1462 | 7.10 |
| 4 | sglang-omni | `speech-mps-on` | 0.60 | 0.83 | 0.0 | 0.00 | — | 1396 | 6.51 |
| 8 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 0.77 | 1.00 | 0.0 | 0.00 | 163 | 1465 | 10.32 |
| 8 | sglang-omni | `speech-mm32k` | 0.89 | 1.17 | 0.0 | 0.00 | — | 1381 | 9.13 |
| 8 | sglang-omni | `speech-mps-on` | 0.85 | 1.15 | 0.0 | 0.00 | — | 1385 | 9.55 |
| 16 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 1.08 | 1.40 | 0.0 | 0.00 | 192 | 1469 | 14.66 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 3.40 | 6.97 | 579.7 | 14.14 | 1796 | 1314 | 4.01 |
| 16 | sglang-omni | `speech-mps-on` | 1.18 | 1.69 | 0.0 | 0.00 | — | 1385 | 13.45 |
| 32 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 1.48 | 1.89 | 0.7 | 0.10 | 273 | 2900 | 21.36 |
| 32 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 3.79 | 26.94 | 4219.3 | 93.53 | 9295 | 2581 | 4.04 |
| 32 | sglang-omni | `speech-mps-off` | 1.61 | 2.37 | 0.0 | 0.00 | — | 2730 | 18.73 |

The engine streams several audio events per request (5.4 on average at C=32). Its largest stall over all runs is 0 up
to C=8, 1.4 ms at C=16 and 0.10 s at C=32 (3200 requests). vLLM-Omni's MRv2 profile stalls for seconds at C ≥ 16 (see [baselines.md](baselines.md)). The
frameworks generate slightly different amounts of audio for the same requests (Talker sampling), which throughput
counts as delivered.

## Against the target

The target is 2× the best baseline cell for each C and metric, with the one-sided 95% lower bound at or above 2.0. The
best cell per metric is the configuration selected on the tuning requests, so it can differ from the faster cell shown
above (at C=8 throughput it is `speech-mm32k`, 985.0 frames/s). Point ratio / LCB; bold cells pass.

| C | Throughput | TTFA p50 | TTFA p95 | RTF (mean) |
|---:|---:|---:|---:|---:|
| 1 | 1.46× / 1.45 | **2.35× / 2.33** | **2.47× / 2.41** | 1.49× / 1.49 |
| 2 | 1.25× / 1.24 | **2.14× / 2.09** | **2.19× / 2.14** | 1.27× / 1.26 |
| 4 | 1.14× / 1.13 | **2.19× / 2.17** | **2.53× / 2.48** | 1.16× / 1.14 |
| 8 | 1.20× / 1.19 | **2.33× / 2.29** | **2.66× / 2.59** | 1.21× / 1.20 |
| 16 | 1.16× / 1.13 | **2.18× / 2.13** | **2.82× / 2.71** | 1.17× / 1.14 |
| 32 | 1.21× / 1.20 | 1.59× / 1.55 | **2.95× / 2.87** | 1.23× / 1.22 |

11 of 24 pass, all TTFA (the 2026-10-19 snapshot: 10; TTFA p50 at C=2 now passes at LCB 2.09). TTFA p50 at C=32 needs
about 1.3× more. Throughput and RTF need 1.4–1.8× more at every C. **Throughput-and-latency 2× gate: FAIL.**

These are development measurements on an L20X with K=5. The planned acceptance runs on an H200 with a fixed K=10,
duration guards and the quality gates; they have not been run.

## Run ranges

Minimum–maximum over the measured runs of each cell. No request failed in any measured run.

| C | Framework | Config | K | Requests / run | Failed | Throughput | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |
|---:|---|---|---:|---:|---:|---|---|---|---|
| 1 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 0.28 | 0.37 | 0.0 | 0.00 | 45 | 1461 | 3.56 |
| 1 | sglang-omni | `speech-mm32k` | 5 | 320 | 0 | 277.10–280.23 | 378.5–384.9 | 498.7–505.5 | 0.0926–0.0936 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 5 | 320 | 0 | 166.84–199.08 | 293.1–297.6 | 329.6–354.6 | 0.1273–0.1482 |
| 2 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 0.40 | 0.53 | 0.0 | 0.00 | 76 | 1461 | 4.93 |
| 2 | sglang-omni | `speech-mps-on` | 5 | 320 | 0 | 439.78–456.90 | 465.7–484.3 | 626.5–653.2 | 0.1134–0.1184 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 5 | 320 | 0 | 306.15–330.05 | 366.6–394.4 | 447.1–484.9 | 0.1538–0.1655 |
| 4 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 0.56 | 0.73 | 0.0 | 0.00 | 133 | 1462 | 7.10 |
| 4 | sglang-omni | `speech-mps-on` | 5 | 320 | 0 | 699.98–719.34 | 588.2–606.6 | 807.4–849.2 | 0.1442–0.1478 |
| 8 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 0.77 | 1.00 | 0.0 | 0.00 | 163 | 1465 | 10.32 |
| 8 | sglang-omni | `speech-mm32k` | 5 | 320 | 0 | 972.19–1003.36 | 872.9–911.5 | 1119.9–1232.2 | 0.2063–0.2137 |
| 8 | sglang-omni | `speech-mps-on` | 5 | 320 | 0 | 1020.19–1040.43 | 835.2–870.3 | 1116.2–1203.4 | 0.1980–0.2033 |
| 16 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 1.08 | 1.40 | 0.0 | 0.00 | 192 | 1469 | 14.66 |
| 16 | sglang-omni | `speech-mps-on` | 10 | 320 | 0 | 1386.19–1518.73 | 1083.1–1286.2 | 1518.3–1926.0 | 0.2712–0.2971 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 5 | 320 | 0 | 403.16–416.38 | 1026.0–1126.0 | 1951.8–2053.0 | 1.0029–1.0281 |
| 32 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r16` | 1.48 | 1.89 | 0.7 | 0.10 | 273 | 2900 | 21.36 |
| 32 | sglang-omni | `speech-mps-off` | 5 | 640 | 0 | 1969.90–2019.52 | 1567.6–1672.3 | 2293.6–2479.2 | 0.4069–0.4186 |
| 32 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 5 | 640 | 0 | 404.51–409.67 | 1009.3–1117.5 | 1902.1–2133.1 | 1.9736–2.0628 |

## Numerical and quality status

| Check | Status |
|---|---|
| Thinker parity (bf16, GPU) | Passes the noise-relative rule: no further from HF bf16 eager than HF's own SDPA path |
| Talker conditioning and logits | Teacher-forced codec logits within rel-L2 1e-4 of HF |
| Token2Wav, FP32 reference path | Within 1e-4 of the reference with seeded HiFT noise |
| Fast DiT, FP16 flow | Within the FP16 mel bound (5e-3) |
| TF32 HiFT alone | Max chunk 5.3e-3 (bound 1e-2), 190 chunks |
| TF32 HiFT + batched setup encoder (`track_s_fast_r11`) | **Fails on 1 of 190 chunks: 1.06e-2.** Strict expected failure in `tests/engine/test_hift_tf32.py` until repaired |
| TF32 HiFT without stage 0's transposed convolution + batched setup encoder (`track_s_fast_r16`) | Max chunk 3.2e-3 (bound 1e-2), 190 chunks: B24 repaired |
| Speech quality gates (WER, SIM, UTMOS vs the frozen baseline reference, 1088 rows × 3 seeds at C=1 and C=32) | `track_s_fast_r16` (round 16): absolute floors **pass** at C=1 and C=32 over three seeds (UTMOS 4.1658 / 4.1670); the comparative UTMOS gate **fails** (4.17 vs the non-streaming 4.30, margin 0.05), computed with the gate formula on cells not declared against the frozen Q_ref ([engine.md](engine.md), round 16). `track_s_fast_r11`'s single seed-0 scoring missed the UTMOS floor by 0.0009 |
| Multimodal accuracy (MMMU, MMSU, Daily-Omni) | Evaluators and references frozen; engine gates not run |
| Robustness (C=32, 30 min, cancellation, overload, long prefills, memory ±1%) | Passed on `track_s_fast` (33,291 requests, 0 invalid, max stall 0.18 s); not repeated on `track_s_fast_r11` or `track_s_fast_r16` |

## Speech quality

Acceptance (user decision, 2026-10-21): the engine's speech quality passes on the **absolute floors**, pooled over
three seeds of all 1088 rows at C=1 and C=32. The comparative gates against the frozen baseline reference are computed
and reported but do not decide: their UTMOS reference (4.298) comes from SGLang-Omni, which does not stream audio.

`track_s_fast_r16`, seeds 0, 1 and 2 (`quality/engine_final`):

| | Rows | WER (filtered) | Catastrophic rows | SIM | UTMOS (seeds 0 / 1 / 2) | UTMOS pooled | Silence / repetition / truncation |
|---|---:|---:|---:|---:|---|---:|---|
| Engine `track_s_fast_r16`, C=1 | 1088 × 3 | 1.38% | 7 | 49.11 | 4.163 / 4.168 / 4.167 | **4.166** | 0 / 0 / 0.06% |
| Engine `track_s_fast_r16`, C=32 | 1088 × 3 | 1.46% | 12 | 49.09 | 4.165 / 4.172 / 4.165 | **4.167** | 0.06% / 0 / 0 |
| Absolute floors | | ≤ 2.69% | | ≥ 41.58 | | ≥ 4.163 | |
| Baseline reference (best per metric) | 1088 × 3 | 1.56% (corpus) | 4 | 49.56 | | 4.298 | 0 / 0 / 0 |

- Absolute floors: **pass** at both C (and for every seed except C=1 seed 0, 4.1626, one draw; see
  [engine.md](engine.md), round 16, for the attribution to the Talker chunk-prefill graphs' padding).
- Comparative gates (reported only; the gate formula applied to these reports, which were not declared against the
  frozen reference before running): UTMOS fails at both C (4.17 vs 4.30, margin 0.05); WER passes at C=1 and is
  inconclusive at C=32 (2.00% corpus vs 1.56%, seed noise above the margin); catastrophic rows are inconclusive at
  both C; SIM and the silence, repetition and truncation rates pass.
- The UTMOS gap is concentrated in the low tail: at C=1 seed 0 the engine's 10th percentile is 3.76 (SGLang-Omni 4.01,
  vLLM-Omni turn 3.79) and its 90th percentile 4.46 (4.48, 4.46). The streaming vLLM-Omni turn profiles score about
  4.20; the engine's 8-frame first chunk accounts for about 0.025 (320-row ablation).
- Thinker greedy text equals the reference on 99.9% of rows.

Test-suite results, the end-to-end smoke of this repository and the robustness acceptance are in [testing.md](testing.md).

## Evidence

- `evidence/20261021-l20x/<candidate>_c<C>/`: the six `track_s_fast_r16` cells: each cell's `experiment.json` (the
  frozen spec, fingerprints and the server launch), `result.json` (verification, manifest, run summaries), each run's
  `summary.json` and gzipped `requests.jsonl`, and the gzipped server logs. Audio and wire dumps stay with the raw cells.
- `evidence/20261021-l20x/campaign/`: the engine-versus-envelope ratios computed on the raw cells
  (`r16_vs_envelope.json`, `r16_envelope.log`) and the cell verification log (`r16_verify.log`).
- `evidence/20261021-l20x/quality/engine_final/`: the three-seed speech quality reports at C=1 and C=32 with per-row
  records.
- `evidence/20261021-l20x/manifest.json`: every file's SHA-256 and its source file.
- `evidence/20261021-l20x/baselines.json` points to `evidence/20261019-l20x`, which holds the unchanged baseline
  evaluation cells, the frozen envelope (`campaign/envelope.json`), the tuning winners (`campaign/winners.json`), the
  42 tuning cells, the baseline stage traces, the baseline qualification records and the frozen speech reference
  (`quality/`), the robustness acceptance of `track_s_fast` and the previous engine snapshot (`track_s_fast_r11`).
- The engine cells ran from frozen worktrees at the commits recorded in each `experiment.json`.
