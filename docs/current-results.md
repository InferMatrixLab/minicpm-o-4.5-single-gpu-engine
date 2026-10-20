# Current results

Snapshot of 2026-10-19. The engine's six cells were measured on 2026-10-18 and 2026-10-19, the baseline cells on
2026-10-15 and 2026-10-16, all on the same L20X host. Everything below is rebuilt from
[`evidence/20261019-l20x`](../evidence/20261019-l20x) by `python tools/report_results.py evidence/20261019-l20x`
(also [current-results.json](current-results.json)).

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
| 1 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 397.43 | 1.43× | 126.2 | 133.2 | 0.0635 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 187.52 | 0.67× | 295.3 | 339.9 | 0.1354 |
| 1 | sglang-omni | `speech-mm32k` | 278.72 | 1.00× | 381.2 | 502.9 | 0.0931 |
| 2 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 521.11 | 1.16× | 190.2 | 222.5 | 0.0975 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 318.61 | 0.71× | 380.0 | 465.1 | 0.1596 |
| 2 | sglang-omni | `speech-mps-on` | 450.37 | 1.00× | 471.0 | 637.7 | 0.1152 |
| 4 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 761.92 | 1.07× | 267.1 | 308.2 | 0.1335 |
| 4 | vllm-omni | not evaluated at this C | — | — | — | — | — |
| 4 | sglang-omni | `speech-mps-on` | 709.73 | 1.00× | 597.6 | 825.6 | 0.1463 |
| 8 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 1118.93 | 1.08× | 361.4 | 425.6 | 0.1817 |
| 8 | vllm-omni | not evaluated at this C | — | — | — | — | — |
| 8 | sglang-omni | `speech-mps-on` | 1033.40 | 1.00× | 853.9 | 1150.4 | 0.2001 |
| 16 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 1674.36 | 1.15× | 483.5 | 569.6 | 0.2422 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 411.03 | 0.28× | 1057.2 | 1982.3 | 1.0148 |
| 16 | sglang-omni | `speech-mps-on` | 1455.39 | 1.00× | 1179.2 | 1686.0 | 0.2829 |
| 32 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 2437.10 | 1.22× | 654.6 | 794.6 | 0.3345 |
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
| 1 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 0.29 | 0.39 | 0.0 | 0.00 | 54 | 1461 | 3.48 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 0.55 | 0.80 | 3.6 | 1.57 | 91 | 1409 | 1.70 |
| 1 | sglang-omni | `speech-mm32k` | 0.38 | 0.50 | 0.0 | 0.00 | — | 1389 | 2.57 |
| 2 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 0.44 | 0.58 | 0.0 | 0.00 | 88 | 1461 | 4.56 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 0.68 | 0.97 | 0.0 | 0.00 | 186 | 1411 | 2.89 |
| 2 | sglang-omni | `speech-mps-on` | 0.47 | 0.64 | 0.0 | 0.00 | — | 1388 | 4.15 |
| 4 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 0.59 | 0.80 | 0.0 | 0.00 | 134 | 1461 | 6.68 |
| 4 | sglang-omni | `speech-mps-on` | 0.60 | 0.83 | 0.0 | 0.00 | — | 1396 | 6.51 |
| 8 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 0.82 | 1.06 | 0.0 | 0.00 | 198 | 1466 | 9.77 |
| 8 | sglang-omni | `speech-mm32k` | 0.89 | 1.17 | 0.0 | 0.00 | — | 1381 | 9.13 |
| 8 | sglang-omni | `speech-mps-on` | 0.85 | 1.15 | 0.0 | 0.00 | — | 1385 | 9.55 |
| 16 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 1.09 | 1.41 | 0.0 | 0.03 | 251 | 1471 | 14.57 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 3.40 | 6.97 | 579.7 | 14.14 | 1796 | 1314 | 4.01 |
| 16 | sglang-omni | `speech-mps-on` | 1.18 | 1.69 | 0.0 | 0.00 | — | 1385 | 13.45 |
| 32 | minicpm-o-4.5-single-gpu-engine | `track_s_fast_r11` | 1.47 | 1.88 | 0.4 | 0.09 | 261 | 2906 | 21.47 |
| 32 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 3.79 | 26.94 | 4219.3 | 93.53 | 9295 | 2581 | 4.04 |
| 32 | sglang-omni | `speech-mps-off` | 1.61 | 2.37 | 0.0 | 0.00 | — | 2730 | 18.73 |

The engine streams several audio events per request (5.4 on average at C=32). Its largest stall over all runs is 0 up
to C=8, 0.03 s at C=16 and 0.09 s at C=32 (3200 requests). vLLM-Omni's MRv2 profile stalls for seconds at C ≥ 16 (see [baselines.md](baselines.md)). The
frameworks generate slightly different amounts of audio for the same requests (Talker sampling), which throughput
counts as delivered.

## Against the target

The target is 2× the best baseline cell for each C and metric, with the one-sided 95% lower bound at or above 2.0. The
best cell per metric is the configuration selected on the tuning requests, so it can differ from the faster cell shown
above (at C=8 throughput it is `speech-mm32k`, 985.0 frames/s). Point ratio / LCB; bold cells pass.

| C | Throughput | TTFA p50 | TTFA p95 | RTF (mean) |
|---:|---:|---:|---:|---:|
| 1 | 1.43× / 1.42 | **2.34× / 2.33** | **2.55× / 2.50** | 1.47× / 1.46 |
| 2 | 1.16× / 1.15 | 2.00× / 1.95 | **2.09× / 2.04** | 1.18× / 1.17 |
| 4 | 1.07× / 1.06 | **2.24× / 2.22** | **2.68× / 2.63** | 1.10× / 1.08 |
| 8 | 1.14× / 1.12 | **2.36× / 2.33** | **2.75× / 2.67** | 1.16× / 1.15 |
| 16 | 1.15× / 1.13 | **2.19× / 2.14** | **2.96× / 2.85** | 1.17× / 1.15 |
| 32 | 1.22× / 1.21 | 1.59× / 1.55 | **2.99× / 2.92** | 1.24× / 1.23 |

10 of 24 pass, all TTFA. TTFA p50 at C=2 misses narrowly (LCB 1.95); at C=32 it needs about 1.3× more. Throughput and
RTF need 1.4–1.9× more at every C. **Throughput-and-latency 2× gate: FAIL.**

These are development measurements on an L20X with K=5. The planned acceptance runs on an H200 with a fixed K=10,
duration guards and the quality gates; they have not been run.

## Run ranges

Minimum–maximum over the measured runs of each cell. No request failed in any measured run.

| C | Framework | Config | K | Requests / run | Failed | Throughput | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) |
|---:|---|---|---:|---:|---:|---|---|---|---|
| 1 | engine | `track_s_fast_r11` | 5 | 320 | 0 | 394.94–399.99 | 125.5–127.2 | 130.8–134.4 | 0.0630–0.0639 |
| 1 | sglang-omni | `speech-mm32k` | 5 | 320 | 0 | 277.10–280.23 | 378.5–384.9 | 498.7–505.5 | 0.0926–0.0936 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 5 | 320 | 0 | 166.84–199.08 | 293.1–297.6 | 329.6–354.6 | 0.1273–0.1482 |
| 2 | engine | `track_s_fast_r11` | 5 | 320 | 0 | 518.74–524.53 | 189.7–191.0 | 220.9–225.2 | 0.0968–0.0979 |
| 2 | sglang-omni | `speech-mps-on` | 5 | 320 | 0 | 439.78–456.90 | 465.7–484.3 | 626.5–653.2 | 0.1134–0.1184 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 5 | 320 | 0 | 306.15–330.05 | 366.6–394.4 | 447.1–484.9 | 0.1538–0.1655 |
| 4 | engine | `track_s_fast_r11` | 5 | 320 | 0 | 749.03–773.83 | 264.4–269.3 | 304.0–312.8 | 0.1314–0.1360 |
| 4 | sglang-omni | `speech-mps-on` | 5 | 320 | 0 | 699.98–719.34 | 588.2–606.6 | 807.4–849.2 | 0.1442–0.1478 |
| 8 | engine | `track_s_fast_r11` | 5 | 320 | 0 | 1106.09–1134.23 | 356.6–364.5 | 419.0–431.2 | 0.1789–0.1836 |
| 8 | sglang-omni | `speech-mm32k` | 5 | 320 | 0 | 972.19–1003.36 | 872.9–911.5 | 1119.9–1232.2 | 0.2063–0.2137 |
| 8 | sglang-omni | `speech-mps-on` | 5 | 320 | 0 | 1020.19–1040.43 | 835.2–870.3 | 1116.2–1203.4 | 0.1980–0.2033 |
| 16 | engine | `track_s_fast_r11` | 5 | 320 | 0 | 1662.10–1699.34 | 480.0–486.2 | 555.4–579.7 | 0.2386–0.2443 |
| 16 | sglang-omni | `speech-mps-on` | 10 | 320 | 0 | 1386.19–1518.73 | 1083.1–1286.2 | 1518.3–1926.0 | 0.2712–0.2971 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 5 | 320 | 0 | 403.16–416.38 | 1026.0–1126.0 | 1951.8–2053.0 | 1.0029–1.0281 |
| 32 | engine | `track_s_fast_r11` | 5 | 640 | 0 | 2410.63–2452.86 | 647.0–662.8 | 777.7–806.2 | 0.3321–0.3380 |
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
| Speech quality gates (WER, SIM, UTMOS vs the frozen baseline reference, 1088 rows × 3 seeds at C=1 and C=32) | **Not run** for `track_s_fast_r11`. Development run of `track_s_fast` below: passes the absolute floors, UTMOS 0.13 below the reference |
| Multimodal accuracy (MMMU, MMSU, Daily-Omni) | Evaluators and references frozen; engine gates not run |
| Robustness (C=32, 30 min, cancellation, overload, long prefills, memory ±1%) | Passed on `track_s_fast` (33,291 requests, 0 invalid, max stall 0.18 s); not repeated on `track_s_fast_r11` |

## Speech quality (development)

`track_s_fast` (the profile without the three options of `track_s_fast_r11`), all 1088 rows, seed 0
(`quality/engine_dev`), against the frozen reference: the best value over the qualified baselines' full-set runs
(1088 rows × 3 seeds, `quality/speech_qref_final.json`).

| | Rows | WER (filtered) | Catastrophic rows | SIM | UTMOS | Silence / repetition / truncation |
|---|---:|---:|---:|---:|---:|---|
| Engine `track_s_fast`, C=1 | 1088 | 1.40% | 2 | 48.93 | 4.166 | 0 / 0 / 0 |
| Engine `track_s_fast`, C=32 | 1088 | 1.37% | 2 | 48.88 | 4.166 | 0 / 0 / 0 |
| Baseline reference (best per metric) | 1088 × 3 | 1.56% (corpus) | 4 | 49.56 | 4.298 | 0 / 0 / 0 |
| Absolute floors | | ≤ 2.69% | | ≥ 41.58 | ≥ 4.163 | |

The engine passes the absolute floors; its UTMOS is 0.13 below SGLang-Omni's (4.30–4.31) and 0.03–0.04 below vLLM-Omni's
turn profiles (4.20). The 8-frame first-chunk ramp accounts for about 0.025 of it (320-row ablation). Thinker greedy
text equals the reference on 99.9% of rows. The comparative gates need all three seeds and run on the exact headline
candidate; they have not been run.

Test-suite results, the end-to-end smoke of this repository and the robustness acceptance are in [testing.md](testing.md).

## Evidence

- `evidence/20261019-l20x/<candidate>_c<C>/`: each cell's `experiment.json` (the frozen spec, fingerprints and the
  server launch), `result.json` (verification, manifest, run summaries), each run's `summary.json` and gzipped
  `requests.jsonl`, and the gzipped server logs. Audio and wire dumps stay with the raw cells.
- `evidence/20261019-l20x/manifest.json`: every file's SHA-256 and its source file.
- `evidence/20261019-l20x/campaign/`: the frozen baseline envelope (`envelope.json`), the tuning winners
  (`winners.json`), the engine-versus-envelope ratios computed on the raw cells (`r11c_vs_envelope.json`,
  `r11c_envelope.log`) and the cell verification log.
- `evidence/20261019-l20x/tuning/`: the 42 tuning cells of the 7 qualified baseline configurations (no server logs).
- `evidence/20261019-l20x/traces/`: baseline stage-trace summaries (`tools/trace_baselines.py`).
- `evidence/20261019-l20x/quality/`: the engine development quality reports with per-row records, the baseline
  qualification records and the frozen speech reference.
- `evidence/20261019-l20x/robustness_track_s_fast_8338bda/`: the robustness acceptance report and its replay.
- The engine cells ran from a frozen worktree at the commit recorded in each `experiment.json` (the code of this
  snapshot's `track_s_fast_r11` path, before the repository was extracted).
