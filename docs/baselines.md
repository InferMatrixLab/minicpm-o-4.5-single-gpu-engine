# Baselines: what was measured and why some cells are missing

Both baselines were run unmodified from pinned checkouts on the same L20X host: vLLM-Omni
[`2ec5f26`](https://github.com/vllm-project/vllm-omni/commit/2ec5f265c69520cdd5e5f0a19db485d38885843e) and SGLang-Omni
[`22d4ace`](https://github.com/sgl-project/sglang-omni/commit/22d4ace88f9547777b819a69d7691204bb9ce847). All numbers
here come from [`evidence/20261019-l20x`](../evidence/20261019-l20x) (`tuning/`, `traces/`, `quality/`) and are
rebuilt by `tools/report_results.py`.

## Selection protocol

1. **Candidates.** Every shipped MiniCPM-o configuration, each with and without MPS: three SGLang-Omni variants
   (`speech-mps-off`, `speech-mps-on`, `speech-mm32k`) and six vLLM-Omni variants (`minicpmo_4_5_turn`, `_turn_mrv2`,
   `_turn_mrv2_h200`, each MPS off and on).
2. **Class R.** Each candidate must send the same prompt token IDs and reference audio bytes as the reference path.
   All nine do.
3. **Quality qualification.** 320 tuning rows × 3 seeds at C=8 (960 generations) against absolute floors (filtered WER ≤ 2.69%, SIM ≥ 41.58,
   UTMOS ≥ 4.163). Two vLLM-Omni candidates fail on UTMOS and are excluded (table below).
4. **Tuning.** Each of the 7 qualified candidates at every C on the tuning requests (K=2).
5. **Selection.** Per C and metric, the best tuning value wins (`campaign/winners.json`).
6. **Evaluation.** Only the winners are measured on the evaluation requests (K=5; K=10 for SGLang-Omni at C=16).
   Those cells form the envelope the engine is compared with.

**This is why vLLM-Omni has no evaluation cell at C=4 and C=8:** during tuning SGLang-Omni won every metric at those
concurrencies, so no vLLM-Omni configuration was carried into the evaluation. It was measured there, on the tuning
requests; the full table is below.

## Quality qualification (320 tuning rows × 3 seeds at C=8, pooled)

| Framework | Config | Filtered WER | SIM | UTMOS | Qualified |
|---|---|---:|---:|---:|---|
| sglang-omni | `speech-mm32k` | 1.39% | 49.13 | 4.310 | yes |
| sglang-omni | `speech-mps-off` | 1.40% | 48.76 | 4.304 | yes |
| sglang-omni | `speech-mps-on` | 1.33% | 49.17 | 4.306 | yes |
| vllm-omni | `minicpmo_4_5_turn-mps-off` | 1.39% | 48.90 | 4.201 | yes |
| vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 1.31% | 48.93 | 4.200 | yes |
| vllm-omni | `minicpmo_4_5_turn_mrv2-mps-off` | 2.32% | 49.04 | 4.165 | yes |
| vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 2.39% | 48.94 | 4.164 | yes |
| vllm-omni | `minicpmo_4_5_turn_mrv2_h200-mps-off` | 2.49% | 49.03 | 4.161 | **no** (UTMOS floor 4.163) |
| vllm-omni | `minicpmo_4_5_turn_mrv2_mps-mps-on` | 2.43% | 48.88 | 4.154 | **no** (UTMOS floor 4.163) |

The frozen speech reference for the engine's comparative gates is the best value over the qualified candidates'
full-set runs (1088 rows × 3 seeds): WER 1.557%, 4 catastrophic rows, SIM 49.56, UTMOS 4.298
(`quality/speech_qref_final.json`).

## Every qualified configuration on the tuning requests

K=2 runs of 320 requests (384 at C=32) on the tuning pool. These values selected the winners; they are not the
evaluation numbers in the README, which use another request pool and K=5.

| C | Framework | Config | Throughput (frames/s) | TTFA p50 (ms) | TTFA p95 (ms) | RTF (mean) | Stall mean (ms) | Stall max (s) |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 1 | vllm-omni | `minicpmo_4_5_turn-mps-off` | 196.3 | 301.8 | 337.6 | 0.1291 | 0.0 | 0.00 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 194.1 | 300.7 | 336.8 | 0.1305 | 0.0 | 0.00 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mrv2-mps-off` | 194.5 | 395.6 | 434.7 | 0.1318 | 0.0 | 0.00 |
| 1 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 197.4 | 384.0 | 422.6 | 0.1299 | 0.0 | 0.00 |
| 1 | sglang-omni | `speech-mm32k` | 270.7 | 386.1 | 508.3 | 0.0959 | 0.0 | 0.00 |
| 1 | sglang-omni | `speech-mps-off` | 265.2 | 394.8 | 517.3 | 0.0980 | 0.0 | 0.00 |
| 1 | sglang-omni | `speech-mps-on` | 256.3 | 407.5 | 536.8 | 0.1013 | 0.0 | 0.00 |
| 2 | vllm-omni | `minicpmo_4_5_turn-mps-off` | 305.8 | 393.0 | 518.5 | 0.1663 | 0.0 | 0.00 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 312.3 | 379.3 | 460.5 | 0.1627 | 0.0 | 0.00 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mrv2-mps-off` | 263.1 | 512.5 | 666.0 | 0.1980 | 0.0 | 0.00 |
| 2 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 284.6 | 436.3 | 592.7 | 0.1831 | 0.0 | 0.00 |
| 2 | sglang-omni | `speech-mm32k` | 424.9 | 497.7 | 655.5 | 0.1227 | 0.0 | 0.00 |
| 2 | sglang-omni | `speech-mps-off` | 431.1 | 488.5 | 647.7 | 0.1206 | 0.0 | 0.00 |
| 2 | sglang-omni | `speech-mps-on` | 437.6 | 479.2 | 634.9 | 0.1189 | 0.0 | 0.00 |
| 4 | vllm-omni | `minicpmo_4_5_turn-mps-off` | 355.0 | 893.7 | 1047.9 | 0.2935 | 0.0 | 0.00 |
| 4 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 383.1 | 715.9 | 963.9 | 0.2699 | 0.0 | 0.00 |
| 4 | vllm-omni | `minicpmo_4_5_turn_mrv2-mps-off` | 331.6 | 786.4 | 974.3 | 0.3121 | 0.0 | 0.00 |
| 4 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 353.2 | 632.6 | 940.5 | 0.2932 | 0.0 | 0.00 |
| 4 | sglang-omni | `speech-mm32k` | 636.0 | 655.3 | 890.4 | 0.1633 | 0.0 | 0.00 |
| 4 | sglang-omni | `speech-mps-off` | 642.3 | 654.8 | 883.5 | 0.1615 | 0.0 | 0.00 |
| 4 | sglang-omni | `speech-mps-on` | 687.8 | 602.0 | 827.9 | 0.1512 | 0.0 | 0.00 |
| 8 | vllm-omni | `minicpmo_4_5_turn-mps-off` | 373.6 | 1188.8 | 1633.0 | 0.5460 | 1.3 | 0.25 |
| 8 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 405.5 | 1048.1 | 1527.6 | 0.5025 | 1.0 | 0.27 |
| 8 | vllm-omni | `minicpmo_4_5_turn_mrv2-mps-off` | 355.9 | 986.2 | 1595.5 | 0.5733 | 8.3 | 0.41 |
| 8 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 385.4 | 941.5 | 1545.5 | 0.5292 | 7.0 | 0.41 |
| 8 | sglang-omni | `speech-mm32k` | 980.8 | 890.1 | 1130.3 | 0.2096 | 0.0 | 0.00 |
| 8 | sglang-omni | `speech-mps-off` | 937.6 | 942.3 | 1185.6 | 0.2215 | 0.0 | 0.00 |
| 8 | sglang-omni | `speech-mps-on` | 980.8 | 871.3 | 1185.0 | 0.2098 | 0.0 | 0.00 |
| 16 | vllm-omni | `minicpmo_4_5_turn-mps-off` | 382.9 | 1860.6 | 2865.9 | 1.0629 | 74.0 | 1.39 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 417.6 | 1584.5 | 2320.6 | 0.9706 | 39.2 | 1.41 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mrv2-mps-off` | 362.9 | 1181.3 | 2077.4 | 1.1330 | 723.2 | 21.40 |
| 16 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 407.4 | 1022.9 | 1934.9 | 1.0188 | 628.6 | 16.27 |
| 16 | sglang-omni | `speech-mm32k` | 1272.9 | 1288.4 | 1900.7 | 0.3229 | 0.0 | 0.00 |
| 16 | sglang-omni | `speech-mps-off` | 1307.6 | 1293.0 | 1766.2 | 0.3168 | 0.0 | 0.00 |
| 16 | sglang-omni | `speech-mps-on` | 1367.0 | 1223.7 | 1675.2 | 0.3010 | 0.0 | 0.00 |
| 32 | vllm-omni | `minicpmo_4_5_turn-mps-off` | 397.1 | 3619.7 | 5045.3 | 2.0408 | 1684.7 | 4.44 |
| 32 | vllm-omni | `minicpmo_4_5_turn_mps-mps-on` | 425.1 | 3396.7 | 5131.2 | 1.8993 | 1305.1 | 3.61 |
| 32 | vllm-omni | `minicpmo_4_5_turn_mrv2-mps-off` | 369.4 | 1222.7 | 3184.2 | 2.1654 | 4521.3 | 53.44 |
| 32 | vllm-omni | `minicpmo_4_5_turn_mrv2_h200_mps-mps-on` | 402.5 | 1023.6 | 3374.0 | 1.9988 | 4221.5 | 66.87 |
| 32 | sglang-omni | `speech-mm32k` | 1682.9 | 1957.6 | 2807.4 | 0.4952 | 0.0 | 0.00 |
| 32 | sglang-omni | `speech-mps-off` | 1862.4 | 1777.8 | 2515.5 | 0.4422 | 0.0 | 0.00 |
| 32 | sglang-omni | `speech-mps-on` | 1799.5 | 1861.9 | 2517.9 | 0.4623 | 0.0 | 0.00 |

## Why vLLM-Omni stops scaling at about 400 frames/s

Every vLLM-Omni configuration delivers 330–425 frames/s from C=4 to C=32, about 16 seconds of audio per second. With
more streams than that rate covers, each stream gets slower than real time: RTF reaches ~1.0 at C=16 and ~2.0 at C=32,
playback stalls appear (MRv2 profiles: up to 67 s in one request), and TTFA grows with C. Its TTFA p50 at C=16–32
stays the best among the baselines only in the MRv2 profiles, which start first audio early and then stall.

The stage trace (`tools/trace_baselines.py`, `minicpmo_4_5_turn.yaml`, `traces/vllm_c*`; p50 ms from arrival; the GPU
columns come from a separate window under the torch profiler and are diagnostic) places the limit in Stage 2,
Code2Wav:

| C | frames/s | Thinker TTFT | Thinker done | Talker first | Talker done | Code2Wav first | Code2Wav done | GPU s per audio s (stage 0 / 1 / 2) |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 177 | 25 | 86 | 133 | 416 | 301 | 544 | 0.0150 / 0.0199 / 0.0318 |
| 8 | 376 | 49 | 160 | 295 | 768 | 1555 | 2112 | 0.0095 / 0.0058 / 0.0342 |
| 16 | 397 | 52 | 181 | 348 | 972 | 2357 | 4247 | 0.0037 / 0.0023 / 0.0280 |
| 32 | 409 | 64 | 231 | 376 | 889 | 3877 | 8125 | 0.0077 / 0.0036 / 0.0318 |

- At C=32 the Talker has a request's codec tokens after ~0.9 s, but Code2Wav emits the first audio after 3.9 s and the
  last after 8.1 s: requests queue in front of Stage 2.
- Stage 2 runs outer-eager (`enforce_eager: true`) and launched 1.5 million kernels for 283 s of audio in the C=32
  kernel window (5.8 s of CPU launch time, 9.0 s of GPU kernel time over a 17.5 s span). Its HiFT graphs are captured
  for batch 1 only (`hift_graph_exact_batch_sizes: [1]`). So Stage 2 is launch-bound, not GPU-bound: at 0.032 GPU-s
  per audio-second its GPU work alone would allow about 31 audio-seconds per second.
- SGLang-Omni's whole pipeline needs 0.009 GPU-s per audio-second at C=32 in the same kind of window.

## SGLang-Omni does not stream audio

Turn-mode MiniCPM-o speech in SGLang-Omni at `22d4ace` returns one terminal waveform per request (1 audio event per
request in every measured run; vLLM-Omni sends ~4.5, the engine ~5.4). Its TTFA is therefore its end-to-end latency, and
its stall and chunk-gap columns are empty. This is why its TTFA is 2–3× the engine's even where its throughput is close:
the envelope's TTFA winners at C=4, C=8 and part of C=16–32 are non-streaming cells. The harness records these cells as
non-streaming (`bench/adapters/sglang_omni.py`); a streaming SGLang-Omni would have a lower TTFA and the TTFA ratios
would shrink.
