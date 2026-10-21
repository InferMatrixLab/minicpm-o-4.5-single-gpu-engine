# Runtime optimizations

Not every feature in the source is enabled by every profile. The table separates the core implementation from options.
Local kernel or component improvements are not counted as service improvements: the "served" column comes from
paired served runs or formal sweeps (K=5 runs, evaluation requests) on one L20X. Each optimization's measurements and
tests are in [engine.md](engine.md); the order in which they were tried is in [optimization-log.md](optimization-log.md).

| Optimization | How it reduces work or overhead | Served effect | Qualification |
|---|---|---|---|
| One process for Thinker and Talker | Paged FlashInfer KV per model; the Talker is conditioned in place on the Thinker's hidden states, with no copy across processes and one scheduler for both | Basis of every profile | Core; Talker conditioning and teacher-forced logits checked against HF (rel-L2 < 1e-4) |
| CUDA-graph decode | Thinker and Talker decode replayed per batch bucket, plans skipped between page crossings; the Talker runs K=4 frames per replay | Basis of every profile | Core; graph == eager exactly |
| Fused Thinker kernels | Triton residual add + RMSNorm, q/k norm + RoPE + v copy, SiLU-multiply, each reproducing the reference rounding with integer round-to-nearest-even | Part of the core | Core; same greedy tokens as the torch reference |
| Token2Wav worker process under MPS | Moves the launch-heavy Token2Wav thread out of the engine's GIL; MPS lets both run kernels concurrently | C=8: 374–419 → 525–699 frames/s, TTFA p50 0.83–0.95 → 0.62–0.70 s (smoke) | Core; 4 feature processes chosen by measurement (equal to 12) |
| Batched streaming Token2Wav | Windows of all streams in one flow encoder, DiT and HiFT call; ragged prompts and merged final windows | Part of the core | Core; batched == per-stream |
| Fast DiT solve | Attention caches as page-size-1 slots (the reference trimming becomes slot bookkeeping), packed ragged rows, graphs per (rows, tokens) bucket, FP16 flow | Part of `track_s_fast` | Approved approximation; mels within the FP16 bound of FP32 |
| First-chunk ramp | First audio window after 8 frames instead of 25 | Lower TTFA | Approved approximation; costs ~0.025 UTMOS in a 320-row ablation |
| First-chunk lane and bounded prefills | New voices' setups first while continuations keep 100 ms of slack; Thinker prefills within 2048 tokens a tick | C=32 vs the earlier policy: TTFA p95 −11.4% (LCB 1.089), p50 −5.1%, stall 0.109 vs 0.142 s | Core (`track_s_fast`); chosen by a rule fixed before the ablation |
| KV lifetime commitments | Thinker and Talker KV granted only for a request's whole lifetime, in admission order | Throughput −0.3 to −2.8% (separate sessions); no admission wait observed | Core; required for correctness under a full pool |
| TF32 HiFT upsampling | Only the upsampling convolutions in TF32; f0 predictor, source and input/output convolutions stay FP32; in `track_s_fast_r16` also stage 0's transposed convolution | Throughput +2.3 to +10.3% at every C (C=32 2050 → 2262 frames/s), TTFA p50 −1.7 to −9.4%; keeping stage 0's transposed convolution in FP32 costs nothing measurable | Option `t2w_hift_decode_tf32`; max chunk error 5.3e-3 alone, 3.2e-3 with the batched setup encoder and stage 0's transposed convolution in FP32 (bound 1e-2) |
| Batched prompt-setup encoder | All prompts of a setup call in one right-padded flow-encoder batch, masked attention keys, caches cut per prompt | With Talker prefill graphs: throughput +3.7 to +6.8% at every C (C=32 2262 → 2353) | Option `t2w_batch_setup_encoder`; served chunks 3.71e-3 (FP16 bound 5e-3) |
| Talker chunk-prefill graphs | Graphs per row count (1–4) and token bucket for prefills that append to existing sequences; first frames sampled in one call | (included in the row above) | Option `talker_prefill_graphs`; equal to eager within row-batching noise |
| Copies without waiting | Queued host-to-device copies instead of `torch.tensor(..., device="cuda")`; FlashInfer plans from host inputs, each waiting only for its predecessor's upload | Throughput +0.5 to +3.6% (C=32 2353 → 2437); the worker's flow call 30.9 → 7.3 ms host | Core; the plan-ordering fix removed an upload race found by the formal sweep |
| Overlapped engine tick | The Thinker step is queued on its own CUDA stream before the Talker step and collected after it, so the tick's two GPU waits overlap; graphs of the two lanes are captured on separate streams (cuBLAS keeps a workspace per stream). Ticks that complete a first text chunk, and ticks with more than 16 live requests, run serially | Throughput +1.6 to +8.0% at C=1–16 against r11c, C=32 unchanged; C=2 TTFA p50 −7% | Option `tick_overlap` (`tick_overlap_first_chunks: false`, `tick_overlap_max_live: 16`); same Talker codes as the serial tick at C=1 (all 1088 durations equal) |

The current profile `track_s_fast_r16` enables these options and the overlapped tick. It passes the numerical bounds
and the speech-quality absolute floors over three seeds; see [current-results.md](current-results.md).

## Measured and not adopted

| Experiment | Result |
|---|---|
| Asynchronous Token2Wav results (`t2w_async_results`) | Negative: C=8 1097 → 1058 frames/s, TTFA p50 +40 ms at C=8–32. More, smaller decode calls queue on a GPU already at 67–69% SM utilization |
| HiFT graph buckets above 16 rows (`t2w_hift_rows`) | Neutral served (within ±1%) although a HiFT call is 4–19% faster in isolation; HiFT is a small part of the decode call |
| Whole-K Talker capture (`talker_whole_k`) | Neutral served; bitwise exact. The K-step stays ~10 ms of host time: the host waits for the K frames' GPU work, not for launches |
| Lazy Thinker rows, adaptive K up to 8 | Unchanged at C=8 and C=32 |
| MPS SM cap on the worker (75%, 60%) | Slower: the Talker speeds up, the worker slows (C=32 2210 / 2177 vs 2288 frames/s) |
| Decode-first worker policy, prefill slack floor | Worse TTFA p95: +13.6% at C=32 (decode first), +16.7% at C=16 (300 ms slack floor); the floor also adds stall |
| FP8 Thinker | Teacher-forced top-1 0.978 against the 0.98 floor; off |
| Vendored vLLM runner for the Thinker | Faster decode step in isolation, but per-token hidden states need vLLM's forward context and its fixed memory split; not chosen |

## What the measurements say

Three host-side launch reductions in a row (asynchronous results, HiFT buckets, whole-K capture) measured neutral.
Unperturbed, the GPU's SMs are about two thirds busy at C ≥ 8 with low occupancy (15–22%) and DRAM bandwidth
(21–23%): many small kernels from two processes. The engine tick waits for GPU work twice (Thinker step, then Talker
step), and the worker idles while its decode call runs. [performance-gap.md](performance-gap.md) ranks the remaining
levers.
