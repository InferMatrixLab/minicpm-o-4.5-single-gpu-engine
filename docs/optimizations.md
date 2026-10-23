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

### Round 17-18 options (candidate `track_s_fast_r18c1`, not yet accepted)

Served effects are time-paired windows (spans-only, C=16 and C=32), each step against the previous candidate; see
[engine.md](engine.md) round 18. Every row is bitwise equal to the path it replaces unless the qualification says otherwise.

| Optimization | How it reduces work or overhead | Served effect (C=32) | Qualification |
|---|---|---|---|
| Merged DiT solve (`t2w_merge_solve`) | One packed solve for a decode call's window groups instead of one per group | Part of r17 | Bitwise; the FP16-autocast HiFT defect it exposed is fixed (3fa580c) |
| Fused prompt setups (`t2w_fuse_setup_frames`) | Pending setups ride the decode call's solve | TTFA p50 −6 to −11%, throughput unchanged | Bitwise |
| Lean Talker frame (`talker_lean_frame`) | Strided q/k/v views, in-place residual GEMMs, one-kernel counter RNG | Frame graph −17% isolated | Bitwise |
| Lean HiFT, fused DiT residuals (`t2w_hift_lean`, `t2w_dit_fused_residual`) | One Snake kernel, folded weight norms; residual add fused with the next norm | +2.5 to +5.1%, TTFA −4 to −9% | DiT bitwise; HiFT within its run-to-run spread |
| Thinker multi-row prefill graphs (`thinker_chunk_prefill_graphs`) | 1-4-row prompt prefills replay graphs | +3.8% | Graph == eager of the same padded batch |
| Overlap at any load (`tick_overlap_max_live` null) | No serial ticks above 16 live requests | +10 to +12.6%, TTFA −5 to −7% | Scheduling only |
| HF-rounding prefill kernels (`prefill_hf_kernels`) | The decode's fused Triton kernels in every prefill | +7.2% (C=8 +5.4%, C=16 +7.0%) | AC-3.1 within HF's own SDPA noise; changes prefill numerics |
| Fast DiT kernels (`t2w_dit_fused_conv` and two unconditional changes) | q/k norms one program per token in the per-head order; no attention zero-fill; norms write the conv im2col | +3.1%, TTFA −4 to −8% | Bitwise |
| HiFT bias fusion (`t2w_hift_fused_bias`) | ResBlock bias, residual and next Snake in one kernel | Neutral (−0.1%) | Bitwise ResBlocks |
| Setup prefetch (`t2w_setup_prefetch`) | Pending setups' flow encoder while a decode call's GPU work runs | Neutral; TTFA −1.7% (C=16 −2.2%) | Bitwise |
| Talker prefill RoPE kernel (`prefill_rope_hf`) | RoPE + v copy in one kernel | +2.0% | Bitwise |
| HiFT rows by 2 to 16, 20, 24 (`t2w_hift_rows`) | Padding 20.7% → 5.6% of HiFT rows | +1.2% | Within the HiFT bound (graph buckets) |
| Worker below normal MPS priority (`t2w_mps_client_priority` 1) | The engine's kernels go first | +3.7%, TTFA +0.7% | Scheduling only |
| Lean Talker codes (`talker_lean_codes`) | One next-code / live write per block instead of per frame | +4.0% | Same row state (CPU and GPU tests) |
| Batched encoder cache IO (`t2w_encoder_batched_io`) | One gather and one scatter kernel for every stream's conformer cache | In r18c1 | Bitwise |
| All of the last three on r18n (`track_s_fast_r18c1`) | — | **+6.8% (about 3640-3710 frames/s), C=16 +4.5%, TTFA unchanged** | Needs K=5, quality, robustness, multimodal |


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
| Held decodes (`t2w_hold_*`) | C=32 unchanged, C=8 −1 to −3% |
| Fixed Talker K=6 / K=8 | C=32 +1.6 to +2.2%, TTFA p50 +14 to +23% |
| HiFT channels-last | 2–8% of HiFT time isolated, waveform moves ~1e-3; not built |
| GELU in the DiT fc1 GEMM epilogue | 27% of elements change (rel-L2 3.8e-4) for ~0.5 ms a solve; not built |
| Engine below normal MPS priority | C=32 −1.6% |
| Talker stream at the highest CUDA priority | Unchanged (Talker K-step 11.09 vs 11.16 ms) |
| Every tick overlapped (`tick_overlap_first_chunks` true) | C=32 +1.6%, TTFA p50 +3.4% |
| Vendored vLLM runner for the Thinker | Faster decode step in isolation, but per-token hidden states need vLLM's forward context and its fixed memory split; not chosen |

## What the measurements say

Three host-side launch reductions in a row (asynchronous results, HiFT buckets, whole-K capture) measured neutral.
Unperturbed, the GPU's SMs are about two thirds busy at C ≥ 8 with low occupancy (15–22%) and DRAM bandwidth
(21–23%): many small kernels from two processes. The engine tick waits for GPU work twice (Thinker step, then Talker
step), and the worker idles while its decode call runs. [performance-gap.md](performance-gap.md) ranks the remaining
levers.
