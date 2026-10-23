# Optimization log

Chronological record of the engine's development, 2026-10-14 to 2026-10-23, on L20X GPUs. "Envelope" is the best
baseline cell per concurrency and metric; "pass" is a ratio whose one-sided 95% lower bound reaches 2.0 (24 cells:
6 concurrencies × 4 metrics). Full measurements: [engine.md](engine.md).

| Step | Change | Served result | Envelope passes |
|---|---|---|---:|
| 0 | Harness, metrics, baselines pinned; effective inputs checked (prompt token IDs and reference audio bytes equal across stacks) | — | — |
| 1–3 | Baseline campaign: class-R qualification, per-C tuning on tuning requests, evaluation on separate requests; frozen speech and multimodal quality references | Envelope frozen | — |
| 2 | Architecture spike: one process with a custom paged FlashInfer runtime, against an embedded vLLM runner and two processes | Layout (a) chosen ([architecture_decision.md](architecture_decision.md)) | — |
| 3–4 | Thinker, Talker, Token2Wav ports with parity tests; Track S engine loop; graph decoders; batched Token2Wav | First end-to-end cells | — |
| 4 | Token2Wav in a worker process under MPS | C=8 smoke 374–419 → 525–699 frames/s | — |
| 4–5 | Fast DiT solve, FP16 flow, first-chunk ramp, 4 feature processes instead of 12 (15 → 7 GPU processes) | Headline `track_s_fast` | — |
| 6–7 | Scheduling: work-conserving EDF worker, first-chunk lane, bounded prefills, chunked long prompts, KV backpressure | Lower stall (C=32 run max 0.27 → 0.16 s) | — |
| 8 | Scheduling ablation with a rule fixed in advance; memory repairs for the 30-minute soak (TorchCodec leak, jemalloc in feature processes) | C=32 TTFA p95 −11.4% against the earlier policy; soak PASS | 8 |
| 9 | Talker KV committed for a request's lifetime (no deadlock under a full pool); acceptance under a recorded policy | Throughput −0.3 to −2.8%; acceptance PASS (33,291 requests, 0 invalid) | 8 |
| 10 | Stage tracing; TF32 HiFT upsampling stages | Throughput +2.3 to +10.3% at every C; C=32 2050 → 2262 frames/s | 9 |
| 11 | Kernel attribution (CUPTI, GPM, inventory); batched setup encoder; Talker chunk-prefill graphs | Throughput +3.7 to +6.8%; C=32 2353 frames/s | 10 |
| 11 | Copies and FlashInfer plans without stream synchronization; plan-upload race found by the formal sweep and fixed | Throughput +0.5 to +3.6%; C=32 2437 frames/s (`r11c`) | 10 |
| 12 | Clean attribution of the unperturbed server; asynchronous Token2Wav results | Negative (C=8 −3.6%, TTFA p50 +40 ms); not adopted | 10 |
| 13 | HiFT graph buckets above 16 rows | Neutral; not adopted | 10 |
| 14 | Whole-K Talker capture (one graph per K-frame block); matched HiFT comparison | Neutral; not adopted. Found that TF32 HiFT with the batched setup encoder exceeds the HiFT bound on 1 of 190 chunks | 10 |
| 15 | Overlapped engine tick: the Thinker step on its own stream beside the Talker step; first-chunk ticks and ticks above 16 live requests serial (`track_s_fast_r15d`); per-lane graph capture streams | Throughput +1.6 to +7.3% at C=1–16, C=32 unchanged; C=2 TTFA p50 −7%. First speech-quality scoring of `track_s_fast_r11`: UTMOS 0.001–0.003 below the floor (r15 unchanged at C=1) | 11 |
| 16 | B24 repair (stage 0's HiFT transposed convolution back in FP32); UTMOS attribution; three-seed speech quality (`track_s_fast_r16`) | Performance unchanged (throughput +0.5 to +8.0% vs r11c at C=1–16, C=32 −0.7%); absolute floors pass at C=1 and C=32 over three seeds; comparative UTMOS gate fails (4.17 vs 4.30 non-streaming reference) | 11 |
| 17 | 32k served context; merged DiT solve; linear resampling of understanding audio (MMSU 0.5200 vs SGLang 0.5192) | `track_s_fast_r17` (measured with an FP16-autocast HiFT defect, fixed in round 18) | — |
| 18 | Fused setups, lean Talker frame, lean HiFT, fused DiT residuals (r18f); multi-row Thinker prefill graphs; overlap at any load; HF-rounding prefill kernels (r18i); bitwise DiT kernels (r18j); Talker prefill RoPE kernel and HiFT rows (r18n); worker below normal MPS priority, lean Talker codes, batched encoder cache IO (`track_s_fast_r18c1`) | Paired windows at C=32: r18f 2613 → r18i 3267 → r18c1 ~3640–3710 frames/s; r18f vs r16 C=8/16 +17 to +23%; TTFA p50 C=32 ~0.42 s. Not yet accepted | — |

## Lessons recorded on the way

- **A faster component is not a faster server.** HiFT buckets (4–19% faster in isolation) and whole-K capture (fewer
  launches) were both neutral served. Check the component's share of the served critical path first.
- **Measure the unperturbed server.** CUPTI cost 9–20% of throughput and made SM utilization read ~50%; spans-only
  and untraced controls read ~67%.
- **Removing a synchronization can remove an ordering.** Host-side FlashInfer plan inputs removed the implicit wait that
  had ordered the asynchronous plan-buffer uploads. Two plans of one wrapper could then race, and only the formal sweep
  caught it (13 of 30 runs with failed requests).
- **Justify every process by measurement.** 12 feature processes were no faster than 4; 15 GPU processes became 7.
- **Combined options need their own numerical check.** TF32 HiFT and the batched setup encoder each pass the HiFT bound
  and fail it together.
- **Concurrent graph replays need separate capture streams.** cuBLAS keeps a workspace per (handle, stream) and a
  captured GEMM keeps its capture stream's. Two lanes captured on torch's shared capture stream share one workspace;
  replaying them concurrently hung the GPU.
- **Overlap helps only where the GPU has room.** The overlapped tick gained 2–7% up to C=16 and lost 3% at C=32, where
  the Token2Wav worker is GPU bound and the extra, smaller engine steps take GPU time from it. A paired profile gave
  the gate's real firing rate (17% at 24 live requests), which the K=5 cell alone did not.
- **Score every candidate's quality, including the earlier headline.** `track_s_fast_r11` had performance evidence but
  had never been scored; its first scoring misses the UTMOS floor.
- **A single seed is not a quality verdict near a floor.** Padding the Talker's chunk prefills changed the bf16 rounding,
  150 of 1088 rows sampled other codes and seed 0 fell 0.0008 under the UTMOS floor; the other seeds and the pooled
  score pass. Attribute by ablation at C=1, where the output is deterministic, before changing precision.
- **Declare the frozen references in the sweep spec before running a gate candidate.** The comparative gate accepts only
  cells whose experiment names the frozen Q_ref (`frozen_refs.speech_qref`).
- **At C=32 the engine tick is the throughput.** The benchmark is closed-loop, so throughput follows a request's
  duration, which the tick sets. After the worker got lighter, worker-only speedups measured neutral (the work-conserving
  worker turns saved time into more, smaller calls), while letting the engine's kernels go first under MPS and removing
  engine host work each gained ~4%. An option rejected under one bottleneck (the Talker prefill's RoPE kernel, −1 to
  −3% while the worker bounded C=32) can help under the next (+2%).
- **Concurrent paired-window scripts need distinct ports.** `serve_engine.sh` keys its MPS directory on the port, so two
  scripts on the same ports removed each other's MPS daemons; one window then ran without MPS at ~2300 instead of ~3350
  frames/s and still exited cleanly. Check the GPU inventory for the MPS server before using a window.
