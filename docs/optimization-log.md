# Optimization log

Chronological record of the engine's development, 2026-10-14 to 2026-10-19, on L20X GPUs. "Envelope" is the best
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
