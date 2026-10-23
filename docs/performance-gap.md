# Performance gap

The engine meets 2× on first-audio latency at most concurrencies, because a request's first window needs only a short
prefill, one 10-token Thinker chunk, 8 Talker frames and one Token2Wav call. Throughput and RTF are a different matter:
they need the whole pipeline to do twice the audio per second, and at C=32 the engine delivers 1.22× SGLang-Omni.
This page records where the time goes and which levers remain. The measurements are from `tools/engine_profile.py`
(stage spans, CUPTI kernel windows in every GPU process, NVML GPM counters) on `track_s_fast_r11`.

**Round 18 update** (paired spans-only windows; [engine.md](engine.md) round 18). Levers 1, 3 and 4 below were built:
the overlapped tick now runs at any load, multi-row Thinker prefills replay graphs, every prefill uses the fused
HF-rounding kernels, and the worker's DiT, HiFT and setups do less GPU and host work. C=32 went from 2613 (r18f) to
about 3640–3710 frames/s (`track_s_fast_r18c1`, not yet accepted). With the worker lighter, C=32 follows the engine
tick (~24 ms, 4 Talker frames a row): worker-only savings measured neutral, while giving the engine's kernels MPS
priority over the worker's and removing per-frame host work in the Talker each gained ~4%. The device still has a
kernel running only 85% of the time at 65% SM utilization and 21% occupancy; a Talker chunk prefill takes ~7.9 ms of
the tick for ~2 ms of kernels.

## What the GPU is doing

Unperturbed (spans-only control, untraced throughput 2490 frames/s at C=32):

| C | SM utilization | Occupancy | DRAM bandwidth |
|---:|---:|---:|---:|
| 1 | 45.7% | — | — |
| 8 | 64.4% | 15–22% | 21–23% |
| 16 | 65.8% | 15–22% | 21–23% |
| 32 | 67.2% | 15–22% | 21–23% |

The GPU is busy but not saturated: many small, latency-bound kernels from two processes (the engine launches ~400
kernels per Talker frame). In the CUPTI run at C=32 the engine has a kernel running 48% of the time and the worker 68%.
Measurement cost was checked: spans cost 2–3% of throughput and CUPTI 9–20%, so host times come from spans-only runs
and kernel times from CUPTI runs.

## Where the engine tick goes (C=32)

- Kernel time per call: Talker K-step decode 6.9 ms; worker DiT solve 18.9 ms, HiFT 28.1 ms per decode call, setup
  solve 23.1 ms.
- The tick serializes two GPU waits: the Thinker step, then the Talker step. Whole-K Talker capture removed the
  per-frame replays and clones and changed nothing: the K-step stays ~10 ms of host time because the host waits for the
  K frames' GPU work at the copy.
- About half of the tick is still host gaps: the Talker's per-row bookkeeping (~4 ms), multi-row Thinker prefills run
  eagerly (22 ms host against 12 ms of kernels), and the two synchronizations.
- The first text chunk (283 ms of a ~660 ms C=32 first audio) takes 11 Thinker steps at the tick rate, so it scales
  with the tick. This is what holds TTFA p50 at C=32 to 1.59×.

## Where the worker goes (C=32)

- Decode calls (≈6/s × 130 ms) and setups (≈9/s × 27 ms) fill the worker's second.
- A decode call waits once, at its final device-to-host copy, for GPU work that runs beside the engine's under MPS.
- Keeping two calls in flight (asynchronous results) made the worker launch more, smaller calls that queued behind each
  other: negative. HiFT buckets above 16 rows made HiFT 4–19% faster in isolation and moved nothing served.

## GPU work per second of audio (isolated, C=32 rates)

Setups ~1.6 ms plus host time, HiFT 1.9, Thinker 1.9, Talker 1.8, DiT 1.3, flow encoder 0.3, S3 tokens 0.7 (before the
TF32 HiFT and host-copy changes). Doubling throughput needs this roughly halved; no single component is that large.

## Remaining levers, ranked by the measurements

1. **Overlap the tick's two GPU waits.** Run the Thinker step and the Talker step on separate streams, or pipeline the
   next Thinker step behind the current Talker step, so the host does not idle twice per tick.
2. **Codec transport over CUDA IPC.** Windows currently cross as Python integer lists; a device-resident hand-off
   removes a host round trip per window and lets the worker start without the engine's synchronization.
3. **Multi-row Thinker prefill graphs** and graph-captured encoder buckets (22 ms host vs 12 ms kernels per prefill).
4. **The worker's GPU work:** the DiT solve and the setup solve, and fewer, larger decode calls.
5. **Kernel fusion in the Talker frame** (~400 kernels per frame at 15–22% occupancy).
6. **Thinker weight bandwidth:** FP8 missed its accuracy floor (top-1 0.978 against 0.98); NVFP4 needs a GPU that
   supports it.

Each lever is to be bounded before it is built: estimate the largest served gain from the measured times, and stop if
it cannot move the cell beyond the run-to-run spread.
