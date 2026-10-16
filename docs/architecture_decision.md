# Architecture decision: single process with a custom paged runtime (layout a)

Status: decided on L20X spike data (task10). Measurements: `results/l20x/spike/{a,b,c}_b{1,8,32}.json`, produced by
`tools/spike/`. Common workload: B requests, 60-token prompts, 30-token replies, 100 Talker frames, Token2Wav windows.

## Candidates

| | (a) custom FlashInfer, one process | (b) vLLM runner embedded for the Thinker | (c) two processes, CUDA IPC (control) |
|---|---|---|---|
| Thinker decode step, B=8 | 27.6 ms eager; graph-captured: see `a_b8.json` notes | 5.25 ms (vLLM, CUDA graphs) | same kernels as (a) |
| Thinker prefill, B=8 | 29.1 ms (eager) | 18.4 ms | same as (a) |
| Talker frame, B=8 | 14.1 ms eager; graph-captured: see notes | not provided by vLLM (custom in either case) | same as (a) |
| Thinker to Talker handoff, B=8 | 0.97 ms (in place; condition build only) | not reachable through vLLM's API | 0.68 ms + process wake-up (CUDA IPC) |
| Token2Wav window | 160 ms (reference stepaudio2 path, fp32, per request) | same | same |
| KV budgeting | one explicit page pool per model, sized by the engine | vLLM reserves `gpu_memory_utilization` of the device at start; Talker, Token2Wav and their graphs must fit in the rest | a fixed split between the processes, decided at launch |
| Forward-context ownership | ours | vLLM's global forward context, model runner and scheduler | ours, per process |
| Graph capture and pool isolation | our own graph pools per component | vLLM captures its own graphs; ours must avoid its pool | isolated by process |
| Scheduling integration | one loop schedules Thinker, Talker and Token2Wav together | vLLM's engine loop owns batching; the Talker and Token2Wav loop runs beside it | two loops plus IPC |
| Per-token hidden states for the Talker | available directly | needs a model-runner hook inside vLLM (this is what vllm-omni does) | available, then copied across |

## Decision

**(a)**: one process, a custom paged FlashInfer runtime for the Thinker and the Talker, and Token2Wav in the same
process and CUDA context.

- (b)'s main advantage is a mature, graph-captured decode step. The engine can reach that with its own graph-captured
  paged decode (`engine/runtime/thinker_decode.py`, measured in the spike notes). (b) would still require owning
  vLLM's forward context to read per-token hidden states. It would also fix the memory split before any request is
  seen, and run a second scheduler beside vLLM's. The plan allows (b) (vLLM's runner embedded in-process for the
  Thinker at TP=1); what it rules out is reusing the vllm-omni / sglang-omni serving runtimes. (b) was not chosen for
  the reasons above, not because it is disallowed.
- (c) shows the process boundary costs little for the handoff itself (0.68 ms). But it gives up cross-component
  scheduling, which is how the engine can prioritise first chunks over continuations. It also splits KV memory
  statically between the processes.
- Performance work follows the trace order (task14 onward). The Token2Wav reference path (about 160 ms per window,
  fp32, batch 1) dominates the first-audio critical path. The Thinker and Talker graph decode come next.

## Revision (round 4, measured on the Track S engine): Token2Wav in its own process

The single-process layout was built first (`engine/runtime/backend_s.py`, Token2Wav on its own thread and CUDA stream).
Under load the engine loop is launch-bound Python on both sides, and the two threads share one GIL:

| Track S smoke, L20X, C=8, 24 requests | Thinker step | Talker tick (K=4) | Token2Wav call | frames/s | TTFA p50 |
|---|---|---|---|---|---|
| one process, Token2Wav thread | 7-8 ms | 11-12 ms | 230-300 ms (about 6 windows) | 374-419 | 0.83-0.95 s |
| Token2Wav worker process, no MPS | 18 ms | 49 ms | (time-sliced contexts) | 296 | 1.22 s |
| Token2Wav worker process, MPS | 8.7-10 ms | 12-14.5 ms | 85-126 ms | 525-699 | 0.62-0.70 s |

- Thinker and Talker stay in one process (layout (a)): paged FlashInfer, CUDA-graph decode, the Talker conditioned in
  place from the Thinker's hidden states.
- Token2Wav (reference-voice features, prompt setups, batched window decodes) runs in a worker process
  (`engine/runtime/t2w_worker.py`). The engine sends token windows (small integer lists) and receives audio. No
  tensors cross the process boundary.
- The worker and the engine must run under CUDA MPS. Without it the two contexts time-slice, and the engine loop slows
  2-4x. The launch starts a private MPS daemon (short pipe path, GPU addressed by UUID).
- Remaining contention is inside each process: feature extraction threads in the worker. These move to their own
  processes (`feature_processes`).

These numbers are indicative only. They were measured on a GPU shared with a scoring job. The baseline-equivalent
measurement is the harness sweep (task13/task17).

## Processes per engine (round 5)

One engine on one GPU runs 7 GPU processes (headline `track_s_fast`). Each was kept only where a measurement shows it is needed:

| Process | GPU memory (L20X) | Why it exists |
|---|---|---|
| Engine: API server, Thinker, Talker | ~57 GB | weights, paged KV, graphs |
| Token2Wav worker | ~33 GB | flow (fast DiT) and HiFT, 65k-slot cache pool. In the engine process, its launch-heavy Python thread contends for the GIL with the Thinker/Talker loop. A separate process under MPS raised C=8 from 374–419 to 525–699 frames/s (round 4). |
| 4 feature processes | ~1.3 GB each | CAMPPlus (CPU ONNX) plus the S3 tokenizer (GPU) for each reference voice, outside the worker's GIL. 4 match 12 (C=32 smoke: 1873 vs 1813 frames/s, TTFA 0.87 vs 0.97 s; C=1 equal), and 24 were no faster than 12: the wait for a voice is the worker loop, not the pool. Running the S3 tokenizer in a thread of the worker, with CPU-only feature processes and no extra CUDA context, measured slower (C=32 1638 frames/s, TTFA 1.0 s; C=1 TTFA +20 ms), so it is not the default. (The worker reads `feature_gpu_tokens: false`; the Track S backend does not forward that key from a config yet, so it is reachable only through the worker's own config.) |
| CUDA MPS server | <0.1 GB | lets the engine and the worker run kernels concurrently |

The previous headline used 12 feature processes (15 GPU processes, ~10 GB more). Feature processes return freed heap to the OS after each job (`malloc_trim`), because their RSS crept up during a 30-minute soak. Feature-process memory (rounds 5-7). The growth over the first few hundred distinct voices (~35-40 MB per process), which failed every 30-minute soak's 1% bound, and the slow decay seen after the round-6 soak's fault scenarios (-1.2% in 4 minutes) are oneDNN's primitive cache: 1,024 shape-keyed entries (the 16 to 24 kHz resampler's conv1d is planned per input length), whose content and size follow the recent mix of voice lengths. Raw FFTs do not grow (round 6's attribution to FFT plans was wrong). The feature processes therefore run with `ONEDNN_PRIMITIVE_CACHE_CAPACITY=0` (no measurable cost, bit-identical features) and a fixed glibc mmap threshold (`MALLOC_MMAP_THRESHOLD_=131072`, so large job buffers do not creep into the heap); the ONNX Runtime CPU arena stays on and is filled before load with a 60 s voice (without the arena each job returns ~40 MB to the OS and takes it back, and RSS swings ±3% with the sampling instant). The 1,200-length shape warm-up is kept. Evidence: `tools/spike/feature_scenario_rss.py` runs the worker's own pool through the robustness scenario order (mixed, cancel, EOS, 48 s fallback, overload, then the sustained rows) and measures per-PID RSS against the reference at the sustain job count of the soak's first 60 s (`results/engine_dev/r7_memory/`: arena off ±1.3-4.1%, arena on +60 s warm 1.0-1.4%, plus oneDNN cache off 0.76-1.21%, plus the mmap threshold 0.64-0.85% over 12,000 jobs but 0.96-1.15% over 32,000, the soak's length, a slow creep with glibc that 2,000 warm-up jobs do not remove). The feature processes therefore run on jemalloc (`feature_allocator`, `LD_PRELOAD` of the system `libjemalloc.so.2` for the spawned feature processes only; a missing library fails the start): 0.58-0.90% per process (total 0.66%) over 32,000 jobs and a plateau after ~9,000, bit-identical features and the same job time (`allocator_equivalence.log`).

The codec tokens reach the worker as Python lists through a multiprocessing queue, not by a GPU-resident handoff. This was measured adequate (a window is 25 integers) but it is not the plan's planned transport (task10).
