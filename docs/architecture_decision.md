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
  seen, and run a second scheduler beside vLLM's. Reusing vLLM's serving pipeline in that way is ruled out by the
  plan.
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
