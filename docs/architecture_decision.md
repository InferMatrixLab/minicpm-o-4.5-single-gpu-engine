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
