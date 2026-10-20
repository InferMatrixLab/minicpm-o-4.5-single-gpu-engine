# Benchmark harness

One client drives vllm-omni, sglang-omni and the new engine with the same
Seed-TTS-eval EN voice-clone workload.  Every metric is computed from client
`time.perf_counter()` timestamps.

| Module | Role |
|---|---|
| `workload.py` | Seed-TTS rows, canonical request, disjoint warmup/tuning/eval ID pools, per-request seeds |
| `adapters/` | Map a canonical request to each engine's native API; parse its SSE events or JSON body |
| `audio.py` | Strict WAV decoding, -50 dBFS silence, per-engine audio assembly with retransmission detection |
| `client.py` | Timestamped request execution; stops reading at `[DONE]`; `replay()` runs recordings through the same path |
| `metrics.py` | TTFA, RTF, duration, playback stalls per request; frames/s = 25·ΣD/W and percentiles per run |
| `protocol.py` | Closed loop at fixed C, warmup + discarded run 0 + K runs, cache policy (`restart` / `flush`), raw wire capture |
| `servers.py` | Start and stop baseline servers on a reserved GPU, one process group each |
| `manifest.py` | Reproducibility manifest and validator |
| `stats.py` | Complete-run bootstrap ratio estimator, final (K=10) and development (K=5..10) validators, duration guard |
| `eligibility.py` | Class-R classification and envelope validation |
| `effective_inputs.py`, `effective_dump.py`, `effective_report.py` | Effective-input equality across stacks |
| `cli.py` | `smoke`, `cell`, `sweep`, `validate-manifest` |

Effective-input rule: prompt token IDs and the reference file bytes must match
the canonical inputs exactly.  A baseline's own reference preprocessing is
recorded as a documented difference.  The new engine must reproduce the HF
reference path exactly.

Run the unit tests with `.venv/bin/python -m pytest tests/bench`, and the
vllm-omni native-client cross-check with the vllm-omni environment:
`"$OMNI_VLLM_REPO"/.venv/bin/python -m pytest tests/bench/test_vllm_native_crosscheck.py`.
