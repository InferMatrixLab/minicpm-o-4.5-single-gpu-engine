# 2026-10-19 · L20X

Engine `track_s_fast_r11` (six cells, K=5, measured 2026-10-18/19) and the frozen baseline envelope cells (vLLM-Omni
`2ec5f26`, SGLang-Omni `22d4ace`, measured 2026-10-15/16) on one L20X each, Seed-TTS English evaluation requests.

- Tables and figure: [docs/current-results.md](../../docs/current-results.md), rebuilt with
  `python tools/report_results.py evidence/20261019-l20x`.
- Target ratios against the envelope (computed on the raw cells): [campaign/r11c_vs_envelope.json](campaign/r11c_vs_envelope.json).
- Every file and its source: [manifest.json](manifest.json).

All 17 cells pass `bench.verify` on their full artifacts (audio, wire dumps and traces included), rechecked on the
recording host on 2026-10-20; `campaign/r11c_verify.log` is the original engine check. No measured run has a failed
request.

Also here:

- `tuning/`: the 42 tuning cells (7 qualified baseline configurations × 6 concurrencies, K=2) that selected the
  envelope, with their own `manifest.json`; summarized in [docs/baselines.md](../../docs/baselines.md).
- `traces/`: per-stage trace summaries of both baselines at C = 1/8/16/32 (`tools/trace_baselines.py`).
- `quality/`: the engine's development speech-quality reports (`track_s_fast`, 1088 rows, seed 0, C=1 and C=32) with
  per-row records, the nine baseline qualification records and the frozen speech reference.
- `robustness_track_s_fast_8338bda/`: the 30-minute C=32 robustness acceptance report and its replay.
