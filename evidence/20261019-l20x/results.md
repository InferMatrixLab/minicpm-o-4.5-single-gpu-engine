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
