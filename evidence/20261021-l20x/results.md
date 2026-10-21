# 2026-10-21 · L20X

Engine `track_s_fast_r16` (round 16: `track_s_fast_r15d` with the B24 HiFT repair; six cells, K=5, measured
2026-10-21 on GPU0 and GPU1) on one L20X each, Seed-TTS English evaluation requests, against the frozen baseline
envelope of [20261019-l20x](../20261019-l20x/results.md) (named in [baselines.json](baselines.json)).

- Tables and figure: [docs/current-results.md](../../docs/current-results.md), rebuilt with
  `python tools/report_results.py evidence/20261021-l20x`.
- Target ratios against the envelope (computed on the raw cells): [campaign/r16_vs_envelope.json](campaign/r16_vs_envelope.json);
  11 of 24 reach LCB 2.0, all TTFA.
- All six cells pass `bench.verify` on their full artifacts ([campaign/r16_verify.log](campaign/r16_verify.log)); no
  measured run has a failed request.
- Speech quality, three seeds at C=1 and C=32: [quality/engine_final](quality/engine_final). The absolute floors pass
  at both C (UTMOS 4.166 / 4.167 against 4.163); the comparative UTMOS gate is reported and fails against the
  non-streaming reference, which by the 2026-10-21 decision does not decide acceptance.
- Robustness acceptance (policy v1, C=32, 30 minutes, 28f8584 on GPU2): PASS, replay equal
  ([robustness_track_s_fast_r16_28f8584](robustness_track_s_fast_r16_28f8584)).
- Every file and its source: [manifest.json](manifest.json).
