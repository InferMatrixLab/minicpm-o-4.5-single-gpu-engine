#!/usr/bin/env bash
# A development quality sweep from a worktree on one GPU, then development scoring of its cell.
# Usage: ablation_run.sh <gpu> <worktree> <spec> <out-dir> <score-out.json> <d_ref.json>
set -uo pipefail
G=$1; W=$2; SPEC=$(realpath -m "$3"); OUT=$(realpath -m "$4"); SCORE=$(realpath -m "$5"); DREF=$(realpath -m "$6")
Q=/workspace/cgzheng/cgzheng/qwen3_omni
cd "$W"; source scripts/env.sh
echo "[$(date +%T)] sweep start"
timeout 7200 $Q/.venv/bin/python -m bench.cli sweep --spec "$SPEC" --out "$OUT"
echo "[$(date +%T)] sweep rc=$?"
cd $Q
cell=$(ls -d $OUT/*_c* | head -1)
.venv/bin/python -m evals.speech.suite score --cell "$cell" --scope development --gpu "$G" --d-ref "$DREF" --out "$SCORE" > "${SCORE%.json}.log" 2>&1
echo "[$(date +%T)] score rc=$?"
