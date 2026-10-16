#!/usr/bin/env bash
# Rerun full multimodal sets from the campaign worktree on one GPU (their earlier reports recorded an older evaluator).
# Usage: mm_rerun.sh <gpu> <worktree>
set -uo pipefail
G=$1; W=$2; Q=/workspace/cgzheng/cgzheng/qwen3_omni
cd "$W"; source scripts/env.sh
run() {  # name port candidate-args...
  local name=$1 port=$2; shift 2
  echo "[$(date +%T)] start $name"
  $Q/.venv/bin/python -m evals.multimodal.suite run --gpu "$G" --port "$port" --subset full --scope final --concurrency 16 \
    --out "$Q/results/multimodal/full/$name" "$@" > "$Q/results/multimodal/full_${name}.log" 2>&1
  echo "[$(date +%T)] end $name rc=$?"
}
run sglang_mm32k_r2 18641 --adapter sglang-omni --config speech --label mm32k --env OMNI_SGLANG_MAX_SEQ=32768
run vllm_turn_mps-off_r2 18642 --adapter vllm-omni --config minicpmo_4_5_turn.yaml --label mps-off --mps off
