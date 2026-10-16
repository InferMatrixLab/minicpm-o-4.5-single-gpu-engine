#!/usr/bin/env bash
# Run an engine evaluation sweep from a frozen worktree once a given PID has exited and the GPU is idle.
# Usage: engine_eval_after.sh <wait-pid> <gpu> <worktree> <spec (relative to the worktree)> <out-dir>
set -uo pipefail
WAIT=$1; G=$2; W=$3; SPEC=$4; OUT=$5
[ "$WAIT" != 0 ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 20; done
for i in $(seq 60); do nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
cd "$W"; source scripts/env.sh
echo "[$(date +%T)] sweep start ($W $SPEC)"
timeout 14400 /workspace/cgzheng/cgzheng/qwen3_omni/.venv/bin/python -m bench.cli sweep --spec "$SPEC" --out "$OUT"
echo "[$(date +%T)] sweep rc=$?"
