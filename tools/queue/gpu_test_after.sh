#!/usr/bin/env bash
# Run GPU pytest targets on one GPU once a PID has exited and the GPU is idle.
# Usage: gpu_test_after.sh <wait-pid> <gpu> <log> <pytest args...>
set -uo pipefail
WAIT=$1; G=$2; LOG=$(realpath -m "$3"); shift 3
while kill -0 "$WAIT" 2>/dev/null; do sleep 20; done
for i in $(seq 60); do nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
cd /workspace/cgzheng/cgzheng/qwen3_omni; source scripts/env.sh
echo "[$(date +%T)] start $*" > "$LOG"
CUDA_VISIBLE_DEVICES=$G ENGINE_GPU_TESTS=1 timeout 3600 .venv-engine/bin/python -m pytest "$@" -q -s -p no:cacheprovider >> "$LOG" 2>&1
echo "[$(date +%T)] rc=$?" >> "$LOG"
