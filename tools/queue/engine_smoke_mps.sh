#!/usr/bin/env bash
# In-process engine smoke runs under a private MPS daemon on one GPU (development; the GPU must be otherwise idle).
# Usage: engine_smoke_mps.sh <gpu-index> <out-dir> <config> <c>[,<c>...] [extra engine_smoke args...]
set -uo pipefail
G=$1; OUT=$2; CFG=$3; CS=$4; shift 4
Q=/workspace/cgzheng/cgzheng/qwen3_omni; cd $Q; source scripts/env.sh; mkdir -p $OUT
UUID=$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
M=/tmp/cgz-mps-sm$G; rm -rf $M; mkdir -p $M/pipe $M/log
export CUDA_MPS_PIPE_DIRECTORY=$M/pipe CUDA_MPS_LOG_DIRECTORY=$M/log
CUDA_VISIBLE_DEVICES=$UUID nvidia-cuda-mps-control -d
for c in ${CS//,/ }; do
  n=$((c == 1 ? 8 : 4 * c))
  name=$(basename $CFG .json)_c$c
  CUDA_VISIBLE_DEVICES=$UUID timeout 2400 .venv-engine/bin/python tools/engine_smoke.py --config $CFG --n $n --c $c "$@" > $OUT/$name.log 2>&1
  echo "[$(date +%T)] $name rc=$? $(grep '^{"n"' $OUT/$name.log | cut -c1-200)"
done
echo quit | nvidia-cuda-mps-control; sleep 2; rm -rf $M
