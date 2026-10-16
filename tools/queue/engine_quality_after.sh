#!/usr/bin/env bash
# GPU2: after a PID exits, the engine full-set quality sweep (frozen worktree), then development-scope scoring per C.
# Usage: engine_quality_after.sh <wait-pid> <worktree>
set -uo pipefail
WAIT=$1; W=$2; Q=/workspace/cgzheng/cgzheng/qwen3_omni; OUT=$Q/results/engine_sweeps/quality_dev_gpu2
while kill -0 "$WAIT" 2>/dev/null; do sleep 20; done
for i in $(seq 60); do nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
cd "$W"; source scripts/env.sh
echo "[$(date +%T)] quality sweep start"
timeout 10800 $Q/.venv/bin/python -m bench.cli sweep --spec $Q/sweeps/engine/quality_dev_gpu2.json --out $OUT
echo "[$(date +%T)] quality sweep rc=$?"
cd $Q
for c in 1 32; do
  cell=$(ls -d $OUT/*_c$c | head -1)
  echo "[$(date +%T)] score C=$c $cell"
  .venv/bin/python -m evals.speech.suite score --cell "$cell" --scope development --gpu 2 --d-ref results/l20x/campaign/dref/dref_truncation_c$c.json \
    --out results/engine_sweeps/quality_dev_c$c.json > results/engine_sweeps/quality_dev_c$c.log 2>&1
  echo "[$(date +%T)] score C=$c rc=$?"
done
