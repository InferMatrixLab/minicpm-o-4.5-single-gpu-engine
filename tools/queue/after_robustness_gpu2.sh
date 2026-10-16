#!/usr/bin/env bash
# GPU2 follow-up: wait for the robustness driver, stop its engine server, then extend the C=16 evaluation cell whose
# TTFA p95 interval exceeds 10% of the estimate (K 5 -> 10) from the campaign worktree.
set -uo pipefail
Q=/workspace/cgzheng/cgzheng/qwen3_omni; DRIVER=$1; SERVER_PORT=$2
while kill -0 "$DRIVER" 2>/dev/null; do sleep 15; done
P=$(ps -eo pid,args | grep "serve_engine.s[h] track_s_fast $SERVER_PORT" | awk '{print $1}')
[ -n "$P" ] && kill -TERM $P; for i in $(seq 30); do [ -n "$P" ] && kill -0 $P 2>/dev/null || break; sleep 2; done
for i in $(seq 60); do nvidia-smi -i 2 --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
cd /workspace/cgzheng/cgzheng/qwen3_omni_runs/f2e15fb && source scripts/env.sh
echo "[$(date +%T)] extend c16"
$Q/.venv/bin/python -m bench.cli extend --cell $Q/results/l20x/campaign/eval_c16/01_sglang-omni_speech-mps-on_ac0e44a46d2d_c16 --k 10 \
  --reason "envelope: ttfa_p95 90% interval exceeds 10% of the estimate at K=5" --port 18530
echo "[$(date +%T)] extend rc=$?"
