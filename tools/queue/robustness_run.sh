#!/usr/bin/env bash
# Start the engine server from a worktree on one GPU under MPS, run tools/robustness.py against it, stop the server.
# Usage: robustness_run.sh <wait-pid|0> <gpu> <worktree> <config-name> <port> <out-dir> [robustness args...]
set -uo pipefail
WAIT=$1; G=$2; W=$3; CFG=$4; PORT=$5; OUT=$(realpath -m "$6"); shift 6
[ "$WAIT" != 0 ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 20; done
for i in $(seq 60); do nvidia-smi -i "$G" --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
mkdir -p "$OUT"; cd "$W"; source scripts/env.sh
ENGINE_MPS=1 CUDA_VISIBLE_DEVICES=$G setsid bash scripts/serve_engine.sh "$CFG" "$PORT" > "$OUT/server.log" 2>&1 &
SP=$!
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do sleep 10; kill -0 $SP 2>/dev/null || { echo "server died"; exit 1; }; done
echo "[$(date +%T)] robustness start"
/workspace/cgzheng/cgzheng/qwen3_omni/.venv/bin/python tools/robustness.py --url "http://127.0.0.1:$PORT" --gpu "$G" --config "$W/configs/engine/$CFG.json" \
  --out "$OUT" "$@" > "$OUT/run.log" 2>&1
echo "[$(date +%T)] robustness rc=$?"
kill -TERM $SP; for i in $(seq 30); do kill -0 $SP 2>/dev/null || break; sleep 2; done
