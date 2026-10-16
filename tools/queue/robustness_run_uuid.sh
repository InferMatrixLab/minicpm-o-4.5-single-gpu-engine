#!/usr/bin/env bash
# Start the engine server from a worktree on one GPU under MPS, run tools/robustness.py against it, stop the server.
# The GPU is declared to the checker by UUID (resolved here from the index); the checker fails before load if the
# server's /info reports another device.
# Usage: robustness_run_uuid.sh <wait-pid|0> <gpu-index> <worktree> <config-name> <port> <out-dir> [robustness args...]
set -uo pipefail
WAIT=$1; G=$2; W=$3; CFG=$4; PORT=$5; OUT=$(realpath -m "$6"); shift 6
UUID=$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
[ -n "$UUID" ] || { echo "no GPU $G"; exit 1; }
[ "$WAIT" != 0 ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 20; done
for i in $(seq 60); do nvidia-smi -i "$UUID" --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
mkdir -p "$OUT"; cd "$W"; source scripts/env.sh
ENGINE_MPS=1 CUDA_VISIBLE_DEVICES=$UUID setsid bash scripts/serve_engine.sh "$CFG" "$PORT" > "$OUT/server.log" 2>&1 &
SP=$!
echo "$SP" > "$OUT/server.pid"
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do sleep 10; kill -0 $SP 2>/dev/null || { echo "server died"; exit 1; }; done
echo "[$(date +%T)] robustness start (GPU $G = $UUID)"
/workspace/cgzheng/cgzheng/qwen3_omni/.venv/bin/python tools/robustness.py --url "http://127.0.0.1:$PORT" --gpu-uuid "$UUID" \
  --config "$W/configs/engine/$CFG.json" --out "$OUT" "$@" > "$OUT/run.log" 2>&1
echo "[$(date +%T)] robustness rc=$?"
kill -TERM $SP; for i in $(seq 30); do kill -0 $SP 2>/dev/null || break; sleep 2; done
