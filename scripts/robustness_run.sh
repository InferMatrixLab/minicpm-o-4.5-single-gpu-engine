#!/usr/bin/env bash
# Start the engine server from a checkout on one GPU under MPS, run tools/robustness.py against it, stop the server,
# and exit with the checker's status.  The GPU is declared to the checker by UUID (resolved here from the index); the
# checker fails before load if the server's /info reports another device.  The server (its own session) is stopped and
# reaped on every exit path, also when this script is interrupted.
# Usage: robustness_run.sh <wait-pid|0> <gpu-index> <worktree> <config-name> <port> <out-dir> [robustness args...]
# CHECKER (optional): the checker command (default: the repository's tools/robustness.py); tests use a stub.
set -uo pipefail
WAIT=$1; G=$2; W=$3; CFG=$4; PORT=$5; OUT=$(realpath -m "$6"); shift 6
UUID=$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader 2>/dev/null | tr -d ' ')
case "$UUID" in GPU-*) ;; *) UUID=${FAKE_GPU_UUID:-} ;; esac  # nvidia-smi prints its errors on stdout
[ -n "$UUID" ] || { echo "no GPU $G"; exit 2; }
[ "$WAIT" != 0 ] && while kill -0 "$WAIT" 2>/dev/null; do sleep 20; done
for i in $(seq 60); do nvidia-smi -i "$UUID" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q . || break; sleep 5; done
mkdir -p "$OUT"; cd "$W"; source scripts/env.sh
SP=""
stop_server() {
  [ -n "$SP" ] || return 0
  kill -TERM "$SP" 2>/dev/null
  for i in $(seq 30); do kill -0 "$SP" 2>/dev/null || break; sleep 2; done
  kill -0 "$SP" 2>/dev/null && kill -KILL "$SP" 2>/dev/null
  wait "$SP" 2>/dev/null
  SP=""
}
trap 'stop_server; exit 143' TERM INT
trap 'stop_server' EXIT
ENGINE_MPS=1 CUDA_VISIBLE_DEVICES=$UUID setsid bash ${SERVE:-scripts/serve_engine.sh} "$CFG" "$PORT" > "$OUT/server.log" 2>&1 &
SP=$!
echo "$SP" > "$OUT/server.pid"
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; do sleep ${HEALTH_POLL_S:-10}; kill -0 $SP 2>/dev/null || { echo "server died"; exit 3; }; done
echo "[$(date +%T)] robustness start (GPU $G = $UUID)"
${CHECKER:-${BENCH_PYTHON:-.venv/bin/python} tools/robustness.py} --url "http://127.0.0.1:$PORT" --gpu-uuid "$UUID" \
  --config "$W/configs/engine/$CFG.json" --out "$OUT" "$@" > "$OUT/run.log" 2>&1
RC=$?
echo "[$(date +%T)] robustness rc=$RC"
stop_server
exit "$RC"
