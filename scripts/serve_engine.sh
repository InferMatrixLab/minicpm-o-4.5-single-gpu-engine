#!/usr/bin/env bash
# Usage: serve_engine.sh <config-name|path.json> <port> [extra args...]   (CUDA_VISIBLE_DEVICES selects the GPU)
#
# ENGINE_MPS=1: run the server under a private CUDA MPS daemon (the Token2Wav worker process and the engine then run
# kernels concurrently).  The daemon's pipe directory is short (Unix socket paths are limited to ~108 bytes), the GPU
# is addressed by UUID for the daemon and the clients (so a missing daemon cannot redirect work to another GPU), and
# the daemon is quit and its directory removed when the server exits or is terminated.
set -euo pipefail
source "$(dirname "$0")/env.sh"
CFG=$1; PORT=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$CFG" in /*|*.json) ;; *) CFG="$ROOT/configs/engine/$CFG.json" ;; esac
cd "$ROOT"
PY="${ENGINE_PYTHON:-$ROOT/.venv-engine/bin/python}"
[ -x "$PY" ] || PY=python3
if [ "${ENGINE_MPS:-0}" != 1 ]; then
  exec "$PY" -m engine.server --host 127.0.0.1 --port "$PORT" --config "$CFG" "$@"
fi
GPU="${CUDA_VISIBLE_DEVICES:?ENGINE_MPS=1 needs CUDA_VISIBLE_DEVICES}"
case "$GPU" in GPU-*) UUID=$GPU ;; *) UUID=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader | tr -d ' ') ;; esac
D="/tmp/minicpmo-engine-mps-$PORT"
rm -rf "$D"
mkdir -p "$D/pipe" "$D/log"
export CUDA_MPS_PIPE_DIRECTORY="$D/pipe" CUDA_MPS_LOG_DIRECTORY="$D/log"
CHILD=""
cleanup() {
  [ -n "$CHILD" ] && kill -TERM "$CHILD" 2>/dev/null && wait "$CHILD" 2>/dev/null || true
  echo quit | nvidia-cuda-mps-control >/dev/null 2>&1 || true
  sleep 1
  rm -rf "$D"
}
trap cleanup EXIT
trap 'exit 143' TERM INT
CUDA_VISIBLE_DEVICES="$UUID" nvidia-cuda-mps-control -d
export CUDA_VISIBLE_DEVICES="$UUID"
"$PY" -m engine.server --host 127.0.0.1 --port "$PORT" --config "$CFG" "$@" &
CHILD=$!
wait "$CHILD"
