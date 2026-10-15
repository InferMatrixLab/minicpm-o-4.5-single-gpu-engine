#!/usr/bin/env bash
# Usage: serve_engine.sh <config-name|path.json> <port> [extra args...]   (CUDA_VISIBLE_DEVICES selects the GPU)
set -euo pipefail
source "$(dirname "$0")/env.sh"
CFG=$1; PORT=$2; shift 2
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
case "$CFG" in /*|*.json) ;; *) CFG="$ROOT/configs/engine/$CFG.json" ;; esac
cd "$ROOT"
PY="${ENGINE_PYTHON:-$ROOT/.venv-engine/bin/python}"
[ -x "$PY" ] || PY=/workspace/cgzheng/cgzheng/qwen3_omni/.venv-engine/bin/python  # frozen worktrees share the engine environment
exec "$PY" -m engine.server --host 127.0.0.1 --port "$PORT" --config "$CFG" "$@"
