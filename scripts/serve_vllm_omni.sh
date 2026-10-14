#!/usr/bin/env bash
# Usage: serve_vllm_omni.sh <deploy-config.yaml> <port> [extra args...]
# Run under canhazgpu so CUDA_VISIBLE_DEVICES is set.
set -euo pipefail
source "$(dirname "$0")/env.sh"
CFG=$1; PORT=$2; shift 2
cd "$OMNI_VLLM_REPO"
source .venv/bin/activate
case "$CFG" in /*) ;; *) CFG="$OMNI_VLLM_REPO/vllm_omni/deploy/$CFG" ;; esac
exec python -m vllm_omni.entrypoints.cli.main serve "$MODEL_PATH" --omni \
  --served-model-name openbmb/MiniCPM-o-4_5 --host 127.0.0.1 --port "$PORT" \
  --deploy-config "$CFG" --trust-remote-code "$@"
