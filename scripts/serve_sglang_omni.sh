#!/usr/bin/env bash
# Usage: serve_sglang_omni.sh <port> [extra args...]   (runbook seed-tts flags)
set -euo pipefail
source "$(dirname "$0")/env.sh"
PORT=$1; shift
cd "$OMNI_SGLANG_REPO"
source .venv/bin/activate
unset SGLANG_OMNI_TORCH_COMPILE_DEFAULT PYTORCH_ALLOC_CONF
export SGLANG_OMNI_STRICT_PORT=1 SGLANG_OMNI_STARTUP_TIMEOUT=1800
MPS_ARGS=()
if [ "${OMNI_SGLANG_MPS:-0}" = "1" ]; then
  # MPS-on candidate: sglang-omni's own pipeline MPS runtime (config field ``mps``).
  MPS_ARGS=(--mps on)
fi
exec sgl-omni serve --model-path "$MODEL_PATH" --model-name minicpmo \
  --thinker.engine.mem_fraction_static 0.55 --talker.engine.mem_fraction_static 0.15 \
  --thinker.factory.max_seq_len "${OMNI_SGLANG_MAX_SEQ:-8192}" --port "$PORT" "${MPS_ARGS[@]}" "$@"
