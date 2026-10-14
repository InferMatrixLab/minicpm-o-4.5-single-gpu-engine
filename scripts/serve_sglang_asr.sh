#!/usr/bin/env bash
# Usage: serve_sglang_asr.sh <qwen3-asr|whisper> <port>
# Serves the pinned ASR snapshot with sglang-omni, as its CI does for seed-tts WER.
set -euo pipefail
source "$(dirname "$0")/env.sh"
KIND=$1; PORT=$2; shift 2
case "$KIND" in
  qwen3-asr) M=/workspace/models/hub/models--Qwen--Qwen3-ASR-1.7B/snapshots/7278e1e70fe206f11671096ffdd38061171dd6e5; NAME=Qwen/Qwen3-ASR-1.7B ;;
  whisper)   M=/workspace/models/hub/models--openai--whisper-large-v3/snapshots/06f233fe06e710322aca913c1bc4249a0d71fce1; NAME=openai/whisper-large-v3 ;;
  *) echo "unknown ASR kind $KIND" >&2; exit 2 ;;
esac
cd "$OMNI_SGLANG_REPO"
source /workspace/cgzheng/cgzheng/omni-baselines/eval-env/bin/activate
export SGLANG_OMNI_STRICT_PORT=1 SGLANG_OMNI_STARTUP_TIMEOUT=1800
exec sgl-omni serve --model-path "$M" --model-name "$NAME" --port "$PORT" "$@"
