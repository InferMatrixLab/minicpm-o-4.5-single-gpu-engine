# Shared environment for baseline and engine launches.
# Each value keeps a caller's setting. scripts/env.local.sh (not tracked) is sourced first for machine-specific locations.
_ENV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[ -f "$_ENV_ROOT/scripts/env.local.sh" ] && source "$_ENV_ROOT/scripts/env.local.sh"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export HF_MODULES_CACHE="${HF_MODULES_CACHE:-$HF_HOME/modules}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export MODEL_PATH="${MODEL_PATH:-$HF_HUB_CACHE/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582}"
export OMNI_VLLM_REPO="${OMNI_VLLM_REPO:-$_ENV_ROOT/../vllm-omni}"
export OMNI_SGLANG_REPO="${OMNI_SGLANG_REPO:-$_ENV_ROOT/../sglang-omni}"
export OMNI_EVAL_ENV="${OMNI_EVAL_ENV:-$_ENV_ROOT/../omni-eval-env}"
unset _ENV_ROOT
