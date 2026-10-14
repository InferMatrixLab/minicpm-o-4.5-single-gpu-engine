# Shared environment for baseline and engine launches.
export HF_HUB_CACHE=/workspace/models/hub
export HF_MODULES_CACHE=/workspace/models/cgzheng/hf_modules
export HF_HUB_OFFLINE=1
export MODEL_PATH=/workspace/models/hub/models--openbmb--MiniCPM-o-4_5/snapshots/503e754207c94da6bb26850b4469f367c9ea3582
export OMNI_VLLM_REPO=/workspace/cgzheng/cgzheng/vllm-omni
export OMNI_SGLANG_REPO=/workspace/cgzheng/cgzheng/omni-baselines/sglang-omni-main
