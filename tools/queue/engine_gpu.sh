#!/usr/bin/env bash
# Engine GPU parity tests and the architecture spike on GPU $1 (run from the main repo).
set -uo pipefail
G=$1; Q=/workspace/cgzheng/cgzheng/qwen3_omni; cd $Q; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=$G ENGINE_GPU_TESTS=1
mkdir -p results/l20x/spike results/engine_parity
.venv-engine/bin/python -m pytest tests/engine -q -p no:cacheprovider > results/engine_parity/gpu_tests.log 2>&1; echo "engine tests rc=$?"
for b in 1 8 32; do .venv-engine/bin/python tools/spike/a_custom.py $b > results/l20x/spike/a_b$b.log 2>&1; echo "spike a b=$b rc=$?"; done
for b in 8; do .venv-engine/bin/python tools/spike/c_two_process.py $b > results/l20x/spike/c_b$b.log 2>&1; echo "spike c rc=$?"; done
for b in 1 8 32; do $OMNI_VLLM_REPO/.venv/bin/python tools/spike/b_vllm_thinker.py $b > results/l20x/spike/b_b$b.log 2>&1; echo "spike b b=$b rc=$?"; done
