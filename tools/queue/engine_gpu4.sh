#!/usr/bin/env bash
# AC-3 slot 4: HF noise floor with per-prompt records, then the engine GPU parity tests (Thinker vs HF noise, Talker, Token2Wav,
# paged/graph decode equivalence) and the engine report.
set -uo pipefail
G=$1; Q=/workspace/cgzheng/cgzheng/qwen3_omni; cd $Q; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=$G ENGINE_GPU_TESTS=1
[ -f results/engine_parity/hf_bf16_sdpa_per_prompt.pt ] || .venv-hfref/bin/python tools/hfref/ac3_noise_floor.py results/engine_parity/ac3_noise_floor.json > results/engine_parity/ac3_noise_floor.log 2>&1; echo "noise floor rc=$?"
.venv-engine/bin/python -m pytest tests/engine -q -p no:cacheprovider -k "gpu or stream or decode" > results/engine_parity/gpu_tests4.log 2>&1; echo "engine tests4 rc=$?"
.venv-engine/bin/python tools/ac3_engine_report.py results/engine_parity/ac3_engine_report.json > results/engine_parity/ac3_engine_report.log 2>&1; echo "engine report rc=$?"
