#!/usr/bin/env bash
# AC-3 measurement slot: HF's own bf16 noise floor and the engine's metrics against the HF references.
set -uo pipefail
G=$1; Q=/workspace/cgzheng/cgzheng/qwen3_omni; cd $Q; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=$G
.venv-hfref/bin/python tools/hfref/ac3_noise_floor.py results/engine_parity/ac3_noise_floor.json > results/engine_parity/ac3_noise_floor.log 2>&1; echo "noise floor rc=$?"
.venv-engine/bin/python tools/ac3_engine_report.py results/engine_parity/ac3_engine_report.json > results/engine_parity/ac3_engine_report.log 2>&1; echo "engine report rc=$?"
