#!/usr/bin/env bash
# Second engine GPU slot: numerics probe, GPU parity reruns, the spike runs that failed for a missing decoder.
set -uo pipefail
G=$1; Q=/workspace/cgzheng/cgzheng/qwen3_omni; cd $Q; source scripts/env.sh
export CUDA_VISIBLE_DEVICES=$G ENGINE_GPU_TESTS=1
.venv-engine/bin/python tools/numerics_probe.py results/engine_parity/thinker_ref_bf16.pt 8 > results/engine_parity/numerics_probe.log 2>&1; echo "probe rc=$?"
.venv-engine/bin/python -m pytest tests/engine -q -p no:cacheprovider -k "gpu or stream or paged or decode" > results/engine_parity/gpu_tests2.log 2>&1; echo "engine tests2 rc=$?"
for b in 1 8; do .venv-engine/bin/python tools/spike/a_custom.py $b > results/l20x/spike/a_b$b.log 2>&1; echo "spike a b=$b rc=$?"; done
