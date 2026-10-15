#!/usr/bin/env bash
# Engine validation window (round 4): GPU tests one file per process, the HF Track S reference and its parity test,
# in-process smoke runs under a private MPS daemon, and the first harness sweep of the engine.
# Usage: engine_validate_r4.sh <gpu-index>      (the GPU must be held for this window: no timing job on it)
set -uo pipefail
G=$1; Q=/workspace/cgzheng/cgzheng/qwen3_omni; cd $Q; source scripts/env.sh
OUT=results/engine_validate; mkdir -p $OUT
UUID=$(nvidia-smi -i "$G" --query-gpu=uuid --format=csv,noheader | tr -d ' ')
export ENGINE_GPU_TESTS=1
step() { echo "[$(date +%T)] $*"; }

step "GPU tests"
for f in test_kernels test_thinker_decode test_talker_decode test_token2wav_stream test_thinker_gpu_parity test_talker_gpu_parity test_token2wav_chunker test_track_s_chunks; do
  CUDA_VISIBLE_DEVICES=$UUID timeout 2400 .venv-engine/bin/python -m pytest tests/engine/$f.py -q -p no:cacheprovider > $OUT/$f.log 2>&1
  step "$f rc=$? $(tail -1 $OUT/$f.log)"
done

step "HF Track S reference"
[ -f results/engine_parity/track_s_ref_bf16.pt ] || CUDA_VISIBLE_DEVICES=$UUID timeout 3600 .venv-hfref/bin/python tools/hfref/track_s_reference.py \
  results/engine_parity/track_s_ref_bf16.pt 16 > $OUT/track_s_reference.log 2>&1
step "reference rc=$?"
CUDA_VISIBLE_DEVICES=$UUID timeout 2400 .venv-engine/bin/python -m pytest tests/engine/test_track_s_gpu_parity.py -q -p no:cacheprovider > $OUT/test_track_s_gpu_parity.log 2>&1
step "track_s parity rc=$? $(tail -1 $OUT/test_track_s_gpu_parity.log)"

step "smoke under MPS"
M=/tmp/cgz-mps-val$G; rm -rf $M; mkdir -p $M/pipe $M/log
export CUDA_MPS_PIPE_DIRECTORY=$M/pipe CUDA_MPS_LOG_DIRECTORY=$M/log
CUDA_VISIBLE_DEVICES=$UUID nvidia-cuda-mps-control -d
for cfg in track_s_proc track_s_proc_euler; do
  for c in 1 8 32; do
    n=$((c == 1 ? 8 : 4 * c))
    CUDA_VISIBLE_DEVICES=$UUID timeout 2400 .venv-engine/bin/python tools/engine_smoke.py --config configs/engine/$cfg.json --n $n --c $c \
      > $OUT/smoke_${cfg}_c$c.log 2>&1
    step "smoke $cfg C=$c rc=$? $(grep '^{"n"' $OUT/smoke_${cfg}_c$c.log | cut -c1-200)"
  done
done
echo quit | nvidia-cuda-mps-control; sleep 2; rm -rf $M
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

step "harness sweep (engine through the HTTP server, ENGINE_MPS=1)"
timeout 7200 .venv/bin/python -m bench.cli sweep --spec sweeps/engine/dev_gpu$G.json --out results/engine_sweeps/dev_gpu$G > $OUT/sweep.log 2>&1
step "sweep rc=$?"
grep -E "run[0-9]+:|valid=" $OUT/sweep.log | tail -8
step "done"
