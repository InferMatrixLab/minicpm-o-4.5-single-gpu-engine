#!/usr/bin/env bash
# Regenerate the 32-row effective-input dumps with provenance on GPU $1 (vllm-omni dump needs a live server).
set -euo pipefail
W=$2; Q=/workspace/cgzheng/cgzheng/qwen3_omni; G=$1; PORT=18690
OUT=$Q/results/effective_inputs_v2; mkdir -p $OUT
source $W/scripts/env.sh
cd $W
CUDA_VISIBLE_DEVICES=$G $OMNI_VLLM_REPO/.venv/bin/python -m bench.effective_dump canonical --out $OUT/canonical.jsonl
CUDA_VISIBLE_DEVICES=$G $OMNI_SGLANG_REPO/.venv/bin/python -m bench.effective_dump sglang-omni --out $OUT/sglang-omni.jsonl
CUDA_VISIBLE_DEVICES=$G setsid bash $W/scripts/serve_vllm_omni.sh minicpmo_4_5_turn.yaml $PORT > $OUT/vllm_server.log 2>&1 &
SPID=$!
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do sleep 5; kill -0 $SPID || exit 1; done
CUDA_VISIBLE_DEVICES=$G $OMNI_VLLM_REPO/.venv/bin/python -m bench.effective_dump vllm-omni --out $OUT/vllm-omni.jsonl --url http://127.0.0.1:$PORT
kill -TERM $SPID; sleep 20; kill -TERM -- -$SPID 2>/dev/null || true
for i in $(seq 60); do nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
cd $Q && .venv/bin/python -m bench.effective_report $OUT
echo DUMPS_DONE
