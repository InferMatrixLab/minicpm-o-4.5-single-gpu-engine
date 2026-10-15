#!/usr/bin/env bash
# Native cross-check on GPU $1 from worktree $2: repo $3 (vllm|sglang), concurrency $4, prompts $5.
# Our client: a fresh-server K=1 performance cell on the evaluation IDs; native client: the repo's own benchmark on a fresh server.
set -uo pipefail
G=$1; W=$2; R=$3; C=$4; N=$5; PORT=$((18700 + C))
Q=/workspace/cgzheng/cgzheng/qwen3_omni; OUT=$Q/results/l20x/campaign/xcheck; mkdir -p $OUT
source $W/scripts/env.sh
DS=/workspace/models/hub/datasets--zhaochenyang20--seed-tts-eval/snapshots/8f5e1aa2a35d42f42e940074c1983358b9491f89
cd $W
if [ "$R" = vllm ]; then
  $Q/.venv/bin/python -m bench.cli cell --adapter vllm-omni --config minicpmo_4_5_turn.yaml --label mps-off --gpu $G --port $PORT \
    --concurrency $C --k 1 --id-set eval --out $OUT/ours_vllm_c$C
  CUDA_VISIBLE_DEVICES=$G setsid bash $W/scripts/serve_vllm_omni.sh minicpmo_4_5_turn.yaml $PORT > $OUT/native_vllm_c${C}_server.log 2>&1 &
else
  $Q/.venv/bin/python -m bench.cli cell --adapter sglang-omni --config speech --label mps-off --gpu $G --port $PORT \
    --concurrency $C --k 1 --id-set eval --out $OUT/ours_sglang_c$C
  CUDA_VISIBLE_DEVICES=$G setsid bash $W/scripts/serve_sglang_omni.sh $PORT > $OUT/native_sglang_c${C}_server.log 2>&1 &
fi
SPID=$!
until curl -sf http://127.0.0.1:$PORT/health >/dev/null; do sleep 5; kill -0 $SPID || { echo "server died"; exit 1; }; done
if [ "$R" = vllm ]; then
  (cd $OMNI_VLLM_REPO && .venv/bin/vllm bench serve --omni --host 127.0.0.1 --port $PORT --model openbmb/MiniCPM-o-4_5 --trust-remote-code \
    --backend openai-chat-omni --endpoint /v1/chat/completions --dataset-name seed-tts --dataset-path $DS --seed-tts-locale en \
    --disable-shuffle --no-oversample --num-prompts $N --max-concurrency $C --request-rate inf --num-warmups $C \
    --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}' \
    --percentile-metrics ttft,e2el,audio_ttfp,audio_rtf,audio_duration --metric-percentiles 50,95,99 \
    --save-result --save-detailed --result-dir $OUT --result-filename native_vllm_c$C.json) > $OUT/native_vllm_c$C.log 2>&1
else
  (cd $OMNI_SGLANG_REPO && .venv/bin/python -m benchmarks.eval.benchmark_omni_seedtts --generate-only --port $PORT --model minicpmo --lang en \
    --meta $DS/en/meta.lst --voice-clone --reference-audio-field audio.ref_audio --max-concurrency $C --max-samples $N --temperature 0.0 \
    --output-dir $OUT/native_sglang_c$C) > $OUT/native_sglang_c$C.log 2>&1
fi
echo "native rc=$?"
kill -TERM $SPID; for i in $(seq 24); do kill -0 $SPID 2>/dev/null || break; sleep 5; done; kill -TERM -- -$SPID 2>/dev/null
for i in $(seq 60); do nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -q . || break; sleep 5; done
kill -KILL -- -$SPID 2>/dev/null; true
