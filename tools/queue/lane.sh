#!/usr/bin/env bash
# Usage: lane.sh <gpu> <worktree> <wait-pid> <job>...   jobs: "sweep:<spec-name>:<out-dir>" | "mm:<name>:<port>:<args...>" | "dumps"
set -uo pipefail
G=$1; W=$2; WAIT=$3; shift 3
Q=/workspace/cgzheng/cgzheng/qwen3_omni; S=$(dirname "$0")
while kill -0 "$WAIT" 2>/dev/null; do sleep 30; done
cd "$W"
for job in "$@"; do
  echo "[$(date +%F' '%T)] start $job"
  IFS=: read -r kind a b c <<< "$job"
  case $kind in
    sweep) $Q/.venv/bin/python -m bench.cli sweep --spec "$W/sweeps/campaign/$a" --out "$Q/results/l20x/campaign/$b" ;;
    mm) $Q/.venv/bin/python -m evals.multimodal.suite run --gpu "$G" --port "$b" --subset full --scope final --concurrency 16 \
          --out "$Q/results/multimodal/full/$a" $c ;;
    dumps) bash "$S/dumps.sh" "$G" "$W" ;;
    xcheck) bash "$S/xcheck.sh" "$G" "$W" "$a" "$b" "$c" ;;
    score) $Q/.venv/bin/python -m evals.speech.suite score --gpu "$G" $c ;;
    main) (cd $Q && bash -c "$a") ;;
    trace) (cd $Q && .venv/bin/python tools/trace_baselines.py run --stack "$a" --gpu "$G" --port $((18800 + $b)) --concurrency "$b"               --out "$Q/results/l20x/traces/${a}_c$b" && .venv/bin/python tools/trace_baselines.py extract --trace-dir "$Q/results/l20x/traces/${a}_c$b") ;;
    engine) bash "$S/engine_gpu.sh" "$G" ;;
  esac
  echo "[$(date +%F' '%T)] end $job rc=$?"
done
