#!/usr/bin/env bash
# Run separated/fused x NPU Graph off/on serving ablation on NPU 6,7.

set -eo pipefail

BASELINE_CHECKOUT=${BASELINE_CHECKOUT:-/data/liuke/llama_lite_npu-worktrees/matmul-swiglu-rmsnorm-unfused-baseline}
FUSED_CHECKOUT=${FUSED_CHECKOUT:-/data/liuke/llama_lite_npu-worktrees/matmul-swiglu-rmsnorm-fused-optimization}
RESULT_ROOT=${1:-/data/liuke/rmsnorm_matmul_swiglu_20260730/e2e}
PORT=${PORT:-8017}
MASTER_PORT=${MASTER_PORT:-29617}
REPEATS=${REPEATS:-3}
REQUESTS=${REQUESTS:-4}
MAX_TOKENS=${MAX_TOKENS:-32}

HARNESS="$FUSED_CHECKOUT/benchmarks/swiglu"
CLIENT="$HARNESS/e2e_client.py"
START_SERVER="$HARNESS/start_e2e_server.sh"
STOP_SERVER="$HARNESS/stop_e2e_server.sh"
DATASET_ROOT="$FUSED_CHECKOUT/benchmarks/qwen3_32b_tp2_fp16/datasets/20260714_length_matrix"
P128_WORKLOAD="$DATASET_ROOT/p128_n32-formal.jsonl"
P512_WORKLOAD="$DATASET_ROOT/p512_n32-formal.jsonl"

declare -A CHECKOUTS=(
  [baseline]="$BASELINE_CHECKOUT"
  [fused-triton]="$FUSED_CHECKOUT"
)
declare -A BRANCHES=(
  [baseline]=test/matmul-swiglu-rmsnorm-unfused-baseline
  [fused-triton]=feature/matmul-swiglu-rmsnorm-fused-optimization
)

mkdir -p "$RESULT_ROOT"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
npu-smi info > "$RESULT_ROOT/npu-before.txt"
env | sort > "$RESULT_ROOT/environment.txt"

active_server_root=
cleanup() {
  if [[ -n "$active_server_root" ]]; then
    bash "$STOP_SERVER" "$active_server_root" "$PORT" || true
  fi
}
trap cleanup EXIT

run_client() {
  local checkout=$1
  local branch=$2
  local workload=$3
  local case_id=$4
  local repeat=$5
  local requests=$6
  local warmups=$7
  local output=$8
  local commit
  commit=$(git -C "$checkout" rev-parse HEAD)

  mkdir -p "$(dirname "$output")"
  python "$CLIENT" \
    --base-url "http://127.0.0.1:$PORT" \
    --workload "$workload" \
    --checkout "$FUSED_CHECKOUT" \
    --server-branch "$branch" \
    --server-commit "$commit" \
    --case-id "$case_id" \
    --repeat "$repeat" \
    --concurrency 1 \
    --requests "$requests" \
    --warmup-requests "$warmups" \
    --max-tokens "$MAX_TOKENS" \
    --output "$output" \
    > "${output%.json}.log"
}

for implementation in baseline fused-triton; do
  checkout=${CHECKOUTS[$implementation]}
  branch=${BRANCHES[$implementation]}

  for graph_mode in off on; do
    server_root="$RESULT_ROOT/$implementation/graph-$graph_mode/server"
    mkdir -p "$server_root"
    active_server_root=$server_root

    BATCHING_MODE=legacy \
    NPU_DEVICES=6,7 \
    NPU_GRAPH_MODE="$graph_mode" \
      bash "$START_SERVER" "$checkout" "$server_root" "$PORT" "$MASTER_PORT"

    run_client \
      "$checkout" "$branch" "$P128_WORKLOAD" \
      "prewarm-p128-c1-o$MAX_TOKENS" 0 1 0 \
      "$RESULT_ROOT/$implementation/graph-$graph_mode/prewarm/p128.json"
    run_client \
      "$checkout" "$branch" "$P512_WORKLOAD" \
      "prewarm-p512-c1-o$MAX_TOKENS" 0 1 0 \
      "$RESULT_ROOT/$implementation/graph-$graph_mode/prewarm/p512.json"

    curl -fsS "http://127.0.0.1:$PORT/debug/stats" \
      > "$server_root/formal-start-stats.json"
    curl -fsS "http://127.0.0.1:$PORT/metrics" \
      > "$server_root/formal-start-metrics.prom"

    for repeat in $(seq 1 "$REPEATS"); do
      repeat_id=$(printf '%02d' "$repeat")
      run_client \
        "$checkout" "$branch" "$P128_WORKLOAD" \
        "p128-c1-o$MAX_TOKENS" "$repeat" "$REQUESTS" 1 \
        "$RESULT_ROOT/$implementation/graph-$graph_mode/p128-c1-o$MAX_TOKENS/repeat-$repeat_id/result.json"
      run_client \
        "$checkout" "$branch" "$P512_WORKLOAD" \
        "p512-c1-o$MAX_TOKENS" "$repeat" "$REQUESTS" 1 \
        "$RESULT_ROOT/$implementation/graph-$graph_mode/p512-c1-o$MAX_TOKENS/repeat-$repeat_id/result.json"
    done

    curl -fsS "http://127.0.0.1:$PORT/debug/stats" \
      > "$server_root/formal-end-stats.json"
    curl -fsS "http://127.0.0.1:$PORT/metrics" \
      > "$server_root/formal-end-metrics.prom"
    npu-smi info > "$server_root/npu-after.txt"

    bash "$STOP_SERVER" "$server_root" "$PORT"
    active_server_root=
  done
done

npu-smi info > "$RESULT_ROOT/npu-after.txt"
date --iso-8601=seconds > "$RESULT_ROOT/run-complete.txt"
