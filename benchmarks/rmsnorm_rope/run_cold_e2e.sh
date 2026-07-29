#!/usr/bin/env bash
# Measure one cold request per server lifecycle.
#
# The current scheduler can leave requests queued after the first completed
# request.  Restarting isolates that known server limitation and prevents a
# scheduler wake-up failure from being misreported as operator latency.

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
START_SERVER="$SCRIPT_DIR/start_e2e_server.sh"
STOP_SERVER="$SCRIPT_DIR/stop_e2e_server.sh"
CLIENT="$SCRIPT_DIR/e2e_client.py"

BASELINE_CHECKOUT=${BASELINE_CHECKOUT:-/data/liuke/llama_lite_npu-worktrees/rmsnorm-rope-unfused-baseline}
FUSED_CHECKOUT=${FUSED_CHECKOUT:-/data/liuke/llama_lite_npu-worktrees/rmsnorm-rope-fused-optimization}
WORKLOAD_ROOT=${WORKLOAD_ROOT:-/data/liuke/rmsnorm_rope_validation_20260729/workloads}
RESULT_ROOT=${1:-/data/liuke/rmsnorm_rope_validation_20260729/e2e-cold}
PORT=${PORT:-8017}
MASTER_PORT=${MASTER_PORT:-29617}
REPEATS=${REPEATS:-3}
MAX_TOKENS=${MAX_TOKENS:-32}

declare -A CHECKOUTS=(
  [baseline]="$BASELINE_CHECKOUT"
  [fused]="$FUSED_CHECKOUT"
)
declare -A BRANCHES=(
  [baseline]=test/rmsnorm-rope-unfused-baseline
  [fused]=feature/rmsnorm-rope-fused-optimization
)
declare -A WORKLOADS=(
  [p128]="$WORKLOAD_ROOT/p128-fixed.jsonl"
  [p512]="$WORKLOAD_ROOT/p512-fixed.jsonl"
)

active_server_root=
cleanup() {
  if [[ -n "$active_server_root" ]]; then
    bash "$STOP_SERVER" "$active_server_root" "$PORT" || true
  fi
}
trap cleanup EXIT

source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p "$RESULT_ROOT"
npu-smi info > "$RESULT_ROOT/npu-before.txt"
env | sort > "$RESULT_ROOT/environment.txt"

for implementation in baseline fused; do
  checkout=${CHECKOUTS[$implementation]}
  branch=${BRANCHES[$implementation]}
  commit=$(git -C "$checkout" rev-parse HEAD)
  for case_name in p128 p512; do
    workload=${WORKLOADS[$case_name]}
    for repeat in $(seq 1 "$REPEATS"); do
      run_root="$RESULT_ROOT/$implementation/$case_name/repeat-$repeat"
      server_root="$run_root/server"
      active_server_root=$server_root

      NPU_DEVICES=6,7 \
      NPU_GRAPH_MODE=off \
      BATCHING_MODE=continuous \
      MODEL_DIR=/data/liuke/llama_lite_npu/my_weight/Qwen3-1.7B \
      MAX_SEQ_LEN=1024 \
      MAX_BATCH_SIZE=8 \
        bash "$START_SERVER" \
          "$checkout" "$server_root" "$PORT" "$MASTER_PORT"

      python "$CLIENT" \
        --base-url "http://127.0.0.1:$PORT" \
        --workload "$workload" \
        --checkout "$FUSED_CHECKOUT" \
        --server-branch "$branch" \
        --server-commit "$commit" \
        --case-id "$case_name-c1-o$MAX_TOKENS" \
        --repeat "$repeat" \
        --concurrency 1 \
        --requests 1 \
        --warmup-requests 0 \
        --max-tokens "$MAX_TOKENS" \
        --output "$run_root/result.json" \
        > "$run_root/client.log"

      bash "$STOP_SERVER" "$server_root" "$PORT"
      active_server_root=
    done
  done
done

npu-smi info > "$RESULT_ROOT/npu-after.txt"
date --iso-8601=seconds > "$RESULT_ROOT/run-complete.txt"
