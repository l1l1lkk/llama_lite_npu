#!/usr/bin/env bash
# Capture matched fused Triton decode traces with NPU Graph disabled/enabled.

set -eo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CHECKOUT=${CHECKOUT:-/data/liuke/llama_lite_npu-worktrees/swiglu-fused-optimization}
RESULT_ROOT=${1:-/data/liuke/swiglu_graph_validation_20260728/profiler}
PORT=${PORT:-8016}
MASTER_PORT=${MASTER_PORT:-29616}
PROFILE_DURATION=${PROFILE_DURATION:-8}
WORKLOAD="$CHECKOUT/benchmarks/qwen3_32b_tp2_fp16/datasets/20260714_length_matrix/p128_n32-formal.jsonl"
COMMIT=$(git -C "$CHECKOUT" rev-parse HEAD)

source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p "$RESULT_ROOT"

active_server_root=
cleanup() {
  if [[ -n "$active_server_root" ]]; then
    bash "$SCRIPT_DIR/stop_e2e_server.sh" "$active_server_root" "$PORT" || true
  fi
}
trap cleanup EXIT

run_request() {
  local output=$1
  local requests=$2
  python "$SCRIPT_DIR/e2e_client.py" \
    --base-url "http://127.0.0.1:$PORT" \
    --workload "$WORKLOAD" \
    --checkout "$CHECKOUT" \
    --server-branch feature/swiglu-fused-optimization \
    --server-commit "$COMMIT" \
    --case-id p128-c1-o32 \
    --repeat 1 \
    --concurrency 1 \
    --requests "$requests" \
    --warmup-requests 0 \
    --max-tokens 32 \
    --output "$output" \
    > "${output%.json}.log"
}

for graph_mode in off on; do
  mode_root="$RESULT_ROOT/graph-$graph_mode"
  server_root="$mode_root/server"
  mkdir -p "$server_root"
  active_server_root=$server_root

  BATCHING_MODE=legacy \
  NPU_DEVICES=6,7 \
  NPU_GRAPH_MODE="$graph_mode" \
  PROFILING_MODE=dynamic \
    bash "$SCRIPT_DIR/start_e2e_server.sh" \
      "$CHECKOUT" "$server_root" "$PORT" "$MASTER_PORT"

  run_request "$mode_root/prewarm.json" 1
  curl -fsS "http://127.0.0.1:$PORT/debug/stats" \
    > "$mode_root/pre-profile-stats.json"

  launcher_pid=$(cat "$server_root/server.pid")
  rank_pid=$(
    ps -eo pid=,ppid=,args= |
      awk -v parent="$launcher_pid" \
        '$2 == parent && $0 ~ /python -u server.py/ {print $1}' |
      sort -n |
      head -n 1
  )
  if [[ -z "$rank_pid" ]]; then
    echo "failed to find rank process under launcher $launcher_pid" >&2
    exit 1
  fi
  printf '%s\n' "$rank_pid" > "$mode_root/profiled-rank-pid.txt"

  {
    printf 'start\n'
    sleep "$PROFILE_DURATION"
    printf 'stop\nquit\n'
  } | msprof \
      --dynamic=on \
      --pid="$rank_pid" \
      --output="$mode_root/msprof" \
      --ascendcl=on \
      --runtime-api=on \
      --task-time=on \
      --hccl=on \
      --aic-metrics=PipeUtilization \
      > "$mode_root/msprof.log" 2>&1 &
  profiler_pid=$!
  sleep 2
  run_request "$mode_root/profiled-request.json" 1
  wait "$profiler_pid"

  curl -fsS "http://127.0.0.1:$PORT/debug/stats" \
    > "$mode_root/post-profile-stats.json"
  bash "$SCRIPT_DIR/stop_e2e_server.sh" "$server_root" "$PORT"
  active_server_root=
done

date --iso-8601=seconds > "$RESULT_ROOT/run-complete.txt"
