#!/usr/bin/env bash
set -eo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/benchmarks/qwen3_32b_tp2_fp16/baseline.env"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

GRAPH_MODE=${1:-on}
CAMPAIGN=${2:?usage: start_server.sh on|off CAMPAIGN_ID}
case "$GRAPH_MODE" in
  on) GRAPH_FLAG=--compiled_model ;;
  off) GRAPH_FLAG=--no_compiled_model ;;
  *) echo "graph mode must be on or off" >&2; exit 2 ;;
esac

SERVER_LIFECYCLE_ID=${SERVER_LIFECYCLE_ID:-}
RESULT_NAMESPACE=${RESULT_NAMESPACE:-$GRAPH_MODE}
DECODE_PRIORITY_MODE=${DECODE_PRIORITY_MODE:-on}
case "$DECODE_PRIORITY_MODE" in
  on) DECODE_PRIORITY_FLAG=--decode_priority ;;
  off) DECODE_PRIORITY_FLAG=--no_decode_priority ;;
  *) echo "DECODE_PRIORITY_MODE must be on or off" >&2; exit 2 ;;
esac
if [[ -n "$SERVER_LIFECYCLE_ID" ]]; then
  RESULT_ROOT="$ROOT/benchmark-results/$CAMPAIGN/$RESULT_NAMESPACE/lifecycles/$SERVER_LIFECYCLE_ID"
else
  RESULT_ROOT="$ROOT/benchmark-results/$CAMPAIGN/$RESULT_NAMESPACE"
fi
mkdir -p "$RESULT_ROOT/server"
PID_FILE="$RESULT_ROOT/server/server.pid"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "refusing to start: campaign server is already running" >&2
  exit 1
fi
if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
  echo "refusing to start: $SERVER_URL already has a healthy service" >&2
  exit 1
fi

COMMAND=(
  python -m torch.distributed.run
  --nproc_per_node="$TP_SIZE"
  --master_addr=127.0.0.1
  --master_port="$MASTER_PORT"
  server.py
  --checkpoints_dir "$MODEL_DIR"
  --host 0.0.0.0
  --port "$SERVER_PORT"
  --max_seq_len "$MAX_SEQ_LEN"
  --page_size "$PAGE_SIZE"
  "$GRAPH_FLAG"
  --continuous_batching
  --max_batch_size "$MAX_BATCH_SIZE"
  "$DECODE_PRIORITY_FLAG"
)
TRACE_PREFIX="$RESULT_ROOT/server/request-timing-trace"
printf '%q ' ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES" LITE_LLAMA_REQUEST_TIMING_TRACE="$TRACE_PREFIX" "${COMMAND[@]}" > "$RESULT_ROOT/server/start-command.txt"
printf '\n' >> "$RESULT_ROOT/server/start-command.txt"

cd "$ROOT"
nohup env ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES" \
  LITE_LLAMA_REQUEST_TIMING_TRACE="$TRACE_PREFIX" "${COMMAND[@]}" \
  > "$RESULT_ROOT/server/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

for _ in $(seq 1 180); do
  if curl -fsS "$SERVER_URL/health" > "$RESULT_ROOT/server/health.json" 2>/dev/null; then
    curl -fsS "$SERVER_URL/debug/stats" > "$RESULT_ROOT/server/start-stats.json"
    curl -fsS "$SERVER_URL/metrics" > "$RESULT_ROOT/server/start-metrics.prom"
    echo "server ready: pid=$SERVER_PID graph=$GRAPH_MODE decode_priority=$DECODE_PRIORITY_MODE"
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited before readiness; inspect $RESULT_ROOT/server/server.log" >&2
    exit 1
  fi
  sleep 2
done
echo "server readiness timed out" >&2
exit 1
