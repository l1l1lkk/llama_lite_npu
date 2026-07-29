#!/usr/bin/env bash
set -eo pipefail

CHECKOUT=${1:?usage: start_e2e_server.sh CHECKOUT RESULT_ROOT PORT MASTER_PORT}
RESULT_ROOT=${2:?usage: start_e2e_server.sh CHECKOUT RESULT_ROOT PORT MASTER_PORT}
PORT=${3:?usage: start_e2e_server.sh CHECKOUT RESULT_ROOT PORT MASTER_PORT}
MASTER_PORT=${4:?usage: start_e2e_server.sh CHECKOUT RESULT_ROOT PORT MASTER_PORT}

MODEL_DIR=${MODEL_DIR:-/data/liuke/llama_lite_npu/my_weight/Qwen3-1.7B}
NPU_DEVICES=${NPU_DEVICES:-6,7}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
MAX_BATCH_SIZE=${MAX_BATCH_SIZE:-8}

source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p "$RESULT_ROOT"
PID_FILE="$RESULT_ROOT/server.pid"
SERVER_URL="http://127.0.0.1:$PORT"

if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "server already running from $PID_FILE" >&2
  exit 1
fi
if curl -fsS "$SERVER_URL/health" >/dev/null 2>&1; then
  echo "port $PORT already has a healthy service" >&2
  exit 1
fi

COMMAND=(
  python -m torch.distributed.run
  --nproc_per_node=2
  --master_addr=127.0.0.1
  --master_port="$MASTER_PORT"
  server.py
  --checkpoints_dir "$MODEL_DIR"
  --host 127.0.0.1
  --port "$PORT"
  --max_seq_len "$MAX_SEQ_LEN"
  --page_size 16
  --no_compiled_model
  --continuous_batching
  --max_batch_size "$MAX_BATCH_SIZE"
  --no_decode_priority
)

{
  printf 'ASCEND_RT_VISIBLE_DEVICES=%q ' "$NPU_DEVICES"
  printf 'LITE_LLAMA_REQUEST_TIMING_TRACE=%q ' "$RESULT_ROOT/request-timing"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
} > "$RESULT_ROOT/start-command.txt"
git -C "$CHECKOUT" status --short --branch > "$RESULT_ROOT/git-status.txt"
git -C "$CHECKOUT" rev-parse HEAD > "$RESULT_ROOT/git-head.txt"
npu-smi info > "$RESULT_ROOT/npu-before.txt"

cd "$CHECKOUT"
nohup env \
  ASCEND_RT_VISIBLE_DEVICES="$NPU_DEVICES" \
  LITE_LLAMA_REQUEST_TIMING_TRACE="$RESULT_ROOT/request-timing" \
  "${COMMAND[@]}" > "$RESULT_ROOT/server.log" 2>&1 &
SERVER_PID=$!
echo "$SERVER_PID" > "$PID_FILE"

for _ in $(seq 1 180); do
  if curl -fsS "$SERVER_URL/health" > "$RESULT_ROOT/health.json" 2>/dev/null; then
    curl -fsS "$SERVER_URL/debug/stats" > "$RESULT_ROOT/start-stats.json"
    curl -fsS "$SERVER_URL/metrics" > "$RESULT_ROOT/start-metrics.prom"
    echo "server ready: pid=$SERVER_PID port=$PORT"
    exit 0
  fi
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "server exited before readiness; inspect $RESULT_ROOT/server.log" >&2
    exit 1
  fi
  sleep 2
done

echo "server readiness timed out" >&2
exit 1
