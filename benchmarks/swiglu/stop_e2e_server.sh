#!/usr/bin/env bash
set -eo pipefail

RESULT_ROOT=${1:?usage: stop_e2e_server.sh RESULT_ROOT PORT}
PORT=${2:?usage: stop_e2e_server.sh RESULT_ROOT PORT}
PID_FILE="$RESULT_ROOT/server.pid"

source /usr/local/Ascend/ascend-toolkit/set_env.sh

if [[ ! -f "$PID_FILE" ]]; then
  echo "missing PID file: $PID_FILE" >&2
  exit 1
fi
PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
  echo "server process already exited: $PID"
else
  CMDLINE=$(tr '\0' ' ' < "/proc/$PID/cmdline")
  if [[ "$CMDLINE" != *"torch.distributed.run"* || "$CMDLINE" != *"server.py"* ]]; then
    echo "refusing to stop unexpected PID $PID: $CMDLINE" >&2
    exit 1
  fi
  kill -TERM "$PID"
  for _ in $(seq 1 90); do
    if ! kill -0 "$PID" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$PID" 2>/dev/null; then
    echo "server did not stop after SIGTERM: $PID" >&2
    exit 1
  fi
fi

for _ in $(seq 1 30); do
  if ! curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    npu-smi info > "$RESULT_ROOT/npu-after.txt"
    echo "server stopped: pid=$PID port=$PORT"
    exit 0
  fi
  sleep 1
done

echo "port $PORT remains healthy after server exit" >&2
exit 1
