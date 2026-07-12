#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/benchmarks/qwen3_32b_tp2_fp16/baseline.env"
GRAPH_MODE=${1:?usage: stop_server.sh on|off CAMPAIGN_ID}
CAMPAIGN=${2:?usage: stop_server.sh on|off CAMPAIGN_ID}
RESULT_ROOT="$ROOT/benchmark-results/$CAMPAIGN/$GRAPH_MODE"
PID_FILE="$RESULT_ROOT/server/server.pid"

curl -fsS "$SERVER_URL/debug/stats" > "$RESULT_ROOT/server/stop-stats.json" 2>/dev/null || true
curl -fsS "$SERVER_URL/metrics" > "$RESULT_ROOT/server/stop-metrics.prom" 2>/dev/null || true
if [[ ! -f "$PID_FILE" ]]; then
  echo "no campaign pid file; nothing stopped"
  exit 0
fi
PID=$(cat "$PID_FILE")
if ! kill -0 "$PID" 2>/dev/null; then
  echo "campaign process already stopped: $PID"
  exit 0
fi
CMDLINE=$(tr '\0' ' ' < "/proc/$PID/cmdline")
if [[ "$CMDLINE" != *"torch.distributed.run"* || "$CMDLINE" != *"server.py"* ]]; then
  echo "refusing to stop unexpected pid $PID: $CMDLINE" >&2
  exit 1
fi
kill -TERM "$PID"
for _ in $(seq 1 30); do
  kill -0 "$PID" 2>/dev/null || { echo "server stopped: $PID"; exit 0; }
  sleep 1
done
echo "server did not stop after SIGTERM; no force signal sent" >&2
exit 1

