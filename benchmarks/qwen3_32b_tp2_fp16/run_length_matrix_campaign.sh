#!/usr/bin/env bash
set -u

repo=${REPO_ROOT:-/data/liuke/llama_lite_npu}
campaign=${CAMPAIGN:-20260714_qwen3_32b_tp2_fp16_length_matrix}
campaign_root="$repo/benchmark-results/$campaign"
plan=${PLAN_PATH:-$repo/benchmarks/qwen3_32b_tp2_fp16/campaigns/20260714_length_matrix.json}
workloads=${WORKLOAD_ROOT:-$repo/benchmarks/qwen3_32b_tp2_fp16/datasets/20260714_length_matrix}
namespace=matrix
log="$campaign_root/orchestration.log"
start_order=${START_ORDER:-1}
stall_no_progress_seconds=1200
hard_review_seconds=21600

cd "$repo"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
mkdir -p "$campaign_root/workload"
cp "$plan" "$campaign_root/campaign-plan.json"
cp "$workloads"/* "$campaign_root/workload/"
exec >>"$log" 2>&1
echo "ORCHESTRATION_START $(date -Iseconds)"
echo "WATCHDOG_PROTOCOL revision=progress-v2 start_order=$start_order no_progress_seconds=$stall_no_progress_seconds hard_review_seconds=$hard_review_seconds signals=completed_requests,generated_tokens,graph_replays stall_requires=no_progress_and_idle_or_waiting_without_running"

python - "$plan" <<'PY' > /tmp/task5-sequence.tsv
import json, sys
for item in json.load(open(sys.argv[1], encoding="utf-8"))["sequence"]:
    print(item["order"], item["repeat"], item["prompt"], item["output"], item["lifecycle_id"], sep="\t")
PY

while IFS=$'\t' read -r order repeat prompt output lifecycle; do
  if (( order < start_order )); then
    echo "UNIT_SKIP order=$order lifecycle=$lifecycle reason=start_order"
    continue
  fi
  echo "UNIT_START order=$order repeat=$repeat prompt=$prompt output=$output lifecycle=$lifecycle time=$(date -Iseconds)"
  if curl -fsS --max-time 2 http://127.0.0.1:8213/health >/dev/null 2>&1; then
    echo "PRECONDITION_FAILED port_open lifecycle=$lifecycle"; exit 20
  fi
  if ps -eo args | grep -q '[s]erver.py'; then
    echo "PRECONDITION_FAILED server_process lifecycle=$lifecycle"; exit 21
  fi
  aicore6=$(npu-smi info -t usages -i 6 | awk -F: '/Aicore Usage Rate/{gsub(/[^0-9.]/,"",$2); print $2; exit}')
  aicore7=$(npu-smi info -t usages -i 7 | awk -F: '/Aicore Usage Rate/{gsub(/[^0-9.]/,"",$2); print $2; exit}')
  echo "PRE_NPU lifecycle=$lifecycle aicore6=${aicore6:-unknown} aicore7=${aicore7:-unknown}"
  if [[ "${aicore6:-1}" != 0 || "${aicore7:-1}" != 0 ]]; then
    echo "PRECONDITION_FAILED npu_busy lifecycle=$lifecycle"; exit 22
  fi

  if ! env SERVER_LIFECYCLE_ID="$lifecycle" RESULT_NAMESPACE="$namespace" DECODE_PRIORITY_MODE=off \
    bash benchmarks/qwen3_32b_tp2_fp16/start_server.sh on "$campaign"; then
    if ! curl -fsS --max-time 3 http://127.0.0.1:8213/health >/dev/null 2>&1; then
      echo "START_FAILED lifecycle=$lifecycle"; exit 23
    fi
    echo "START_RETURNED_NONZERO_BUT_HEALTHY lifecycle=$lifecycle"
  fi

  formal="$workloads/p${prompt}_n32-formal.jsonl"
  warmup="$workloads/p${prompt}_n8-warmup.jsonl"
  eval_prompt=$((prompt - 22))
  monitor="$campaign_root/$namespace/lifecycles/$lifecycle/server/lifecycle-monitor.jsonl"
  setsid env \
    SEPARATE_WARMUP=1 WARMUP_REQUESTS_PER_WORKER=2 RUN_REPETITIONS="$repeat" \
    RESULT_NAMESPACE="$namespace" BENCHMARK_VARIANT=length_matrix DATASET_KIND=line_by_line \
    FORMAL_DATASET_PATH="$formal" WARMUP_DATASET_PATH="$warmup" \
    DATASET_OFFSET_OVERRIDE=0 WARMUP_DATASET_OFFSET_OVERRIDE=0 \
    WORKLOAD_ID=qwen3_32b_length_matrix_v1 SERVER_LIFECYCLE_ID="$lifecycle" \
    LIFECYCLE_ORDER_INDEX="$order" LIFECYCLE_SEQUENCE=length-matrix-balanced-v1 \
    TIMESERIES_INTERVAL_S=1 NPU_SAMPLE_INTERVAL_S=5 \
    bash benchmarks/qwen3_32b_tp2_fp16/run_case.sh "$campaign" on "$prompt" "$eval_prompt" "$output" 4 32 &
  run_pid=$!
  run_started_epoch=$(date +%s)
  last_progress_epoch=$run_started_epoch
  last_success=-1
  last_generated=-1
  last_replays=-1
  hard_review_recorded=0
  while kill -0 "$run_pid" 2>/dev/null; do
    stats=$(curl -fsS --max-time 3 http://127.0.0.1:8213/debug/stats 2>/dev/null || echo '{}')
    metrics=$(curl -fsS --max-time 3 http://127.0.0.1:8213/metrics 2>/dev/null || true)
    waiting=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("scheduler",{}).get("waiting",0))' "$stats" 2>/dev/null || echo 0)
    running=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("scheduler",{}).get("running",0))' "$stats" 2>/dev/null || echo 0)
    success=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("requests",{}).get("chat:success",0))' "$stats" 2>/dev/null || echo 0)
    replays=$(python -c 'import json,sys; print(json.loads(sys.argv[1]).get("npu_graph",{}).get("replays",0))' "$stats" 2>/dev/null || echo 0)
    generated=$(awk '/^lite_llama_generated_tokens_total\{endpoint="chat"\}/{print int($2); exit}' <<<"$metrics")
    generated=${generated:-0}
    aicore=$(npu-smi info -t usages -i 6 | awk -F: '/Aicore Usage Rate/{gsub(/[^0-9.]/,"",$2); print $2; exit}')
    python - "$monitor" "$stats" "${aicore:-null}" <<'PY'
import datetime, json, sys
try: stats=json.loads(sys.argv[2])
except Exception: stats={"status":"parse_error"}
with open(sys.argv[1], "a", encoding="utf-8") as f:
    f.write(json.dumps({"timestamp_utc":datetime.datetime.now(datetime.timezone.utc).isoformat(),"stats":stats,"aicore6_pct":None if sys.argv[3]=="null" else float(sys.argv[3])})+"\n")
PY
    now_epoch=$(date +%s)
    if (( success > last_success || generated > last_generated || replays > last_replays )); then
      last_progress_epoch=$now_epoch
    fi
    last_success=$success
    last_generated=$generated
    last_replays=$replays
    no_progress_seconds=$((now_epoch - last_progress_epoch))
    run_elapsed_seconds=$((now_epoch - run_started_epoch))
    stalled_evidence=0
    if (( waiting > 0 && running == 0 )); then stalled_evidence=1; fi
    if [[ "${aicore:-1}" == 0 ]]; then stalled_evidence=1; fi
    if (( no_progress_seconds >= stall_no_progress_seconds && stalled_evidence == 1 )); then
      diagnostic="$campaign_root/server-only-rejected/stall-$lifecycle"
      mkdir -p "$diagnostic"
      cp "$monitor" "$diagnostic/lifecycle-monitor.jsonl"
      printf '%s\n' "$stats" > "$diagnostic/final-stats.json"
      ps -eo pid,ppid,stat,etime,args > "$diagnostic/process-tree.txt"
      npu-smi info -t usages -i 6 > "$diagnostic/npu6.txt" 2>&1 || true
      npu-smi info -t usages -i 7 > "$diagnostic/npu7.txt" 2>&1 || true
      tail -n 300 "$campaign_root/$namespace/lifecycles/$lifecycle/server/server.log" > "$diagnostic/server-log-tail.txt" 2>/dev/null || true
      printf 'reason=progress-watchdog-stall\nno_progress_seconds=%s\ncompleted_requests=%s\ngenerated_tokens=%s\ngraph_replays=%s\nwaiting=%s\nrunning=%s\naicore6_pct=%s\n' "$no_progress_seconds" "$success" "$generated" "$replays" "$waiting" "$running" "${aicore:-unknown}" > "$diagnostic/watchdog-state.txt"
      echo "STALL_DETECTED lifecycle=$lifecycle no_progress_seconds=$no_progress_seconds success=$success generated=$generated replays=$replays waiting=$waiting running=$running aicore=$aicore time=$(date -Iseconds)"
      kill -TERM -- "-$run_pid" 2>/dev/null || true; wait "$run_pid" 2>/dev/null || true
      env SERVER_LIFECYCLE_ID="$lifecycle" RESULT_NAMESPACE="$namespace" bash benchmarks/qwen3_32b_tp2_fp16/stop_server.sh on "$campaign" || true
      exit 50
    fi
    if (( run_elapsed_seconds >= hard_review_seconds && hard_review_recorded == 0 )); then
      diagnostic="$campaign_root/server-only-rejected/hard-review-$lifecycle"
      mkdir -p "$diagnostic"
      cp "$monitor" "$diagnostic/lifecycle-monitor.jsonl" 2>/dev/null || true
      printf '%s\n' "$stats" > "$diagnostic/current-stats.json"
      printf 'reason=hard-review-threshold-reached-while-progressing\nrun_elapsed_seconds=%s\nno_progress_seconds=%s\ncompleted_requests=%s\ngenerated_tokens=%s\ngraph_replays=%s\nwaiting=%s\nrunning=%s\naicore6_pct=%s\nautomatic_termination=false\n' "$run_elapsed_seconds" "$no_progress_seconds" "$success" "$generated" "$replays" "$waiting" "$running" "${aicore:-unknown}" > "$diagnostic/watchdog-state.txt"
      echo "HARD_REVIEW_REQUIRED lifecycle=$lifecycle elapsed_seconds=$run_elapsed_seconds no_progress_seconds=$no_progress_seconds success=$success generated=$generated replays=$replays time=$(date -Iseconds)"
      hard_review_recorded=1
    fi
    sleep 30
  done
  wait "$run_pid"; rc=$?
  if (( rc != 0 )); then
    diagnostic="$campaign_root/server-only-rejected/failed-$lifecycle"
    mkdir -p "$diagnostic"
    cp "$monitor" "$diagnostic/lifecycle-monitor.jsonl" 2>/dev/null || true
    tail -n 300 "$campaign_root/$namespace/lifecycles/$lifecycle/server/server.log" > "$diagnostic/server-log-tail.txt" 2>/dev/null || true
    echo "RUN_FAILED lifecycle=$lifecycle rc=$rc time=$(date -Iseconds)"
    env SERVER_LIFECYCLE_ID="$lifecycle" RESULT_NAMESPACE="$namespace" bash benchmarks/qwen3_32b_tp2_fp16/stop_server.sh on "$campaign" || true
    exit 30
  fi
  echo "RUN_OK lifecycle=$lifecycle time=$(date -Iseconds)"
  env SERVER_LIFECYCLE_ID="$lifecycle" RESULT_NAMESPACE="$namespace" bash benchmarks/qwen3_32b_tp2_fp16/stop_server.sh on "$campaign" || true
  for _ in $(seq 1 60); do
    if ! curl -fsS --max-time 1 http://127.0.0.1:8213/health >/dev/null 2>&1 && ! ps -eo args | grep -q '[s]erver.py'; then break; fi
    sleep 1
  done
  if curl -fsS --max-time 2 http://127.0.0.1:8213/health >/dev/null 2>&1 || ps -eo args | grep -q '[s]erver.py'; then
    echo "STOP_FAILED lifecycle=$lifecycle"; exit 40
  fi
  echo "UNIT_DONE order=$order lifecycle=$lifecycle time=$(date -Iseconds)"
done < /tmp/task5-sequence.tsv

echo "ORCHESTRATION_DONE $(date -Iseconds)"
