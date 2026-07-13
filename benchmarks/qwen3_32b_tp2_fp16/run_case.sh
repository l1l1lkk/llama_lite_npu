#!/usr/bin/env bash
set -eo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$ROOT/benchmarks/qwen3_32b_tp2_fp16/baseline.env"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
set -u

CAMPAIGN=${1:?usage: run_case.sh CAMPAIGN GRAPH TARGET_PROMPT EVALSCOPE_PROMPT OUTPUT CONCURRENCY [REQUESTS]}
GRAPH_MODE=${2:?}
TARGET_PROMPT=${3:?}
EVALSCOPE_PROMPT=${4:?}
OUTPUT_TOKENS=${5:?}
CONCURRENCY=${6:?}
REQUESTS=${7:-$(( CONCURRENCY * 6 ))}
(( REQUESTS >= CONCURRENCY )) || { echo "requests must be >= concurrency" >&2; exit 2; }

CASE_ID="p${TARGET_PROMPT}_o${OUTPUT_TOKENS}_c${CONCURRENCY}"
CASE_ROOT="$ROOT/benchmark-results/$CAMPAIGN/$GRAPH_MODE/$CASE_ID"
mkdir -p "$CASE_ROOT"
curl -fsS "$SERVER_URL/health" >/dev/null

RUN_REPETITIONS=${RUN_REPETITIONS:-$(seq 1 "$FORMAL_REPEATS")}
for REP in $RUN_REPETITIONS; do
  RUN_ID="${CAMPAIGN}_${GRAPH_MODE}_${CASE_ID}_r${REP}"
  RUN_ROOT="$CASE_ROOT/run-$(printf '%02d' "$REP")"
  CLIENT_ROOT="$RUN_ROOT/client"
  SERVER_ROOT="$RUN_ROOT/server"
  mkdir -p "$CLIENT_ROOT" "$SERVER_ROOT"
  DEFAULT_OFFSET=$(( TARGET_PROMPT * 100000 + CONCURRENCY * 1000 + REP * REQUESTS ))
  OFFSET=${DATASET_OFFSET_OVERRIDE:-$DEFAULT_OFFSET}
  WARMUP=$(( CONCURRENCY * WARMUP_REQUESTS_PER_WORKER ))
  DEFAULT_WARMUP_OFFSET=$(( OFFSET + 700000000 ))
  WARMUP_OFFSET=${WARMUP_DATASET_OFFSET_OVERRIDE:-$DEFAULT_WARMUP_OFFSET}
  EVALSCOPE_VERSION=$(python -c 'from importlib.metadata import version; print(version("evalscope"))')
  CACHE_STATE=${CACHE_STATE_OVERRIDE:-kv-cache-cold-unique-offset}
  SEPARATE_WARMUP=${SEPARATE_WARMUP:-0}
  PAIR_ID="${CASE_ID}_r${REP}"
  DATASET_KIND=${DATASET_KIND:-random}
  FORMAL_DATASET_PATH=${FORMAL_DATASET_PATH:-}
  WARMUP_DATASET_PATH=${WARMUP_DATASET_PATH:-$FORMAL_DATASET_PATH}
  WORKLOAD_ID=${WORKLOAD_ID:-unspecified}
  SERVER_LIFECYCLE_ID=${SERVER_LIFECYCLE_ID:-default}
  LIFECYCLE_ORDER_INDEX=${LIFECYCLE_ORDER_INDEX:-0}
  LIFECYCLE_SEQUENCE=${LIFECYCLE_SEQUENCE:-unspecified}
  TIMESERIES_INTERVAL_S=${TIMESERIES_INTERVAL_S:-0}
  NPU_SAMPLE_INTERVAL_S=${NPU_SAMPLE_INTERVAL_S:-5}
  FORMAL_DATASET_SHA256=""
  WARMUP_DATASET_SHA256=""
  if [[ -n "$FORMAL_DATASET_PATH" ]]; then
    FORMAL_DATASET_SHA256=$(sha256sum "$FORMAL_DATASET_PATH" | awk '{print $1}')
  fi
  if [[ -n "$WARMUP_DATASET_PATH" ]]; then
    WARMUP_DATASET_SHA256=$(sha256sum "$WARMUP_DATASET_PATH" | awk '{print $1}')
  fi

  cat > "$RUN_ROOT/run-metadata.json" <<EOF
{"run_id":"$RUN_ID","pair_id":"$PAIR_ID","campaign":"$CAMPAIGN","graph":"$GRAPH_MODE","target_server_input_tokens":$TARGET_PROMPT,"evalscope_prompt_tokens":$EVALSCOPE_PROMPT,"min_tokens":$OUTPUT_TOKENS,"output_tokens":$OUTPUT_TOKENS,"concurrency":$CONCURRENCY,"requests":$REQUESTS,"warmup_requests":$WARMUP,"warmup_dataset_offset":$WARMUP_OFFSET,"separate_warmup":$SEPARATE_WARMUP,"dataset_kind":"$DATASET_KIND","formal_dataset_path":"$FORMAL_DATASET_PATH","formal_dataset_sha256":"$FORMAL_DATASET_SHA256","warmup_dataset_path":"$WARMUP_DATASET_PATH","warmup_dataset_sha256":"$WARMUP_DATASET_SHA256","workload_id":"$WORKLOAD_ID","server_lifecycle_id":"$SERVER_LIFECYCLE_ID","lifecycle_order_index":$LIFECYCLE_ORDER_INDEX,"lifecycle_sequence":"$LIFECYCLE_SEQUENCE","run_order_in_lifecycle":$REP,"timeseries_interval_s":$TIMESERIES_INTERVAL_S,"npu_sample_interval_s":$NPU_SAMPLE_INTERVAL_S,"seed":$SEED,"dataset_offset":$OFFSET,"temperature":0.0,"top_p":1.0,"sampling":"greedy","evalscope_version":"$EVALSCOPE_VERSION","strict_workload":true,"cache_state":"$CACHE_STATE","metric_boundary":"formal server metrics exclude the separately recorded warmup"}
EOF

  RUN_STARTED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  WARMUP_STARTED_AT_UTC=""
  WARMUP_ENDED_AT_UTC=""
  FORMAL_WARMUP=$WARMUP
  if (( SEPARATE_WARMUP == 1 )); then
    WARMUP_STARTED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    WARMUP_ROOT="$CLIENT_ROOT/warmup"
    mkdir -p "$WARMUP_ROOT"
    curl -fsS "$SERVER_URL/metrics" > "$SERVER_ROOT/pre-warmup-metrics.prom"
    curl -fsS "$SERVER_URL/debug/stats" > "$SERVER_ROOT/pre-warmup-stats.json"
    WARMUP_COMMAND=(
      evalscope perf
      --url "$SERVER_URL/v1/chat/completions"
      --api openai
      --model "$MODEL_NAME"
      --tokenizer-path "$MODEL_DIR"
      --dataset "$DATASET_KIND"
      --number "$WARMUP"
      --parallel "$CONCURRENCY"
      --warmup-num 0
      --min-prompt-length "$EVALSCOPE_PROMPT"
      --max-prompt-length "$EVALSCOPE_PROMPT"
      --min-tokens "$OUTPUT_TOKENS"
      --max-tokens "$OUTPUT_TOKENS"
      --temperature 0
      --top-p 1
      --seed "$SEED"
      --dataset-offset "$WARMUP_OFFSET"
      --stream
      --no-test-connection
      --total-timeout 21600
      --outputs-dir "$WARMUP_ROOT/evalscope"
      --no-timestamp
      --name "${RUN_ID}_warmup"
    )
    if [[ -n "$WARMUP_DATASET_PATH" ]]; then
      WARMUP_COMMAND+=(--dataset-path "$WARMUP_DATASET_PATH")
    fi
    printf '%q ' "${WARMUP_COMMAND[@]}" > "$WARMUP_ROOT/command.txt"
    printf '\n' >> "$WARMUP_ROOT/command.txt"
    set +e
    "${WARMUP_COMMAND[@]}" > "$WARMUP_ROOT/stdout.log" 2>&1
    WARMUP_STATUS=$?
    set -e
    echo "$WARMUP_STATUS" > "$WARMUP_ROOT/exit-code.txt"
    curl -fsS "$SERVER_URL/metrics" > "$SERVER_ROOT/before-metrics.prom" || true
    curl -fsS "$SERVER_URL/debug/stats" > "$SERVER_ROOT/before-stats.json" || true
    (( WARMUP_STATUS == 0 )) || {
      echo "$RUN_ID warmup failed with exit code $WARMUP_STATUS" >&2
      exit "$WARMUP_STATUS"
    }
    WARMUP_ENDED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    python "$ROOT/benchmarks/qwen3_32b_tp2_fp16/extract_workload_fingerprint.py" \
      "$WARMUP_ROOT/evalscope" \
      --tokenizer-path "$MODEL_DIR" \
      --repo-root "$ROOT" \
      --expected-prompt-tokens "$TARGET_PROMPT" \
      --expected-completion-tokens "$OUTPUT_TOKENS" \
      --output "$WARMUP_ROOT/workload-fingerprint.json"
    FORMAL_WARMUP=0
  else
    curl -fsS "$SERVER_URL/metrics" > "$SERVER_ROOT/before-metrics.prom"
    curl -fsS "$SERVER_URL/debug/stats" > "$SERVER_ROOT/before-stats.json"
  fi

  COMMAND=(
    evalscope perf
    --url "$SERVER_URL/v1/chat/completions"
    --api openai
    --model "$MODEL_NAME"
    --tokenizer-path "$MODEL_DIR"
    --dataset "$DATASET_KIND"
    --number "$REQUESTS"
    --parallel "$CONCURRENCY"
    --warmup-num "$FORMAL_WARMUP"
    --min-prompt-length "$EVALSCOPE_PROMPT"
    --max-prompt-length "$EVALSCOPE_PROMPT"
    --min-tokens "$OUTPUT_TOKENS"
    --max-tokens "$OUTPUT_TOKENS"
    --temperature 0
    --top-p 1
    --seed "$SEED"
    --dataset-offset "$OFFSET"
    --stream
    --no-test-connection
    --total-timeout 21600
    --outputs-dir "$CLIENT_ROOT/evalscope"
    --no-timestamp
    --name "$RUN_ID"
  )
  if [[ -n "$FORMAL_DATASET_PATH" ]]; then
    COMMAND+=(--dataset-path "$FORMAL_DATASET_PATH")
  fi
  printf '%q ' "${COMMAND[@]}" > "$CLIENT_ROOT/command.txt"
  printf '\n' >> "$CLIENT_ROOT/command.txt"
  FORMAL_STARTED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  FORMAL_STARTED_EPOCH=$(date +%s.%N)
  SAMPLER_PID=""
  SAMPLER_STATUS=0
  if [[ "$TIMESERIES_INTERVAL_S" != "0" ]]; then
    SAMPLER_COMMAND=(
      python "$ROOT/benchmarks/qwen3_32b_tp2_fp16/sample_runtime_timeseries.py"
      --server-url "$SERVER_URL"
      --output "$SERVER_ROOT/timeseries.jsonl"
      --interval "$TIMESERIES_INTERVAL_S"
      --npu-interval "$NPU_SAMPLE_INTERVAL_S"
      --npu-devices "$NPU_DEVICES"
    )
    printf '%q ' "${SAMPLER_COMMAND[@]}" > "$SERVER_ROOT/timeseries-command.txt"
    printf '\n' >> "$SERVER_ROOT/timeseries-command.txt"
    "${SAMPLER_COMMAND[@]}" > "$SERVER_ROOT/timeseries-sampler.log" 2>&1 &
    SAMPLER_PID=$!
    sleep 0.25
  fi
  set +e
  "${COMMAND[@]}" > "$CLIENT_ROOT/stdout.log" 2>&1
  STATUS=$?
  set -e
  if [[ -n "$SAMPLER_PID" ]]; then
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    set +e
    wait "$SAMPLER_PID"
    SAMPLER_STATUS=$?
    set -e
    echo "$SAMPLER_STATUS" > "$SERVER_ROOT/timeseries-exit-code.txt"
  fi
  FORMAL_ENDED_EPOCH=$(date +%s.%N)
  FORMAL_ENDED_AT_UTC=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  FORMAL_DURATION_S=$(python -c 'import sys; print(float(sys.argv[2]) - float(sys.argv[1]))' "$FORMAL_STARTED_EPOCH" "$FORMAL_ENDED_EPOCH")
  cat > "$RUN_ROOT/run-timing.json" <<EOF
{"run_started_at_utc":"$RUN_STARTED_AT_UTC","warmup_started_at_utc":"$WARMUP_STARTED_AT_UTC","warmup_ended_at_utc":"$WARMUP_ENDED_AT_UTC","formal_started_at_utc":"$FORMAL_STARTED_AT_UTC","formal_ended_at_utc":"$FORMAL_ENDED_AT_UTC","formal_started_epoch_s":$FORMAL_STARTED_EPOCH,"formal_ended_epoch_s":$FORMAL_ENDED_EPOCH,"formal_duration_s":$FORMAL_DURATION_S}
EOF
  curl -fsS "$SERVER_URL/metrics" > "$SERVER_ROOT/after-metrics.prom" || true
  curl -fsS "$SERVER_URL/debug/stats" > "$SERVER_ROOT/after-stats.json" || true
  echo "$STATUS" > "$CLIENT_ROOT/exit-code.txt"
  (( STATUS == 0 )) || { echo "$RUN_ID failed with exit code $STATUS" >&2; exit "$STATUS"; }
  (( SAMPLER_STATUS == 0 )) || { echo "$RUN_ID sampler failed with exit code $SAMPLER_STATUS" >&2; exit "$SAMPLER_STATUS"; }
  python "$ROOT/benchmarks/qwen3_32b_tp2_fp16/extract_workload_fingerprint.py" \
    "$CLIENT_ROOT/evalscope" \
    --tokenizer-path "$MODEL_DIR" \
    --repo-root "$ROOT" \
    --expected-prompt-tokens "$TARGET_PROMPT" \
    --expected-completion-tokens "$OUTPUT_TOKENS" \
    --output "$CLIENT_ROOT/workload-fingerprint.json"
  python "$ROOT/benchmarks/qwen3_32b_tp2_fp16/extract_request_metrics.py" \
    "$CLIENT_ROOT/evalscope" \
    --output "$CLIENT_ROOT/request-metrics.json"
done
