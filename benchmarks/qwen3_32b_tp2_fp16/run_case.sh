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
  EVALSCOPE_VERSION=$(python -c 'from importlib.metadata import version; print(version("evalscope"))')
  CACHE_STATE=${CACHE_STATE_OVERRIDE:-kv-cache-cold-unique-offset}

  curl -fsS "$SERVER_URL/metrics" > "$SERVER_ROOT/before-metrics.prom"
  curl -fsS "$SERVER_URL/debug/stats" > "$SERVER_ROOT/before-stats.json"
  cat > "$RUN_ROOT/run-metadata.json" <<EOF
{"run_id":"$RUN_ID","campaign":"$CAMPAIGN","graph":"$GRAPH_MODE","target_server_input_tokens":$TARGET_PROMPT,"evalscope_prompt_tokens":$EVALSCOPE_PROMPT,"min_tokens":$OUTPUT_TOKENS,"output_tokens":$OUTPUT_TOKENS,"concurrency":$CONCURRENCY,"requests":$REQUESTS,"warmup_requests":$WARMUP,"seed":$SEED,"dataset_offset":$OFFSET,"temperature":0.0,"top_p":1.0,"sampling":"greedy","evalscope_version":"$EVALSCOPE_VERSION","strict_workload":true,"cache_state":"$CACHE_STATE","metric_boundary":"client and server snapshots stored separately"}
EOF

  COMMAND=(
    evalscope perf
    --url "$SERVER_URL/v1/chat/completions"
    --api openai
    --model "$MODEL_NAME"
    --tokenizer-path "$MODEL_DIR"
    --dataset random
    --number "$REQUESTS"
    --parallel "$CONCURRENCY"
    --warmup-num "$WARMUP"
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
  printf '%q ' "${COMMAND[@]}" > "$CLIENT_ROOT/command.txt"
  printf '\n' >> "$CLIENT_ROOT/command.txt"
  set +e
  "${COMMAND[@]}" > "$CLIENT_ROOT/stdout.log" 2>&1
  STATUS=$?
  set -e
  curl -fsS "$SERVER_URL/metrics" > "$SERVER_ROOT/after-metrics.prom" || true
  curl -fsS "$SERVER_URL/debug/stats" > "$SERVER_ROOT/after-stats.json" || true
  echo "$STATUS" > "$CLIENT_ROOT/exit-code.txt"
  (( STATUS == 0 )) || { echo "$RUN_ID failed with exit code $STATUS" >&2; exit "$STATUS"; }
done
