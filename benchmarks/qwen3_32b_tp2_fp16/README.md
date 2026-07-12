# Qwen3-32B TP=2 FP16 benchmark contract

This directory defines the reproducible primary server baseline for
Qwen3-32B on 2 x Atlas 910B3. It replaces ad-hoc table-only measurements;
historical figures remain historical and are not silently mixed into this
baseline.

## Fixed environment

- Runtime: `server.py`, OpenAI-compatible chat completion, streaming.
- Model/checkpoint: `Qwen3-32B` at the exact `MODEL_DIR` in `baseline.env`.
- Execution dtype: FP16. The upstream checkpoint metadata may say BF16, but
  `lite_llama/executor/model_executor.py` loads and asserts FP16 parameters and
  FP16 KV storage; retain both facts in the environment snapshot.
- Hardware: two Atlas 910B3 devices, physical NPU IDs `6,7`, TP=2.
- Git SHA and dirty status, CANN, PyTorch, torch_npu, triton-ascend and
  EvalScope versions are mandatory outputs of `collect_env.sh`; do not copy
  versions from a previous campaign.
- Server defaults fixed here: max sequence length 4096, page size 16,
  continuous batching, max batch size 32 and decode priority enabled.
- Sampling: greedy (`temperature=0`, `top_p=1`), streaming, one choice.

`config.json` metadata alone is not proof of execution dtype. The environment
snapshot, server log, git SHA and implementation assertion together define the
runtime.

## Workload and run rules

The P0 matrix is prompt/output `128/256` at concurrency 1, 4, 8 and 16, plus
`512/256/c4` and `2048/256/c4`. Every case has two warmup requests per worker
and three formal runs. EvalScope random data uses seed 42. Each formal run has
a deterministic, unique `dataset_offset`, which prevents exact-prefix KV-cache
hits between formal runs while keeping the generated workload reproducible.

The target prompt length is the token count observed by the server after the
Qwen3 chat template. `EVALSCOPE_PROMPT` is calibrated separately because the
chat template adds tokens. A run is valid only when the EvalScope summary
shows the expected average input and output lengths (or a documented deviation
with request-level evidence), all requests succeed, and greedy settings match.

Fixed-output campaigns set both `--min-tokens` and `--max-tokens`. The server
masks EOS logits independently for each batch row until its `min_tokens`
threshold; a table-only average is not sufficient evidence. Run
`validate_strict_workload.py` to check the summary plus every reported input
and output token percentile, request counts, exit codes and greedy arguments.
Rejected calibration attempts remain server-only and are indexed by the
campaign manifest rather than mixed into formal aggregates.

Cache vocabulary:

- `kv-cache-cold`: unique prompt token sequences/offsets with no reusable exact
  prefix. The model and graph are already warmed; this does not mean a cold
  process or uncaptured graph.
- `kv-cache-warm`: an intentional replay of exactly the same prompt set against
  the same server process. Warm-cache results must use a different case label
  and must not be merged with this cold primary baseline.
- `process-cold`: a newly started server before model/graph warmup. Never use
  process-cold requests as formal measurements.

## Metric contract

Client-side metrics come only from EvalScope raw artifacts:

- TTFT: client request dispatch to first streamed output token.
- ITL: mean interval between adjacent streamed output tokens.
- TPOT: `(E2E latency - TTFT) / (output tokens - 1)` per request, then the
  EvalScope aggregate.
- E2E latency: client dispatch to final response completion.
- Output throughput: total completed output tokens / formal wall time.
- Total throughput: completed input plus output tokens / formal wall time.
- QPS: successful formal requests / formal wall time.
- Success rate: successful / total formal requests.

Server-side metrics come only from `/metrics` and `/debug/stats` before/after
snapshots. Queue wait is scheduler enqueue to first admission and is summarized
from the delta of `lite_llama_queue_wait_seconds_sum/count`. Server request
latency starts at scheduler entry, so it is not interchangeable with EvalScope
E2E latency. Every table column must say `client_` or `server_` when ambiguity
is possible.

Percentiles may be reported only when the corresponding EvalScope
`benchmark_percentile.json` or Prometheus histogram buckets are retained. This
contract does not infer percentiles from averages and does not claim profiler
data unless a profiler artifact exists.

## Artifact layout

Every accepted campaign has two explicitly separated artifact layers.

For paired Graph ablation, freeze each input sequence as a line-by-line JSONL
dataset and retain it under `workload/` in the bundle. The same concurrency pair
must have identical formal/warmup dataset SHA256 values, EvalScope arguments,
request counts, seed and offsets. Because concurrent responses can complete in
a different order, observed request fingerprints are compared as an exact
token-sequence multiset; the frozen JSONL SHA proves original input order.

The Git-tracked reproducibility bundle is the durable source for published
numbers and ships with the project:

```text
benchmarks/results/<campaign>/
  campaign-manifest.json
  environment/**
  workload/**
  <graph-mode>/server/{start-command.txt,start/stop metrics and stats,...}
  <graph-mode>/<case>/run-XX/
    run-metadata.json
    client/{command.txt,exit-code.txt,evalscope/**/{args,summary,percentile}.json}
    server/{before/after metrics.prom,before/after stats.json}
  summary.csv
  aggregate.csv
```

The server-only bulky layer is optional retention for databases, HTML,
complete logs and failed/calibration attempts:

```text
benchmark-results/<campaign>/
  environment/{baseline.env,git.txt,runtime.txt,model-config.json,...}
  <graph-mode>/server/{start-command.txt,server.log,...}
  <graph-mode>/<case>/run-XX/
    run-metadata.json
    client/{command.txt,stdout.log,exit-code.txt,evalscope/**}
    server/{before-metrics.prom,after-metrics.prom,before-stats.json,after-stats.json}
  summary.csv
```

`campaign-manifest.json` lists every selected file plus every omitted
server-only file with source path, size and SHA256. SQLite, HTML, full stdout,
full server logs and profiler output may be omitted only when the tracked JSON
and metrics remain sufficient to rebuild all published statistics. A
performance number is publishable only when it can be rebuilt from the
Git-tracked bundle; server retention is not a dependency. Hand-copied tables
without per-run evidence are not accepted.

## Execution

```bash
cd /data/liuke/llama_lite_npu
CAMPAIGN=20260711_qwen3_32b_tp2_fp16
bash benchmarks/qwen3_32b_tp2_fp16/collect_env.sh "$CAMPAIGN"
bash benchmarks/qwen3_32b_tp2_fp16/start_server.sh on "$CAMPAIGN"

# EVALSCOPE_PROMPT must first be calibrated to the target server input length.
bash benchmarks/qwen3_32b_tp2_fp16/run_case.sh "$CAMPAIGN" on 128 106 256 1 12
bash benchmarks/qwen3_32b_tp2_fp16/run_case.sh "$CAMPAIGN" on 128 106 256 4 24
bash benchmarks/qwen3_32b_tp2_fp16/run_case.sh "$CAMPAIGN" on 128 106 256 8 48

python benchmarks/qwen3_32b_tp2_fp16/summarize.py \
  "benchmark-results/$CAMPAIGN"
bash benchmarks/qwen3_32b_tp2_fp16/stop_server.sh on "$CAMPAIGN"

# Run this where the server campaign is available, then copy the output into
# the Git worktree at benchmarks/results/$CAMPAIGN.
python benchmarks/qwen3_32b_tp2_fp16/build_bundle.py \
  "benchmark-results/$CAMPAIGN" "/tmp/${CAMPAIGN}_bundle"

# Must pass without access to benchmark-results/ or the server.
python benchmarks/qwen3_32b_tp2_fp16/validate_bundle.py \
  "benchmarks/results/$CAMPAIGN"
python benchmarks/qwen3_32b_tp2_fp16/validate_strict_workload.py \
  "benchmarks/results/$CAMPAIGN" --compare
```

Graph off is a separate server lifecycle using `start_server.sh off`; never
toggle graph mode inside a running campaign process. Only a PID created in the
campaign artifact directory may be stopped by `stop_server.sh`.
