# vLLM-Ascend Performance Benchmark

> Historical/unverified：本文是旧手工测试记录，不具备 canonical v2 的冻结 workload、统一 strict validator 和 Git-tracked raw bundle。当前跨框架规则见 [`docs/benchmarks/README.md`](benchmarks/README.md)，不得把本文数字升级为 `v0.0.15rc2` 严格对比。

This document records vLLM-Ascend benchmark results used as the external comparison baseline for `llama_lite_npu`.

The historical `vllm-ascend 0.8.4rc2` screenshot is treated only as historical reference. New comparison data should be collected from the current vLLM-Ascend environment with complete launch parameters and EvalScope outputs.

## Test Record Requirements

For every vLLM-Ascend benchmark run, record:

- vLLM launch command and relevant environment variables.
- Version information:
  - `torch`
  - `torch_npu`
  - `vllm`
  - `vllm_ascend`
  - CANN / driver environment if available.
- Model, dtype, tensor parallel size, graph mode, block size, max sequence settings.
- EvalScope command.
- EvalScope summary metrics:
  - average input tokens
  - average output tokens
  - average latency
  - TTFT
  - ITL
  - TPOT
  - output throughput
  - total throughput
  - request throughput
  - success / failed requests
- Workload throughput table if EvalScope prints it.

Recommended version snapshot command:

```bash
python - <<'PY'
import torch
import torch_npu
import vllm

try:
    import vllm_ascend
except Exception:
    vllm_ascend = None

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("vllm:", vllm.__version__)
print("vllm_ascend:", getattr(vllm_ascend, "__version__", "unknown"))
PY

npu-smi info
```

## Recommended vLLM-Ascend Launch Baseline

Target baseline:

- Model: Qwen3-32B
- Device: Atlas 910B3, 2 cards
- Tensor Parallel: 2
- Dtype: FP16
- API: OpenAI-compatible server
- Graph mode: record exact `--compilation-config` used

Example launch command:

```bash
export ASCEND_RT_VISIBLE_DEVICES=6,7
export OMP_NUM_THREADS=1
export TASK_QUEUE_ENABLE=1
export HCCL_OP_EXPANSION_MODE=AIV
export VLLM_ASCEND_ENABLE_FLASHCOMM1=1

vllm serve /data/model_weights/Qwen3-32B \
  --host 0.0.0.0 \
  --port 8000 \
  --served-model-name Qwen3-32B \
  --trust-remote-code \
  --distributed-executor-backend mp \
  --tensor-parallel-size 2 \
  --dtype float16 \
  --max-model-len 4096 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 4096 \
  --block-size 128 \
  --gpu-memory-utilization 0.90 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,4,8,16,32]}'
```

If `FULL_DECODE_ONLY` is not available in the target environment, record the actual fallback mode explicitly, for example:

```bash
--compilation-config '{"cudagraph_mode":"PIECEWISE"}'
```

## Benchmark Matrix

### Matrix A: Single-request Greedy Decode, 128 input / 256 output

EvalScope command:

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --dataset random \
  --tokenizer-path /data/workspace/model_weight/Qwen3-32B \
  --number 20 \
  --parallel 1 \
  --min-prompt-length 128 \
  --max-prompt-length 128 \
  --min-tokens 256 \
  --max-tokens 256 \
  --temperature 0 \
  --stream \
  --name vllm-qwen3-32b-fp16-p1-128x256-greedy
```

Result collected at `2026-06-18 03:38:23`.

| Field | Value |
|---|---:|
| Framework | vLLM-Ascend |
| Model | Qwen3-32B |
| API | OpenAI chat completions |
| Dataset | random |
| Requests | 20 |
| Success / Failed | 20 / 0 |
| Parallel | 1 |
| Prompt length | 128 |
| Output length | 256 |
| Temperature | 0 |
| Stream | true |
| Test duration | 181.21 s |
| Request throughput | 0.11 req/s |
| Avg latency | 9.06 s |
| TTFT | 307.71 ms |
| ITL | 34.19 ms |
| TPOT | 34.32 ms |
| Avg input tokens | 128.00 |
| Avg output tokens | 256.00 |
| Output throughput | 28.25 tok/s |
| Total throughput | 42.38 tok/s |
| Decode throughput | 29.14 tok/s |

Percentile highlights:

| Metric | P50 | P90 | P99 | Max |
|---|---:|---:|---:|---:|
| Latency | 8.94 s | 8.95 s | 11.33 s | 11.33 s |
| TTFT | 188.99 ms | 195.86 ms | 2563.29 ms | 2563.29 ms |
| ITL | 34.29 ms | 34.40 ms | 35.28 ms | 37.67 ms |
| TPOT | 34.32 ms | 34.33 ms | 34.38 ms | 34.38 ms |
| Output tok/s | 28.63 | 28.64 | 28.66 | 28.66 |
| Total tok/s | 42.95 | 42.97 | 42.99 | 42.99 |

Workload throughput:

| Metric | Overall | Last 30s | Steady drop 20% |
|---|---:|---:|---:|
| Total Prompt tok/s | 14.13 | 14.32 | 14.32 |
| New Prompt tok/s | 14.13 | 14.32 | 14.32 |
| Cached Prompt tok/s | 0.00 | 0.00 | 0.00 |
| Completion tok/s | 28.25 | 28.63 | 28.63 |

Notes:

- EvalScope printed `PyTorch was not found`, but tokenizer loading and benchmark execution completed successfully.
- `uvloop` was not installed. This is acceptable for `parallel=1`, but high-concurrency runs should install `uvloop` or `evalscope[perf]` to reduce rate-control noise.
- The exact vLLM/vLLM-Ascend version and server launch command still need to be attached to this run before using it as a final published comparison.

### Matrix B: Single-request Top-P Decode, 128 input / 256 output

EvalScope command:

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --dataset random \
  --tokenizer-path /data/workspace/model_weight/Qwen3-32B \
  --number 20 \
  --parallel 1 \
  --min-prompt-length 128 \
  --max-prompt-length 128 \
  --min-tokens 256 \
  --max-tokens 256 \
  --temperature 0.6 \
  --top-p 0.9 \
  --stream \
  --name vllm-qwen3-32b-fp16-p1-128x256-topp
```

Result collected at `2026-06-18 03:49:58`.

| Field | Value |
|---|---:|
| Framework | vLLM-Ascend |
| Model | Qwen3-32B |
| API | OpenAI chat completions |
| Dataset | random |
| Requests | 20 |
| Success / Failed | 20 / 0 |
| Parallel | 1 |
| Prompt length | 128 |
| Output length | 256 |
| Temperature | 0.6 |
| Top-p | 0.9 |
| Stream | true |
| Test duration | 180.87 s |
| Request throughput | 0.11 req/s |
| Avg latency | 9.04 s |
| TTFT | 185.28 ms |
| ITL | 34.60 ms |
| TPOT | 34.74 ms |
| Avg input tokens | 128.00 |
| Avg output tokens | 256.00 |
| Output throughput | 28.31 tok/s |
| Total throughput | 42.46 tok/s |
| Decode throughput | 28.79 tok/s |

Percentile highlights:

| Metric | P50 | P90 | P99 | Max |
|---|---:|---:|---:|---:|
| Latency | 9.04 s | 9.05 s | 9.05 s | 9.05 s |
| TTFT | 185.66 ms | 187.62 ms | 188.38 ms | 188.38 ms |
| ITL | 34.71 ms | 34.79 ms | 35.61 ms | 39.20 ms |
| TPOT | 34.74 ms | 34.74 ms | 34.75 ms | 34.75 ms |
| Output tok/s | 28.31 | 28.32 | 28.33 | 28.33 |
| Total tok/s | 42.46 | 42.48 | 42.49 | 42.49 |

Workload throughput:

| Metric | Overall | Last 30s | Steady drop 20% |
|---|---:|---:|---:|
| Total Prompt tok/s | 14.15 | 14.15 | 14.15 |
| New Prompt tok/s | 14.15 | 14.15 | 14.15 |
| Cached Prompt tok/s | 0.00 | 0.00 | 0.00 |
| Completion tok/s | 28.31 | 28.31 | 28.31 |

Notes:

- This Top-P result is effectively the same speed as the Greedy run under the same vLLM-Ascend setup.
- Compared with Matrix A, TPOT changes from `34.32 ms` to `34.74 ms`, about `+1.2%`.
- TTFT is lower than Matrix A because Matrix A had one high-tail TTFT request; the steady decode rate is the more useful signal here.

### Matrix C: Concurrent Greedy Decode, parallel=4

EvalScope command:

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --dataset random \
  --tokenizer-path /data/workspace/model_weight/Qwen3-32B \
  --number 80 \
  --parallel 4 \
  --min-prompt-length 128 \
  --max-prompt-length 128 \
  --min-tokens 256 \
  --max-tokens 256 \
  --temperature 0 \
  --stream \
  --name vllm-qwen3-32b-fp16-p4-128x256-greedy
```

Result collected at `2026-06-18 06:06:58`.

| Field | Value |
|---|---:|
| Framework | vLLM-Ascend |
| Model | Qwen3-32B |
| API | OpenAI chat completions |
| Dataset | random |
| Requests | 80 |
| Success / Failed | 80 / 0 |
| Parallel | 4 |
| Prompt length | 128 |
| Output length | 256 |
| Temperature | 0 |
| Stream | true |
| Test duration | 186.94 s |
| Request throughput | 0.43 req/s |
| Avg latency | 9.35 s |
| TTFT | 392.77 ms |
| ITL | 34.97 ms |
| TPOT | 35.11 ms |
| Avg input tokens | 128.00 |
| Avg output tokens | 256.00 |
| Output throughput | 109.56 tok/s |
| Total throughput | 164.33 tok/s |
| Decode throughput | 28.48 tok/s |

Percentile highlights:

| Metric | P50 | P90 | P99 | Max |
|---|---:|---:|---:|---:|
| Latency | 9.34 s | 9.35 s | 9.42 s | 9.42 s |
| TTFT | 485.68 ms | 496.03 ms | 551.71 ms | 551.71 ms |
| ITL | 34.67 ms | 34.80 ms | 35.92 ms | 177.71 ms |
| TPOT | 35.23 ms | 35.77 ms | 35.90 ms | 35.90 ms |
| Output tok/s | 27.40 | 27.42 | 27.42 | 27.42 |
| Total tok/s | 41.10 | 41.13 | 41.14 | 41.14 |

Workload throughput:

| Metric | Overall | Last 30s | Steady drop 20% |
|---|---:|---:|---:|
| Total Prompt tok/s | 54.78 | 54.81 | 54.80 |
| New Prompt tok/s | 54.78 | 54.81 | 54.80 |
| Cached Prompt tok/s | 0.00 | 0.00 | 0.00 |
| Completion tok/s | 109.56 | 109.63 | 109.61 |

Notes:

- Compared with Matrix A `parallel=1`, aggregate output throughput scales from `28.25 tok/s` to `109.56 tok/s`, about `3.88x`.
- Per-request TPOT changes from `34.32 ms` to `35.11 ms`, only about `+2.3%`, so vLLM-Ascend keeps decode latency stable while increasing concurrency.
- TTFT increases from `307.71 ms` to `392.77 ms`, which is expected with more concurrent prefill/decode work.

### Matrix D: Concurrent Greedy Decode, parallel=8

EvalScope command:

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --dataset random \
  --tokenizer-path /data/workspace/model_weight/Qwen3-32B \
  --number 160 \
  --parallel 8 \
  --min-prompt-length 128 \
  --max-prompt-length 128 \
  --min-tokens 256 \
  --max-tokens 256 \
  --temperature 0 \
  --stream \
  --name vllm-qwen3-32b-fp16-p8-128x256-greedy
```

Result collected at `2026-06-18 06:34:21`.

| Field | Value |
|---|---:|
| Framework | vLLM-Ascend |
| Model | Qwen3-32B |
| API | OpenAI chat completions |
| Dataset | random |
| Requests | 160 |
| Success / Failed | 160 / 0 |
| Parallel | 8 |
| Prompt length | 128 |
| Output length | 256 |
| Temperature | 0 |
| Stream | true |
| Test duration | 195.34 s |
| Request throughput | 0.82 req/s |
| Avg latency | 9.77 s |
| TTFT | 404.49 ms |
| ITL | 36.57 ms |
| TPOT | 36.71 ms |
| Avg input tokens | 128.00 |
| Avg output tokens | 256.00 |
| Output throughput | 209.69 tok/s |
| Total throughput | 314.53 tok/s |
| Decode throughput | 27.24 tok/s |

Percentile highlights:

| Metric | P50 | P90 | P99 | Max |
|---|---:|---:|---:|---:|
| Latency | 9.73 s | 9.84 s | 9.94 s | 9.94 s |
| TTFT | 398.73 ms | 517.48 ms | 583.99 ms | 584.57 ms |
| ITL | 36.56 ms | 36.71 ms | 37.72 ms | 216.68 ms |
| TPOT | 36.57 ms | 37.26 ms | 37.74 ms | 37.84 ms |
| Output tok/s | 26.32 | 26.33 | 26.34 | 26.34 |
| Total tok/s | 39.48 | 39.50 | 39.51 | 39.51 |

Workload throughput:

| Metric | Overall | Last 30s | Steady drop 20% |
|---|---:|---:|---:|
| Total Prompt tok/s | 104.84 | 105.27 | 105.05 |
| New Prompt tok/s | 104.84 | 105.27 | 105.05 |
| Cached Prompt tok/s | 0.00 | 0.00 | 0.00 |
| Completion tok/s | 209.69 | 210.55 | 210.09 |

Notes:

- Compared with Matrix A `parallel=1`, aggregate output throughput scales from `28.25 tok/s` to `209.69 tok/s`, about `7.42x`.
- Compared with Matrix C `parallel=4`, aggregate output throughput scales from `109.56 tok/s` to `209.69 tok/s`, about `1.91x`.
- Per-request TPOT changes from `34.32 ms` at `parallel=1` to `36.71 ms` at `parallel=8`, about `+7.0%`.
- vLLM-Ascend is still scaling efficiently at `parallel=8`; the main visible cost is slightly higher TPOT and TTFT.

### Matrix E: Historical Long-output Case, short prompt / 2048 output

EvalScope command:

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --dataset random \
  --tokenizer-path /data/workspace/model_weight/Qwen3-32B \
  --number 15 \
  --parallel 1 \
  --min-prompt-length 20 \
  --max-prompt-length 45 \
  --max-tokens 2048 \
  --temperature 0 \
  --stream \
  --name vllm-qwen3-32b-fp16-p1-shortprompt-2048-greedy
```

Result collected at `2026-06-18 06:39:45`.

| Field | Value |
|---|---:|
| Framework | vLLM-Ascend |
| Model | Qwen3-32B |
| API | OpenAI chat completions |
| Dataset | random |
| Requests | 15 |
| Success / Failed | 15 / 0 |
| Parallel | 1 |
| Prompt length range | 20-45 |
| Max output tokens | 2048 |
| Temperature | 0 |
| Stream | true |
| Test duration | 383.88 s |
| Request throughput | 0.04 req/s |
| Avg latency | 25.59 s |
| TTFT | 211.71 ms |
| ITL | 34.40 ms |
| TPOT | 34.40 ms |
| Avg input tokens | 32.33 |
| Avg output tokens | 737.87 |
| Output throughput | 28.83 tok/s |
| Total throughput | 30.10 tok/s |
| Decode throughput | 29.07 tok/s |

Percentile highlights:

| Metric | P50 | P90 | P99 | Max |
|---|---:|---:|---:|---:|
| Latency | 20.55 s | 42.59 s | 57.41 s | 57.41 s |
| TTFT | 213.69 ms | 215.66 ms | 216.43 ms | 216.43 ms |
| ITL | 34.37 ms | 34.80 ms | 35.49 ms | 37.97 ms |
| TPOT | 34.36 ms | 34.54 ms | 34.68 ms | 34.68 ms |
| Input tokens | 34.00 | 41.00 | 44.00 | 44.00 |
| Output tokens | 593.00 | 1228.00 | 1650.00 | 1650.00 |
| Output tok/s | 28.84 | 28.89 | 28.90 | 28.90 |
| Total tok/s | 30.46 | 31.00 | 31.16 | 31.16 |

Workload throughput:

| Metric | Overall | Last 30s | Steady drop 20% |
|---|---:|---:|---:|
| Total Prompt tok/s | 1.26 | 0.71 | 1.24 |
| New Prompt tok/s | 1.26 | 0.71 | 1.24 |
| Cached Prompt tok/s | 0.00 | 0.00 | 0.00 |
| Completion tok/s | 28.83 | 28.89 | 28.83 |

Notes:

- This run corresponds to the earlier historical vLLM-style benchmark path: short prompt and long generated output.
- Compared with the old external screenshot baseline of vLLM-Ascend `0.8.4rc2` reporting about `7.64 output tok/s`, the current vLLM-Ascend result is about `28.83 / 7.64 = 3.77x` faster.
- Decode speed remains close to Matrix A despite much longer outputs: `34.40 ms` TPOT here versus `34.32 ms` in the 128x256 greedy run.
- Average output length is `737.87`, not 2048, because the server/model stopped earlier for many requests.

### Matrix F: Prefill Stress, 1024 input / 128 output, parallel=4

EvalScope command:

```bash
evalscope perf \
  --url http://127.0.0.1:8000/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --dataset random \
  --tokenizer-path /data/workspace/model_weight/Qwen3-32B \
  --number 40 \
  --parallel 4 \
  --min-prompt-length 1024 \
  --max-prompt-length 1024 \
  --min-tokens 128 \
  --max-tokens 128 \
  --temperature 0 \
  --stream \
  --name vllm-qwen3-32b-fp16-p4-1024x128-greedy
```

Result collected at `2026-06-18 06:49:52`.

| Field | Value |
|---|---:|
| Framework | vLLM-Ascend |
| Model | Qwen3-32B |
| API | OpenAI chat completions |
| Dataset | random |
| Requests | 40 |
| Success / Failed | 40 / 0 |
| Parallel | 4 |
| Prompt length | 1024 |
| Output length | 128 |
| Temperature | 0 |
| Stream | true |
| Test duration | 53.97 s |
| Request throughput | 0.74 req/s |
| Avg latency | 5.39 s |
| TTFT | 650.76 ms |
| ITL | 37.05 ms |
| TPOT | 37.34 ms |
| Avg input tokens | 1024.00 |
| Avg output tokens | 128.00 |
| Output throughput | 94.87 tok/s |
| Total throughput | 853.85 tok/s |
| Decode throughput | 26.78 tok/s |

Percentile highlights:

| Metric | P50 | P90 | P99 | Max |
|---|---:|---:|---:|---:|
| Latency | 5.39 s | 5.39 s | 5.46 s | 5.46 s |
| TTFT | 895.98 ms | 897.28 ms | 957.78 ms | 957.78 ms |
| ITL | 35.38 ms | 35.50 ms | 36.71 ms | 436.00 ms |
| TPOT | 38.51 ms | 40.07 ms | 40.18 ms | 40.18 ms |
| Input tokens | 1024.00 | 1024.00 | 1024.00 | 1024.00 |
| Output tokens | 128.00 | 128.00 | 128.00 | 128.00 |
| Output tok/s | 23.75 | 23.76 | 23.76 | 23.76 |
| Total tok/s | 213.79 | 213.84 | 213.84 | 213.84 |

Workload throughput:

| Metric | Overall | Last 30s | Steady drop 20% |
|---|---:|---:|---:|
| Total Prompt tok/s | 759.00 | 759.98 | 829.92 |
| New Prompt tok/s | 759.00 | 759.98 | 829.92 |
| Cached Prompt tok/s | 0.00 | 0.00 | 0.00 |
| Completion tok/s | 94.87 | 95.00 | 103.74 |

Notes:

- This run stresses prefill much more than the 128x256 decode-heavy cases.
- Compared with Matrix C `parallel=4, 128x256`, output throughput drops from `109.56 tok/s` to `94.87 tok/s`, about `-13.4%`.
- Total throughput rises sharply from `164.33 tok/s` to `853.85 tok/s` because prompt tokens dominate this workload.
- TTFT rises from `392.77 ms` to `650.76 ms`, reflecting the higher prefill cost at 1024 input tokens.
- This is one of the key workloads for evaluating scheduler, packed prefill, chunked prefill, and KV-cache engine design.

## Comparison Notes Against llama_lite_npu

Current directly comparable `llama_lite_npu` data is incomplete because recent local results were mostly collected with `examples/benchmark_tp.py`, while this vLLM-Ascend record is an EvalScope OpenAI-server benchmark.

Known `llama_lite_npu` Qwen3-32B records:

| Version / Path | Test Type | Sampling | Input / Output | Main Result |
|---|---|---|---|---:|
| v0.0.6rc2 | `benchmark_tp.py`, batch=4 | greedy | ~128 / 256 | 22.1 tok/s per sequence, 88.6 batch tok/s, 45.16 ms/token |
| v0.0.6rc2 | `benchmark_tp.py`, batch=4 | temp=0.6, top_p=0.9 | ~128 / 256 | 17.9 tok/s per sequence, 71.7 batch tok/s, 55.83 ms/token |
| Earlier NPU Graph server run | EvalScope, parallel=1 | not fully aligned | ~59.5 / 494.5 avg | 24.06 output tok/s, 26.96 total tok/s |

Preliminary reading:

- On the 128x256 single-request greedy path, vLLM-Ascend reports `TPOT=34.32 ms` and `Output throughput=28.25 tok/s`.
- The latest `llama_lite_npu` local batch benchmark reports `45.16 ms/token` and `22.1 tok/s` per sequence.
- If compared only as a rough decode-latency signal, vLLM-Ascend is about `45.16 / 34.32 = 1.32x` faster per generated token.
- This is not a strict apples-to-apples result yet because the measurement frontends differ:
  - vLLM-Ascend: EvalScope -> OpenAI server, `parallel=1`.
  - llama_lite_npu: local benchmark script, `batch_size=4`.
- To publish a strict comparison, collect the same EvalScope matrix for `llama_lite_npu` and vLLM-Ascend with identical prompt/output length, sampling, stream mode, and concurrency.
