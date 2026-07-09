<div align="center">

# Lite Llama NPU

**A learning-oriented LLM inference engine for Huawei Ascend NPUs**

Built with PyTorch, `torch_npu`, Triton Ascend, and HCCL. The project implements
the core execution path instead of wrapping a high-level inference library.

[English](README.md) | [中文](README_CN.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)
![Ascend](https://img.shields.io/badge/Ascend-910B3-red)
![Version](https://img.shields.io/badge/version-0.0.13rc2-blue)

</div>

## Why This Project

Lite Llama NPU is a compact inference engine for studying how production-style
LLM systems work on Ascend hardware. It exposes the model executor, KV cache,
attention kernels, tensor parallel communication, continuous batching,
sampling, graph replay, and profiling paths as readable Python and Triton code.

The current development focus is multi-card inference for **Qwen3 Dense** and
**Qwen3 MoE** on Atlas 910B3.

This repository is intended for learning, profiling, and systems experiments.
It is not positioned as a production replacement for vLLM-Ascend or MindIE.

## Architecture

```text
                   OpenAI-compatible HTTP API
                              |
                              v
                 +--------------------------+
                 | Continuous Batch Scheduler|
                 | token budget / admission |
                 +-------------+------------+
                               |
                     Prefill   |   Decode
                               v
                 +--------------------------+
                 |      ModelExecutor       |
                 | request state / sampling |
                 +------+------------+------+
                        |            |
              +---------+--+      +--+----------------+
              | Qwen3 Dense|      | Qwen3 MoE         |
              | TP layers  |      | GMM / Routed GEMV |
              +---------+--+      +--+----------------+
                        |            |
                        +------+-----+
                               |
                 +-------------v------------+
                 | Paged KV + PagedAttention|
                 | FlashAttention / Decode  |
                 +-------------+------------+
                               |
                 +-------------v------------+
                 | NPU Graph / HCCL / NPU   |
                 +--------------------------+
```

## Capability Matrix

| Area | Capability | Status |
|---|---|---|
| Models | Qwen3-32B Dense | Supported |
| Models | Qwen3-30B-A3B MoE | Supported |
| Models | Qwen3-VL | Supported |
| Parallelism | Tensor Parallel | Supported |
| Parallelism | Single-node Expert Parallel | Experimental |
| Serving | OpenAI Chat/Completions API | Supported |
| Scheduling | Continuous Batching | Supported |
| Scheduling | Token-budget admission | Supported |
| Scheduling | Adaptive Chunked Prefill | Supported |
| KV cache | Paged KV Cache / PagedAttention | Supported |
| KV cache | Exact block-level Prefix Cache | Supported |
| Attention | FlashAttention2 no-pad Prefill | Supported |
| Attention | Flash Decoding | Supported |
| Graph execution | Decode NPU Graph | Supported for stable Dense shapes |
| MoE | `torch_npu` GMM | Supported |
| MoE | Routed GEMV and Triton Gather/Scatter | Supported |
| Profiling | Ascend Profiler / MindStudio Insight | Supported |
| Observability | Prometheus metrics, sampling-path counters, and runtime debug snapshot | Supported |
| Evaluation | EvalScope | Supported |
| Precision | FP16 | Main validated path |
| Precision | BF16 / W8A8 / W4A8 / FP8 | Not yet stabilized |
| Distributed | Multi-node TP/EP | Not yet supported |

## Benchmarks

All numbers below were measured on **2 × Atlas 910B3** with Qwen3-32B and
`TP=2`. They are selected from
[`docs/inference_performance_history.md`](docs/inference_performance_history.md).

### EvalScope service benchmarks

Latest v0.0.13rc2 server A/B, EvalScope random dataset, Qwen3-32B, TP=2,
concurrency 4, average input 178 tokens, Top-P sampling:

| Mode | Output throughput | Total throughput | Avg latency | TTFT | TPOT | ITL | Success |
|---|---:|---:|---:|---:|---:|---:|---:|
| `--decode_priority` | **39.00 tok/s** | 66.13 tok/s | 25.88 s | 13.438 s | 48.82 ms | 48.73 ms | 40 / 40 |
| `--no_decode_priority` | **56.06 tok/s** | 97.42 tok/s | 16.59 s | 1.889 s | 61.66 ms | 61.21 ms | 40 / 40 |

`--decode_priority` lowers decode-step latency but delays new prefill. Use it for
serving stability; use `--no_decode_priority` for offline throughput tests.

| Workload | Version | Concurrency | Avg input / output | Output throughput | Total throughput | TTFT | TPOT | ITL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Fixed-length Greedy | v0.0.8rc1 | 1 | 184 / 256 | **24.7089 tok/s** | 42.4685 tok/s | 702.7 ms | 37.9 ms | 37.7 ms |
| Fixed-length Greedy | v0.0.7rc1 | 4 | 155.975 / 254.125 | **63.3523 tok/s** | 102.2360 tok/s | 2.2958 s | 53.0 ms | 52.9 ms |
| Fixed-length Top-P | v0.0.7rc1 | 4 | 156 / 231.325 | **57.7176 tok/s** | 96.6409 tok/s | 1.4861 s | 62.9 ms | 61.6 ms |
| Mixed-length Greedy | v0.0.8rc1 | 4 | 285.475 / 245.7 | **50.1580 tok/s** | 108.436 tok/s | 5.1176 s | 57.8 ms | 57.1 ms |

Current v0.0.13rc2 changes are scheduler and sampling-path changes. The release
reports include the exact commands that should be used for fresh server-side
measurements before adding new benchmark rows.

### MoE kernel and graph evolution

Qwen3-30B-A3B, FP16, Batch 4, prompt approximately 128 tokens, output 256
tokens:

| Execution path | NPU Graph | Per-sequence throughput | Batch throughput | Time per token |
|---|---:|---:|---:|---:|
| v0.0.4rc1 TP + GMM + graph replay | Enabled | **31.7 tok/s** | **126.9 tok/s** | 31.53 ms |
| v0.0.5rc2 EP eager | Disabled | 5.5 tok/s | 22.1 tok/s | 181.09 ms |

> Benchmark caution: rows with different versions, prompt distributions,
> sampling modes, or output lengths are not strict apples-to-apples comparisons.
> The history document keeps the original test context and known limitations.

## Current Release Notes

- [v0.0.13rc2 Release Report](docs/releases/v0.0.13rc2.md) - adaptive chunked prefill and scheduler policy.
- [v0.0.11rc1 Release Report](docs/releases/v0.0.11rc1.md) - batched vocabulary-parallel Top-P sampling.
- [v0.0.10rc3 Release Report](docs/releases/v0.0.10rc3.md) - Prometheus observability.

## Quick Start

### Requirements

- Python 3.10
- PyTorch 2.7
- `torch_npu` 2.7
- CANN / Ascend Toolkit
- Atlas 910B3

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -r requirement.txt
export MODEL_DIR=/data/models/Qwen3-32B
```

### Start the Qwen3-32B server on two cards

```bash
cd /data/liuke/llama_lite_npu

ASCEND_RT_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --nproc_per_node=2 server.py \
  --checkpoints_dir "$MODEL_DIR" \
  --host 0.0.0.0 \
  --port 8213 \
  --page_size 16 \
  --max_seq_len 4096 \
  --compiled_model \
  --continuous_batching \
  --max_batch_size 32
```

```bash
curl http://127.0.0.1:8213/health
curl http://127.0.0.1:8213/metrics
curl http://127.0.0.1:8213/debug/stats
```

### Send a streaming request

```bash
curl -N http://127.0.0.1:8213/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": "Explain PagedAttention."}],
    "max_tokens": 128,
    "temperature": 0,
    "stream": true
  }'
```

### Run an EvalScope benchmark

```bash
evalscope perf \
  --url http://127.0.0.1:8213/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --tokenizer-path "$MODEL_DIR" \
  --dataset random \
  --number 40 \
  --parallel 4 \
  --min-prompt-length 128 \
  --max-prompt-length 512 \
  --max-tokens 256 \
  --temperature 0 \
  --stream
```

## Profiling

`examples/benchmark_tp.py` can collect Ascend Profiler traces:

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --nproc_per_node=2 examples/benchmark_tp.py \
  --checkpoints_dir "$MODEL_DIR" \
  --batch_size 4 \
  --prompt_len 128 \
  --max_gen_len 256 \
  --page_size 16 \
  --compiled_model \
  --warmup 2 \
  --iterations 5 \
  --profile \
  --profile_dir ./profiler_output
```

Open the generated directory with MindStudio Insight to inspect operator,
communication, memory, and timeline data.

## Observability

The server exports Prometheus metrics at `GET /metrics` and a compact JSON
runtime snapshot at `GET /debug/stats`. Metrics cover request QPS, failures by
stable reason, latency, TTFT, ITL, queue depth, scheduler states, token
throughput, KV page usage, preemption, and NPU Graph activity.

See the [observability guide](docs/observability.md) for metric names, PromQL
examples, and a Prometheus scrape configuration.

## Repository Guide

```text
lite_llama/
  continuous_batching.py     request lifecycle and scheduling
  executor/
    model_executor.py        model loading and execution orchestration
    npu_graph.py             fixed-shape decode graph capture/replay
    paged_attention.py       paged KV allocation and request page tables
  kernels/                   Triton Ascend and torch_npu kernels
  models/
    qwen3.py                 Dense Qwen3
    qwen3_moe.py             Qwen3 MoE
server.py                    OpenAI-compatible server
examples/benchmark_tp.py     TP benchmark and profiler entry point
docs/bug_records.md          engineering mistakes and root-cause reviews
```

## Documentation

- [Chinese README](README_CN.md)
- [Documentation index](docs/README.md)
- [Performance history](docs/inference_performance_history.md)
- [Engineering bug records](docs/bug_records.md)
- [Observability guide](docs/observability.md)
- [v0.0.10rc3 release notes](docs/releases/v0.0.10rc3.md)

## Current Limitations

- FP16 is the primary validated precision path.
- Multi-node TP/EP and mature MoE All-to-All are not available.
- Chunked Prefill is retained as an experimental path and is not recommended for
  the current short-to-medium prompt benchmark shape.
- Dynamic MoE routing and some HCCL operations restrict NPU Graph coverage.
- Performance data should always be interpreted with its exact workload.

## Acknowledgements

This project is based on and extends
[harleyszhang/lite_llama](https://github.com/harleyszhang/lite_llama).

## Citation

```bibtex
@misc{lite_llama,
  title        = {lite_llama},
  author       = {Litellama AI team},
  howpublished = {\url{https://github.com/harleyszhang/lite_llama}},
  year         = {2024}
}
```
