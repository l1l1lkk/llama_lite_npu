<div align="center">

# Lite Llama NPU

**面向昇腾 NPU 的学习型大模型推理引擎**

基于 PyTorch、`torch_npu`、Triton Ascend 与 HCCL，自主实现模型执行、KV
缓存、并行通信、连续批处理、采样、图执行和性能分析链路。

[English](README.md) | [中文](README_CN.md)

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)
![Ascend](https://img.shields.io/badge/Ascend-910B3-red)
![Version](https://img.shields.io/badge/version-0.0.10rc2-blue)

</div>

## 项目定位

Lite Llama NPU 是一个用于学习昇腾大模型推理系统的轻量级引擎。项目不依赖
高层推理框架封装，而是直接展示模型执行器、KV Cache、Attention 内核、张量
并行通信、连续批处理、采样、NPU Graph 和性能分析的实现。

当前重点是 **Qwen3 Dense** 与 **Qwen3 MoE** 在 Atlas 910B3 上的多卡推理。

项目适合用于推理框架学习、性能分析和系统实验，不定位为 vLLM-Ascend 或
MindIE 的生产替代品。

## 系统架构

```text
                    OpenAI 兼容 HTTP 接口
                              |
                              v
                 +--------------------------+
                 |   Continuous Batching    |
                 | 请求准入 / Token 预算调度 |
                 +-------------+------------+
                               |
                     Prefill   |   Decode
                               v
                 +--------------------------+
                 |      ModelExecutor       |
                 | 请求状态 / 采样 / 执行编排 |
                 +------+------------+------+
                        |            |
              +---------+--+      +--+----------------+
              | Qwen3 Dense|      | Qwen3 MoE         |
              | TP 模型层   |      | GMM / Routed GEMV |
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

## 能力矩阵

| 领域 | 能力 | 状态 |
|---|---|---|
| 模型 | Qwen3-32B Dense | 已支持 |
| 模型 | Qwen3-30B-A3B MoE | 已支持 |
| 模型 | Qwen3-VL | 已支持 |
| 并行 | Tensor Parallel | 已支持 |
| 并行 | 单机 Expert Parallel | 实验性支持 |
| 服务 | OpenAI Chat/Completions 接口 | 已支持 |
| 调度 | Continuous Batching | 已支持 |
| 调度 | Token 预算准入 | 已支持 |
| KV 缓存 | Paged KV Cache / PagedAttention | 已支持 |
| KV 缓存 | 完整块 Prefix Cache | 已支持 |
| Attention | FlashAttention2 no-pad Prefill | 已支持 |
| Attention | Flash Decoding | 已支持 |
| 图执行 | Decode NPU Graph | Dense 稳定形状可用 |
| MoE | `torch_npu` GMM | 已支持 |
| MoE | Routed GEMV 与 Triton Gather/Scatter | 已支持 |
| 分析 | Ascend Profiler / MindStudio Insight | 已支持 |
| 测试 | EvalScope | 已支持 |
| 精度 | FP16 | 当前主要验证路径 |
| 精度 | BF16 / W8A8 / W4A8 / FP8 | 尚未稳定 |
| 分布式 | 多机 TP/EP | 尚未支持 |

## 性能数据

以下结果均来自 **2 × Atlas 910B3、Qwen3-32B、TP=2** 的实测记录，原始上下文
见 [`docs/inference_performance_history.md`](docs/inference_performance_history.md)。

### EvalScope 服务测试

| 测试场景 | 版本 | 并发 | 平均输入 / 输出 | 输出吞吐 | 总吞吐 | TTFT | TPOT | ITL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 固定长度 Greedy | v0.0.8rc1 | 1 | 184 / 256 | **24.7089 tok/s** | 42.4685 tok/s | 702.7 ms | 37.9 ms | 37.7 ms |
| 固定长度 Greedy | v0.0.7rc1 | 4 | 155.975 / 254.125 | **63.3523 tok/s** | 102.2360 tok/s | 2.2958 s | 53.0 ms | 52.9 ms |
| 固定长度 Top-P | v0.0.7rc1 | 4 | 156 / 231.325 | **57.7176 tok/s** | 96.6409 tok/s | 1.4861 s | 62.9 ms | 61.6 ms |
| 混合长度 Greedy | v0.0.8rc1 | 4 | 285.475 / 245.7 | **50.1580 tok/s** | 108.436 tok/s | 5.1176 s | 57.8 ms | 57.1 ms |

### MoE 算子与图执行演进

Qwen3-30B-A3B，FP16，Batch 4，Prompt 约 128 tokens，输出 256 tokens：

| 执行路径 | NPU Graph | 单序列吞吐 | Batch 吞吐 | 单 token 耗时 |
|---|---:|---:|---:|---:|
| v0.0.4rc1 TP + GMM + Graph Replay | 开启 | **31.7 tok/s** | **126.9 tok/s** | 31.53 ms |
| v0.0.5rc2 EP Eager | 关闭 | 5.5 tok/s | 22.1 tok/s | 181.09 ms |

> 注意：不同版本、Prompt 分布、采样方式或输出长度的数据不能直接视为严格横向
> 对比。性能历史文档保留了每组数据的测试条件和已知限制。

## 快速开始

### 环境要求

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

### 双卡启动 Qwen3-32B 服务

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
```

### 发送流式请求

```bash
curl -N http://127.0.0.1:8213/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": "解释一下 PagedAttention"}],
    "max_tokens": 128,
    "temperature": 0,
    "stream": true
  }'
```

### 运行 EvalScope

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

## 性能分析

`examples/benchmark_tp.py` 支持采集 Ascend Profiler 数据：

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

采集完成后，可以使用 MindStudio Insight 查看算子、通信、内存和时间线数据。

## 代码导航

```text
lite_llama/
  continuous_batching.py     请求生命周期与调度
  executor/
    model_executor.py        模型加载和执行编排
    npu_graph.py             固定形状 Decode Graph 捕获与回放
    paged_attention.py       Paged KV 分配和请求页表
  kernels/                   Triton Ascend 与 torch_npu 算子
  models/
    qwen3.py                 Qwen3 Dense
    qwen3_moe.py             Qwen3 MoE
server.py                    OpenAI 兼容服务
examples/benchmark_tp.py     TP 性能测试和 Profiler 入口
docs/bug_records.md          错误复盘和根因记录
```

## 文档

- [英文 README](README.md)
- [文档索引](docs/README.md)
- [性能历史记录](docs/inference_performance_history.md)
- [错误复盘记录](docs/bug_records.md)
- [v0.0.10rc2 发布记录](docs/releases/v0.0.10rc2.md)

## 当前限制

- FP16 是当前主要验证的精度路径；
- 尚未支持成熟的多机 TP/EP 和 MoE All-to-All；
- Chunked Prefill 保留为实验路径，当前中短 Prompt 测试不建议默认开启；
- 动态 MoE 路由和部分 HCCL 算子会限制 NPU Graph 覆盖范围；
- 所有性能数据都必须结合具体 workload 理解。

## 致谢

本项目基于 [harleyszhang/lite_llama](https://github.com/harleyszhang/lite_llama)
学习和扩展。

## 引用

```bibtex
@misc{lite_llama,
  title        = {lite_llama},
  author       = {Litellama AI team},
  howpublished = {\url{https://github.com/harleyszhang/lite_llama}},
  year         = {2024}
}
```
