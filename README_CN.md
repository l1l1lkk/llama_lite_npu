# Lite Llama NPU

**面向昇腾 NPU 的学习型大模型推理引擎**

[English](README.md) | [中文](README_CN.md)

![Version](https://img.shields.io/badge/version-0.0.13rc1-blue)

## 项目定位

Lite Llama NPU 是一个用于学习推理框架内部机制的项目。它不是对 vLLM-Ascend 或 MindIE 的上层封装，而是直接实现模型执行、KV 缓存、并行通信、连续批处理、采样和可观测性链路。

当前重点是 Atlas 910B3 上的 Qwen3 Dense、Qwen3 MoE 和 Qwen3-VL 推理。

## 架构概览

```text
OpenAI 兼容 API -> Continuous Batch Scheduler -> ModelExecutor -> Paged KV / Attention -> NPU Graph / HCCL
```

## 能力矩阵

| 领域 | 能力 | 状态 |
|---|---|---|
| 模型 | Qwen3-32B Dense | 已支持 |
| 模型 | Qwen3-30B-A3B MoE | 已支持 |
| 模型 | Qwen3-VL | 已支持 |
| 并行 | Tensor Parallel | 已支持 |
| 服务 | OpenAI Chat/Completions API | 已支持 |
| 调度 | Continuous Batching | 已支持 |
| 调度 | Token 预算准入 | 已支持 |
| 调度 | 自适应 Chunked Prefill | 已支持 |
| KV 缓存 | Paged KV Cache / PagedAttention | 已支持 |
| KV 缓存 | 完整块 Prefix Cache | 已支持 |
| Attention | FlashAttention2 no-pad Prefill | 已支持 |
| 图执行 | Decode NPU Graph | Dense 稳定形状可用 |
| 采样 | Vocab-parallel Greedy / Top-P | 已支持 |
| 可观测性 | Prometheus 指标与运行时调试快照 | 已支持 |

## 性能数据

以下结果来自 **2 张 Atlas 910B3、Qwen3-32B、TP=2** 的历史实测。原始上下文见 [`docs/inference_performance_history.md`](docs/inference_performance_history.md)。

| 测试场景 | 版本 | 并发 | 平均输入 / 输出 | 输出吞吐 | 总吞吐 | TTFT | TPOT | ITL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 固定长度 Greedy | v0.0.8rc1 | 1 | 184 / 256 | **24.7089 tok/s** | 42.4685 tok/s | 702.7 ms | 37.9 ms | 37.7 ms |
| 固定长度 Greedy | v0.0.7rc1 | 4 | 155.975 / 254.125 | **63.3523 tok/s** | 102.2360 tok/s | 2.2958 s | 53.0 ms | 52.9 ms |
| 固定长度 Top-P | v0.0.7rc1 | 4 | 156 / 231.325 | **57.7176 tok/s** | 96.6409 tok/s | 1.4861 s | 62.9 ms | 61.6 ms |
| 混合长度 Greedy | v0.0.8rc1 | 4 | 285.475 / 245.7 | **50.1580 tok/s** | 108.436 tok/s | 5.1176 s | 57.8 ms | 57.1 ms |

v0.0.11rc1 和 v0.0.13rc1 是采样路径与调度策略更新，新的服务器实测数据会按 release 文档中的命令采集后再更新到性能表。

## 当前版本说明

- [v0.0.13rc1 发布记录](docs/releases/v0.0.13rc1.md)：自适应 Chunked Prefill 与调度策略。
- [v0.0.11rc1 发布记录](docs/releases/v0.0.11rc1.md)：批量 vocab-parallel Top-P 采样。
- [v0.0.10rc3 发布记录](docs/releases/v0.0.10rc3.md)：Prometheus 可观测性。

## 快速启动

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
pip install -r requirement.txt
export MODEL_DIR=/data/liuke/llama_lite_npu/my_weight/Qwen3-32B

ASCEND_RT_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --nproc_per_node=2 server.py \
  --checkpoints_dir ${MODEL_DIR} \
  --host 0.0.0.0 \
  --port 8213 \
  --max_seq_len 4096 \
  --page_size 16 \
  --compiled_model \
  --continuous_batching \
  --max_batch_size 32
```

## 可观测性

```bash
curl http://127.0.0.1:8213/metrics
curl http://127.0.0.1:8213/debug/stats
```

重点指标包括请求延迟、队列深度、TTFT、ITL、KV 页使用量、NPU Graph replay/fallback、采样 candidate/fallback 统计等。

## 致谢

本项目基于 [lite_llama](https://github.com/harleyszhang/lite_llama) 的思想继续扩展，主要用于学习昇腾推理框架实现。
