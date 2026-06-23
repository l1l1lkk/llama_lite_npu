<div align="center">

# Lite Llama NPU

**面向昇腾 NPU 的轻量级大模型推理框架**

基于 PyTorch、torch_npu 与 Triton Ascend，从模型结构、KV Cache、Attention、算子融合、张量并行、连续批处理和性能分析等环节探索大模型推理优化。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)
![Ascend](https://img.shields.io/badge/Ascend-910B3-red)
![Version](https://img.shields.io/badge/version-0.0.10rc1-blue)

</div>

## 项目简介

Lite Llama NPU 的目标不是封装 Transformers，而是实现一条可以观察、修改、验证和复盘的昇腾大模型推理链路。项目当前重点围绕 **Qwen3 Dense、Qwen3 MoE 与 Qwen3-VL 在 Atlas 910B3 上的多卡推理** 展开。

当前项目适合用于：

- 学习大模型推理框架内部结构；
- 理解 Prefill、Decode、KV Cache、PagedAttention 和 Continuous Batching；
- 验证 Tensor Parallel、Expert Parallel、NPU Graph 和算子融合；
- 使用 EvalScope、Ascend Profiler 与 MindStudio Insight 分析性能瓶颈；
- 记录框架设计中的错误和修复过程。

项目仍处于持续开发阶段，不建议直接作为生产服务使用。

## 最新版本

当前版本：**0.0.10rc1**，发布日期：**2026-06-23**。

本版本重点清理 Decode 热路径，为后续合并主分支做准备：

- 默认关闭 TP worker 每 token 的 Decode 状态 Host 校验；
- Worker 侧 Decode 默认不再返回 Host token 列表；
- 跳过 Worker-only Prefix Cache host copy；
- 保留调试开关 `LLAMA_LITE_NPU_VALIDATE_TP_DECODE_STATE=1`；
- 完成文档索引和错误复盘记录清理。

相关文档：

- [v0.0.10rc1 发布记录](docs/releases/v0.0.10rc1.md)
- [更新日志](CHANGELOG.md)
- [版本管理与发布规范](docs/versioning.md)
- [推理性能历史记录](docs/inference_performance_history.md)
- [文档索引](docs/README.md)
- [错误复盘记录](docs/bug_records.md)

## 当前支持能力

### 模型

| 模型 | 状态 | 说明 |
|---|---|---|
| Qwen3-32B | 已支持 | 当前 Dense 主测试模型 |
| Qwen3-30B-A3B | 已支持 | MoE 主测试模型 |
| Qwen3-VL | 已支持 | 多模态路径保留 |
| Llama / Qwen2 / Llava | 历史继承 | 来自上游 lite_llama，当前不是主要优化目标 |

### 推理框架特性

- FP16 权重加载与推理；
- Tensor Parallel；
- Qwen3 MoE Tensor Parallel；
- 单机 Expert Parallel 实验路径；
- Continuous Batching；
- Paged KV Cache；
- PagedAttention；
- Prefix Cache；
- Decode NPU Graph；
- FlashAttention2 no-pad Prefill；
- Flash Decoding；
- MoE GMM；
- MoE Routed GEMV；
- Triton Gather / Scatter；
- OpenAI Chat Completions 兼容接口；
- EvalScope 性能测试；
- Ascend Profiler 采集；
- MindStudio Insight 可视化分析。

## 当前架构

```text
用户请求
  ↓
OpenAI 兼容 Server
  ↓
Continuous Batching 调度器
  ↓
Prefill / Decode 执行路径
  ↓
ModelExecutor
  ↓
Qwen3 Dense / Qwen3 MoE / Qwen3-VL
  ↓
Paged KV Cache + PagedAttention
  ↓
torch_npu / Triton Ascend / HCCL
```

## 性能摘要

完整历史数据见：[推理性能历史记录](docs/inference_performance_history.md)。

### Qwen3-32B 与 vLLM-Ascend 对比基线

测试条件：2 × Atlas 910B3，TP=2，并发 1，EvalScope，Qwen3-32B。

| 指标 | Lite Llama NPU | vLLM-Ascend 0.8.4rc2 | 对比 |
|---|---:|---:|---:|
| Output Throughput | 24.0606 tok/s | 7.6409 tok/s | 约 3.15 倍 |
| Total Throughput | 26.9575 tok/s | 7.8122 tok/s | 约 3.45 倍 |
| TPOT | 40.6 ms | 131.1 ms | 降低约 69.0% |
| TTFT | 261.2 ms | 392.7 ms | 降低约 33.5% |

说明：外部 vLLM-Ascend 数据来自第三方公开测试结果，测试脚本、版本、Prompt 分布和采样参数可能存在差异，因此该表只作为阶段性参考。

### Qwen3-30B-A3B MoE 双卡测试

测试条件：2 × Atlas 910B3，FP16，TP=2，Batch=4，Prompt 约 128 tokens，生成 256 tokens。

| 版本与路径 | NPU Graph | 单序列吞吐 | Batch 吞吐 | 单 token 耗时 |
|---|---:|---:|---:|---:|
| v0.0.4rc1 TP Graph | 开启 | 31.7 tok/s | 126.9 tok/s | 31.53 ms |
| v0.0.5rc2 EP Eager | 关闭 | 5.5 tok/s | 22.1 tok/s | 181.09 ms |

MoE 的性能高度依赖专家路由、专家并行、GMM、NPU Graph 和调度路径。当前 EP 路径主要用于验证架构，不代表最终 MoE 性能上限。

## 环境要求

推荐环境：

- Python 3.10；
- PyTorch 2.7；
- torch_npu 2.7；
- CANN 与 Ascend Toolkit；
- Atlas 910B3；
- 已执行 Ascend 环境变量脚本。

示例：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

安装 Python 依赖：

```bash
pip install -r requirement.txt
```

## 快速启动

### Qwen3-32B OpenAI 兼容服务

```bash
cd /data/liuke/llama_lite_npu

ASCEND_RT_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --nproc_per_node=2 server.py \
  --checkpoints_dir /data/liuke/llama_lite_npu/my_weight/Qwen3-32B/ \
  --host 0.0.0.0 \
  --port 8213 \
  --page_size 16 \
  --max_seq_len 4096 \
  --compiled_model \
  --continuous_batching \
  --max_batch_size 32
```

健康检查：

```bash
curl http://127.0.0.1:8213/health
```

### Qwen3-32B CLI 推理

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --nproc_per_node=2 cli_qwen3_tp.py \
  --checkpoints_dir /data/liuke/llama_lite_npu/my_weight/Qwen3-32B/ \
  --page_size 16 \
  --max_seq_len 4096 \
  --max_gen_len 1024 \
  --compiled_model \
  --disable_thinking
```

### Qwen3-30B-A3B MoE CLI 推理

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --nproc_per_node=2 cli_qwen3_moe_tp.py \
  --checkpoints_dir /data/liuke/llama_lite_npu/my_weight/Qwen3-30B-A3B/ \
  --page_size 16 \
  --max_seq_len 4096 \
  --max_gen_len 1024 \
  --disable_thinking \
  --no_compiled_model
```

说明：MoE 路径包含动态专家路由，当前不建议默认开启 Decode NPU Graph。

## EvalScope 测试

### 固定长度 Greedy 单并发

```bash
evalscope perf \
  --url http://127.0.0.1:8213/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --tokenizer-path /data/liuke/llama_lite_npu/my_weight/Qwen3-32B/ \
  --dataset random \
  --number 20 \
  --parallel 1 \
  --min-prompt-length 184 \
  --max-prompt-length 184 \
  --max-tokens 256 \
  --temperature 0 \
  --stream \
  --name lite_llama_npu_greedy_p1
```

### 混合长度并发测试

```bash
evalscope perf \
  --url http://127.0.0.1:8213/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --tokenizer-path /data/liuke/llama_lite_npu/my_weight/Qwen3-32B/ \
  --dataset random \
  --number 40 \
  --parallel 4 \
  --min-prompt-length 128 \
  --max-prompt-length 512 \
  --max-tokens 256 \
  --temperature 0 \
  --stream \
  --name lite_llama_npu_mixed_p4
```

## Benchmark 脚本

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --nproc_per_node=2 examples/benchmark_tp.py \
  --checkpoints_dir /data/liuke/llama_lite_npu/my_weight/Qwen3-32B/ \
  --batch_size 4 \
  --prompt_len 128 \
  --max_gen_len 256 \
  --page_size 16 \
  --compiled_model \
  --warmup 2 \
  --iterations 5
```

## Ascend Profiler

```bash
ASCEND_RT_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --nproc_per_node=2 examples/benchmark_tp.py \
  --checkpoints_dir /data/liuke/llama_lite_npu/my_weight/Qwen3-32B/ \
  --batch_size 4 \
  --prompt_len 128 \
  --max_gen_len 256 \
  --page_size 16 \
  --compiled_model \
  --warmup 2 \
  --iterations 5 \
  --profile \
  --profile_dir /data/liuke/llama_lite_npu/profiler_output \
  --profile_level Level1 \
  --profile_aic_metrics PipeUtilization \
  --no_profile_data_simplification
```

采集完成后，将 `profiler_output` 下载到本地，用 MindStudio Insight 打开。

## 当前限制

- 当前主要验证 FP16，BF16、W8A8、W4A8、FP8 尚未形成稳定路径；
- MoE Expert Parallel 仍是单机实验路径，还未实现成熟的多机 All-to-All；
- Chunked Prefill 已有安全路径，但短 Prompt 场景下通常不如 Packed Prefill；
- Paged Chunk FlashAttention 仍处于实验阶段；
- NPU Graph 对动态 shape、动态专家路由和部分 HCCL 场景有限制；
- 文档中的性能数据必须结合测试工具、并发、Prompt 分布和采样参数理解。

## 文档入口

- [文档索引](docs/README.md)
- [错误复盘记录](docs/bug_records.md)
- [推理性能历史记录](docs/inference_performance_history.md)
- [性能优化记录](docs/performance_optimization.md)
- [vLLM-Ascend 性能基线](docs/vllm_ascend_benchmark.md)

## 项目结构

```text
lite_llama/
  executor/              # 模型执行、TP 控制、NPU Graph、Paged KV
  models/                # Qwen3、Qwen3 MoE、Qwen3-VL 等模型结构
  kernels/               # Triton / torch_npu 自定义算子
  continuous_batching.py # 连续批处理调度
examples/
  benchmark_tp.py        # TP benchmark 与 profiler 采集
docs/
  bug_records.md         # 错误复盘
  releases/              # 版本发布记录
server.py                # OpenAI 兼容服务入口
cli_qwen3_tp.py          # Qwen3 Dense CLI
cli_qwen3_moe_tp.py      # Qwen3 MoE CLI
```

## 致谢

本项目基于 [harleyszhang/lite_llama](https://github.com/harleyszhang/lite_llama) 学习和改造，重点面向昇腾 NPU 推理链路、Qwen3 系列模型、多卡并行和性能分析进行扩展。

## Citation

```bibtex
@misc{lite_llama,
  title        = {lite_llama},
  author       = {Litellama AI team},
  howpublished = {\url{https://github.com/harleyszhang/lite_llama}},
  year         = {2024}
}
```
