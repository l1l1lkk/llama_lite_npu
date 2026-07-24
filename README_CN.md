# Lite Llama NPU

**面向昇腾 NPU 的学习型大模型推理引擎**

[English](README.md) | [中文](README_CN.md)

![Version](https://img.shields.io/badge/version-0.0.15rc2-blue)

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
| 服务 | 逐请求 `min_tokens` 固定输出控制 | Continuous Batching 已支持 |
| 调度 | Continuous Batching | 已支持 |
| 调度 | Token 预算准入 | 已支持 |
| 调度 | 自适应 Chunked Prefill | 已支持 |
| KV 缓存 | Paged KV Cache / PagedAttention | 已支持 |
| KV 缓存 | 完整块 Prefix Cache | 已支持 |
| Attention | FlashAttention2 no-pad Prefill | 已支持 |
| 图执行 | Decode NPU Graph | Dense 稳定形状可用 |
| 采样 | Vocab-parallel Greedy / Top-P | 已支持 |
| 可观测性 | Prometheus 指标与运行时调试快照 | 已支持 |

## MoE Runtime 正确性

v0.0.15rc2 保留 v0.0.15rc1 引入的 DeepSeek V2/V3 MoE 组件能力，并正式规定每个
版本对应一个 release 分支和一个 annotated tag。该组件在保持 tuple 解包兼容的通用
MoE 边界上增加
兼容：softmax/sigmoid grouped top-k、V3 correction bias 仅参与选择、routed/shared
experts、官方字段配置、HF→canonical→runtime 权重布局和有界单层 safetensors 读取。
Qwen3 参数名、checkpoint layout、state-dict key 和默认执行行为保持不变。

独立 CPU FP32 reference 是数值 oracle。显式 DeepSeek component 命令在物理 NPU 6
完成 2/2、无 skip，覆盖 V2/V3 × FP16/BF16 和真实 GMM；Qwen 保护套件在物理 NPU 7
完成 6/6、无 skip。这是组件 correctness 证据，不是完整模型或性能结论。

DeepSeek 边界不包含 MLA、attention、KV/RoPE、完整 decoder/CausalLM、W8A8、
all-to-all/EPLB 或非连续 expert map。EP 仅为 replicated-token 连续 expert slice，
真实 TP2/EP2 NPU collective 尚未验证；DeepSeek decode Graph 保持 fail-closed，V4 routing
不支持。详见 [v0.0.15rc2 发布记录](docs/releases/v0.0.15rc2.md)和
[DeepSeekMoE Runtime 设计](docs/deepseek_moe_runtime_design.md)。

## 性能数据

以下结果来自 **2 张 Atlas 910B3、Qwen3-32B、TP=2** 的历史实测。原始上下文见 [`docs/inference_performance_history.md`](docs/inference_performance_history.md)。

### 最新严格 NPU Graph benchmark（v0.0.13rc3）

Qwen3-32B、TP=2、FP16、prompt 128、`min_tokens=max_tokens=256`、greedy；每个 on/off 配对使用相同冻结请求并完成三轮正式测试：

| 并发 | Graph on E2E | Graph off E2E | E2E 降低 | Graph on 输出吞吐 | Graph off 输出吞吐 | 吞吐比 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.2149 s | 49.6692 s | **79.43%** | 25.060 tok/s | 5.154 tok/s | **4.862x** |
| 4 | 21.0396 s | 96.3659 s | **78.17%** | 45.332 tok/s | 9.946 tok/s | **4.558x** |

12 个正式 run 全部严格满足输入 128、输出 256、失败数 0。这是项目内部 Graph on/off 消融，不是与 vLLM-Ascend 的对比。逐 run JSON、metrics、token 指纹和离线重建工具位于 `benchmarks/results/20260712_qwen3_32b_tp2_fp16_graph_ablation/`。

| 测试场景 | 版本 | 并发 | 平均输入 / 输出 | 输出吞吐 | 总吞吐 | TTFT | TPOT | ITL |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 固定长度 Greedy | v0.0.8rc1 | 1 | 184 / 256 | **24.7089 tok/s** | 42.4685 tok/s | 702.7 ms | 37.9 ms | 37.7 ms |
| 固定长度 Greedy | v0.0.7rc1 | 4 | 155.975 / 254.125 | **63.3523 tok/s** | 102.2360 tok/s | 2.2958 s | 53.0 ms | 52.9 ms |
| 固定长度 Top-P | v0.0.7rc1 | 4 | 156 / 231.325 | **57.7176 tok/s** | 96.6409 tok/s | 1.4861 s | 62.9 ms | 61.6 ms |
| 混合长度 Greedy | v0.0.8rc1 | 4 | 285.475 / 245.7 | **50.1580 tok/s** | 108.436 tok/s | 5.1176 s | 57.8 ms | 57.1 ms |

v0.0.13rc3 在 Continuous Batching 路径新增逐请求 `min_tokens`，并在达到阈值前按 batch row 于采样前屏蔽 EOS。

## 当前版本说明

- [v0.0.15rc2 发布记录](docs/releases/v0.0.15rc2.md)：每版本独立 release 分支/tag 规范与冻结源码跨平台校验。
- [v0.0.15rc1 发布记录](docs/releases/v0.0.15rc1.md)：DeepSeek V2/V3 MoE 组件兼容与单卡 NPU correctness。
- [v0.0.14rc1 发布记录](docs/releases/v0.0.14rc1.md)：通用 Qwen3 MoE runtime 边界与分层 reference correctness。
- [v0.0.13rc3 发布记录](docs/releases/v0.0.13rc3.md)：固定输出控制与严格 Graph 消融。
- [v0.0.13rc2 发布记录](docs/releases/v0.0.13rc2.md)：发布验证兼容性。
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

需要观察真实请求经过 Scheduler、Prefill、Decode 和模型层的进展时，可以启动
Trace CLI：

```bash
python -m lite_llama.trace_cli run --trace-level layer --no-open -- \
  --checkpoints_dir my_weight/Qwen3-32B --port 8213
```

浏览器访问 `http://127.0.0.1:8213/debug/trace`。详细的事件语义、SSE 接口、
JSONL 记录、TP Rank 文件和 NPU Graph 限制见
[推理请求与模型层可视化](docs/inference_trace.md)。

## 致谢

本项目基于 [lite_llama](https://github.com/harleyszhang/lite_llama) 的思想继续扩展，主要用于学习昇腾推理框架实现。
