# 文档索引

## 当前Ascend推理文档

- [项目主文档](../README.md)：环境、能力矩阵、启动命令、性能和限制。
- [版本管理规范](versioning.md)：版本号、CHANGELOG、发布报告与Tag规则。
- [推理性能历史](inference_performance_history.md)：不同版本和测试工具的实测记录。
- [性能优化说明](performance_optimization.md)：项目中的主要性能方向。
- [Qwen3-VL支持方案](qwen3vl_support_plan.md)：多模态结构与实现说明。
- [`v0.0.6rc2`版本报告](releases/v0.0.6rc2.md)：Batch级Greedy通信修复。

## 设计与实现记录

- [`docs/superpowers/specs/`](superpowers/specs/)：功能设计文档。
- [`docs/superpowers/plans/`](superpowers/plans/)：实现计划和阶段拆分。
- [`docs/releases/`](releases/)：各版本相对上一版本的变化、验证和限制。

## 历史与上游资料

以下文档主要来自项目早期CUDA/ROCm阶段或保留的模型结构记录，不代表当前Atlas 910B3
推荐配置：

- `benchmark.md`
- `benchmark_models.md`
- `benchmark_models_history.md`
- `LlamaForCausalLM.md`
- `Qwen2ForCausalLM.md`
- `LlavaForConditionalGeneration.md`
- `LlavaNextForConditionalGeneration.md`

使用当前Qwen3 Dense、Qwen3 MoE、Continuous Batching、PagedAttention、NPU Graph和
Ascend Profiler时，应优先阅读根目录[`README.md`](../README.md)和最新版本报告。
