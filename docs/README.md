# 文档索引

本目录记录 Lite Llama NPU 的设计、版本演进、性能数据和错误复盘。除保留的外部英文基线文档外，项目自有中文文档统一使用中文描述；模型名、命令行参数、文件路径、算子名和错误原文保留英文。

## 当前项目文档

- [项目主页](../README.md)：项目能力、启动方式、性能摘要和当前路线图。
- [版本管理规范](versioning.md)：版本号、更新日志、发布文档和 Tag 规则。
- [更新日志](../CHANGELOG.md)：按时间倒序记录版本变更。
- [推理性能历史记录](inference_performance_history.md)：记录各版本核心测试结果。
- [性能优化记录](performance_optimization.md)：记录主要优化方向和阶段性结论。
- [错误复盘记录](bug_records.md)：记录框架开发中的错误、排查过程和修复经验。
- [Qwen3-VL 支持方案](qwen3vl_support_plan.md)：记录多模态模型适配方案。

## 发布文档

- [v0.0.10rc2 发布记录](releases/v0.0.10rc2.md)：GitHub 首页、中英文 README、性能表与核心模块说明。
- [v0.0.10rc1 发布记录](releases/v0.0.10rc1.md)：Decode 热路径清理。
- [v0.0.9rc5 发布记录](releases/v0.0.9rc5.md)：Paged Chunk FlashAttention 运行时 Tile 选择。
- [v0.0.9rc4 发布记录](releases/v0.0.9rc4.md)：Paged Chunk FlashAttention 在 910B3 上的编译回退。
- [v0.0.9rc3 发布记录](releases/v0.0.9rc3.md)：TP Chunked Prefill Rank 同步修复。
- [v0.0.9rc2 发布记录](releases/v0.0.9rc2.md)：Chunked Prefill 接入 Paged Chunk FlashAttention。
- [v0.0.7rc2 发布记录](releases/v0.0.7rc2.md)：KV 引擎和调度改造记录。

## 外部对比基线

- [vLLM-Ascend 性能基线](vllm_ascend_benchmark.md)：英文文档，记录外部 vLLM-Ascend 对比数据和 EvalScope 测试矩阵。

## 历史继承文档

以下文档主要来自上游 lite_llama 或早期 CUDA/ROCm 适配阶段，仅作为历史参考，不代表当前 Atlas 910B3、Qwen3 Dense、Qwen3 MoE、Continuous Batching、PagedAttention、NPU Graph 或 Ascend Profiler 的最新状态：

- `benchmark.md`
- `benchmark_models.md`
- `benchmark_models_history.md`
- `benchamrk_kernels.md`
- `LlamaForCausalLM.md`
- `Qwen2ForCausalLM.md`
- `LlavaForConditionalGeneration.md`
- `LlavaNextForConditionalGeneration.md`
