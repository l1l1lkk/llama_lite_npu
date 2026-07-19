# 文档索引

本目录记录 Lite Llama NPU 的设计、版本演进、性能数据和错误复盘。除必要的
外部英文资料外，项目自有文档统一使用中文；命令、文件路径、接口名称和代码
原文保留英文。

## 当前项目文档

- [项目首页](../README.md)：英文项目介绍、快速开始、性能摘要和当前路线。
- [中文项目首页](../README_CN.md)：与英文首页同步的中文版本。
- [版本管理规范](versioning.md)：版本号、变更日志、发布文档与 Tag 规则。
- [变更日志](../CHANGELOG.md)：按时间倒序记录版本变化。
- [推理性能历史](inference_performance_history.md)：记录各版本核心测试结果。
- [性能优化记录](performance_optimization.md)：记录主要优化方向和阶段结论。
- [错误复盘记录](bug_records.md)：记录框架开发中的错误、排查和修复经验。
- [可观测性与 Prometheus 指标](observability.md)：指标、PromQL 和抓取配置。
- [Qwen3-VL 支持方案](qwen3vl_support_plan.md)：多模态模型适配说明。
- [Qwen3 MoE Runtime 设计](qwen3_moe_runtime_design.md)：独立 reference、通用
  router/executor 边界与连续 TP/EP placement。
- [Qwen3 MoE Runtime 正确性验证](qwen3_moe_runtime_validation.md)：CPU/NPU、
  Graph、TP/EP 与独立 FP32 oracle 的分层验证证据。
- [DeepSeekMoE Runtime 设计](deepseek_moe_runtime_design.md)：V2/V3 grouped routing、
  shared expert、权重布局、单层 checkpoint reader 与组件级 NPU 正确性契约。

## 发布文档

- [v0.0.15rc2 发布记录](releases/v0.0.15rc2.md)：每版本独立 release 分支/tag 规范与冻结源码跨平台校验。
- [v0.0.15rc1 发布记录](releases/v0.0.15rc1.md)：DeepSeek V2/V3 MoE 组件兼容与单卡 NPU correctness。
- [v0.0.14rc1 发布记录](releases/v0.0.14rc1.md)：通用 Qwen3 MoE runtime 边界与分层 reference correctness。
- [v0.0.13rc3 发布记录](releases/v0.0.13rc3.md)：`min_tokens` 固定输出控制与严格 NPU Graph 消融。
- [v0.0.13rc2 发布记录](releases/v0.0.13rc2.md)：发布验证兼容性。
- [v0.0.13rc1 发布记录](releases/v0.0.13rc1.md)：Decode Priority 调度。
- [v0.0.10rc3 发布记录](releases/v0.0.10rc3.md)：Prometheus 可观测性。
- [v0.0.10rc2 发布记录](releases/v0.0.10rc2.md)：GitHub 首页与文档整理。
- [v0.0.10rc1 发布记录](releases/v0.0.10rc1.md)：Decode 热路径清理。
- [v0.0.9rc5 发布记录](releases/v0.0.9rc5.md)：Paged Chunk FlashAttention
  运行时 Tile 选择。
- [v0.0.9rc4 发布记录](releases/v0.0.9rc4.md)：Paged Chunk FlashAttention
  在 910B3 上的编译回退。
- [v0.0.9rc3 发布记录](releases/v0.0.9rc3.md)：TP Chunked Prefill Rank
  同步修复。
- [v0.0.9rc2 发布记录](releases/v0.0.9rc2.md)：Chunked Prefill 接入
  Paged Chunk FlashAttention。
- [v0.0.7rc2 发布记录](releases/v0.0.7rc2.md)：KV 引擎和调度更新。

## 外部对比基线

- [vLLM-Ascend 性能基线](vllm_ascend_benchmark.md)：外部
  vLLM-Ascend 对比数据和 EvalScope 测试说明。

## 历史教程文档

下列文档主要来自上游 lite_llama 的 CUDA/ROCm 学习阶段，仅作为历史参考，
不代表当前 Atlas 910B3、Qwen3 Dense、Qwen3 MoE、Continuous Batching、
PagedAttention、NPU Graph 与 Ascend Profiler 的支持状态：

- `benchmark.md`
- `benchmark_models.md`
- `benchmark_models_history.md`
- `benchamrk_kernels.md`
- `LlamaForCausalLM.md`
- `Qwen2ForCausalLM.md`
- `LlavaForConditionalGeneration.md`
- `LlavaNextForConditionalGeneration.md`
