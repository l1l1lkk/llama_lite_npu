# Changelog

所有触发版本升级的变更按发布时间倒序记录。详细规则见[版本管理与发布规范](docs/versioning.md)。

## [0.0.2rc1] - 2026-06-10

新增Qwen3-30B-A3B MoE模型的正确性优先适配。

### 核心能力

- 新增`qwen3_moe`配置、模型注册和独立双卡CLI；
- 复用现有Qwen3 Attention、RoPE、FlashAttention、Flash Decoding和Paged KV链路；
- 支持128专家、TopK=8 Router以及按命中专家执行的SwiGLU专家MLP；
- 支持专家内部Tensor Parallel，Router复制，专家中间维切分并在输出端AllReduce；
- 权重转换器支持官方Qwen3-30B-A3B权重，并严格检查每层专家完整性；
- MoE首版显式关闭Decode NPU Graph，避免动态专家路径被错误Capture。

### 验证状态

- 配置、Router、专家计算、权重堆叠、TP切分和Graph降级单元测试通过；
- Windows CPU开发环境完成静态编译检查；
- Atlas 910B3双卡权重加载、端到端生成和性能数据需要在目标服务器继续验证。

### 文档

- [v0.0.2rc1完整版本报告](docs/releases/v0.0.2rc1.md)

## [0.0.1rc1] - 2026-06-09

首个带版本号的候选版本。

### 核心能力

- 支持Qwen3-32B在2 × Atlas 910B3上的FP16张量并行推理；
- 支持OpenAI兼容接口、真实SSE Streaming及EvalScope token usage统计；
- 支持Paged KV Cache、Flash Decoding和Triton Ascend融合算子；
- NPU Graph使用官方`NPUGraph`接口，并按128-token长度Bucket进行Capture/Replay；
- Paged KV Decode改为增量更新token映射，避免每个Token重建完整映射表；
- Graph与Eager模式的确定性输出验证通过。

### 性能

- Output Throughput：`24.0606 tok/s`
- TPOT：`40.6 ms`
- ITL：`40.9 ms`
- 相对Graph修复前的`5.696 tok/s`，吞吐提升约`322.4%`；
- 对比第三方vLLM-Ascend 0.8.4rc2结果，当前输出吞吐约为`3.15×`。

### 文档

- [v0.0.1rc1完整版本报告](docs/releases/v0.0.1rc1.md)

[0.0.1rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.1rc1
[0.0.2rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.2rc1
