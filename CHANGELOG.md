# Changelog

所有触发版本升级的变更按发布时间倒序记录。详细规则见[版本管理与发布规范](docs/versioning.md)。

## [0.0.3rc1] - 2026-06-11

将Qwen3-30B-A3B MoE专家执行从动态Python循环升级为Ascend Grouped MatMul与Triton设备侧路由。

### 核心能力

- Gate/Up与Down投影分别使用`torch_npu.npu_grouped_matmul`；
- Triton在NPU侧完成专家计数、按专家Gather和routing weight加权Scatter；
- 移除GMM热路径中的`torch.unique(...).tolist()`及逐专家Python循环；
- 模型加载时将现有`.pth`专家权重一次性转换为GMM原生`[expert, input, output]`布局，无需重新转换权重；
- 保留`eager`参考后端，并支持每个Sparse MoE层在TP AllReduce前进行数值对齐。

### 验证状态

- 17项Qwen3 MoE CPU单元与契约测试通过；
- 新增2项Atlas NPU真实GMM数值测试，本地无NPU环境时明确跳过；
- Python静态编译通过；
- Atlas 910B3端到端数值与性能结果需在目标服务器完成后写入，不在本版本文档中填写预测数据。

### 文档

- [v0.0.3rc1完整版本报告](docs/releases/v0.0.3rc1.md)

## [0.0.2rc2] - 2026-06-10

修复Qwen3-30B-A3B MoE在Prefill阶段因SwiGLU错误读取非连续Gate/Up视图而产生无关回答的问题。

### Bug修复

- SwiGLU Triton内核分别接收Gate、Up和输出张量的行跨度；
- 修复融合Gate/Up经过`chunk()`后输入stride大于输出stride时的错误寻址；
- Dense MLP连续张量路径保持兼容。

### 验证状态

- Qwen3 MoE单元测试由9项增加到10项并全部通过；
- Python静态编译和`git diff --check`通过；
- Atlas 910B3端到端回答正确性等待目标服务器验证。

### 文档

- [v0.0.2rc2完整版本报告](docs/releases/v0.0.2rc2.md)

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
[0.0.2rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.2rc2
[0.0.3rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.3rc1
