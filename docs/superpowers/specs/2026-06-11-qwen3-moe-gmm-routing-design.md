# Qwen3 MoE GMM 与设备侧路由设计

## 目标

将 Qwen3 MoE 当前按专家执行的 Python eager 循环替换为 Ascend
`torch_npu.npu_grouped_matmul`，并用 Triton 完成 token-expert
任务的设备侧重排、Gather 和加权 Scatter。保留 eager 实现作为
CPU参考、显式调试后端和 GMM 不可用时的安全回退。

## 执行路径

默认 `auto` 后端：

1. Router 继续执行 FP32 softmax 和 TopK，不改变专家选择语义。
2. 将 `[tokens, top_k]` 路由展开为 `tokens * top_k` 条任务。
3. Triton 统计每个专家任务数，并按专家分区 Gather hidden states。
4. 使用累计 `group_list` 调用 Gate/Up Grouped MatMul。
5. 继续复用现有 Triton SwiGLU。
6. 调用 Down Grouped MatMul。
7. Triton 将 routing weight 乘入专家输出，并按原 token 原子累加。
8. TP 模式保持每层一次 HCCL AllReduce。

CPU、非 NPU 设备、GMM 接口不可用或显式设置
`LITE_LLAMA_MOE_BACKEND=eager` 时使用现有 eager 路径。

## 数值对齐

设置 `LITE_LLAMA_MOE_VALIDATE=1` 时，每个 MoE 层同时计算：

- eager 本地专家输出；
- GMM 本地专家输出。

在 TP AllReduce 之前比较两者，默认使用 `rtol=1e-2`、
`atol=1e-2`。不一致时抛出包含层内最大绝对误差、平均绝对误差
和张量形状的错误。Router 只执行一次，因此 TopK 专家和权重天然
共享；测试额外验证路由任务映射、专家计数、Gather 和 Scatter。

## 组件边界

- `lite_llama/kernels/moe_routing.py`
  - PyTorch reference routing，供 CPU 单元测试使用。
  - Triton NPU routing，负责 count/sort/gather/scatter。
- `lite_llama/models/moe.py`
  - eager 专家参考实现。
  - `torch_npu.npu_grouped_matmul` 兼容调用。
  - backend 选择、回退和逐层数值验证。
- `tests/models/test_qwen3_moe.py`
  - CPU reference、后端选择、GMM调用契约和数值对齐测试。
- `tests/npu/test_qwen3_moe_gmm.py`
  - 910B3 上 eager/GMM 的真实逐层数值测试。

## 约束

- 不改变转换后权重文件格式；GMM 调用时使用权重转置视图。
- 不引入 Expert Parallel。
- 本版本不启用 MoE NPU Graph；先稳定固定拓扑的 GMM eager 路径。
- GMM 运行时错误不静默回退，避免隐藏数值或接口问题；只有接口
  不存在时 `auto` 后端才回退 eager。
