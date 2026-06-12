# Qwen3 MoE GMM Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用 Ascend Grouped MatMul 和 Triton 设备侧路由替换 Qwen3 MoE 专家 Python 循环，并提供逐层 eager 数值对齐。

**Architecture:** Router 输出保持不变；Triton 将 token-expert assignment 按专家整理成连续输入，两个 `torch_npu.npu_grouped_matmul` 完成 Gate/Up 和 Down，Triton 再加权 Scatter。eager 路径保留为参考与回退，验证模式在 TP AllReduce 前逐层比较本地输出。

**Tech Stack:** PyTorch 2.7、torch_npu 2.7.1、Triton Ascend 3.2、unittest。

---

### Task 1: 路由参考语义与失败测试

**Files:**
- Create: `lite_llama/kernels/moe_routing.py`
- Modify: `tests/models/test_qwen3_moe.py`

- [ ] 添加测试，验证按专家重排后的 token id、routing weight、
  expert counts、Gather 数据和加权 Scatter 与直接 `index_add_` 一致。
- [ ] 运行目标测试并确认因API不存在而失败。
- [ ] 实现纯 PyTorch reference helper，使CPU测试通过。

### Task 2: GMM 后端契约

**Files:**
- Modify: `lite_llama/models/moe.py`
- Modify: `tests/models/test_qwen3_moe.py`

- [ ] 添加假 `torch_npu` GMM 测试，验证 Gate/Up 与 Down 各调用一次、
  `group_list` 为累计专家任务数、权重布局为 `[E,K,N]`。
- [ ] 运行目标测试并确认失败。
- [ ] 提取 `_forward_eager_local`、`_forward_grouped_local` 和
  `_npu_grouped_matmul`，实现最小兼容调用。

### Task 3: Triton count/gather/scatter

**Files:**
- Modify: `lite_llama/kernels/moe_routing.py`
- Modify: `lite_llama/kernels/__init__.py`
- Modify: `tests/models/test_qwen3_moe.py`

- [ ] 添加AST/接口测试，要求 Triton count、gather 和 scatter kernel
  存在，且NPU入口不调用 `.tolist()`。
- [ ] 运行目标测试并确认失败。
- [ ] 实现专家计数、前缀和、按专家分配位置并Gather、加权Scatter。
- [ ] 运行CPU测试和静态编译。

### Task 4: 自动后端和逐层数值验证

**Files:**
- Modify: `lite_llama/models/moe.py`
- Modify: `tests/models/test_qwen3_moe.py`
- Create: `tests/npu/test_qwen3_moe_gmm.py`

- [ ] 添加 `auto/eager/gmm` 选择及无GMM接口回退测试。
- [ ] 添加验证模式成功与误差超限测试。
- [ ] 实现环境变量控制和本地输出对齐，再执行一次TP AllReduce。
- [ ] 编写910B3真实NPU对齐测试。

### Task 5: 版本与验证

**Files:**
- Modify: `VERSION`
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Create: `docs/releases/v0.0.3rc1.md`

- [ ] 更新版本为 `0.0.3rc1`，记录GMM、Triton routing和已知限制。
- [ ] 运行 MoE 单元测试、全量可运行测试、`py_compile` 和
  `git diff --check`。
- [ ] 输出910B3验证命令；不在本地伪造NPU通过结论。
