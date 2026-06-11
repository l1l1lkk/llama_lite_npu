# Qwen3 MoE Host Sync and Decode Graph Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 清理Qwen3 MoE Decode热路径中的非必要Host同步，并以安全回退方式启用NPU Graph Capture/Replay。

**Architecture:** PagedAttention在Prefill缓存CPU请求ID，Decode不再读取NPU请求Tensor。NPU Graph不再按模型类型静态排除MoE，而是通过真实Capture确定算子兼容性；失败Key永久回退Eager，成功Key按Batch和序列Bucket复用。

**Tech Stack:** Python、PyTorch、torch_npu NPUGraph、Ascend GMM、Triton Ascend、unittest。

---

### Task 1: P3请求ID缓存

**Files:**
- Modify: `tests/test_decode_p0.py`
- Modify: `lite_llama/executor/model_executor.py`

- [ ] 添加失败测试，要求Paged Decode使用Prefill缓存的CPU请求ID，且不调用NPU Tensor `.tolist()`。
- [ ] 运行`python -m unittest tests.test_decode_p0 -v`确认测试因接口缺失失败。
- [ ] 在`prefill_alloc_kv_cache`缓存请求ID元组，在`decode_alloc_kv_cache`复用。
- [ ] 运行测试确认通过。

### Task 2: P4 MoE Graph能力与安全回退

**Files:**
- Modify: `tests/test_decode_p0.py`
- Modify: `lite_llama/executor/npu_graph.py`
- Modify: `lite_llama/executor/model_executor.py`

- [ ] 将“MoE被排除”的测试改为“MoE允许尝试Capture”。
- [ ] 添加失败Capture只尝试一次并持续Eager的测试。
- [ ] 运行测试确认新行为失败。
- [ ] 移除模型类型静态排除，保留按Graph Key的失败缓存。
- [ ] 为Graph Runner增加模型类型日志上下文，不改变Capture/Replay接口。
- [ ] 运行测试确认通过。

### Task 3: Atlas MoE Graph验证入口

**Files:**
- Modify: `tests/npu/test_qwen3_moe_gmm.py`
- Modify: `examples/benchmark_tp.py`

- [ ] 添加NPU测试，真实Capture一个固定Shape MoE层并验证两组路由输入Replay结果。
- [ ] 无NPU环境时明确Skip。
- [ ] Benchmark打印Graph capture/replay/fallback计数，便于确认并非静默Eager。
- [ ] 运行CPU可执行测试和静态编译。

### Task 4: v0.0.4rc1发布文档

**Files:**
- Modify: `VERSION`
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Create: `docs/releases/v0.0.4rc1.md`
- Modify: `docs/inference_performance_history.md`

- [ ] 将版本升级为`0.0.4rc1`。
- [ ] 记录P3、P4、Graph安全回退及Atlas验证命令。
- [ ] README只声明能力，不填写未实测性能。
- [ ] 运行`git diff --check`和全部相关测试。
