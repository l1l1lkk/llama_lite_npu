# Qwen3 MoE Decode Host同步清理与NPU Graph设计

## 目标

在保持Qwen3-30B-A3B现有GMM与Triton路由数值路径不变的前提下：

1. 删除Decode热路径中可避免的NPU到Host同步。
2. 为MoE Decode增加NPU Graph Capture/Replay能力。
3. Graph不兼容时只失败一次并稳定回退Eager，禁止错误Replay。

## P3：Decode Host同步

PagedAttention的请求页表由CPU管理，因此请求ID本身保留为CPU元数据。Prefill时将
`b_req_idx`转换并缓存为Python整数元组，后续Decode直接复用，避免每个Token调用
NPU Tensor的`.tolist()`。

流式文本输出必须把采样Token交给CPU tokenizer，不能完全消除D2H同步。本版本不改变
公开流式语义，只清理KV管理和模型前向内部的非必要同步；Benchmark中的显式
`torch.npu.synchronize()`仅用于准确计时，不属于待删除范围。

## P4：MoE Decode NPU Graph

MoE执行拓扑保持固定：

```text
Router -> TopK -> Count -> Argsort -> Cumsum
       -> Gather -> GMM Gate/Up -> SwiGLU
       -> GMM Down -> Scatter -> TP AllReduce
```

专家ID和`group_list`的数值允许每步变化，但Batch、TopK、专家数和所有Buffer Shape固定。
Graph Runner继续按`(batch_size, 128-token sequence bucket)`缓存图，并复制动态输入。

MoE Graph采用显式开关和能力隔离：

- `qwen3_moe`允许创建Graph Runner。
- 首次Capture尝试真实MoE前向。
- 任意算子不支持Capture时，将对应`(batch, bucket)`加入失败集合。
- 后续相同Key直接Eager，不重复Capture。
- Capture成功后必须Replay；模型输出不允许在Host侧参与控制流。

现有路由函数中的临时Tensor仍由PyTorch/Triton创建。是否能被NPUGraph内存池稳定管理
由服务器真实Capture验证；本版本不自行实现第二套静态内存分配器。

## 正确性与回退

- CPU测试验证MoE不再被静态排除。
- 测试验证Capture失败只发生一次。
- 测试验证不同动态路由输入在同一Shape下进入同一Graph Key。
- Atlas测试必须比较Graph与Eager的Greedy输出，并覆盖跨128/256 Bucket边界。
- 如果GMM动态`group_list`或Triton算子不支持Replay，保留安全Eager路径，并记录失败原因。

## 版本

该变更新增MoE Decode Graph能力并优化Decode执行，版本升级为`0.0.4rc1`。
