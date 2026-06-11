# Continuous Batching、Decode小Batch专家内核与Expert Parallel设计

## 目标

`v0.0.5rc1`在保持现有Qwen3/Qwen3-MoE模型接口的前提下增加：

1. 文本Server Continuous Batching；
2. Qwen3 MoE Decode小Batch专用专家内核；
3. 单机多卡Qwen3 MoE Expert Parallel。

视觉模型、跨节点EP、Chunked Prefill和抢占调度不在首版范围内。

## P7：Continuous Batching

Server使用单一后台调度线程拥有Generator与KV Cache。HTTP处理函数只负责创建请求、
入队和消费独立输出队列，避免多个请求线程并发修改执行器状态。

每个调度周期允许新请求加入已有Decode集合：

```text
等待请求 -> 按Prompt长度分组Prefill -> 加入活动集合
        -> 对已有活动请求执行一步Decode
        -> 独立结束并释放Paged KV -> 下一轮继续接纳请求
```

不同Prompt长度首版按长度分组，避免Padding改变No-padding Attention语义。Decode批次通过
每个请求独立的`request_id`、`seq_len`和Paged KV页表访问各自上下文。

TP rank 0广播`prefill/decode/release`步骤命令，其他Rank执行完全相同的模型步骤。

## P8：Decode小Batch专家内核

保留GMM作为Prefill和较大路由批次后端。新增`routed_gemv`用于小Batch Decode：

```text
token/top-k assignment
 -> 直接读取目标专家Gate/Up
 -> 融合SwiGLU
 -> Down投影
 -> routing weight
 -> 固定TopK归约
```

该路径不执行通用`argsort/bincount/gather/scatter`。默认当
`tokens * top_k <= 64`时自动启用，可通过
`LITE_LLAMA_MOE_GEMV_MAX_ASSIGNMENTS`调整。`LITE_LLAMA_MOE_BACKEND`支持
`auto/eager/gmm/routed_gemv`。

## P9：Expert Parallel

Attention继续使用现有TP，因此每层进入MoE前的token hidden states在各Rank上相同。
在这个拓扑下无需先做All-to-All分发token；每个Rank只持有并计算自己的完整专家：

```text
rank 0: experts 0..63
rank 1: experts 64..127

replicated hidden states
 -> each rank computes locally-owned expert contributions
 -> one HCCL AllReduce sums partial token outputs
```

这种实现将专家权重内存和有效专家计算按Rank切分，同时复用现有MoE层末尾的AllReduce。
相比在相同TP组内增加All-to-All，它减少了通信阶段和调度元数据。

权重文件格式不变。加载时：

- `tp`模式沿专家中间维切分每个专家；
- `ep`模式沿expert维切分完整Gate/Up和Down权重；
- Router在每卡复制；
- 启动参数为`--moe_parallel_mode tp|ep`，默认`tp`保持兼容。

小BatchEP先在NPU上压缩本地assignment，再只启动本地专家程序；较大Prefill批次按本地
专家过滤后继续使用GMM。动态压缩会改变中间Tensor Shape，因此EP模式的NPU Graph可能
安全回退Eager；`tp`模式仍保持固定Shape Routed-GEMV Graph路径。

## 正确性与回退

- Continuous Batching覆盖动态加入、容量限制和请求独立退出；
- Routed-GEMV与Eager逐层数值对齐；
- EP验证两Rank局部专家输出之和等于完整专家参考输出；
- Graph Capture失败继续使用现有按Key缓存的Eager安全回退；
- Atlas上的Triton、GMM、HCCL与Graph兼容性必须通过NPU测试确认。

## 版本

本次新增调度、MoE Decode内核和并行策略，版本升级为`0.0.5rc1`。
