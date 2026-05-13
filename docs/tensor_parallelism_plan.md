# Tensor Parallelism 多卡推理方案

## 核心思想：Megatron 风格的列切 + 行切

每层做矩阵乘法时把权重按列/行切开，分到不同 GPU 上并行算，最后通过通信原语合并结果。

---

## 1. 切分方案

### Attention 的 QKV 投影（列切）

```
Q = x @ W_q    
K = x @ W_k    
V = x @ W_v    
```

把 `W_q` 沿 head 维度切成 2 份，假设 2 卡：

```
GPU0:  Q_0 = x @ W_q_0    (heads 0-15)
GPU1:  Q_1 = x @ W_q_1    (heads 16-31)
```

- 输入 x 完全相同（从上一层 all_gather 得到）
- 每卡独立算自己的 head
- KV cache 各管各的 head，不重复

### Attention 输出投影 O（行切 + all_reduce）

```
GPU0:  out_0 = attn_output_0 @ W_o_0    (部分结果)
GPU1:  out_1 = attn_output_1 @ W_o_1

all_reduce(out_0 + out_1) → 完整输出
```

### FFN gate/up（列切）

```
GPU0:  h_0 = silu(x @ W_g_0) * (x @ W_u_0)
GPU1:  h_1 = silu(x @ W_g_1) * (x @ W_u_1)
```

每卡算一部分 intermediate 维度，输入 x 相同。

### FFN down（行切 + all_reduce）

```
GPU0:  out_0 = h_0 @ W_d_0
GPU1:  out_1 = h_1 @ W_d_1

all_reduce(out_0 + out_1)
```

### Embedding / lm_head（列切）

把词表沿 vocab 维度切开：

```
GPU0:  embed = lookup(W_embed_0, token_ids)
GPU1:  embed = lookup(W_embed_1, token_ids)
all_reduce(embed)  # 或 all_gather
```

lm_head 也同理切分，最后 all_gather 得到完整 logits。

---

## 2. 通信原语

| 操作 | 作用 | 出现位置 |
|---|---|---|
| `all_reduce` | 累加各卡部分结果，广播到所有卡 | O_proj 后、FFN down 后、Embedding |
| `all_gather` | 收集各卡数据拼接 | 每层输入（替代方案） |
| `reduce_scatter` | all_reduce 优化版，结果分散在各卡 | 可替代 all_reduce |

NPU 上使用 HCCL（对标 CUDA NCCL）：
```python
import torch_npu.distributed as dist
dist.all_reduce(tensor)           # 累加 + 广播
dist.all_gather(tensor_list, t)   # 收集拼接
dist.reduce_scatter(output, inputs) # 累加 + 分散
```

---

## 3. 在本项目中需要改动的文件

### ① `models/qwen3.py` — 权重切片 + 插入通信

`Qwen3Attention.__init__` 根据 world_size 切片：

```python
# 原本：所有权重在一张卡
self.q_proj_weight = (num_heads * head_dim, hidden_size)

# 切分后：每卡只存自己的 head
local_heads = num_heads // world_size
self.q_proj_weight = (local_heads * head_dim, hidden_size)
self.kv_proj_weight = (2 * local_kv_heads * head_dim, hidden_size)
self.o_proj_weight = (hidden_size, local_heads * head_dim)  # 行切
```

`Qwen3Attention.forward` 插入通信：

```python
output = F.linear(attn_output, self.o_proj_weight.data)
dist.all_reduce(output)  # ← 新增：O_proj 后 all_reduce
```

`FusedMLP.forward` 插入通信：

```python
return dist.all_reduce(self.down_proj(swiglu_forward(...)))  # ← FFN down 后
```

### ② `executor/model_executor.py` — 多卡初始化

```python
# 启动时
dist.init_process_group(backend='hccl')
local_rank = dist.get_rank()
torch.npu.set_device(local_rank)

# 加载权重时只加载当前 rank 的 slice
model = load_with_tensor_parallel(ckpt_path, world_size, rank)
```

### ③ `kernels/` — Attention 内核无需改动

FlashAttention 和 FlashDecoding 的 Triton kernel 本身不需要修改。每个 GPU 只看到自己的 head，计算逻辑完全一样。

唯一的调整：KV cache 按 head 维度分片，每卡只分配 `local_kv_heads` 份。

---

## 4. 实现步骤

| 阶段 | 内容 | 复杂度 |
|---|---|---|
| 1 | 权重自动切片（Q/K/V/O/gate/up/down/embed/lm_head） | 中 |
| 2 | 插入通信原语到 forward pass | 中 |
| 3 | KV cache 按 head 分片 | 小 |
| 4 | 多进程启动 + 权重加载（torchrun） | 中 |
| 5 | 测试验证精度对齐单卡 | 大 |

## 5. vLLM 额外做的事

- **Continuous Batching**：请求不会等整个 batch 完成，有请求结束立即插入新请求
- **PagedAttention**：KV cache 像操作系统分页一样管理，避免碎片
- **Prefix Caching**：相同 prefix 的 KV cache 复用
- **Quantization**：AWQ/GPTQ 量化减少显存

本项目目前是**静态批处理 + 连续 KV cache**，可以逐步演进。
