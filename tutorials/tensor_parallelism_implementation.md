# Tensor Parallelism 实现详解

本文以 lite_llama 项目的 TP 实现为例，从零讲解如何在推理框架中加入多卡并行。

---

## 目录

1. [为什么需要 TP](#1-为什么需要-tp)
2. [TP 的核心思想](#2-tp-的核心思想)
3. [列切 vs 行切](#3-列切-vs-行切)
4. [lite_llama 的实现](#4-lite_llama-的实现)
   - [4.1 tp_utils.py：基础设施](#41-tp_utilspy基础设施)
   - [4.2 qwen3.py：TP 感知的模型层](#42-qwen3py-tp-感知的模型层)
   - [4.3 model_executor.py：权重切片 + 加载](#43-model_executorpy权重切片--加载)
   - [4.4 生成同步：采样广播](#44-生成同步采样广播)
5. [使用方式](#5-使用方式)
6. [踩坑记录](#6-踩坑记录)

---

## 1. 为什么需要 TP

单卡显存有限。Qwen3-32B 的 fp16 权重约 64GB，910B3 单卡只有 60GB。把权重切到两张卡上，每卡只要约 32GB，就能跑起来了。

```
单卡:    [████████████████████████████████]  64GB → 装不下
TP 2卡:  [████████████] + [████████████]     各 32GB → 装得下
```

---

## 2. TP 的核心思想

Megatron 提出的方案：**把一层的权重矩阵切开，多卡同时算，最后通信合并**。

关键洞察：Triton kernel 不需要任何改动。FlashAttention、FlashDecoding、RMSNorm、SwiGLU——这些 kernel 每张卡各自算自己的那份，完全不知道 TP 的存在。只有权重形状和 forward 中的通信是 TP 特有的。

---

## 3. 列切 vs 行切

以 Qwen3 的 Q 投影为例，`W_q` 是 `(num_heads * head_dim, hidden_size)` = `(4096, 2560)` 的矩阵：

### 列切（Column Parallel）

把矩阵**沿输出维度**切成两半，各自算一半输出：

```
W_q 完整: (4096, 2560)
  GPU0: W_q[0:2048, :]  →  (2048, 2560)
  GPU1: W_q[2048:4096, :]  →  (2048, 2560)

输入 x 完全相同: (batch, 2560)
  GPU0: x @ W_q_0^T  →  (batch, 2048)   ← heads 0-15
  GPU1: x @ W_q_1^T  →  (batch, 2048)   ← heads 16-31
```

**适用**：Q 投影、K 投影、V 投影、FFN gate/up

### 行切（Row Parallel）

把矩阵**沿输入维度**切成两半，各算一部分，最后 **all_reduce** 累加：

```
W_o 完整: (2560, 4096)
  GPU0: W_o[:, 0:2048]  →  (2560, 2048)
  GPU1: W_o[:, 2048:4096]  →  (2560, 2048)

GPU0: attn_0 @ W_o_0^T  →  (batch, 2560)  ← 部分结果
GPU1: attn_1 @ W_o_1^T  →  (batch, 2560)  ← 部分结果

all_reduce(GPU0_out + GPU1_out)  →  (batch, 2560)  ← 完整结果
```

**适用**：O 投影、FFN down

### 口诀

```
列切：输入相同，输出各自一半
行切：输出形状相同，需要 all_reduce 求和
```

### Embedding 和 lm_head 的处理

```
Embedding: (vocab, hidden)
  全部复制。两个 rank 有完全相同的词表，因为输入 token 必须能在本卡查到

lm_head: (vocab, hidden)
  列切（沿 vocab 维度）。各算各的 vocab 部分，
  最后 all_gather 拼接成完整 logits
```

---

## 4. lite_llama 的实现

TP 实现分布在 5 个文件中。核心设计原则：**改动最小化，复用最大化**。

```
executor/tp_utils.py         # TP 基础设施（新增）
models/qwen3.py              # TP 感知的模型层（修改）
models/qwen3vl.py            # 透传 tp_config（修改）
executor/model_executor.py   # 权重切片 + 加载（修改）
cli_qwen3_tp.py / cli_qwen3vl_tp.py  # 入口（新增）
```

### 4.1 tp_utils.py：基础设施

```python
@dataclass
class TPConfig:
    world_size: int = 1    # 几张卡
    rank: int = 0          # 当前是第几张
    backend: str = "hccl"  # "hccl" (NPU) 或 "nccl" (CUDA)
```

**进程组初始化**：

```python
def init_tp(world_size, rank, backend="hccl") -> TPConfig:
    torch.distributed.init_process_group(
        backend=backend,
        world_size=world_size,
        rank=rank,
    )
    return TPConfig(world_size=world_size, rank=rank, backend=backend)
```

由 `torchrun --nproc_per_node=2` 自动设置 RANK/WORLD_SIZE 环境变量，`detect_tp_env()` 读取并初始化。

**权重切片函数**：

```python
def shard_attention_q(weight, tp):
    """沿 dim=0 切：每 rank 拿自己的 head"""
    return weight[_shard_slice(weight.shape[0], tp.world_size, tp.rank)].clone()

def shard_attention_o(weight, tp):
    """沿 dim=1 切：每 rank 拿输入的一部分列"""
    return weight[:, _shard_slice(weight.shape[1], ...)].clone()
```

**通信原语**：

```python
def tp_all_reduce(tensor):
    """跨 TP group 求和并广播"""
    torch.distributed.all_reduce(tensor, group=_TP_GROUP)
    return tensor

def tp_all_gather(tensor, dim=-1):
    """收集各 rank 的数据并拼接"""
    chunks = [torch.empty_like(tensor) for _ in range(world_size)]
    torch.distributed.all_gather(chunks, tensor, group=_TP_GROUP)
    return torch.cat(chunks, dim=dim)
```

### 4.2 qwen3.py：TP 感知的模型层

改了 5 个类，都在构造函数和 forward 中：

**Qwen3Attention**：

```python
class Qwen3Attention(nn.Module):
    def __init__(self, config, tp_config=None):
        self.tp = tp_config or TPConfig()
        # 每卡只算自己的 head
        self.num_heads = config.num_heads // self.tp.world_size   # 32 → 16
        self.num_kv_heads = config.num_kv_heads // self.tp.world_size  # 8 → 4

        # 权重自动变成 sharded 形状
        self.q_proj_weight = nn.Parameter(
            torch.rand(self.num_heads * self.head_dim, self.hidden_size))
        # 单卡: (4096, 2560) → TP2: (2048, 2560)

    def forward(self, x, ...):
        xq, xk, xv = self._get_qkv(x, ...)   # 各自算自己的 head
        attn_output = self.attn(...)           # 各自做 attention
        output = F.linear(attn_output, self.o_proj_weight.data)
        output = tp_all_reduce(output)         # ← 唯一新增：行切需要求和
        return output
```

**FusedMLP**：

```python
class FusedMLP(nn.Module):
    def __init__(self, config, tp_config=None):
        # intermediate_size 也切
        self.intermediate_size = config.intermediate_size // tp.world_size

        # gate/up: 列切
        self.gate_proj = nn.Linear(hidden, intermediate // tp, ...)
        self.up_proj = nn.Linear(hidden, intermediate // tp, ...)
        # down: 行切
        self.down_proj = nn.Linear(intermediate // tp, hidden, ...)

    def forward(self, x):
        h = swiglu_forward(self.gate_proj(x), self.up_proj(x))
        out = self.down_proj(h)
        out = tp_all_reduce(out)   # ← 行切求和
        return out
```

**Qwen3Model**：

```python
class Qwen3Model(nn.Module):
    def __init__(self, config, tp_config=None):
        # Embedding: 复制（所有 rank 有完整词表）
        self.embed_tokens = nn.Embedding(config.vocab_size, hidden)

        # lm_head: 列切（vocab 维度）
        vocab_local = config.vocab_size // tp.world_size
        self.lm_head_weight = nn.Parameter(torch.rand(vocab_local, hidden))

    def forward(self, input_ids, ...):
        h = self.embed_tokens(input_ids)         # 复制，各算各的
        for layer in self.layers:
            h, residual = layer(...)              # 每层内有 all_reduce

        output = F.linear(h, self.lm_head_weight.data)
        output = tp_all_gather(output, dim=-1)    # ← 收集拼接完整 logits
        return output
```

### 4.3 model_executor.py：权重切片 + 加载

权重转换产出**全量** .pth 文件。TP 加载时，每 rank 读同一个文件，各取自己的切片：

```python
def _load_model_weight(model_config, checkpoints_dir, device, tp_config):
    # 1. 初始化模型（TP 感知，权重形状已经是 sharded 的）
    with init_empty_weights():
        model = Qwen3Model(config, tp_config=tp_config)

    # 2. 加载全量权重到 CPU（mmap 大文件，不占满 RAM）
    state_dict = torch.load(ckpt_path, mmap=True, map_location="cpu")

    # 3. 切片（CPU 上操作）
    if tp_config.enabled:
        state_dict = _shard_state_dict(state_dict, num_layers, tp_config, model_config)

    # 4. 加载切片后的权重到 NPU
    model.to(device).half()
    model.load_state_dict(state_dict, strict=True, assign=True)
```

**切片逻辑 `_shard_state_dict`**：

```python
for i in range(num_layers):
    p = f"layers.{i}.self_attn"
    state_dict[f"{p}.q_proj_weight"] = shard_attention_q(
        state_dict[f"{p}.q_proj_weight"], tp)       # 列切
    state_dict[f"{p}.kv_proj_weight"] = shard_attention_kv(
        state_dict[f"{p}.kv_proj_weight"], ...)     # 列切
    state_dict[f"{p}.o_proj_weight"] = shard_attention_o(
        state_dict[f"{p}.o_proj_weight"], tp)       # 行切
    # FFN gate/up: 列切, down: 行切
    ...

state_dict["lm_head_weight"] = shard_lm_head(...)   # 列切(vocab)
```

### 4.4 生成同步：采样广播

这是一个关键的踩坑点。TP 下所有 rank 同时跑 forward（靠 all_reduce 同步），但 forward 之后的**采样步骤是独立的**。

问题场景：
```
Prefill 完成 → logits 正确（all_gather 保证一样）
Rank 0: multinomial 抽出 "你好"
Rank 1: multinomial 抽出 "？"    ← NPU RNG 不同！

下一轮：
Rank 0: embed("你好") + forward
Rank 1: embed("？") + forward
→ all_reduce 把两个不同 token 的计算结果求和 → 垃圾
```

解决：采样后立即广播，确保所有 rank 用同一个 token。

```python
# generate_stream.py
next_token = sample_top_p(probs, top_p)
if torch.distributed.is_initialized():
    torch.distributed.broadcast(next_token, src=0)  # ← 关键：强制同步
```

同时设置相同随机种子兜底：

```python
torch.manual_seed(42)
torch.npu.manual_seed(42)
```

---

## 5. 使用方式

```bash
# 纯文本 Qwen3
ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
    --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \
    cli_qwen3_tp.py --checkpoints_dir my_weight/Qwen3-32B/

# VL 模型
ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
    --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \
    cli_qwen3vl_tp.py --checkpoints_dir my_weight/Qwen3-VL-32B-Instruct/

# Benchmark
ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
    --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \
    examples/benchmark_tp.py --checkpoints_dir my_weight/Qwen3-32B/ \
    --batch_size 4 --max_gen_len 256 --iterations 5
```

权重不需重新转换——同一个 .pth 文件，每 rank 自动切片。

---

## 6. 踩坑记录

| 问题 | 现象 | 原因 | 修复 |
|---|---|---|---|
| **采样不同步** | 输出第一个 token 就错，后面全崩 | 两 rank 各抽各的 token | 采样后 broadcast |
| **缺少 ChatML 模板** | 纯文本模型输出乱码 | Qwen3-instruct 需要 `<\|im_start\|>` 格式 | 加 `get_prompter` |
| **权重加载 OOM** | 32B 模型 load_state_dict OOM | 全量加载到 NPU | 先加载到 CPU，切片后再搬 NPU |
| **HCCL broadcast CPU tensor** | `No backend type for cpu` | HCCL 只支持 NPU tensor | 改用 `broadcast_object_list` |
| **model.to vs load_state_dict 顺序** | `Cannot copy out of meta tensor` | meta tensor 还没填充就调 to(device) | 先 load_state_dict，再 to(device) |
| **KV cache 分配使用全量 heads** | KV cache OOM 或形状不对 | `num_kv_heads` 未除以 world_size | 用 `local_kv_heads` |

### 扩展指南

目前只支持 Qwen3/Qwen3-VL。要加其他模型，按以下步骤：

1. **模型 `__init__`**：接受 `tp_config`，`num_heads`/`num_kv_heads`/`intermediate_size` 除以 `world_size`
2. **模型 forward**：O_proj 和 FFN down 之后加 `tp_all_reduce`，lm_head 之后加 `tp_all_gather`
3. **model_executor._initialize_model**：传 `tp_config`
4. **model_executor._shard_state_dict**：加该模型的权重 key 映射
