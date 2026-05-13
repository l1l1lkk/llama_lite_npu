# LLM 推理全流程：从输入到输出

本文以 lite_llama 项目为例，追踪一次完整的模型推理过程，帮助你从零理解 LLM 是怎么一步步算出结果的。

---

## 目录

1. [概览：一次推理经历了什么](#1-概览一次推理经历了什么)
2. [Tokenizer：把文字变成数字](#2-tokenizer把文字变成数字)
3. [Embedding：数字变成向量](#3-embedding数字变成向量)
4. [位置编码 RoPE：告诉模型顺序](#4-位置编码-rope告诉模型顺序)
5. [Transformer 层：核心计算](#5-transformer-层核心计算)
   - [5.1 SkipRMSNorm：残差 + 归一化](#51-skiprmsnorm残差--归一化)
   - [5.2 Attention：信息检索](#52-attention信息检索)
   - [5.3 SwiGLU FFN：知识存储](#53-swiglu-ffn知识存储)
6. [lm_head + 采样：选出下一个 token](#6-lm_head--采样选出下一个-token)
7. [Prefill vs Decode：两种计算模式](#7-prefill-vs-decode两种计算模式)
8. [优化技术清单](#8-优化技术清单)

---

## 1. 概览：一次推理经历了什么

用户输入 `"什么是大语言模型"`，模型逐字输出回答。整个流程分为两大阶段：

```
┌─────────────────────────────────────────────────────────┐
│ Prefill（预填充）：一次性处理整个 prompt                  │
│                                                         │
│ Tokenize → Embedding → RoPE → 36层 Transformer          │
│   → logits → 采样得到第1个 token                        │
│                                                         │
│ 特点：并行计算全部 prompt token，耗时但只需要一次          │
├─────────────────────────────────────────────────────────┤
│ Decode（解码）：逐个 token 生成                          │
│                                                         │
│ 第1个 token → Embedding → RoPE → 36层 Transformer       │
│   → logits → 采样得到第2个 token → ...循环直到 EOS       │
│                                                         │
│ 特点：每次只处理1个 token，但需要读完整 KV cache          │
└─────────────────────────────────────────────────────────┘
```

> **关键洞察**：Prefill 和 Decode 的根本区别在于 `seq_len`。
> - Prefill: `seq_len = 所有 prompt token 数`（如 25），`Q @ K^T` 产生 25×25 的矩阵
> - Decode: `seq_len = 1`，`Q @ K^T` 产生 1×N 的矩阵（N = 已缓存的 KV 长度）

---

## 2. Tokenizer：把文字变成数字

**代码位置**：`lite_llama/generate.py` 第 202 行

```python
input_ids = self.tokenizer(prompts, return_tensors="pt", padding=True).input_ids
```

**做了什么**：
```
"什么是大语言模型" → [1, 74892, 104198, 106252, 1773, 100638, 103939, 109107]
                                                     ↑
                                            每个数字是一个 token ID
```

**Padding**：batch 中不同长度的 prompt 补 pad_token（如 0）到相同长度，用一个 `input_text_mask` 标记哪些位置是真实的。

> **学习点**：token 不一定是完整的汉字，也可能是一个词的一部分。BPE/WordPiece 等算法把常用词组合成单个 token，生僻词拆成多个 token。

---

## 3. Embedding：数字变成向量

**代码位置**：`lite_llama/models/qwen3.py` 第 296 行

```python
h = self.get_input_embeddings(input_ids)  # h.shape: (batch, seq_len, hidden_size)
```

**做了什么**：

```
输入: [1, 74892, 104198, ...]    shape: (1, 8)     每个是整数
输出:                            shape: (1, 8, 2560)  每个变成 2560 维向量

Embedding 表: (vocab_size=152000, hidden_size=2560)
查表操作:   token_id=74892 → embedding[74892] → 一个 2560 维的向量
```

**直观理解**：每个 token 被映射到一个 2560 维空间的点。语义相近的词在这个空间里距离也近。

---

## 4. 位置编码 RoPE：告诉模型顺序

**代码位置**：`lite_llama/models/RotaryEmbedding.py`

**为什么需要**：Attention 机制本身**没有顺序概念**。"狗咬人" 和 "人咬狗" 的 token 集合完全一样，不加位置信息模型根本分不出来。

### 4.1 标准 RoPE 原理

RoPE（Rotary Position Embedding）的核心思想：**用旋转矩阵给 Q 和 K 编码位置信息**。

```
位置 m 的 token，其 Q 向量的第 i 对维度旋转 m * θ_i 角度。
位置 n 的 token，其 K 向量的第 i 对维度旋转 n * θ_i 角度。

Q_m · K_n = (旋转后的 Q_m) · (旋转后的 K_n)
          = 原始 Q_m · R(m - n) · 原始 K_n     ← R(m-n)是旋转矩阵
```

关键性质：**内积只依赖于相对位置 m - n**，模型能感知 token 之间的相对距离。

### 4.2 计算步骤

```python
# 1. 计算频率 inv_freq
inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2) / dim))
# 例：dim=128, base=10000, inv_freq = [1/1, 1/1.156, 1/1.337, ...]

# 2. 计算旋转角度
freqs = position_ids @ inv_freq  # (seq_len, dim/2)

# 3. 应用旋转: 每一对维度 [x_{2i}, x_{2i+1}] 旋转 θ
cos, sin = cos(freqs), sin(freqs)
q_rotated = q * cos + rotate_half(q) * sin
k_rotated = k * cos + rotate_half(k) * sin
```

**代码位置**：Triton kernel 实现 `lite_llama/kernels/rope_emb.py`

### 4.3 M-RoPE（Qwen3-VL 专用）

Qwen3-VL 使用 3D 位置编码，因为有**三种类型**的 token：
- **文本 token**：1D 递增位置
- **图像 token**：2D (行, 列) 空间位置
- **视频 token**：3D (时间, 行, 列) 时空位置

```python
position_ids: (3, batch, seq_len)
# dim 0: temporal (时间)
# dim 1: height   (行)  
# dim 2: width    (列)
```

三种维度各自计算 RoPE 频率，然后按照 `[THWTHWTHW...]` 模式交织。

> **代码位置**：`lite_llama/models/RotaryEmbedding.py` Qwen3VLTextRotaryEmbedding 类

---

## 5. Transformer 层：核心计算

模型把输入重复经过 36 层相同的 Transformer block，每层逐渐精炼表示。

**代码位置**：`lite_llama/models/qwen3.py` 第 306 行

```python
for i, layer in enumerate(self.layers):
    h, residual = layer(h, atten_info, i, position_embeddings, qk_scale, residual)
```

每层内部有两个子层：

```
输入 h (batch, seq_len, hidden_size)
    │
    ├─→ SkipRMSNorm ──→ Attention (FlashAttention/FlashDecoding)
    │       │
    │       └─→ 残差连接 ──→ h = h + attention_out
    │
    ├─→ SkipRMSNorm ──→ SwiGLU FFN
    │       │
    │       └─→ 残差连接 ──→ h = h + ffn_out
    │
    输出 h (batch, seq_len, hidden_size)
```

### 5.1 SkipRMSNorm：残差 + 归一化

**代码位置**：`lite_llama/kernels/skip_rmsnorm.py`

这是一个**融合算子**，把两个操作合并为一个 Triton kernel，减少显存读写：

```
传统做法（两次显存读写）：
  step1: residual = x + residual     (写回显存)
  step2: y = RMSNorm(residual)        (再读显存)

融合做法（一次显存读写）：
  y, residual = skip_rmsnorm(x, residual, weight)
```

**RMSNorm 计算**：
```
RMS(x) = x / sqrt(mean(x²) + ε) * weight
```

把输入归一化到均方根为 1，避免数值爆炸/消失，让深层网络能稳定训练和推理。

### 5.2 Attention：信息检索

**代码位置**：`lite_llama/models/qwen3.py` Attention 类

这是模型最核心的计算。你可以这样直观理解：

> **类比**：你在一段文章里找"什么地方跟我现在想的内容最相关"。
> - Q（Query）：你现在在想什么
> - K（Key）：每段内容的标签
> - V（Value）：每段内容本身
> - Attention 分数 = "想的内容"跟"标签"的匹配程度
> - 最终输出 = 各段内容按匹配程度加权平均

#### 5.2.1 计算步骤

```
输入: x (batch=4, seq_len=25, hidden_size=2560)

# 步骤 1: QKV 投影
Q = x @ W_q     # (4, 25, 32*128=4096)
K = x @ W_k     # (4, 25, 8*128=1024)   ← GQA: KV head 比 Q head 少
V = x @ W_v     # (4, 25, 8*128=1024)

# reshape: (4, 25, num_heads, head_dim)
Q: (4, 25, 32, 128)
K: (4, 25, 8, 128)
V: (4, 25, 8, 128)

# 步骤 2: RoPE (给 K 加上位置信息)
Q, K = rope(Q, K)

# 步骤 3: 计算 attention 分数
scores = Q @ K^T / sqrt(128)       # (4, 32, 25, 25)
#          ↑ 除以 sqrt(d) 防止梯度爆炸

# 步骤 4: Causal mask (不能偷看未来)
scores[i, j] = -∞  for i < j       # 上三角遮掉

# 步骤 5: Softmax 归一化
weights = softmax(scores)           # 每行概率和为 1

# 步骤 6: 加权求和
output = weights @ V                # (4, 32, 25, 128)

# 步骤 7: 输出投影
output = output.view(4, 25, 4096) @ W_o   # (4, 25, 2560)
```

#### 5.2.2 GQA（Grouped Query Attention）

**代码中体现**：`num_q_heads=32, num_kv_heads=8`

```
32 个 Q head 对应 8 个 KV head
每 4 个 Q head 共享 1 个 KV head pair

为什么这样做：
- MHA: 大量 KV cache 显存，读带宽瓶颈
- GQA: KV cache 减少 4 倍，decode 速度提升
- 精度损失几乎为零
```

代码中通过 `num_kv_groups` 实现：`kv_head_idx = q_head_idx // num_kv_groups`

#### 5.2.3 FlashAttention：IO 优化的注意力计算

**代码位置**：`lite_llama/kernels/flashattention2_nopad.py`

**为什么需要**：标准 Attention 的 Q@K^T 产生 (seq_len, seq_len) 的中间矩阵，需要写回 HBM（显存）再读出来，当 seq_len=2048 时这个矩阵是 (2048, 2048)，非常占带宽。

**FlashAttention 的做法**：
- 把 Q, K, V 分块（tile）
- 每块在 SRAM（片上缓存）里完成 QK^T → softmax → O 的计算
- 使用 **online softmax** 算法在分块过程中逐步更新归一项
- **结果**：中间矩阵从不写回 HBM，带宽节省 3-4 倍

```
传统 Attention:
HBM ← Q,K,V → SRAM → 完整 QK^T 写回 HBM → 读回 softmax → 写回 O

FlashAttention:
Q,K,V 分块读入 SRAM → 每次更新 O 累加器 → 最终 O 直接写回 HBM
（QK^T 中间矩阵从未离开 SRAM）
```

#### 5.2.4 FlashDecoding：Decode 阶段的特殊优化

**代码位置**：`lite_llama/kernels/flashdecoding.py`

Decode 阶段 `seq_len_q = 1`，但 `seq_len_kv` 可能长达数万。直接用 FlashAttention 的话 GPU 利用率低（只有一个 Q token，并行度不够）。

**FlashDecoding 的做法**：把 KV cache 切成多个 partition，每个 partition 独立计算部分 attention，最后再 reduce。

```
Stage 1 (分区计算):
  KV_partition_1  vs Q → mid_o[0], mid_logsumexp[0]
  KV_partition_2  vs Q → mid_o[1], mid_logsumexp[1]
  KV_partition_3  vs Q → mid_o[2], mid_logsumexp[2]
  ... 这些可以并行计算

Stage 2 (归约):
  对 mid_o 用 online softmax 归约 → 最终输出
```

#### 5.2.5 KV Cache

**代码位置**：`lite_llama/executor/mem_manager.py` KVCacheMemoryManager

**为什么需要**：Decode 阶段每次只输入 1 个 token，但 attention 需要所有历史 token 的 K 和 V。如果每次都把全部历史 token 重新算一遍 K 和 V，计算量会随生成长度平方增长。

**KV Cache 机制**：
```
Prefill: 算所有 prompt token 的 K,V → 存入 buffer
Decode step 1: 只算新 token 的 K,V → 追加到 buffer → attention 用 buffer 里的全部 K,V
Decode step 2: 只算新 token 的 K,V → 追加到 buffer → attention 用 buffer 里的全部 K,V
...
```

**内存管理**（本项目的特色优化）：
- 预分配 `(max_tokens, 2*kv_heads, head_dim)` 的大 buffer（每层一个）
- 引用计数管理：分配时 `add_ref`，释放时 `release_ref`
- 支持非连续分配：不同请求的 KV cache 可以散布在 buffer 中
- 通过 `b_req_tokens_table` 索引表找到每个请求的 KV cache 位置

### 5.3 SwiGLU FFN：知识存储

**代码位置**：`lite_llama/models/qwen3.py` FusedMLP 类

```python
h = swiglu_forward(gate_proj(x), up_proj(x))  # SiLU(gate) * up
out = down_proj(h)
```

**三层网络的作用**：
```
                ┌─→ gate_proj → SiLU ─┐
x (2560) ──────┤                      ├─→ × ─→ down_proj → (2560)
                └─→ up_proj ──────────┘
```

- **gate_proj + up_proj**：把输入从 2560 维扩展到 10240 维（intermediate_size），用门控机制激活
- **SwiGLU**（Sigmoid Linear Unit with Gating）：`SiLU(x) * y`，门控决定哪些信息通过
- **down_proj**：压缩回 2560 维

> **学习点**：Attention 负责"从上下文中检索信息"，FFN 负责"存储知识"。模型学到的事实、概念主要存储在 FFN 的权重中。

**算子融合**：`swiglu_forward` 把 SiLU(gate) × up 合并为一个 Triton kernel，避免中间变量写回显存。

---

## 6. lm_head + 采样：选出下一个 token

### 6.1 lm_head 投影

**代码位置**：`lite_llama/models/qwen3.py` 第 313 行

```python
h, _ = skip_rmsnorm(h, residual, self.norm_weight.data, self.rmsnorm_eps)
output = F.linear(h, self.lm_head_weight.data)
```

最后一步：把 2560 维的向量投影到 152000 维（vocab_size），得到每个 token 的"分数"（logits）。

### 6.2 采样算法

**代码位置**：`lite_llama/generate.py` `sample_top_p`

```python
# 1. Temperature 缩放
probs = softmax(logits / temperature)    # T高→更均匀（随机），T低→更集中（确定）

# 2. Top-p (Nucleus) 采样
probs_sort, probs_idx = sort(probs, descending=True)
cumsum_probs = cumsum(probs_sort)        # 累积概率
mask = cumsum_probs - probs_sort > p     # 累积超 p 的截断
probs_sort[mask] = 0.0
probs_sort = probs_sort / sum(probs_sort)  # 重归一化

# 3. 按概率采样
next_token = multinomial(probs_sort)
```

**Top-p 采样直观理解**：不取概率最大的单个 token，而是在"概率累加到 p 的最小 token 集合"中随机挑一个。p=0.9 意味着模型 90% 确定的那几个词里随机选，避免重复和枯燥。

---

## 7. Prefill vs Decode：两种计算模式

**代码位置**：`lite_llama/models/qwen3.py` Qwen3Attention.forward 第 139 行

```python
if seq_len > 1:
    attn_output = self.attn.context_forward(...)   # Prefill
else:
    attn_output = self.attn.token_forward(...)     # Decode
```

| | Prefill | Decode |
|---|---|---|
| seq_len | > 1（整个 prompt） | == 1（单个 token） |
| Attention kernel | FlashAttention2 (NoPad) | FlashDecoding |
| QK^T 形状 | (N, N) 方阵 | (1, cache_len) |
| 瓶颈 | 计算（矩阵乘法） | 显存带宽（读 KV cache） |
| KV cache | 写入 | 读取 + 追加 1 行 |
| qk_scale | `1/sqrt(d) * 1.4427` (用 exp2) | `1/sqrt(d)` (用 exp) |

**NoPad Batching**：不同长度的 prompt 拼接在一起，而不是 pad 到相同长度。通过 `b_start_loc` 和 `b_seq_len` 记录每个样本的起止位置。

```
普通批处理：           NoPad 批处理：
[1,2,3,4,0,0,0]       [1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7]  
[1,2,3,4,5,6,7]   →    b_start_loc = [0, 9]
[1,2,3,0,0,0,0]        b_seq_len   = [9, 7]
         ↑                             ↑
     浪费计算                 节省 30-50% 计算量
```

---

## 8. 优化技术清单

### 算子融合（减少显存读写）

| 融合算子 | 原操作 | 文件 |
|---|---|---|
| skip_rmsnorm | residual + x → RMSNorm | `kernels/skip_rmsnorm.py` |
| swiglu_forward | SiLU(gate) × up | `kernels/swiglu.py` |
| rope_emb_forward | Q 和 K 同时做 RoPE | `kernels/rope_emb.py` |
| update_kv_buffer | KV 写入 cache | `kernels/update_kv_buffer.py` |
| KV 权重融合 | K_proj + V_proj → kv_proj_weight | `models/qwen3.py` |

### 内存管理

| 技术 | 说明 | 文件 |
|---|---|---|
| FlashAttention | QK^T 从不写回 HBM | `kernels/flashattention2_nopad.py` |
| FlashDecoding | KV 分块并行 + 归约 | `kernels/flashdecoding.py` |
| GQA | KV head 比 Q head 少 | `models/qwen3.py` |
| KV Cache 动态分配 | 按需分配，引用计数 | `executor/mem_manager.py` |
| GQA KV head 索引 | 代替 repeat_kv | Attention 类 |

### 计算优化

| 技术 | 说明 | 文件 |
|---|---|---|
| exp2 替代 exp | Prefill 用更快指令 | `models/qwen3.py` (qk_scale × 1.4427) |
| NoPad Batching | 变长序列拼接 | `kernels/flashattention2_nopad.py` |
| Online Softmax | 分块 softmax 不写中间结果 | FlashAttention kernel |
| M-RoPE | 3D 位置编码（多模态） | `models/RotaryEmbedding.py` |

---

## A. 关键数据结构

```python
# attention 信息结构体（贯穿整个推理过程）
atten_info = AttentionInfo(
    kv_buffer = [             # 每层一个 KV cache buffer
        tensor(max_tokens, 2*kv_heads, head_dim),   # layer 0
        tensor(max_tokens, 2*kv_heads, head_dim),   # layer 1
        ...36 层...
    ],
    cur_select_index = tensor([0, 1, 2, ..., 24]),  # 当前要写入的 KV cache 位置
    b_start_loc = tensor([0]),     # 每个样本在 batch 中的起始位置
    b_seq_len = tensor([25]),      # 每个样本的实际序列长度
    b_req_tokens_table = tensor([[0,1,2,...,2047]]), # 每个请求的 token 索引映射
    max_actual_seq_len = 25,       # 当前最大序列长度
)
```

---

## B. 完整的一次推理（以 Qwen3-VL-4B 为例）

```
用户: "图片内容是什么" + 🐕图片
─────────────────────────────────────────────

[Prefill]
  tokenize("图片内容是什么") → [1, 104198, ..., 151655x144, ...]  (410 tokens)
  embed_tokens → (1, 410, 2560)
  
  vision_encode(image):
    Conv3D patch → ViT blocks × 27 → PatchMerger → DeepStack特征
    → image_embeds (144, 2560), deepstack[3层]
  
  masked_scatter: 把 image_embeds 替换到 input_ids 的 <image> token 位置
  M-RoPE: 计算 3D 位置编码 (T,H,W)
  
  For layer 0..35:
    skip_rmsnorm → FlashAttention2_NoPad (410×410 causal attention)
    skip_rmsnorm → SwiGLU FFN
    (前3层额外注入 DeepStack 视觉特征)
  
  skip_rmsnorm → lm_head → (1, 410, 152000)
  取最后一个位置 → softmax → sample_top_p → 第1个 token: "这"

[Decode step 2]
  embed_tokens("这") → (1, 1, 2560)
  RoPE(position=410+0)
  
  For layer 0..35:
    skip_rmsnorm → FlashDecoding (1×411 attention, KV cache分块并行)
    skip_rmsnorm → SwiGLU FFN
  
  lm_head → softmax → sample_top_p → 第2个 token: "张"

[Decode step 3]
  类似上面 → "图"
  
...循环直到生成 EOS 或达到 max_gen_len...
  
最终输出: "这张图片展示了一只三色伯恩山犬..."
─────────────────────────────────────────────
耗时: Prefill ~1-2s, 每 token decode ~10-20ms
```

---

## C. 推荐阅读路径

1. **入口**：`cli_qwen3vl.py` → `qwen3vl_generate_stream.py`（理解整体流程）
2. **模型结构**：`models/qwen3.py`（LlamaDecoderLayer → Attention → MLP）
3. **注意力机制**：`models/qwen3.py` Attention 类 + `kernels/flashattention2_nopad.py` + `kernels/flashdecoding.py`
4. **优化技巧**：`kernels/skip_rmsnorm.py` + `kernels/swiglu.py` + `kernels/rope_emb.py`
5. **内存管理**：`executor/mem_manager.py` + `executor/model_executor.py`
6. **多模态扩展**：`models/qwen3vl.py` + `models/qwen3vl_vision.py` + `models/RotaryEmbedding.py`（M-RoPE 部分）
