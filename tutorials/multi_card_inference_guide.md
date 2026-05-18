# 多卡推理深度解析

本文以 lite_llama 在 2×910B3 上运行 Qwen3-32B (TP=2) 为例，逐层拆解多卡推理的计算过程、通信开销和优化方向。

---

## 目录

1. [整体架构：TP 下的数据流](#1-整体架构tp-下的数据流)
2. [逐层拆解：每一步在算什么](#2-逐层拆解每一步在算什么)
3. [通信分析：时间花在哪](#3-通信分析时间花在哪)
4. [优化方案](#4-优化方案)
   - [4.1 CUDA/NPU Graph](#41-cudanpu-graph)
   - [4.2 PagedAttention](#42-pagedattention)
   - [4.3 Continuous Batching](#43-continuous-batching)
   - [4.4 Prefill/Decode 调度优化](#44-prefilldecode-调度优化)
   - [4.5 算子融合](#45-算子融合)
   - [4.6 All-Reduce 优化](#46-all-reduce-优化)
5. [优化路线图](#5-优化路线图)

---

## 1. 整体架构：TP 下的数据流

先回顾 Tensor Parallelism 的核心思想：**每层权重切开，多卡同时算，通信合并结果**。

```
输入: "你是谁" (tokenized → 410 tokens)

                    GPU 0 (npu:4)                         GPU 1 (npu:5)
                    ────────────                          ────────────
Step 1: Embed      embed("你是谁")  [相同，复制]           embed("你是谁")
                    ↓ (1, 410, 2560)                     ↓ (1, 410, 2560)

Step 2: Layer 0    ┌─QKV proj (heads 0-15) ─┐           ┌─QKV proj (heads 16-31) ─┐
   Attention        │ FlashAttn (heads 0-15)  │            │ FlashAttn (heads 16-31)  │
                    │ O_proj → all_reduce ◄───┼────────────┼─── O_proj                │
                    └─────────────────────────┘           └───────────────────────────┘
                    ↓ (1, 410, 2560)                     ↓ (1, 410, 2560)

                    ┌─gate/up (intermediate/2)─┐        ┌─gate/up (intermediate/2)─┐
   FFN              │ SwiGLU                    │         │ SwiGLU                    │
                    │ down → all_reduce ◄───────┼─────────┼─── down                   │
                    └──────────────────────────┘         └───────────────────────────┘
                    ↓ (1, 410, 2560)                     ↓ (1, 410, 2560)

...重复 64 层...

Step 3: lm_head    lm_head (vocab/2)                     lm_head (vocab/2)
                         │                                    │
                         └──── all_gather ────────────────────┘
                                        ↓
                              full logits (1, 410, 152064)
                                        ↓
                              sample → "这"

Step 4: Decode     embed("这") [相同，广播]               embed("这") [相同，广播]
  (循环 255 次)     ↓ (1, 1, 2560)                       ↓ (1, 1, 2560)
                    64 层 Attention + FFN (同 Step 2)     64 层 Attention + FFN (同 Step 2)
                    (但 seq_len=1, 用 FlashDecoding)       (但 seq_len=1, 用 FlashDecoding)
                    ↓                                      ↓
                    all_gather → sample → "张"             all_gather → sample → "张"
                    ↓ 广播确保一致                          ↓ 接收广播
```

**关键变化**：
- Prefill 和 Decode 的 Attention kernel 不同（Prefill 用 FlashAttention2, Decode 用 FlashDecoding）
- 每层有 3 次通信（O_proj all_reduce + FFN down all_reduce + lm_head all_gather）
- Decode 每 token 都要走完整 64 层

---

## 2. 逐层拆解：每一步在算什么

以下以 **Decode 阶段生成 1 个 token** 为例，分析单层的耗时构成。时间是针对 910B3 NPU + Qwen3-32B TP2 的估计值（数量级正确，精确值因环境而异）。

### 2.1 输入准备（约 0.05 ms）

```
位置编码 (RoPE):
  position_ids = arange(cur_pos)
  cos, sin = rotary_emb(cur_pos)    # 查表，极快
  
输入:
  h = embed_tokens(token_id)        # (1, 1, 2560) 查表操作，很快
```

时间忽略不计。

### 2.2 QKV 投影（约 0.3 ms）

```
Q = h @ W_q^T          # (1, 2560) × (2560, 2048) → (1, 2048)  ← TP: 16 local heads × 128 dim
K = h @ W_k^T          # (1, 2560) × (2560, 512)  → (1, 512)   ← TP: 4 local kv heads × 128 dim
V = h @ W_v^T          # 同 K

QK norm (Qwen3 特有):
  Q = RMSNorm(Q)        # head_dim=128 维归一化，逐元素操作
  K = RMSNorm(K)
```

三路矩阵乘是独立计算的。Q 投影稍大（2048 输出 vs 512），加起来约 0.3ms。

### 2.3 FlashDecoding Attention（约 0.8 ms）

```
RoPE:
  Q, K = rope(Q, K, cos, sin)    # 旋转位置编码，逐元素操作

KV cache 更新:
  KV_buffer[cur_pos] = concat(K, V)  # 追加新 token 的 KV

FlashDecoding Stage 1（分块并行）:
  把 KV cache 切成 128-token 的块
  每块独立算 Q @ K^T → softmax → 加权 V → 局部结果
  块数 = cur_pos / 128

FlashDecoding Stage 2（归约）:
  用 online softmax 合并所有块的局部结果
```

**这是 Decode 最耗时的部分**。瓶颈不是计算（Q 只有 1 个 token），而是**读取 KV cache**。

```
seq_len=2048 时：
  每层读 KV cache: 2048 × (2 × 4 × 128) × 2 bytes = 4 MB
  64 层总计: 256 MB
  加上 FlashDecoding Stage 1 的中间结果
  
时间: ~0.8ms（读取 HBM 约占 0.5ms，计算约占 0.3ms）
```

### 2.4 O 投影 + AllReduce（约 0.4 ms）

```
attn_output = FlashDecoding 结果    # (1, 2048) — 16 heads × 128 dim

out = attn_output @ W_o^T        # (1, 2048) × (2048, 2560) → (1, 2560)

[通信] all_reduce(out)             # GPU0 的 out + GPU1 的 out → 广播到两卡
```

矩阵乘约 0.15ms，all_reduce 约 0.2-0.3ms（HCCL 延迟 + 数据传输）。

### 2.5 FFN gate/up → SwiGLU（约 0.3 ms）

```
gate = h @ W_gate^T      # (1, 2560) × (2560, 5120) → 两路独立
up   = h @ W_up^T        # TP: 每卡算一半 intermediate

h = SiLU(gate) × up      # swiglu_forward fused kernel
```

SwiGLU 融合 kernel 约 0.05ms，矩阵乘约 0.25ms。

### 2.6 FFN down + AllReduce（约 0.4 ms）

```
out = h @ W_down^T       # (1, 5120) × (5120, 2560) → (1, 2560)

[通信] all_reduce(out)    # 同 2.4
```

### 2.7 SkipRMSNorm（约 0.02 ms × 2）

```
# Attention 前
h = x + residual           # 在 skip_rmsnorm kernel 内融合
h = RMSNorm(h) * weight

# FFN 前
h = x + residual
h = RMSNorm(h) * weight
```

融合 kernel，极快。

### 2.8 单层合计

| 步骤 | 计算 (ms) | 通信 (ms) | 合计 (ms) |
|---|---|---|---|
| QKV 投影 | 0.30 | 0 | 0.30 |
| FlashDecoding | 0.80 | 0 | 0.80 |
| O 投影 + all_reduce | 0.15 | 0.25 | 0.40 |
| FFN gate/up | 0.30 | 0 | 0.30 |
| FFN down + all_reduce | 0.15 | 0.25 | 0.40 |
| RMSNorm × 2 | 0.04 | 0 | 0.04 |
| **单层总计** | **1.74** | **0.50** | **2.24** |

### 2.9 每 token 总耗时

```
64 层计算:  64 × 2.24 = 143 ms
lm_head + all_gather:           3 ms
采样 + 其他开销:                 10 ms
──────────────────────────────────
每 token:   ~156 ms  (约等于实测 183 ms 的 85%，剩下是 kernel launch 等开销)
```

**瓶颈分析**：
- FlashDecoding 的 KV cache 读取占 36%（64 × 0.8 = 51ms）
- all_reduce 通信占 22%（64 × 0.5 = 32ms）
- 矩阵乘法占 27%

---

## 3. 通信分析：时间花在哪

### 3.1 通信量

```
每层通信:
  O_proj all_reduce:   2560 个 float16 = 5 KB
  FFN down all_reduce: 2560 个 float16 = 5 KB
  ─────────────────────────────────────────
  单层: 10 KB

64 层: 640 KB (数据量很小！)
lm_head all_gather: 152064 个 float16 = 304 KB (只发生一次)

瓶颈不是数据量，是通信次数 × 延迟
```

HCCL 的每次 all_reduce 有固定延迟（约 0.1ms）加上数据传输时间。64 层 × 2 次 = 128 次通信，128 × 0.2ms ≈ 26ms。

### 3.2 为什么通信占比这么大

```
数据量 (640 KB / token) → HBM 带宽 ~1.5 TB/s → 纯传输只需 ~0.4 μs
实际耗时 ~32 ms → 几乎全是延迟（kernel launch + HCCL handshake）
```

**根源**：128 次独立的 all_reduce 调用，每次都要走 HCCL 的调度开销。如果能合并成 1 次，延迟会降低 100 倍。

---

## 4. 优化方案

### 4.1 CUDA/NPU Graph

**当前状态**：已实现但 buggy，代码在 `executor/cuda_graph.py`。

**原理**：

```
正常推理流程:
  for i in range(256):           ← Python 循环
      input → kernel_1 → k_2 → ... → k_50 → output
      ↑ 每次循环开销：Python 调度 + kernel launch × 50

Graph 模式:
  graph = capture_once(          ← 只捕获一次
      input → k_1 → k_2 → ... → k_50 → output
  )
  for i in range(256):
      graph.replay(input)        ← 一次 GPU 调用，零 CPU 开销
```

**收益**：消除 128 次 kernel launch 和 all_reduce 调度的 CPU 开销。在 910B3 上预期节省 **15-25ms/token**。

**为什么之前 buggy**：KV cache 地址在每次 decode 后变化（`cur_select_index` 不同），Graph 需要动态更新 tensor 内容而非地址。需要：
1. 将 `cur_select_index` 作为 graph 的输入 tensor（每次 replay 前更新值）
2. 将 `b_seq_len` 等变化的状态也作为输入
3. （最难）FlashDecoding 的 `b_req_tokens_table` 是非连续索引，graph 捕获时形状固定但值变化

**实现要点**：Graph 捕获 decode 路径（seq_len=1），留出可变 slot 作为输入。主要难度在 FlashDecoding kernel 的参数处理。

### 4.2 PagedAttention

**当前状态**：未实现。现有 KV cache 是连续分配的。

**原理**：

```
当前方式 (Contiguous):
  req0: [████████████████░░░░░░░░░░░░░░░░░░░░]  分配 max_len 但只用前几段
  req1: [░░░░░░░░░░░░░░░░████████░░░░░░░░░░░░]  大量空洞
  问题：碎片化，内存利用率低

PagedAttention:
  把 KV cache 切成固定大小的 page (如 16 tokens)
  req0: [P0]→[P1]→[P2]→[P5]     ← 用链表连起来，不连续但无碎片
  req1: [P3]→[P4]→[P6]
  空闲: [P7] [P8] [P9] ...       ← 空闲 page pool，按需分配
```

**对 Kernel 的改动**：

```
当前 FlashDecoding kernel:
  b_req_tokens_table[i, :seq_len] = [连续索引 a, a+1, a+2, ...]

Paged FlashDecoding kernel:
  b_req_tokens_table[i, :seq_len] = [P0_first, ..., P0_last, P1_first, ...]
  ↑ 索引跳过 page 边界，不再是连续递增
  
  其他逻辑不变！FlashDecoding 本来就是通过 b_req_tokens_table 
  做间接索引的，天然支持 page 化。
```

**FlashAttention2-NoPad (Prefill)**：需要特殊处理，因为 prefill 是一次性全量 attention，不需要 KV cache 分页。但 batch 中不同请求的 prompt 长度不同，可以通过 NoPad batching 拼接来处理。

**收益**：KV cache 利用率从 ~60% 提升到 ~95%，同显存可服务更多请求。对单请求延迟无影响，但对吞吐有显著提升。

### 4.3 Continuous Batching

**当前状态**：未实现。现有是静态批处理（所有请求同时开始，等待最后完成的）。

**原理**：

```
静态批处理 (Iteration-level):
  Time ──────────────────────────────────────────→
  req0: [Prefill────────][D][D][D][D][D][D][D]
  req1:                     [Prefill──────────][D][D]
  req2:                                [Prefill][D][D][D][D]
         ↑ req0 完成时 req2 才开始                 ↑ 总有空位

Continuous Batching:
  req0: [Prefill][D][D][D][D][D][D][D] ← 完成，释放
  req1:          [Prefill──────────][D][D]   ← 插入
  req2:                   [Prefill────][D][D][D][D]
  req3:                         [Prefill][D][D]      ← 动态插入
  req4:                              [Prefill][D][D]
```

**实现要点**：

```
调度循环:
  while True:
    # 1. 收集新请求，拼入当前 batch
    new_reqs = pull_waiting_queue()
    for r in new_reqs:
        alloc_kv_cache(r.prompt_len)
        batch.add_prefill(r)
    
    # 2. 运行一个 forward step
    #  - Prefill: 新请求的 prompt (多 token)
    #  - Decode: 已有请求的下一个 token (单 token)
    #  两种模式在一次 forward 中混合
    logits = model.forward(combined_batch)
    
    # 3. 采样 + 更新
    for req in batch:
        if req.is_prefill:
            req.start_decode(logits[req.idx, -1])
        else:
            req.next_token = sample(logits[req.idx])
    
    # 4. 清理完成的请求
    for req in batch.completed:
        free_kv_cache(req)
```

混合 Prefill/Decode 的关键：用 NoPad batching 把 prefill 和 decode 的 token 拼在一起。Prefill 请求有多个 token，Decode 各 1 个。FlashAttention2-NoPad 已经支持变长序列。

**收益**：
- 吞吐提升 2-5 倍（在高并发场景）
- GPU 利用率从 ~30% 提升到 ~80%
- 单请求延迟不受影响（只改善 batch 利用率）

### 4.4 Prefill/Decode 调度优化

**问题**：Prefill 和 Decode 的计算特性完全不同，同时执行会互相干扰。

```
Prefill: compute-bound（GPU 算力瓶颈）
  - 大量矩阵乘法，GPU Tensor Core 跑满
  - 耗时与 prompt_len² 成正比

Decode: memory-bound（显存带宽瓶颈）
  - 大量 KV cache 读取
  - 每 token 耗时与 seq_len 成正比
```

**策略 1：Prefill-prioritized**

```
把 Prefill 拆分 (chunked prefill):
  长 prompt (4000 tokens) → 切成 4 段，每段 1000 tokens
  每轮 forward: 处理 1 段 prefill + 所有 decode
  避免一个长 prefill 阻塞所有 decode
```

**策略 2：Decode-prioritized**

```
每个 schedule step:
  1. 优先跑所有 decode（延迟敏感）
  2. 剩余计算能力跑 prefill
  
  如果 GPU 算力 100%，decode 用 40%，prefill 用 60%
```

**策略 3：Split Prefill/Decode**

```
在长 prefill 期间，把 decode 请求迁移到另一组 GPU
  GPU 0-1: 处理 prefill
  GPU 2-3: 处理 decode
  避免互相干扰
```

### 4.5 算子融合

lite_llama 已实现的融合：
- `skip_rmsnorm`: residual + RMSNorm
- `swiglu_forward`: SiLU(gate) × up
- `rope_emb_forward`: Q 和 K 同时 RoPE
- `update_kv_buffer`: KV 写入缓存
- KV 权重融合：K_proj + V_proj → 一个矩阵

还可以再做：

**QKV 投影融合**：

```
当前:
  Q = x @ W_q     ← 一次矩阵乘
  K = x @ W_k     ← 又一次
  V = x @ W_v     ← 又一次
  
融合后:
  QKV = x @ W_qkv_concat     ← 一次矩阵乘
  Q, K, V = split(QKV)
  
  节省 2/3 的 kernel launch 开销
  （注意：TP 下 W_q 和 W_kv 的切法不同，融合需要特殊处理）
```

**FlashDecoding + KV cache read 融合**：

```
当前:
  1. 从 HBM 读 KV cache (独立 kernel)
  2. FlashDecoding (读 KV + 计算)
  
融合后:
  FlashDecoding kernel 直接从 KV cache buffer 读数据
  消除一次 HBM 往返
```

### 4.6 All-Reduce 优化

**问题**：64 层 × 每层 2 次 all_reduce = 128 次通信，延迟累加。

**方案 1：All-Reduce 合批**

```
当前:
  for layer in layers:
      h = attention(h)
      all_reduce(h)            ← 第 1 次
      h = ffn(h)
      all_reduce(h)            ← 第 2 次

合批后:
  for layer in layers:
      h = attention(h)         ← 暂存 partial 结果
      h = ffn(h)
      # 最后统一 all_reduce（等价于两次 all_reduce 合并）
  # 或者多层的 all_reduce 合并：
  all_reduce_batch([l0_out, l1_out, l2_out, ...])
```

难点：all_reduce 是同步屏障，合并后前一层必须等后一层算完，打乱了流水线。

**方案 2：Reduce-Scatter + All-Gather 替代 All-Reduce**

```
All-Reduce: 每个 rank 得到完整结果 (求和 + 广播)
  rank0: a0 → a0+b0 → 完整
  rank1: b0 → a0+b0 → 完整
  
Reduce-Scatter: 结果分散存储 (只求和，各自拿一部分)
  rank0: a0 → a0+b0 的前半
  rank1: b0 → a0+b0 的后半

如果下一个操作是列切（如 gate_proj），不需要完整结果，只需要部分
→ 用 reduce_scatter 代替 all_reduce，通信量减半
```

**方案 3：通信隐藏 (Overlap)**

```
把第 N 层的 all_reduce 和第 N+1 层的计算重叠：
  
  Layer N:   [计算] [all_reduce────────]
  Layer N+1:        [无关计算...] [all_reduce────────]
                              ↑ 重叠执行
```

在 HCCL 上需要 CUDA Stream 支持（不同 stream 上的 kernel 可以并行）。

---

## 5. 优化路线图

按性价比排序：

```
阶段 1: 低风险、高收益（1-2 周）
├── ✅ 算子融合 (QKV 合并、FlashDecoding+KV read 融合)
├── ✅ NPU Graph 修复 (消除 kernel launch 开销)
└── 预期收益: 15-25% decode 加速

阶段 2: 中等风险、中等收益（3-4 周）
├── PagedAttention (KV cache 分页)
├── All-Reduce 合批 (减少通信次数)
└── 预期收益: KV cache 利用率 +50%, decode 加速 10%

阶段 3: 高复杂度、最大吞吐收益（4-8 周）
├── Continuous Batching (动态调度)
├── Chunked Prefill (长 prompt 拆分)
└── 预期收益: 吞吐 2-5 倍, 多请求场景质变

阶段 4: 极限优化（按需）
├── 通信隐藏 (overlap compute/comm)
├── FP8 量化 (降低权重大小)
└── 预期收益: 30-50% 吞吐提升
```
```
● 已推送。实现总结：

  阶段 1：算子融合 + NPU Graph

  优化: NPU Graph
  文件: executor/npu_graph.py
  状态: 新增。NpuGraphRunner 捕获 decode forward 为 graph，消除 ~128 次 kernel launch 开销。自动检测 NPU 是否支持，不支持则静默 fallback
  ────────────────────────────────────────
  优化: QKV 融合
  文件: —
  状态: K/V 权重早已融合（kv_proj_weight），Q 因 TP head 数不同无法合并
  ────────────────────────────────────────
  优化: FlashDecoding+KV read
  文件: —
  状态: KV cache 读取已在 FlashDecoding kernel 内部

  阶段 2：PagedAttention + AllReduce

  优化: PagedAttention
  文件: executor/paged_attention.py
  状态: 新增。PagedKVCacheManager 以 page_size=16 为粒度分配 KV cache。kernel 零改动——FlashDecoding 已用 b_req_tokens_table 间接索引
  ────────────────────────────────────────
  优化: All-Reduce 合批
  文件: —
  状态: 分析后不可行：O_proj 的 all_reduce 结果被 FFN RMSNorm 消费，无法延迟到 FFN down 之后

  为什么 All-Reduce 合批做不了
  Layer N:
    O_proj → [all_reduce A] → residual add → RMSNorm → FFN → down → [all_reduce B]
                                      ↑
                            FFN 需要 A 的结果，不能等 B 一起做

  这不是代码问题，是数学依赖。要减少通信次数只能靠 reduce_scatter
  替代或通信-计算 overlap，两者都需要更底层改动。
```
---

## A. 参考：各模型在 910B3 TP2 下的预估性能

| 模型 | 每 token decode | 256 token 总时间 | 备注 |
|---|---|---|---|
| Qwen3-4B | ~25 ms | ~6.4 s | 单卡即可 |
| Qwen3-8B | ~45 ms | ~11.5 s | 单卡或 TP2 |
| Qwen3-14B | ~85 ms | ~21.8 s | TP2 |
| Qwen3-32B | ~180 ms | ~46 s | TP2 (当前实测) |
| Qwen3-72B | ~380 ms | ~97 s | TP4 或 TP8 |

---

## B. 常用性能分析命令

```bash
# 查看 HCCL 通信拓扑
hccl_info

# 查看 NPU 利用率
npu-smi info watch -n 0

# Triton kernel 性能分析
ASCEND_LAUNCH_BLOCKING=1 python ...

# 通信带宽测试
mpirun -np 2 all_reduce_perf -b 1M -e 256M -f 2
```
