# Triton 高性能算子编程指南

以 lite_llama 项目中的实际 kernel 为例，从零讲解如何用 Triton 写出高性能 GPU 算子。

---

## 目录

1. [GPU 是怎么算的](#1-gpu-是怎么算的)
2. [Triton 是什么](#2-triton-是什么)
3. [Triton 编程模型](#3-triton-编程模型)
4. [实战 1：RMSNorm — 逐元素操作的并行化](#4-实战-1rmsnorm--逐元素操作的并行化)
5. [实战 2：SwiGLU — 算子融合的价值](#5-实战-2swiglu--算子融合的价值)
6. [实战 3：RoPE — 向量化读写](#6-实战-3rope--向量化读写)
7. [实战 4：FlashAttention — Online Softmax 与分块计算](#7-实战-4flashattention--online-softmax-与分块计算)
8. [实战 5：FlashDecoding — 两级归约](#8-实战-5flashdecoding--两级归约)
9. [性能调优清单](#9-性能调优清单)
10. [补充：矩阵乘法基础](#10-补充矩阵乘法基础)

---

## 1. GPU 是怎么算的

### 1.1 核心概念：延迟隐藏

GPU 的本质设计哲学：**用并行换吞吐，用计算隐藏延迟**。

```
CPU（低延迟、少核心）：
  1 个任务 → 极快完成 → 下一个 → 极快完成
  
GPU（高延迟、海量核心）：
  同时启动 10000 个任务
  → 每个都很慢（读显存要几百个时钟周期）
  → 但 10000 个同时跑，总吞吐极高
  → 只要有一个完成了，就换上下一个，掩盖读取的延迟
```

**warp**（CUDA 术语）是 GPU 调度的基本单位：32 个线程一组，同时执行同一条指令。NPU 有类似机制。

### 1.2 显存层级

```
┌─────────────────────────────────────────────┐
│ HBM（显存/Global Memory）                    │
│ 容量：最大（40-80 GB）                        │
│ 速度：慢（~1.5 TB/s）                         │
│ 存什么：权重、KV cache、输入输出               │
├─────────────────────────────────────────────┤
│ L2 Cache                                     │
│ 容量：中等（几十 MB）                          │
│ 速度：中                                      │
├─────────────────────────────────────────────┤
│ SRAM（Shared Memory / L1）                   │
│ 容量：小（~256 KB per SM）                    │
│ 速度：快（~20 TB/s）                           │
│ 存什么：当前 tile 的数据                       │
├─────────────────────────────────────────────┤
│ 寄存器                                       │
│ 容量：极小（~256 KB per SM）                  │
│ 速度：最快                                    │
│ 存什么：当前线程的局部变量                     │
└─────────────────────────────────────────────┘
```

**瓶颈在哪里？** 不是计算，是**数据搬运**。

```
加一次浮点数乘法：  ~0.01 ns（可以忽略）
从 HBM 读一个数：  ~500 ns（比计算慢 50000 倍）

结论：你在纸上算某个算法的时间复杂度 = O(N³)，在 GPU 上实际时间 
      ≈ 从 HBM 读写了多少数据 / HBM 带宽。
```

这就是为什么 **FlashAttention 能快 3-4 倍**的原因——不是算得更快，是读写更少。

---

## 2. Triton 是什么

**Triton** 是 OpenAI 推出的 GPU 编程语言，定位介于手写 CUDA 和 PyTorch 之间。

```
手写 CUDA kernel
  优点：完全控制，能榨干每一点性能
  缺点：难写、难调试、不同 GPU 要适配、显存管理要手写

PyTorch 算子组合
  优点：简单
  缺点：每个算子都写回 HBM，浪费带宽

Triton（中间道路）
  优点：用 Python 写 kernel 逻辑，自动 tiling，自动优化
  缺点：不如手写 CUDA 灵活，但足够用
```

**Triton 的核心抽象**：你只需描述每个 **program（线程块）** 做什么，Triton 自动把 program 映射到 GPU 的 SM（Streaming Multiprocessor）上执行。

---

## 3. Triton 编程模型

### 3.1 基本结构

```python
import triton
import triton.language as tl

# 这是 Triton kernel —— 跑在 GPU 上的函数
@triton.jit
def my_kernel(
    x_ptr,        # 输入张量的指针（指向 HBM 中的地址）
    y_ptr,        # 输出张量的指针
    N,            # 张量大小
    BLOCK_SIZE: tl.constexpr,  # 编译期常量（决定 tile 大小）
):
    # 获取当前 program 的 ID
    pid = tl.program_id(0)
    
    # 计算当前 program 负责的元素范围
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N          # 处理边界（N 不一定是 BLOCK_SIZE 的倍数）
    
    # 从 HBM 加载数据到 SRAM（Triton 自动管理 SRAM 分配）
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # 在 SRAM 中计算（这部分代码会被编译为 GPU 指令）
    y = x * x + 1.0
    
    # 写回 HBM
    tl.store(y_ptr + offsets, y, mask=mask)

# 这是 Python wrapper —— 跑在 CPU 上
def my_function(x):
    y = torch.empty_like(x)
    N = x.numel()
    BLOCK_SIZE = 1024
    
    # 计算需要多少个 program（grid）
    grid = (triton.cdiv(N, BLOCK_SIZE),)
    
    # 启动 kernel
    my_kernel[grid](
        x, y, N, BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,        # 每个 program 的 warp 数量
        num_stages=2,       # 流水线阶段数
    )
    return y
```

### 3.2 关键概念

**tl.program_id(axis)**：当前 program 在 grid 中的位置。`pid = tl.program_id(0)` 返回 0, 1, 2, ..., M-1，每个 program 处理不同的数据块。

**grid**：总共有多少个 program。`grid = (M,)` 表示 1D grid，有 M 个 program。

**tl.constexpr**：编译期常量。BLOCK_SIZE 在编译时确定，不在运行时改变。这允许 Triton 编译器针对特定 BLOCK_SIZE 生成专门的机器码。

**tl.arange**：生成一个从 0 开始的向量。类似 Python 的 `range()`。

**tl.load / tl.store**：从 HBM 读取/写入数据。Triton 自动管理 SRAM 缓存。

**mask**：处理边界条件。如果 N 不是 BLOCK_SIZE 的倍数，最后一个 block 会越界，mask 防止读到非法内存。

### 3.3 数据流

```
Python (CPU 端)
    │
    │  my_kernel[grid](x_ptr, y_ptr, ...)
    │  分配 grid，把参数传到 GPU
    ▼
───────────────────────────────────────────── GPU 边界
    │
    │  Triton 编译器生成 PTX（类似汇编）
    │  GPU 驱动加载到 SM 上执行
    ▼
GPU SM (每个 SM 运行多个 program)
    │
    │  每个 program:
    │    tl.load  →  从 HBM 读到 SRAM
    │    tl.xxx   →  在寄存器上计算
    │    tl.store →  从 SRAM 写回 HBM
    ▼
─────────────────────────────────────────────
    │
    │  kernel 执行完毕，控制返回 Python
    ▼
Python (CPU 端) — 拿到结果
```

---

## 4. 实战 1：RMSNorm — 逐元素操作的并行化

**文件**：`lite_llama/kernels/skip_rmsnorm.py`

### 4.1 算法

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) * weight

其中 mean(x²) 是每个 token 独立计算的均值
输入 x: (batch, seq_len, hidden_size) → 展平为 (total_rows, hidden_size)
每行独立归一化，不同行之间没有依赖
```

### 4.2 并行策略

```python
# 每行由一个 program 处理
# grid = (total_rows,)  即 batch * seq_len 个 program

@triton.jit
def rms_norm_kernel(X, Y, W, ..., N, eps, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)  # 当前处理第几行
    X += pid * x_stride_r   # 定位到当前行的首地址
    Y += pid * y_stride_r
    
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N  # N = hidden_size（如 2560）
    
    # 1. 加载当前行的全部元素到 SRAM
    x = tl.load(X + cols * x_stride_c, mask=mask, other=0.0).to(tl.float32)
    
    # 2. 计算 RMS = sqrt(mean(x²) + ε)
    var = tl.sum(x * x, axis=0) / N         # 所有 BLOCK_SIZE 个元素求和
    rrms = 1.0 / tl.sqrt(var + eps)
    
    # 3. 归一化并乘 weight
    w = tl.load(W + cols, mask=mask, other=0.0)
    y = (x * rrms).to(tl.float16) * w      # 结果转为 fp16
    
    # 4. 写回
    tl.store(Y + cols * y_stride_c, y, mask=mask)
```

**关键理解**：`tl.sum(x * x, axis=0)` 这行把 BLOCK_SIZE 个元素的平方累加成 1 个标量——这是 **program 内归约**（warp-level reduction）。Triton 自动生成高效的 shuffle 指令实现。

### 4.3 SkipRMSNorm：残差融合

```python
@triton.jit
def skip_rms_norm_kernel(Y, X, R, W, ..., N, eps, BLOCK_SIZE: tl.constexpr):
    # ...同上加载 x 和 r...
    x = tl.load(X + ..., mask=mask).to(tl.float32)
    r = tl.load(R + ..., mask=mask).to(tl.float32)
    
    # 融合点：先做残差加法
    x = x + r
    tl.store(R + ..., x, mask=mask)  # 写回残差（供下一层用）
    
    # 再做 RMSNorm（同上）
    var = tl.sum(x * x, axis=0) / N
    rrms = 1.0 / tl.sqrt(var + eps)
    w = tl.load(W + ..., mask=mask)
    y = (x * rrms).to(tl.float16) * w
    tl.store(Y + ..., y, mask=mask)
```

**融合的价值**：
```
未融合（两个操作，两次 HBM 往返）：
  x = x + r           → 写回 HBM（读 r + 写 x）
  y = RMSNorm(x)       → 读 x + 写 y
  共：4 次 HBM 操作

融合（一个 kernel，一次 HBM 往返）：
  读 x, r → 算 x+r → 直接算 RMSNorm → 写 y, r
  共：2 次 HBM 操作
  
带宽节省：50%
```

---

## 5. 实战 2：SwiGLU — 算子融合的价值

**文件**：`lite_llama/kernels/swiglu.py`

### 5.1 算法

```
SwiGLU(a, b) = SiLU(a) * b = (a * sigmoid(a)) * b

未融合：
  gate = SiLU(x @ W_gate)    # 一个 kernel
  up   = x @ W_up            # 另一个 kernel
  out  = gate * up           # 第三个 kernel
  
融合后：
  out = swiglu_forward(a, b) # 一个 kernel
  其中 a = x @ W_gate, b = x @ W_up（还是两次矩阵乘，但逐元素乘法融合了）
```

### 5.2 Triton 实现

```python
@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    row_stride,
    n_cols: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 每行由一个 program 处理
    pid = tl.program_id(0)
    a_ptr += pid * row_stride
    b_ptr += pid * row_stride
    c_ptr += pid * row_stride
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # 加载 a（gate）和 b（up）到 SRAM
    a_row = tl.load(a_ptr + offsets, mask=mask, other=0).to(tl.float32)
    b_row = tl.load(b_ptr + offsets, mask=mask, other=0)
    
    # 融合计算：SiLU(a) * b，不写回 HBM
    c_row = silu(a_row) * b_row
    
    tl.store(c_ptr + offsets, c_row, mask=mask)

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)
```

**为什么 `a` 转 float32 而 `b` 不转**：`sigmoid` 在 float16 下容易精度不够，所以 a 在 float32 下计算 sigmoid，b 保持 float16。

---

## 6. 实战 3：RoPE — 向量化读写

**文件**：`lite_llama/kernels/rope_emb.py`

### 6.1 算法

对一个 head_dim 维的向量，把相邻元素成对旋转：

```
对于第 i 对 (x_2i, x_2i+1)：
  [x_2i'] = [cos(θ_i)  -sin(θ_i)] [x_2i ]
  [x_2i+1']= [sin(θ_i)   cos(θ_i)] [x_2i+1]

展开写：
  x_2i'   = x_2i * cos(θ_i) - x_2i+1 * sin(θ_i)
  x_2i+1' = x_2i * sin(θ_i) + x_2i+1 * cos(θ_i)
```

### 6.2 Triton 实现

```python
@triton.jit
def _triton_rope_emb(
    q_ptr, ..., k_ptr, ...,
    cos, sin, ...,
    sl, bs, n_qh, n_kh, hd, ...,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_id = pid // sl
    cos_row_idx = pid % sl
    
    # 定位到当前行（batch_id 的第 cos_row_idx 个 token）
    q_ptr += pid * q_row_stride
    k_ptr += pid * k_row_stride
    
    # 定位到对应的 cos/sin 行
    cos_ptr = cos + batch_id * cos_b_stride + cos_row_idx * cos_s_stride
    sin_ptr = sin + batch_id * sin_b_stride + cos_row_idx * sin_s_stride
    
    # 加载 cos 和 sin（半个 head_dim，因为每对用同一个旋转角）
    cos_offsets = tl.arange(0, pad_hd // 2)
    cos_mask = cos_offsets < hd // 2
    cos_row = tl.load(cos_ptr + cos_offsets, mask=cos_mask)
    sin_row = tl.load(sin_ptr + cos_offsets, mask=cos_mask)
    
    # 加载 Q 和 K 的前半部分（偶数位置 x_{2i}）
    first_half_q_offsets = (
        tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
    )
    q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=...)
    k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=...)
    
    # 加载后半部分（奇数位置 x_{2i+1}）
    second_half_q_offsets = first_half_q_offsets + (hd // 2)
    q_tile_2 = tl.load(q_ptr + second_half_q_offsets, mask=...)
    k_tile_2 = tl.load(k_ptr + second_half_k_offsets, mask=...)
    
    # 应用旋转
    new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
    new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
    
    # 写回
    tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=...)
    tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=...)
```

**向量化读写的技巧**：
```
读 Q 前半：
  offsets = tl.arange(n_qh)[:, None] * hd + tl.arange(hd//2)[None, :]
  → 产生一个 (n_qh, hd//2) 的偏移矩阵
  → tl.load 一次性加载 n_qh * hd//2 个元素
  → 相当于一个 coalesced memory access（合并访存）
```

---

## 7. 实战 4：FlashAttention — Online Softmax 与分块计算

**文件**：`lite_llama/kernels/flashattention2_nopad.py`

这是本项目中**最复杂的 kernel**，也是最重要的一个。

### 7.1 问题：为什么标准 Attention 慢

```python
# 标准实现
scores = Q @ K.T / sqrt(d)        # (seq, seq) → 写回 HBM！
masked  = causal_mask(scores)     # 读 HBM → 写 HBM
weights = softmax(masked)         # 读 HBM → 写 HBM
output  = weights @ V             # 读 HBM → 写 HBM

# 问题：scores 和 weights 都是 (seq_len, seq_len) 的矩阵
# seq_len=4096 时：4096*4096*2 bytes = 32 MB（fp16）
# 每次 forward 都要读写这个中间矩阵，带宽瓶颈严重
```

### 7.2 核心思想：分块 + Online Softmax

**分块**：把 Q 和 K 分成小块（tiles），每次只加载一小块到 SRAM。

```
把 Q 沿行切成 M // BLOCK_M 块，把 K 沿列切成 N // BLOCK_N 块
每对 (Q块, K块) 在 SRAM 中计算局部 attention，逐步累积结果
```

**Online Softmax**：标准 softmax 需要全部元素才能算。但推导一个巧妙的递推公式：

```
标准 softmax：
  m = max(x), s = sum(exp(x - m)), y = exp(x - m) / s

Online softmax（分两个块 x1, x2）：
  块1：m1 = max(x1), s1 = sum(exp(x1 - m1)), acc1 = exp(x1 - m1) / s1 * V1
  块2：m12 = max(m1, m2)
       new_acc = s1/s12 * exp(m1 - m12) * acc1 + s2/s12 * exp(m2 - m12) * acc2
       s12 = s1 * exp(m1 - m12) + s2 * exp(m2 - m12)

通过 m 和 s 两个统计量，可以在不知道后续块的情况下随时更新结果
```

### 7.3 伪代码走读

```python
@triton.jit
def flash_attention2_nopad_kernel(Q, K, V, O, ...):
    # 获取当前 program 负责的 (batch, head) 组合
    cur_bh = tl.program_id(1)       # 0 ... batch*heads-1
    cur_batch = cur_bh // heads
    cur_head = cur_bh % heads
    
    # 获取该 batch 的序列长度和起始位置（NoPad 批处理）
    cur_seq_len = tl.load(B_Seqlen + cur_batch)
    cur_seq_start = tl.load(B_Start_Loc + cur_batch)
    
    # 当前 program 负责的 Q 行范围（tile）
    block_m_idx = tl.program_id(0)
    offs_m = block_m_idx * BLOCK_M_SIZE + tl.arange(BLOCK_M_SIZE)
    
    # 加载 Q 的 tile 到 SRAM
    q_offs = (cur_seq_start + offs_m) * stride_q_bs + cur_head * stride_q_heads + ...
    q = tl.load(Q + q_offs, mask=offs_m < cur_seq_len, other=0.0)
    
    # 初始化 online softmax 的统计量
    m_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32) - float("inf")  # 当前最大值
    d_i = tl.zeros((BLOCK_M_SIZE,), dtype=tl.float32)                  # 当前分母
    acc = tl.zeros((BLOCK_M_SIZE, HEAD_DIM), dtype=tl.float32)        # 输出累加器
    
    # 遍历 K 的所有块
    for start_n in range(0, cur_seq_len, BLOCK_N_SIZE):
        # 加载 K 块
        k = tl.load(K + (cur_seq_start + start_n) * stride_k_bs + ...)
        
        # 计算 Q @ K^T（在 SRAM 中）
        qk = tl.dot(q, k)           # (BLOCK_M, BLOCK_N)
        
        # Causal mask
        casual_mask = offs_m[:, None] >= (start_n + offs_n[None, :])
        qk = tl.where(casual_mask, qk * sm_scale, -1.0e8)
        
        # Online softmax 更新
        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        qk -= m_ij[:, None]                              # 稳定化
        p = tl.math.exp2(qk)                             # 用 exp2 更快
        d_ij = tl.sum(p, 1)
        
        alpha = tl.math.exp2(m_i - m_ij)                 # 旧值的缩放因子
        d_i = d_i * alpha + d_ij                         # 更新分母
        acc = acc * alpha[:, None]                       # 缩放旧累加器
        
        # 加载 V 块，计算 P @ V
        v = tl.load(V + ...)
        acc = tl.dot(p.to(v.dtype), v, acc)              # 累加到输出
        
        m_i = m_ij                                       # 更新最大值
    
    # 最终归一化并写回
    acc = acc / d_i[:, None]
    tl.store(O + ..., acc, mask=...)
```

**关键点解析**：

1. **grid 是 2D 的**：`(num_M_blocks, batch * heads)`。每个 program 处理一个 (Q_tile_row, head) 组合。

2. **Causal mask 用 `>=` 而非 `<`**：`offs_m >= offs_k` 保证下三角为有效值，上三角被 mask 为 `-inf`。

3. **`exp2` 替代 `exp`**：qk_scale 乘以 `1.4427`（= 1/ln2），然后用 `exp2(x)` 算，GPU 上 `exp2` 比 `exp` 快。

4. **两个累加器 `m_i` 和 `d_i`** 实现 online softmax——不需要知道完整 softmax 结果就能逐步更新。

---

## 8. 实战 5：FlashDecoding — 两级归约

**文件**：`lite_llama/kernels/flashdecoding.py`

### 8.1 问题

Decode 阶段 `seq_len_q = 1`，但 `seq_len_kv` 可能长达数万。直接分块的话，Grid 维度不够并行化——因为 Q 只有 1 行，`M 方向` 是 1，所有 parallel program 都在 `head 方向`，数量有限（比如 32 个）。

### 8.2 方案：两级归约

```
Stage 1（分块计算）：把 KV cache 切成 PARTITION_SIZE 的块
  每个块做一个局部的 FlashAttention → mid_o, mid_logsumexp
  
  并行维度：(batch, head, num_partitions)
  如果 num_partitions = 128，batch=4，heads=32：
  → 4 × 32 × 128 = 16384 个 parallel program！

Stage 2（归约）：把每个块的 (mid_o, mid_logsumexp) 归约成最终结果
  用 online softmax 的公式合并
  
  并行维度：(batch, head)
  每个 program 归约自己负责的那些分区
```

### 8.3 Stage 1 实现

```python
@triton.jit
def _flash_decoding_stage1_kernel(
    Q, K, V, ..., Mid_O, Mid_O_LogExpSum, ...,
):
    batch_pid = tl.program_id(0)    # 哪个 batch
    head_pid = tl.program_id(1)     # 哪个 head
    seq_block_pid = tl.program_id(2) # 哪个 KV 分区
    
    # 计算当前分区的 KV 范围
    cur_batch_partition_start = seq_block_pid * BLOCK_SEQ
    cur_batch_partition_end = min(cur_batch_seq_len, start + BLOCK_SEQ)
    
    # 加载 Q（单个 token）
    q = tl.load(Q + batch_pid * q_bs_stride + head_pid * q_heads_stride + ...)
    
    # 初始化
    d_i = 0.0
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    # 迭代当前分区的所有 KV 块
    for start_n in range(0, num_blocks, 1):
        # 通过 b_req_tokens_table 间接索引 KV cache
        k_loc = tl.load(b_req_tokens_table + ...)       # 获取 KV cache 的真实位置
        k = tl.load(K + k_loc[:, None] * k_bs_stride + ...)
        v = tl.load(V + k_loc[:, None] * v_bs_stride + ...)
        
        # Q @ K^T（Q 是单个 token，K 是块）
        qk = tl.sum(q[None, :] * k, axis=1)            # (BLOCK_N,)
        qk *= qk_scale
        
        # Online softmax 更新
        current_max = tl.max(qk)
        m_ij = tl.maximum(m_i, current_max)
        p = tl.exp(qk - m_ij)
        
        alpha = tl.exp(m_i - m_ij)
        d_i = alpha * d_i + tl.sum(p, axis=0)
        acc = alpha * acc + tl.sum(p[:, None] * v, axis=0)  # (BLOCK_DMODEL,)
        m_i = m_ij
    
    # 如果当前分区有数据，写 mid_o 和 mid_logsumexp
    tl.store(Mid_O + ..., acc / d_i)
    tl.store(Mid_O_LogExpSum + ..., m_i + tl.log(d_i))
```

**与 FlashAttention 的区别**：
- Q 是单个 token（一维向量），不是矩阵
- `qk = tl.sum(q * k, axis=1)` 用逐元素乘 + 求和，而非 `tl.dot`
- 额外用了 `b_req_tokens_table` 间接索引 KV cache（因为 KV cache 可能是非连续分配的）
- 输出不是最终结果，而是 `(mid_o, mid_logsumexp)`，留给 Stage 2 归约

### 8.4 Stage 2 实现

```python
@triton.jit
def _flash_decoding_stage2_kernel(
    Mid_O, Mid_O_LogExpSum, Output, ..., B_Seqlen, ...
):
    batch_pid = tl.program_id(0)
    head_pid = tl.program_id(1)
    
    # 多少个分区需要归约
    num_partitions = (cur_batch_seq_len + BLOCK_SEQ - 1) // BLOCK_SEQ
    
    d_i = 0.0
    m_i = -float("inf")
    acc = tl.zeros([BLOCK_DMODEL], dtype=tl.float32)
    
    for block_seq_n in range(0, num_partitions, 1):
        part_v = tl.load(Mid_O + block_seq_n * mido_partitions_stride)   # mid_o[p]
        part_max = tl.load(Mid_O_LogExpSum + block_seq_n)                # logsumexp[p]
        
        # online softmax 归约
        m_ij = tl.maximum(part_max, m_i)
        alpha = tl.exp(m_i - m_ij)
        p = tl.exp(part_max - m_ij)
        acc = alpha * acc + p * part_v
        d_i = alpha * d_i + p
        m_i = m_ij
    
    # 最终结果
    tl.store(Output + ..., acc / d_i)
```

**归约的数学原理**：跟 online softmax 一样——`part_max`（= m_p + log(d_p)）作为归约的"分数"，`part_v`（= acc_p / d_p）作为归约的"值"。

---

## 9. 性能调优清单

### 9.1 BLOCK_SIZE 选择

```
规则：BLOCK_SIZE 应是 2 的幂，且在 64-256 之间

太小（< 32）：program 太多，调度开销占主导
太大（> 512）：SRAM 不够用导致 register spilling（溢出到 HBM）
```

实际操作：先设 128 或 256，用 Triton 的 `@triton.autotune` 自动搜索最优值。

### 9.2 Coalesced Access（合并访存）

```python
# 好的访存：连续地址（同一行的相邻列）
offsets = tl.arange(0, BLOCK_SIZE)         # [0, 1, 2, ..., 255]
x = tl.load(X + row * stride + offsets)    # 一次加载 256 个连续元素

# 不好的访存：跨步访存（访问不同行的同一列）
offsets = tl.arange(0, BLOCK_SIZE) * stride # [0, 2560, 5120, ...]
x = tl.load(X + col + offsets)              # 每读一个元素要跳到新地址
```

### 9.3 数值稳定性

```python
# 问题：大数 exp 会溢出
exp(1000) = inf  → 整个计算崩掉

# 解决：先用最大值做偏移
m = max(x)
safe_x = x - m          # 最大值变成 0，其他 < 0
exp(safe_x)             # 所有值在 (0, 1] 之间，不会溢出
```

### 9.4 pow2 vs pad

```
triton.next_power_of_2(N) 的使用场景：
- BLOCK_SIZE 通常设为 2 的幂，便于 GPU 内存对齐
- 对于不是 2 的幂的维度，用 mask 处理边界
```

### 9.5 精度取舍

```
经验法则：
- 矩阵乘法（Q@K^T）：fp32 累加，避免精度损失
- 逐元素操作（RMSNorm, SwiGLU）：中间用 fp32，结果转回 fp16
- exp/softmax：fp32（数值敏感的运算）
- RoPE：直接 fp16（旋转不放大误差）
```

---

## 10. 补充：矩阵乘法基础

Triton 的 `tl.dot` 是调用了 GPU 的 Tensor Core，能在一个时钟周期完成 4×4 的矩阵乘法。

```python
# tl.dot 的用法
a = tl.randn((64, 128))   # (M, K)
b = tl.randn((64, 128))   # (N, K) — 注意 K 在最后一个维度
c = tl.dot(a, b.T)        # (M, N) = (64, 64)

# 等价于 PyTorch 的：
# c = a @ b.T
```

**Tensor Core 的要求**：
- 维度最好是 16 或 32 的倍数
- 数据类型最好是 fp16/bf16（fp32 速度慢很多）

### 10.1 一个简单的矩阵乘法 kernel

```python
@triton.jit
def matmul_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 2D grid: (M/BLOCK_M, N/BLOCK_N) 个 program
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    # 当前 program 负责 C 的哪一块
    offs_m = pid_m * BLOCK_M + tl.arange(BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(BLOCK_N)
    offs_k = tl.arange(BLOCK_K)
    
    # 初始化累加器（在寄存器中）
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # 遍历 K 维度
    for k in range(0, K, BLOCK_K):
        # 加载 A 块和 B 块到 SRAM
        a = tl.load(A + offs_m[:, None] * stride_am + (k + offs_k[None, :]) * stride_ak,
                     mask=(offs_m[:, None] < M) & ((k + offs_k[None, :]) < K), other=0.0)
        b = tl.load(B + (k + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn,
                     mask=((k + offs_k[:, None]) < K) & (offs_n[None, :] < N), other=0.0)
        
        # Tensor Core 矩阵乘法
        acc += tl.dot(a, b)
    
    # 写回
    tl.store(C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))
```

这个 kernel 跟 FlashAttention 的 `tl.dot(q, k)` 是完全一样的机制——只是 FlashAttention 在此基础上叠加了 causal mask 和 online softmax。

---

## A. 推荐学习顺序

1. **先看懂 RMSNorm kernel**：逐元素操作，无跨元素依赖，最简单
2. **再看 SwiGLU kernel**：理解算子融合的意义
3. **然后 RoPE kernel**：理解向量化 load/store
4. **再看 FlashAttention**：理解分块 + online softmax，最核心的优化
5. **最后 FlashDecoding**：两级归约

每看一个 kernel 时，建议用极小的数据手算一遍（比如 BLOCK_SIZE=4，手动追踪 program 0 和 1 分别做了什么），这样比读代码快得多。

---

## B. 常用 Triton API 速查

| API | 作用 |
|---|---|
| `tl.program_id(axis)` | 当前 program 在 grid 中的索引 |
| `tl.arange(start, end)` | 生成向量 |
| `tl.load(ptr, mask=..., other=...)` | 从 HBM 加载 |
| `tl.store(ptr, val, mask=...)` | 写入 HBM |
| `tl.dot(a, b)` | Tensor Core 矩阵乘法 |
| `tl.sum(x, axis=...)` | 向量归约 |
| `tl.max(x, axis=...)` | 求最大值 |
| `tl.where(cond, a, b)` | 条件选择 |
| `tl.exp(x)` / `tl.math.exp2(x)` | 指数函数 |
| `tl.sigmoid(x)` | Sigmoid |
| `tl.zeros(shape, dtype)` | 零张量 |
| `tl.multiple_of(x, N)` | 编译器提示（x 是 N 的倍数） |
