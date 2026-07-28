# Ascend NPU Triton SwiGLU 融合测试报告

## 1. 项目结论

本次工作将 Qwen3 Dense MLP 的正式 SwiGLU 路径改为自定义 Triton
kernel。实现不是 `torch_npu.npu_swiglu` 的包装：生产文件中通过
`@triton.jit` 定义 `_swiglu_packed_kernel`，模型先把 `gate_proj` 和
`up_proj` 权重打包为一个 `gate_up_proj`，一次 Linear 生成连续的
`[gate, up]` 半区，随后由二维 Triton grid 在一个 kernel 中完成
FP32 SiLU 和逐元素乘法。

结论必须分成设备侧和完整调用侧：

- NPU Profiler 中，Prefill 设备执行时间从 43.5688 us 降至
  24.1124 us，降低 44.66%，1.807x；Decode 从 16.0080 us 降至
  2.9720 us，降低 81.43%，5.386x。
- 每步 NPU kernel 从 6 个降为 1 个，减少 83.33%。
- 但当前 eager Python/Triton launch 路径存在显著 host dispatch 开销。
  14 个 FP16 shape 的 p50 几何平均 speedup 只有 0.629x，即整体延迟
  约为基线的 1.589 倍；只有长序列 `B=1,S=2048,D=6144` 获得
  14.30% 的 p50 降低。
- Qwen3-1.7B TP=2 端到端中，TTFT 小幅改善，但 TPOT 和总延迟轻微
  回退。因此该版本证明了 Triton 设备 kernel 融合有效，但尚未达到
  端到端性能准入条件。

这份结果没有把纯设备时间改善包装成端到端收益；正式履历应表述为
“完成 Triton 融合实现、定位 host launch 瓶颈并给出下一阶段优化方向”。

## 2. 分支与隔离

两条分支均从同一源代码检查点建立独立 worktree，不修改原工作目录：

| 角色 | 分支 | 本地 worktree | 服务器 worktree |
|---|---|---|---|
| 未融合基线 | `test/swiglu-unfused-baseline` | `D:\lite_llama_npu-worktrees\swiglu-unfused-baseline` | `/data/liuke/llama_lite_npu-worktrees/swiglu-unfused-baseline` |
| Triton 融合 | `feature/swiglu-fused-optimization` | `D:\lite_llama_npu-worktrees\swiglu-fused-optimization` | `/data/liuke/llama_lite_npu-worktrees/swiglu-fused-optimization` |

未融合基线提交为
`ce55d4b08d179a84c2885432ff004ee202b6e324`。最终融合分支 SHA 以交付
时本地、服务器和 origin 的三方核对结果为准。

## 3. 测试环境

- 服务器：`218.106.157.54:10013`。
- 实际容器：`triton-llama-lite`。
- 单算子与 Profiler：物理 NPU 7。
- 端到端：物理 NPU 6、7，Tensor Parallel=2。
- NPU：Ascend Atlas 910B3。
- PyTorch：2.7.1；torch-npu：2.7.1；Triton：3.2.0。
- 模型：Qwen3-1.7B，FP16，28 层，hidden size 2048，
  intermediate size 6144，16 heads，head dimension 128。
- 端到端关闭 compiled model，使用 legacy request-at-a-time 调度，
  concurrency=1，输出 32 tokens。

SwiGLU 没有 attention head 轴。报告中的 64/128/256/512 是与常见
head dimension 等宽的 feature proxy；6144 和 25600 才是真实 FFN
中间维度。

## 4. 基线是什么

未融合分支执行：

```python
gate_fp32 = gate.float()
output = (
    gate_fp32
    * torch.sigmoid(gate_fp32)
    * up.float()
).to(gate.dtype)
```

Profiler 每个 active step 包含：

- Cast 3 个；
- Mul 2 个；
- Sigmoid 1 个；
- 合计 6 个 NPU kernel。

该基线保留两次独立的 Gate/Up Linear 和 FP32 中间张量，作为未融合
对照。

## 5. 慢在哪里，Profiler 看到了什么

### 5.1 未融合 Prefill

匹配 shape 为 `[4,128,6144]`：

- 设备时间 43.5688 us/step；
- MTE2 weighted ratio 0.6217，为占比最高的流水；
- 分类为 memory access；
- 多次 Cast/Mul/Sigmoid 反复读写中间张量。

因此 Prefill 基线主要是访存和多 kernel 中间数据搬运问题。

### 5.2 未融合 Decode

匹配 shape 为 `[16,1,6144]`：

- 设备时间 16.0080 us/step；
- scalar weighted ratio 0.5756；
- 分类为 scalar compute/small-kernel；
- 6 个很短的 kernel 放大了调度和启动开销。

Decode 不是通信或 HCCL 同步问题，而是小 kernel 数量与标量执行问题。

### 5.3 Triton 融合后

| 阶段 | 基线设备时间 | Triton 设备时间 | 降低 | speedup | kernel/step |
|---|---:|---:|---:|---:|---:|
| Prefill | 43.5688 us | 24.1124 us | 44.66% | 1.807x | 6 → 1 |
| Decode | 16.0080 us | 2.9720 us | 81.43% | 5.386x | 6 → 1 |

融合 kernel 的 dominant pipe 变为 vector compute：

- Prefill AIV vector ratio 0.6846；
- Decode AIV vector ratio 0.5949。

这说明设备侧已经从多次访存/标量小算子转为一个向量化 Triton kernel。
Profiler 中没有 HCCL/通信 kernel，排除同步和通信瓶颈。

## 6. Triton 实现与数据布局

### 6.1 正式 kernel

kernel 使用二维 grid：

```text
grid = (row_count, ceil(feature_width / BLOCK_SIZE))
```

每个 program：

1. 从连续 gate 半区加载一个 tile；
2. 从连续 up 半区加载对应 tile；
3. 将 gate 提升到 FP32；
4. 执行 `gate * sigmoid(gate) * up`；
5. 一次写回输出。

宽 shape 不再把整行 padding 到 next power of two。最大 tile 为 8192，
因此 25600 宽度拆成多个 program，不会生成 32768 大块。

### 6.2 权重与投影布局

Tensor Parallel 分片后执行：

```text
gate_up_proj.weight = cat([gate_proj.weight, up_proj.weight], dim=0)
```

一次 Linear 直接输出 `[gate, up]` 连续半区，正式 Dense Qwen3 路径
不发生运行时 `cat`。兼容的双输入 API 仍可工作，但会显式支付一次
packing 成本。

### 6.3 为什么不是原仓库 Triton kernel

原 kernel 一行只启动一个 program，并把宽度向上取整为 2 的幂。
在 25600 宽度下会申请 32768 block，实测编译失败：

```text
ub overflow, requires 2621440 bits while 1572864 bits available
```

二维 tile 设计消除了这个 shape 上限。

## 7. 调优闭环

### 7.1 Tile 扫描

在 256、512、1024、2048、4096、8192、16384 之间扫描：

- Decode `[16,1,6144]` 的独立双指针候选在 512/4096 附近较快；
- Prefill `[4,128,6144]` 在 8192 为 0.2554 ms；
- 宽 shape `[1,128,25600]` 在 8192 为 0.2545 ms；
- 16384 相比 8192 没有稳定收益。

生产 packed 单指针路径又单独复测。尝试按行数切换 4096/8192 后，
Decode wall-time 反而从约 0.175 ms 增至约 0.182 ms，因此撤销
row-aware 策略，最终使用最大 8192 的统一策略。该结果说明候选 kernel
的 tile 结论不能不经验证直接迁移到不同输入布局。

### 7.2 被否决的交错布局

还测试了：

```text
[gate0, up0, gate1, up1, ...]
```

它允许一维 grid，但在 Ascend 上产生 stride=2 的输入读取。长序列
`[1,2048,6144]` 增至约 23.6 ms，真实模型宽度约 6.2 ms，远差于
连续半区布局，因此撤销。原始数据保存在
`tuning/interleaved-layout-smoke.json`。

## 8. 跨 shape 微基准

配置：30 samples、10 warmup，FP16/BF16，NPU 7。比较同时保留 mean、
p50、p90、p99 和 CV。共享服务器偶发单点抖动，因此稳定延迟以 p50
为主，CV/p99 用于观察尾部。

### 8.1 FP16

- 14/14 shape 数值通过；
- p50 几何平均 speedup：0.629x；
- p50 speedup 范围：0.526x～1.167x；
- FP16 最大绝对误差：0.00390625；
- 唯一明确正收益为 `B=1,S=2048,D=6144`：
  0.3326 → 0.2850 ms，降低 14.30%。

代表结果：

| Shape | 基线 p50 | Triton p50 | 变化 |
|---|---:|---:|---:|
| `[1,1,6144]` | 0.1074 ms | 0.1794 ms | +66.95% |
| `[16,1,6144]` | 0.0986 ms | 0.1751 ms | +77.56% |
| `[1,128,6144]` | 0.1377 ms | 0.2129 ms | +54.64% |
| `[1,512,6144]` | 0.1733 ms | 0.2422 ms | +39.79% |
| `[1,2048,6144]` | 0.3326 ms | 0.2850 ms | -14.30% |
| `[4,128,6144]` | 0.1704 ms | 0.2440 ms | +43.22% |
| `[1,128,25600]` | 0.1747 ms | 0.2430 ms | +39.11% |

这里的“+”表示延迟回退，“-”表示延迟降低。

### 8.2 BF16

- 14/14 shape 数值通过；
- 本次 BF16 输出与量化后的 FP32 参考完全一致，最大绝对误差为 0；
- p50 几何平均 speedup：0.665x；
- p50 最大 speedup 1.144x，同样出现在长序列场景。

### 8.3 稳定性

绝大多数 FP16 case 的 CV 小于 3.4%。共享环境中少量 case 出现单个
p99 异常值，使 CV 达到 14%～23%，但对应 p50/p90 仍集中。报告同时
保留原始 30 个样本统计，不删除异常值。

结论是：

- 数值和 shape 覆盖稳定；
- 设备 kernel 对长序列有效；
- 小 batch、小 feature 和 Decode 的完整调用延迟不满足性能准入；
- 下一阶段应降低 Triton host launch/dispatch 开销，而不是继续只优化
  设备算术。

## 9. 端到端 TTFT/TPOT

模型 Qwen3-1.7B，TP=2，NPU 6/7。每个分支：

- p128、p512 两个固定 workload；
- 每种 workload 重复 3 次；
- 每次 4 个正式请求和 1 个 warmup；
- 共 24 个正式请求；
- 0 失败。

| Case | 指标 | 基线 | Triton | 变化 |
|---|---|---:|---:|---:|
| p128 | TTFT | 83.876 ms | 82.615 ms | 改善 1.50% |
| p128 | TPOT | 75.402 ms | 76.466 ms | 回退 1.41% |
| p128 | E2E | 2421.334 ms | 2453.077 ms | 回退 1.31% |
| p128 | throughput | 13.218 tok/s | 13.048 tok/s | 回退 1.29% |
| p512 | TTFT | 463.658 ms | 461.795 ms | 改善 0.40% |
| p512 | TPOT | 74.251 ms | 74.960 ms | 回退 0.96% |
| p512 | E2E | 2765.437 ms | 2785.569 ms | 回退 0.73% |
| p512 | throughput | 11.570 tok/s | 11.487 tok/s | 回退 0.72% |

设备侧单 kernel 的收益没有转化为 TPOT 正收益，主要证据链是：

1. Profiler 设备时间已经显著降低；
2. 微基准 wall-time 在小 shape 仍回退；
3. Decode 每层都经过 Python/Triton launch；
4. 28 层累积后抵消设备 kernel 节省。

因此当前主要瓶颈已经从设备 kernel 内部转移到 host dispatch/框架集成。

## 10. 数值误差与输出一致性

单算子参考为 FP32：

```text
float(gate) * sigmoid(float(gate)) * float(up)
```

再量化回目标 dtype。门禁为：

```text
atol=0.002, rtol=0.002
```

结果：

- FP16 14/14 通过，最大绝对误差 0.00390625；
- BF16 14/14 通过，本次最大绝对误差 0；
- FP16 在 `|reference| > 1e-2` 区域的最大相对误差低于约 0.001。

真实 Qwen3 第一层、TP rank0 的 Gate/Up、SwiGLU 和 Down 路径分别测试
`[1,1,2048]`、`[4,1,2048]`、`[1,128,2048]`：

- activation 和 down projection 全部通过 `atol=rtol=0.02`；
- 最大绝对误差 0.00390625；
- cosine similarity 约为 1。

端到端文本：

- p128：4/4 跨分支完全一致；
- p512：3/4 完全一致；
- 每个分支内部三次重复均确定性一致。

p512 的一次措辞变化意味着严格逐字节输出门禁未全部通过，但数值门禁
全部通过。

## 11. 测试门禁

服务器最终代码执行：

- 基线相关测试：58/58 通过；
- Triton 融合相关测试：60/60 通过；
- `python -m compileall` 通过；
- `git diff --check` 通过；
- FP16/BF16 微基准 28/28 shape 通过；
- 真实模型路径 3/3 通过；
- 端到端两个分支各 24 个正式请求，0 失败。

合同测试明确检查：

- 正式包选择 `swiglu_fused.py`；
- 文件导入 Triton；
- 存在 `_swiglu_packed_kernel`；
- 正式实现中不存在 `npu_swiglu` 调用。

## 12. 下一阶段优化建议

当前版本不应宣称端到端加速完成。建议按以下顺序继续：

1. 将 Triton kernel 注册为 `torch.library` custom op，减少 Python
   wrapper 和动态参数构造；
2. 验证 compiled model/静态图捕获，比较 graph on/off 的 launch 成本；
3. 为 Decode 开发 packed 单指针的多行 tile，而不是直接照搬双指针
   候选的 row-aware 参数；
4. 用 Host Self Time、ACL API 时间和设备时间做三段分解；
5. 只有 TPOT/E2E 在多轮测试中稳定为正后，才把该 Triton 版本升级为
   性能生产方案。

## 13. 可用于履历的准确表述

> 在 Ascend 910B3 上实现自定义 Triton SwiGLU 融合算子，将 Qwen3
> Gate/Up 权重打包为单投影并以二维 UB-safe tile 融合 FP32 SiLU 与
> 逐元素乘法，解决 25,600 宽度下原单行 Triton kernel 的 UB overflow。
> NPU Profiler 显示 kernel 数由 6 降至 1，Prefill/Decode 设备时间分别
> 降低 44.66%/81.43%；完成 FP16/BF16 28 组 shape、真实模型权重路径和
> TP=2 端到端验证。进一步定位到 eager Triton host dispatch 抵消设备
> 收益，当前 TPOT 回退 0.96%～1.41%，据此提出 custom op/静态图集成的
> 下一阶段方案。

这段表述同时包含成果和限制，不应改写为“端到端已提升”。
