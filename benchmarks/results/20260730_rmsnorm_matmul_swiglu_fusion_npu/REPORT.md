# MatMul / SwiGLU / RMSNorm Triton 融合与 NPU Graph 验证报告

## 结论

本轮验证的真实连续数据流是：

`Skip-RMSNorm -> packed gate/up MatMul -> SwiGLU`

在 Ascend 910B3、Qwen3-1.7B、TP=2 上，三算子 Triton 单 kernel **不应合入生产路径**。
它把 batch=1 的算子中位延迟从 `0.6204 ms` 增加到 `3.2259 ms`
（负收益 `-419.98%`）。NPU Graph 能把分开链的 TPOT 从
`76.80 ms` 降到 `6.29 ms`，但只能把融合链从 `92.50 ms` 降到
`90.36 ms`。最佳方案是 **保留 CANN MatMul + Triton RMSNorm/SwiGLU
的分开实现，并开启 NPU Graph**。

这不是“融合没有减少 kernel 数”：算子 Profiler 明确显示每步从 3 个
kernel 降为 1 个。负收益来自单 kernel 内部低效的 decode GEMV 和重复
RMS 归约，属于设备端数据搬运/标量流水问题，不是同步问题。

## 测试对象与边界

- 服务器：`218.106.157.54:10013`
- 容器：`triton-llama-lite`
- 设备：算子测试/Profiler 使用物理 NPU 7；E2E TP=2 使用 NPU 6、7
- NPU：Ascend 910B3
- 模型：`/data/liuke/llama_lite_npu/my_weight/Qwen3-1.7B`
- 精度：FP16
- 模型宽度：hidden size 2048，TP 后 intermediate size 3072
- E2E：并发 1，输出 32 tokens，p128/p512，各 3 次重复，每次 4 请求

没有把 `SwiGLU -> down MatMul -> 下一层 RMSNorm` 当作可融合边界。
在 TP=2 下，down projection 后存在 `tp_all_reduce`/HCCL；跨越该通信
做单 kernel 融合会改变并行语义。

## 三种方案

1. 分开/eager：Triton Skip-RMSNorm + CANN packed MatMul + Triton
   packed SwiGLU。
2. 融合/eager：一个 Triton kernel 完成 residual add、RMSNorm、两路
   packed GEMV 和 SwiGLU；仅用于 `sequence_length=1`、batch 不超过 16
   的 decode，prefill 保持分开路径。
3. 融合/Graph：同一个 Triton kernel 被 decode NPU Graph 捕获和 replay。

为分离 Graph 自身收益，额外测了“分开/Graph”，形成四象限。

## 算子延迟与 batch 稳定性

统计口径：NPU 7，10 次 warmup，50 次同步计时，中位数。

| Batch | 分开链 (ms) | Triton 单 kernel (ms) | 融合收益 |
|---:|---:|---:|---:|
| 1 | 0.6204 | 3.2259 | -419.98% |
| 2 | 0.6128 | 3.2050 | -423.05% |
| 4 | 0.6070 | 3.2074 | -428.42% |
| 8 | 0.6080 | 3.2112 | -428.18% |

融合延迟在 batch 1–8 间只波动约 0.65%，因此结果“稳定”，但稳定地是
负收益。该边界不含 attention head，head dimension 不参与 kernel
索引；当前模型 head dimension=128，不能把 head dimension 当作本算子
的独立变量。sequence length 128/512 走明确的 prefill fallback，因而
TTFT 不期望从该 decode-only kernel 获益。

## NPU Profiler 看到什么

算子级捕获：物理 Card 7、单进程、Stream 47、batch=1、5 个 active
steps、Profiler Level1、PipeUtilization、L2 cache。

| 实现 | 每步 kernel | 平均设备时间 |
|---|---|---:|
| 分开 | `skip_rms_norm_kernel` | 2.008 μs |
| 分开 | `aclnnMatmul_MatMulCommon_MatMulV2` | 20.404 μs |
| 分开 | `_swiglu_packed_kernel` | 1.856 μs |
| 融合 | `_rmsnorm_matmul_swiglu_decode_kernel` | 2982.848 μs |

分开链的三个 device kernel 合计约 `24.268 μs`；融合 kernel 约
`2982.848 μs`，设备执行时间约为前者的 122.9 倍。

融合 kernel 的关键流水指标：

- AIC MAC ratio：`0.005`，即约 0.5%；
- AIV MTE2 ratio：`0.8918`，即约 89.18%；
- AIV scalar ratio：`0.7180`，即约 71.80%；
- 48 个 N tile，每个 tile 为得到自己的输出列块都会重新读取输入并
  计算 RMS 统计量；
- BLOCK_N=256 会在编译期报 UB overflow，需要 2,097,152 bit，而设备
  可用 1,572,864 bit；最终可编译配置为 BLOCK_N=64。

因此瓶颈分类是：

- **主要：访存/数据布局与重复归约**；
- **次要：Triton 标量/向量流水，decode GEMV 未达到 CANN MatMul 效率**；
- **不是：HCCL 同步**（该边界内部无 collective）；
- **不是：shape 抖动**（batch 1–8 延迟稳定）；
- eager 分开链另有明显 host launch gap，这一部分正是 NPU Graph
  擅长解决的。

完整算子 Profiler 压缩包见
`raw/operator-profiler-full.tar.gz`，可直接解压后用 MindStudio Insight
打开；汇总见 `raw/profiler/operator-profiler-summary.json`。

## E2E 四象限结果

### p128 / 输出 32 tokens

| 实现 | Graph | TTFT (ms) | TPOT (ms) | E2E (ms) |
|---|---|---:|---:|---:|
| 分开 | off | 84.788 | 76.803 | 2465.682 |
| 分开 | on | 85.566 | 6.291 | 280.584 |
| 融合 Triton | off | 85.262 | 92.501 | 2952.801 |
| 融合 Triton | on | 88.101 | 90.363 | 2889.360 |

### p512 / 输出 32 tokens

| 实现 | Graph | TTFT (ms) | TPOT (ms) | E2E (ms) |
|---|---|---:|---:|---:|
| 分开 | off | 462.089 | 76.608 | 2836.936 |
| 分开 | on | 461.282 | 6.866 | 674.132 |
| 融合 Triton | off | 462.656 | 93.104 | 3348.873 |
| 融合 Triton | on | 466.092 | 91.046 | 3288.523 |

关键收益：

- 分开链 Graph on vs off：
  - p128：TPOT `+91.81%`，E2E `+88.62%`；
  - p512：TPOT `+91.04%`，E2E `+76.24%`。
- Triton 融合 Graph on vs off：
  - p128：TPOT `+2.31%`，E2E `+2.15%`；
  - p512：TPOT `+2.21%`，E2E `+1.80%`。
- Triton 融合 eager vs 分开 eager：
  - p128：TPOT `-20.44%`，E2E `-19.76%`；
  - p512：TPOT `-21.53%`，E2E `-18.05%`。
- “融合 + Graph”对“分开 + eager”的联合结果仍为负：
  - p128：TPOT `-17.66%`，E2E `-17.18%`；
  - p512：TPOT `-18.85%`，E2E `-15.92%`。

TTFT 基本不变或轻微变慢是预期行为：本实现只替换 decode
`sequence_length=1`，prefill 明确走分开 fallback；当前 NPU Graph 也只
捕获 decode。

## NPU Graph 捕获证据

四象限正式区间中，Graph on 的两种实现都满足：

- captures：2；
- replays：930；
- fallbacks：0。

动态 msprof 的融合分支 p128 单请求显示：

| 指标 | Graph off | Graph on | 减少 |
|---|---:|---:|---:|
| FftsPlusTaskLaunch count | 1826 | 90 | 95.07% |
| FftsPlusTaskLaunch time | 14444.70 μs | 907.83 μs | 93.72% |
| EventRecord count | 5478 | 270 | 95.07% |
| EventRecord time | 45864.96 μs | 2854.42 μs | 93.78% |
| ContextGetCurrent count | 1826 | 90 | 95.07% |

Graph 确实减少了运行时发射与事件 API，但融合 kernel 的设备计算仍在，
所以 TPOT 只能改善约 2.2%。这验证了：Graph 能解决调度/同步开销，
不能修复单 kernel 内的低效访存和 GEMV。

## 数值验证

- 算子矩阵 batch 1/2/4/8 全部通过
  `torch.allclose(atol=5e-2, rtol=5e-2)`；
- 融合输出最大绝对误差：`0.00390625`；
- 新 residual 最大绝对误差：`0`；
- E2E p128/p512 的 12/12 请求在以下比较中输出 SHA256 100% 一致：
  - 分开 Graph on/off；
  - 融合 Graph on/off；
  - 融合 vs 分开（Graph off）；
  - 融合 vs 分开（Graph on）。

## 建议

1. 生产路径保持三算子分开，并开启 decode NPU Graph。
2. 不合入当前 MatMul/RMSNorm/SwiGLU Triton 单 kernel；该分支作为负向
   实验和 Profiler 证据保留。
3. 如果继续研究，应优先使用 CANN/AscendC 的高性能 GEMV primitive，
   或支持跨 kernel producer-consumer locality 的编译器融合，而不是在
   每个 N tile 内重复 RMS 归约。
4. 不跨 `tp_all_reduce` 融合 down MatMul 与下一层 RMSNorm。

## 原始数据

- `raw/micro/baseline.json`
- `raw/micro/fused.json`
- `raw/e2e/comparison.json`
- `raw/e2e/comparison.csv`
- `raw/profiler/operator-profiler-summary.json`
- `raw/profiler/e2e-graph-off/*.csv`
- `raw/profiler/e2e-graph-on/*.csv`
- `raw/e2e-results-full.tar.gz`
- `raw/operator-profiler-full.tar.gz`
- `raw/SHA256SUMS`

服务器保留的完整动态 msprof（约 953 MiB）位于：

`/data/liuke/rmsnorm_matmul_swiglu_20260730/e2e/profiler`
