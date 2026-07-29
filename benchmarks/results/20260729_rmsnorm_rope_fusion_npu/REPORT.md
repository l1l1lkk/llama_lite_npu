# RMSNorm + RoPE Triton 融合算子 NPU 测试报告

## 1. 项目目标与结论

本项目面向 Qwen3 Attention 的 Q/K 归一化与旋转位置编码链路，将原来的
Q RMSNorm、K RMSNorm、RoPE 三次 Triton launch 融合为一次 Triton launch，
并完成算子级、Profiler、数值与真实服务端到端验证。

测试结论如下：

- 融合后每步 kernel 数由 3 降为 1，减少 66.667%。
- 算子微基准在 FP16/BF16、11 组不同 batch/sequence/head dimension 下均有
  稳定正收益，延迟降低中位数分别为 72.775% 和 74.659%。
- NPU Profiler 显示原链路的主要问题是 scalar compute 占用高，以及小 kernel
  启动和 shape 展开导致的碎片化；不是 HCCL 通信或显式同步瓶颈。
- Qwen3-1.7B、TP=2 服务测试中，TPOT 降低 12.437%–13.419%，端到端耗时降低
  10.689%–11.227%；TTFT 基本持平，说明优化主要作用于逐 token 解码路径。
- FP16/BF16 共 22 个 case 全部通过容差校验，端到端生成结果的跨分支哈希一致。

## 2. 代码与版本隔离

| 用途 | 分支 | 性能测试代码提交 |
|---|---|---|
| 非融合基线 | `test/rmsnorm-rope-unfused-baseline` | `ffde6897bc41881fcc70016216a23366d3b3309e` |
| Triton 融合 | `feature/rmsnorm-rope-fused-optimization` | `cccbbbdebeb83e5a2b7d635b43845797643074c2` |

两个分支都从 `e058280` 建立独立 worktree，未修改主工作目录，也未修改原有
SwiGLU 分支。基线和融合分支暴露相同的 `qk_rmsnorm_rope_forward` 接口，使
Qwen3 模型接入、基准工具和服务启动参数保持一致，避免比较中混入接口差异。

## 3. 基线是什么

非融合基线复用项目已有算子：

1. `skip_rmsnorm(q, ...)`：对 Q 的每个 head 做 RMSNorm。
2. `skip_rmsnorm(k, ...)`：对 K 的每个 head 做 RMSNorm。
3. `rope_emb_forward(q, k, ...)`：读取归一化后的 Q/K 并执行 RoPE。

因此一次 Attention 前处理包含 3 次 Triton kernel launch。归一化后的 Q/K 会
先写回显存，再被 RoPE 重新读出。只计算 Q/K tensor 本身，读写流量约为
`4E`（RMSNorm 读写各一次、RoPE 读写各一次），其中 `E` 为 Q/K 元素总量。

## 4. 慢在哪里，Profiler 看到了什么

Profiler 使用物理卡 7，FP16，Q heads=16、K heads=4、head dimension=128，
每组采集 5 个 active step：

| 场景 | Shape | 基线 kernel/step | 融合 kernel/step | 基线设备时间/step | 融合设备时间/step | 降低 |
|---|---:|---:|---:|---:|---:|---:|
| Prefill | B=1, S=512 | 3 | 1 | 774.479 µs | 45.004 µs | 94.189% |
| Decode | B=32, S=1 | 3 | 1 | 54.645 µs | 4.612 µs | 91.560% |

基线中每 5 个 step 捕获到 10 个 `rms_norm_kernel` 和 5 个
`_triton_rope_emb`；融合后只捕获到 5 个 `_qk_rmsnorm_rope_kernel`。

PipeUtilization 的加权比例为：

| 场景 | 版本 | Scalar | MTE2 | MTE3 | Vector |
|---|---|---:|---:|---:|---:|
| Prefill | 基线 | 0.768 | 0.136 | 0.075 | 0.045 |
| Prefill | 融合 | 0.382 | 0.331 | 0.345 | 0.238 |
| Decode | 基线 | 0.808 | 0.097 | 0.071 | 0.046 |
| Decode | 融合 | 0.505 | 0.260 | 0.260 | 0.282 |

据此将瓶颈归类为：

- **主要瓶颈：计算和 shape/launch 碎片化。** 基线 Scalar 比例为
  0.768–0.808，且同一逻辑被拆成三个 kernel；融合后 scalar 比例显著下降，
  load/store 与 vector pipe 利用更均衡。
- **次要瓶颈：访存。** 基线需要落盘并重新读取 RMSNorm 中间结果；融合后
  中间值停留在寄存器中，Q/K tensor 流量从约 `4E` 降为 `2E`，减少 50%。
- **不是同步/通信瓶颈。** 匹配的算子采集中没有 HCCL kernel；收益来自单卡
  kernel 合并，而非通信重叠或同步规避。
- **存在 shape 上限问题。** 原 RMSNorm grid 按 `token × local_q_heads` 展开，
  Qwen3-1.7B、TP=2 的安全 prefill 上限为 7500 token；融合 kernel 改成每 token
  一个 program 后，将安全上限提升至 60000 token。

原始采集在 `profiler/npu-profiler-raw.tar.gz`，结构化结果在
`profiler/profiler-comparison.json`。

## 5. 做了什么融合与数据布局优化

融合 kernel 的 grid 为 `(batch × sequence,)`，每个 Triton program 同时完成
当前 token 的全部 Q heads 和 K heads：

1. 一次读取当前 position 的 cos/sin。
2. 以 `[head, half_dim]` 二维 tile 读取 Q/K 的前后半维。
3. 在 FP32 中完成平方和、RMS reduction 与 reciprocal sqrt。
4. 转回输入 dtype 后乘 RMSNorm weight。
5. 直接在寄存器中完成 RoPE 旋转并原位写回 Q/K。

head 数和 half dimension 使用 `next_power_of_2` padding 配合 mask，支持本次
测试中的 head dimension 64/128/256。Q、K、weight、cos、sin 在入口处保证
contiguous，使 stride 规则固定，避免不规则布局导致的额外寻址开销。

## 6. 算子延迟与 shape 稳定性

测试配置：物理卡 7，warmup=10，samples=30，固定随机种子 20260729。
Q heads=16、K heads=4。

### 6.1 FP16

| 维度族 | Shape | 基线 P50 | 融合 P50 | 延迟降低 | 加速比 |
|---|---|---:|---:|---:|---:|
| Batch | B=1,S=1,D=128 | 0.543 ms | 0.151 ms | 72.109% | 3.585x |
| Batch | B=32,S=1,D=128 | 0.532 ms | 0.147 ms | 72.345% | 3.616x |
| Sequence | B=1,S=512,D=128 | 0.991 ms | 0.229 ms | 76.835% | 4.317x |
| Sequence | B=1,S=2048,D=128 | 3.381 ms | 0.366 ms | 89.186% | 9.247x |
| Head dim | B=4,S=128,D=64 | 0.857 ms | 0.178 ms | 79.253% | 4.820x |
| Head dim | B=4,S=128,D=256 | 1.004 ms | 0.228 ms | 77.340% | 4.413x |

11 个 shape 的延迟降低范围为 71.244%–89.186%，中位数为 72.775%；
融合延迟 CV 中位数为 1.298%，最大为 2.796%。

### 6.2 BF16

| 维度族 | Shape | 基线 P50 | 融合 P50 | 延迟降低 | 加速比 |
|---|---|---:|---:|---:|---:|
| Batch | B=1,S=1,D=128 | 0.584 ms | 0.150 ms | 74.310% | 3.893x |
| Batch | B=32,S=1,D=128 | 0.579 ms | 0.156 ms | 72.973% | 3.700x |
| Sequence | B=1,S=512,D=128 | 1.035 ms | 0.222 ms | 78.541% | 4.660x |
| Sequence | B=1,S=2048,D=128 | 3.365 ms | 0.351 ms | 89.582% | 9.599x |
| Head dim | B=4,S=128,D=64 | 0.870 ms | 0.176 ms | 79.812% | 4.954x |
| Head dim | B=4,S=128,D=256 | 1.021 ms | 0.223 ms | 78.185% | 4.584x |

11 个 shape 的延迟降低范围为 72.973%–89.582%，中位数为 74.659%。
首次正式运行中 B=4、S=1 出现 20.932% CV，因此对全部 BF16 shape 独立复测；
复测最大 CV 为 4.674%，中位数为 1.244%，确认收益稳定。首次结果和确认结果
均保留在 `microbenchmark/`，未删除异常样本。

覆盖矩阵：

- Batch：1、4、16、32（S=1，D=128）。
- Sequence length：16、128、512、2048（B=1，D=128）。
- Head dimension：64、128、256（B=4，S=128）。

## 7. 端到端 TTFT/TPOT

服务配置：Qwen3-1.7B，TP=2，物理卡 6、7，continuous batching，NPU Graph
关闭，最大序列长度 1024，temperature=0，输出 32 token。每个 case 运行 3 次，
每次启动全新服务并只发送 1 个请求，结果取均值。

| Prompt | 指标 | 基线 | 融合 | 变化 |
|---|---|---:|---:|---:|
| 128 token | TTFT | 313.444 ms | 311.480 ms | 降低 0.626% |
| 128 token | TPOT | 88.614 ms | 77.594 ms | 降低 12.437% |
| 128 token | E2E | 3060.491 ms | 2716.889 ms | 降低 11.227% |
| 128 token | 输出吞吐 | 10.453 tok/s | 11.777 tok/s | 提升 12.664% |
| 512 token | TTFT | 663.616 ms | 666.618 ms | 上升 0.452% |
| 512 token | TPOT | 87.380 ms | 75.655 ms | 降低 13.419% |
| 512 token | E2E | 3372.393 ms | 3011.916 ms | 降低 10.689% |
| 512 token | 输出吞吐 | 9.488 tok/s | 10.623 tok/s | 提升 11.962% |

TTFT 在 ±0.7% 范围内，不能声称有确定提升；TPOT 的 12%–13% 改善与融合算子
在每层、每个 decode token 上重复执行的预期一致。

### 7.1 端到端限制与异常处理

当前服务调度器在一次生命周期的首请求结束后，会出现
`waiting=1, running=0` 且后续请求不再推进的问题。为避免把调度器故障混入
融合算子比较，正式数据采用“每请求一个全新服务生命周期”的隔离协议。因此
本报告能证明单请求 TTFT/TPOT 与输出一致性，不能替代多请求并发吞吐验收。

首次融合 p128 请求包含新 shape 的 Triton 编译，TPOT 为 179.856 ms；该值不
属于稳态，未计入正式均值，但完整保留在 `e2e/e2e-diagnostics.tar.gz`。正式
p128 使用已完成编译后的三次独立服务结果。正式逐次日志、Git HEAD、NPU
状态、Prometheus 快照和请求 timing 均在 `e2e/e2e-formal-raw.tar.gz`。

## 8. 数值误差如何验证

算子级参考路径是非融合的 Q RMSNorm + K RMSNorm + RoPE。每个 shape 使用同一
随机输入、weight、cos/sin，比较融合输出 Q/K：

- FP16：`atol=0.03, rtol=0.03`。
- BF16：`atol=0.15, rtol=0.05`。
- 同时记录 max absolute error、max relative error、mean absolute error 和
  `torch.allclose`。

结果：

| Dtype | 通过 | Q 最大绝对误差 | K 最大绝对误差 | Q 最大平均绝对误差 | K 最大平均绝对误差 |
|---|---:|---:|---:|---:|---:|
| FP16 | 11/11 | 0.001953125 | 0.001953125 | 1.612e-8 | 2.142e-8 |
| BF16 | 11/11 | 0.125 | 0.0625 | 1.977e-3 | 1.858e-3 |

端到端使用 temperature=0，并保存输出 SHA-256。所有正式重复的分支内输出一致，
基线与融合分支输出哈希也完全一致。

## 9. 测试门禁

- 基线分支：相关单元测试 **7 passed**。
- 融合分支：相关单元测试 **9 passed**。
- FP16 正确性：11/11 passed。
- BF16 正确性：11/11 passed，另做一次全矩阵确认。
- 正式端到端：2 个 prompt 长度 × 2 个分支 × 3 次重复，输出一致。

测试日志位于 `tests/`。算子、Profiler、比较和服务测试脚本位于
`benchmarks/rmsnorm_rope/`，可以在相同容器环境复现。

## 10. 证据索引

| 文件 | 内容 |
|---|---|
| `microbenchmark/comparison-fp16.json` | FP16 每 shape 的基线/融合对比 |
| `microbenchmark/comparison-bf16.json` | BF16 每 shape 的基线/融合对比 |
| `microbenchmark/fused-bf16-confirmation.json` | BF16 稳定性复测 |
| `profiler/profiler-comparison.json` | kernel 数、设备时间、PipeUtilization |
| `profiler/npu-profiler-raw.tar.gz` | 四组完整 NPU Profiler 原始目录 |
| `e2e/comparison.json` | TTFT、TPOT、E2E、吞吐与输出一致性 |
| `e2e/e2e-formal-raw.tar.gz` | 正式端到端逐次原始证据 |
| `e2e/e2e-diagnostics.tar.gz` | 首次编译异常值和调度器诊断证据 |
| `tests/*.log` | 两分支单元测试结果 |

