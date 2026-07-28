# Ascend NPU SwiGLU 融合优化与测试报告

## 1. 结论

本项目在两条隔离分支上完成了未融合基线与融合优化：

- 未融合基线：`test/swiglu-unfused-baseline`，提交
  `ce55d4b08d179a84c2885432ff004ee202b6e324`；
- 融合优化：`feature/swiglu-fused-optimization`，生产实现提交
  `c95ed1d7db8d0fd500185b659caba223989b86ed`，测试与证据工具继续演进至
  `fdf206aee6cf0ebce4e8ae0771d90496aa579160`。

在物理 NPU 7 上，FP16 14 个 shape 全部通过数值门禁，算子延迟降低
31.45%–57.49%，几何平均加速 1.99 倍。NPU Profiler 显示，单次调用由
3 个 Cast、1 个 Sigmoid、2 个 Mul 共 6 个 kernel 收敛为 1 个原生
`SwiGlu` kernel；代表 prefill shape 的纯设备时间降低 58.50%，decode
shape 降低 46.94%。

Qwen3-1.7B、TP=2、物理 NPU 6/7 的正式端到端对比中：

| Prompt | TTFT | TPOT | E2E | 输出吞吐 |
|---:|---:|---:|---:|---:|
| 128 | 88.558 → 80.491 ms，-9.11% | 76.716 → 72.548 ms，-5.43% | 2466.752 → 2329.486 ms，-5.56% | 12.971 → 13.738 tok/s，+5.92% |
| 512 | 464.723 → 463.192 ms，-0.33% | 76.268 → 71.711 ms，-5.98% | 2829.043 → 2686.226 ms，-5.05% | 11.311 → 11.911 tok/s，+5.31% |

长 prompt 的 TTFT 主要由 attention 和线性投影主导，因此 512-token
TTFT 只改善 0.33%；decode 每层都会执行 SwiGLU，TPOT 改善在两种 prompt
下保持在 5.4%–6.0%。

## 2. 可直接用于工作履历的项目描述

**项目：Ascend 910B3 上 Qwen3 SwiGLU 融合与端到端性能优化**

负责在双 Atlas 910B3 环境中建立 SwiGLU 未融合/融合双分支基线，设计
batch、sequence length、特征宽度、FP16/BF16 的可复现算子矩阵，并使用
torch-npu Profiler 定位 eager 路径的 6 次 kernel 下发和中间张量访存
开销。针对原 Triton 单行大块策略在 25,600 宽度下 UB overflow、在
6,144 宽度下性能不佳的问题，将 Dense Qwen3 的 Gate/Up 权重按 TP 分片后
打包为单个 `gate_up_proj`，通过一次线性投影直接生成 `[gate, up]` 连续
布局，再调用 CANN 原生 `npu_swiglu` 完成单 kernel 融合。最终 FP16
14 个 shape 的算子延迟降低 31.45%–57.49%，Profiler 设备时间在 prefill/
decode 代表 shape 下分别降低 58.50%/46.94%；Qwen3-1.7B TP=2 端到端
TPOT 降低 5.43%–5.98%，E2E 降低约 5%，并通过真实模型权重的 Gate/Up、
SwiGLU、Down 数值对齐和两分支回归测试。

## 3. 基线是什么

基线函数严格按以下 eager 链路执行：

```python
a_fp32 = a.float()
b_fp32 = b.float()
output = a_fp32 * torch.sigmoid(a_fp32) * b_fp32
return output.to(a.dtype)
```

选择 FP32 中间计算是为了与原 Triton 路径的数值契约一致，避免把降低精度
误算成性能优化。基线分支没有修改原工作区，单独位于：

```text
本地：D:\lite_llama_npu-worktrees\swiglu-unfused-baseline
服务器：/data/liuke/llama_lite_npu-worktrees/swiglu-unfused-baseline
GitLab：test/swiglu-unfused-baseline
```

## 4. 慢在哪里，Profiler 看到了什么

匹配的 profiler shape：

- prefill：`[batch=4, sequence=128, feature=6144]`；
- decode：`[batch=16, sequence=1, feature=6144]`；
- Profiler Level1，`PipeUtilization`，开启 L2 cache，5 个 active step。

| 阶段 | 版本 | 每 step kernel | 设备时间/step | 主要 Pipe 特征 |
|---|---|---:|---:|---|
| Prefill | 未融合 | 6 | 43.353 us | MTE2 0.619，访存占主导 |
| Prefill | 融合 | 1 | 17.993 us | Vector 0.584，MTE2 降至 0.468 |
| Decode | 未融合 | 6 | 16.278 us | Scalar 0.593，小 kernel/调度占主导 |
| Decode | 融合 | 1 | 8.636 us | 单 kernel 后主要剩余输入搬运 |

基线每 step 的精确 kernel 构成为 3×Cast、2×Mul、1×Sigmoid。Prefill
需要反复读写 FP32 中间张量，属于访存问题；decode 数据量小，六次小 kernel
的 scalar 与下发开销更突出。Profiler 中没有通信 kernel，也没有跨流等待
证据，因此瓶颈不是 HCCL 同步。

原项目 Triton 实现还存在 shape 问题：它按一行一个 program，并将 25,600
列扩到 32,768 的 power-of-two block。在 Atlas 910B3 上编译时报：

```text
ub overflow, requires 2621440 bits while 1572864 bits available
```

完整日志保存在 `source-fused/microbenchmark-fp16.log`。2-D Triton tile
256/512/1024/2048/4096 均通过数值验证，但 backend sweep 仍显示 CANN
packed 路径更快，因此没有把“可编译”误当作最终优化。

## 5. 做了什么融合和数据布局优化

1. Dense Qwen3 的 `gate_proj`、`up_proj` 权重在 TP 分片之后沿输出维打包，
   形成 `gate_up_proj.weight = cat([gate, up], dim=0)`。
2. Forward 从两次独立 `F.linear` 改为一次宽输出线性投影。
3. 投影输出天然为最后一维连续的 `[gate, up]` 布局，不在热路径执行
   runtime `cat`。
4. 直接调用 `torch_npu.npu_swiglu(packed, dim=-1)`，在一个 CANN kernel
   内完成 split、SiLU 和乘法。
5. 为仍传入独立 Gate/Up tensor 的其他模型保留兼容入口；Dense Qwen3
   走无额外拷贝的 packed 专用入口。

这既减少 Gate/Up GEMM 下发次数，也消除了 SwiGLU 中间张量落 HBM 和多次
kernel 调度。

## 6. 算子延迟与 shape 稳定性

### 6.1 Batch sweep，S=1，D=6144，FP16

| Batch | 基线 mean | 融合 mean | 延迟降低 | 融合 CV |
|---:|---:|---:|---:|---:|
| 1 | 0.1182 ms | 0.0515 ms | 56.40% | 3.27% |
| 4 | 0.1093 ms | 0.0519 ms | 52.54% | 2.77% |
| 16 | 0.1104 ms | 0.0507 ms | 54.12% | 2.98% |
| 32 | 0.1162 ms | 0.0560 ms | 51.77% | 3.74% |

### 6.2 Sequence sweep，B=1，D=6144，FP16

| Sequence | 基线 mean | 融合 mean | 延迟降低 | 融合 CV |
|---:|---:|---:|---:|---:|
| 16 | 0.1103 ms | 0.0507 ms | 54.05% | 2.30% |
| 128 | 0.1452 ms | 0.0843 ms | 41.93% | 4.94% |
| 512 | 0.1704 ms | 0.1119 ms | 34.33% | 4.41% |
| 2048 | 0.3264 ms | 0.1439 ms | 55.92% | 3.52% |

### 6.3 Head-dimension-sized proxy，B=4，S=128，FP16

SwiGLU 本身没有 attention head 轴，实际计算轴是 FFN intermediate
dimension。因此这里按用户要求使用 64/128/256/512 的
head-dimension-sized feature proxy，同时另测真实模型宽度 6144/25600。

| Feature width | 基线 mean | 融合 mean | 延迟降低 | 融合 CV |
|---:|---:|---:|---:|---:|
| 64 | 0.1101 ms | 0.0513 ms | 53.43% | 5.10% |
| 128 | 0.1176 ms | 0.0500 ms | 57.49% | 3.72% |
| 256 | 0.1128 ms | 0.0491 ms | 56.44% | 2.49% |
| 512 | 0.1164 ms | 0.0570 ms | 51.02% | 2.51% |

真实模型宽度下：

- `[4,128,6144]`：0.1624 → 0.1113 ms，降低 31.45%；
- `[1,128,25600]`：0.1659 → 0.1081 ms，降低 34.85%，且不再 UB
  overflow。

融合分支 14 个 FP16 shape 的 CV 为 2.30%–5.10%，未出现随 batch、
sequence 或 feature width 增长而失效的 shape。

## 7. 端到端 TTFT/TPOT

正式范围：

- 模型：Qwen3-1.7B，FP16；
- 设备：物理 NPU 6、7，TP=2；
- Graph：off；
- serving：legacy request-at-a-time；
- workload：冻结的精确 128/512 prompt token；
- 每个 cell：1 个 warmup，4 个 formal request，3 次 repeat；
- 输出：`min_tokens=max_tokens=32`，greedy；
- 总计：每分支 24 个 formal request，失败 0。

最初 continuous-batching 生命周期完成 2 个 warmup 后出现 scheduler
waiting 请求无法继续 admission。该生命周期没有形成正式结果，完整保留在
`e2e/baseline` 作为 rejected diagnostic，未进入汇总。正式对比改用两分支
完全一致的 legacy 生命周期，避免把调度器异常混入 SwiGLU 因果结论。

128-token TTFT 改善明显，因为 prefill 较短，MLP 优化在首 token 路径中的
占比更高。512-token TTFT 基本持平，说明长 prefill 的主耗时已经转移到
attention/大矩阵投影；TPOT 在两种 prompt 下均稳定改善。

## 8. 数值误差如何验证

数值门禁分三层：

1. **算子级随机矩阵**：固定 seed `20260728`，FP32
   `a * sigmoid(a) * b` 为参考，覆盖 14 个 shape。
   FP16 使用 `atol=rtol=0.002`，14/14 通过；最大绝对误差
   `0.00390625`，大于 `1e-2` 参考值区域的最大相对误差小于
   `0.000976`。BF16 14/14 通过，本轮输出与参考量化结果逐元素相同。
2. **真实模型权重路径**：读取 Qwen3-1.7B 第 0 层权重，按 TP=2 rank 0
   分片，对比“两次 Linear + eager SwiGLU + Down”与“一次 packed Linear
   + CANN SwiGLU + Down”。`[1,1,2048]`、`[4,1,2048]`、
   `[1,128,2048]` 均通过 `atol=rtol=0.02`；activation/Down 最大绝对
   误差 `0.00390625`，余弦相似度约为 1。
3. **端到端 greedy 输出哈希**：每个 repeat 内输出完全稳定。跨分支
   128-token workload 为 4/4 byte-identical；512-token 为 3/4
   byte-identical。唯一差异是一个近似等价的生成分支从 “parse it”
   变为 “parse this”，之后文本分叉。这符合接近决策边界时微小浮点误差
   影响 greedy token 的现象。

因此算子与真实层级数值门禁通过，但如果业务门禁要求所有生成文本
byte-for-byte 完全一致，则本优化在 512-token 样本上不满足该更严格门禁，
不能隐瞒为“完全无差异”。

## 9. 测试与代码质量

- 未融合分支相关回归：58/58 通过；
- 融合分支相关回归：60/60 通过；
- FP16 算子矩阵：14/14；
- BF16 算子矩阵：14/14；
- 真实 Qwen3 模型路径：3/3；
- 正式端到端：两分支各 24/24 请求成功；
- 两分支均通过 `compileall`、`git diff --check`；
- 测试结束后 8228 端口关闭，无本任务 server/client 残留。

## 10. 复现

算子矩阵：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=7 python -m benchmarks.swiglu.benchmark \
  --device npu:0 --physical-device 7 --dtype float16 \
  --output /data/liuke/swiglu-results/microbenchmark-fp16.json
```

Profiler：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python -m benchmarks.swiglu.profile \
  --device npu:0 --physical-device 7 --shape 4x128x6144 \
  --worker-name swiglu-prefill \
  --output-dir /data/liuke/swiglu-results/profiler-prefill
```

Profiler 汇总：

```bash
python -m benchmarks.swiglu.analyze_profiler \
  --baseline-prefill <baseline-prefill> \
  --baseline-decode <baseline-decode> \
  --fused-prefill <fused-prefill> \
  --fused-decode <fused-decode> \
  --output profiler-summary.json
```

真实模型路径：

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python -m benchmarks.swiglu.validate_model_path \
  --checkpoint /data/liuke/llama_lite_npu/my_weight/Qwen3-1.7B/Qwen3-1.7B.pth \
  --device npu:0 --physical-device 7 --tp-size 2 --tp-rank 0 --layer 0 \
  --output model-path-validation.json
```

端到端 server 生命周期使用 `start_e2e_server.sh` 和
`stop_e2e_server.sh`，客户端使用 `e2e_client.py`，聚合使用
`compare_e2e.py`。所有完整命令、server log、request-level JSON、
Profiler CSV/JSON/DB 均已包含在本交付目录中。

## 11. 结论边界

本报告只适用于所记录的 Atlas 910B3、CANN 8.5、Torch/torch-npu 2.7.1、
Triton 3.2.0、Qwen3-1.7B FP16、TP=2、Graph off 环境。算子 shape
矩阵支持判断融合实现的稳定性；端到端数字不应外推到 Graph on、
continuous batching、其他模型或其他并发度。
