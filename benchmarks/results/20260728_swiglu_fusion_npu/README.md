# SwiGLU NPU 融合优化交付包

本目录是 2026-07-28 在 Atlas 910B3 上完成的 SwiGLU 基线、优化、
Profiler、数值与端到端验证闭环。

建议从 [REPORT.md](REPORT.md) 开始阅读。主要机器可读证据：

- `comparison-fp16.json/csv`：FP16 算子矩阵对比；
- `comparison-bf16.json/csv`：BF16 算子矩阵对比；
- `profiler-summary.json`：匹配 shape 的 NPU Profiler 派生结论；
- `model-path-validation.json`：真实 Qwen3 权重的 Gate/Up、SwiGLU、
  Down 数值对齐；
- `e2e-comparison.json/csv`：TTFT、TPOT、E2E 与吞吐对比；
- `baseline/`、`fused/`：微基准和完整 NPU Profiler 原始输出；
- `tuning/`：Triton tile 与 CANN backend 选型数据；
- `source-fused/`：原 Triton kernel 在 25,600 宽度触发 UB overflow 的
  诊断证据；
- `e2e/`：未纳入正式汇总的 continuous-batching 卡住诊断；
- `e2e-legacy/`：正式端到端逐请求数据；
- `environment.json`：软硬件、分支、模型与测试契约；
- `manifest.json`：服务器采集证据的 SHA256 和大小清单。

服务器原始目录为：

```text
/data/liuke/swiglu-fusion-20260728
```

本地完整副本和压缩包为：

```text
D:\lite_llama_npu-artifacts\swiglu-fusion-20260728
D:\lite_llama_npu-artifacts\swiglu-fusion-20260728.tar.gz
```

压缩包 SHA256：

```text
d311c2da88404a004fc058ee65bf0aa0cc1c21132c22da82a86c90014e5d7b5c
```
