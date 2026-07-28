# Triton SwiGLU NPU fusion evidence

本目录是未融合 SwiGLU 与自定义 Triton SwiGLU 融合实现的完整对比证据。

- `REPORT.md`：中文闭环报告与履历表述。
- `baseline/`：未融合分支微基准与 NPU Profiler 原始输出。
- `fused-triton/`：Triton 融合分支微基准与 NPU Profiler 原始输出。
- `tuning/`：tile 扫描和被否决的数据布局实验。
- `e2e-legacy/`：Qwen3-1.7B、TP=2、NPU 6/7 的逐请求端到端数据。
- `comparison-*.json/csv`：FP16、BF16 shape 对比。
- `profiler-summary.json`：匹配窗口的设备 kernel 汇总。
- `e2e-comparison.json/csv`：TTFT、TPOT、E2E 与吞吐对比。
- `model-path-validation.json`：真实 Qwen3 权重路径数值校验。
- `environment.json`：软硬件、模型、分支和测试契约。
- `manifest.json`：原始证据文件 SHA256 与大小清单。

服务器原始目录：

```text
/data/liuke/swiglu-triton-fusion-20260728
```

本地交付目录和归档：

```text
D:\lite_llama_npu-artifacts\swiglu-triton-fusion-20260728
D:\lite_llama_npu-artifacts\swiglu-triton-fusion-20260728.tar.gz
D:\lite_llama_npu-artifacts\swiglu-triton-fusion-20260728.tar.gz.sha256
```

归档摘要以同级 `.sha256` 文件为准；校验文件保存在归档外，避免归档内容
与归档自身摘要形成循环依赖。
