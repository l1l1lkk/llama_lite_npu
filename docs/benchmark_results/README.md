# Benchmark 结果索引

当前主 benchmark campaign 统一遵循 `benchmarks/qwen3_32b_tp2_fp16/README.md`。发布数字的权威结构化复现包位于 `benchmarks/results/<campaign>/`，随代码一起跟踪。SQLite、HTML 和完整日志等大文件可以只保留在服务器，但其路径、大小和 SHA256 必须记录在 bundle manifest 中。

`docs/inference_performance_history.md` 中的历史结果早于当前 artifact 契约。只有版本、checkpoint、精度、Graph 模式、采样、workload 和指标边界都一致时，才允许与当前结果计算加速比。

当前 campaign：

- [`20260711_qwen3_32b_tp2_fp16.md`](20260711_qwen3_32b_tp2_fp16.md)
- [`20260711_qwen3_32b_tp2_fp16_aggregate.csv`](20260711_qwen3_32b_tp2_fp16_aggregate.csv)
- [`Git-tracked reproducibility bundle`](../../benchmarks/results/20260711_qwen3_32b_tp2_fp16/)

任务1固定输出长度验证：

- [`min_tokens` 固定输出长度机制与严格 workload 验证](20260712_min_tokens_fixed_output.md)
- [`Git-tracked strict workload bundle`](../../benchmarks/results/20260712_qwen3_32b_tp2_fp16_min_tokens/)

任务2 NPU Graph 严格消融：
- [`Qwen3-32B TP=2 FP16 NPU Graph 严格消融`](20260712_qwen3_32b_tp2_fp16_graph_ablation.md)
- [`Git-tracked Graph ablation bundle`](../../benchmarks/results/20260712_qwen3_32b_tp2_fp16_graph_ablation/)
