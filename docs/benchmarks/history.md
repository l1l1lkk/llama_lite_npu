# Benchmark 跨版本与跨框架历史

本页是唯一历史索引，不重新抄写旧表中的性能数字。数字及其限制以原报告和 Git bundle 为准。

## Historical strict old-code

以下 campaign 在各自旧提交上通过了当时的严格门禁，但不是 `v0.0.15rc2` 当前基线：

| Campaign | 证据 | 限制 |
| --- | --- | --- |
| 20260712 min_tokens | [报告](../benchmark_results/20260712_min_tokens_fixed_output.md) / [bundle](../../benchmarks/results/20260712_qwen3_32b_tp2_fp16_min_tokens/) | 旧版本固定输出正确性 |
| 20260712 Graph A/B | [报告](../benchmark_results/20260712_qwen3_32b_tp2_fp16_graph_ablation.md) / [bundle](../../benchmarks/results/20260712_qwen3_32b_tp2_fp16_graph_ablation/) | 内部 Graph 严格配对，不是跨框架比较 |
| 20260713 并发扩展 | [报告](../benchmark_results/20260713_qwen3_32b_tp2_fp16_graph_scalability.md) / [bundle](../../benchmarks/results/20260713_qwen3_32b_tp2_fp16_graph_scalability/) | 旧代码、Lite Llama 单框架 |
| 20260713 Decode Priority | [报告](../benchmark_results/20260713_qwen3_32b_tp2_fp16_decode_priority_ablation.md) / [bundle](../../benchmarks/results/20260713_qwen3_32b_tp2_fp16_decode_priority_ablation/) | 旧代码、内部调度消融 |
| 20260714 长度矩阵 | [报告](../benchmark_results/20260714_qwen3_32b_tp2_fp16_length_matrix.md) / [bundle](../../benchmarks/results/20260714_qwen3_32b_tp2_fp16_length_matrix/) | 旧代码、c4、Lite Llama 单框架 |

## Historical unverified

- [20260711 初始基线](../benchmark_results/20260711_qwen3_32b_tp2_fp16.md)：部分 run 提前 EOS，缺统一 strict validation，Graph on/off 请求数也未完全配对。
- [vLLM-Ascend 手工记录](../vllm_ascend_benchmark.md)：缺统一冻结 JSONL、当前环境指纹和 Git-tracked raw bundle；长输出历史 case 未固定到目标长度。
- [旧推理性能记录](../inference_performance_history.md)：混合了 `benchmark_tp.py`、EvalScope random 和 prefix custom client，只能逐条按原口径阅读。
- [`benchmark.md`](../benchmark.md)、[`benchmark_models.md`](../benchmark_models.md)、[`benchmark_models_history.md`](../benchmark_models_history.md)：上游 CUDA/ROCm 学习阶段资料，不代表当前 Ascend serving contract。

## Diagnostic

服务器 screening、warmup、rejected lifecycle、完整日志和 profiler 不进入正式 aggregate。若需要引用，只能链接对应 diagnostic/omitted 索引并说明失败或排查目的。

## 当前版本状态

`v0.0.15rc2` 尚无 strict current campaign。下一步应先验收 canonical v2 本地 harness，再单独执行两个框架的 capability probe；不得用上述历史数字补齐当前矩阵。
