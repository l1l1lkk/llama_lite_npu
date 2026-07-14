# Benchmark 执行手册

本文汇总 Qwen3-32B、2×Atlas 910B3 基线建设任务1至任务5的统一执行方法。所有性能结论必须来自 Git-tracked reproducibility bundle，不能只保留人工表格，也不能依赖服务器上可能被清理的 SQLite、HTML 或完整日志。

## 固定契约

- 记录 branch、git SHA、VERSION、checkpoint/config SHA、FP16、TP=2、NPU `6,7`、CANN/PyTorch/torch_npu/triton-ascend/EvalScope 精确版本。
- 固定 chat template、page size、max batch、max sequence length、scheduler、continuous batching、Graph/Decode Priority 开关、seed、offset 和请求顺序。
- 固定输出使用 `min_tokens=max_tokens`；必须逐请求验证实际 input/output token 长度和 success，不能只看平均值。
- 每个正式 run 使用独立 server lifecycle；warmup/capture 在正式边界前完成。正式区间要求 Graph capture delta=0、fallback=0、replay>0。
- 原始指标区分 client 与 server。Client TTFT/TPOT/ITL/E2E/throughput/QPS 来自 EvalScope；server queue wait、server TTFT、request timing、scheduler/KV/Graph 来自服务端。两者不能无说明地替换。
- 不采集或无法可靠采集的 percentile、continuous batch size 或 profiler 分解必须明确缺失，不能推测或伪造。

## Quick gate

Quick gate 用于普通代码变更的提交前检查或每日冒烟，不替代完整发布验证。

1. 运行相关单测、`scripts/validate_release.py`、UTF-8 和 `git diff --check`。
2. 启动 Qwen3-32B、TP=2、FP16、Graph-on、continuous batching、`--no_decode_priority`。
3. 使用冻结 p128 workload，concurrency=4、output=256、warmup=8、formal=32；至少一个独立正式 run。
4. hard gate：32/32 success、strict input=128、strict output=256、0 failed、formal capture=0、fallback=0、replay>0、fingerprint 匹配。
5. 将 candidate 与任务5 machine-readable baseline 比较。环境或 workload fingerprint 不匹配为 invalid；正确性或 Graph 失败为 hard fail；性能越过 `max(10%, 3×baseline CV)` 为 regression。

单轮 quick gate 适合发现明显回退，但不能替代三轮方差。触发 warning/regression 后应运行对应 cell 三轮，确认不是短窗口噪声。

## Full gate

Full gate 用于发布候选、推理路径或调度配置变更，以及 quick gate 报警后的系统复核。

| 模块 | 矩阵 | 目的 |
|---|---|---|
| 固定输出正确性 | p128/o256，c1/c4，三轮 | 验证 `min_tokens`、逐请求 EOS mask、TP 一致性 |
| Graph A/B | p128/o256，c1/c4，on/off 各三轮 | 严格配对验证 Graph 收益与计数 |
| 并发扩展 | p128/o256，c1/2/4/8/16，各三轮 | 定位吞吐饱和、TTFT 与 queue wait |
| Decode Priority A/B | p128/o256，c1/2/4/8/16，on/off 各三轮 | 验证 admission gate、公平性和 decode 代价 |
| 长度矩阵 | p128/512/1024/2048 × o64/256/512，c4，各三轮 | 建立 TTFT、decode、吞吐、KV/HBM 二维回归面 |

这些都是本项目内部实现/配置消融。Graph on/off 不是与 vLLM 或 vLLM-Ascend 的对比；跨框架比较必须另建同环境、同 workload、同指标边界的 campaign。

## 何时运行

- API、采样停止条件或 token 语义变化：固定输出正确性 + quick gate；必要时全长度矩阵。
- Graph capture/replay、算子或编译路径变化：Graph A/B + quick gate；不得用旧 campaign 补 pair。
- scheduler、continuous batching、admission 或 Decode Priority 变化：并发扩展 + Decode Priority A/B。
- KV cache、page size、max context、prefill/decode kernel 或模型模板变化：完整长度矩阵。
- 纯文档变化：单测、release validator、链接/UTF-8/diff-check；无需制造新性能数据。
- 发布候选：完整 full gate，或明确记录未运行模块及风险接受人。

## 原始证据与留存

每个正式 run 的 Git bundle 至少保留：

- `run-metadata.json`、`run-timing.json`、EvalScope args/summary/percentile、exit code、精确命令；
- 逐请求 client metrics 与 prompt token fingerprint；
- server before/after metrics 和 stats、精确 request timing、Graph counters；
- 一秒 scheduler/KV/Graph 时间序列及安全可得的低频 NPU/HBM 采样；
- server lifecycle 启停命令、health、环境快照、campaign plan 和冻结 workload；
- aggregate/scaling/paired/baseline 等机器生成结果，以及可离线重建的 compare 命令；
- `campaign-manifest.json` 的 included SHA/size；SQLite、HTML、完整 stdout/server log、profiler 等 omitted 文件的服务器原路径、大小和 SHA256。

服务器大文件是可选留存层，不是报告复算依赖。Git bundle 必须在脱离 `benchmark-results/` 的临时目录中通过 SHA 校验和全部 `--compare`。

## 可比性判定

只有以下项目一致才允许计算 ratio、speedup 或回归：模型和 checkpoint/config SHA、精度、TP、硬件、软件栈、代码语义、Graph/Decode Priority、模板、prompt/output token 长度、请求数、并发、seed/offset、prompt token 序列、warmup 和正式边界。

- 版本、硬件或 fingerprint 不同：标为 invalid/sanity reference，不计算回归结论。
- workload 匹配但正确性、failed request 或 Graph fallback：hard fail，不能用性能数字掩盖。
- 三轮数据：报告 mean、sample stdev、CV 和请求级 p50/p90/p99，不用总体平均掩盖 cell 差异。
- 历史 campaign 只能作 sanity reference，不能并入本轮均值。
- 经验斜率和 regression 阈值只适用于当前服务器测量面，不是线性外推或跨硬件 SLA。

## 推荐命令

```bash
python benchmarks/qwen3_32b_tp2_fp16/validate_bundle.py \
  benchmarks/results/<campaign>
python benchmarks/qwen3_32b_tp2_fp16/validate_strict_workload.py \
  benchmarks/results/<campaign> --compare
python benchmarks/qwen3_32b_tp2_fp16/analyze_length_matrix.py \
  benchmarks/results/<length-matrix-campaign> --compare
python benchmarks/qwen3_32b_tp2_fp16/compare_performance_baseline.py \
  benchmarks/results/<campaign>/performance-baseline.json \
  benchmarks/results/<campaign>/performance-baseline.json --expect pass
```

任务1至任务5的逐项中文报告和 Git bundle 入口位于 [`benchmark_results/README.md`](benchmark_results/README.md)。
