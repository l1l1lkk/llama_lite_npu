# Qwen3-32B TP=2 FP16 输入/输出长度二维性能矩阵

## 结论

本轮 36/36 个正式 run 全部通过严格门禁：每个 run 32 个请求，输入长度严格等于 128/512/1024/2048，输出长度严格等于 64/256/512，失败请求为 0；正式区间 Graph capture 增量和 fallback 均为 0，replay 均持续增长。结果构成当前服务器、模型及配置下的机器可读回归基线，不是跨硬件 SLA。

在固定输出长度下，prompt 从 128 增长到 2048 时，mean TTFT 从约 1.15 s 非线性增加到约 159.2 s。p1024→p2048 每增加 1K input token 的经验差分斜率约 133.25–133.37 s，显著高于较短区间；增长同时包含 queue wait 和首 token 服务时间，不能解释为单一 kernel 的线性 prefill 成本。固定 prompt 时，输出增长提高闭环测试中的 output throughput，但延长 E2E 并增加 KV pages；这是固定并发、固定请求数下批处理与摊销共同作用的观测，不能外推为单请求 decode 随长度变快。

## 测试契约与漂移控制

- 模型：Qwen3-32B，2×Atlas 910B3，物理 NPU `6,7`，TP=2，FP16。
- 服务：NPU Graph on、continuous batching、`--no_decode_priority`、page size 16、max batch size 32、max sequence length 4096。
- workload：concurrency=4、greedy、seed=42、offset=0；prompt 为 128/512/1024/2048，output 为 64/256/512，`min_tokens=max_tokens=output`。
- 每个 cell 三个独立 server lifecycle；每个 lifecycle 8 个 warmup 请求、32 个 formal 请求，共 36 个正式 run。
- 36 个 lifecycle 的 prompt/output/repeat 顺序在采集前写入 `campaign-plan.json`，使用平衡交错顺序，完成后未按结果改序。
- 四个 prompt 长度分别冻结 formal/warmup JSONL；同一 prompt 下三个 output cell 复用同一 formal 文件、seed、offset 和行顺序。并发完成导致 EvalScope SQLite rowid 顺序不稳定，因此门禁同时验证冻结 JSONL SHA（提交顺序）与实际观察到的 prompt token 多重集合（内容一致），不把完成顺序误称为提交顺序。

formal JSONL SHA256：

| Prompt | SHA256 |
|---:|---|
| 128 | `11b2c0d603619092f7d2a35eae53cafc5cb32d4e7d642a429e29a41758a428ba` |
| 512 | `59ac796e7874c56f4a4791dba0a4d5962b74ecad38bc523c43db8d3de63c884c` |
| 1024 | `7c96b2bc824a94733278f4f31dbd46179ccb93d6e621ff50133a7acff6bdb3fc` |
| 2048 | `f4e9a1a15a07bc6ac5eaa2977ee0e3ba0d3940a5720fbb2727812fe8d20930e7` |

## 二维核心结果

下表为每个 cell 三轮 mean。TTFT、Queue、Service→1st、TPOT、ITL 单位为 ms，E2E 为 s，吞吐为 token/s。`Service→1st` 是服务端直接记录的非排队首 token 时间，不是 profiler kernel 分解。

| Prompt | Output | TTFT | Queue | Service→1st | TPOT | ITL | E2E | Output tok/s | Total tok/s | KV pages mean | HBM mean % |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 64 | 1,153.22 | 306.01 | 839.87 | 50.28 | 49.63 | 4.321 | 59.218 | 177.655 | 203.5 | 89 |
| 128 | 256 | 1,152.52 | 302.63 | 839.84 | 49.50 | 49.40 | 13.774 | 74.333 | 111.499 | 243.6 | 89 |
| 128 | 512 | 1,153.06 | 301.22 | 839.61 | 52.25 | 52.54 | 27.849 | 73.532 | 91.916 | 283.6 | 90 |
| 512 | 64 | 10,956.06 | 2,610.57 | 8,334.76 | 96.09 | 94.94 | 17.010 | 15.048 | 135.433 | 798.0 | 90 |
| 512 | 256 | 10,958.50 | 2,613.17 | 8,334.99 | 68.66 | 68.50 | 28.468 | 35.967 | 107.902 | 836.0 | 90 |
| 512 | 512 | 10,958.91 | 2,610.61 | 8,334.42 | 67.10 | 67.39 | 45.247 | 45.261 | 90.521 | 877.7 | 90 |
| 1024 | 64 | 22,643.63 | 10,383.25 | 12,251.46 | 490.02 | 484.20 | 53.515 | 4.783 | 81.308 | 1,578.8 | 90 |
| 1024 | 256 | 22,635.13 | 10,378.27 | 12,246.78 | 176.37 | 175.99 | 67.610 | 15.143 | 75.717 | 1,614.4 | 90 |
| 1024 | 512 | 22,645.20 | 10,382.13 | 12,250.54 | 128.16 | 129.01 | 88.136 | 23.234 | 69.702 | 1,658.8 | 90 |
| 2048 | 64 | 159,091.43 | 72,829.49 | 86,248.48 | 3,051.23 | 3,003.46 | 351.319 | 0.729 | 24.046 | 3,158.6 | 90 |
| 2048 | 256 | 159,200.85 | 72,881.76 | 86,304.81 | 831.40 | 829.04 | 371.207 | 2.758 | 24.826 | 3,178.1 | 90 |
| 2048 | 512 | 159,215.68 | 72,888.75 | 86,311.46 | 469.24 | 472.21 | 398.998 | 5.133 | 25.663 | 3,206.1 | 90 |

三轮 sample stdev、CV、请求级 p50/p90/p99、QPS、input throughput、scheduler/NPU/HBM 分位数和 Graph counter 均在 `length-matrix-aggregate.csv`；逐 run 值在 `length-matrix-run-metrics.csv`。TTFT CV 为 0.006%–0.272%，output throughput CV 为 0.044%–0.446%。

## Prompt 与 output scaling

固定 output 时，三个相邻 prompt 区间的 TTFT 每增加 1K input token 经验差分斜率分别约为：

- p128→p512：25.53 s/1K；
- p512→p1024：22.81–22.83 s/1K；
- p1024→p2048：133.25–133.37 s/1K。

该斜率仅为本矩阵相邻点差分，不是线性模型。p2048 的 TTFT 中 queue wait 约 72.9 s、service-to-first-token 约 86.3 s，两部分都显著增长，因此证据不支持把退化只归因于排队或只归因于 prefill。

固定 prompt 时，从 o64→o256、o256→o512 的每 256 output token E2E 增量随 prompt 增长，分别约为：p128 12.60/14.08 s、p512 15.28/16.78 s、p1024 18.79/20.53 s、p2048 26.52/27.79 s。KV pages 随输出长度单调增加；HBM 采样只有整数百分比且基本为 89%–90%，粒度不足以推断小幅显存差异。完整差分见 `prompt-scaling.csv` 与 `output-scaling.csv`，二维机器数据见 `length-matrix-heatmap.csv/json`。

## Task4 sanity reference

Task4 的 p128/o256/c4/`--no_decode_priority` 只作跨 campaign sanity，不并入本轮均值：TTFT 差异 -0.09%，TPOT +0.47%，output throughput -0.42%，均小于 10%。这支持环境和到达口径没有明显漂移，但不把两个 campaign 合并统计。

## 回归基线规则

`performance-baseline.json` 固化 commit、VERSION、模型 config SHA、软件/硬件/环境/workload 指纹以及 12 个 cell 的 mean/stdev/CV 和正确性/Graph 门禁。仅当 fingerprint 完全匹配时比较性能：throughput 下降或 TTFT/TPOT 上升超过 `max(10%, 3×baseline CV)` 标记 regression；strict correctness、失败请求或 Graph fallback 为 hard fail；fingerprint 不匹配为 invalid，不能伪装成性能回归。

当前 baseline 自比较为 pass；单测同时覆盖 pass、regression、hard fail 和 invalid 分支。阈值是本服务器经验门禁，不是跨硬件 SLA。

## Watchdog 异常与结论边界

初始 order=2 的 r1-p2048-o512 使用了不合适的 2400 秒总墙钟 watchdog，在 completed request、generated token、Graph replay 和 AICore 均持续增长时被截断。该 lifecycle 被永久标为 rejected diagnostic，partial warmup/formal 数据未进入 36 个正式 run、aggregate、baseline 或任何比较；服务器原证据在 `server-only-rejected/insufficient-watchdog-r1-p2048-o512`，索引在 `diagnostics-index.json` 和 manifest omitted 清单。

经批准后，剩余序列统一锁定为 progress-v2：只有连续 1200 秒无 completed/output-token/replay 增长且同时存在停滞证据才判 stall；21600 秒仅触发人工复核，不自动终止仍在推进的 run。order=2 从头重跑，参数、请求数和预注册顺序未变。正式 36 run 未出现 OOM、strict failure、Graph fallback 或真实调度停滞。

本报告只适用于所列模型、checkpoint、软件栈、Graph/调度配置、闭环到达、c4 和固定长度 workload；不外推其他并发、开放到达、其他模型、Graph off 或 vLLM-Ascend。内部 Graph 消融也不是 vLLM 对比。

## 复现与离线验证

- Git-tracked bundle：`benchmarks/results/20260714_qwen3_32b_tp2_fp16_length_matrix/`
- 正式 run：36；bundle 总文件 1,222（manifest 内容索引 1,221 个），索引内容 38,340,805 bytes。
- server-only omitted：543 个文件，122,476,001 bytes；路径、大小和 SHA256 均记录在 `campaign-manifest.json`，离线复算不依赖它们。

```bash
python benchmarks/qwen3_32b_tp2_fp16/validate_bundle.py \
  benchmarks/results/20260714_qwen3_32b_tp2_fp16_length_matrix
python benchmarks/qwen3_32b_tp2_fp16/analyze_length_matrix.py --compare \
  benchmarks/results/20260714_qwen3_32b_tp2_fp16_length_matrix
python benchmarks/qwen3_32b_tp2_fp16/compare_performance_baseline.py \
  benchmarks/results/20260714_qwen3_32b_tp2_fp16_length_matrix/performance-baseline.json \
  benchmarks/results/20260714_qwen3_32b_tp2_fp16_length_matrix/performance-baseline.json \
  --expect pass
```
