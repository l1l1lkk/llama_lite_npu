# Qwen3-32B TP=2 FP16 Graph-on 并发扩展与排队分析

## 结论

在固定 `prompt=128`、`min_tokens=max_tokens=256`、32 条冻结请求的 Graph-on workload 中，15/15 个正式 run 均满足 strict input/output、0 failed、正式区间 capture 增量为 0、fallback 为 0 且 replay 增长。

本轮高 TTFT 主要由服务端排队造成。并发 2/4/8/16 的 server queue wait 分别占 server TTFT 的 96.12%/92.53%/83.96%/75.60%。代码路径进一步给出直接解释：`decode_priority=True` 且已有 decode 请求时，调度器把 `admission_capacity` 置为 0，后到的请求必须等待当前 decode wave 排空后才能 prefill。时间序列与此一致：c2 的峰值为 waiting=1、running=1；c4/c8/c16 的 running 峰值分别为 3/7/14，均有请求留在 waiting wave。

预先约定的工程饱和判据首次在 c1→c2 命中：output throughput 仅增加 0.46%，同时 mean TTFT 增长 2437.19%、mean queue wait 增长约 220.8 万%。这不是硬件吞吐上限；c2→c4、c4→c8、c8→c16 的 throughput 仍分别增长 79.92%、66.15%、33.48%。它表明当前闭环到达时序与 decode-priority admission gate 形成了阶梯式扩展曲线，c2 是第一个工程退化点，而不是全局吞吐峰值。

## 测试契约与环境

- 模型：Qwen3-32B；2×Atlas 910B3，物理 NPU `6,7`；TP=2；执行 FP16。
- 服务：OpenAI chat completions streaming，NPU Graph on，continuous batching，page size 16，max batch size 32，decode priority，max seq len 4096，未启用 chunked prefill。
- 软件：CANN 26.0.rc1、PyTorch 2.7.1、torch_npu 2.7.1、triton-ascend 3.2.0、EvalScope 1.7.1；精确版本、环境变量、checkpoint 和模型配置 SHA 位于 bundle 的 `environment/`。
- workload：同一份 32 条正式 JSONL，服务端实际 input=128，固定 output=256，greedy（temperature=0、top_p=1），seed=42、offset=0、请求顺序不变。
- 正式矩阵：concurrency=1/2/4/8/16，每档 3 轮，每轮 32 请求；每个正式 run 独立 server lifecycle，warmup 数为 `2×concurrency`。
- 冻结正式数据 SHA256：`11b2c0d603619092f7d2a35eae53cafc5cb32d4e7d642a429e29a41758a428ba`。
- 数据采集代码 SHA：`50be2b7c6883def6e77c6575915b15c0c3ff7de2`；最终报告发布 SHA 以本报告所在提交为准。

## 顺序与漂移控制

顺序在正式结果产生前写入 `campaign-plan.json`。原定同一服务连续三轮，但 c1/r1 后的第二轮 warmup 出现停滞，经批准改为每个正式 run 独立生命周期；已完成的 c1/r1 保留，其余顺序不再修改：

1. round1：c1、c8、c2、c16、c4；
2. round2：c4、c16、c2、c8、c1；
3. round3：c2、c1、c4、c8、c16。

每次生命周期启动前检查 8213 关闭且没有 `server.py`/worker 残留，启动后重新 warmup/capture，正式区间独立采集指标。每轮时间、lifecycle ID 和顺序索引位于逐 run metadata。

## 指标定义

- Client TTFT/TPOT/ITL/E2E、output/total throughput 和 QPS 来自 EvalScope 正式请求；逐请求可复算字段保存在 `request-metrics.json`。
- Server queue wait 和 server TTFT mean 来自正式前后 Prometheus sum/count 差值。`非排队首 token` 是 `server TTFT mean - server queue wait mean` 的估算，不是 profiler 分解。
- Queue wait p50/p90/p99 来自 Prometheus histogram bucket 差值并在桶内线性插值。10–30 秒桶很宽，因此只可作为粗粒度估计，不应解释为精确分位数。
- waiting/prefilling/running、KV pages、Graph counters 每秒采样；NPU utilization/HBM 每 5 秒采样。当前服务没有独立 continuous batch size 指标，报告以 running/prefilling/system requests 作为可观测代理。

## 并发扩展结果

下表为三轮 `mean ± sample stdev`。延迟单位见列名，吞吐为 token/s。

| 并发 | Client TTFT ms | Server queue ms | 非排队首 token估算 ms | TPOT ms | ITL ms | E2E s | Output tok/s | Total tok/s | QPS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 406.28 ± 1.53 | 0.45 ± 0.04 | 399.11 ± 0.81 | 38.70 ± 0.27 | 38.62 ± 0.27 | 10.275 ± 0.068 | 24.915 ± 0.164 | 37.373 ± 0.246 | 0.0973 ± 0.0007 |
| 2 | 10308.10 ± 18.88 | 9901.48 ± 18.81 | 399.70 ± 0.88 | 38.54 ± 0.07 | 38.46 ± 0.07 | 20.136 ± 0.037 | 25.029 ± 0.046 | 37.544 ± 0.070 | 0.0978 ± 0.0002 |
| 4 | 11271.69 ± 39.35 | 10420.99 ± 39.45 | 841.41 ± 0.27 | 43.44 ± 0.15 | 43.36 ± 0.16 | 22.350 ± 0.079 | 45.032 ± 0.161 | 67.548 ± 0.241 | 0.1759 ± 0.0006 |
| 8 | 12600.07 ± 29.37 | 10566.36 ± 30.72 | 2018.74 ± 1.53 | 55.83 ± 0.25 | 55.71 ± 0.25 | 26.835 ± 0.089 | 74.821 ± 0.252 | 112.232 ± 0.378 | 0.2923 ± 0.0010 |
| 16 | 16031.83 ± 25.89 | 12095.45 ± 24.53 | 3903.79 ± 2.76 | 76.06 ± 0.28 | 75.93 ± 0.28 | 35.428 ± 0.098 | 99.868 ± 0.270 | 149.802 ± 0.405 | 0.3901 ± 0.0010 |

三轮合并后的逐请求分位数如下。Queue wait 是粗粒度 histogram 估计。

| 并发 | Client TTFT p50/p90/p99 ms | Queue p50/p90/p99 ms（桶估算） | Client E2E p50/p90/p99 s | Client TPOT p50/p90/p99 ms |
|---:|---:|---:|---:|---:|
| 1 | 405.34 / 409.56 / 415.87 | 2.50 / 4.50 / 4.95 | 10.265 / 10.352 / 10.370 | 38.66 / 39.01 / 39.08 |
| 2 | 10630.39 / 10649.19 / 10665.30 | 19677 / 27935 / 29794 | 20.462 / 20.494 / 20.510 | 38.55 / 38.63 / 38.66 |
| 4 | 11272.93 / 12875.38 / 12904.82 | 19677 / 27935 / 29794 | 22.757 / 22.807 / 22.974 | 44.96 / 45.14 / 45.42 |
| 8 | 12509.70 / 12563.25 / 17565.60 | 19677 / 27935 / 29794 | 27.390 / 27.459 / 27.461 | 58.36 / 58.49 / 58.51 |
| 16 | 15687.39 / 15741.80 / 36042.53 | 20000 / 28533 / 50400 | 35.891 / 36.534 / 46.633 | 81.02 / 81.56 / 81.59 |

## 时间序列与设备状态

| 并发 | waiting mean / peak | running mean / peak | system requests mean / peak | KV pages mean / peak | NPU util mean / peak | HBM mean / peak |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.04 / 1 | 0.91 / 1 | 0.95 / 1 | 169 / 321 | 87.56% / 100% | 89% / 89% |
| 2 | 0.92 / 1 | 0.91 / 1 | 1.84 / 2 | 187 / 339 | 88.03% / 100% | 89% / 89% |
| 4 | 1.69 / 3 | 1.78 / 3 | 3.48 / 4 | 228 / 405 | 82.39% / 100% | 89% / 89% |
| 8 | 2.83 / 7 | 3.60 / 7 | 6.42 / 8 | 315 / 537 | 76.11% / 100% | 90% / 90% |
| 16 | 3.95 / 15 | 6.22 / 14 | 10.17 / 16 | 500 / 777 | 71.84% / 100% | 90% / 90% |

HBM 采样值粒度为整数百分比；NPU utilization 是低频设备采样，不能替代 kernel profiler。随着并发增大，平均 utilization 下降而峰值仍为 100%，与“prefill/decode wave 和等待间隙”一致，但不足以单独证明 kernel 级原因。

## Little's Law 检查

`QPS × mean E2E` 分别为 1.000、1.969、3.931、7.844、13.820，对目标并发 1/2/4/8/16 的误差约为 0.03%、1.57%、1.72%、1.95%、13.62%。c1–c8 闭环基本成立；c16 偏低主要来自固定 32 请求只覆盖约两批，启动/收尾边界在短测量窗口中占比更大，同时逐请求 E2E 与 campaign wall-time 的边界并不完全相同。该误差不用于修正吞吐结果。

## 根因判断

请求路径为 OpenAI handler → `BatchRequest` → continuous batching pending queue → `_step_once()` admission/prefill → decode。`lite_llama/continuous_batching.py` 中 `_step_once()` 在 `decode_priority and decode_active` 时明确执行 `admission_capacity = 0`。由于 EvalScope 并发请求并非在同一个 scheduler tick 原子到达，首批请求一旦开始 decode，稍后到达的请求就会留在 pending queue，直到 decode wave 完成。

证据链为：

1. c2 相比 c1，非排队首 token 估算仍约 400 ms、TPOT 也基本不变，但 queue wait 从 0.45 ms 跃升到 9.90 s，解释了几乎全部 TTFT 增量；
2. c2 的 output throughput 与 c1 几乎相同，waiting/running 峰值均为 1，符合两个请求被拆成两个串行 wave；
3. c4/c8/c16 的 queue share 逐步下降，但非排队首 token估算增长到 0.84/2.02/3.90 s，说明更大 prefill batch 的服务时间也开始贡献 TTFT；TPOT 同时升至 43.44/55.83/76.06 ms；
4. 所有正式区间 Graph capture 增量为 0、fallback 为 0，排除正式阶段重新 capture/fallback 作为本轮 TTFT 主因。

因此，任务2中 c4 高 TTFT/queue wait 的直接原因是 decode-priority admission policy 与请求到达微小错位共同造成的 wave 排队；并发继续升高后，prefill/decode 服务时间增长成为次要但不断增大的因素。本任务没有修改公开推理语义；是否调整 admission policy、启用 chunked prefill 或改变 decode priority 需另设严格消融。

## 诊断生命周期与正式结果分离

c1/r1 后，同一长生命周期的第二轮 warmup 曾持续至少 936 秒处于 waiting=1、running=0、AICore idle、Graph replay 不增长。该生命周期标记为 rejected/diagnostic，不占用正式 run ID，也未混入 15 个正式结果。现有证据只支持“疑似调度状态/唤醒停滞”，不足以断言代码根因。诊断快照、日志尾部、进程/端口/NPU 状态只保留在服务器，并由 bundle manifest 的 omitted 清单记录原路径、大小和 SHA256。

随后 11 个全新独立生命周期以及先前完成的 4 个正式生命周期均未复现该停滞。这说明独立生命周期能隔离状态污染，但不能证明长生命周期问题已消失。

## 正确性、Graph 门禁与异常

- 15/15 正式 run：32/32 成功，strict input=128，strict output=256，0 failed。
- 15/15：同一正式 JSONL SHA、同一 prompt token 多重集合、同一 EvalScope 参数（除 concurrency）、同一服务启动参数。
- 15/15：warmup 后 Graph 已 capture；formal capture delta=0、fallback=0、replay delta>0。
- 15/15：每秒时间序列覆盖正式区间，NPU 采样按 5 秒粒度保留。
- 停服脚本只发送 SIGTERM；父进程在等待窗口内偶尔未立即消失，但随后 8213、server.py/worker 均清空后才启动下一生命周期，未使用 force kill。

## 结论边界与复现

任务2的 c1/c4 仅作跨 campaign sanity reference，未与本轮 32 请求独立生命周期数据混合计算。本文结论仅适用于固定软件栈、短 prompt、固定 256 输出、decode priority 和当前闭环客户端到达模式；不外推到长 prompt、其他模型/精度、Graph off 或 vLLM-Ascend。

- Git-tracked bundle：`benchmarks/results/20260713_qwen3_32b_tp2_fp16_graph_scalability/`
- 逐 run 聚合：`scalability-run-metrics.csv`
- 并发汇总：`scalability-aggregate.csv`
- 时间序列汇总：`timeseries-summary.csv`
- 饱和判定：`saturation-analysis.json`
- 自动门禁：`task3-validation.json`、`strict-validation.json`

```bash
python benchmarks/qwen3_32b_tp2_fp16/validate_bundle.py \
  benchmarks/results/20260713_qwen3_32b_tp2_fp16_graph_scalability
python benchmarks/qwen3_32b_tp2_fp16/validate_strict_workload.py --compare \
  benchmarks/results/20260713_qwen3_32b_tp2_fp16_graph_scalability
python benchmarks/qwen3_32b_tp2_fp16/analyze_scalability.py --compare \
  benchmarks/results/20260713_qwen3_32b_tp2_fp16_graph_scalability
```
