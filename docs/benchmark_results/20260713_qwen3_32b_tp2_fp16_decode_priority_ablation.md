# Qwen3-32B TP=2 FP16 Decode Priority 严格配对消融

## 结论

本轮 30/30 个正式 run 均满足 input=128、output=256、0 failed，15/15 个 on/off pair 的冻结数据集、请求顺序、EvalScope 参数和服务配置均匹配，正式区间 Graph capture 增量为 0、fallback 为 0、replay 持续增长。

结果支持预注册因果假设。在 concurrency=2/4/8/16 时，关闭 Decode Priority 使 mean server queue wait 分别降低 97.85%/97.09%/96.71%/94.67%，均超过 80% 因果门槛；同时 output throughput 提升 78.80%/66.09%/49.04%/46.75%。代价是 TPOT/ITL 分别恶化约 10.1%/13.3%/12.3%/20.4%，均达到 10% decode 干扰门槛。c1 无排队压力，两种模式差异小于 1%。因此，当前短 prompt 闭环 workload 的高 queue wait/TTFT 主要由 Decode Priority admission gate 引起，而不是 Graph capture/fallback。

该结论只适用于本报告的模型、短 prompt、固定输出、闭环到达和软件栈，不外推到长 prompt、其他模型、Graph off 或 vLLM-Ascend。

## 预注册假设与严格配对设计

- 模型：Qwen3-32B，2×Atlas 910B3，物理 NPU `6,7`，TP=2，FP16。
- 服务：NPU Graph on、continuous batching、page size 16、max batch size 32、max sequence length 4096；只切换现有 `--decode_priority` 与 `--no_decode_priority`。
- workload：prompt=128、`min_tokens=max_tokens=256`、greedy、seed=42、offset=0；每个正式 run 32 个请求。
- 冻结 formal JSONL SHA256：`11b2c0d603619092f7d2a35eae53cafc5cb32d4e7d642a429e29a41758a428ba`。
- 矩阵：concurrency=1/2/4/8/16，每个 mode 三轮，共 30 个独立 server lifecycle；每轮 warmup=`2×concurrency`。
- 配对：相同 concurrency/repeat 构成 pair；pair 内轮换 on/off 先后，并发顺序也轮换。锁定顺序见 bundle 的 `campaign-plan.json`，正式结果产生后未修改。
- 预注册阈值：queue wait 降低 ≥80% 视为主要因果因素；TPOT 或 ITL 恶化 ≥10% 记录 decode 干扰代价；output throughput 提升 ≥15% 记录吞吐收益。

软件精确版本、checkpoint/config SHA、环境变量、完整命令和每轮时间位于 bundle 的 `environment/`、`lifecycles/` 和逐 run metadata。

## 三轮汇总

表中为三轮 `mean ± sample stdev`。延迟单位见列名，吞吐为 token/s。

| 模式 | 并发 | TTFT ms | Queue ms | 非排队首 token 估算 ms | TPOT ms | ITL ms | E2E s | Output tok/s | Total tok/s | QPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| on | 1 | 407.30 ± 1.15 | 0.48 ± 0.03 | 400.35 | 38.93 ± 0.09 | 38.85 ± 0.09 | 10.334 | 24.77 | 37.16 | 0.0968 |
| off | 1 | 408.38 ± 1.09 | 0.44 ± 0.05 | 400.98 | 38.86 ± 0.17 | 38.78 ± 0.17 | 10.317 | 24.81 | 37.22 | 0.0969 |
| on | 2 | 10349.67 ± 39.07 | 9941.89 ± 38.15 | 400.65 | 38.70 ± 0.15 | 38.62 ± 0.15 | 20.219 | 24.93 | 37.39 | 0.0974 |
| off | 2 | 619.11 ± 1.44 | 213.53 ± 0.23 | 399.19 | 42.62 ± 0.10 | 42.53 ± 0.10 | 11.486 | 44.57 | 66.85 | 0.1741 |
| on | 4 | 11300.39 ± 41.89 | 10449.33 ± 41.26 | 841.53 | 43.50 ± 0.23 | 43.41 ± 0.24 | 22.393 | 44.95 | 67.42 | 0.1756 |
| off | 4 | 1153.56 ± 1.44 | 303.88 ± 0.32 | 840.03 | 49.26 ± 0.19 | 49.16 ± 0.19 | 13.715 | 74.65 | 111.97 | 0.2916 |
| on | 8 | 12672.18 ± 67.55 | 10637.68 ± 68.53 | 2019.92 | 56.17 ± 0.29 | 56.06 ± 0.28 | 26.996 | 74.38 | 111.56 | 0.2905 |
| off | 8 | 2385.32 ± 1.24 | 349.84 ± 0.97 | 2019.97 | 63.08 ± 0.08 | 62.96 ± 0.07 | 18.471 | 110.85 | 166.27 | 0.4330 |
| on | 16 | 16073.67 ± 65.57 | 12138.26 ± 61.89 | 3903.53 | 76.32 ± 0.54 | 76.18 ± 0.54 | 35.534 | 99.60 | 149.40 | 0.3891 |
| off | 16 | 4559.84 ± 6.96 | 646.99 ± 0.60 | 3880.45 | 91.93 ± 0.38 | 91.75 ± 0.38 | 28.001 | 146.16 | 219.24 | 0.5709 |

请求级 p50/p90/p99、每项三轮 sample stdev 和逐 run 值均在 `decode-priority-aggregate.csv` 与 `decode-priority-run-metrics.csv`，不是从表格反推。

## 严格配对 ratio 与因果门槛

延迟 ratio 定义为 on/off（大于 1 表示 off 更低），吞吐 ratio 定义为 off/on（大于 1 表示 off 更高）。表中前三项是三个 pair ratio，末项是 ratio-of-means。

| 并发 | Queue on/off（三 pair；均值比） | TTFT on/off（三 pair；均值比） | TPOT on/off（三 pair；均值比） | ITL on/off（三 pair；均值比） | Output off/on（三 pair；均值比） |
|---:|---:|---:|---:|---:|---:|
| 1 | 1.287/0.982/1.043；1.093 | 0.999/0.991/1.002；0.997 | 1.004/1.000/1.001；1.002 | 1.004/1.000/1.001；1.002 | 1.004/1.000/1.001；1.002 |
| 2 | 46.386/46.565/46.727；46.559 | 16.660/16.740/16.750；16.717 | 0.904/0.910/0.911；0.908 | 0.904/0.910/0.911；0.908 | 1.781/1.791/1.793；1.788 |
| 4 | 34.271/34.444/34.445；34.387 | 9.741/9.825/9.823；9.796 | 0.876/0.889/0.884；0.883 | 0.876/0.889/0.884；0.883 | 1.649/1.672/1.662；1.661 |
| 8 | 30.619/30.226/30.376；30.407 | 5.334/5.278/5.325；5.313 | 0.893/0.886/0.893；0.891 | 0.893/0.885/0.893；0.890 | 1.495/1.482/1.495；1.490 |
| 16 | 18.672/18.819/18.792；18.761 | 3.514/3.531/3.530；3.525 | 0.827/0.832/0.831；0.830 | 0.827/0.832/0.832；0.830 | 1.464/1.469/1.469；1.467 |

c2–c16 均同时通过 queue wait 因果门槛和吞吐收益门槛；也均达到 TPOT/ITL 干扰代价门槛。c1 三项门槛均未触发。

## 公平性与时间序列

按服务端记录的 formal submission order，priority-on 在 c2/4/8/16 的三轮合计均有 93/96 个请求 queue wait≥1s；priority-off 对应为 0/0/0/6。priority-on 的 queue p99 分别为 10.29/12.48/17.20/32.22s，最大值分别为 10.33/12.55/17.27/35.55s；priority-off 分别降至 p99 0.400/0.400/0.404/4.681s，最大值 0.407/0.401/0.407/4.804s。后半批相对前半批的 mean queue wait 差在 on 模式为 0.636/0.771/1.048/1.902s，而 off 模式仅为 0.002/0.001/0.001/0.009s，说明关闭 gate 显著减少 wave starvation 和到达顺序长尾。

一秒时间序列中，c2/4/8/16 的 waiting mean 从 on 的 0.92/1.69/2.80/3.94 降至 off 的 0.07/0.27/0.92/1.22；running mean 从 0.91/1.78/3.62/6.21 升至 1.72/3.17/5.56/10.21。off 模式同时提高 KV pages 使用，符合更多请求被及时接纳。NPU utilization 为低频设备采样、HBM 为整数百分比，不能替代 kernel profiler；完整 mean/peak 在 aggregate CSV，原始样本在逐 run `timeseries.jsonl`。

所有 30 个 formal 区间的 Graph capture delta 合计为 0、fallback delta 合计为 0；priority-on/off 的 replay 均持续增长。prefilling 独立 gauge 在当前采样粒度下为 0，不能据此断言没有 prefill，只能结合 running、queue 和服务事件解释。

## Pareto 建议

- 在线低 TTFT：本短 prompt workload 在并发≥2 时优先 `--no_decode_priority`，queue 与 TTFT 大幅降低，公平性更好。
- Decode 平稳性：`--decode_priority` 的 TPOT/ITL 更低；若逐 token 稳定性比首 token 和吞吐更重要，可保留该模式。
- 离线吞吐：本 workload 优先 `--no_decode_priority`，c2–c16 output throughput 提升 46.75%–78.80%。

这些建议不是通用默认值；长 prompt、开放到达、不同 batch/page 配置必须重新测试。

## 正确性、异常与边界

- 30/30：32/32 成功、strict input=128、strict output=256、0 failed。
- 15/15 pair：同一冻结 JSONL、seed/offset、prompt token 多重集合、请求顺序、EvalScope 参数；服务命令归一化后只剩 Decode Priority flag 差异。
- 30/30：独立 lifecycle，formal 前 capture 完成，formal capture delta=0、fallback=0、replay delta>0，时间序列覆盖完整。
- 本轮没有出现 Task3 的 `waiting>0/running=0/AICore idle` 长停滞，没有 rejected formal lifecycle。
- 停服脚本未使用 force kill；每个后续 lifecycle 都在 8213 关闭、无 server/worker 残留、NPU 6/7 AICore idle 后启动。
- “非排队首 token”是 server TTFT mean 减 queue wait mean 的估算，不是 profiler 分解。

## 复现与离线校验

- Git-tracked bundle：`benchmarks/results/20260713_qwen3_32b_tp2_fp16_decode_priority_ablation/`
- 逐 run 指标：`decode-priority-run-metrics.csv`
- mode/并发汇总：`decode-priority-aggregate.csv`
- 逐 pair ratio：`decode-priority-paired-ratios.csv`
- 公平性：`decode-priority-fairness.csv`
- 因果阈值：`decode-priority-causal.json`
- 自动门禁：`task4-validation.json`、`strict-validation.json`

```bash
python benchmarks/qwen3_32b_tp2_fp16/validate_bundle.py \
  benchmarks/results/20260713_qwen3_32b_tp2_fp16_decode_priority_ablation
python benchmarks/qwen3_32b_tp2_fp16/validate_strict_workload.py --compare \
  benchmarks/results/20260713_qwen3_32b_tp2_fp16_decode_priority_ablation
python benchmarks/qwen3_32b_tp2_fp16/compare_decode_priority.py --compare \
  benchmarks/results/20260713_qwen3_32b_tp2_fp16_decode_priority_ablation
```

Git bundle 共 1012 个文件、9,590,950 bytes；manifest 的 SHA256 索引覆盖其中 1011 个内容文件、9,263,643 bytes，manifest 自身不自引用。467 个 server-only 大文件（完整 stdout/log、SQLite/HTML、lifecycle monitor 等）共 101,011,524 bytes，均在 `campaign-manifest.json` 的 omitted 清单记录服务器原路径、大小和 SHA256；复算本报告不依赖它们。
