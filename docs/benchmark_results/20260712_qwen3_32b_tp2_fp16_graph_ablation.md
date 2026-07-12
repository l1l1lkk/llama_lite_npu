# Qwen3-32B TP=2 FP16 NPU Graph 严格消融

## 结论

在完全配对的 `prompt=128`、固定输出 256 token workload 下，NPU Graph 对解码阶段有稳定且显著的收益：c1 的客户端 E2E 延迟降低 79.43%、输出吞吐提高 4.862 倍；c4 的 E2E 延迟降低 78.17%、输出吞吐提高 4.558 倍。c1 TTFT 基本不变（降低 0.18%，三轮有正有负），不能据此宣称 Graph 改善单请求 prefill。c4 的 TTFT 和 queue wait 同时下降约 78%～79%，说明本 workload 下吞吐改善显著缓解了串行排队。

这些结论只适用于本报告固定的软件、模型、设备、模板和短 prompt workload，不与任务1或 20260711 的非配对历史数据混算，也不外推到其他并发、上下文长度或精度。

## 测试假设和环境

- 模型：Qwen3-32B；checkpoint 见 bundle 的 `environment/baseline.env` 和模型配置 SHA。
- 硬件：2×Atlas 910B3，物理 NPU `6,7`，TP=2，FP16。
- 服务：OpenAI chat completions、streaming、page size 16、continuous batching、max batch size 32、decode priority、max seq len 4096。
- workload：服务端实际 input=128，`min_tokens=max_tokens=256`，greedy（temperature=0、top_p=1），相同 Qwen3 chat template。
- 软件：CANN 26.0.rc1，PyTorch 2.7.1，torch_npu 2.7.1，triton-ascend 3.2.0，EvalScope 1.7.1。
- 数据采集代码 SHA：`ebf984d01215507f1421f1530adb909da1fb07c3`；分支 `release/0.0.12rc1`。最终报告与 bundle 的发布 SHA 以仓库提交为准。

## 严格配对方法

c1 每轮正式 6 请求、warmup 2 请求；c4 每轮正式 8 请求、warmup 8 请求。每档运行 3 轮。每个 on/off 对共享相同 seed=42、dataset offset、warmup offset、正式/预热 JSONL、JSONL SHA、请求数、token 序列和 EvalScope 参数。Graph on 与 off 使用独立服务生命周期，启动命令归一化后仅 `--compiled_model` / `--no_compiled_model` 不同。

EvalScope 的 random 插件内部使用 Python `set`，不同进程即使 seed/offset 相同也可能产生不同 token；因此早期随机尝试被拒绝并保留在 server-only artifacts，未进入任何结果。正式测试使用冻结的 line-by-line JSONL。JSONL SHA 证明输入顺序一致；由于并发 SQLite 按完成顺序落库，on/off 完成次序可以不同，比较器另外对实际观察到的完整 token ID 做带重复多重集合比较，六对均通过。

正式 run ID 统一为 `20260712_qwen3_32b_tp2_fp16_graph_ablation_{on|off}_p128_o256_c{1|4}_r{1|2|3}`。

## 三轮汇总

延迟类 ratio 定义为 `off/on`，数值大于 1 表示 Graph on 更快；吞吐类 ratio 定义为 `on/off`，数值大于 1 表示 Graph on 吞吐更高。表中为 mean ± sample stdev。

| 并发 | 指标 | Graph on | Graph off | 汇总 ratio | on 相对改善 |
|---:|---|---:|---:|---:|---:|
| 1 | E2E latency (s) | 10.2149 ± 0.0108 | 49.6692 ± 0.3409 | 4.8624× | 延迟 -79.43% |
| 1 | TTFT (ms) | 406.46 ± 1.03 | 407.20 ± 0.82 | 1.0018× | 延迟 -0.18% |
| 1 | TPOT (ms) | 38.463 ± 0.035 | 193.187 ± 1.341 | 5.0226× | 延迟 -80.09% |
| 1 | ITL (ms) | 38.310 ± 0.040 | 192.427 ± 1.332 | 5.0229× | 延迟 -80.09% |
| 1 | output throughput (tok/s) | 25.060 ± 0.027 | 5.154 ± 0.035 | 4.8621× | 吞吐 +386.21% |
| 1 | total throughput (tok/s) | 37.591 ± 0.040 | 7.731 ± 0.053 | 4.8621× | 吞吐 +386.21% |
| 1 | QPS | 0.0979 ± 0.0001 | 0.02013 ± 0.00012 | 4.8626× | 吞吐 +386.26% |
| 1 | server queue wait (ms) | 0.673 ± 0.072 | 0.461 ± 0.133 | 0.6853× | on 高 45.91% |
| 4 | E2E latency (s) | 21.0396 ± 0.0112 | 96.3659 ± 0.4851 | 4.5802× | 延迟 -78.17% |
| 4 | TTFT (ms) | 10047.37 ± 10.81 | 45128.86 ± 646.99 | 4.4916× | 延迟 -77.74% |
| 4 | TPOT (ms) | 43.107 ± 0.006 | 200.930 ± 0.649 | 4.6612× | 延迟 -78.55% |
| 4 | ITL (ms) | 42.943 ± 0.015 | 200.173 ± 0.611 | 4.6613× | 延迟 -78.55% |
| 4 | output throughput (tok/s) | 45.332 ± 0.022 | 9.946 ± 0.034 | 4.5580× | 吞吐 +355.80% |
| 4 | total throughput (tok/s) | 67.998 ± 0.032 | 14.918 ± 0.051 | 4.5580× | 吞吐 +355.80% |
| 4 | QPS | 0.1771 ± 0.0001 | 0.03883 ± 0.00015 | 4.5605× | 吞吐 +356.05% |
| 4 | server queue wait (ms) | 9196.55 ± 9.96 | 44275.63 ± 647.56 | 4.8144× | 延迟 -79.23% |

逐 run 数值与 ratio 的权威来源为 bundle 的 `paired-ratios.csv`；三轮 mean/stdev 和 ratio-of-means 来源为 `graph-comparison.csv`，均可由逐 run JSON/metrics 离线重建。

## 正确性和 Graph 计数

- 12/12 formal run：strict input 128、strict output 256、失败请求 0。
- 6/6 on/off 配对：metadata、EvalScope args、冻结数据集 SHA、观察 token 多重集合全部一致。
- Graph on：c1 首次 warmup 完成 2 次 capture，c4 首次 warmup 把累计 capture 扩展到 4；每个 formal 区间 capture 增量为 0、fallback 为 0，replay 持续增加。
- Graph off：所有 pre-warmup / before-formal / after-formal 快照中的 capture_attempts、captures、replays、fallbacks 全为 0。

完整计数按 pair 保存在 `pair-validation.json`，正确性门禁保存在 `strict-validation.json`。

## 异常与边界

1. 早期 random workload 因 Python hash 随机化导致跨进程 token 不一致，被比较器拒绝；未计算或发布其 speedup。原文件只在服务器保留，并由 campaign manifest 给出路径、大小和 SHA256。
2. Graph on/off 停止时，父进程在停止脚本等待窗口内未立即消失；服务端口和子 rank 随后均退出，第二生命周期只在确认无 `server.py` 进程后启动。未使用 force kill。
3. c1 queue wait 均低于 1 ms，差值相对于该尺度较小且方向不利，不应解读为 Graph 的排队优势；主要收益来自 TPOT/ITL。
4. c4 高 TTFT 主要包含服务端排队。本阶段只报告现象，不做根因分析或任务3复测。

## 可复核路径

- Git bundle：`benchmarks/results/20260712_qwen3_32b_tp2_fp16_graph_ablation/`
- 冻结 workload：`benchmarks/qwen3_32b_tp2_fp16/datasets/20260712_graph_ablation/`，bundle 内另有 `workload/` 自包含副本。
- 离线验证：`python benchmarks/qwen3_32b_tp2_fp16/validate_bundle.py benchmarks/results/20260712_qwen3_32b_tp2_fp16_graph_ablation`
- 配对重建：`python benchmarks/qwen3_32b_tp2_fp16/compare_graph.py benchmarks/results/20260712_qwen3_32b_tp2_fp16_graph_ablation --compare`
