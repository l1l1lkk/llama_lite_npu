# 推理性能历史记录

本文按时间倒序记录项目的核心推理性能，用于观察版本演进和性能回归。

## 记录规则

- 只记录已经实际运行得到的数据，不填写预测值。
- 必须保留模型、测试工具、TP、Batch/并发、输入/输出长度和 NPU Graph 状态。
- `benchmark_tp.py` 的 `Avg throughput` 表示单序列生成步吞吐，`Batch throughput` 才是批量总输出吞吐。
- EvalScope 的 `Output Throughput` 表示服务在测试期间生成的总输出 Token 吞吐。
- 不同测试工具、输入长度、输出长度或并发配置的数据不能直接横向比较。
- 历史数据没有保存 Git commit 时明确标记为“commit 未记录”，不将其绑定到具体版本。

## 核心记录

| 日期 | 项目版本 | 模型 | 测试工具 | TP / Batch或并发 | 输入 / 输出 | NPU Graph | 执行路径 | 核心吞吐 | 单Token指标 | 备注 |
|---|---|---|---|---|---|---|---|---:|---:|---|
| 2026-06-11 | 0.0.3rc2 | Qwen3-30B-A3B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 关闭 | `auto`，应解析为GMM + Triton路由 | 5.0 tok/s；Batch 20.0 tok/s | 199.81 ms/token | 5次平均51.152s；模型与KV约54.6GB |
| 2026-06-09 | 0.0.1rc1 | Qwen3-32B | EvalScope | TP=2 / 并发=1 | 平均59.53 / 494.47 | 开启 | Decode Graph，128-token bucket | Output 24.0606 tok/s；Total 26.9575 tok/s | TPOT 40.6ms；ITL 40.9ms | 15请求；TTFT 261.2ms |
| 2026-06-09 | 0.0.1rc1前基线 | Qwen3-32B | EvalScope | TP=2 / 并发=1 | 平均60.73 / 564.73 | 关闭 | Eager Decode | Output 5.696 tok/s；Total 6.3085 tok/s | TPOT 175.2ms；ITL 174.7ms | 15请求；commit未记录 |
| 2026-05-18 | 历史代码，commit未记录 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 关闭 | Eager Decode | 5.4 tok/s；Batch 21.6 tok/s | 185.56 ms/token | 5次平均47.503s |
| 2026-05-18 | 历史代码，commit未记录 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 开启但捕获效果未确认 | Graph runner已创建 | 5.3 tok/s；Batch 21.1 tok/s | 189.39 ms/token | 5次平均48.485s；不能作为有效Graph加速结果 |
| 2026-05-18 | 历史代码，commit未记录 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 关闭 | Eager Decode | 5.3 tok/s；Batch 21.3 tok/s | 187.91 ms/token | 早期Dense基线；5次平均48.106s |

## 当前同口径结论

使用相同的 `benchmark_tp.py` 配置比较：

| 模型 | Avg throughput | Batch throughput | ms/token |
|---|---:|---:|---:|
| Qwen3-32B Dense历史最佳基线 | 5.4 tok/s | 21.6 tok/s | 185.56 |
| Qwen3-30B-A3B MoE v0.0.3rc2 | 5.0 tok/s | 20.0 tok/s | 199.81 |
| MoE相对Dense | -7.4% | -7.4% | +7.7% |

因此，本次 MoE 实测比 Dense 慢约 7%～8%；考虑运行波动，可概括为接近 10%。

## 后续测试必须补充

MoE 测试建议显式设置后端，避免 `auto` 随环境变化：

```bash
export LITE_LLAMA_MOE_BACKEND=gmm
```

`v0.0.4rc1`新增MoE Decode NPU Graph。该版本尚无Atlas性能记录，必须确认Benchmark中的
`captured > 0`且`replays > 0`后，才能将结果作为Graph性能写入上表；如果
`fallbacks > 0`且`captured = 0`，结果仍属于Eager路径。

后续每次记录至少保留：

```text
Git commit / 版本
模型与dtype
测试工具
TP、Batch或并发
输入长度、输出长度
NPU Graph状态
MoE backend
吞吐、TPOT或ms/token
```
