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
| 2026-06-22 | 0.0.8rc1 | Qwen3-32B | EvalScope | TP=2 / concurrency=1 | avg 184 / 256 | enabled | Continuous Batching; Greedy; fixed-length prompt; packed prefill not stressed | Output 24.7089 tok/s; Total 42.4685 tok/s | TTFT 702.7ms; TPOT 37.9ms; ITL 37.7ms | 20 requests; fixed-length baseline; no decode regression vs v0.0.7rc6 |
| 2026-06-22 | 0.0.8rc1 | Qwen3-32B | EvalScope | TP=2 / concurrency=4 | avg 285.475 / 245.7 | enabled | Continuous Batching; Greedy; mixed prompt lengths; packed prefill path | Output 50.158 tok/s; Total 108.436 tok/s | TTFT 5117.6ms; TPOT 57.8ms; ITL 57.1ms | 40 requests; mixed-length prefill stress; compare only against same workload |
| 2026-06-18 | 0.0.7rc6 | Qwen3-32B | EvalScope | TP=2 / concurrency=1 | avg 184.0 / 255.95 | enabled | Continuous Batching; Greedy; exact Prefix Cache on by default; partial Prefix Cache off | Output 24.3149 tok/s; Total 41.7947 tok/s | TTFT 706.3ms; TPOT 38.5ms; ITL 38.4ms | 20 requests; random dataset; rc5 default-partial regression fixed |
| 2026-06-18 | 0.0.7rc6 | Qwen3-32B | EvalScope | TP=2 / concurrency=4 | avg 184.0 / 215.775 | enabled | Continuous Batching; Greedy; exact Prefix Cache on by default; partial Prefix Cache off | Output 62.4598 tok/s; Total 115.7218 tok/s | TTFT 1502.2ms; TPOT 65.0ms; ITL 56.8ms | 40 requests; random dataset; average output shorter than 256, compare with caution |
| 2026-06-18 | 0.0.7rc6 | Qwen3-32B | `benchmark_prefix_cache.py` | TP=2 / concurrency=1 | approx 62 / 256 | enabled | same prompt; Exact Prefix Cache | Output 26.56 tok/s/request-time; Wall 388.45 tok/s/wall | Avg TTFT 22.9ms; P50 TTFT 6.0ms | 20 requests; first request TTFT 344ms, later requests about 5-6ms |
| 2026-06-17 | 0.0.6rc2 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 开启 | Batch级Vocab Parallel Greedy；temperature=0；Top-P inactive | 22.1 tok/s；Batch 88.6 tok/s | 45.16 ms/token | 5次平均11.560s；模型与KV约54.3GB；Graph attempts=3、captured=3、replays=1785、fallbacks=0 |
| 2026-06-17 | 0.0.6rc2 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 开启 | 精确Vocab Parallel Top-P；temperature=0.6、top_p=0.9 | 17.9 tok/s；Batch 71.7 tok/s | 55.83 ms/token | 5次平均14.293s；模型与KV约54.3GB；Graph attempts=3、captured=3、replays=1785、fallbacks=0 |
| 2026-06-12 | 0.0.5rc2 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 开启 | 完整词表Logits AllGather + Greedy；temperature=0 | 21.3 tok/s；Batch 85.0 tok/s | 47.04 ms/token | 5次平均12.043s；模型与KV约54.2GB；Graph attempts=3、captured=3、replays=1785、fallbacks=0 |
| 2026-06-12 | 0.0.6rc1 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 开启 | Vocab Parallel Greedy；temperature=0 | 19.7 tok/s；Batch 78.6 tok/s | 50.87 ms/token | 5次平均13.023s；模型与KV约54.2GB；Graph attempts=3、captured=3、replays=1785、fallbacks=0 |
| 2026-06-12 | 0.0.6rc1 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 开启 | 精确Vocab Parallel Top-P；temperature=0.6、top_p=0.9 | 17.7 tok/s；Batch 70.9 tok/s | 56.41 ms/token | 5次平均14.441s；模型与KV约54.2GB；Graph attempts=3、captured=3、replays=1785、fallbacks=0 |
| 2026-06-12 | 0.0.5rc2 | Qwen3-30B-A3B | `benchmark_tp.py` | EP=2 / Batch=4 | 约128 / 256 | 关闭 | Expert Parallel Eager；MoE backend需由启动日志确认 | 5.5 tok/s；Batch 22.1 tok/s | 181.09 ms/token | 5次平均46.359s；模型与KV约54.6GB；EP Graph按设计自动禁用 |
| 2026-06-11 | 0.0.4rc1 | Qwen3-30B-A3B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 开启 | GMM + Triton路由 + Decode Graph | 31.7 tok/s；Batch 126.9 tok/s | 31.53 ms/token | 5次平均8.071s；Graph attempts=3、captured=3、replays=1785、fallbacks=0 |
| 2026-06-11 | 0.0.3rc2 | Qwen3-30B-A3B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 关闭 | `auto`，应解析为GMM + Triton路由 | 5.0 tok/s；Batch 20.0 tok/s | 199.81 ms/token | 5次平均51.152s；模型与KV约54.6GB |
| 2026-06（补录） | 0.0.2，具体RC未确认 | Qwen3-30B-A3B | `benchmark_tp.py` | TP=2 / Batch=4 | 输入长度未保存 / 256 | 关闭 | P0～P3前专家Python循环 | 1.0 tok/s；Batch 4.1 tok/s | 987.06 ms/token | 5次平均252.687s；模型与KV约54.6GB |
| 2026-06-09 | 0.0.1rc1 | Qwen3-32B | EvalScope | TP=2 / 并发=1 | 平均59.53 / 494.47 | 开启 | Decode Graph，128-token bucket | Output 24.0606 tok/s；Total 26.9575 tok/s | TPOT 40.6ms；ITL 40.9ms | 15请求；TTFT 261.2ms |
| 2026-06-09 | 0.0.1rc1前基线 | Qwen3-32B | EvalScope | TP=2 / 并发=1 | 平均60.73 / 564.73 | 关闭 | Eager Decode | Output 5.696 tok/s；Total 6.3085 tok/s | TPOT 175.2ms；ITL 174.7ms | 15请求；commit未记录 |
| 2026-05-18 | 历史代码，commit未记录 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 关闭 | Eager Decode | 5.4 tok/s；Batch 21.6 tok/s | 185.56 ms/token | 5次平均47.503s |
| 2026-05-18 | 历史代码，commit未记录 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 开启但捕获效果未确认 | Graph runner已创建 | 5.3 tok/s；Batch 21.1 tok/s | 189.39 ms/token | 5次平均48.485s；不能作为有效Graph加速结果 |
| 2026-05-18 | 历史代码，commit未记录 | Qwen3-32B | `benchmark_tp.py` | TP=2 / Batch=4 | 约128 / 256 | 关闭 | Eager Decode | 5.3 tok/s；Batch 21.3 tok/s | 187.91 ms/token | 早期Dense基线；5次平均48.106s |

## 2026-06-22 v0.0.8rc1 EvalScope 固定长度与混合长度测试

本轮记录 Qwen3-32B、TP=2、NPU Graph enabled、Greedy 采样下，v0.0.8rc1 packed prefill 改动后的表现。

| 场景 | 并发 | 请求数 | 平均输入/输出 | Output Throughput | Total Throughput | Avg Latency | TTFT | TPOT | ITL |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 固定长度单并发 | 1 | 20 | 184 / 256 | 24.7089 tok/s | 42.4685 tok/s | 10.3601s | 702.7ms | 37.9ms | 37.7ms |
| 混合长度并发 | 4 | 40 | 285.475 / 245.7 | 50.158 tok/s | 108.436 tok/s | 19.1015s | 5117.6ms | 57.8ms | 57.1ms |

固定长度单并发与 v0.0.7rc6 固定长度单并发相比：Output Throughput 从 24.3149 提升到 24.7089 tok/s，约 +1.6%；Total Throughput 从 41.7947 提升到 42.4685 tok/s，约 +1.6%；Latency 从 10.5259s 降到 10.3601s，约 -1.6%；TPOT 从 38.5ms 降到 37.9ms，约 -1.6%。结论是 fixed-length decode 路径没有回归，结果略好但属于小幅收益/测试波动范围。

混合长度并发结果不能直接与之前固定长度或短输入随机测试横向比较：本轮平均输入为 285.475 tokens，并且 P10 到 P99 输入长度约从 104 到 540 tokens，prefill 压力明显更高。TTFT 升高到 5.1176s 主要来自更长、更分散的 prompt prefill；ITL 为 57.1ms，说明 decode 阶段保持在接近 v0.0.7 并发测试的量级。该结果说明 packed prefill 的混合长度路径可以正常承载请求，但若要量化“提升多少”，需要在 v0.0.7rc6 和 v0.0.8rc1 上跑完全相同的混合长度命令。

Chunked prefill 本轮未记录性能结果，因为测试报错，待补充完整 traceback 后单独记录。

## 当前同口径结论

使用相同的 `benchmark_tp.py` 配置比较：

| 模型 | Avg throughput | Batch throughput | ms/token |
|---|---:|---:|---:|
| Qwen3-32B Dense v0.0.6rc2 Graph Greedy | 22.1 tok/s | 88.6 tok/s | 45.16 |
| Qwen3-32B Dense v0.0.6rc2 Graph Top-P | 17.9 tok/s | 71.7 tok/s | 55.83 |
| Qwen3-32B Dense v0.0.5rc2 Graph Greedy | 21.3 tok/s | 85.0 tok/s | 47.04 |
| Qwen3-32B Dense v0.0.6rc1 Graph Greedy | 19.7 tok/s | 78.6 tok/s | 50.87 |
| Qwen3-32B Dense v0.0.6rc1 Graph Top-P | 17.7 tok/s | 70.9 tok/s | 56.41 |
| Qwen3-30B-A3B MoE 0.0.2（P0～P3前） | 1.0 tok/s | 4.1 tok/s | 987.06 |
| Qwen3-32B Dense历史最佳基线 | 5.4 tok/s | 21.6 tok/s | 185.56 |
| Qwen3-30B-A3B MoE v0.0.3rc2 | 5.0 tok/s | 20.0 tok/s | 199.81 |
| Qwen3-30B-A3B MoE v0.0.5rc2 EP Eager | 5.5 tok/s | 22.1 tok/s | 181.09 |
| Qwen3-30B-A3B MoE v0.0.4rc1 | 31.7 tok/s | 126.9 tok/s | 31.53 |

v0.0.6rc1的两组Qwen3-32B结果使用完全相同的Graph和Benchmark配置，仅改变采样策略：

- Greedy吞吐为19.7 tok/s，比Top-P的17.7 tok/s高约11.3%；
- Top-P平均生成时间增加1.418s，增幅约10.9%；
- Top-P单Token耗时增加5.54ms，增幅约10.9%；
- 两组均为`captured=3`、`replays=1785`、`fallbacks=0`，差异不是Graph回退导致。

Greedy路径只需在每个Rank求局部最大值并交换少量最大值和Token ID。精确Top-P还需要
全局Softmax归一化、各Rank Top-K候选提取、候选AllGather、排序、累计概率和Multinomial
采样，因此17.7 tok/s是符合当前实现预期的结果。该差异反映采样开销，不代表模型精度
下降。

但是v0.0.6rc1 Greedy相对v0.0.5rc2 Greedy出现性能回归：

- 吞吐由21.3下降到19.7 tok/s，下降约7.5%；
- 平均时间由12.043s增加到13.023s，增加约8.1%；
- 单Token耗时由47.04ms增加到50.87ms，增加约8.1%。

原因是v0.0.6rc1首版分布式Greedy按Batch逐行执行。Batch=4时，每个Decode Step会为
4行分别执行Value和Token ID的AllGather，共8次小Collective；v0.0.5rc2虽然传输了完整
词表Logits，但AllGather位于模型Forward/NPU Graph中，通信调用次数更少。当前双卡小
Batch场景受HCCL启动延迟影响，减少通信字节没有抵消增加通信次数的成本。后续应把
Greedy改成整个Batch一次性求局部最大值并批量通信，同时将Top-P候选通信向量化。

v0.0.6rc2完成Batch级Greedy通信修复后：

- Greedy吞吐由v0.0.6rc1的19.7提升到22.1 tok/s，提升约12.2%；
- Greedy单Token耗时由50.87ms下降到45.16ms，降低约11.2%；
- 相对v0.0.5rc2的21.3 tok/s，v0.0.6rc2高约3.8%；
- Top-P由17.7提升到17.9 tok/s，仅提升约1.1%，基本不变。

该结果说明rc2准确修复了rc1的Greedy通信粒度回归，并略高于旧版完整词表AllGather路径。
但收益没有达到“显著加速”的原因是：旧版完整Logits AllGather在Decode Graph内，调用次数少
且通信带宽利用较好；新版虽然降低通信字节数，但局部Argmax、候选打包和小AllGather仍在
Graph外执行，Decode总耗时主要仍由Transformer层MatMul、Attention、HCCL AllReduce和
Graph外采样开销共同决定。Top-P路径尚未做Batch级向量化，仍包含全局归一化、候选通信、
排序和随机采样，因此这次rc2对Top-P提升很小。

从v0.0.3rc2的MoE TP Eager基线到v0.0.5rc2的EP Eager结果：

- 单序列吞吐由5.0 tok/s提升到5.5 tok/s，提升约10.0%；
- Batch吞吐由20.0 tok/s提升到22.1 tok/s，提升约10.5%；
- 单Token耗时由199.81ms下降到181.09ms，降低约9.4%；
- EP Eager与Dense历史Eager基线基本持平，差异处于约2%的量级。

该结果说明当前两卡EP路径已经能够正确运行，但尚未形成数量级性能收益。TP=2时，TP路径
对8个命中专家分别计算一半中间维，EP路径平均在每卡计算约4个完整专家；两者每卡有效
专家计算量理论上接近。EP还需要承担本地专家筛选、动态路由压缩和负载不均衡，因此两卡
小Batch场景下主要价值是验证框架能力，而不是显著降低Decode时延。

从v0.0.3rc2到v0.0.4rc1，在相同模型和Benchmark配置下：

- 单序列吞吐由5.0 tok/s提升到31.7 tok/s，约为原来的6.34倍；
- Batch吞吐由20.0 tok/s提升到126.9 tok/s，约为原来的6.35倍；
- 单Token耗时由199.81ms下降到31.53ms，降低约84.2%；
- 平均生成时间由51.152s下降到8.071s，降低约84.2%。

`attempts=3`、`captured=3`、`replays=1785`、`fallbacks=0`表明测试覆盖的三个
序列长度Bucket均成功Capture，正式迭代持续使用Graph Replay，没有回退Eager。

`v0.0.5rc2`已经完成双卡EP Eager Atlas基线。当前仍缺少同版本、同后端的TP Eager
复测，因此5.5 tok/s与v0.0.3rc2的5.0 tok/s只能作为阶段性参考，不能完全拆分版本代码、
后端自动选择和EP并行方式各自的贡献。

## 后续测试必须补充

MoE 测试建议显式设置后端，避免 `auto` 随环境变化：

```bash
export LITE_LLAMA_MOE_BACKEND=gmm
```

MoE TP Graph结果必须确认Benchmark中的`captured > 0`且`replays > 0`后，才能作为
Graph性能写入上表；如果`fallbacks > 0`且`captured = 0`，结果仍属于Eager路径。

MoE EP在`v0.0.5rc2`后按设计自动关闭Decode Graph。EP结果应标记为Eager，不再用
Graph Capture统计判断执行路径。

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


## 2026-06-17 v0.0.7rc1 EvalScope Server Measurements

Environment and command shape:

- Model: Qwen3-32B
- Hardware: 2 x Atlas 910B3
- Runtime: OpenAI-compatible `server.py` with Continuous Batching enabled
- Branch/version: `release/0.0.7rc1`
- TP: 2
- PagedAttention: `page_size=16`
- Decode NPU Graph: enabled by server startup option
- Scheduler: `max_batch_size=32`, `max_prefill_tokens=1024`, `max_decode_tokens=8`
- Dataset: EvalScope random
- Prompt/output target: 128 prompt tokens, 256 max output tokens
- Stream: enabled

### Raw results

| Test | Concurrency | Requests | Temperature / Top-p | Avg input tokens | Avg output tokens | Output throughput | Total throughput | Req throughput | Avg latency | TTFT | TPOT | ITL |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v007_greedy_p1` | 1 | 20 | temperature=0 | 156.000 | 256.000 | 24.2603 tok/s | 39.0439 tok/s | 0.0948 req/s | 10.5516 s | 0.6670 s | 0.0388 s | 0.0386 s |
| `v007_greedy_p4` | 4 | 40 | temperature=0 | 155.975 | 254.125 | 63.3523 tok/s | 102.2360 tok/s | 0.2493 req/s | 15.7068 s | 2.2958 s | 0.0530 s | 0.0529 s |
| `v007_topp_p4` | 4 | 40 | temperature=0.6, top_p=0.9 | 156.000 | 231.325 | 57.7176 tok/s | 96.6409 tok/s | 0.2495 req/s | 15.7261 s | 1.4861 s | 0.0629 s | 0.0616 s |

### Interpretation

Greedy concurrency scaling, comparing `v007_greedy_p4` with `v007_greedy_p1`:

- Output throughput: 24.2603 -> 63.3523 tok/s, +161.1%, 2.61x.
- Total throughput: 39.0439 -> 102.2360 tok/s, +161.8%, 2.62x.
- Request throughput: 0.0948 -> 0.2493 req/s, +162.9%, 2.63x.
- Average latency: 10.5516 -> 15.7068 s, +48.9%.
- TPOT: 38.8 -> 53.0 ms, +36.6%.

This shows the scheduler/KV refactor mainly improves service-side aggregate throughput under concurrent requests. It does not make a single decode stream faster by itself. The concurrency efficiency is about 65% of ideal 4x scaling, which is reasonable for this stage because decode still shares model compute, TP communication, sampling, HTTP streaming, and scheduler overhead.

Top-P overhead, comparing `v007_topp_p4` with `v007_greedy_p4`:

- Output throughput: 63.3523 -> 57.7176 tok/s, -8.9%.
- Total throughput: 102.2360 -> 96.6409 tok/s, -5.5%.
- TPOT: 53.0 -> 62.9 ms, +18.7%.
- ITL: 52.9 -> 61.6 ms, +16.4%.
- Request throughput is effectively unchanged: 0.2493 -> 0.2495 req/s.

Top-P is slower because it still needs probability normalization, candidate filtering/sorting, and random sampling work that Greedy avoids. The average output length is also shorter in this run, so request throughput is not directly comparable to output-token throughput.

### What v0.0.7rc1 improved

v0.0.7rc1 introduced token-budget scheduling and KV metadata foundations. In this measurement, the visible improvement is not from live KV prefix reuse yet; prefix cache and chunked prefill are metadata/planning only in this release. The measurable gain is that the server can batch concurrent decode work through Continuous Batching and maintain much higher aggregate throughput:

- Greedy single concurrency output throughput: 24.2603 tok/s.
- Greedy concurrency 4 output throughput: 63.3523 tok/s.
- Aggregate output throughput gain from concurrency batching: +39.092 tok/s, +161.1%.

This should be treated as a server-scheduling gain, not a MatMul/kernel-level gain.

## 2026-06-18 v0.0.7rc4 Prefix Cache Server Measurements

Environment and command shape:

- Model: Qwen3-32B
- Runtime: OpenAI-compatible `server.py`
- Benchmark script: `examples/benchmark_prefix_cache.py`
- Requests: 20
- Concurrency: 1
- Max output tokens: 256
- Temperature: 0
- Stream: enabled
- Thinking: off

### Raw results

| Dataset | Requests | Success | Avg latency | P50 / P90 / P99 latency | Avg TTFT | P50 / P90 / P99 TTFT | Avg output tokens | Output throughput | Avg total tokens |
|---|---:|---:|---:|---|---:|---|---:|---:|---:|
| `same` | 20 | 20 | 9.5967 s | 9.4165 / 9.4432 / 12.9406 s | 0.0232 s | 0.0057 / 0.0067 / 0.3509 s | 256.0 | 26.68 tok/s/request-time | 318.0 |
| `random` | 20 | 20 | 11.1117 s | 11.1857 / 11.2068 / 12.4304 s | 1.2658 s | 1.1005 / 1.4423 / 1.4451 s | 256.0 | 23.04 tok/s/request-time | 517.5 |

### Prefix Cache effect

The repeated-prompt run shows a clear Prefix Cache hit pattern:

- Request 1 TTFT: 0.351 s.
- Requests 2-20 TTFT: stable around 0.005-0.008 s.
- Cached-request average TTFT after excluding the first request: about 0.006 s.
- Random prompts stay around 1.09-1.44 s TTFT and do not show cache reuse.

Compared with the random-prompt baseline:

- Average TTFT drops from 1.2658 s to 0.0232 s, about 54.6x lower.
- P50 TTFT drops from 1.1005 s to 0.0057 s, about 193x lower.
- Average latency drops from 11.1117 s to 9.5967 s, about 13.6% lower.
- Output throughput by request-time improves from 23.04 tok/s to 26.68 tok/s, about +15.8%.

Caveat: the repeated prompt and random prompt runs do not have identical input-token counts (`Avg total tokens` differs), so total latency is not a pure cache-only comparison. The TTFT collapse after the first repeated request is the strongest evidence that Prefix Cache reuse is working.

## 2026-06-18 v0.0.7rc6 EvalScope and Prefix Cache Measurements

Environment and command shape:

- Model: Qwen3-32B
- Hardware: 2 x Atlas 910B3
- Runtime: OpenAI-compatible `server.py` with Continuous Batching enabled
- Branch/version: `release/0.0.7rc6`
- TP: 2
- PagedAttention: `page_size=16`
- Decode NPU Graph: enabled by server startup option
- Partial Prefix Cache: disabled for EvalScope random tests; exact Prefix Cache remains enabled by default
- Dataset: EvalScope random for throughput tests; `examples/benchmark_prefix_cache.py` for cache tests
- Stream: enabled

### Raw results

| Test | Tool | Concurrency | Requests | Temperature / Top-p | Avg input tokens | Avg output tokens | Output throughput | Total throughput | Req throughput | Avg latency | TTFT | TPOT | ITL |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v007rc6_greedy_p1` | EvalScope random | 1 | 20 | temperature=0 | 184.0 | 255.95 | 24.3149 tok/s | 41.7947 tok/s | 0.0950 req/s | 10.5259 s | 0.7063 s | 0.0385 s | 0.0384 s |
| `v007rc6_greedy_p4` | EvalScope random | 4 | 40 | temperature=0 | 184.0 | 215.775 | 62.4598 tok/s | 115.7218 tok/s | 0.2895 req/s | 13.7312 s | 1.5022 s | 0.0650 s | 0.0568 s |
| `same` | Prefix cache script | 1 | 20 | temperature=0 | ~62.0 | 256.0 | 26.56 tok/s/request-time | 388.45 tok/s/wall | n/a | 9.6386 s | 0.0229 s | n/a | n/a |
| `random` | Prefix cache script | 1 | 20 | temperature=0 | ~312.5 | 256.0 | 12.37 tok/s/request-time | 231.10 tok/s/wall | n/a | 20.6874 s | 10.7590 s | n/a | n/a |

### Interpretation

The `v0.0.7rc6` EvalScope random result confirms the `v0.0.7rc5` TTFT regression was fixed:

- `v0.0.7rc5` greedy p1 random had Output 16.4127 tok/s and TTFT 5.8119 s.
- `v0.0.7rc6` greedy p1 random has Output 24.3149 tok/s and TTFT 0.7063 s.
- Output throughput recovered by about 48.1% relative to the rc5 regression run.
- TTFT dropped by about 87.8%.
- ITL is 38.4ms, essentially the same decode speed as the healthy v0.0.7 line.

Compared with `v0.0.7rc1` greedy p1 random:

- Output throughput is 24.3149 vs 24.2603 tok/s, effectively unchanged (+0.2%).
- TTFT is 0.7063 vs 0.6670 s. The new run also has a longer measured input length: 184 vs 156 tokens, so the small TTFT increase is expected.
- Decode ITL is 38.4ms vs 38.6ms, effectively unchanged.

For concurrency 4:

- Output throughput is 62.4598 tok/s, close to the previous v0.0.7rc1 value of 63.3523 tok/s.
- Total throughput is higher, 115.7218 vs 102.2360 tok/s, but this run has longer inputs and shorter average outputs, so total throughput is not a clean improvement signal.
- TTFT improved from 2.2958s in rc1 to 1.5022s in this run, but TPOT worsened from 53.0ms to 65.0ms. Because average output tokens dropped from 254.125 to 215.775, this run should be treated as broadly comparable rather than strictly faster.

The prefix cache script shows exact Prefix Cache is still working:

- First repeated prompt TTFT: 0.344s.
- Requests 2-20 TTFT: about 5-6ms.
- Avg TTFT: 22.9ms.

The `benchmark_prefix_cache.py random` result is not directly comparable to EvalScope random. It produced about 312.5 input tokens per request (`568.5 total - 256 output`), while EvalScope random used 184 input tokens. Its high TTFT around 10.76s is therefore a long-prefill/no-cache control case, not evidence that rc6 random serving is slow.
