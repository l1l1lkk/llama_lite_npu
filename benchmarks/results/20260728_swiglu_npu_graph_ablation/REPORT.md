# Dense Decode NPU Graph × Triton SwiGLU 四象限验证报告

## 1. 结论

本次验证确认：对于无路由、每个 token 重复执行同一算子链的 Dense decode，
NPU Graph 有非常明显的收益，并且能够让 Triton SwiGLU 的设备侧融合收益更稳定地
转化为端到端 TPOT/E2E 收益。

- 未融合基线开启 Graph 后，TPOT 改善 **90.70%～91.53%**；
- Triton 融合开启 Graph 后，TPOT 改善 **91.03%～91.77%**；
- 在 Graph-on 条件下，Triton 融合相对未融合仍额外改善 TPOT
  **2.82%～3.45%**；
- 两个 Graph-on 分支正式阶段均为 **2 次预捕获、930 次 replay、
  0 次新 capture、0 fallback**；
- NPU Profiler 显示 `FftsPlusTaskLaunch` 调用数从 1826 降至 90，
  减少 **95.07%**，其 Host 时间减少 **94.65%**；
- 同时设备算子总执行时间仅从 157094.71 us 变为 156157.62 us，
  变化约 **0.60%**，说明核心收益来自 Host/Runtime 发射与调度开销下降，
  不是计算量减少。

因此，前一轮 Graph-off 测试中出现的“单 kernel 更快、TPOT/E2E 没有变快”
确实属于 Graph 可以显著改善的 launch-bound 问题。

## 2. 测试对象

| 项目 | 未融合基线 | Triton 融合 |
|---|---|---|
| branch | `test/swiglu-unfused-baseline` | `feature/swiglu-fused-optimization` |
| commit | `ce55d4b08d179a84c2885432ff004ee202b6e324` | `098f35ec305fd89ad60e23400ff6cee4af9c24c9` |
| 模型 | Qwen3-1.7B Dense | Qwen3-1.7B Dense |
| TP | 2 | 2 |
| NPU | 6、7 | 6、7 |
| 容器 | `triton-llama-lite` | `triton-llama-lite` |
| batching | legacy，单请求 | legacy，单请求 |

正式矩阵：

- 实现：未融合、Triton 融合；
- NPU Graph：off、on；
- prompt length：128、512；
- output tokens：32；
- concurrency：1；
- 每个 cell：3 轮，每轮 4 个正式请求，共 12 个请求；
- 每轮正式请求前另有 1 个不计入统计的 warmup；
- Graph-on 服务正式测试前分别预跑 p128、p512，提前完成图捕获。

## 3. 四象限正式结果

### 3.1 绝对结果

| 实现 | Graph | Case | TTFT (ms) | TPOT (ms) | E2E (ms) | Throughput (tok/s) |
|---|---:|---|---:|---:|---:|---:|
| 未融合 | off | p128 | 82.673 | 76.299 | 2447.954 | 13.076 |
| 未融合 | on | p128 | 85.693 | 6.462 | 286.010 | 111.786 |
| Triton 融合 | off | p128 | 84.197 | 75.787 | 2433.588 | 13.150 |
| Triton 融合 | on | p128 | 85.357 | 6.239 | 278.762 | 114.681 |
| 未融合 | off | p512 | 462.990 | 75.131 | 2792.037 | 11.460 |
| 未融合 | on | p512 | 463.598 | 6.988 | 680.230 | 47.020 |
| Triton 融合 | off | p512 | 462.636 | 75.742 | 2810.632 | 11.384 |
| Triton 融合 | on | p512 | 461.396 | 6.791 | 671.920 | 47.605 |

### 3.2 Graph-on 相对 Graph-off

| 实现 | Case | TTFT | TPOT | E2E |
|---|---|---:|---:|---:|
| 未融合 | p128 | 回退 3.65% | **改善 91.53%** | **改善 88.32%** |
| 未融合 | p512 | 回退 0.13% | **改善 90.70%** | **改善 75.64%** |
| Triton 融合 | p128 | 回退 1.38% | **改善 91.77%** | **改善 88.55%** |
| Triton 融合 | p512 | 改善 0.27% | **改善 91.03%** | **改善 76.09%** |

TTFT 没有获得类似 TPOT 的收益符合执行边界：当前 NPU Graph 只包
`seq_len == 1` 的 decode，prefill 仍走 eager。p128 的几毫秒 TTFT 波动小于
单轮间波动量级，不应解释为 Graph 对 prefill 的稳定负收益。

### 3.3 Graph-on 下 Triton 融合的增量收益

| Case | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| p128 | 改善 0.39% | **改善 3.45%** | **改善 2.53%** |
| p512 | 改善 0.47% | **改善 2.82%** | **改善 1.22%** |

Graph-off 时融合收益受 Host 发射开销和运行波动掩盖：p128 TPOT 改善
0.67%，p512 TPOT 回退 0.81%。Graph-on 后两个长度都稳定为正，说明 Graph
消除了主要的提交噪声，Triton 的设备侧 kernel 收益开始进入端到端路径。

相对“未融合 + Graph off”的最终组合收益：

| Case | TTFT | TPOT | E2E |
|---|---:|---:|---:|
| p128 | 回退 3.25% | **改善 91.82%** | **改善 88.61%** |
| p512 | 改善 0.34% | **改善 90.96%** | **改善 75.93%** |

## 4. Graph 是否真实生效

### 4.1 正式阶段计数器

| 实现 | 预捕获 captures | 正式 capture delta | 正式 replay delta | fallback delta |
|---|---:|---:|---:|---:|
| 未融合 + Graph-on | 2 | 0 | 930 | 0 |
| Triton 融合 + Graph-on | 2 | 0 | 930 | 0 |

930 次 replay 与测试负载完全一致：

`2 cases × 3 repeats × (4 formal + 1 warmup) × 31 decode intervals = 930`

这说明正式指标未包含首次捕图成本，Triton kernel 也确实进入了整模型 Graph，
没有静默回退到 eager。

### 4.2 稳定性

Graph-on 的三轮 TPOT 均值变异系数：

| 实现 | p128 CV | p512 CV |
|---|---:|---:|
| 未融合 | 0.932% | 0.504% |
| Triton 融合 | **0.161%** | **0.125%** |

两个长度下 Triton + Graph 的轮间波动均低于 0.2%，收益稳定。

## 5. NPU Profiler 证据

Profiler 采用 CANN `msprof` dynamic attach，采集对象为：

- Card/Device：物理 NPU 6；
- Process：TP rank0 的实际 `server.py` 进程；
- Workload：Triton 融合分支，p128、output 32、单请求；
- 对照：Graph off 与 Graph on；
- Graph-on 在采集前已捕获，采集请求产生 31 次 replay、0 fallback。

### 5.1 Host Runtime API

| Runtime API | Graph off count | Graph on count | 数量减少 | off time | on time | 时间减少 |
|---|---:|---:|---:|---:|---:|---:|
| `FftsPlusTaskLaunch` | 1826 | 90 | **95.07%** | 16403.59 us | 876.92 us | **94.65%** |
| `EventRecord` | 5478 | 270 | **95.07%** | 49316.71 us | 2633.47 us | **94.66%** |
| `ContextGetCurrent` | 1826 | 90 | **95.07%** | 701.36 us | 31.10 us | **95.57%** |

### 5.2 设备侧

| 指标 | Graph off | Graph on | 变化 |
|---|---:|---:|---:|
| 设备算子记录数 | 14796 | 14425 | -2.51% |
| 设备算子累计时间 | 157094.71 us | 156157.62 us | -0.60% |

设备工作量基本不变，而 Host launch/event 数量下降约 95%。因此该场景的
Graph 收益类型可以明确判定为：

> **Host/Runtime launch 与同步调度优化，而不是访存或计算优化。**

Profiler 会扰动绝对延迟，所以正式收益使用无 Profiler 的四象限结果；
Profiler 只用于证明因果来源。

## 6. 输出一致性

同一实现的 Graph on/off 输出：

| 实现 | p128 | p512 |
|---|---:|---:|
| 未融合 Graph on vs off | 12/12，100% | 12/12，100% |
| Triton Graph on vs off | 12/12，100% | 12/12，100% |

因此 NPU Graph 没有改变任一实现的生成结果。

跨实现比较中，p128 为 12/12 一致；p512 为 9/12 一致。差异固定发生在同一个
prompt，并且在 Graph off/on 下完全相同，说明这是既有的融合分支生成差异，
不是 NPU Graph 引入。融合算子的数值 allclose 门禁见原始 SwiGLU 融合报告。

## 7. 文件说明

- `comparison.json`：完整聚合结果、收益、CV、计数器和输出一致性；
- `comparison.csv`：四象限核心指标；
- `baseline/`、`fused-triton/`：24 份正式结果及服务日志/metrics；
- `profiler/graph-off/`：Graph-off 原始采集、MindStudio JSON/DB/CSV；
- `profiler/graph-on/`：Graph-on 原始采集、MindStudio JSON/DB/CSV；
- `profiler.tar.gz`：完整 profiler 归档；
- `manifest.json`：文件 SHA256 清单。

Profiler 归档 SHA256：

`771d2ff9d7ef2c600d2b6766d1b8eedcfb0ec4fb8183629aaca2105bdd512873`

## 8. 最终判断

Dense decode 的算子和 shape bucket 可复用时，NPU Graph 对当前问题不是
“小幅补偿”，而是决定性优化：它消除了逐 token、逐算子的 Host 发射链，
使 TPOT 降低约 91%，并让 Triton SwiGLU 在 Graph-on 稳态下继续贡献
2.82%～3.45% 的额外 TPOT 收益。

推荐部署组合为：

> **Triton SwiGLU 融合 + 服务启动预捕获 + NPU Graph replay**

生产门禁应继续要求正式流量阶段 `capture_delta=0`、`fallback_delta=0`，
并按 batch size 与 128-token sequence bucket 完成预捕获。
