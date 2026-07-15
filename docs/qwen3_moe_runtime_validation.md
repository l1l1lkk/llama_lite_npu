# Qwen3 MoE Runtime 正确性验证

本文记录通用 softmax top-k router、routed-expert executor 与连续 TP/EP
placement 的正确性证据。它面向维护者和推理引擎面试复盘：重点不是“模型能跑”，
而是说明 oracle 如何保持独立、并行分片怎样闭合、Graph 如何证明没有改变冻结输入的
行为，以及数值近并列时为什么不能把跨并行模式 token 完全一致当成唯一标准。

架构背景见 [Qwen3 MoE Runtime 设计](qwen3_moe_runtime_design.md)，精简机器数据见
[summary.json](validation_data/20260715_qwen3_30b_a3b_moe_runtime/summary.json) 和
[manifest.json](validation_data/20260715_qwen3_30b_a3b_moe_runtime/manifest.json)。

## 1. 验证目标与不变量

本轮冻结源码 commit 为
`84239ed2e2afd9a277fdc2a5becab8623c1d31c0`，验证以下不变量：

- router 仍执行 `linear -> FP32 softmax -> top-k -> optional renorm -> cast back`，
  `RoutingResult` 同时支持字段访问与三元 tuple 解包；
- generic/compatibility router 与 executor 在同权重、同输入下行为一致；
- `ExpertPlacement` 只表达连续 TP intermediate shard 或连续 EP expert slice，且不进入
  state dict；
- `Qwen3SparseMoeBlock` 的 parameter key、`last_router_logits` 和输出契约不变；
- eager、GMM、routed GEMV 对独立 reference 对齐；
- TP/EP 的 rank-local contribution、partition sum、post-allreduce 与 ownership 闭合；
- NPUGraph 实际 capture/replay，冻结 TP greedy 输出与 eager 相同且无 fallback。

这些不变量共同覆盖接口、数值、分片、collective 和 Graph，不能由单一“生成文本看起来
正常”替代。

## 2. 环境与冻结模型

真实 NPU 验证环境为 Ascend 910B3、Ubuntu 22.04.5 aarch64、Python 3.10.12、
PyTorch 2.7.1、torch_npu 2.7.1、CANN 8.5.0、driver 26.0.rc1。

完整模型为 Qwen3-30B-A3B：48 层、`H=2048`、`E=128`、`K=8`、MoE
intermediate size `I=768`，`norm_topk_prob=true`。checkpoint artifact 是 BF16，
加载和分片后 runtime 转成 FP16；因此 D2 主 oracle 明确采用：

```text
checkpoint BF16 -> FP16 quantized values -> CPU FP32 math
```

checkpoint SHA256 为
`d9456e599c153a1b62c74ce4da9e796dbe58c683c3335358b6258af737b59fe1`，
config SHA256 为
`2850ddb3bf7aecad20b611e2d44f3077fc8193f4827c93beddd4c02ad63c2297`。
仓库只保留这些指纹和精简数据，不包含 61 GB checkpoint、raw tensor 或完整服务器日志。

## 3. Phase 1-3：CPU Reference 与兼容边界

Phase 1 在 `tests/reference/moe_reference.py` 建立纯 PyTorch CPU reference。它不导入
生产 router、executor、routing helper 或 kernel，内部用 FP32 计算 router softmax 和
expert 累加，并显式表达 GMM-native runtime layout：

```text
router  [E, H]
gate_up [E, H, 2I]
down    [E, I, H]
```

Phase 2 引入 `RoutingResult` 与 `SoftmaxTopKRouter`，保留
`Qwen3MoeTopKRouter` 兼容符号、tuple ABI、权重 key 与 `last_router_logits`。Phase 3
引入不可变 `ExpertPlacement` 和 `RoutedExpertExecutor`，保留 `Qwen3MoeExperts`
兼容子类、constructor/public 属性、backend 选择、all-reduce 位置和执行算法。

CPU 测试覆盖 T=0/1/多 token、top-k、renorm、non-contiguous 输入、极端有限 logits、
TP intermediate decomposition 和 EP expert contribution。inference-only 契约不要求
backward，NaN/Inf 不定义为稳定跨设备 ABI，tie 不锁定 expert id 顺序。

## 4. Phase 4A：单卡 NPU、GMM/GEMV 与 Graph

在加入本阶段 durable regression test 之前，物理 NPU 6 和 7 各自实际执行已有 5 项
正式测试，均为 5/5 通过、0 fail/error/skip：

- GMM 与 eager；
- validation mode；
- routed GEMV 与 eager；
- EP local contribution；
- dynamic routing NPUGraph capture/replay。

Graph 测试确实进入 capture/replay，没有 skip 或 correctness fallback。ArgSort 的 AiCPU
告警只记录为性能风险，不作为 correctness 失败。一次性通用边界探针使用
`T=4/H=64/E=8/I=32/K=2`、FP16 NPU 与 FP32 CPU oracle；generic/compatibility
router 和 executor 最大差为 0，GMM/GEMV/block 对 oracle 的最大误差约 `2e-6`。

Phase 4C1 将该探针固化为第 6 项正式 NPU 测试。当前本地没有 NPU，只验证该 suite 可
发现 6 项并安全 skip；第 6 项尚未在服务器执行，状态为 **待 Phase 4C2 复跑**。因此
本文不能把历史 5/5 写成新 suite 的 6/6。

Phase 4A 原始清单 self SHA256 为
`1b3028449c0eb22335013c2e3be3bb4814d2c63fdc6165f219791efbe471ebe0`。

## 5. Phase 4B：TP2/EP2/Graph 三角 Smoke

三组生命周期使用相同模型、prompt 生成逻辑、seed 42、prompt length 64、batch 1、
greedy、generation length 16 和 page size 16；唯一预注册自变量是：

| 组别 | 并行模式 | MoE backend | Graph |
| --- | --- | --- | --- |
| A | TP2 | eager | off |
| B | EP2 | eager | off |
| C | TP2 | auto | on |

三组均完成双 rank 权重加载和生成，exit 0，无 OOM、HCCL error、ChildFailedError、
Traceback 或非有限异常。C 的 Graph counters 为
`attempts=1/captured=1/replays=15/fallbacks=0`。

A 与 C 的完整 16-token 序列和文本 SHA256 exact 相同，支持 TP eager 与真实 Graph
路径在冻结输入上行为一致。A 与 B 前 15 token 相同，仅 0-based token index 15
分叉：A 选择 27362，B 选择 17719。该现象触发诊断，而不是被隐藏或直接判为 EP bug。
本次生命周期耗时受权重加载和编译缓存影响，不能作为 benchmark 或性能结论。

## 6. D1：逐层捕获与初始 INCONCLUSIVE

D1 在不修改仓库算法的前提下捕获第 16 次 forward 的 logits、router、rank-local expert
输出、post-allreduce 与 decoder hidden/residual，并精确复现 A/B 原 token 与文本 SHA。

关键事实：

- 48/48 层同模式 rank0/rank1 post-allreduce 逐元素 exact；
- 48/48 层跨模式 post 与 decoder hidden 通过 `rtol=1e-2, atol=1e-2`；
- 唯一 router 顺序差异在 layer 32 的 expert 104/106，selected set 相同，且不是
  top-k 边界翻转；
- 两个最终候选在各自模式中的 winner margin 都是 `0.03125`，跨模式 logit 扰动也是
  `0.03125`，即一个 FP16 量化步；
- residual 从 layer 37 起部分 elementwise absolute gate 失败，layer 37 relative L2
  约 `0.24%`。

由于当时还没有独立 FP32 层 oracle，D1 严格判为 `INCONCLUSIVE`。local contribution
在 CPU FP32 相加只是一种诊断重构，不是 HCCL bit-exact oracle。

D1 原始清单 self SHA256 为
`92649b6e54fd616d3a469e43ebb68d2caef661da1a7b214a7cf9dedf0a6a843e`。

## 7. D2：独立 FP32 单层 Oracle

D2 只使用 D1 capture、checkpoint mmap 和独立 CPU PyTorch FP32 数学，禁止导入生产
MoE/router/executor、分片 helper 或 kernel。AST/symbol 扫描的禁止命中为 0。

目标层是 0、32、37、47，覆盖首个 MoE 数值差异、唯一 router order 差异、residual
首次绝对门失败和最大 post 差异。结果为
`BOTH_PATHS_REFERENCE_ALIGNED`：

- 4 层 × 2 模式 × 2 rank = 16 个 captured local contribution 全部对独立 FP32
  partial 通过 `1e-2/1e-2`；
- 4 层 × 2 模式 = 8 个 independent partial sum 对 EXEC_ORACLE 全部通过严格
  `rtol=1e-5, atol=1e-6`；
- 8 个 captured post 对 EXEC_ORACLE 全部通过 `1e-2/1e-2`；
- 8 个 router selected set 全部一致，128 条 ownership 失败为 0，非有限指标行为 0；
- TP rank0/rank1 覆盖 `I[0,384)`/`I[384,768)` 且无重叠；EP rank0/rank1 覆盖
  experts `[0,64)`/`[64,128)`，每个 selected expert 恰有一个 owner。

layer 32 的 B 路径把 104/106 logits 都量化为 `-2.876953125`，导致已选集合内部顺序
交换；独立 oracle 的 selected set 与 runtime 相同，K/K+1 边界没有翻转。这不是路由
选择错误。

D2 说明 D1 的 residual 超门是下游 FP16 数值差异累积的观测，不是已证明的 MoE
placement/executor/collective 缺陷。但 D2 只验证四个 MoE 层，不证明另外 44 层或完整
attention/norm decoder 的 FP32 正确性。

D2 原始清单 self SHA256 为
`ba7be8d6741ff1b50e3f8f9fcb12a7e56c0f0a3e024f27a6c14e86550c011029`。
完整 72/8/128 行表由仓库内 manifest 做 size/SHA256 封闭校验。

## 8. 最终 Correctness Contract

后续 MoE 变更应使用分层门，而不是只比较跨模式生成 token：

1. TP/EP 各自 rank-local 与 post 输出对独立 FP32 oracle 使用
   `rtol=1e-2, atol=1e-2`；容差需在新 dtype/NPU 上重新实测校准。
2. 独立 partition partial sum 对 EXEC_ORACLE 使用
   `rtol=1e-5, atol=1e-6`。
3. 同模式 rank post exact、所有 tensor finite、TP coverage/无重叠、EP ownership 唯一
   是硬门。
4. router 锁定 selected set、weights 和 K/K+1 boundary；真实 tie 不锁 selected order。
5. TP eager 与 TP Graph 对冻结 greedy 输入可以要求 token exact；TP 与 EP 由于 FP16
   分解和累加顺序不同，不要求 token bit-exact。
6. 当 winner margin 小于或等于跨模式数值扰动时，必须报告候选 logits、margin、扰动
   和首次差异位置，不能只报文本差异。
7. decoder residual 同时报告 raw max/mean、relative L2 和 elementwise close。本次约
   `0.24%` 只是观测值，不固化为通用阈值。

这套契约既能检出 ownership、partition 和 collective 错误，也允许可解释的低精度
近并列翻转。

## 9. 可复现命令

CPU reference、兼容和证据完整性：

```bash
python -m unittest tests.test_moe_validation_evidence -v
python -m unittest tests.models.test_moe_reference tests.models.test_qwen3_moe -v
```

本地无 NPU 时只验证 discover/skip；服务器 NPU 回归必须显式运行并保留真实结果：

```bash
python -m unittest tests.npu.test_qwen3_moe_gmm -v
python scripts/validate_release.py
```

公开复现不依赖内部服务器目录，而以 Git tracked compact data 和 manifest 为入口。原始
diagnostics 仅通过 self hash 建立来源链；raw `.pt`、完整日志和 checkpoint 不进入仓库。

## 10. 结论与限制

当前证据支持：Qwen3 softmax top-k router、连续 TP/EP placement、generic routed
executor、eager/GMM/GEMV 与 TP NPUGraph 在已覆盖范围内符合 reference correctness
契约。跨 TP/EP 的最后一个 greedy token 分叉由独立 oracle 证明不是 placement 或
collective 失败，且符合近似并列 logits 被 FP16 扰动放大的解释。

尚未覆盖 DeepSeek grouped top-k、sigmoid/route scale/correction bias、shared expert、
all-to-all、非连续 expert map、W8A8/其他量化、MLA，也没有形成性能结论。模型 smoke
仅一个 prompt 和 16 token；D2 仅四个 MoE 层。新增第 6 项 NPU 回归还必须在
Phase 4C2 真实复跑后才能更新为服务器通过。

本阶段 `VERSION` 仍为 `0.0.13rc3`；目标 `0.0.14rc1` 只在最终发布阶段更新。
