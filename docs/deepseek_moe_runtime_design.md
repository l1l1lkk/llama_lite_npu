# DeepSeekMoE Runtime 独立 Reference 与行为契约

## 1. 背景、面试价值与本阶段目标

Lite Llama NPU 在 `v0.0.14rc1` 已建立 `RoutingResult`、
`SoftmaxTopKRouter`、`ExpertPlacement` 与 `RoutedExpertExecutor`，并保持
Qwen3 兼容符号和 state dict 不变。但这些通用边界当前只实现 Qwen3 的
softmax top-k 与连续 TP/EP placement，不能据此声称已经支持 DeepSeekMoE。

第 2 周 Phase 6B 的目标是先建立一个可解释、可复核、与生产实现相互独立的
CPU 数学 oracle。它对面试和工程实践的价值在于：

- 能准确解释 DeepSeek-V2 与 DeepSeek-V3 路由差异，而不是只说“多了一个
  grouped top-k”；
- 能说明 correction bias 为什么只影响选择、不能污染最终 combine weight；
- 能把 routed expert、shared expert、TP intermediate partial 与 EP expert
  ownership 分开验证；
- 在生产重构前先冻结错误条件、dtype、shape、tie 和非有限输入边界；
- 为后续 NPU、Graph、量化和通信适配提供独立 oracle，而不是让优化路径验证
  自己。

本阶段只新增测试侧 reference、CPU 单元测试和本文档；没有修改
`lite_llama/**`，没有接入真实 checkpoint loader，也没有实现 DeepSeek 模型。

## 2. 独立性与总体 API

独立实现位于：

```text
tests/reference/deepseek_moe_reference.py
```

它只导入标准库、`torch` 和 `torch.nn.functional`，不导入或调用生产代码中的
router、executor、MoE block、TP shard helper、GMM、GEMV 或 NPU kernel。
所有公开函数均使用 `torch.no_grad()`，只接受 CPU tensor，内部浮点运算统一为
FP32。

最小 API 为：

```python
DeepSeekRoutingSpec(...)

deepseek_route_reference(
    hidden_states,
    router_weight,
    spec,
    correction_bias=None,
)

deepseek_routed_experts_reference(
    hidden_states,
    selected_experts,
    routing_weights,
    gate_up_weight,
    down_weight,
    expert_start=0,
)

deepseek_shared_expert_reference(
    hidden_states,
    gate_up_weight=None,
    down_weight=None,
)

deepseek_moe_reference(...)
```

`deepseek_moe_reference` 返回：

```text
output, router_logits, routing_weights, selected_experts
```

其中 output、logits 和 weights 为 FP32，selected expert 是 global expert ID，
dtype 固定为 `torch.int64`。空 token 输入 `T=0` 分别返回 `[0,H]`、
`[0,E]`、`[0,K]` 和 `[0,K]`。

## 3. DeepSeek-V2/V3 路由数学

设：

- flat hidden states 为 `X ∈ R[T,H]`；
- router weight 为 `Wg ∈ R[E,H]`；
- expert 数为 `E`；
- 每个 token 选择 `K` 个 expert；
- expert 被均匀划分为 `G=n_group` 个连续组；
- 每个 token 先保留 `Gk=topk_group` 个组。

### 3.1 原始分数

router logits 统一在 FP32 中计算：

```text
L = X_fp32 @ Wg_fp32^T
```

DeepSeek-V2 的原始分数为：

```text
S = softmax(L)
```

DeepSeek-V3 的原始分数为：

```text
S = sigmoid(L)
```

本 reference 只接受已知的两个组合：

| 模型语义 | score_func | topk_method | correction bias |
|---|---|---|---|
| V2 | `softmax` | `group_limited_greedy` | 禁止 |
| V3 | `sigmoid` | `noaux_tc` | 必需 |

其他组合不会静默退化，而是 fail-closed 抛出明确错误。

### 3.2 correction bias 只参与选择

V3 使用 `e_score_correction_bias=b` 时：

```text
selection_scores = S + b
combine_scores   = S
```

bias 同时影响组选择和组内 expert 选择，但最终 routing weight 必须从未加 bias
的原始 `S` 按 selected ID gather。测试直接构造一个被 bias 推入 top-k 的 expert，
验证返回 weight 等于其原始 sigmoid score，而不等于加 bias 后的数值。

### 3.3 组选择

每组 expert 数为：

```text
Eg = E / G
```

V2 `group_limited_greedy` 的组分数为选择分数的组内最大值：

```text
group_score[g] = max(selection_scores[g, :])
```

V3 `noaux_tc` 的组分数为组内最高两个修正后选择分数之和：

```text
group_score[g] = top1(selection_scores[g, :])
               + top2(selection_scores[g, :])
```

先选择 `topk_group` 个组，屏蔽其他组，再从保留组中选择 K 个 expert。配置必须
满足：

```text
E % G == 0
1 <= topk_group <= G
1 <= K <= topk_group * Eg
```

`noaux_tc` 还要求 `Eg >= 2`，否则无法定义组内 top-2 sum。

### 3.4 combine、归一化与缩放

得到 global selected IDs 后，从原始分数 gather：

```text
C = gather(S, selected_experts)
```

若 `norm_topk_prob=True`：

```text
C = C / sum(C, dim=-1)
```

即使输入和logits有限，极端负的FP32 sigmoid logits也可能数值下溢，使所有
selected score均为0。若请求归一化，reference会在除法前检查逐token的
selected-weight sum；分母必须finite且严格大于0，否则明确抛出`ValueError`。
这里不会添加epsilon或clamp来伪造新的路由数学，也不会静默返回NaN。
`norm_topk_prob=False`时不执行这项归一化分母检查，合法的全零sigmoid combine
weights保持有限零值并继续应用route scale。该规则是本独立reference定义的
数值安全域，不能据此声称DeepSeek、vLLM或其他生产实现已经存在相同检查。

最后无论是否归一化，都应用：

```text
routing_weights = C * routed_scaling_factor
```

实现和测试分别锁定“先归一化、再缩放”以及 `norm=False` 仍然缩放的行为。
缩放后还必须保持routing weights全部finite；有限但超出FP32可表示结果范围的
scale会明确抛出`ValueError`，而不是返回Inf。该检查同样只定义reference的
安全域，不改变常规DeepSeek route scale下的计算值。

### 3.5 tie、determinism 与非有限输入

- tie 不绑定 selected group 或 expert 的内部顺序；测试只锁合法集合、选中组容量
  和等权性质；
- 对固定 seed、有限且非 tie 的输入，重复调用要求输出完全一致；
- NaN 和 Inf 明确属于 out-of-contract，不定义跨设备 ABI，reference 会拒绝；
- 不支持的 `score_func`（包括 V4 的 `sqrtsoftplus`）和不支持的
  `topk_method` 必须抛错，不能回退到 softmax 或 greedy。

## 4. Routed expert 与 shared expert

### 4.1 Routed expert

reference 使用显式 token/slot/expert 循环，不复用生产 grouping 或排序算法。
对于 expert e：

```text
[gate, up] = X @ W_gate_up[e]
activated  = silu(gate) * up
expert_out = activated @ W_down[e]
output    += routing_weight * expert_out
```

global expert ID 不属于 `[expert_start, expert_start + E_local)` 时忽略。这只描述
本项目当前 replicated-token EP 中一个 rank 的 local contribution，不实现
collective 或 token dispatch。

### 4.2 Shared expert

shared expert 是独立 SwiGLU：

```text
[shared_gate, shared_up] = X @ W_shared_gate_up
shared_out = (silu(shared_gate) * shared_up) @ W_shared_down
```

`n_shared_experts=0` 用两个 `None` weight 表达，返回 FP32 零张量。一个或多个
shared expert 在 reference 中可等价合并为更宽的 shared intermediate：

```text
I_shared = n_shared_experts * moe_intermediate_size
```

完整 MoE 输出为：

```text
output = routed_output + shared_output
```

shared expert 不进入 routed expert ID 空间，也不参与 grouped top-k。

## 5. TP/EP 的单进程 reference 语义

### 5.1 TP intermediate shard

TP rank 仍持有全部 routed experts，但只持有连续 intermediate 区间。对完整
gate/up 的 gate 半和 up 半分别切相同 `[start,end)`，再拼回本 rank 的
`[H,2*I_local]`；down 切 `[I_local,H]`。各 rank FP32 partial sum 应与完整
routed oracle 在严格门内一致：

```text
sum(tp_routed_partial[rank]) == routed_full
```

shared expert以相同方式切 intermediate，独立验证：

```text
sum(tp_shared_partial[rank]) == shared_full
```

reference 不执行 all-reduce；它只验证 reduce 前的数学分解，因此生产路径后续
必须证明 routed/shared 各自只 reduce 一次。

### 5.2 连续 EP expert slice

EP rank 持有连续 global expert 区间和完整 intermediate。所有 rank 接收相同
token/routing，非本地 expert 贡献为零：

```text
sum(ep_local_contribution[rank]) == routed_full
```

测试锁定每个被选 global expert 恰好属于一个 rank，没有丢失或重复 ownership。
这不是 DeepSeek 原生 all-to-all、EPLB 或非连续 expert map；这些能力不得从
当前 reference 外推。

## 6. checkpoint 与 runtime 布局契约

本阶段不实现 loader，但明确后续适配必须面对的 checkpoint 语义：

```text
model.layers.L.mlp.gate.weight                         [E,H]
model.layers.L.mlp.gate.e_score_correction_bias       [E]
model.layers.L.mlp.experts.e.gate_proj.weight         [I,H]
model.layers.L.mlp.experts.e.up_proj.weight           [I,H]
model.layers.L.mlp.experts.e.down_proj.weight         [H,I]
model.layers.L.mlp.shared_experts.gate_proj.weight    [I_shared,H]
model.layers.L.mlp.shared_experts.up_proj.weight      [I_shared,H]
model.layers.L.mlp.shared_experts.down_proj.weight    [H,I_shared]
```

测试 reference 直接接受项目计划使用的 runtime 侧布局：

```text
hidden                  [T,H]
router_weight           [E,H]
routed_gate_up          [E_local,H,2I]  # gate 后 up
routed_down             [E_local,I,H]
shared_gate_up          [H,2I_shared]   # gate 后 up
shared_down             [I_shared,H]
```

这些布局只是行为契约，不能写成“当前 loader 已支持”。真实 checkpoint key、
transpose、stack、TP/EP shard 和 state dict 兼容必须在后续独立阶段验证。

## 7. 与 v0.0.14rc1 通用边界的未来关系

后续生产阶段可以复用的边界：

- `RoutingResult` 继续保持三字段 tuple ABI：raw logits、最终 combine weights、
  global selected expert IDs；
- `RoutedExpertExecutor` 可继续消费 selected IDs 和 combine weights；
- `ExpertPlacement` 可继续描述现有连续 TP intermediate shard 和 EP expert
  slice。

但 Phase 6B 没有修改这些生产类，也没有声称它们已支持 DeepSeek。后续最小
生产设计应：

- 新增具体 grouped router，并在内部显式区分 selection score 与 combine
  score；
- 不扩展 `RoutingResult` 字段，避免破坏 Qwen tuple ABI；诊断 margin 应放在
  测试或独立 diagnostics 中；
- 将 shared expert 作为独立模块，由上层 MoE block 与 routed output 组合；
- 对未知 DeepSeek Graph/backend 明确 fail-closed，不能落入默认可用路径；
- 保持 Qwen softmax router、Qwen expert executor、state dict 和默认行为不变。

## 8. 冻结官方参考与许可证边界

本阶段冻结以下官方源：

- DeepSeek-V2 `ec98ee3cbffc32104cd55dba8af884b3d772602a`：
  <https://github.com/deepseek-ai/DeepSeek-V2/commit/ec98ee3cbffc32104cd55dba8af884b3d772602a>
- DeepSeek-V2-Lite `604d5664dddd88a0433dbae533b7fe9472482de0`：
  <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/commit/604d5664dddd88a0433dbae533b7fe9472482de0>
- DeepSeek-V3 `9b4e9788e4a3a731f7567338ed15d3ec549ce03b`：
  <https://github.com/deepseek-ai/DeepSeek-V3/commit/9b4e9788e4a3a731f7567338ed15d3ec549ce03b>
- DeepSeek-V4 `b5968e9190ef611bbf34a7229255be88a0e937c1`：
  <https://huggingface.co/deepseek-ai/DeepSeek-V4/commit/b5968e9190ef611bbf34a7229255be88a0e937c1>
- vLLM `3935829f897616fdb14fb7324875d41dd4914dff`：
  <https://github.com/vllm-project/vllm/commit/3935829f897616fdb14fb7324875d41dd4914dff>
- vLLM-Ascend `32b086054694fcae13123ecc76a65825746e3565`：
  <https://github.com/vllm-project/vllm-ascend/commit/32b086054694fcae13123ecc76a65825746e3565>

DeepSeek官方代码仓库采用MIT或模型仓库声明的相应许可证；vLLM与
vLLM-Ascend采用Apache-2.0。当前Lite Llama NPU仓库没有顶层
`LICENSE`/`NOTICE`，因此本阶段只依据公开数学定义和接口思想独立实现，没有
复制上述项目的非平凡源码、注释或kernel。任何源码复用和attribution策略必须
先由维护者明确，不能在后续实现中机械复制。

## 9. TDD 矩阵与真实结果

### 9.1 RED

先新增仅引用目标API的最小测试，在reference文件不存在时运行：

```text
python -m unittest tests.models.test_deepseek_moe_reference -v
```

真实结果：测试模块导入失败，错误为
`ModuleNotFoundError: No module named 'tests.reference.deepseek_moe_reference'`，
exit code `1`。RED 状态没有 stage、commit 或 push。

### 9.2 Phase 6B focused GREEN

实现独立 reference 和完整测试矩阵后运行同一命令：

```text
python -m unittest tests.models.test_deepseek_moe_reference -v
```

真实结果：24 tests 全部通过，exit code `0`。

24项测试覆盖：

- T=0、1、多token以及多个E/K/group/topk_group组合；
- softmax、sigmoid、bias selection-only；
- group max、group top-2 sum、归一化与route scale；
- group/K边界、tie集合语义、非连续输入、极端有限logits、V3 sigmoid归一化
  下溢fail-closed与determinism；
- hotspot/空expert、shared 0/1/多等价宽度；
- routed/shared TP partial sum、EP ownership/local sum；
- inference-only、shape/配置/非有限输入fail-closed；
- V4 score/method拒绝和AST/import独立性检查。

Phase 6B-R1针对有限V3 logits的sigmoid下溢补充了定向RED：

```text
python -m unittest \
  tests.models.test_deepseek_moe_reference.\
DeepSeekMoeRoutingReferenceTest.\
test_sigmoid_normalization_underflow_fails_closed_only_when_enabled -v
```

修复前该测试因`ValueError not raised`失败，exit code `1`；只读探针同时确认
返回weights为`[[nan, nan]]`且`finite=False`。在归一化除法前增加positive-finite
分母检查后，同一定向测试通过，exit code `0`；`norm=False`分支仍返回有限全零
weights，没有引入epsilon、clamp或新的fallback数学。

同一安全域审计还发现，有限但极大的route scale可以让FP32输出溢出为Inf。
定向测试`test_routing_weight_scaling_overflow_fails_closed`在输出finite检查加入前
因`ValueError not raised`失败，exit code `1`；缩放后增加明确finite检查后通过，
exit code `0`。常规有效route scale的数值路径保持不变。

### 9.3 完整GREEN门

Qwen reference/runtime兼容回归：

```text
python -m unittest \
  tests.models.test_moe_reference \
  tests.models.test_qwen3_moe -v
```

真实结果：56 tests全部通过，exit code `0`。

版本文档与既有MoE compact evidence门：

```text
python -m unittest \
  tests.test_repository_docs \
  tests.test_moe_validation_evidence -v
```

真实结果：8 tests全部通过，exit code `0`。

独立性测试单独执行：

```text
python -m unittest \
  tests.models.test_deepseek_moe_reference.\
DeepSeekMoeReferenceIndependenceTest -v
```

真实结果：1 test通过，exit code `0`。AST/import检查只允许标准库和PyTorch，
并禁止生产MoE符号、shard helper与优化kernel名字。

完整release validator：

```text
python scripts/validate_release.py
```

真实结果：Markdown UTF-8、VERSION文档同步、compile和diff-check通过；默认
discovery运行160 tests并全部通过，exit code `0`。当前validator的默认
discovery不替代显式的`tests.models.test_deepseek_moe_reference`命令，因此24项
新增契约测试必须作为独立硬门保留。

最后独立执行：

```text
git diff --check
```

真实结果：exit code `0`。

以上均为本地CPU结果，不包含服务器或NPU执行，也不能用来声称性能、Graph或
完整DeepSeek模型能力。

CPU reference内部统一FP32，分片partial sum测试使用`rtol=1e-5, atol=1e-6`
的严格门。FP16/BF16输入只验证会提升到FP32 oracle；未来生产FP16/BF16与
NPU对齐可暂以`rtol=1e-2, atol=1e-2`作为起始建议，但必须同时报告原始误差和
relative L2，并在真实NPU阶段重新校准，不能把建议值写成最终跨设备ABI。

## 10. 明确不在本阶段的能力

Phase 6B 明确不实现或验证：

- DeepSeek-V4 `sqrtsoftplus`、static-hash MoE；
- MLA、完整DeepSeek CausalLM、tokenizer或生成；
- all-to-all、EPLB、非连续expert map或token dispatcher；
- 量化、W8A8、量化kernel和checkpoint量化格式；
- NPU、GMM/GEMV、HCCL、Graph capture/replay；
- 性能优化、benchmark、吞吐或延迟结论；
- backward、训练或梯度正确性。

本阶段 `VERSION` 仍为 `0.0.14rc1`。DeepSeekMoE属于功能级更新，目标
`0.0.15rc1` 只能在全部生产、checkpoint、CPU/NPU和发布门完成后的最终发布
阶段更新。

## 11. 下一阶段停止门

Phase 6B 完成后必须先由控制任务审查 reference、测试和本文档。本阶段不进入
Phase 6C。若后续授权 Phase 6C，才可以建立生产 grouped router policy，并以
本reference作为独立oracle，同时继续锁定Qwen默认行为与state dict。
