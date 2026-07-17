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

## 12. Phase 6C：生产 Grouped Router 边界

### 12.1 本阶段目标与API

Phase 6C只把Phase 6B已经冻结的V2/V3 valid-domain routing数学接入生产侧，
没有创建DeepSeek MoE block、shared expert、模型配置、checkpoint loader或模型
注册。新增生产API仍位于依赖轻的`lite_llama/models/moe.py`：

```python
@dataclass(frozen=True)
class GroupedTopKConfig:
    num_groups: int
    topk_groups: int
    score_func: str
    topk_method: str
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.0

class DeepSeekGroupedTopKRouter(nn.Module):
    ...
```

router constructor显式接收：

```text
hidden_size, num_experts, top_k,
num_groups, topk_groups, score_func, topk_method,
norm_topk_prob, routed_scaling_factor, dtype
```

构造时会把grouped policy冻结为`GroupedTopKConfig`。该配置只保存Python标量，
不进入state dict，也不会在forward中产生tensor值驱动控制流。只接受已审计的
两个组合：

```text
softmax + group_limited_greedy
sigmoid + noaux_tc
```

`sqrtsoftplus`、static hash、未知score/method、非法group划分、K超过已选组
容量及noaux组内不足两个expert均fail-closed。

### 12.2 参数与state dict

两个router都持有：

```text
weight [E,H]，dtype由constructor指定
```

V2 `group_limited_greedy` state dict精确为：

```text
weight
```

V3 `noaux_tc`额外创建可加载的FP32参数：

```text
e_score_correction_bias [E]
```

因此V3 router state dict精确为：

```text
weight
e_score_correction_bias
```

本阶段没有创建上层DeepSeek block。未来嵌套为`gate`后才会自然形成
`gate.weight`和`gate.e_score_correction_bias`；这里不能写成当前checkpoint
loader已经支持这些key。

### 12.3 valid-domain生产数学

forward先把任意前导维输入flatten为`[-1,H]`：

```text
router_logits = F.linear(flat_hidden, weight)
```

selection内部统一FP32：

```text
V2 original_scores = softmax(router_logits.float())
V3 original_scores = sigmoid(router_logits.float())
```

V3 correction bias只构造：

```text
selection_scores = original_scores + e_score_correction_bias
```

组选择和expert选择使用selection scores；最终combine仍从未加bias的
`original_scores`按selected IDs gather。V2组分数为组内max；V3/noaux组分数
为修正后组内top-2之和。选择top groups后再选择K个global expert。随后按静态
配置执行optional normalization和route scale。

返回值继续是既有三字段ABI：

```text
RoutingResult(router_logits, routing_weights, selected_experts)
```

`routing_weights`最终cast回`router_logits.dtype`，public expert IDs保持
`torch.int64`。`T=0`返回`[0,E]`与两个`[0,K]`；tie测试只锁合法集合和等权
性质，不锁内部顺序。

### 12.4 Reference与production职责差异

Phase 6B reference是CPU FP32、fail-closed oracle，负责定义非法数值域；它在
sigmoid归一化分母为0/非有限或route scale产生非有限weights时抛出明确错误。

生产router只覆盖本阶段已验证的valid domain。为保持设备热路径无host sync，
没有把reference的`.item()`式异常判断搬进forward，也没有用epsilon/clamp改变
公式。极端sigmoid下溢仍由reference定向测试锁定；后续若设备侧需要异常遥测，
必须另立不破坏Graph的设计，不能在当前hot path加入tensor值驱动Python分支。

FP32生产结果对独立reference使用严格对齐。FP16稳定margin样本使用
`rtol=atol=2e-3`，BF16使用`rtol=atol=1e-2`；这些仅是本地CPU测试容差，不是
NPU或跨设备最终ABI。

### 12.5 Qwen与既有执行路径兼容

Phase 6C没有修改以下既有类的AST：

```text
RoutingResult
SoftmaxTopKRouter
Qwen3MoeTopKRouter
ExpertPlacement
RoutedExpertExecutor
Qwen3MoeExperts
```

Qwen router constructor signature、`weight` state dict、softmax/normalization输出、
tuple ABI均继续由独立Qwen reference验证。既有Qwen block的`gate.weight`、
`experts.gate_up_weight`、`experts.down_weight`和`last_router_logits`继续由原回归
套件锁定。executor、GMM/GEMV、TP/EP reduce、Graph policy和server flag没有
修改。

新增router继续支持`spec_from_file_location`直接加载`moe.py`，没有导入完整
`lite_llama`包或新增第三方依赖。forward源码AST检查证明只存在基于冻结配置的
静态分支，并禁止`.item/.tolist/.cpu/.numpy`、日志、warning、动态adapter或
hidden/logits/weights驱动的Python `if`。

### 12.6 TDD与真实结果

RED命令：

```text
python -m unittest \
  tests.models.test_deepseek_moe_router.\
DeepSeekMoeRouterContractTest.\
test_grouped_topk_config_is_available -v
```

实现前真实结果：1 test error，
`AttributeError: module ... has no attribute 'GroupedTopKConfig'`，exit code `1`；
RED状态未stage、commit或push。

最小API加入后同一定向测试1/1通过，exit code `0`。完整Phase 6C focused命令：

```text
python -m unittest tests.models.test_deepseek_moe_router -v
```

真实结果：12 tests全部通过，exit code `0`。覆盖immutable config、非法组合/V4
拒绝、V2/V3 FP32 oracle矩阵、bias selection-only、norm/scale、T=0/1/multi、
FP16/BF16稳定margin、tie集合、state dict/dtype/load、Qwen兼容、direct loader
与hot-path AST。

独立reference回归：

```text
python -m unittest tests.models.test_deepseek_moe_reference -v
```

真实结果：24 tests全部通过，exit code `0`；两个Phase 6B文件SHA256保持不变。

Qwen reference/runtime兼容回归：

```text
python -m unittest \
  tests.models.test_moe_reference \
  tests.models.test_qwen3_moe -v
```

真实结果：56 tests全部通过，exit code `0`。

版本文档与既有MoE evidence门：

```text
python -m unittest \
  tests.test_repository_docs \
  tests.test_moe_validation_evidence -v
```

真实结果：8 tests全部通过，exit code `0`。

```text
python scripts/validate_release.py
```

真实结果：Markdown UTF-8、VERSION文档同步、compile和diff-check通过；默认
discovery运行160 tests并全部通过，exit code `0`。独立`git diff --check`也为
exit code `0`。本阶段没有运行服务器或NPU。

### 12.7 明确不包含的能力与版本

Phase 6C不包含shared expert、DeepSeek MoE block、layer schedule、配置/模型注册、
checkpoint转换或加载、all-to-all/EPLB、非连续expert map、MLA、量化、NPU、
Graph或性能优化。新增router单独通过CPU测试，不能据此声称完整DeepSeek模型或
checkpoint可用。

`VERSION`仍为`0.0.14rc1`；目标`0.0.15rc1`只在所有后续生产、checkpoint、
CPU/NPU和发布门通过后的最终发布阶段更新。

## 13. Phase 6D1：Shared Expert与最小Block编排

### 13.1 范围与API

Phase 6D1只建立shared expert的权重/归约所有权，以及把Phase 6C router、既有
routed executor和shared MLP组合起来的最小block。本阶段新增：

```python
class SharedExpertPlacement(NamedTuple): ...
class SharedExpertMLP(nn.Module): ...
class DeepSeekMoeBlock(nn.Module): ...
```

`SharedExpertPlacement`是不可变纯Python元数据，不持有tensor或module，也不进入
state dict。字段为：

```text
parallel_mode, world_size, rank,
shared_intermediate_size, local_intermediate_size,
intermediate_start, intermediate_end, reduce_output
```

它独立于routed `ExpertPlacement`，shared expert不占用global routed expert ID。
非法mode、非正world size、越界rank、非正shared intermediate，以及TP不能整除
均在构造时fail-closed。

### 13.2 TP/EP ownership与归约

设完整shared intermediate宽度为`I_s`：

| 模式 | 本rank参数 | local区间 | local output后的动作 |
| --- | --- | --- | --- |
| world size 1 | 完整`I_s` | `[0,I_s)` | 不归约 |
| TP，world size `W` | `I_s/W`连续切片 | `[r*I_s/W,(r+1)*I_s/W)` | 恰好一次sum all-reduce |
| EP，world size `W` | 每rank完整复制`I_s` | `[0,I_s)` | 不归约 |

TP中每个rank只计算一个intermediate partial，两个rank的FP32 partial之和必须对齐
Phase 6B完整shared oracle，因此需要一次reduce。EP中每个rank都拥有同一个完整
shared expert；routed experts虽然按global expert slice产生不同partial并由既有
executor归约，shared output却已经完整，若再按EP rank求和会得到`W`倍放大。
所以EP shared禁止reduce，也禁止把各rank shared结果相加当oracle。

归约调用封装在窄范围module helper `_shared_expert_all_reduce`中，真实包加载路径
仍调用现有`tp_all_reduce`。`SharedExpertMLP.forward`只在冻结placement的
`reduce_output=True`时调用helper；测试可patch helper精确锁定调用次数，不需要
引入完整`lite_llama`包或修改`tp_utils`。真实collective不属于本阶段结论。

### 13.3 SharedExpertMLP数学与布局

本rank runtime参数布局为：

```text
gate_up_weight [H, 2*I_local]  # gate后up
down_weight    [I_local, H]
```

local数学为：

```text
gate_up = hidden @ gate_up_weight
gate, up = split(gate_up)
activated = silu(gate) * up
local_output = activated @ down_weight
```

CPU FP32结果直接与Phase 6B独立`deepseek_shared_expert_reference`比较。TP测试从
完整oracle权重独立切出gate、up和down的连续区间，再验证两个生产local partial
之和；没有用生产shared函数同时充当full oracle和partial oracle。NPU tensor未来
可复用既有SwiGLU策略，但D1没有运行或声称NPU、Graph或性能结果。

单独`SharedExpertMLP` state dict精确为：

```text
gate_up_weight
down_weight
```

placement与reduce ownership均不产生额外参数或buffer。

### 13.4 最小DeepSeekMoeBlock

block只执行以下编排：

```text
保存original shape并flatten
RoutingResult = gate(flat)
last_router_logits = RoutingResult.router_logits
routed_output = experts(flat, selected_experts, routing_weights)
shared_output = shared_experts(flat)
output = routed_output + shared_output
reshape回original shape
```

`RoutedExpertExecutor`继续独占routed TP/EP partial的reduce，`SharedExpertMLP`独占
shared TP partial的reduce；block源码不调用`tp_all_reduce`或shared reduce helper，
避免double reduce。2D、3D和`T=0`均保持shape语义，`last_router_logits`保留router
返回tensor的identity。

V2 block state dict精确为：

```text
gate.weight
experts.gate_up_weight
experts.down_weight
shared_experts.gate_up_weight
shared_experts.down_weight
```

V3额外包含：

```text
gate.e_score_correction_bias
```

未来checkpoint adapter负责把官方gate/up/down与shared参数转换到上述runtime布局；
当前key契约不能写成checkpoint loader已经支持DeepSeek。

### 13.5 Reference、mock collective与兼容门

Phase 6B reference仍是完整routed/shared FP32 oracle，并保持字节不变。D1生产测试
分别验证：single-rank V2/V3 block对完整oracle、TP shared partial sum、EP每rank
完整shared输出，以及mock reduce调用次数：

```text
TP world size 2: 1次
EP world size 2: 0次
world size 1:    0次
```

mock只验证调用ownership，不冒充HCCL/NPU collective。AST门锁定既有
`RoutingResult`、router、`ExpertPlacement`、`RoutedExpertExecutor`及Qwen类未改，
尤其既有executor `forward`中的all-reduce位置没有移动。新block无block级reduce，
shared forward无`.item()`/`.tolist()`等host sync。dependency-light direct-file
loader继续可构造并运行single-rank CPU block。

### 13.6 TDD与真实结果

RED命令：

```text
python -m unittest \
  tests.models.test_deepseek_moe_block.\
DeepSeekMoeBlockContractTest.\
test_shared_expert_placement_is_available -v
```

实现前真实结果：1 test error，`AttributeError: module ... has no attribute
'SharedExpertPlacement'`，exit code `1`。RED未stage、commit或push。最小API加入后
同一定向测试1/1通过，exit code `0`。

完整D1 focused命令：

```text
python -m unittest tests.models.test_deepseek_moe_block -v
```

真实结果：12 tests全部通过，exit code `0`。覆盖placement公式/错误、shared
FP32 oracle、TP/EP ownership、reduce call-count、V2/V3 block、2D/3D/T=0、
last logits identity、state dict、AST与direct loader。

Phase 6C router和Phase 6B reference联合回归：

```text
python -m unittest \
  tests.models.test_deepseek_moe_router \
  tests.models.test_deepseek_moe_reference -v
```

真实结果：36 tests全部通过，exit code `0`。Qwen回归：

```text
python -m unittest \
  tests.models.test_moe_reference \
  tests.models.test_qwen3_moe -v
```

真实结果：56 tests全部通过，exit code `0`。仓库文档/evidence门仍为8 tests，
release validator默认discovery仍为160 tests；最终复验均通过，exit code `0`，
独立`git diff --check`也为`0`。这些均为本地CPU结果。

### 13.7 明确不包含的能力与版本

Phase 6D1不包含checkpoint adapter、DeepSeek config/model registry、dense-vs-MoE
layer schedule、完整CausalLM、MLA、all-to-all/EPLB、非连续expert map、量化、
服务器/NPU、真实collective、Graph或性能优化。D1 block只是可独立验证的最小编排，
不能据此加载官方checkpoint或声称完整模型可用。

`VERSION`仍为`0.0.14rc1`；目标`0.0.15rc1`只在后续checkpoint、模型集成、
CPU/NPU与发布门全部完成后的最终发布阶段更新。

## 14. Phase 6D2：Checkpoint Canonicalizer与Shared布局Adapter

### 14.1 阶段边界

Phase 6D2只提供两个可以独立测试的纯转换边界：

1. `stack_deepseek_moe_weights`把显式指定的DeepSeek-V2/V3 HF MoE层转换为
   canonical checkpoint key/layout；
2. `prepare_shared_expert_gate_up/down`把canonical shared权重转换为Phase 6D1
   `SharedExpertMLP`的本rank runtime布局。

本阶段没有把它们接入`apply_weight_convert.py`、ModelExecutor、model config/registry、
完整checkpoint loader或完整模型。helper只依赖stdlib与PyTorch；不导入模型、
kernel、transformers或上游实现。

### 14.2 HF key到canonical checkpoint

对每个显式MoE层`L`，转换关系为：

| HF输入 | canonical输出 | canonical形状/所有权 |
| --- | --- | --- |
| `model.layers.L.mlp.gate.weight` | `layers.L.mlp.gate.weight` | `[E,H]`，沿用tensor identity |
| `...gate.e_score_correction_bias` | `layers.L.mlp.gate.e_score_correction_bias` | `[E]`，规范为FP32 |
| `experts.e.gate_proj.weight` + `up_proj.weight` | `experts.gate_up_weight` | numeric expert order的`[E,2I,H]`，gate rows后up rows |
| `experts.e.down_proj.weight` | `experts.down_weight` | numeric expert order的`[E,H,I]` |
| `shared_experts.gate_proj.weight` + `up_proj.weight` | `shared_experts.gate_up_weight` | `[2I_s,H]`，gate rows后up rows |
| `shared_experts.down_proj.weight` | `shared_experts.down_weight` | `[H,I_s]`，沿用tensor identity |

router和shared down无需改layout，直接沿用源tensor；bias只在必要时执行FP32 cast。
expert stack与gate/up cat必然产生新tensor。转换不会clone未发生layout变化的大权重，
也不读取、转换或消费未列入`moe_layer_indices`的dense层及其他checkpoint内容。

`moe_layer_indices`保持调用方给出的稳定顺序，并拒绝空集合、重复、负数和非整数；
expert按`0..E-1`显式numeric顺序收集。缺key、非tensor、非浮点、rank/shape、
hidden/intermediate、expert count、dtype或device不一致均fail-closed，错误包含layer、
expert和key上下文。普通权重要求全转换范围dtype/device一致；correction bias允许源
dtype不同，但device必须一致，canonical结果统一FP32。

V2调用`use_correction_bias=False`时不要求、不输出也不消费bias，即使源dict中存在
bias也保持原样。V3调用`True`时bias是必需key。该布尔API不接受`sqrtsoftplus`
等字符串或静默fallback，因此没有把DeepSeek-V4 static-hash路由伪装为V3。

### 14.3 consume原子性与大权重ownership

`consume=False`保持源dict的key顺序、tensor identity和内容不变。`consume=True`
采用“先完整验证并构造全部目标，再统一删除”的两阶段语义：

```text
validate every requested layer/key/layout
build complete converted mapping
only after success: delete exact consumed HF keys
```

因此后续层缺key、shape或dtype失败时，较早层也不会留下partial pop。成功时只删除
显式MoE层实际消费的router、routed expert、shared expert和可选bias keys；dense层、
未列层和其他权重保持不变。这个原子性是Python dict所有权契约，不声称解决真实
61GB checkpoint加载时的峰值内存；后续loader仍需在同一语义下设计分层释放。

### 14.4 Canonical shared到TP/EP runtime

shared canonical输入与D1 runtime输出：

| 权重 | canonical | runtime |
| --- | --- | --- |
| gate/up | `[2I_s,H]` | `[H,2I_local]` |
| down | `[H,I_s]` | `[I_local,H]` |

world size 1只执行transpose+contiguous：

```text
gate_up_runtime = gate_up_checkpoint.T
down_runtime = down_checkpoint.T
```

TP令`slice_r=[r*I_s/W,(r+1)*I_s/W)`，gate和up必须使用同一个连续切片：

```text
local_gate = canonical_gate[slice_r]
local_up   = canonical_up[slice_r]
runtime_gate_up = cat(local_gate, local_up).T
runtime_down = canonical_down[:, slice_r].T
```

测试把rank0/rank1 runtime转回canonical方向后，分别重组gate、up和down，必须逐元素
恢复完整canonical权重。EP不做intermediate切片，每个rank都只transpose完整shared
权重，各rank结果分别等于world size 1，不能把两rank相加或乘world size。

两个helper只接受floating CPU tensor，只做layout，不通信、不all-reduce，也不导入
`SharedExpertPlacement`/`SharedExpertMLP`。mode/world/rank/divisibility/shape均独立
fail-closed。既有routed `shard_moe_*`、`prepare_moe_*`、`tp_all_reduce`和错误契约
保持AST不变。

### 14.5 Canonical到D1 strict-load

single-rank集成测试把canonical routed权重交给既有：

```text
prepare_moe_gate_up_for_gmm: [E,2I,H] -> [E,H,2I]
prepare_moe_down_for_gmm:    [E,H,I]  -> [E,I,H]
```

并把canonical shared权重交给本阶段新增helper：

```text
[2I_s,H] -> [H,2I_s]
[H,I_s]  -> [I_s,H]
```

去除单层`layers.L.mlp.`前缀后，V2/V3 mapping均对Phase 6D1
`DeepSeekMoeBlock.load_state_dict(strict=True)`实现零missing/zero unexpected；V3
最终`gate.e_score_correction_bias`参数保持FP32。加载后的single-rank block输出继续
对齐Phase 6B独立`deepseek_moe_reference`。expected canonical与数值oracle在测试中
使用直接cat/stack/transpose公式构造，没有调用production converter自证。

### 14.6 TDD与真实结果

RED命令：

```text
python -m unittest \
  tests.models.test_deepseek_moe_weights.\
DeepSeekMoeWeightContractTest.\
test_shared_gate_up_layout_helper_is_available -v
```

实现前真实结果：1 test error，`AttributeError: module ... has no attribute
'prepare_shared_expert_gate_up'`，exit code `1`；RED未stage、commit或push。最小helper
加入后同一测试1/1通过，exit code `0`。

完整focused命令：

```text
python -m unittest tests.models.test_deepseek_moe_weights -v
```

扩展矩阵首轮14项通过、1项error：strict-load V3数值样本沿用了为检查布局顺序而
加入`layer_id*1000`的权重标记，随机hidden导致sigmoid全零，按Phase 6B契约正确
触发normalization fail-closed。该问题不属于converter差异；定向数值样本缩放到
稳定valid domain后，最终15 tests全部通过，exit code `0`。布局/顺序测试仍保留
原始大标记值。

DeepSeek 6B/6C/6D1联合回归：

```text
python -m unittest \
  tests.models.test_deepseek_moe_block \
  tests.models.test_deepseek_moe_router \
  tests.models.test_deepseek_moe_reference -v
```

真实结果：48 tests全部通过，exit code `0`。Qwen回归：

```text
python -m unittest \
  tests.models.test_moe_reference \
  tests.models.test_qwen3_moe -v
```

真实结果：56 tests全部通过，exit code `0`。最终仓库文档/evidence 8项、release
validator和`git diff --check`也作为独立门复跑；这些结果均为本地CPU，不包含
真实TP/EP collective、服务器、NPU或Graph。

### 14.7 许可证、限制与版本

本阶段依据已冻结公开数学、checkpoint key和接口契约独立实现，没有复制DeepSeek、
vLLM或vLLM-Ascend非平凡源码。仓库当前仍缺顶层LICENSE/NOTICE；正式对外复用前
必须单独解决许可证与attribution边界。

Phase 6D2不包含DeepSeek-V4、MLA、all-to-all/EPLB、非连续expert map、量化、
CLI、ModelExecutor、config/model registry、完整loader/full model、服务器/NPU、
Graph或性能优化。TP/EP测试仅验证单进程layout decomposition，不能冒充collective
正确性。

`VERSION`仍为`0.0.14rc1`；目标`0.0.15rc1`只在后续完整集成、真实NPU/并行与
发布门全部完成后的最终发布阶段更新。

## 15. Phase 6E1：配置驱动的MoE组件与CPU层状态装载

### 15.1 阶段目标与严格边界

Phase 6E1把已经独立验收的三段能力串成一个仍可单独测试的CPU组件：

```text
官方V2/V3 MoE字段子集
  -> DeepSeekMoeConfig（只描述MoE组件）
  -> build_deepseek_moe_block（只构造命中的MoE层）

HF MoE tensors
  -> Phase 6D2 canonical state
  -> prepare_deepseek_moe_layer_state（单层、单rank runtime state）
  -> Phase 6D1 DeepSeekMoeBlock.load_state_dict(strict=True)
```

该链路没有定义`DeepSeekModel`、decoder或CausalLM，也没有构造dense MLP、attention、
MLA、KV cache、RoPE、embedding、LM head或采样器。`DeepSeekMoeConfig`没有加入
`CONFIG_CLASS_MAP`，`ModelExecutor`也不会把`deepseek_v2/deepseek_v3`识别为可运行
完整模型。由此，组件级strict-load不能被表述为完整DeepSeek checkpoint已经可加载。

### 15.2 冻结官方字段与规范化映射

字段证据继续冻结在Phase 6A确定的官方来源，不跟随main漂移：

- DeepSeek-V2-Lite `604d5664dddd88a0433dbae533b7fe9472482de0`：
  <https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite/commit/604d5664dddd88a0433dbae533b7fe9472482de0>
- DeepSeek-V3 `9b4e9788e4a3a731f7567338ed15d3ec549ce03b`：
  <https://github.com/deepseek-ai/DeepSeek-V3/commit/9b4e9788e4a3a731f7567338ed15d3ec549ce03b>

组件配置映射如下：

| 官方字段 | 组件字段 | V2-Lite冻结值 | V3冻结值 |
| --- | --- | ---: | ---: |
| `hidden_size` | `hidden_size` | 2048 | 7168 |
| `num_hidden_layers` | `num_layers` | 27 | 61 |
| `intermediate_size` | `intermediate_size` | 10944 | 18432 |
| `n_routed_experts` | `num_experts` | 64 | 256 |
| `num_experts_per_tok` | 同名 | 6 | 8 |
| `moe_intermediate_size` | 同名 | 1408 | 2048 |
| `n_shared_experts` | `num_shared_experts` | 2 | 1 |
| `scoring_func` | `score_func` | `softmax` | `sigmoid` |
| `topk_method` | `topk_method` | `greedy` | `noaux_tc` |
| `n_group` | `num_groups` | 1 | 8 |
| `topk_group` | `topk_groups` | 1 | 4 |
| `norm_topk_prob` | 同名 | false | true |
| `routed_scaling_factor` | 同名 | 1.0 | 2.5 |
| `first_k_dense_replace` | 同名 | 1 | 3 |
| `moe_layer_freq` | 同名 | 1 | 1 |
| `torch_dtype` | 同名（仅记录） | `bfloat16` | `bfloat16` |

V2-Lite冻结HF配置使用`topk_method="greedy"`。本项目已审计的生产router把同一
one-group/group-limited数学明确命名为`group_limited_greedy`，所以`from_dict`只在
`model_type="deepseek_v2"`时执行这一个显式别名规范化。其他score/method组合不会
被静默改写：V2只接受`softmax + group_limited_greedy`，V3只接受
`sigmoid + noaux_tc`。V4、`sqrtsoftplus`、`static_hash`和hash-routing字段均
fail-closed。

派生值为：

```text
shared_intermediate_size = num_shared_experts * moe_intermediate_size
uses_correction_bias = (score_func == sigmoid and topk_method == noaux_tc)
```

未知attention等extra字段仍由现有`from_dict`过滤；测试逐一锁定上表关键字段，防止
它们因过滤而静默丢失。官方配置入口同时实行三层来源门：

1. `deepseek_v2`只接受精确列表`["DeepseekV2ForCausalLM"]`，`deepseek_v3`只接受
   精确列表`["DeepseekV3ForCausalLM"]`；缺失、交叉、混合、额外或未知architecture
   均fail-closed。该检查也位于dataclass公共校验门，因此直接构造不能绕过。
2. `from_dict`要求上表全部直接字段显式存在，并要求六组官方alias/canonical字段
   （层数、routed/shared expert数、score、group数、top-k group数）每组至少出现一个；
   不允许借V2 dataclass默认值补齐shape、schedule、scale或ownership。直接构造的默认
   V2配置仍然有效，因为这项“来源完整性”门只作用于外部mapping解析。
3. alias与canonical同时出现时，相同值可接受，异值明确拒绝；该规则只在
   `DeepSeekMoeConfig.from_dict`内生效，不改变`BaseConfig`或Qwen解析。

V2的`greedy`规范化只在来源字段完整、alias无冲突且model/architecture精确配对后执行。
`model_type`与`architectures`只保存来源事实，不触发完整模型注册。

### 15.3 0-based层调度与并行校验

层调度严格使用用户可见的0-based公式：

```text
0 <= layer_index < num_layers
layer_index >= first_k_dense_replace
layer_index % moe_layer_freq == 0
```

`is_moe_layer`对负数、越界和非整数返回false；`moe_layer_indices()`按升序稳定列出
所有命中层。V2-Lite因此只有layer 0是dense，V3只有layer 0/1/2是dense。frequency
大于1时仍按全局0-based层号取模，而不是相对`first_k_dense_replace`重新编号。
空MoE调度、非正层数/frequency、越界dense前缀均被拒绝。本阶段只拒绝dense层构造，
不替它创建dense MLP。

`validate_moe_parallel(world_size, mode)`只验证MoE ownership：

| 模式 | 必须整除 | shared语义 |
| --- | --- | --- |
| TP | routed `I % W == 0`且shared `I_s % W == 0` | shared intermediate切片 |
| EP | `E % W == 0` | shared完整复制，不要求`I_s % W == 0` |

mode、world size和rank在factory/adapter入口再次fail-closed。这里不验证attention heads、
KV heads或vocab，因为它们不属于MoE组件契约。

### 15.4 Component factory

新增dependency-light API：

```python
build_deepseek_moe_block(
    config,
    layer_index,
    tp_config=None,
    dtype=torch.float16,
)
```

factory只在`config.is_moe_layer(layer_index)`为true时构造Phase 6D1
`DeepSeekMoeBlock`，并从config完整传入H/E/K、routed/shared I、group策略、norm、
route scale、layer index和TP/EP配置。dense、不命中frequency及越界层均清晰拒绝。
V2/V3 state dict继续保持Phase 6D1的五key/六key契约；factory没有新增参数、buffer或
checkpoint key。

直接文件测试为`model_config.py`、`moe.py`和`tp_utils.py`安装三个明确别名，然后加载
component模块；没有`try/except ImportError`宽泛降级，也没有引入`transformers`、
`accelerate`或完整包初始化。

### 15.5 Canonical整层状态到rank-local block状态

第二个API为：

```python
prepare_deepseek_moe_layer_state(
    canonical_state,
    config,
    layer_index,
    tp_config=None,
)
```

它只读取`layers.L.mlp.`下的目标MoE层，返回去掉单层前缀、可直接strict-load的block
state。另一MoE层、dense层和无关key不会进入输出。输入mapping的key、tensor identity
和内容前后保持不变；函数不消费、不`torch.load`、不搬设备、不通信。

转换关系为：

| canonical | runtime local key | world1/TP | EP |
| --- | --- | --- | --- |
| `gate.weight [E,H]` | `gate.weight` | 沿用identity | 沿用identity |
| V3 bias `[E] FP32` | `gate.e_score_correction_bias` | 沿用identity | 沿用identity |
| routed gate/up `[E,2I,H]` | `experts.gate_up_weight` | I切片后`[E,H,2I_local]` | expert切片后`[E_local,H,2I]` |
| routed down `[E,H,I]` | `experts.down_weight` | I切片后`[E,I_local,H]` | expert切片后`[E_local,I,H]` |
| shared gate/up `[2I_s,H]` | `shared_experts.gate_up_weight` | I_s切片后`[H,2I_s_local]` | 完整`[H,2I_s]` |
| shared down `[H,I_s]` | `shared_experts.down_weight` | I_s切片后`[I_s_local,H]` | 完整`[I_s,H]` |

world size 1也必须执行canonical到runtime的transpose。V3 bias必须存在、shape为`[E]`、
保持FP32；V2即使canonical mapping含额外bias也不输出、不消费。所有普通权重必须是
同dtype/device的floating CPU tensor，并与config H/E/I严格一致；bias只允许FP32且
device一致。

### 15.6 Strict-load、TP/EP decomposition与独立oracle

测试先使用Phase 6D2 converter产生多层canonical state，再调用本阶段adapter。world1
的V2/V3 runtime mapping均对factory block实现`load_state_dict(strict=True)`零missing、
零unexpected，随后完整block输出对齐Phase 6B `deepseek_moe_reference`。

TP2为rank0/1分别构造block并strict-load：routed和shared intermediate各持有连续一半；
测试使用独立reference产生的selected IDs/weights，分别调用rank-local routed/shared数学，
两个rank的四个partial相加后对齐完整oracle。EP2为rank0/1分别持有连续且唯一的两名
routed experts，shared在每rank完整复制；正确组合是：

```text
routed_rank0 + routed_rank1 + one_copy_of_shared
```

测试显式证明把两份shared相加不会等于完整oracle。以上只验证单进程layout与ownership，
没有执行或模拟HCCL collective，也不构成NPU正确性证据。

### 15.7 TDD与真实结果

RED命令：

```text
python -m unittest \
  tests.models.test_deepseek_moe_component.\
DeepSeekMoeComponentContractTest.\
test_component_config_api_exists -v
```

实现前真实结果：1 test error，`AttributeError: module ... has no attribute
'DeepSeekMoeConfig'`，exit code `1`；RED未stage、commit或push。

完整focused命令：

```text
python -m unittest tests.models.test_deepseek_moe_component -v
```

首轮14项功能测试通过；冻结AST审计因Windows `subprocess`默认GBK解码UTF-8 Git blob而
产生9个subtest error。测试改为对Git输出显式UTF-8解码，并把正则改为raw string；
没有修改生产数学。最终15 tests全部通过，exit code `0`。

Phase 6E1-R1补充了三类真实fail-closed RED：

- architecture来源：V2/V3交叉、缺失、混合与未知architecture此前会被接受（空列表还
  触发了非契约`TypeError`）；定向命令运行1 test，出现8 failures和1 error，exit code
  `1`；
- required source fields：逐项删除shape、schedule、scale直接字段及六组alias后，配置
  会静默继承V2默认值或只在后续策略校验偶然失败；
- alias冲突：六组alias/canonical异值双写均被静默覆盖。

后两类合并定向命令运行2 tests，共22 failures，exit code `1`。加入统一来源门后，三类
定向命令运行3 tests全部通过，exit code `0`；完整component suite增至18 tests并继续
作为最终GREEN门。R1全过程未stage、commit或push。

冻结DeepSeek回归：

```text
python -m unittest \
  tests.models.test_deepseek_moe_weights \
  tests.models.test_deepseek_moe_block \
  tests.models.test_deepseek_moe_router \
  tests.models.test_deepseek_moe_reference -v
```

真实结果：63 tests全部通过，exit code `0`。Qwen回归命令
`python -m unittest tests.models.test_moe_reference tests.models.test_qwen3_moe -v`
真实结果为56 tests全部通过，exit code `0`。仓库文档/evidence 8项、release validator、
`git diff --check`和自定义config/schedule/strict-load/frozen-AST probe也作为最终独立门
复跑；结果只代表本地CPU。

### 15.8 冻结兼容、许可证与后续阻塞

测试把当前`model_config.py`与HEAD中的既有Base/Llama/Qwen/Llava配置类逐类AST比较，
并对`moe.py`、`tp_utils.py`、Phase 6B/C/D reference、router、block、weights文件执行
`git diff --exit-code`；这些冻结边界均保持不变。新增配置也明确不出现在
`executor_struct.py`的`CONFIG_CLASS_MAP`。

本阶段依据冻结公开字段、数学和既有项目接口独立实现，没有复制DeepSeek、vLLM或
vLLM-Ascend非平凡源码。仓库仍缺顶层LICENSE/NOTICE，正式对外复用前必须单独解决
许可证和attribution边界。

仍不支持：DeepSeek-V4、完整checkpoint/file loader、`apply_weight_convert` CLI、
`ModelExecutor`、config/model registry、dense layer实现、完整decoder/CausalLM、MLA、
attention/KV/RoPE、all-to-all/EPLB、非连续expert map、量化、服务器/NPU、Graph及性能
优化。`VERSION`仍为`0.0.14rc1`；目标`0.0.15rc1`只在后续完整模型边界、真实并行/NPU
与发布门全部完成后的最终发布阶段更新。
