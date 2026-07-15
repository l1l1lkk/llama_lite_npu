# Qwen3 MoE Runtime 正确性基线与最小演进设计

## 1. 背景、目标与面试价值

当前实现已经具备 Qwen3 MoE 的 FP16 推理、TP/实验性 EP、Ascend GMM、
routed GEMV 和 decode Graph 路径，但模型结构、路由、专家布局、后端选择与
collective 仍集中在少数 Qwen3 专用类中。这样的实现可以运行，却难以单独回答
以下推理引擎问题：

- router 与 expert executor 的数值契约是什么；
- TP 和 EP 分别切分什么，global/local expert id 如何转换；
- 优化算子与独立数学定义是否一致；
- Graph、fallback、量化和后续模型适配应在哪一层扩展。

本阶段的目标是先建立一个独立、CPU 可运行、容易审查的 PyTorch 数学参考，
再用 characterization tests 锁定当前 Qwen3 行为。它能形成 AI Infra/LLM 推理
引擎面试中可解释的证据链：从权重布局和路由公式出发，说明单卡 reference、
TP/EP 分解、优化后端以及 Graph 风险，而不是只展示“模型能跑”。

本阶段不修改生产代码，不引入通用 runtime 抽象，也不声称 NPU 正确性或性能。

## 2. 基线与范围

- GitLab 事实源起点：`release/0.0.12rc1`，commit
  `5595fefe1d6e0c88b6dba00fa6c9c1aee38f4708`。
- 隔离开发分支：`feature/moe-runtime-0.0.14rc1`。
- 当前 `VERSION` 仍为 `0.0.13rc3`。
- 通用 MoE runtime 属于功能级更新，目标版本建议为 `0.0.14rc1`，只在最终发布
  阶段同步 VERSION、CHANGELOG、README、release 文档与 tag。
- Phase 1 只新增测试 reference、CPU 测试和本文档。

## 3. 当前 Qwen3 MoE 调用链与耦合

### 3.1 配置与权重

1. `executor_struct.py` 根据 `model_type=qwen3_moe` 选择 `Qwen3MoeConfig`；
   `ModelExecutor` 读取 checkpoint 配置并注入 `moe_parallel_mode=tp|ep`。
2. `apply_weight_convert.py` 和 `utils/qwen3_moe_weights.py` 把 HF 的逐专家
   gate/up/down 权重堆叠，并保留 router 的 `gate.weight`。
3. checkpoint gate/up 为 `[E, 2I, H]`、down 为 `[E, H, I]`。加载器根据 TP/EP
   切分后转成 GMM-native runtime 布局 `[E_local, H, 2I_local]` 和
   `[E_local, I_local, H]`。
4. router 权重 `[E, H]` 在 TP/EP rank 间复制。

### 3.2 模型与执行

`Qwen3MoeDecoderLayer` 在 sparse layer 中用 `Qwen3SparseMoeBlock` 替换 dense
MLP。`Qwen3SparseMoeBlock` 同时负责：

- 把任意前导维度展平为 `[T, H]` 并在结束时恢复；
- 持有 `mlp.gate.weight`，计算 logits、top-k weights/ids；
- 保存 `last_router_logits`；
- 调用 `Qwen3MoeExperts`。

`Qwen3MoeExperts` 又同时持有专家权重、TP/EP placement、backend 环境变量、
eager/GMM/GEMV dispatch、local combine 和最终 `tp_all_reduce`。因此 router
weight、expert execution、parallel placement 与 backend policy 尚未形成清晰边界。

### 3.3 当前并行与 Graph 语义

- TP：每 rank 持有所有专家的 intermediate shard；每个 rank 计算 hidden partial，
  最后 all-reduce。
- EP：每 rank 持有连续的完整专家区间；所有 rank 仍看到全部 token 和同一组路由，
  筛选本地 global id 后转换为 local id，再 all-reduce 各 rank 的局部贡献。当前
  EP 不是 token all-to-all runtime。
- decode Graph：Qwen3 MoE TP 可尝试捕获；EP 因动态 `nonzero` 路径禁用；捕获失败
  的 key 记忆后回 eager。

checkpoint key、runtime layout、FP32 softmax/累加、backend 环境变量、all-reduce
位置及 Graph eligibility 都是后续重构的兼容性风险。

## 4. 独立 CPU Reference

实现位于 `tests/reference/moe_reference.py`。该文件只导入 Python typing 与
PyTorch，不导入 `lite_llama`、Qwen3 router/expert、`moe_routing`、routed GEMV、
GMM helper 或生产 eager 路径。测试还通过 AST/import 检查锁定这一独立性。

### 4.1 API

- `route_topk_reference(hidden_states, router_weight, *, top_k, norm_topk_prob)`
  返回 `(router_logits, routing_weights, selected_experts)`。
- `expert_forward_reference(hidden_states, selected_experts, routing_weights,
  gate_up_weight, down_weight, *, expert_start=0, local_num_experts=None)`
  返回一个 rank 的 FP32 局部贡献。
- `moe_forward_reference(...)` 组合上述两步，返回
  `(output_fp32, router_logits, routing_weights, selected_experts)`。

全部 API 为 `torch.no_grad()` inference-only，只接受 CPU flat token 输入 `[T,H]`。
`T=0` 返回 `[0,H]`；合法 `K` 为 `1..E`。

### 4.2 路由数学定义

给定 hidden states `X ∈ R^(T×H)` 和 router 权重 `W_r ∈ R^(E×H)`：

```text
L = X W_r^T
P = softmax_fp32(L)
(A, J) = topk(P, K)
A = A / sum(A)    if norm_topk_prob=True
```

当前生产契约中，`L` 保留 linear 的输入 dtype，softmax 输入显式提升为 FP32，
选中权重 `A` 最后 cast 回 `L.dtype`，expert ids `J` 是 global id。

### 4.3 专家数学定义与布局

reference 直接接受 runtime GMM-native 布局：

```text
router_weight  [E,       H]
gate_up_weight [E_local, H, 2I]
down_weight    [E_local, I, H]
```

对 token `t`、top-k slot `s`、专家 `e=J[t,s]`：

```text
[g, u] = X[t] @ gate_up_weight[e]
z      = silu(g) * u
y_e    = z @ down_weight[e]
Y[t]  += float32(A[t,s]) * float32(y_e)
```

专家 matmul 与 token combine 使用 FP32。`expert_start/local_num_experts` 定义一个
连续 EP 本地专家区间；区间外 expert id 被忽略，从而得到单 rank local
contribution。reference 不实现 collective、dispatch 通信或 all-to-all。

TP 的单进程验证把 gate 和 up 的 intermediate 维按相同区间切分，并切分 down 的
输入 intermediate 维；各 rank partial sum 应等于完整专家输出。EP 验证把专家维
切成连续区间；各 rank local contribution 之和应等于完整输出。

## 5. 当前稳定契约

Phase 1 测试锁定以下 CPU 行为：

- `T=0`、`T=1`、多 token；`E=1/2/4/8`；`K=1`、多 K、`K=E`；
- `norm_topk_prob=True/False`，启用时 selected weights 和为 1，关闭时保留未选
  expert 的概率质量；
- router softmax 使用 FP32，selected weights 回 cast 到 logits dtype；
- Qwen3 CPU eager block 与独立 FP32 oracle 对齐；
- 无 token expert、所有 token 命中热点 expert、较均匀命中所有 expert；
- contiguous 与 non-contiguous `[T,H]` 输入得到相同输出；
- 有限极端 logits（测试使用 `±1e4`）保持有限并与 reference 对齐；
- 非 tie、固定输入的 ids/weights/logits/output 可重复；
- TP intermediate partial sum 和 EP local contribution sum 等于 full oracle；
- inference-only，不测试 backward。

FP32 production/reference 比较使用 `rtol=1e-5, atol=1e-6`；纯相同算子路径的
characterization 可使用 PyTorch 默认严格 close。该数值只代表本阶段 CPU FP32。

## 6. 不稳定或 Unsupported 边界

- NaN/Inf 输入不属于当前稳定跨设备 ABI；本阶段不为生产路径增加异常，也不绑定
  某个 top-k 非有限输出。
- tie 的 expert id 顺序未定义。tie 测试只检查 id 合法且不重复、权重性质；用于
  输出比较的并列专家具有相同权重，避免依赖 tie-break。
- FP16/BF16/NPU tolerance 尚未定案。可把 FP16 `1e-2`、BF16 `2e-2` 作为 Phase 4
  初始实验量级，但必须用真实 NPU 的 max/mean error 校准，不能作为当前结论。
- 不验证 backward、真实 HCCL collective、NPU GMM/GEMV、全模型 Graph、性能、
  W8A8 或其他量化。
- 不支持 shared experts、grouped top-k、sigmoid、route scale、score correction、
  DeepSeekMoE 或 MLA；这些不是 Phase 1 的隐藏实现。

## 7. 与冻结参考实现的对比

### 7.1 vLLM

冻结 commit：
[`66b6c684ab5a32dabcd2e69daaa60ff5d5198392`](https://github.com/vllm-project/vllm/commit/66b6c684ab5a32dabcd2e69daaa60ff5d5198392)。

vLLM 把 router、routed experts/quant method、expert mapping、parallel config 和
runner 分离；测试使用独立 iterative PyTorch MoE 对照 fused top-k/experts。可借鉴
的是接口职责、logical/physical expert mapping、custom routing/quant contract 和
reference test 组织方式，不是 CUDA custom op、CUDA Triton、CUDA Graph、NCCL、
DeepEP 或 GPU 量化布局。

### 7.2 vLLM-Ascend

冻结 commit：
[`63a194ef73d821d1b4b024d341383537a5030998`](https://github.com/vllm-project/vllm-ascend/commit/63a194ef73d821d1b4b024d341383537a5030998)。

vLLM-Ascend 在通用接口后分离 expert selector、PrepareAndFinalize、token dispatcher
和 MoE MLP/quant method，并根据能力在 fused selector 与 native path 间选择。可
借鉴 NPU 适配层边界及显式 int32 expert id/quant contract；不能直接搬用特定 CANN
版本、`torch_npu` 签名、NZ 权重布局、MC2/HCCL group、A2/A3/A5 分支或量化 scale
格式。

### 7.3 Apache-2.0 Attribution

vLLM 与 vLLM-Ascend 的冻结版本均为 Apache-2.0。本阶段依据公开数学定义和接口
思想独立编写测试 reference，没有复制两仓库的非平凡源码。若后续需要复用具体
实现，必须先处理本仓库当前缺失的 tracked LICENSE/NOTICE，并保留来源、版权、
许可证和修改说明；默认仍应独立实现。

## 8. Phase 2 最小边界建议

Phase 2 才考虑以下最小生产边界：

1. `RoutingResult`：只承载 `[T,K]` ids/weights 和可选 logits，不包含 GMM 排序计划。
2. `Router`：保留 `mlp.gate.weight` checkpoint key 和当前 Qwen3 默认 softmax/top-k
   行为。
3. `ExpertPlacement`：只表达 TP/EP、本地 global id 区间和 final reduce 责任。
4. `ExpertExecutor`：接收 flat hidden 与 routing，保持 eager/GMM/GEMV policy 和
   output combine；暂不强拆 backend-specific Combine。
5. `MoELayer`：负责 flatten/restore、router、executor 和 exactly-once reduce。

Phase 2 必须先用本阶段 oracle 保护默认 Qwen3 行为。高风险点包括 state dict key/
layout、`last_router_logits`、FP32 softmax/累加、TP/EP local id、backend fallback、
Graph capture storage/eligibility 及 server 默认配置。shared experts、DeepSeek routing、
量化和通信框架不应提前塞入这次最小重构。

## 9. 测试矩阵与真实结果

| 项目 | 覆盖 | Phase 1 结果 |
|---|---|---|
| tokens | 0、1、5/7 | 通过 |
| experts | 1、2、4、8 | 通过 |
| top-k | 1、2、K=E | 通过 |
| renormalize | True/False | 通过 |
| dtype | FP32 oracle；BF16 router cast characterization | 通过 |
| routing load | 无 token expert、热点、每 expert 命中 | 通过 |
| input layout | contiguous/non-contiguous | 通过 |
| logits | 有限 `±1e4` | 通过 |
| determinism | 非 tie 固定输入重复 | 通过 |
| tie | 不绑定 id 顺序；相同 expert 权重下输出 | 通过 |
| TP | 两个 intermediate shard partial sum | 通过（单进程数学模拟） |
| EP | 两个连续 expert slice local sum | 通过（单进程数学模拟） |
| gradients | inference-only/no_grad | 通过；未运行 backward |
| NPU/GMM/GEMV/Graph | 仅保留既有 CPU policy tests | 未做真实 NPU 验证 |

### 9.1 RED

```text
python -m unittest tests.models.test_moe_reference -v
```

首次执行 exit code `1`，真实失败为
`ModuleNotFoundError: No module named 'tests.reference'`。这证明新 reference/契约在
实现前不存在；RED 状态未 commit/push。

### 9.2 GREEN 与回归

- 修改前 baseline：
  `python -m unittest tests.models.test_qwen3_moe -v`，26 tests，exit `0`。
- 新增 reference：
  `python -m unittest tests.models.test_moe_reference -v`，16 tests，exit `0`。
- MoE 权重转换/TP layout：7 tests，exit `0`。
- NPU Graph policy、graph ablation、observability CPU suites：28 tests，exit `0`。
- 一次组合相关测试中，除旧 `tests/others/test_load_weight.py` 外的 38 tests 通过；
  该旧文件在收集阶段因环境缺少 `accelerate` 而 exit `1`，发生在导入未修改的
  `lite_llama` 包时，与 Phase 1 改动无关。
- `python -m pytest tests/kernels/test_cuda_graph.py -q` exit `1`：旧 CUDA 示例的
  `test_model_gen(input_text)` 缺少 pytest fixture。它不是当前 NPU Graph 契约套件；
  实际相关 CPU policy suite `tests.test_decode_p0` 已包含在上述 28 tests 并通过。

- focused GREEN：`python -m unittest tests.models.test_moe_reference
  tests.models.test_qwen3_moe tests.test_decode_p0 tests.test_graph_ablation
  tests.test_observability tests.test_model_executor_packed_prefill
  tests.test_tp_control -v`，82 tests，exit `0`。
- `python scripts/validate_release.py`：markdown UTF-8 scan 通过，版本文档仍为
  `0.0.13rc3`，unittest discovery 157 tests 通过，compile 检查和
  `git diff --check` 通过，exit `0`。

## 10. 限制与后续验证门

本阶段的结果只能证明 CPU 数学 reference 与当前 CPU production eager 行为，以及
单进程 TP/EP 分解语义。Phase 4 才能在记录 CANN、torch_npu、设备、shape、dtype
的真实环境中校准 FP16/BF16 tolerance，并验证 GMM、routed GEMV、HCCL、full-model
Graph capture/replay/fallback。任何 NPU 结果都不能由本阶段 CPU 测试推断。

## 11. Phase 2：RoutingResult 与通用 TopK Router 边界

### 11.1 最终模块边界

Phase 2 最终选择在现有 `lite_llama/models/moe.py` 内建立最小路由边界，没有新增
`moe_runtime.py`：

- `RoutingResult` 是不可变、tuple-compatible 的三字段返回值；
- `SoftmaxTopKRouter` 持有当前通用 softmax top-k 数学实现；
- `Qwen3MoeTopKRouter` 是不增加参数或行为的兼容子类；
- `Qwen3SparseMoeBlock` 仍持有 `gate` 和 `experts`，调用与 combine 路径不变。

选择同文件不是放弃模块化，而是遵循当前 dependency-light 测试约束。现有 CPU
测试用 `spec_from_file_location` 直接加载 `moe.py`，此时 `__package__` 为空。在
已有文件内定义边界，不需要相对导入、动态路径加载或 broad `ImportError`
fallback，也不会为了测试导入整个 `lite_llama` 包及其 `accelerate` 依赖。等未来
仓库建立稳定的轻量模块加载约定后，可以在不改变 API 的前提下移动定义。

### 11.2 RoutingResult 语义与 tuple 兼容

字段顺序保持原三元返回 ABI：

```text
router_logits, routing_weights, selected_experts
```

调用方既可以继续：

```python
logits, weights, expert_ids = router(hidden_states)
```

也可以使用属性访问：

```python
result = router(hidden_states)
result.router_logits
result.routing_weights
result.selected_experts
```

测试锁定字段和 tuple 行为，不依赖具体 `repr`。`RoutingResult` 不承载排序后的
token、expert counts、group list 或执行计划，避免把 GMM/GEMV 细节泄漏到 router
边界。

### 11.3 Router 行为保持

`SoftmaxTopKRouter` 原样保留：

```text
F.linear
-> softmax(router_logits.float())
-> topk
-> optional selected-weight renormalization
-> cast weights back to router_logits.dtype
```

constructor、默认 `norm_topk_prob=True`、weight `[E,H]`、expert id dtype、
`T=0` shape 和 Qwen3 symbol 均保持。Phase 1 的真实数学矩阵继续对
`Qwen3MoeTopKRouter` 运行，而不是用 mock 替代路由计算。

本阶段没有加入 sigmoid、grouped top-k、route scale、correction bias、custom
routing 或 DeepSeek 参数占位；这些行为需要各自的 reference 契约和阶段验收。

### 11.4 兼容证据

`Qwen3SparseMoeBlock.state_dict()` 和 `named_parameters()` 仍精确只有：

```text
gate.weight
experts.gate_up_weight
experts.down_weight
```

因此没有新增可训练参数或 checkpoint key。测试在一次真实 block 调用中捕获 router
返回值，确认 `last_router_logits` 与该 `RoutingResult.router_logits` 是同一 tensor，
并再次比较 block 输出与 Phase 1 独立 FP32 oracle。

dependency-light direct-file loader 继续直接加载 `moe.py`，无需额外模块预加载。
source contract 对 `SoftmaxTopKRouter.forward` 做 AST 检查：没有 `.item()`、
`.tolist()`、`.cpu()`、`.numpy()` 或日志；唯一 Python 分支是静态配置
`self.norm_topk_prob`。没有增加 shape assertion、device copy 或 tensor-value-driven
控制流。

### 11.5 明确不进入的范围

本阶段没有修改 `Qwen3MoeExperts` 的执行逻辑、backend policy、GMM/GEMV、routing
kernel、TP/EP placement、all-reduce、Graph 实现或 server flags。ExpertExecutor 与
placement 属于下一独立阶段；shared experts、DeepSeek routing 和量化不在本阶段。

CPU Graph policy tests 只能证明 eligibility/fallback 源码路径没有回归。没有运行
NPU Graph capture/replay，因此不能声称 NamedTuple 在真实 Ascend Graph 中已经完成
验证；热路径无 host sync 的源码契约只是进入后续 NPU 验证的必要条件。

### 11.6 Phase 2 TDD 与真实结果

RED：

```text
python -m unittest \
  tests.models.test_moe_reference.MoeReferenceContractTest.\
test_generic_router_boundary_is_tuple_compatible -v
```

实现前 exit code `1`，真实失败为 production module 不存在
`SoftmaxTopKRouter` 的 `AttributeError`。RED 状态未 stage、commit 或 push。

最小 GREEN：同一测试在实现后 1 test 通过，exit `0`。

Phase 2 reference/compatibility：

```text
python -m unittest tests.models.test_moe_reference -v
```

22 tests 通过，exit `0`。新增覆盖 tuple/属性访问、Qwen3 兼容符号、空输入
shape/dtype、state dict/parameter keys、`last_router_logits`、direct-file loader 和
无 host-visible tensor conversion。

MoE、Graph policy、weight conversion、TP/EP 相关 CPU 回归：

```text
python -m unittest \
  tests.models.test_moe_reference \
  tests.models.test_qwen3_moe \
  tests.test_decode_p0 \
  tests.test_graph_ablation \
  tests.test_observability \
  tests.test_model_executor_packed_prefill \
  tests.test_tp_control -v
```

88 tests 通过，exit `0`。

最终 focused gate：

```text
python -m unittest \
  tests.models.test_moe_reference \
  tests.models.test_qwen3_moe -v
```

48 tests 通过，exit `0`。其中 22 项是独立 reference/Phase 2 契约，26 项是已有
Qwen3 MoE 回归；不能只依赖默认 discovery 来代替这条显式命令。

`python scripts/validate_release.py` 的 markdown UTF-8、版本文档、compile、
`git diff --check` 全部通过，默认 discovery 157 tests 通过，exit `0`。独立执行
`git diff --check` 也为 exit `0`。

Phase 2 不更新版本；`VERSION` 仍为 `0.0.13rc3`，目标 `0.0.14rc1` 仍只在最终
发布阶段更新。

## 12. Phase 3：ExpertPlacement 与通用 Routed-Expert Executor

### 12.1 ExpertPlacement 字段与推导

`ExpertPlacement` 是位于 `lite_llama/models/moe.py` 的不可变 NamedTuple，只保存
纯 Python 字符串和整数：

```text
parallel_mode
world_size
rank
local_num_experts
expert_start
expert_end
local_intermediate_size
```

它不保存 tensor、`nn.Module`、通信 group、expert map 或执行 plan，因此不会进入
state dict，也不会给 Graph 热路径增加 tensor 操作。推导入口为：

```python
ExpertPlacement.from_config(
    num_experts=...,
    intermediate_size=...,
    tp_config=...,
)
```

记全局专家数为 `E`、intermediate size 为 `I`、并行 world size 为 `W`、rank 为
`R`。

TP 保持所有 expert、本 rank 只持有 intermediate shard：

```text
local_num_experts       = E
expert_start            = 0
expert_end              = E
local_intermediate_size = I / W
```

约束仍为 `I % W == 0`。TP 的 rank 被记录为元数据，但不改变 expert 范围。

EP 保持连续完整 expert 区间、本 rank 持有完整 intermediate：

```text
local_num_experts       = E / W
expert_start            = R * local_num_experts
expert_end              = expert_start + local_num_experts
local_intermediate_size = I
```

约束仍为 `E % W == 0`。模式仍只允许 `tp`/`ep`。invalid mode、TP divisibility、
EP divisibility 的异常类型和完整消息均由测试锁定，与 Phase 3 前一致。本阶段没有
新增 rank range、world size 或其他验证，以免无意改变既有行为。

### 12.2 通用 Executor 与 Qwen3 兼容层

原 `Qwen3MoeExperts` 的实现主体提升为 `RoutedExpertExecutor`。constructor 签名、
weight parameter、public 属性、backend 配置和所有执行方法保留。constructor 仅把
原地 placement 计算替换为 `ExpertPlacement.from_config`，再从元数据回填原属性：

```text
parallel_mode
local_num_experts
expert_start
expert_end
local_intermediate_size
```

同时继续保留 `tp_config`、`layer_index`、`backend`、GEMV threshold、validation
开关和 tolerance 等已有属性。新增 `placement` 只是不可变纯 Python 元数据，不是
parameter/buffer。

`Qwen3MoeExperts` 现在是 `RoutedExpertExecutor` 的无覆盖兼容子类：不定义自己的
`__init__` 或 `forward`，不增加参数和状态。`inspect.signature` 证实两者 constructor
一致；`Qwen3SparseMoeBlock` 仍显式构造 `Qwen3MoeExperts`，外部符号和模型结构不变。

### 12.3 保持不变的执行契约

生产 diff 没有修改以下方法体：

- `_forward_eager_local`
- `_forward_grouped_local`
- `_forward_routed_local`
- `_should_use_routed_gemv`
- `_validate_local_outputs`
- `_use_grouped_backend`
- `forward`

因此 expert 数学、EP global/local id 过滤、GMM/GEMV 调用、backend env、threshold、
fallback/warning、validation tolerance 和最终 reduce 算法均保持。source contract 还
确认 `forward(hidden_states, selected_experts, routing_weights)` 没有增加
`RoutingResult`/`isinstance` 动态适配、`.item()`/`.tolist()`/`.cpu()` 或日志；
`tp_all_reduce` 调用仍只有一次，条件仍为
`getattr(self.tp_config, "enabled", False)`。

generic executor 与 Qwen3 compatibility executor 在相同真实权重、hidden、routing
下运行 CPU eager，二者均与 Phase 1 独立 FP32 oracle 对齐。没有用 mock 替代 expert
数值路径。

executor 自身 state dict 对两种类型都仍精确为：

```text
gate_up_weight
down_weight
```

完整 `Qwen3SparseMoeBlock` 的 state dict/named parameters 继续精确为：

```text
gate.weight
experts.gate_up_weight
experts.down_weight
```

### 12.4 明确不实现的能力

`ExpertPlacement` 只描述当前连续 TP/EP placement，不实现 all-to-all、非连续 expert
map、冗余 expert、负载均衡或 EPLB。Phase 3 也不增加 shared expert、DeepSeekMoE、
quant method、W8A8 参数或未来能力占位大全。

RoutedExpertExecutor 只是清晰命名和兼容边界，不是新执行算法。本阶段不修改 kernel、
weight converter、executor、Graph、server 或 collective。shared expert combine、
DeepSeek routing、量化和更通用通信必须在各自阶段建立独立 reference 与验证门。

### 12.5 Direct Loader、Graph 与 NPU 边界

placement 和 executor 继续定义在现有 `moe.py`，dependency-light direct-file loader
无需新增 import/fallback，也不会导入整个 `lite_llama`/`accelerate` 依赖。

CPU Graph policy tests 只能证明 Qwen3 MoE TP/EP eligibility、capture/fallback policy
没有源码回归。没有运行真实 NPU Graph、GMM/GEMV 或 HCCL；新增纯 Python placement
不进入 forward 热路径，但这不能替代后续 Ascend capture/replay 实测。本阶段不声称
性能变化。

### 12.6 Phase 3 TDD 与真实结果

RED：

```text
python -m unittest \
  tests.models.test_moe_reference.MoeReferenceContractTest.\
test_generic_executor_boundary_exposes_placement -v
```

实现前 1 test 失败，exit `1`；真实原因是 production module 不存在
`ExpertPlacement` 的 `AttributeError`。RED 状态未 stage、commit 或 push。

最小 GREEN：同一测试在实现后通过，exit `0`。

Phase 3 reference/compatibility：

```text
python -m unittest tests.models.test_moe_reference -v
```

30 tests 通过，exit `0`。新增覆盖 TP world size 1/2、不同 rank；EP rank 0/1 连续
范围；完整异常消息；constructor signature/MRO；generic/Qwen eager/oracle；executor
state dict；public backend/placement 属性；forward source/reduce 契约。

MoE、Graph policy、weight conversion、TP/EP 和 backend 相关 CPU 回归：

```text
python -m unittest \
  tests.models.test_moe_reference \
  tests.models.test_qwen3_moe \
  tests.test_decode_p0 \
  tests.test_graph_ablation \
  tests.test_observability \
  tests.test_model_executor_packed_prefill \
  tests.test_tp_control -v
```

96 tests 通过，exit `0`。

最终 focused gate：

```text
python -m unittest \
  tests.models.test_moe_reference \
  tests.models.test_qwen3_moe -v
```

56 tests 通过，exit `0`，其中 30 项是独立 reference/Phase 2/Phase 3 契约，26 项
是已有 Qwen3 MoE 回归。

`python scripts/validate_release.py` 的 markdown UTF-8、版本文档、compile、
`git diff --check` 全部通过，默认 discovery 157 tests 通过，exit `0`。独立执行
`git diff --check` 也为 exit `0`。

Phase 3 不更新版本；`VERSION` 仍为 `0.0.13rc3`，目标 `0.0.14rc1` 仍只在最终
发布阶段更新。
