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
