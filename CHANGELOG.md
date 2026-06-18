# Changelog

所有触发版本升级的变更按发布时间倒序记录。详细规则见[版本管理与发布规范](docs/versioning.md)。

## [0.0.7rc6] - 2026-06-18

Bugfix release for the v0.0.7 KV-cache line.

### Fixed

- Changed page-aligned partial Prefix Cache reuse from default-on to explicit opt-in via `--partial_prefix_cache`.
- Preserved exact Prefix Cache as the default greedy repeated-prompt optimization.
- Avoided EvalScope random-prompt TTFT regression caused by conservative token-by-token suffix replay on shared chat-template prefixes.

### Docs

- [v0.0.7rc6 release report](docs/releases/v0.0.7rc6.md)

## [0.0.7rc5] - 2026-06-18

KV-cache optimization closeout for the v0.0.7 line.

### Core changes

- Added live page-aligned partial prefix reuse for greedy requests. Exact prompt hits still skip prefill entirely; prefix-extension prompts now share cached pages and replay only the uncached suffix.
- Changed chunked prefill from admission-only scheduling into a real multi-tick execution path using safe incremental prompt replay.
- Added TP continuous-batching `prefill_chunk` control messages so worker ranks mirror chunked prefill state correctly.
- Kept mixed-length packed prefill as a tested metadata contract; no-padding packed prefill kernels remain future work.

### Compatibility and limitations

- Prefix reuse remains disabled for sampling requests (`temperature>0`).
- Partial suffix replay is correctness-first and token-by-token; expected benefit is TTFT reduction on repeated prefixes, not maximum raw prefill throughput.
- No Atlas performance number is recorded yet.

### Docs

- [v0.0.7rc5 release report](docs/releases/v0.0.7rc5.md)

## [0.0.7rc4] - 2026-06-17

Runtime-benefit release for the v0.0.7 scheduler/KV-engine line.

### Core changes

- Added exact-prompt live Prefix Cache for greedy requests (`temperature=0`). Repeated identical prompts can skip the full prefill forward and share cached Paged KV pages plus the first sampled token.
- Prefix Cache is intentionally disabled for sampling requests (`temperature>0`) to avoid changing stochastic generation semantics.
- Added Paged KV request sharing APIs backed by page refcounts. Shared pages are released only after all request/cache references are gone.
- Added bounded LRU ownership for cached prompt pages to avoid unbounded KV retention.
- Improved chunked-prefill scheduling behavior: long prompts accumulate chunk credit and can be deferred while shorter prompts are admitted, improving mixed long/short prompt responsiveness without unsafe suffix-prefill execution.

### Expected test-visible benefit

- Repeated exact greedy prompts should show lower TTFT because prefill forward is skipped on cache hits.
- Mixed long/short prompt concurrency should show better short-request responsiveness when `--chunked_prefill --prefill_chunk_size` and a prefill token budget are enabled.
- Random datasets with no repeated prompts should not show Prefix Cache gains.

### Limitations

- This is exact full-prompt caching, not arbitrary partial-prefix reuse yet.
- Chunked prefill is scheduler interleaving, not true suffix-prefill kernel execution.
- Prefix Cache currently targets greedy correctness; stochastic Top-P requests stay on the uncached path.

### Tests

- Added regression tests for shared Paged KV pages, exact Prefix Cache hits, sampling-cache bypass, and chunked long-prompt deferral.

### Docs

- [v0.0.7rc4 release report](docs/releases/v0.0.7rc4.md)

## [0.0.7rc3] - 2026-06-17

Complete the safe runtime pieces of the v0.0.7 scheduler/KV-engine refactor.

### Core changes

- Added scheduler-level KV-pressure preemption. When prefill/decode reports KV capacity pressure, the scheduler can release one active request, requeue it, and rebuild its context from `prompt_tokens + generated_token_ids` without duplicating streamed tokens.
- Added `--max_preemptions` server option for continuous batching; default is `1`, and `0` disables preemption.
- Added live PagedAttention page reference counts, so shared/future prefix-cache pages are not returned to the free pool until the last reference is released.
- Added request page introspection for Paged KV debugging and future prefix-cache integration.
- Added mixed-length no-padding prefill packing metadata (`MixedLengthPrefillPacker`) as the stable contract for future packed prefill kernels.
- Existing chunked-prefill planner remains the explicit chunk contract; runtime execution stays conservative until the Attention path supports suffix-prefill safely.

### Compatibility and limitations

- Default behavior remains unchanged unless KV capacity pressure occurs or `--max_preemptions` is changed.
- True live prefix-cache reuse and no-padding/chunked prefill execution still require Attention/KV writer changes and are not falsely enabled in this release.
- No Atlas performance numbers are recorded for this release.

### Tests

- Added tests for KV page refcounts, mixed-length prefill packing, preempted-request context rebuild, and scheduler KV-pressure recovery.

### Docs

- [v0.0.7rc3 release report](docs/releases/v0.0.7rc3.md)

## [0.0.7rc2] - 2026-06-17

Bugfix release for TP continuous batching idle stability on Ascend.

### Bug fix

- Replaced continuous-batching TP control-plane HCCL tensor broadcast with CPU `StoreCommandChannel` backed by `torch.distributed.TCPStore`.
- Fixed rank 1 idle-time watchdog failure: `ACL stream synchronize failed, error code:507048` / `fftsplus timeout`.
- Worker ranks now block on CPU store metadata while idle and only enter NPU/HCCL for actual model execution.
- Added regression tests to prevent server continuous batching from using `TensorCommandChannel` again.

### Docs

- [v0.0.7rc2 release report](docs/releases/v0.0.7rc2.md)
- [Bug records](docs/bug_records.md)

## [0.0.7rc1] - 2026-06-17

Scheduler and KV-engine refactor foundation release.

### Core changes

- Continuous Batching adds `max_prefill_tokens` for prefill token-budget admission per scheduler tick.
- Oversized prompts can be admitted alone to avoid long-prompt starvation.
- Continuous Batching adds `max_decode_tokens` for active decode-row budgeting per scheduler tick.
- Added CPU-side `KVBlockRefCounter` for logical KV block refcount metadata.
- Added `PrefixCache` for block-aligned longest-prefix matching metadata.
- Added `ChunkedPrefillPlanner` as the planning entry for future chunked prefill execution.
- Server CLI adds `--max_prefill_tokens`, `--max_decode_tokens`, `--chunked_prefill`, and `--prefill_chunk_size`.

### Compatibility and limitations

- Defaults remain compatible: when token budgets are omitted, scheduling still follows `max_batch_size`.
- Prefix cache is metadata-only and is not wired into live PagedAttention KV reuse yet.
- Chunked prefill is a planning/configuration entry and does not change model execution semantics yet.
- No predicted Atlas 910B3 performance numbers are recorded in this release.

### Docs

- [v0.0.7rc1 release report](docs/releases/v0.0.7rc1.md)

## [0.0.6rc2] - 2026-06-12

修复v0.0.6rc1首版Vocab Parallel Greedy按Batch逐行发起小Collective导致的性能回归。

### Bug修复

- Greedy对整个Batch一次性计算局部最大Logit和Token ID；
- 将最大值和精确float32 Token ID打包为`[batch, 2]`，每个Decode Step只执行一次
  AllGather；
- 保持全局Greedy选择与旧版完整词表Argmax语义一致；
- 移除Batch=4时每Token八次小AllGather产生的HCCL启动开销；
- Benchmark在`temperature=0`时打印`Top-p: inactive (temperature=0)`，避免把默认
  `top_p=0.9`误解为实际启用。

### 验证状态

- 采样单元测试覆盖Batch级单Collective、跨Rank全局Token选择和Top-P状态显示；
- 本地CPU回归和Python静态编译通过；
- Atlas 910B3需要复测是否消除v0.0.6rc1相对v0.0.5rc2约7.5%的Greedy回归。

### 文档

- [v0.0.6rc2完整版本报告](docs/releases/v0.0.6rc2.md)

## [0.0.6rc1] - 2026-06-12

优化Qwen3 TP与Continuous Batching的Decode热路径，减少每Token的全词表通信、
Host同步、重复反分词和Python对象广播。

### 核心能力

- Qwen3 Dense、Qwen3 MoE和Qwen3-VL在TP模式下保留本地LM Head词表分片；
- Greedy采样仅交换各Rank局部最大值和全局Token ID；
- Top-P采样先交换有界候选集，并在无法证明候选集覆盖精确nucleus时自动回退完整
  Logits Gather，保证采样语义不变；
- Continuous Batching将最新Token和Decode Position保留在NPU；
- Rank 0每个模型Step只执行一次批量Token D2H，worker Rank不再复制Token到Host；
- 流式输出使用有界后缀增量反分词，边界不稳定时自动回退完整解码；
- TP Continuous Batching控制面由`broadcast_object_list`改为固定头部和张量Payload。

### 兼容性与验证

- 不改变现有`.pth`权重、PagedAttention、NPU Graph Bucket和OpenAI API；
- 完整Logprobs API仍按需Gather全词表Logits；
- Legacy单请求与多模态请求初始化仍可使用对象广播，它们不位于逐Token热路径；
- 本地相关CPU单元测试和Python静态编译通过；
- Atlas 910B3 TP=2吞吐与输出一致性需要服务器实测，本版本不填写预测性能。

### 文档

- [v0.0.6rc1完整版本报告](docs/releases/v0.0.6rc1.md)

## [0.0.5rc3] - 2026-06-12

同步最近版本的文档、Atlas实测结果和当前功能边界；相对v0.0.5rc2不修改推理执行逻辑。

### 文档与实测

- README新增模型、并行、Continuous Batching和NPU Graph支持矩阵；
- MoE启动示例明确区分TP Graph与EP Eager；
- 补录Qwen3-30B-A3B双卡EP Eager结果：5.5 tok/s、Batch 22.1 tok/s、
  181.09ms/token；
- Profiler示例更新为EP/TP Eager通信对照采集；
- 明确EP当前使用本地专家计算加AllReduce，并非Token All-to-All；
- 明确当前只支持单机多卡，尚未实现多机TP × EP二维并行；
- README.zh与主README同步，避免继续展示上游CUDA/ROCm旧说明。

### 文档

- [v0.0.5rc3完整版本报告](docs/releases/v0.0.5rc3.md)

## [0.0.5rc2] - 2026-06-12

修复Qwen3 MoE Expert Parallel启动Decode NPU Graph时因`aclnnNonzero`导致进程退出的问题。

### Bug修复

- EP路由需要使用`torch.nonzero`压缩本地专家assignment；
- Ascend `aclnnNonzero`会同步执行stream，不能进入NPU Graph Capture；
- EP模式现在启动时直接禁用Decode Graph并明确记录Eager回退；
- MoE TP模式和Dense模型继续保留现有NPU Graph路径；
- 避免尝试失败的Capture污染stream，不能仅依赖异常捕获后继续执行。

### 验证状态

- 新增EP禁用Graph、TP保留Graph的回归测试；
- 相关CPU测试和静态编译通过；
- Atlas服务器需确认EP能够完成Warmup与正式Benchmark。

### 文档

- [v0.0.5rc2完整版本报告](docs/releases/v0.0.5rc2.md)

## [0.0.5rc1] - 2026-06-11

增加文本服务Continuous Batching、MoE Decode小Batch专家内核和单机Expert Parallel。

### 核心能力

- OpenAI兼容文本Server由单请求串行执行升级为共享Continuous Batching调度器；
- 每个请求独立管理Paged KV request ID、序列长度、输出队列和结束释放；
- TP进程使用Prefill、Decode、Release步骤级命令保持动态Batch一致；
- 新增Triton Routed-GEMV专家后端，小Decode批次跳过通用专家排序与Gather/Scatter；
- `auto`后端按`tokens * top_k`在Routed-GEMV和Ascend GMM间选择；
- 新增`--moe_parallel_mode ep`，每卡持有部分完整专家并通过HCCL AllReduce合并局部输出；
- CLI、Server和Benchmark均可选择MoE TP或EP执行模式。

### 验证状态

- 47项Continuous Batching、Paged KV、NPU Graph和Qwen3 MoE CPU测试通过；
- 5项Atlas NPU测试入口在无NPU本地环境中按预期跳过；
- Python静态编译通过；
- 本地无Atlas NPU，Triton Ascend内核、EP双卡完整模型、服务并发与Graph Replay需要在910B3验证；
- 本版本不填写预测性能，实测结果后续写入性能历史记录。

### 文档

- [v0.0.5rc1完整版本报告](docs/releases/v0.0.5rc1.md)

## [0.0.4rc1] - 2026-06-11

完成Qwen3 MoE Decode热路径Host同步清理，并开放带安全回退的NPU Graph Capture/Replay。

### 核心能力

- PagedAttention在Prefill阶段缓存CPU请求ID，Decode不再每个Token读取NPU请求Tensor；
- 流式生成复用Token解码时已有的D2H结果判断EOS，删除额外的`eos_reached.all()`同步；
- Qwen3 MoE允许按`(batch_size, 128-token bucket)`尝试NPU Graph Capture；
- Graph Capture失败的Key只尝试一次，后续稳定回退Eager；
- Benchmark输出Graph attempts、captured、replays和fallbacks计数；
- 新增动态专家路由与GMM `group_list` Graph Replay的Atlas NPU测试。

### 验证状态

- Windows CPU契约与回归测试通过；
- Atlas 910B3需运行新增NPU测试确认当前CANN/torch_npu组合支持GMM、Triton路由和HCCL Graph Replay；
- 未填写预测性能，实测后写入性能历史记录。

### 文档

- [v0.0.4rc1完整版本报告](docs/releases/v0.0.4rc1.md)

## [0.0.3rc2] - 2026-06-11

修复Qwen3 MoE Triton路由Gather在Ascend Triton 3.2编译阶段失败的问题。

### Bug修复

- 不再读取`tl.atomic_add`返回的旧值作为专家分组写入位置；
- 改为在NPU上使用`torch.argsort`生成专家顺序，再由Triton融合Gather与路由元数据写入；
- 保持路由过程无CPU同步、无`.tolist()`和无逐专家Python循环；
- 移除Ascend Triton不建议手动传入的`num_warps`参数。

### 验证状态

- 新增Ascend Triton原子返回值兼容性回归测试；
- 19项Qwen3 MoE CPU单元与契约测试通过；
- Atlas 910B3需重新执行NPU GMM测试和双卡端到端启动。

### 文档

- [v0.0.3rc2完整版本报告](docs/releases/v0.0.3rc2.md)

## [0.0.3rc1] - 2026-06-11

将Qwen3-30B-A3B MoE专家执行从动态Python循环升级为Ascend Grouped MatMul与Triton设备侧路由。

### 核心能力

- Gate/Up与Down投影分别使用`torch_npu.npu_grouped_matmul`；
- Triton在NPU侧完成专家计数、按专家Gather和routing weight加权Scatter；
- 移除GMM热路径中的`torch.unique(...).tolist()`及逐专家Python循环；
- 模型加载时将现有`.pth`专家权重一次性转换为GMM原生`[expert, input, output]`布局，无需重新转换权重；
- 保留`eager`参考后端，并支持每个Sparse MoE层在TP AllReduce前进行数值对齐。

### 验证状态

- 17项Qwen3 MoE CPU单元与契约测试通过；
- 新增2项Atlas NPU真实GMM数值测试，本地无NPU环境时明确跳过；
- Python静态编译通过；
- Atlas 910B3端到端数值与性能结果需在目标服务器完成后写入，不在本版本文档中填写预测数据。

### 文档

- [v0.0.3rc1完整版本报告](docs/releases/v0.0.3rc1.md)

## [0.0.2rc2] - 2026-06-10

修复Qwen3-30B-A3B MoE在Prefill阶段因SwiGLU错误读取非连续Gate/Up视图而产生无关回答的问题。

### Bug修复

- SwiGLU Triton内核分别接收Gate、Up和输出张量的行跨度；
- 修复融合Gate/Up经过`chunk()`后输入stride大于输出stride时的错误寻址；
- Dense MLP连续张量路径保持兼容。

### 验证状态

- Qwen3 MoE单元测试由9项增加到10项并全部通过；
- Python静态编译和`git diff --check`通过；
- Atlas 910B3端到端回答正确性等待目标服务器验证。

### 文档

- [v0.0.2rc2完整版本报告](docs/releases/v0.0.2rc2.md)

## [0.0.2rc1] - 2026-06-10

新增Qwen3-30B-A3B MoE模型的正确性优先适配。

### 核心能力

- 新增`qwen3_moe`配置、模型注册和独立双卡CLI；
- 复用现有Qwen3 Attention、RoPE、FlashAttention、Flash Decoding和Paged KV链路；
- 支持128专家、TopK=8 Router以及按命中专家执行的SwiGLU专家MLP；
- 支持专家内部Tensor Parallel，Router复制，专家中间维切分并在输出端AllReduce；
- 权重转换器支持官方Qwen3-30B-A3B权重，并严格检查每层专家完整性；
- MoE首版显式关闭Decode NPU Graph，避免动态专家路径被错误Capture。

### 验证状态

- 配置、Router、专家计算、权重堆叠、TP切分和Graph降级单元测试通过；
- Windows CPU开发环境完成静态编译检查；
- Atlas 910B3双卡权重加载、端到端生成和性能数据需要在目标服务器继续验证。

### 文档

- [v0.0.2rc1完整版本报告](docs/releases/v0.0.2rc1.md)

## [0.0.1rc1] - 2026-06-09

首个带版本号的候选版本。

### 核心能力

- 支持Qwen3-32B在2 × Atlas 910B3上的FP16张量并行推理；
- 支持OpenAI兼容接口、真实SSE Streaming及EvalScope token usage统计；
- 支持Paged KV Cache、Flash Decoding和Triton Ascend融合算子；
- NPU Graph使用官方`NPUGraph`接口，并按128-token长度Bucket进行Capture/Replay；
- Paged KV Decode改为增量更新token映射，避免每个Token重建完整映射表；
- Graph与Eager模式的确定性输出验证通过。

### 性能

- Output Throughput：`24.0606 tok/s`
- TPOT：`40.6 ms`
- ITL：`40.9 ms`
- 相对Graph修复前的`5.696 tok/s`，吞吐提升约`322.4%`；
- 对比第三方vLLM-Ascend 0.8.4rc2结果，当前输出吞吐约为`3.15×`。

### 文档

- [v0.0.1rc1完整版本报告](docs/releases/v0.0.1rc1.md)

[0.0.1rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.1rc1
[0.0.2rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.2rc1
[0.0.2rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.2rc2
[0.0.3rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.3rc1
[0.0.3rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.3rc2
[0.0.4rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.4rc1
[0.0.5rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.5rc1
[0.0.5rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.5rc2
[0.0.5rc3]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.5rc3
[0.0.6rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.6rc1
[0.0.6rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.6rc2
