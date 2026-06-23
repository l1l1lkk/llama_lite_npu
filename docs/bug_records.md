# 错误复盘记录

本文件用于记录 Lite Llama NPU 开发过程中出现过的关键错误。项目的目标是个人学习和框架理解，所以这里不只记录“改了什么”，也记录“为什么会错、如何排查、以后如何避免”。

每个问题固定包含以下部分：

- 问题描述
- 排查过程
- 发现问题
- 分析问题
- 如何解决
- 预防措施

---

## 2026-06-10：MoE SwiGLU 内核复用行跨度错误，导致 Qwen3-30B-A3B 回答错位

### 问题描述

Qwen3-30B-A3B 能够完成模型加载和生成，但回答内容与用户问题明显无关。例如用户输入“你是谁”，模型输出却像是在回答 Python 函数题。该问题不是简单的性能慢，而是模型计算结果已经偏离正确语义。

### 排查过程

先排除采样随机性，将 `temperature=0`、`thinking=off`、`NPU Graph=off` 后问题仍存在。随后使用原始 Hugging Face 权重和相同 Prompt 验证，原始模型能够给出合理 top token，说明 tokenizer、chat template 和原始权重本身不是根因。继续检查转换后的 MoE 专家执行路径，重点对比 Router、TopK、专家 MLP、SwiGLU 和 Down 投影。

### 发现问题

问题集中在 MoE 专家 MLP 的 SwiGLU 内核调用。专家输入经过 Gate/Up 合并后，实际行跨度和内核假设不一致，导致多 token 或多专家场景下读取了错误位置的数据。Dense Qwen3 没有暴露该问题，因为 Dense MLP 的输入布局更简单，不经过动态专家分组。

### 分析问题

MoE 正确性不能只看“能生成文本”。语言模型即使中间计算错位，也可能输出语法正常但语义完全错误的内容。这个问题说明 MoE 的路由、专家分组、Gather、SwiGLU、Scatter 必须逐层数值对齐，不能只用端到端肉眼输出判断。

### 如何解决

修正 SwiGLU 内核输入布局和行跨度假设，并增加 MoE 路由与专家输出的参考对齐测试。验证顺序必须先对齐单专家，再对齐单层 MoE，最后再做端到端生成。

### 预防措施

新增 MoE 相关功能时，必须保留一组小尺寸 PyTorch 参考实现，用于验证 Router、TopK 权重归一化、专家 MLP 和最终 `index_add` 合并结果。

---

## 2026-06-11：Ascend Triton 的 `atomic_add` 不稳定，影响 MoE 路由聚合

### 问题描述

在实现 MoE Triton Gather/Scatter 时，使用 `tl.atomic_add` 统计专家写入位置。代码在部分环境下编译或运行失败，影响 Qwen3-30B-A3B 的专家分组路径。

### 排查过程

报错链路集中在 MoE 路由准备阶段：Router 输出 TopK 后，需要统计每个专家命中的 token 数量并生成专家偏移。编译失败与普通 MatMul 无关，而是出现在 Triton 生成 Ascend 后端代码时。

### 发现问题

Ascend Triton 对某些 atomic 模式支持不完整，尤其在动态 shape、TP 多进程和复杂索引组合下更容易暴露问题。

### 分析问题

MoE 路由阶段的统计、排序和回填属于控制密集型算子，不像 MatMul 那样稳定。把 CUDA Triton 写法直接搬到 Ascend 上风险较高，需要尽量减少复杂 atomic 依赖。

### 如何解决

将首版实现改为更保守的分阶段路径：能用 torch_npu GMM 的地方优先使用 GMM；Triton 只负责相对可控的 Gather/Scatter；遇到不稳定 atomic 模式时回退到更简单的张量路径。

### 预防措施

Triton Ascend 新内核必须先做最小 shape 编译测试，再做 TP 双卡测试，最后再接入完整模型。不能只在单卡或小输入上验证。

---

## 2026-06-17：TP Continuous Batching 空闲等待触发 HCCL Watchdog 超时

### 问题描述

Server 空闲一段时间后，Rank 1 在接收控制命令时崩溃，报错包含：

```text
RuntimeError: ACL stream synchronize failed, error code: 507048
HCCL watchdog thread terminated with exception
```

健康检查接口仍可能返回正常，导致问题看起来像“不影响运行”。

### 排查过程

Traceback 指向 `tp_control.py` 中对 NPU 张量执行 `.cpu().tolist()` 的控制面接收逻辑。Rank 0 和 Rank 1 在空闲时仍通过 HCCL 控制通道同步命令，某些情况下等待事件查询触发超时。

### 发现问题

问题不是业务请求失败，而是 TP worker 控制通道在空闲场景下仍依赖 NPU 事件同步；当 HCCL 任务没有按预期完成时，`.cpu()` 会触发同步并放大为 watchdog 异常。

### 分析问题

控制面不应过度依赖设备侧同步。模型计算可以在 NPU 上，调度命令应该尽量稳定、轻量，并保证主 Rank 和 worker Rank 的状态一致。

### 如何解决

后续版本将控制命令结构化，并减少每步不必要的 Host 同步。对于调试校验类 `.cpu()`，改为默认关闭，只在环境变量打开时启用。

### 预防措施

TP 控制面新增命令时，必须验证空闲、并发、异常取消和长时间运行场景。健康检查只能说明主进程 HTTP 层可用，不能代替 TP worker 健康检查。

---

## 2026-06-22：`ModelExecutor` 构造函数被 safe-prefill helper 意外截断

### 问题描述

`v0.0.8rc8` 后 Server 能启动，但第一次 Continuous Batching Prefill 时 Rank 1 报错：

```text
AttributeError: 'ModelExecutor' object has no attribute '_paged_prefix_cache'
```

### 排查过程

Traceback 指向 Prefix Cache 查询：

```text
ContinuousBatchModelBackend.prefill()
  -> _cache_lookup()
  -> ModelExecutor.share_paged_request_from_cache()
  -> self._paged_prefix_cache
```

`_paged_prefix_cache` 理应在 `ModelExecutor.__init__` 中初始化。检查 diff 后发现 `_infer_safe_prefill_tokens()` 被插入到了构造函数中间。

### 发现问题

新增 helper 方法提前结束了 `__init__` 代码块，导致后半段初始化逻辑不可达，包括请求管理器、Attention 信息、Prefix Cache 和 Graph Runner。

### 分析问题

这是代码放置错误，不是运行时竞态。`py_compile` 无法发现，因为语法仍然合法；只有真实实例化 `ModelExecutor` 才会暴露缺失属性。

### 如何解决

将 `_infer_safe_prefill_tokens()` 移到完整构造函数之后，并增加源码级回归测试，确认 helper 不会插入到构造函数中间。

### 预防措施

不要在长构造函数中间插入新方法。大类改动后至少要实例化核心对象，不能只跑静态编译。

---

## 2026-06-22：非 Chunked Packed Prefill 超过 Triton Grid 上限

### 问题描述

不开启 Chunked Prefill 时，混合长输入并发请求仍然可能在 Prefill 阶段失败。报错本质是 Packed Prefill 的 flat token 数过大，导致底层 Triton grid 或 kernel launch 超出安全范围。

### 排查过程

最初只关注 Chunked Prefill，但用户指出不开 Chunked Prefill 也会报错。继续检查发现 Server 端虽然有 `max_prefill_tokens`，但普通 Prefill 路径没有严格按预算拆分，仍可能把过大的 Prefill batch 一次性送入模型。

### 发现问题

调度层缺少类似 vLLM 的 Prefill token budget 约束。不开 Chunked Prefill 并不等于可以无限大 Prefill；普通 Prefill 也需要在 forward 前做安全切分。

### 分析问题

Prefill 是按输入 token 数放大计算量的阶段。并发数不高时，只要单请求或混合请求输入足够长，仍会触发大矩阵和大 attention grid。

### 如何解决

在调度层按 `max_prefill_tokens` 做预算控制，并在 forward 前增加安全校验。如果当前 batch 超出安全预算，降级为多次小 Prefill，用时间换稳定性。

### 预防措施

任何 Prefill 入口都必须经过 token budget，不允许只有 Chunked Prefill 路径做限制。

---

## 2026-06-22：Server 未暴露 `max_seq_len`，导致长 Prompt 被错误限制到 1024

### 问题描述

用户传入超过 1024 token 的测试后，Paged KV 分配失败，日志显示：

```text
current_tokens=1024, max_seq_len=1024
```

但启动命令里没有显式设置 `max_seq_len`，用户期望可以测试更长输入。

### 排查过程

检查 Server 参数发现，CLI 层没有暴露 `--max_seq_len`，模型执行器只能使用默认值。EvalScope 端即使生成更长输入，Server 端仍按 1024 处理。

### 发现问题

问题不是 Paged KV 没有空闲页，而是请求已经到达执行器允许的最大序列长度边界。

### 分析问题

`max_seq_len` 是模型执行和 KV 管理的硬约束，必须从 Server 启动参数传入，并在日志中打印，否则测试端参数和服务端容量会不一致。

### 如何解决

在 `server.py` 暴露 `--max_seq_len`，传入模型加载和执行器初始化，并在启动日志中显示。

### 预防措施

服务端所有影响容量和性能的参数必须显式打印，包括 `max_seq_len`、`page_size`、`max_prefill_tokens`、`max_batch_size` 和 `compiled_model`。

---

## 2026-06-22：Paged KV 在 `max_seq_len` 边界继续扩展导致分配失败

### 问题描述

Decode 阶段出现：

```text
Paged KV allocation failed: current_tokens=1024, max_seq_len=1024, free_pages=10331
```

空闲页很多，但请求仍然失败。

### 排查过程

查看 `extend_paged_requests()` 后发现，它只看到还有 free pages，却没有在扩展前稳定处理“请求已经达到最大长度”的情况。

### 发现问题

失败原因不是显存不足，而是请求已经达到最大序列长度。此时继续申请新 KV 页没有意义，应该停止该请求或返回长度结束原因。

### 分析问题

KV 管理有两类失败：容量不足和长度越界。二者日志必须区分，否则会误判为 page allocator 问题。

### 如何解决

在扩展 KV 前检查 `current_tokens >= max_seq_len`，将请求标记为结束，避免继续分配。

### 预防措施

KV 分配错误日志必须包含 `current_tokens`、`max_seq_len`、`free_pages` 和 request id，便于判断是容量问题还是长度问题。

---

## 2026-06-22：TP 命令派发没有等待 Worker 确认，导致状态推进过快

### 问题描述

Continuous Batching 中主 Rank 派发 Prefill/Decode 命令后，可能在 Worker 完成前推进下一步状态，最终造成 Rank 间请求状态不一致。

### 排查过程

对比 Rank 0 和 Rank 1 日志，发现主 Rank 已经进入下一轮 Decode，而 Worker 仍在处理上一个阶段。问题在短请求下不明显，在 Chunked Prefill 和混合长度输入下更容易出现。

### 发现问题

控制面缺少命令完成确认。主 Rank 认为命令已经发出就可以继续，但 Worker Rank 还没有完成对应的执行阶段。

### 分析问题

TP 多进程中，调度状态必须是全局一致的。只广播命令不等待确认，会让请求长度、KV 页状态和 active batch 发生分叉。

### 如何解决

为连续批处理控制命令增加确认机制，主 Rank 在推进状态前等待 Worker ack。

### 预防措施

所有会改变请求状态或 KV 状态的 TP 命令，都必须具备完成确认或等价同步点。

---

## 2026-06-22：TP Worker 收到未知 Continuous Batching 控制命令

### 问题描述

Rank 1 报错：

```text
RuntimeError: unknown continuous batching control_id: 19
```

### 排查过程

检查控制命令编码后发现，主 Rank 发送的新命令 ID 没有在 Worker 侧同步更新解析逻辑，或 Worker 解析到了与当前协议版本不匹配的控制值。

### 发现问题

TP 控制协议没有集中定义，新增命令时容易出现主 Rank 和 Worker Rank 的枚举不一致。

### 分析问题

这类问题属于协议版本不一致。即使模型计算代码正确，控制面协议错位也会直接导致多进程崩溃。

### 如何解决

将控制命令 ID 集中管理，Worker 对未知命令输出明确错误，同时保证新增命令时主从两端一起更新。

### 预防措施

控制协议新增字段或命令时，必须增加最小化双 Rank 测试，不能只测单进程路径。

---

## 2026-06-22：Chunked Prefill 中途 Replay 时 Paged KV 分配失败

### 问题描述

Chunked Prefill 测试中，Rank 1 在中途 replay 时失败：

```text
RuntimeError: Paged KV allocation failed for request 1
```

### 排查过程

最初尝试只在失败点前增加释放或扩容逻辑，但问题仍然复现。继续跟踪发现，中途 replay 会多次扩展同一个请求的 KV，调度层和执行器对“当前 chunk 已写入多少 KV”的理解不一致。

### 发现问题

Chunked Prefill 的增量执行路径不是普通 Decode，也不是完整 Prefill。它需要同时维护历史 KV、当前 chunk KV 和请求总长度，不能复用简单的一步一 token Decode 状态。

### 分析问题

问题发生在 KV 生命周期层：调度器、请求状态和 Paged KV 管理器没有形成单一事实来源。只修 allocator 不能根治。

### 如何解决

按 vLLM 思路重构：调度层先决定本轮 chunk 大小和 token budget，执行层只按给定 token 范围写 KV；请求状态在阶段完成后统一推进。

### 预防措施

Chunked Prefill 必须有明确状态机：未开始、Prefill 中、Decode 中、完成。不能让多个层同时修改请求长度。

---

## 2026-06-23：TP Chunked Prefill 首块请求状态分叉，导致 HCCL 超时

### 问题描述

开启 Chunked Prefill 后，NPU Graph 捕获阶段出现 HCCL 通信超时，错误信息中包含 `HcclAllreduce` 和 Rank 间连接失败。

### 排查过程

日志显示 first chunk 在某些 Rank 走了 FlashAttention 路径，另一些 Rank 走了 fallback 或不同 batch 形状。随后 Decode Graph 捕获开始等待未完成的 HCCL work，最终超时。

### 发现问题

TP Rank 间对同一批请求的 chunk 形状和执行路径产生了分叉。NPU Graph 捕获要求通信状态干净且 Rank 对齐，一旦某个 Rank 还在处理上一阶段 HCCL，就会失败。

### 分析问题

这是同步协议问题，不是单个 attention kernel 的数学问题。Chunked Prefill 中只要 Rank 间 batch、seq 或路径不一致，就会破坏后续 AllReduce 和 Graph Capture。

### 如何解决

确保主 Rank 在派发 chunk 命令时携带完整 shape 信息，Worker 不自行推断；Graph 捕获前等待必要同步，避免 pending HCCL work 进入 capture。

### 预防措施

TP + Chunked Prefill + NPU Graph 的组合必须优先保证 Rank 形状一致，再考虑性能优化。

---

## 2026-06-23：Paged Chunk FlashAttention 在 910B3 上触发 UB 溢出

### 问题描述

为 Chunked Prefill 编写 Triton Paged Chunk FlashAttention 后，部分 tile 配置在 Atlas 910B3 上编译或运行失败。

### 排查过程

检查 kernel 后确认数学逻辑方向正确：Q 来自当前 chunk，K/V 通过 paged KV 读取历史和当前块。但在较大 block 配置下，Triton Ascend 后端更容易触发 UB 资源不足或编译失败。

### 发现问题

问题集中在 tile 太大导致片上资源压力过高。CUDA 上可接受的 block 配置不能直接迁移到 910B3。

### 分析问题

Paged Chunk FlashAttention 比普通 no-pad FlashAttention 更复杂，因为它需要额外处理 paged block table、历史长度、当前 chunk 偏移和 causal mask。索引逻辑和片上缓存压力都会增加。

### 如何解决

增加 910B3 上更保守的 tile fallback，并记录运行时选择的 tile，避免单一大 tile 导致编译崩溃。

### 预防措施

Ascend Triton 内核必须做 runtime tile selection，不能假设一个配置覆盖所有 prompt 长度和 chunk 大小。

---

## 2026-06-23：固定小 Tile 让 Paged Chunk FlashAttention 性能回退

### 问题描述

为了避免 Paged Chunk FlashAttention 崩溃，将 tile 固定调小后，测试虽然能跑，但比未实现 FlashAttention 的版本还慢。

### 排查过程

对比日志发现，后续 chunk 虽然走了新 kernel，但小 tile 导致 kernel 数量增加、访存效率下降，并且 chunked 场景本身 prompt 不够长，收益不足以覆盖额外调度和索引成本。

### 发现问题

“使用 FlashAttention”不等于一定更快。对当前 512～1024 token 的测试，Packed Prefill 已经很快；Chunked Prefill 的价值主要在更长 prompt、显存压力或 Decode/Pefill 交错场景。

### 分析问题

Chunked Prefill 的目标不是优化短 prompt 单次 Prefill，而是控制长上下文的峰值显存和调度阻塞。当前场景下，它引入了额外分块、调度和 kernel launch 成本。

### 如何解决

保留普通 Prefill 和 Packed Prefill 的 FlashAttention 快路径；Chunked Prefill 暂时作为安全能力保留，不作为默认性能路径。后续只有在长上下文和并发场景中继续优化 Paged Chunk FlashAttention。

### 预防措施

性能优化必须区分适用场景。短 prompt、固定 batch 的 benchmark 不应强行证明 Chunked Prefill 收益。
