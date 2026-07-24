# 推理请求与模型层可视化

Lite Llama NPU 的 Trace 工具用于观察真实请求在调度器、Prefill、Decode 和
Decoder Layer 中的进展。它与 Prometheus 指标互补：

- Prometheus 用于低基数、长期运行的聚合指标；
- Trace 用于高基数、短时间诊断和请求级回放。

Trace 默认关闭，并且不会记录 Prompt 文本、输出文本或原始 Token ID。

## 快速开始

推荐通过 Trace CLI 启动：

```bash
python -m lite_llama.trace_cli run \
  --trace-level layer \
  --trace-output traces/qwen3-live.jsonl \
  --no-open \
  -- \
  --checkpoints_dir my_weight/Qwen3-32B \
  --port 8213 \
  --continuous_batching
```

然后访问：

```text
http://127.0.0.1:8213/debug/trace
```

也可以直接使用原有服务入口：

```bash
python server.py \
  --checkpoints_dir my_weight/Qwen3-32B \
  --port 8213 \
  --trace \
  --trace_level scheduler \
  --trace_output traces/server.jsonl
```

## Trace 级别

| 级别 | 事件 | 典型用途 |
|---|---|---|
| `request` | 请求提交、准入、Token、结束、失败 | 在线请求诊断 |
| `scheduler` | Request 事件，以及 Prefill、Chunked Prefill、Decode Batch | 调度与批处理分析 |
| `layer` | Scheduler 事件，以及 Decoder/Visual Layer 进度 | 模型结构学习与逐层诊断 |

`trace_cli run --trace-level layer` 在没有显式指定 Graph 开关时，会自动增加
`--no_compiled_model`。原因是 NPU Graph Replay 作为一个完整图执行，不会在每次
Replay 时重新进入 Python Layer Hook。

如果显式同时使用 `--compiled_model` 和 `--trace_level layer`，页面仍能显示请求、
Scheduler 和 Graph Decode Batch，但逐层事件只可能出现在 Graph Capture 或 Eager
Fallback 路径中，不能解释为每次 Replay 的真实逐层执行。

## HTTP 接口

| 接口 | 用途 |
|---|---|
| `GET /debug/trace` | 自包含 Trace 前端 |
| `GET /debug/trace/snapshot` | 当前会话元数据和环形缓冲区快照 |
| `GET /debug/trace/events` | SSE 实时事件流 |

SSE 事件包含单调递增的 `seq`。客户端重连时可通过 `after_seq` 或
`Last-Event-ID` 继续消费。

## 事件关联

Continuous Batching 下，一次模型 Forward 会同时承载多个请求。因此 Trace 使用：

- `request_id`：OpenAI 兼容请求；
- `batch_id`：一次 Prefill、Prefill Chunk 或 Decode 调用；
- `layer_index`：该 Batch 经过的模型层；
- `rank`：Tensor Parallel Rank。

Layer 事件属于 `batch_id`，并携带该 Batch 的 `request_ids`，不能将一次 Layer
调用错误解释为单请求独占执行。

## 时间含义

页面中的 Layer 时间是 `Host dispatch time`，即 Python 调用和 NPU 命令下发时间。
Hook 不调用 `.cpu()`、`.item()` 或 NPU synchronize，因此不会把设备同步引入
Decode 热路径。

需要准确设备执行时间时，应使用项目 profiler 和 MindStudio Insight。实时 Trace
用于结构、状态迁移和请求关联，不替代设备级性能分析。

## Tensor Parallel

TP 模式下，`--trace_output traces/run.jsonl` 会产生：

```text
traces/run.rank0.jsonl
traces/run.rank1.jsonl
```

Rank 0 页面展示 HTTP/Scheduler 请求生命周期；每个 Rank 的 JSONL 文件保留本 Rank
的 Backend 和 Layer 事件，可按 `batch_id`、`control_ids` 和时间戳进行对齐。

## 性能与安全

- 实时缓冲区默认为 50,000 个事件；
- 缓冲区已满时丢弃最旧事件，并在快照中增加 `dropped_events`；
- JSONL 使用后台线程写入，推理线程不等待磁盘；
- Trace 关闭时不注册 Layer Hook；
- 不记录 Prompt、输出文本和原始 Token；
- `/debug/trace` 属于调试接口，对外暴露服务时应增加网络访问控制。
