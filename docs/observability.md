# 可观测性与 Prometheus 指标

`v0.0.10rc3` 为连续批处理文本服务增加 Prometheus 指标。指标仅由 Rank 0
的 HTTP 服务导出，TP Worker 不重复计数。

## 接口

```bash
curl http://127.0.0.1:8213/metrics
curl http://127.0.0.1:8213/debug/stats
```

- `/metrics` 返回 Prometheus 文本格式。
- `/debug/stats` 返回调度器、KV Cache、NPU Graph、请求与失败统计的 JSON
  快照，用于人工排查。

## 请求指标

| 指标 | 类型 | 标签 | 含义 |
|---|---|---|---|
| `lite_llama_requests_total` | Counter | `endpoint`, `status` | 已结束请求数 |
| `lite_llama_request_failures_total` | Counter | `endpoint`, `reason` | 按稳定原因分类的失败数 |
| `lite_llama_request_latency_seconds` | Histogram | `endpoint` | 请求进入调度器到结束的总耗时 |
| `lite_llama_queue_wait_seconds` | Histogram | `endpoint` | 请求排队到首次准入的时间 |
| `lite_llama_time_to_first_token_seconds` | Histogram | `endpoint` | 请求提交到首 Token 的时间 |
| `lite_llama_inter_token_latency_seconds` | Histogram | `endpoint` | 相邻生成 Token 的时间间隔 |
| `lite_llama_prompt_tokens_total` | Counter | `endpoint` | 已接收 Prompt Token 数 |
| `lite_llama_generated_tokens_total` | Counter | `endpoint` | 已生成 Token 数 |

固定标签集合：

- `endpoint`：`chat`、`completion`、`unknown`
- `status`：`success`、`cancelled`、`error`

## 调度与运行时指标

| 指标 | 类型 | 含义 |
|---|---|---|
| `lite_llama_waiting_requests` | Gauge | 等待准入的请求数 |
| `lite_llama_prefilling_requests` | Gauge | 正在分块 Prefill 的请求数 |
| `lite_llama_running_requests` | Gauge | 正在 Decode 的请求数 |
| `lite_llama_preemptions_total` | Counter | 因 KV 压力被抢占的请求次数 |
| `lite_llama_kv_pages_used` | Gauge | 已使用 Paged KV 页数 |
| `lite_llama_kv_pages_free` | Gauge | 空闲 Paged KV 页数 |
| `lite_llama_graph_capture_attempts_total` | Counter | NPU Graph 捕获尝试次数 |
| `lite_llama_graph_captures_total` | Counter | 成功捕获次数 |
| `lite_llama_graph_replays_total` | Counter | Graph Replay 次数 |
| `lite_llama_graph_fallbacks_total` | Counter | 回退 Eager 的次数 |

KV 与 NPU Graph 指标只在抓取 `/metrics` 或访问 `/debug/stats` 时从
`ModelExecutor` 同步，避免在每个 Decode Token 上增加设备状态读取。

## 失败归因

异常文本不会直接作为 Prometheus 标签，避免产生高基数时间序列。固定
`reason` 包括：

| 原因 | 含义 |
|---|---|
| `queue_full` | 等待队列已满 |
| `invalid_request` | 请求参数不合法 |
| `context_limit` | Prompt 或总长度超过 `max_seq_len` |
| `kv_capacity` | KV 页不足、分配失败或 OOM |
| `client_cancelled` | 客户端断开或主动取消 |
| `model_execution` | 模型 Forward 或采样失败 |
| `hccl` | HCCL、Watchdog 或分布式通信错误 |
| `npu_runtime` | ACL、NPU Runtime 或 Ascend 执行错误 |
| `graph_failure` | NPU Graph 捕获或回放失败 |
| `internal` | 其他内部错误 |

完整异常仍应写入服务日志，并通过 `request_id` 定位具体失败样本。

## 常用 PromQL

成功 QPS：

```promql
sum(rate(lite_llama_requests_total{status="success"}[1m]))
```

P99 总延迟：

```promql
histogram_quantile(
  0.99,
  sum by (le) (rate(lite_llama_request_latency_seconds_bucket[5m]))
)
```

P99 TTFT：

```promql
histogram_quantile(
  0.99,
  sum by (le) (rate(lite_llama_time_to_first_token_seconds_bucket[5m]))
)
```

P99 ITL：

```promql
histogram_quantile(
  0.99,
  sum by (le) (rate(lite_llama_inter_token_latency_seconds_bucket[5m]))
)
```

最近五分钟最大队列深度：

```promql
max_over_time(lite_llama_waiting_requests[5m])
```

按失败原因统计：

```promql
sum by (reason) (rate(lite_llama_request_failures_total[5m]))
```

输出 Token 吞吐：

```promql
sum(rate(lite_llama_generated_tokens_total[1m]))
```

KV Cache 使用率：

```promql
lite_llama_kv_pages_used
/
(lite_llama_kv_pages_used + lite_llama_kv_pages_free)
```

## Prometheus 抓取配置

```yaml
scrape_configs:
  - job_name: lite-llama-npu
    scrape_interval: 5s
    static_configs:
      - targets:
          - 127.0.0.1:8213
```

## 当前范围

首版完整请求生命周期指标覆盖连续批处理文本请求。旧版逐请求 Eager 路径和
视觉模型路径仍可获取 Python 进程指标、KV 与 Graph 快照，但尚未完整接入
请求级 TTFT、ITL 和失败分类。
