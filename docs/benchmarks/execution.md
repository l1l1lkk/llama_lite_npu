# Canonical v2 Benchmark 执行指南

## 范围

canonical v2 统一 Lite Llama 与 vLLM-Ascend 的 workload、EvalScope/OpenAI client、strict validator 和证据包。框架差异只留在 adapter 的服务启动、健康检查和 metric capability 映射中。

Phase 2 仅实现本地计划与合成校验。以下 dry-run 不启动服务、不连接 endpoint、不执行 EvalScope，也不写入 `benchmarks/results/`：

```bash
python -m benchmarks.serving.run_campaign \
  --framework lite_llama \
  --campaign benchmarks/configs/campaigns/p0_smoke.yaml \
  --dry-run

python -m benchmarks.serving.run_campaign \
  --framework vllm_ascend \
  --campaign benchmarks/configs/campaigns/p0_smoke.yaml \
  --dry-run
```

`--campaign p0_smoke` 与显式路径等价。当前没有实现非 dry-run 执行入口；省略 `--dry-run` 会直接失败，避免 Phase 2 误启动服务器。

## 统一契约

- endpoint：OpenAI `POST /v1/chat/completions`，stream 开启；端口由 adapter 私有配置决定。
- dataset：同一冻结 `line_by_line` JSONL、相同 SHA256、seed 和顺序。formal 与 warmup 使用独立 JSONL，并分别声明相对各自文件的 `formal_offset`、`warmup_offset`；不得把 formal 的大偏移套到 warmup 文件。
- sampling：campaign 统一声明；P0 smoke 为 greedy，`temperature=0`、`top_p=1`。
- fixed output：`min_tokens=max_tokens`。只有 capability probe 与服务端实际 token 校准完成后才可能 strict pass。
- lifecycle：每个正式 run 独立服务生命周期；warmup 在 formal 边界之外。
- failure gates：逐请求 success、actual input/output、count/order、workload/environment fingerprint、Graph/fallback。
- accuracy：performance correctness 只验证服务与 workload 契约；semantic accuracy 使用独立 campaign/schema，不合成总分。

## Adapter 边界

`benchmarks/serving/evalscope_client.py` 是正式 client 命令的唯一生成位置。adapter 不得拼接另一套 client。

| 项目 | Lite Llama | vLLM-Ascend |
| --- | --- | --- |
| 默认端口 | 8213 | 8000 |
| 服务命令 | `torch.distributed.run ... server.py` | `vllm serve` |
| page/block | page size 16 | block size 128 |
| request timing | supported | 当前配置为 unsupported，待 capability probe |
| Graph counters | supported | 当前配置为 unsupported，不能伪造零值 |

page/block 与 Graph 实现是框架内部参数。跨框架只比较预注册 production mode；框架内部 Graph on/off 必须另建 campaign。要求 Graph 因果结论时，`unsupported` 结果为 invalid。

## 配置与 workload

```text
benchmarks/configs/models/<model>.yaml
benchmarks/configs/frameworks/<framework>.yaml
benchmarks/configs/campaigns/<campaign>.yaml
benchmarks/workloads/<workload_id>/{workload.json,formal.jsonl,warmup.jsonl}
```

模型路径使用环境占位符，Phase 3 必须解析到两个框架共同使用的 checkpoint/tokenizer，并记录文件/config SHA。`qwen3_32b_p128_smoke_uncalibrated_v1` 是合成规划 fixture；它不满足实际 input token 校准，不得转成正式结果。

## Bundle v2

```text
benchmarks/results/<campaign_id>/
  campaign.json
  environment/<framework>.json
  workload/{workload.json,formal.jsonl,warmup.jsonl}
  runs/<framework>/<case_id>/repeat-01/
    metadata.json
    timing.json
    client/{command.txt,exit_code.txt,benchmark_args.json,
            benchmark_summary.json,benchmark_percentile.json,requests.json}
    server/{command.txt,before_metrics.prom,after_metrics.prom,
            before_stats.json,after_stats.json,request_timing.json,timeseries.jsonl}
  derived/{aggregate.json,aggregate.csv,...}
  diagnostics/index.json
  omitted.json
  manifest.json
```

`manifest.json` 覆盖除自身外的全部 compact evidence，并记录 size/SHA256。SQLite、HTML、完整 stdout/server log、profiler 和权重不能进入 compact bundle；它们只能在 `omitted.json` 中记录服务器路径、size、SHA256 和重建方法。

离线校验：

```bash
python -m benchmarks.serving.validate.bundle <copied-bundle>
python -m benchmarks.serving.validate.bundle <copied-bundle> --rebuild
```

重建会从 `runs/*/*/repeat-*/client/requests.json` 生成最小 aggregate。重建后若 bundle 已有 manifest，应重新生成 manifest；正式构包流程必须确保 aggregate 先于 manifest 生成。

## 阶段门与停止条件

### 进入 NPU 前

1. 当前代码 SHA、两个运行环境、模型/config/tokenizer SHA 已冻结；
2. 两个 dry-run 的 common client contract 和 run matrix 一致；
3. fixed-output capability 与 chat-template 后实际 token 长度已校准；
4. NPU 6/7 空闲、目标端口关闭、无残留 server/worker/client；
5. campaign 顺序、warmup、cache state、超时和进度 watchdog 已预注册。

### Formal hard gate

- strict input/output、request count/order、success/failed；
- workload/environment fingerprint；
- Graph `pass/fail/unsupported`；只要结构化状态为 `pass`，`fallback_delta>0` 就是 hard fail；Graph required/causal 时还要求 replay 增长、formal capture=0。optional 且 `unsupported` 可以记录后继续，但不能伪造零值；
- request-level evidence、server snapshots 和时序完整。

OOM、strict 失败、fingerprint mismatch、Graph fallback 或真实停滞时保留 rejected diagnostic，不覆盖正式 run ID。不同 checkpoint、模板、输出语义或 workload 的数字不得计算加速比。
