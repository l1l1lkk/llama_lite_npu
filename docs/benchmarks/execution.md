# Canonical v2 Benchmark 执行指南

## 当前阶段

runner 目前只支持 `--dry-run`，不会启动服务、连接 endpoint、执行 EvalScope 或创建输出目录。唯一 capability 配置为：

```bash
python -m benchmarks.serving.run_campaign \
  --framework lite_llama \
  --campaign benchmarks/configs/campaigns/capability_smoke.yaml \
  --runtime-value QWEN3_32B_LITE_CHECKPOINT=/data/liuke/llama_lite_npu/my_weight/Qwen3-32B \
  --runtime-value QWEN3_32B_TOKENIZER=/data/model_weights/Qwen3-32B \
  --client-profile /data/liuke/benchmark-envs/evalscope-client/profile.json \
  --output-root /data/liuke/benchmark-diagnostics/phase3b/<commit>/lite_llama \
  --dry-run

python -m benchmarks.serving.run_campaign \
  --framework vllm_ascend \
  --campaign benchmarks/configs/campaigns/capability_smoke.yaml \
  --base-url http://127.0.0.1:18000 \
  --runtime-value QWEN3_32B_HF_CHECKPOINT=/data/model_weights/Qwen3-32B \
  --runtime-value QWEN3_32B_TOKENIZER=/data/model_weights/Qwen3-32B \
  --client-profile /data/liuke/benchmark-envs/evalscope-client/profile.json \
  --output-root /data/liuke/benchmark-diagnostics/phase3b/<commit>/vllm_ascend \
  --dry-run
```

`--output-root` 是计划中的精确根目录；dry-run 不创建它。旧 `p0_smoke` ID 已废弃，避免 capability 与 performance 各自形成“当前入口”。

`--runtime-value NAME=VALUE` 可重复使用，只能绑定模型配置实际声明的完整占位符。值必须是规范的 Linux 绝对路径，按单个 argv 字符串传递，不做 shell expansion、命令替换或部分字符串替换。未绑定时仍可 planning，但计划会保留占位符并列出 `unresolved_runtime_values`。

`--client-profile` 必须指向已经通过 CPU preflight 的绝对 JSON 路径。未提供 profile 时仍可 dry-run，但 `client_profile_status=unresolved`、`capability_execution_ready=false`，命令首元素为不可执行标识 `__CLIENT_PROFILE_REQUIRED__`，绝不回退到 PATH 中的裸 `evalscope`。只有模型 runtime bindings 和 verified client profile 同时完整时，`capability_execution_ready` 才能为 `true`；它仍不代表 strict correctness 或 performance eligibility 已通过。

## Runtime endpoint

canonical endpoint 永远是 `POST /v1/chat/completions`。base URL 只允许带显式端口的 HTTP/HTTPS origin，禁止 credentials、path、query 和 fragment。resolved base URL 同时驱动：

1. adapter 健康检查 URL；
2. EvalScope client URL；
3. server launch command 的 `--port`。

默认端口为 Lite `8213`、vLLM-Ascend `8000`。Phase 3A 现场中 `8000` 已被无关 uvicorn 占用，`8012` 是长期 Qwen3.5 服务；二者不得干扰。Phase 3B vLLM capability smoke 使用 override `18000`，Lite 使用 `8213`。

## 统一 client 合同

两个框架共用同一 EvalScope/OpenAI chat stream 命令生成器、冻结 JSONL、seed、offset、request order、sampling 和 fixed-output 语义。正式 client argv 只能由 `benchmarks/serving/evalscope_client.py` 生成。

计划额外冻结：

```json
{"TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
```

它只阻止后端自动加载，不代表 server 禁用 NPU，也不能阻止第三方包显式执行 `import torch_npu`。Phase 3B-3 的 rejected diagnostic 已证明：EvalScope 1.7.1 经 ModelScope 1.37.0、Transformers 5.8.0 和 Accelerate 1.6.0 显式导入 `torch_npu`，最终因 CPU client 不可见 `libhccl.so` 而在请求提交前失败。因此该环境变量只是 profile 的必要条件，不是充分条件。环境变量作为独立序列化字段传递，不能拼进 shell argv；paired validator 会逐字段比较。

### Client profile 与 CPU preflight

client profile schema v1 冻结绝对 Python/EvalScope executable、`python_prefix` 环境根、诊断用 `python_base_prefix`、精确包版本、requirements lock SHA、关键 dist-info 指纹、EvalScope flags、tokenizer 关键文件 SHA 与 CPU load 结果。`accelerate`、`torch`、`torch_npu` 必须显式记录；当前 `cpu_isolated_no_torch` policy 要求三者为 absent/null。profile 的 overall fingerprint 不包含生成时间，但覆盖全部执行身份字段。

Python executable 身份使用规范化但不跟随符号链接的路径；禁止使用 `resolve()`/`realpath()` 或仅用 `samefile` 判断 venv。这样 `<venv>/bin/python` 即使链接到基础解释器也保留所选 venv 身份。同时必须满足 executable 位于 `<python_prefix>/bin/`、live `sys.prefix` 等于 profile prefix，且 `sys.prefix != sys.base_prefix`。因此共享同一基础解释器的 sibling venv 和基础解释器本身都不能冒充目标 client。`python_prefix`、`python_base_prefix` 与 executable 都进入 overall fingerprint 和 paired 字段级比较。

推荐候选为 Python 3.10、EvalScope 1.8.0、ModelScope 1.36.3、Transformers 5.5.3 的隔离 client，且不安装 Accelerate、torch 或 torch_npu。该组合目前只是只读参考环境中 imports/help 已通过，尚待服务器 Python 3.10 独立环境和本地 Qwen3 tokenizer CPU load 验证，不能提前标成服务器可用。

计划中的 `client_preflight_command` 同时包含 argv 与环境，必须在任何 adapter/server/NPU lifecycle 前执行。它重新检查 executable、package/import、CPU-only local tokenizer load、EvalScope flags、lock SHA 和 dist-info fingerprint；返回非零时 orchestration guard 保证 adapter launch 次数为 0。Lite 与 vLLM-Ascend 必须使用同一个 profile fingerprint、executable 与 tokenizer identity。

## Diagnostic contract

`kind=capability` 必须同时满足：

- `comparison_scope=capability_only`；
- `publishable=false`；
- `aggregation_allowed=false`；
- `result_namespace=diagnostics`；
- 计划输出 `evidence_class=diagnostic` 和 `performance_eligible=false`。

capability smoke 使用 stream、greedy、`min_tokens=max_tokens=64`，独立 warmup/formal JSONL 和一个有界 cell。workload 仍为 `actual_token_calibrated=false`，不能进入性能 aggregate、baseline 或 current history。

performance campaign 若 token 未校准、workload 未 strict publishable、固定输出未 probe、环境指纹不完整或 checkpoint equivalence 未验证，也必须输出明确的不合格原因。

## Checkpoint 身份

模型配置拆为两层：

- `logical_identity`：model ID、config/tokenizer SHA、dtype、TP、最大上下文等共同身份；
- `representations`：Lite 的 custom PTH 与 vLLM-Ascend 的 HF safetensors 路径、格式、完整 fingerprint 和 provenance；
- `equivalence`：两种 representation 是否已证明等价、是否允许因果比较。

adapter 只能读取本框架 representation。当前 PTH 转换来源与完整权重 fingerprint 未冻结，等价状态为 `unverified` / `valid_for_causal=false`。这不阻止 diagnostic smoke，但会阻止 performance eligibility。伪造或漂移任一 config/tokenizer SHA 是 paired hard fail。

## Production 与 controlled 边界

`production_stack` 可以比较各框架真实发布栈，但必须完整保留 CANN/PyTorch/torch_npu/框架版本、容器、驱动和硬件指纹；不得写成单变量因果实验。

`controlled_stack` 只有在 checkpoint 等价且 CANN、PyTorch、torch_npu 等共享栈字段完全匹配时才有因果资格。当前现场不满足该条件。

## Bundle v2 目录契约

正式 compact evidence 使用以下唯一结构：

```text
<bundle>/
  campaign.json
  environment/
    <framework>.json
  workload/
    workload.json
    formal.jsonl
    warmup.jsonl
  runs/<framework>/<case_id>/repeat-01/
    metadata.json
    timing.json
    client/
      command.txt
      exit_code.txt
      benchmark_args.json
      benchmark_summary.json
      benchmark_percentile.json
      requests.json
    server/
      command.txt
      before_metrics.prom
      after_metrics.prom
      before_stats.json
      after_stats.json
      request_timing.json
      timeseries.jsonl
  derived/
    aggregate.json
    aggregate.csv
  diagnostics/
    index.json
  omitted.json
  manifest.json
```

`manifest.json` 必须覆盖除自身外的全部 compact evidence，并为每个规范相对路径记录 SHA256 与 size；manifest 不得自引用，不得包含绝对路径、`..` 逃逸或重复路径。其 `file_count` 与 `size_bytes` 汇总必须和实际文件一致。

SQLite、HTML、完整 stdout/server log、profiler 和权重不进入 compact bundle。它们只能写入 `omitted.json`，并记录 server path、size、SHA256、保留策略和重新生成命令。omitted 不得删除重建 aggregate、strict correctness 或环境指纹所需的紧凑证据。

## 离线校验与重建

复制到脱离服务器的临时目录后执行：

```bash
python -m benchmarks.serving.validate.bundle <copied-bundle>
python -m benchmarks.serving.validate.bundle <copied-bundle> --rebuild
```

aggregate 必须从 request-level evidence 生成，并且先于 `manifest.json`。`--rebuild` 改写或补建 `derived/aggregate.json` 后，正式封包流程必须重新生成 manifest，再做一次无 `--rebuild` 的全量 SHA/size 校验；不能保留覆盖旧 aggregate 的陈旧 manifest。

## 进入 NPU 前置门

1. 唯一代码 SHA、模型 logical identity、framework representation、runtime bindings 与环境指纹已记录；
2. 两框架 common client/workload contract 与 frozen JSONL SHA 完全一致；
3. capability 运行必须具有 verified client profile，且 CPU preflight 必须先于 server/NPU；模型 bindings 与 profile 均完整后才允许 `capability_execution_ready=true`，但未校准 token 的 diagnostic 仍不得进入性能汇总；
4. NPU 6/7 空闲，目标端口关闭，无残留 server/worker/client；不得干扰 8000 与 8012；
5. lifecycle、warmup/formal 边界、cache state、timeout、progress watchdog 和停止门已预注册。

## Formal hard gates

- success/failed、实际 input/output token、request count/order；
- workload/environment/checkpoint fingerprint；
- fixed-output capability probe 与实际 token 校准；
- Graph 状态只能是 pass/fail/unsupported；结构化状态为pass时 fallback delta大于0即hard fail，required/causal还要求formal capture delta=0且replay增长；
- request-level evidence、server before/after snapshots 与时间序列完整；
- performance correctness 与 semantic accuracy 分开建campaign、分开报告，不合成单一总分。

任一 OOM、strict mismatch、fingerprint mismatch、Graph fallback、真实停滞或生命周期污染都必须保留为 rejected diagnostic，不得覆盖正式 run ID，也不得进入 aggregate、baseline 或 current history。

## Phase 3B 停止与清理门

正式 capability smoke 前必须复核 NPU 6/7、端口、残留进程和模型挂载。OOM、HTTP/stream失败、strict token mismatch、Lite Graph fallback、真实停滞或无法安全判断残留状态时，只保留 rejected diagnostic 并停止；不得写入 aggregate。

每个生命周期结束只能停止本阶段创建的客户端、server、worker或专用容器。必须复验对应端口关闭、无残留进程、NPU 6/7恢复空闲；不得删除旧 checkout、历史 bundle、profiler 或其他用户资产。清理证据本身写入 diagnostic metadata。
