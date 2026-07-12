# `min_tokens` 固定输出长度机制与严格 workload 验证

## 问题现象与根因

第一阶段 EvalScope 命令同时设置了 `--min-tokens 256` 和 `--max-tokens 256`，但服务端的 `ChatCompletionRequest`、`CompletionRequest` 均没有 `min_tokens` 字段。Pydantic 因而丢弃该字段，scheduler 只收到 `max_tokens`。模型在 256 token 前采到 EOS 时会立即结束，造成部分 run 的平均输出长度不足 256。

根因不在 EvalScope 计数，而在请求参数没有贯穿服务端调用链，采样层也没有按请求屏蔽 EOS。

## 固定环境

- 模型/checkpoint：`/data/liuke/llama_lite_npu/my_weight/Qwen3-32B`；`config.json` SHA256 为 `97e295b63283935788fac5e4f8860862a56d4089538cafc93f0431f2ebe483bb`。
- 代码：`release/0.0.12rc1`，实测实现 SHA `f393a6881c66abc33466dff0dcd3d6b83b3a460d`。
- 硬件与执行：2×Atlas 910B3，物理 NPU `6,7`，TP=2，FP16，NPU Graph on，max sequence length 4096，page size 16。
- 软件：CANN `26.0.rc1`、PyTorch `2.7.1`、torch_npu `2.7.1`、triton-ascend `3.2.0`、EvalScope `1.7.1`。
- 完整启动命令保存于 `on/server/start-command.txt`；环境变量、模型配置、NPU 状态和软件包快照保存于 bundle 的 `environment/`。

## 方案选择

采用与现有接口兼容的 `min_tokens`：默认值为 0；当某一请求已经采样的 token 数少于该请求的 `min_tokens` 时，在采样前把该 batch row 的 EOS logit 置为负无穷；达到阈值后恢复 EOS，允许正常停止。

没有采用“采到 EOS 后忽略并继续循环”的做法，因为这种方式已经选择了 EOS，可能产生重复 EOS、空文本和错误的增量解码状态。没有采用 Qwen3 专用 token ID 或 workload 硬编码；EOS 来自当前 tokenizer。

## 请求与采样调用链

```text
EvalScope min_tokens
  -> OpenAI ChatCompletionRequest / CompletionRequest
  -> _submit_continuous_request
  -> ContinuousBatchScheduler.submit
  -> BatchRequest(min_tokens, sampled_token_count)
  -> TP prefill control message（逐请求同步 min_tokens）
  -> ContinuousBatchModelBackend._sample_device
  -> sample_next_token
  -> 按 batch row 和 TP vocab shard 屏蔽 EOS logit
```

每次采样后，rank 0 与 worker rank 都在 device token 记账路径递增 `sampled_token_count`，因此 TP 各 rank 对是否屏蔽 EOS 的判断一致。batch 内不同请求可使用不同阈值。

## 正确性与兼容性风险

- `min_tokens=0` 或未传时不屏蔽任何 token，保持原行为；无屏蔽 token 时不会复制 logits。
- 负值由 schema 和 `BatchRequest` 拒绝；`min_tokens > max_tokens` 同样拒绝。
- greedy 与 Top-P 共用采样前屏蔽逻辑；本阶段服务器实测固定 greedy。
- EOS 在达到 `min_tokens` 后恢复，可以产生正常的 `stop`；达到 `max_tokens` 则返回 `length`。
- TP 使用全局 EOS token ID，根据当前 rank 的 vocab 起始位置换算本地列，只有持有 EOS 的 rank 修改 logits。
- 当前实现作用于 continuous batching 文本端点；非 continuous batching 的旧生成器路径不在本阶段实测范围。

## 单元测试与服务测试

覆盖项包括：schema 默认值与边界校验、chat/completion 参数透传、streaming 参数透传、batch 内不同阈值、greedy 按行屏蔽、TP shard 全局/本地 token ID 换算、阈值后 EOS 正常停止、TP 控制消息往返一致性。

测试命令：

```bash
python -m unittest \
  tests.test_min_tokens tests.test_tp_control \
  tests.test_vocab_parallel_sampling tests.test_continuous_batching \
  tests.test_server_batching tests.test_repository_docs \
  tests.test_observability tests.test_server_stream_usage
```

本地相关测试 97 项通过；实现提交同步服务器并加载 CANN 环境后，服务器相关测试 86 项通过。`py_compile` 与 `git diff --check` 通过。第一阶段旧 bundle 仍能重建 15 个 run、5 个 aggregate row，并验证 171 个文件，证明本次脚本改动没有破坏旧数据包。

## 实测矩阵与原始 run ID

固定环境：Qwen3-32B、FP16、TP=2、Atlas 910B3 NPU 6/7、NPU Graph on、prompt=128、output=256、temperature=0、top_p=1、seed=42。

| 并发 | 正式请求数/run | warmup | 正式轮次 | strict input | strict output | failed | run ID |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 12 | 2 | 3 | 3/3 | 3/3 | 0/36 | `..._c1_r1`、`..._c1_r2`、`..._c1_r3` |
| 4 | 24 | 8 | 3 | 3/3 | 3/3 | 0/72 | `..._c4_r1`、`..._c4_r2`、`..._c4_r3` |

本阶段只验证固定输出机制，不计算或宣传 Graph speedup。

完整 run ID 前缀为 `20260712_qwen3_32b_tp2_fp16_min_tokens_on_p128_o256`。逐 run 的 `benchmark_args.json`、`benchmark_summary.json`、`benchmark_percentile.json`、命令、exit code、server before/after metrics 与 debug stats 均位于 `benchmarks/results/20260712_qwen3_32b_tp2_fp16_min_tokens/`。`strict-validation.json` 依据 summary 平均值与 percentile 文件中的全部已报告输入/输出 token 值重建，6 个正式 run 全部为 pass。

```text
20260712_qwen3_32b_tp2_fp16_min_tokens_on_p128_o256_c1_r1
20260712_qwen3_32b_tp2_fp16_min_tokens_on_p128_o256_c1_r2
20260712_qwen3_32b_tp2_fp16_min_tokens_on_p128_o256_c1_r3
20260712_qwen3_32b_tp2_fp16_min_tokens_on_p128_o256_c4_r1
20260712_qwen3_32b_tp2_fp16_min_tokens_on_p128_o256_c4_r2
20260712_qwen3_32b_tp2_fp16_min_tokens_on_p128_o256_c4_r3
```

聚合数据只作本轮可复核记录，不用于 Graph 加速比：c1 平均 E2E 10.2604 s、TTFT 407.05 ms；c4 平均 E2E 22.2614 s、TTFT 11161.40 ms。相应原始数字和统计计算均可从 bundle 的 `summary.csv`、`aggregate.csv` 及逐 run JSON 重建。

### 被拒绝尝试

c4 原始 r3（offset `12804072`）为 24/24 成功且 output=256，但 1 个请求 input=127，平均 input=127.9583，因此未纳入正式结果。随后使用 output=1 的短请求筛选 offset；`12804096` 平均 input=127.9167，`12804024` 为 24/24 input=128。筛选后的同服务实例发生 scheduler 等待而 AICore=0 的停滞，未产生正式 summary；客户端被终止并安全停止服务。全新服务实例使用已验证的 `12804024` 补跑 r3，最终通过。被拒绝 run、筛选结果、SQLite、HTML 和完整日志均为 server-only，manifest 记录其路径、大小和 SHA256，未将其数字混入正式 aggregate。

## 三端版本一致性矩阵

| 位置 | 分支 | SHA | tracked 状态 | 备注 |
|---|---|---|---|---|
| 本地指定工作树 | `release/0.0.12rc1` | 本报告所在最终提交 | clean | 第一阶段基线 `7468c439`；实测实现 `f393a688` |
| 服务器容器 | `release/0.0.12rc1` | 与本报告所在最终提交一致 | clean | 已知 profiler 未跟踪文件保留 |
| GitLab origin | `release/0.0.12rc1` | 与本报告所在最终提交一致 | 不适用 | 推送后用 `ls-remote` 复验 |
| GitHub github | `release/0.0.12rc1` | 与本报告所在最终提交一致 | 不适用 | 推送后用 SSH `ls-remote` 复验 |

实测环境快照记录的代码 SHA 为 `f393a6881c66abc33466dff0dcd3d6b83b3a460d`。最终提交只追加复现脚本、结构化数据包和本报告，不改变该实现的采样语义；四方最终 SHA 在任务交付记录中给出。

## Git 数据包与离线重建

Git-tracked bundle 共 77 个文件、283207 bytes，其中 manifest 自身不纳入自己的 hash index；76 个 payload 文件全部通过 SHA256。server-only 层共省略 83 个文件、11018014 bytes，主要是 SQLite、HTML、完整 stdout/server log、offset 筛选和被拒绝尝试。manifest 为每个省略文件记录服务器相对路径、大小和 SHA256。

在只复制 Git bundle 的临时目录中执行 `summarize.py`、`validate_bundle.py` 和 `validate_strict_workload.py --compare`，重建得到 6 个 summary row、2 个 aggregate row，CSV 与提交版本一致，76 个 payload hash 全部通过，严格 workload 状态为 pass。

## 已知限制

- 本报告的性能数字只来自本 campaign 的 Git-tracked 原始 JSON/metrics，没有使用被拒绝或筛选尝试的性能数字。
- EvalScope 不导出逐请求 JSON 到精简数据包时，严格长度由 summary 平均值与 percentile 文件中全部已报告分位点共同验证；完整 SQLite/stdout 作为 server-only 大文件登记 SHA256。
- 非 continuous batching 的旧生成器路径尚未接入 `min_tokens`，不属于本阶段 workload。
- c4 r3 为确保 3/3 strict input，在全新服务实例复用了已验证 offset；prefix cache 关闭且服务已重启，metadata 明确标记为 `kv-cache-cold-after-server-restart-reused-validated-offset`，未伪装成 unique offset。
