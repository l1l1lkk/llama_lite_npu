# Benchmark 总入口

本目录是当前 benchmark 的唯一文档入口。新测试使用 canonical v2 harness；旧脚本、旧报告和旧数据包仅用于历史复核，不能自动升级为当前版本的严格可比数据。

## 证据等级

| 等级 | 用途 | 性能发布资格 |
| --- | --- | --- |
| diagnostic | capability smoke、warmup、失败生命周期和现场诊断 | 禁止进入 aggregate、baseline 和 current history |
| strict current | 当前代码、冻结 workload、环境指纹、逐请求门禁和 bundle v2 全部通过 | 仅在 comparison scope 门禁通过时可发布 |
| historical strict old-code | 旧提交上可复算且通过当时严格门禁 | 只能用于历史趋势 |
| historical unverified | 缺少原始证据、固定输出或环境指纹 | 不可用于严格比较 |

当前没有 `v0.0.15rc2` 的 strict current 性能数字。[`capability_smoke.yaml`](../../benchmarks/configs/campaigns/capability_smoke.yaml) 只生成 diagnostic 计划，即使固定输出和 token 门禁通过也不得形成性能结论。

## Comparison scope

- `capability_only`：只验证 endpoint、stream、固定输出、token usage 和指标能力；始终 `performance_eligible=false`。
- `production_stack`：允许框架使用各自发布支持的原生依赖，但必须完整记录环境指纹并验证 checkpoint 等价。结论只能称“同硬件、同 workload 的生产栈比较”，不能称仅框架变量不同。
- `controlled_stack`：除 workload/checkpoint 等价外，还要求 CANN、PyTorch、torch_npu 等共享栈字段一致，才允许因果归因。

Phase 3A 只读现场显示：Lite 环境为 CANN 8.5 / torch 2.7.1，vLLM-Ascend 发布环境为 CANN 9.0 / torch 2.10.0；Lite 使用自定义 PTH，vLLM-Ascend 使用 HF safetensors。关键 config/tokenizer/index SHA 一致，但完整权重等价性和转换 provenance 尚未验证。因此当前只能进行 `capability_only` smoke。

## 文档

- [统一执行指南](execution.md)
- [Campaign 索引](campaigns/README.md)
- [跨版本、跨框架历史索引](history.md)
- [`benchmarks/serving/`](../../benchmarks/serving/)：runner、adapter、validator 与 bundle v2

历史 Graph 消融是项目内部路径比较，不是 vLLM-Ascend 对比。准确率与性能正确性必须分开报告。
