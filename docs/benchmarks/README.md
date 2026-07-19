# Benchmark 总入口

本目录是项目当前唯一的 benchmark 文档入口。新测试使用 canonical v2 harness；旧脚本、旧报告和旧数据包继续保留用于历史复核，但不能因为目录仍存在就视为当前版本的严格基线。

## 证据等级

| 等级 | 含义 | 可否用于当前跨框架比较 |
| --- | --- | --- |
| strict current | 当前代码、冻结 workload、环境指纹、逐请求门禁和 bundle v2 均通过 | 可以，但只能比较完整匹配的 cell |
| historical strict old-code | 旧提交上通过当时严格门禁，原始证据可复算 | 不可以替代当前版本；只作历史趋势和方法复用 |
| historical unverified | 缺原始请求、指纹、固定输出或完整环境证据 | 不可以；只能引用为历史记录 |
| diagnostic | screening、warmup、失败生命周期、profiler 或局部现场 | 不进入 aggregate、baseline 或速度比 |

当前尚无 `v0.0.15rc2` 的 strict current 性能数据。`p0_smoke.yaml` 只是 Phase 2 的计划与合成门禁配置，workload 明确标记为未做服务端 token 校准，不能形成性能结论。

## 当前入口

- [统一执行指南](execution.md)：runner、配置、证据结构、校验门和 NPU 前置条件。
- [Campaign 索引](campaigns/README.md)：canonical v2 campaign 报告入口；当前没有 rc2 正式报告。
- [跨版本/跨框架历史](history.md)：历史证据等级与旧入口映射。
- [`benchmarks/serving/`](../../benchmarks/serving/)：统一 runner、adapter、strict validator 和 bundle v2 实现。
- [`benchmarks/configs/`](../../benchmarks/configs/)：模型、框架和 campaign 声明。
- [`benchmarks/workloads/`](../../benchmarks/workloads/)：冻结 workload 与 SHA256。
- [`benchmarks/results/`](../../benchmarks/results/)：Git-tracked compact evidence；现有内容均为历史 campaign。

## 发布结论门

只有同时满足以下条件才能发布性能数字：

1. 模型/checkpoint/config/tokenizer、代码、软件栈和硬件指纹完整；
2. 两框架使用同一冻结 JSONL、seed、formal/warmup 独立 offset、请求顺序、OpenAI chat stream client contract；
3. 每请求 success、实际 input/output token、request count/order 均严格通过；
4. 固定输出 capability probe 和实际 token 校准已完成；
5. Graph 状态只能是 `pass`、`fail` 或 `unsupported`，不得把 unsupported 写成零；
6. Git bundle 可脱离服务器校验 SHA 并从逐请求 evidence 重建 aggregate；
7. semantic accuracy 与 performance correctness 分开报告。

历史 Graph 消融属于项目内部路径比较，不是 vLLM-Ascend 对比。
