# Canonical v2 Campaign 索引

当前没有 `v0.0.15rc2` 正式性能报告，也没有已执行的 Phase 3B smoke 结果。

唯一当前预备配置是 [`capability_smoke.yaml`](../../../benchmarks/configs/campaigns/capability_smoke.yaml)。它包含两个框架各一个有界 diagnostic cell，固定 `stream=true`、greedy、`min_tokens=max_tokens=64`，并明确禁止发布和聚合。

Phase 3A 已确认的现场差异记录在[执行指南](../execution.md)：Lite 与 vLLM-Ascend 使用不同生产依赖栈和不同 checkpoint representation。只有 capability smoke、权重等价证明与相应 comparison scope 门禁通过后，才能新增 campaign 报告。

每份未来报告必须链接 Git-tracked bundle，列出代码、环境、workload、checkpoint representation/provenance、strict 门禁、异常和结论边界。diagnostic 不能伪装为性能报告。
