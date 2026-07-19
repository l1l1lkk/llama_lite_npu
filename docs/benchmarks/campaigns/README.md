# Canonical v2 Campaign 索引

当前没有 `v0.0.15rc2` 正式性能报告。Phase 3B-3 曾执行一次 Lite capability smoke，但 EvalScope client 在请求提交前失败，server request count 为 0；该生命周期只作为 rejected diagnostic，不是性能结果，也不进入 aggregate、baseline 或 history current。

Phase 3D-1 的 r10 client 也在请求提交前失败（`request_count=0`）：CLI help 门未覆盖
`evalscope.perf.main` 的真实 import，且基础 EvalScope closure 未包含完整 `EvalScope[perf]`
依赖。后续 campaign 必须绑定 schema v2 profile；schema v1 historical profile 不得执行。

唯一当前预备配置是 [`capability_smoke.yaml`](../../../benchmarks/configs/campaigns/capability_smoke.yaml)。它包含两个框架各一个有界 diagnostic cell，固定 `stream=true`、greedy、`min_tokens=max_tokens=64`，并明确禁止发布和聚合。

该配置要求 `client_profile_required=true`。未来计划必须绑定同一个 verified client profile；profile CPU preflight 必须早于任一 server/NPU 生命周期。当前推荐的 Python 3.10 / EvalScope 1.8.0 / ModelScope 1.36.3 / Transformers 5.5.3 隔离组合尚待服务器 CPU 验证，不能据此声称 smoke 已恢复。

Phase 3A 已确认的现场差异记录在[执行指南](../execution.md)：Lite 与 vLLM-Ascend 使用不同生产依赖栈和不同 checkpoint representation。只有 capability smoke、权重等价证明与相应 comparison scope 门禁通过后，才能新增 campaign 报告。

每份未来报告必须链接 Git-tracked bundle，列出代码、环境、workload、checkpoint representation/provenance、strict 门禁、异常和结论边界。diagnostic 不能伪装为性能报告。
