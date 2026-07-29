# RMSNorm + RoPE Triton 融合验证归档

本目录是 2026-07-29 在 Ascend NPU 上完成的闭环验证材料。实现、算子测试、
NPU Profiler、端到端服务测试和可复现脚本均已纳入独立 Git 分支。

## 结论入口

- [完整测试报告](REPORT.md)
- [项目履历表述](PROJECT_RESUME.md)
- [运行环境与测试版本](environment/RUN_METADATA.md)

## 证据目录

- `microbenchmark/`：FP16/BF16 原始 JSON、CSV、复测数据和命令日志。
- `profiler/`：四组原始 Ascend Profiler 采集归档、校验和、结构化对比和解析日志。
- `e2e/`：正式端到端对比、逐次服务生命周期原始证据归档和异常诊断归档。
- `tests/`：基线分支与融合分支的单元测试日志。
- `environment/`：容器环境变量和运行元数据。

压缩包均为原始目录的无损归档，可使用 `tar -xzf <archive>` 解压。对应
`.sha256` 文件用于校验传输完整性。

## 核心结果

- 3 个 Triton kernel（Q RMSNorm、K RMSNorm、RoPE）融合为 1 个 kernel。
- FP16 11 个 shape 的算子延迟降低中位数为 **72.775%**，范围为
  **71.244%–89.186%**。
- BF16 11 个 shape 的算子延迟降低中位数为 **74.659%**，范围为
  **72.973%–89.582%**。
- Ascend Profiler 中，Prefill/Decode 的设备 kernel 时间分别降低
  **94.189% / 91.560%**，每步 kernel 数降低 **66.667%**。
- Qwen3-1.7B、TP=2、输出 32 token：128/512 token prompt 的 TPOT 分别降低
  **12.437% / 13.419%**；TTFT 分别变化 **+0.626% / -0.452%**，可视为基本持平。
- FP16 和 BF16 共 22 个正确性 case 全部通过；端到端跨分支输出哈希完全一致。

