# 证据清单

| 路径 | 内容 |
|---|---|
| `REPORT.md` | 完整结论、Profiler 根因、E2E 收益与建议 |
| `TEST_REPORT.md` | 测试命令、覆盖面和限制 |
| `raw/micro/*.json` | batch 1/2/4/8 算子延迟与误差 |
| `raw/e2e/comparison.json` | 四象限统计、Graph 计数、输出 parity、msprof 汇总 |
| `raw/e2e/comparison.csv` | 四象限核心指标 |
| `raw/profiler/operator-profiler-summary.json` | Card 7 算子 kernel 统计 |
| `raw/profiler/e2e-graph-*/*.csv` | Graph off/on API 与 device op 统计 |
| `raw/operator-profiler-full.tar.gz` | 可导入 MindStudio Insight 的完整算子捕获 |
| `raw/e2e-results-full.tar.gz` | 24 个 E2E repeat、服务状态与环境证据 |
| `raw/SHA256SUMS` | 两个压缩包的 SHA256 |

服务器完整原始目录：

- `/data/liuke/rmsnorm_matmul_swiglu_20260730/micro`
- `/data/liuke/rmsnorm_matmul_swiglu_20260730/profiler`
- `/data/liuke/rmsnorm_matmul_swiglu_20260730/e2e`

完整动态 E2E msprof 体积约 953 MiB，保留在服务器，不放入 Git。
