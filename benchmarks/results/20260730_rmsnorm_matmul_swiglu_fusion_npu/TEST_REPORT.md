# 测试记录

## 分支

- 基线：`test/matmul-swiglu-rmsnorm-unfused-baseline`
- 融合：`feature/matmul-swiglu-rmsnorm-fused-optimization`

两者从同一语义接口出发。基线接口仍发射 Skip-RMSNorm、CANN MatMul、
Triton SwiGLU 三个 device kernel；融合分支只替换 decode 实现。

## 已执行测试

```bash
python -m pytest tests/kernels/test_rmsnorm_matmul_swiglu.py -q
```

融合分支结果：`4 passed`。覆盖 CPU reference、无 residual、非法 packed
width，以及 Triton kernel AST 合约。

```bash
ASCEND_RT_VISIBLE_DEVICES=7 \
python -m benchmarks.rmsnorm_matmul_swiglu.benchmark \
  --output <result.json> --warmup 10 --iterations 50
```

在两个 checkout 分别执行，batch 为 1、2、4、8；所有数值检查通过。

```bash
REPEATS=3 REQUESTS=4 MAX_TOKENS=32 \
bash benchmarks/rmsnorm_matmul_swiglu/run_e2e_matrix.sh \
  /data/liuke/rmsnorm_matmul_swiglu_20260730/e2e
```

四象限 24 个 repeat result 文件完整生成，每个 cell 12 个正式请求。
所有输出 parity 比较均为 12/12。

```bash
ASCEND_RT_VISIBLE_DEVICES=7 \
python -m benchmarks.rmsnorm_matmul_swiglu.profile ...
```

分开和融合各捕获 5 个 active steps。Card 7 / Stream 47 的
`kernel_details.csv` 已收入完整 Profiler 压缩包。

```bash
bash benchmarks/rmsnorm_matmul_swiglu/profile_graph.sh \
  /data/liuke/rmsnorm_matmul_swiglu_20260730/e2e/profiler
```

动态 msprof 附着 TP rank 进程，Graph off/on 均成功导出；Graph on 的
profiled request replay 增量为 31，fallback 为 0。

## 已知限制

- Windows 本地 Python 缺少项目依赖 `accelerate`，本地只执行
  `compileall` 和 `git diff --check`；NPU 功能测试全部在项目容器执行。
- Triton 单 kernel 只为实验目的覆盖 decode FP16、sequence length 1、
  batch 不超过 16；prefill 和非 FP16 走基线 fallback。
- BLOCK_N=256 的 UB overflow 是调优证据，不是最终配置；最终配置为
  BLOCK_M 随 batch 取 1/2/4/8，BLOCK_N=64，BLOCK_K=32。
