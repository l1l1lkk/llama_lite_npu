# RMSNorm + MatMul + SwiGLU

This benchmark covers the contiguous Qwen3 dense FFN boundary:

`Skip-RMSNorm -> packed gate/up MatMul -> SwiGLU`

The down projection is intentionally excluded because TP=2 inserts HCCL
all-reduce before the following layer's RMSNorm.

Run the operator benchmark independently in each checkout:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 \
python -m benchmarks.rmsnorm_matmul_swiglu.benchmark \
  --output /data/liuke/rmsnorm_matmul_swiglu_20260730/micro/result.json
```

Run the four-quadrant end-to-end matrix on NPU 6,7:

```bash
bash benchmarks/rmsnorm_matmul_swiglu/run_e2e_matrix.sh \
  /data/liuke/rmsnorm_matmul_swiglu_20260730/e2e
python -m benchmarks.rmsnorm_matmul_swiglu.analyze_e2e \
  /data/liuke/rmsnorm_matmul_swiglu_20260730/e2e
```
