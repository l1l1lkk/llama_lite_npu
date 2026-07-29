# RMSNorm + RoPE NPU validation

This benchmark isolates the Qwen3 attention sequence

1. Q RMSNorm;
2. K RMSNorm;
3. Q/K rotary position embedding.

The baseline branch exports
`lite_llama.kernels.rmsnorm_rope_unfused.qk_rmsnorm_rope_forward`, which keeps
the existing three Triton launches. The fused branch exports the same public
function from `rmsnorm_rope_fused`, so model and benchmark call sites remain
identical.

## Branches

- baseline: `test/rmsnorm-rope-unfused-baseline`
- fused: `feature/rmsnorm-rope-fused-optimization`

Both branches are isolated worktrees created from source commit `e058280`.

## Microbenchmark

Run modules from the repository root so `profile.py` does not shadow Python's
standard-library `profile` module.

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=7
PYTHONPATH=. python -m benchmarks.rmsnorm_rope.benchmark \
  --device npu:0 \
  --physical-device 7 \
  --dtype float16 \
  --warmup 10 \
  --samples 30 \
  --output /data/liuke/rmsnorm_rope_validation_20260729/result-fp16.json
```

The fixed matrix covers batch `1/4/16/32`, sequence length
`1/16/128/512/2048`, and head dimension `64/128/256`, using the Qwen3 TP2
head layout `QH=16, KH=4`. FP16 is checked with `atol=rtol=0.03`. BF16 uses
`atol=0.15, rtol=0.05`, with actual max/mean absolute and relative errors
recorded for every shape.

## Profiler

```bash
PYTHONPATH=. python -m benchmarks.rmsnorm_rope.profile \
  --device npu:0 \
  --physical-device 7 \
  --shape 1x512x16x4x128 \
  --dtype float16 \
  --worker-name rmsnorm-rope-prefill \
  --output-dir /data/liuke/rmsnorm-rope-profiler-prefill
```

Use `32x1x16x4x128` for decode. Capture baseline and fused with the same
physical card and shapes, then run `benchmarks.rmsnorm_rope.analyze_profiler`
to compare kernel count, device time, and weighted AIV pipeline ratios.

## Serving

The fused branch includes `run_cold_e2e.sh`, `start_e2e_server.sh`,
`stop_e2e_server.sh`, `e2e_client.py`, and `compare_e2e.py`. The cold-run
campaign restarts the server for each formal request because the current
server can leave later requests in the scheduler waiting queue. The report
must therefore label these measurements as isolated cold-request TTFT/TPOT
and preserve the scheduler limitation evidence.
