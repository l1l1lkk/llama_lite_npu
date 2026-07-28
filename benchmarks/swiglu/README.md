# SwiGLU NPU benchmark

This directory provides a branch-neutral benchmark contract for the isolated
unfused and fused implementations.

The tensor shape is `[batch, sequence_length, feature_dimension]`. SwiGLU has no
attention-head axis; the `feature_dimension` sweep at 64/128/256/512 is the
requested head-dimension-sized proxy, while 6144 and 25600 are real Qwen3
intermediate widths.

The completed custom-Triton Atlas 910B3 report and raw evidence are stored in
`benchmarks/results/20260728_swiglu_triton_fusion_npu/`. The earlier
`20260728_swiglu_fusion_npu/` directory is the historical native-CANN candidate
evaluation and is not the final Triton-vs-unfused comparison.

The follow-up Dense decode NPU Graph ablation and full `msprof` evidence are in
`benchmarks/results/20260728_swiglu_npu_graph_ablation/`.

Run the latency and correctness matrix:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
ASCEND_RT_VISIBLE_DEVICES=7 python -m benchmarks.swiglu.benchmark \
  --device npu:0 \
  --physical-device 7 \
  --output /data/swiglu-results/microbenchmark.json
```

Capture an NPU profile:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python -m benchmarks.swiglu.profile \
  --device npu:0 \
  --physical-device 7 \
  --shape 4x128x6144 \
  --worker-name swiglu \
  --output-dir /data/swiglu-results/profiler
```

Compare two latency reports:

```bash
python -m benchmarks.swiglu.compare \
  --baseline /data/swiglu-results/unfused.json \
  --fused /data/swiglu-results/fused.json \
  --output /data/swiglu-results/comparison.json
```

Validate the packed projection with real Qwen3 weights:

```bash
ASCEND_RT_VISIBLE_DEVICES=7 python -m benchmarks.swiglu.validate_model_path \
  --checkpoint /data/liuke/llama_lite_npu/my_weight/Qwen3-1.7B/Qwen3-1.7B.pth \
  --device npu:0 \
  --physical-device 7 \
  --tp-size 2 \
  --tp-rank 0 \
  --layer 0 \
  --output /data/swiglu-results/model-path-validation.json
```

Run and analyze the unfused/fused x Graph off/on serving matrix:

```bash
bash benchmarks/swiglu/run_graph_ablation.sh /data/swiglu-graph-results
python benchmarks/swiglu/analyze_graph_ablation.py /data/swiglu-graph-results
```

Capture matched Graph-off/on decode profiles:

```bash
bash benchmarks/swiglu/profile_graph_ablation.sh \
  /data/swiglu-graph-results/profiler
```

`start_e2e_server.sh` accepts `NPU_GRAPH_MODE=off` (default) or
`NPU_GRAPH_MODE=on`, so existing Graph-off benchmark commands remain unchanged.
