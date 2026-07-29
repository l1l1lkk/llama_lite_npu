# 运行元数据

- 测试日期：2026-07-29
- 服务器：`218.106.157.54:10013`
- 容器：`triton-llama-lite`
- 容器项目目录：`/data/liuke/llama_lite_npu`
- 算子微基准和 Profiler：物理 NPU 7，容器内 `npu:0`
- 端到端：物理 NPU 6、7，Tensor Parallel=2
- 芯片环境标识：A2
- CANN：8.5.0
- Python：3.10.12
- PyTorch：2.7.1+cpu
- torch_npu：2.7.1
- 随机种子：20260729
- 模型：`/data/liuke/llama_lite_npu/my_weight/Qwen3-1.7B`
- 基线测试提交：`ffde6897bc41881fcc70016216a23366d3b3309e`
- 融合算子/Profiler 测试提交：`d1c2257d685312974b69cae0a80a8b66130e0b4f`
- 融合端到端测试提交：`cccbbbdebeb83e5a2b7d635b43845797643074c2`

`container-environment.txt` 是端到端采集时保存的容器环境变量快照。每份
microbenchmark JSON、Profiler capture metadata 和端到端 `result.json` 还包含
对应测试的分支、提交、设备、dtype、容差或请求参数，以上述原始记录为准。

