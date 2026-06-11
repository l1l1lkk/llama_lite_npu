<div align="center">

# Lite Llama NPU

**面向昇腾 NPU 的轻量级大模型推理框架**

基于 PyTorch、torch_npu 与 Triton Ascend，从模型结构、KV Cache、Attention、算子融合、张量并行和性能分析等环节探索大模型推理优化。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)
![Ascend](https://img.shields.io/badge/Ascend-910B3-red)
![Version](https://img.shields.io/badge/version-0.0.3rc2-blue)
![Status](https://img.shields.io/badge/status-active_development-yellow)

</div>

## 项目简介

Lite Llama NPU 的目标不是封装 Transformers，而是实现一条可以观察、修改和验证的昇腾大模型推理链路。目前项目重点围绕 **Qwen3 Dense与MoE模型在Atlas 910B3上的多卡推理**展开，覆盖：

- 模型权重转换与 TP 分片；
- Prefill、Decode、KV Cache 和流式生成；
- FlashAttention、Flash Decoding 与 Triton Ascend 自定义算子；
- PagedAttention、NPU Graph 实验路径；
- OpenAI 兼容服务和 EvalScope 性能测试；
- Ascend PyTorch Profiler 与 MindStudio Insight 可视化分析。

当前项目适合推理框架学习、算子分析和性能优化实验，仍处于持续开发阶段，不建议直接作为生产服务使用。

## 最新版本

当前版本：**0.0.3rc2**（2026-06-11）

- [v0.0.3rc2完整版本报告](docs/releases/v0.0.3rc2.md)
- [完整CHANGELOG](CHANGELOG.md)
- [版本管理与发布规范](docs/versioning.md)

## 主要能力

### 模型与推理

- 支持 Qwen3 文本模型，重点验证 Qwen3-32B；
- 支持 Qwen3-30B-A3B MoE模型的FP16双卡正确性路径；
- 支持 Qwen3-VL 多模态推理路径；
- 保留 Llama、Qwen2、LLaVA 等模型实现；
- 支持流式输出、Top-p、Temperature 和贪心采样；
- 支持 Qwen3 Thinking 模式开启和关闭；
- 提供 OpenAI 兼容接口：
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `GET /v1/models`
  - `GET /health`

### 昇腾推理优化

- **Tensor Parallelism**
  - Q、KV、O、Gate、Up、Down 和 LM Head 权重分片；
  - MoE Router复制与专家内部Tensor Parallel；
  - HCCL AllReduce、AllGather；
  - 支持 `torchrun` 单机多卡推理。
- **Attention**
  - Prefill 使用 FlashAttention No-Pad 内核；
  - Decode 使用 Flash Decoding；
  - GQA 与 KV Cache 索引访问。
- **KV Cache**
  - 预分配 KV Cache；
  - 请求到 Token 的索引表；
  - PagedAttention/Paged KV Cache 实验实现；
  - 可配置 `page_size`。
- **算子融合**
  - K/V Linear 融合；
  - SwiGLU 自定义算子；
  - Qwen3 MoE Gate/Up与Down使用Ascend Grouped MatMul；
  - Triton融合MoE专家分组Gather与routing weight加权Scatter；
  - Skip + RMSNorm 融合；
  - RoPE、RMSNorm、Softmax、KV 更新等 Triton Ascend 内核。
- **图模式**
  - Decode NPU Graph按128-token长度Bucket进行capture/replay；
  - 同一Bucket复用固定Shape Graph并更新动态KV元数据；
  - Shape 不满足 replay 条件时安全回退 Eager。
  - Qwen3 MoE已移除专家Python循环，但NPU Graph仍等待GMM动态分组兼容性验证。
- **性能观测**
  - Ascend PyTorch Profiler；
  - CPU、CANN、NPU 算子、HBM 和 HCCL 通信数据；
  - MindStudio Insight Timeline、算子、内存和集群分析。

## 推理架构

```text
Prompt
  │
  ├─ Tokenizer / Qwen3 Chat Template
  │
  ├─ Prefill
  │    ├─ TP Linear: Q / KV / O / MLP
  │    ├─ FlashAttention No-Pad
  │    └─ KV Cache 写入
  │
  ├─ Decode Loop
  │    ├─ TP Linear + HCCL
  │    ├─ Flash Decoding
  │    ├─ Paged KV / NPU Graph（可选）
  │    ├─ LM Head + Vocab AllGather
  │    └─ Sampling
  │
  └─ Streaming Output / OpenAI API
```

Qwen3-32B、TP=2 时，每个 Decode Token 的主要 Linear 路径为：

```text
64 × (Q + KV + O + Gate + Up + Down) + LM Head
= 385 次 MatMul
```

当前 K/V 已融合为一次投影；Q+KV、Gate+Up 融合仍是后续重点。

## 最新版本性能

以下记录当前已完成目标硬件验证的最新性能，即Qwen3-32B基线。Qwen3-30B-A3B GMM路径尚未产生910B3实测数据，验证方式见[v0.0.3rc1版本报告](docs/releases/v0.0.3rc1.md)。

| 项目 | 配置 |
|---|---|
| 硬件 | 2 × Atlas 910B3，约 64 GB HBM/卡 |
| 模型 | Qwen3-32B |
| 并行 | TP=2 |
| 当前计算路径 | FP16 |
| 并发 | 1 |
| 请求数 | 15 |
| 原始输入长度 | 随机20～45 tokens |
| 最大生成长度 | 2048 tokens |
| PagedAttention | page size 16 |
| NPU Graph | 128-token Bucket Capture/Replay |

| 指标 | 当前结果 |
|---|---:|
| 请求成功率 | 100%（15/15） |
| 平均输入长度 | 59.53 tokens（含ChatML） |
| 平均输出长度 | 494.47 tokens |
| 平均延迟 | 20.5502 s |
| TTFT | 261.2 ms |
| TPOT | 40.6 ms |
| ITL | 40.9 ms |
| Output Throughput | 24.0606 tok/s |
| Total Throughput | 26.9575 tok/s |
| Request Throughput | 0.0487 req/s |

### 与 vLLM-Ascend 0.8.4rc2 的性能对比

双方均为2 × Atlas 910B3、Qwen3-32B、TP=2、并发1。Lite Llama NPU使用本项目最新实测结果；vLLM-Ascend使用第三方公开的`0.8.4rc2`结果。

| 指标 | Lite Llama NPU Qwen3-32B基线 | vLLM-Ascend 0.8.4rc2 | 对比 |
|---|---:|---:|---:|
| Output Throughput | 24.0606 tok/s | 7.6409 tok/s | 本项目约3.15× |
| Total Throughput | 26.9575 tok/s | 7.8122 tok/s | 本项目约3.45× |
| TPOT | 40.6 ms | 131.1 ms | 本项目低约69.0% |
| TTFT | 261.2 ms | 392.7 ms | 本项目低约33.5% |
| 请求成功率 | 100% | 100% | 相同 |

> 该对比不是严格同口径Benchmark：本项目使用FP16，对方使用BF16；输入模板、平均输入/输出长度和EvalScope版本也可能不同。数据用于当前工程基座观察，详细口径见[版本报告](docs/releases/v0.0.1rc1.md)。

## 环境安装

### 前置条件

- Atlas 910B/910B3 服务器；
- 已安装并匹配的 Ascend Driver、Firmware 和 CANN Toolkit；
- Python 3.10 或兼容版本；
- PyTorch 与 torch_npu 版本必须与 CANN 对应。

先确认 NPU 和 CANN 环境：

```bash
npu-smi info
source /usr/local/Ascend/ascend-toolkit/set_env.sh
```

创建 Python 环境：

```bash
conda create -n lite_llama_npu python=3.10 -y
conda activate lite_llama_npu

git clone https://gitlab.com/l1l1lkk/llama_lite_npu.git
cd llama_lite_npu

pip install -U pip setuptools wheel
pip install -r requirement.txt
```

验证环境：

```bash
python - <<'PY'
import torch
import torch_npu

print("torch:", torch.__version__)
print("torch_npu:", torch_npu.__version__)
print("NPU available:", torch.npu.is_available())
PY
```

> 如果 `pip`、`python` 和 `torchrun` 指向不同环境，请使用 `python -m torch.distributed.run` 启动，避免子进程找不到已安装依赖。

## 权重转换

项目使用转换后的单文件 `.pth` 权重，同时保留模型目录中的 tokenizer 和 `config.json`。

```bash
python apply_weight_convert.py /path/to/Qwen3-32B
```

Qwen3-30B-A3B：

```bash
python apply_weight_convert.py \
  /path/to/Qwen3-30B-A3B \
  --model-type qwen3_moe \
  --device cpu
```

转换结果默认写入：

```text
my_weight/Qwen3-32B/
├── Qwen3-32B.pth
├── config.json
├── tokenizer.json
└── ...
```

## 快速开始

### Qwen3-32B 双卡交互推理

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  cli_qwen3_tp.py \
  --checkpoints_dir /data/models/Qwen3-32B/
```

### Qwen3-30B-A3B 双卡交互推理

```bash
export LITE_LLAMA_MOE_BACKEND=auto

ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  cli_qwen3_moe_tp.py \
  --checkpoints_dir /data/models/Qwen3-30B-A3B/ \
  --page_size 16 \
  --max_seq_len 4096 \
  --max_gen_len 1024 \
  --enable_thinking
```

默认`auto`会在NPU且`torch_npu.npu_grouped_matmul`可用时启用GMM。排查数值问题时可使用：

```bash
export LITE_LLAMA_MOE_BACKEND=gmm
export LITE_LLAMA_MOE_VALIDATE=1
```

验证模式会在每个MoE层同时运行GMM和eager参考计算，速度明显变慢；验证通过后应执行`unset LITE_LLAMA_MOE_VALIDATE`。当前MoE NPU Graph仍保持关闭。

Thinking 模式：

```bash
# 开启
--enable_thinking

# 关闭
--disable_thinking
```

### TP 性能测试

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  examples/benchmark_tp.py \
  --checkpoints_dir /data/models/Qwen3-32B/ \
  --batch_size 4 \
  --prompt_len 128 \
  --max_gen_len 256 \
  --page_size 16 \
  --compiled_model \
  --warmup 2 \
  --iterations 5 \
  --disable_thinking
```

### OpenAI 兼容服务

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  server.py \
  --checkpoints_dir /data/models/Qwen3-32B/ \
  --host 0.0.0.0 \
  --port 8213 \
  --page_size 16 \
  --compiled_model
```

调用示例：

```bash
curl http://127.0.0.1:8213/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen3-32B",
    "messages": [{"role": "user", "content": "你是谁？"}],
    "max_tokens": 128,
    "temperature": 0.6,
    "top_p": 0.9,
    "stream": true
  }'
```

## EvalScope 性能测试

```bash
evalscope perf \
  --url http://127.0.0.1:8213/v1/chat/completions \
  --api openai \
  --model Qwen3-32B \
  --number 20 \
  --parallel 1 \
  --dataset random \
  --tokenizer-path /data/models/Qwen3-32B \
  --min-prompt-length 128 \
  --max-prompt-length 128 \
  --max-tokens 256 \
  --temperature 0.6 \
  --top-p 0.9 \
  --stream
```

## Ascend Profiler 与 MindStudio Insight

采集算子、内存和 HCCL 通信数据：

```bash
ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  examples/benchmark_tp.py \
  --checkpoints_dir /data/models/Qwen3-32B/ \
  --batch_size 4 \
  --prompt_len 128 \
  --max_gen_len 32 \
  --page_size 16 \
  --compiled_model \
  --warmup 2 \
  --iterations 3 \
  --profile \
  --profile_dir ./profiler_output \
  --profile_level Level1 \
  --profile_aic_metrics PipeUtilization \
  --no_profile_data_simplification
```

将 `profiler_output` 下载到本地并导入 MindStudio Insight，可查看：

- PyTorch、CANN 和 Ascend Hardware 时间线；
- MatMul、Flash Decoding 等算子耗时；
- HBM 分配、持有和保留；
- HCCL AllReduce/AllGather；
- AI Core、Vector Core、L2 和带宽指标。

Profiler 数据通常包含：

```text
*_ascend_pt/
├── ASCEND_PROFILER_OUTPUT/
│   ├── trace_view.json
│   ├── communication.json
│   └── communication_matrix.json
└── ...
```

## 当前限制

- Prefill 的生成入口仍会把不同长度 Prompt padding 到批内最大长度；
- PagedAttention 已接入主路径，但固定 batch、低并发下不一定带来收益；
- NPU Graph 仍是实验实现，动态 shape 可能导致 replay 回退；
- 服务端尚未实现 Continuous Batching；
- TP 通信为同步 AllReduce/AllGather，尚未实现计算通信重叠；
- Q+KV、Gate+Up 尚未融合；
- LM Head 当前会 AllGather 完整词表 logits；
- Qwen3 MoE GMM已接入，但尚未完成Atlas 910B3性能基线；
- Qwen3 MoE Decode NPU Graph仍未启用；
- 暂未支持 W8A8、INT8、INT4、AWQ 和 SmoothQuant。

## 优化路线

- [x] Qwen3 TP 权重分片与 HCCL 通信；
- [x] Qwen3 Chat Template 与 Thinking 开关；
- [x] FlashAttention No-Pad；
- [x] Flash Decoding；
- [x] KV Cache 动态索引；
- [x] K/V Linear 融合；
- [x] Skip-RMSNorm 与 SwiGLU 自定义算子；
- [x] PagedAttention/Paged KV 实验路径；
- [x] NPU Graph capture/replay 安全回退；
- [x] OpenAI 兼容 API 与真实 SSE Streaming；
- [x] EvalScope 指标适配；
- [x] Ascend Profiler 与 HCCL 数据采集；
- [x] Qwen3 MoE Ascend Grouped MatMul；
- [x] Qwen3 MoE Triton路由Gather/Scatter；
- [x] Qwen3 MoE逐层eager/GMM数值对齐；
- [ ] Packed Prefill，彻底移除 Padding MatMul；
- [ ] Q+KV 融合；
- [ ] Gate+Up 融合；
- [ ] Vocab Parallel Sampling，避免完整 Logits AllGather；
- [ ] NPU Graph 固定 Shape/分桶与命中率统计；
- [ ] Continuous Batching；
- [ ] 通信与计算重叠；
- [ ] W8A8/INT8/INT4 量化；
- [ ] Qwen3.5/Qwen3.6 Hybrid Attention 模型适配。

## 项目结构

```text
lite_llama/
├── executor/
│   ├── model_executor.py      # 模型加载、TP、KV Cache、执行入口
│   ├── tp_utils.py            # TP 初始化、分片与通信
│   ├── paged_attention.py     # Paged KV Cache
│   └── npu_graph.py           # NPU Graph 实验路径
├── kernels/                   # Triton Ascend 自定义算子
│   └── moe_routing.py         # MoE设备侧分组Gather/Scatter
├── models/
│   ├── qwen3.py               # Qwen3 文本模型
│   ├── qwen3_moe.py           # Qwen3 MoE模型
│   ├── qwen3vl.py             # Qwen3-VL 文本与视觉连接
│   └── qwen3vl_vision.py      # Qwen3-VL Vision Encoder
├── generate_stream.py         # 文本流式生成
└── qwen3vl_generate_stream.py # 多模态流式生成

examples/benchmark_tp.py       # TP Benchmark 与 Profiler
server.py                      # OpenAI 兼容服务
apply_weight_convert.py        # Hugging Face 权重转换
```

## 致谢

本项目参考或使用了以下开源项目与技术：

- [PyTorch](https://pytorch.org/)
- [Ascend PyTorch](https://gitee.com/ascend/pytorch)
- [Triton Ascend](https://gitee.com/ascend/triton-ascend)
- [Transformers](https://github.com/huggingface/transformers)
- [Qwen](https://github.com/QwenLM/Qwen3)
- [LightLLM](https://github.com/ModelTC/lightllm)
- [vLLM](https://github.com/vllm-project/vllm)
- [SGLang](https://github.com/sgl-project/sglang)
- [LiteLlama](https://github.com/harleyszhang/lite_llama)

## Citation

如果该项目对你的研究或工程实践有帮助，请引用：

```bibtex
@misc{lite-llama-npu-2026,
  title        = {Lite Llama NPU},
  author       = {{Litellama AI team} and {Lite Llama NPU Contributors}},
  year         = {2026},
  howpublished = {
    \url{https://github.com/harleyszhang/lite_llama} and
    \url{https://gitlab.com/l1l1lkk/llama_lite_npu}
  },
}
```
