<div align="center">

# Lite Llama NPU

**面向昇腾 NPU 的轻量级大模型推理框架**

基于 PyTorch、torch_npu 与 Triton Ascend，从模型结构、KV Cache、Attention、算子融合、张量并行和性能分析等环节探索大模型推理优化。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)
![Ascend](https://img.shields.io/badge/Ascend-910B3-red)
![Status](https://img.shields.io/badge/status-active_development-yellow)

</div>

## 项目简介

Lite Llama NPU 的目标不是封装 Transformers，而是实现一条可以观察、修改和验证的昇腾大模型推理链路。目前项目重点围绕 **Qwen3-32B 在 Atlas 910B3 上的多卡推理**展开，覆盖：

- 模型权重转换与 TP 分片；
- Prefill、Decode、KV Cache 和流式生成；
- FlashAttention、Flash Decoding 与 Triton Ascend 自定义算子；
- PagedAttention、NPU Graph 实验路径；
- OpenAI 兼容服务和 EvalScope 性能测试；
- Ascend PyTorch Profiler 与 MindStudio Insight 可视化分析。

当前项目适合推理框架学习、算子分析和性能优化实验，仍处于持续开发阶段，不建议直接作为生产服务使用。

## 主要能力

### 模型与推理

- 支持 Qwen3 文本模型，重点验证 Qwen3-32B；
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
  - Skip + RMSNorm 融合；
  - RoPE、RMSNorm、Softmax、KV 更新等 Triton Ascend 内核。
- **图模式**
  - Decode NPU Graph capture/replay 实验路径；
  - Shape 不满足 replay 条件时安全回退 Eager。
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

## 当前性能

### 测试环境

| 项目 | 配置 |
|---|---|
| 硬件 | 2 × Atlas 910B3，约 64 GB HBM/卡 |
| 模型 | Qwen3-32B |
| 并行 | TP=2 |
| 当前计算路径 | FP16 |
| Prompt | 约 128 tokens |
| 最大生成长度 | 256 tokens |
| PagedAttention | page size 16 |
| NPU Graph | 实验性开启 |

### 离线 TP Benchmark

测试参数：`batch_size=4`、`warmup=2`、`iterations=5`。

| 指标 | 当前结果 |
|---|---:|
| 平均生成耗时 | 约 47.5～48.5 s |
| 单序列 Decode 吞吐 | 约 5.3～5.4 tok/s |
| 每 Token 延迟 | 约 185～189 ms |
| Batch 聚合吞吐 | 约 21～22 tok/s |
| 模型与 KV Cache 显存 | 约 54.2 GB/卡 |

> `5.3 tok/s` 是一次 Decode Step 的速度；batch=4 时聚合吞吐约为 `21.3 tok/s`。两种指标不能混用。

### EvalScope / OpenAI API

测试参数：并发 1、输入 128 tokens、输出 256 tokens、真实 SSE Streaming。

| 指标 | 当前结果 |
|---|---:|
| TTFT | 约 615.9 ms |
| TPOT | 约 176.0 ms |
| ITL | 约 175.2 ms |
| Decode 吞吐 | 约 5.68 tok/s |
| Output Throughput | 约 5.63 tok/s |

Profiler 会显著增加运行开销，因此开启 `--profile` 后的延迟不作为正式性能成绩。

### 与 vLLM-Ascend 8.2.0rc2 的性能对比

以下数据均来自 **2 × Atlas 910B3、Qwen3-32B、BF16、并发 1** 的 EvalScope 测试。vLLM-Ascend 一侧的版本标识按对比测试环境记录为 `8.2.0rc2`。

| 指标 | Lite Llama NPU | vLLM-Ascend 8.2.0rc2 | 当前差距 |
|---|---:|---:|---:|
| Output Throughput | 5.63 tok/s | 7.64 tok/s | vLLM-Ascend 高约 35.7% |
| Decode Throughput | 5.68 tok/s | 约 7.63 tok/s | vLLM-Ascend 高约 34% |
| TPOT / 每输出 Token 时间 | 176.0 ms | 131.1 ms | Lite Llama NPU 高约 34.2% |
| TTFT | 615.9 ms | 392.7 ms | 测试输入长度不同，仅供参考 |
| 请求成功率 | 100% | 100% | 相同 |

测试口径存在以下差异：

| 项目 | Lite Llama NPU | vLLM-Ascend |
|---|---:|---:|
| 平均输入长度 | 128 tokens | 约 29 tokens |
| 平均输出长度 | 约 256 tokens | 约 1300 tokens |
| 输出上下文范围 | 约 384 tokens | 最高约 1700 tokens |

因此，TTFT 和单请求总延迟不能直接横向比较；TPOT 与 Output Throughput 更能反映当前 Decode 引擎差距。即使 vLLM-Ascend 测试覆盖了更长的 Decode 上下文，其输出吞吐仍高于本项目，说明当前框架在 Decode 路径上仍有约 **25%～35%** 的优化空间。

结合 MindStudio Insight 分析，当前差距主要集中在：

- Decode 阶段大量 small-M `MatMulV2`，每 Token 共 385 次 MatMul；
- Q+KV、Gate+Up 尚未融合；
- 每 Token 执行 128 次同步 HCCL AllReduce，计算与通信尚未重叠；
- NPU Graph 尚未形成稳定的固定 Shape replay；
- LM Head 会 AllGather 完整词表 Logits；
- 服务端尚未实现 Continuous Batching。

> 该对比用于记录当前工程状态。后续将使用完全一致的 Prompt、Output、采样参数和 EvalScope 版本重新测试，形成严格可复现的对照数据。

### Profiler 观察

MindStudio Insight 对当前 TP2 Decode 的分析显示：

- `MatMulV2` 约占已统计算子耗时的 **59.6%**；
- 生成 256 tokens 共触发 **98,560 次 MatMulV2**，与模型结构计算一致；
- 64 层模型每 Token 执行 128 次 AllReduce；
- 256 Token 采集中记录到 **32,768 次 HCCL AllReduce**；
- 当前主要优化方向是 small-M MatMul、TP 通信等待、Graph replay、LM Head 和采样小算子。

这些数据用于定位瓶颈，不代表所有运行环境都会得到完全相同的比例。

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
├── models/
│   ├── qwen3.py               # Qwen3 文本模型
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
