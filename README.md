<div align="center">

# Lite Llama NPU

**面向昇腾 NPU 的轻量级大模型推理框架**

基于 PyTorch、torch_npu 与 Triton Ascend，从模型结构、KV Cache、Attention、算子融合、张量并行和性能分析等环节探索大模型推理优化。

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.7-orange)
![Ascend](https://img.shields.io/badge/Ascend-910B3-red)
![Version](https://img.shields.io/badge/version-0.0.8rc6-blue)
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

Current version: **0.0.8rc6** (2026-06-22)

- [v0.0.8rc6 release report](docs/releases/v0.0.8rc6.md)
- [完整CHANGELOG](CHANGELOG.md)
- [版本管理与发布规范](docs/versioning.md)
- [推理性能历史记录](docs/inference_performance_history.md)
- [文档索引](docs/README.md)

`v0.0.8rc6` adds continuous-batching context-length admission and decode boundary guards. Requests whose prompt plus requested generation exceed `--max_seq_len` now fail clearly instead of driving worker ranks into Paged KV allocation failure. Exact Prefix Cache remains enabled by default; page-aligned partial Prefix Cache remains explicit via `--partial_prefix_cache`.

## 主要能力

### 模型与推理

- 支持 Qwen3 文本模型，重点验证 Qwen3-32B；
- 支持 Qwen3-30B-A3B MoE模型的FP16单机多卡推理；
- 支持 Qwen3-VL 多模态推理路径；
- 保留 Llama、Qwen2、LLaVA 等模型实现；
- 支持流式输出、Top-p、Temperature 和贪心采样；
- Qwen3 TP支持Vocab Parallel Sampling，Greedy路径不再AllGather完整词表Logits；
- 文本OpenAI服务支持Continuous Batching和请求级Paged KV管理；
- 支持 Qwen3 Thinking 模式开启和关闭；
- 提供 OpenAI 兼容接口：
  - `POST /v1/chat/completions`
  - `POST /v1/completions`
  - `GET /v1/models`
  - `GET /health`

### 当前支持矩阵

| 模型路径 | 交互CLI | OpenAI服务 | TP | EP | Continuous Batching | Decode NPU Graph |
|---|---:|---:|---:|---:|---:|---:|
| Qwen3 Dense | 是 | 是 | 是 | 不适用 | 是 | 是 |
| Qwen3-30B-A3B MoE | 是 | 是 | 是 | 单机实验支持 | 是 | TP可用；EP自动关闭 |
| Qwen3-VL | 是 | 是 | 是 | 不适用 | 否 | 否 |
| Llama / Qwen2 / LLaVA | 保留实现 | 部分路径 | 依模型而定 | 否 | 否 | 非当前验证重点 |

> “支持”表示项目中存在对应执行链路；当前持续回归和Atlas性能验证重点是Qwen3
> Dense与Qwen3-30B-A3B。多机并行、量化和生产级容错尚未实现。

### 昇腾推理优化

- **Tensor Parallelism**
  - Q、KV、O、Gate、Up、Down 和 LM Head 权重分片；
  - MoE Router复制与专家内部Tensor Parallel；
  - Qwen3 MoE支持完整专家按Rank切分的单机Expert Parallel；
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
  - Triton Routed-GEMV用于Decode小Batch专家计算；
  - `auto`后端按assignment规模在Routed-GEMV与GMM间选择；
  - Skip + RMSNorm 融合；
  - RoPE、RMSNorm、Softmax、KV 更新等 Triton Ascend 内核。
- **图模式**
  - Decode NPU Graph按128-token长度Bucket进行capture/replay；
  - 同一Bucket复用固定Shape Graph并更新动态KV元数据；
  - Shape 不满足 replay 条件时安全回退 Eager；
  - Qwen3 MoE TP允许Capture动态专家路由，失败Bucket只尝试一次并稳定回退Eager；
  - Qwen3 MoE EP因动态`NonZero` assignment压缩自动关闭Graph，避免Capture stream同步错误；
  - Benchmark输出Graph Capture、Replay和Fallback计数。
- **性能观测**
- **Scheduler and KV engine**
  - Continuous Batching supports request-count and token-budget admission;
  - KV block refcount, Prefix Cache metadata, and Chunked Prefill planner are available as foundation APIs;
  - Paged KV pages now carry live refcounts for safe shared-page ownership;
  - KV-pressure preemption can release/requeue active requests and rebuild context from prompt plus generated tokens;
  - Exact-prompt Prefix Cache is wired into greedy continuous batching and can skip repeated-prompt prefill;
  - Exact Prefix Cache remains enabled by default for greedy repeated prompts;
  - Page-aligned partial Prefix Cache reuse can share cached KV pages and replay only the uncached suffix when `--partial_prefix_cache` is enabled; the partial lookup uses block-level complete-page cache keys instead of full-prompt scanning;
  - Mixed-length packed prefill is wired into live Continuous Batching cache-miss execution, reducing equal-length grouping overhead;
  - Chunked Prefill can process long prompts across scheduler ticks and batches replay across active prefilling requests.
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
  │    ├─ Sharded LM Head
  │    └─ Vocab Parallel Sampling
  │
  └─ Streaming Output / OpenAI API
```

Qwen3-32B、TP=2 时，每个 Decode Token 的主要 Linear 路径为：

```text
64 × (Q + KV + O + Gate + Up + Down) + LM Head
= 385 次 MatMul
```

当前 K/V 已融合为一次投影；Q+KV、Gate+Up 融合仍是后续重点。

## 最新实测性能

`v0.0.8rc6` has not been re-benchmarked on Atlas yet. The table below keeps the latest reproducible measured baselines and is not a new-version performance claim:


1. Qwen3-30B-A3B使用项目Benchmark观察MoE TP/EP执行路径；
2. Qwen3-32B使用EvalScope保留与vLLM-Ascend的服务基线对比。

不同工具的吞吐定义不同，不能把下面两组数据直接横向比较。完整历史见
[推理性能历史记录](docs/inference_performance_history.md)。

### Qwen3-30B-A3B双卡实测

共同配置：2 × Atlas 910B3、FP16、Batch=4、Prompt约128 tokens、生成256 tokens。

| 版本与路径 | NPU Graph | Avg throughput | Batch throughput | 单Token耗时 |
|---|---:|---:|---:|---:|
| v0.0.5rc2 EP Eager | 关闭 | 5.5 tok/s | 22.1 tok/s | 181.09 ms |
| v0.0.4rc1 TP Graph | 开启 | 31.7 tok/s | 126.9 tok/s | 31.53 ms |

EP Eager相对旧MoE TP Eager基线的5.0 tok/s提升约10%，但两卡EP与TP的理论单卡专家
计算量接近。当前EP主要用于验证完整专家切分和扩展能力；由于仍采用本地专家计算后
AllReduce合并，且无法进入Decode Graph，两卡低并发下不会自然获得数量级加速。

### Qwen3-32B EvalScope基线

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

### v0.0.6rc2 Qwen3-32B sampling-path benchmark

共同配置：2 × Atlas 910B3、FP16、TP=2、Batch=4、Prompt约128 tokens、生成256
tokens、Decode NPU Graph成功Replay。

| 采样策略 | Avg throughput | Batch throughput | 单Token耗时 | 平均时间 |
|---|---:|---:|---:|---:|
| v0.0.5rc2 Greedy，temperature=0 | 21.3 tok/s | 85.0 tok/s | 47.04ms | 12.043s |
| v0.0.6rc1 Greedy，temperature=0 | 19.7 tok/s | 78.6 tok/s | 50.87ms | 13.023s |
| v0.0.6rc2 Greedy, temperature=0 | 22.1 tok/s | 88.6 tok/s | 45.16ms | 11.560s |
| v0.0.6rc1 Top-P，temperature=0.6、top_p=0.9 | 17.7 tok/s | 70.9 tok/s | 56.41ms | 14.441s |
| v0.0.6rc2 Top-P, temperature=0.6, top_p=0.9 | 17.9 tok/s | 71.7 tok/s | 55.83ms | 14.293s |

以上Graph统计均为`attempts=3`、`captured=3`、`replays=1785`、`fallbacks=0`。
v0.0.6rc2 fixed the rc1 Greedy small-collective regression: Greedy improved about 12.2% over rc1 and about 3.8% over v0.0.5rc2. Top-P improved about 1.1% because that path still needs global normalization, candidate communication, sorting, and random sampling.

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

### Qwen3-30B-A3B双卡TP + NPU Graph

```bash
export LITE_LLAMA_MOE_BACKEND=auto

ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  cli_qwen3_moe_tp.py \
  --checkpoints_dir /data/models/Qwen3-30B-A3B/ \
  --page_size 16 \
  --max_seq_len 4096 \
  --max_gen_len 1024 \
  --moe_parallel_mode tp \
  --compiled_model \
  --enable_thinking
```

### Qwen3-30B-A3B双卡EP Eager

```bash
export LITE_LLAMA_MOE_BACKEND=auto

ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  cli_qwen3_moe_tp.py \
  --checkpoints_dir /data/models/Qwen3-30B-A3B/ \
  --page_size 16 \
  --max_seq_len 4096 \
  --max_gen_len 1024 \
  --moe_parallel_mode ep \
  --no_compiled_model \
  --enable_thinking
```

默认`auto`在Decode小Batch使用Routed-GEMV，assignment数量超过阈值后尝试Ascend
Grouped MatMul。可通过`LITE_LLAMA_MOE_BACKEND=auto|eager|gmm|routed_gemv`
显式选择。排查数值问题时可使用：

```bash
export LITE_LLAMA_MOE_BACKEND=gmm
export LITE_LLAMA_MOE_VALIDATE=1
```

验证模式会在每个MoE层同时运行优化后端和eager参考计算，速度明显变慢；验证通过后应
执行`unset LITE_LLAMA_MOE_VALIDATE`。MoE TP可尝试NPU Graph；MoE EP当前固定使用
Eager，即使传入`--compiled_model`也会在启动时自动禁用Graph并打印说明。

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

Dense Qwen3：

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

MoE EP Continuous Batching：

```bash
export LITE_LLAMA_MOE_BACKEND=auto

ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  server.py \
  --checkpoints_dir /data/models/Qwen3-30B-A3B/ \
  --host 0.0.0.0 \
  --port 8213 \
  --page_size 16 \
  --moe_parallel_mode ep \
  --no_compiled_model \
  --continuous_batching \
  --max_batch_size 32
```

Continuous Batching在文本服务中默认开启，并要求`page_size > 0`。视觉模型会自动回退到
原请求级执行路径。

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


## Prefix Cache benchmark

Use this script against `server.py` to compare repeated prompts with random prompts. Direct `benchmark_tp.py` does not exercise the server-side Continuous Batching Prefix Cache.

Repeated exact prompt, greedy path, expected to show Prefix Cache TTFT benefit after the first request:

```bash
python examples/benchmark_prefix_cache.py \
  --url http://127.0.0.1:8213/v1/chat/completions \
  --model Qwen3-32B \
  --dataset same \
  --number 20 \
  --parallel 1 \
  --max-tokens 256 \
  --temperature 0
```

Random prompts, no-cache baseline under the same script:

```bash
python examples/benchmark_prefix_cache.py \
  --url http://127.0.0.1:8213/v1/chat/completions \
  --model Qwen3-32B \
  --dataset random \
  --number 20 \
  --parallel 1 \
  --prompt-len 128 \
  --max-tokens 256 \
  --temperature 0
```

For EvalScope random baseline, keep using the EvalScope command above. EvalScope is less convenient for guaranteed identical prompts, so the project script is preferred for Prefix Cache validation.

## Ascend Profiler 与 MindStudio Insight

采集MoE EP Eager算子、内存和HCCL通信数据：

```bash
export LITE_LLAMA_MOE_BACKEND=routed_gemv

ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
  --nproc_per_node=2 \
  examples/benchmark_tp.py \
  --checkpoints_dir /data/models/Qwen3-30B-A3B/ \
  --batch_size 4 \
  --prompt_len 128 \
  --max_gen_len 256 \
  --page_size 16 \
  --moe_parallel_mode ep \
  --warmup 2 \
  --iterations 3 \
  --profile \
  --profile_dir ./profiler_output/ep_eager \
  --profile_wait 0 \
  --profile_warmup 1 \
  --profile_active 1 \
  --profile_level Level1 \
  --profile_aic_metrics PipeUtilization \
  --no_profile_memory \
  --no_profile_data_simplification
```

通信分析应再采集一份仅将`--moe_parallel_mode ep`改为`tp`的Eager对照。Profiler自身
会显著降低吞吐，采集结果只用于分析Timeline、算子和通信，不能作为正式性能成绩。

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

- Legacy benchmark entry points may still pad mixed-length prompts to the batch maximum; the OpenAI server Continuous Batching path now has live mixed-length packed prefill.
- PagedAttention 已接入主路径，但固定 batch、低并发下不一定带来收益；
- NPU Graph 仍是实验实现，动态 shape 可能导致 replay 回退；
- Continuous Batching currently targets text models. Chunked Prefill, preemption, and Prefix Cache are experimental runtime paths, not a production scheduler yet.
- TP 通信为同步 AllReduce/AllGather，尚未实现计算通信重叠；
- 当前只支持单机多卡；设备映射、进程组和权重加载尚未完成多机适配；
- Q+KV、Gate+Up 尚未融合；
- v0.0.8rc6 Prefix Cache defaults to exact repeated greedy prompts only; page-aligned partial prefix reuse is opt-in through `--partial_prefix_cache`;
- Top-P Vocab Parallel Sampling在候选集无法覆盖精确nucleus时会回退完整Logits Gather；
- Rank 0 in Continuous Batching still performs one batched token D2H per step for HTTP streaming; a dedicated suffix-prefill attention kernel is not implemented yet.
- Qwen3 MoE TP Graph兼容性取决于CANN、torch_npu、GMM、Triton和HCCL版本；不兼容时按Bucket回退Eager；
- Qwen3 MoE Expert Parallel首版复用现有TP组，通过本地专家计算加AllReduce合并输出，并非Token All-to-All；
- Qwen3 MoE EP包含动态`NonZero` assignment压缩，因此自动禁用Decode NPU Graph；
- 两卡EP Eager当前为5.5 tok/s，主要价值是验证专家切分能力，不代表EP已具备规模扩展效率；
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
- [x] Qwen3 MoE Decode Host同步清理；
- [x] Qwen3 MoE NPU Graph能力探测与安全回退；
- [x] Mixed-length packed prefill live execution for Continuous Batching;
- [ ] Dedicated suffix-prefill attention kernel for higher Chunked Prefill throughput;
- [ ] Q+KV 融合；
- [ ] Gate+Up 融合；
- [x] Vocab Parallel Sampling，Greedy避免完整Logits AllGather，Top-P保留精确回退；
- [x] Continuous Batching设备端Token/Position状态；
- [x] 有界后缀增量反分词；
- [x] Continuous Batching TP张量控制面；
- [x] NPU Graph 固定 Shape/分桶与命中率统计；
- [x] Continuous Batching；
- [x] Decode小Batch Routed-GEMV专家内核；
- [x] Qwen3 MoE单机Expert Parallel；
- [x] Qwen3 MoE EP Graph不兼容路径自动降级；
- [ ] 通信与计算重叠；
- [ ] 固定容量EP Dispatch Buffer，移除动态`NonZero`；
- [ ] Token All-to-All Dispatch/Combine；
- [ ] TP × EP二维并行组与多机设备映射；
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
│   ├── moe_routing.py         # MoE设备侧分组Gather/Scatter
│   └── moe_routed_gemv.py     # Decode小Batch专家内核
├── models/
│   ├── qwen3.py               # Qwen3 文本模型
│   ├── qwen3_moe.py           # Qwen3 MoE模型
│   ├── qwen3vl.py             # Qwen3-VL 文本与视觉连接
│   └── qwen3vl_vision.py      # Qwen3-VL Vision Encoder
├── continuous_batching.py     # 文本动态Batch调度与模型后端
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
