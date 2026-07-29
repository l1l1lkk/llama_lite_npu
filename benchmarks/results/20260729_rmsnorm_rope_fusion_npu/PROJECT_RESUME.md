# 项目履历：Ascend NPU RMSNorm + RoPE Triton 融合优化

## 项目描述

针对 Qwen3 推理 Attention 前处理链路中 Q/K RMSNorm 与 RoPE 算子碎片化、
中间结果重复访存和长序列 grid 受限问题，在 Ascend NPU 上设计并落地
RMSNorm + RoPE Triton 融合算子，完成从基线构建、NPU Profiler 定位、融合
实现、数值验证、跨 shape 压测到真实服务 TTFT/TPOT 验证的性能闭环。

## 个人工作

- 建立非融合基线与融合优化两条独立 Git 分支和 worktree，以相同 API 接入
  Qwen3 Attention，保证代码、测试和服务配置可进行严格 A/B 对比。
- 使用 Ascend Profiler 对 Prefill 和 Decode 进行同卡、同 shape 配对采集，
  定位基线 3 次 kernel launch、Scalar pipe 占用 0.768–0.808、RMSNorm 中间
  结果重复读写及 `token × head` grid 展开问题。
- 使用 Triton 实现单 kernel 融合：每 token 一个 program，以
  `[head, half_dim]` tile 同时执行 Q/K FP32 RMS reduction、weight 缩放和
  RoPE 旋转；中间值驻留寄存器，Q/K tensor 理论读写流量减少 50%。
- 将每步 kernel 数从 3 降至 1，并把安全 packed-prefill token 上限从 7500
  提升到 60000。
- 构建 FP16/BF16 自动化测试矩阵，覆盖 batch 1–32、sequence length 1–2048、
  head dimension 64/128/256；记录 P50/P90/P99、CV、绝对/相对误差并对 BF16
  抖动 case 独立复测。
- 基于 Qwen3-1.7B、TP=2、NPU 6/7 完成真实服务测试，保存逐次请求日志、NPU
  状态、Prometheus 快照、Git 版本和输出哈希；识别并隔离首次 Triton 编译
  开销及现有服务调度器后续请求停滞问题。

## 量化成果

- 算子微基准：FP16/BF16 延迟降低中位数 **72.775% / 74.659%**，11 个 shape
  均为正收益，最大加速 **9.247x / 9.599x**。
- NPU Profiler：Prefill/Decode 设备 kernel 时间降低 **94.189% / 91.560%**，
  kernel 数降低 **66.667%**。
- 端到端：128/512 token prompt 下 TPOT 降低 **12.437% / 13.419%**，输出吞吐
  提升 **12.664% / 11.962%**，端到端耗时降低 **11.227% / 10.689%**；
  TTFT 在 ±0.7% 内基本持平。
- 数值正确性：FP16/BF16 共 22 个 case 全部通过，端到端跨分支输出哈希一致；
  FP16 Q/K 最大绝对误差均为 **0.001953125**。

## 技术栈

Python、PyTorch、torch_npu、Triton、Ascend CANN、Ascend Profiler、Qwen3、
Tensor Parallel、Docker、Git。

