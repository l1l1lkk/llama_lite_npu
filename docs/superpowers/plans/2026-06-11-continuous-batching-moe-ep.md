# v0.0.5rc1实施清单

## P7 Continuous Batching

- [x] 增加独立请求状态、输出队列和后台调度器。
- [x] Paged KV支持显式请求ID预留、动态Batch元数据和独立释放。
- [x] Server文本接口接入共享调度器。
- [x] TP进程改为`prefill/decode/release`步骤级同步。
- [x] 增加容量、等待队列和轮询参数。

## P8 Decode小Batch专家内核

- [x] 增加PyTorch数值参考实现。
- [x] 增加Triton Gate/Up+SwiGLU、Down和TopK归约内核。
- [x] `auto`后端按assignment数量在Routed-GEMV与GMM间选择。
- [x] 保留`eager/gmm`兼容与逐层数值校验。
- [x] 增加CPU数值测试和Atlas NPU测试入口。

## P9 Expert Parallel

- [x] `TPConfig`增加`moe_parallel_mode`。
- [x] 加载阶段支持按完整专家切分权重。
- [x] 每Rank记录本地专家范围并只计算本地贡献。
- [x] 复用MoE层末尾HCCL AllReduce合并局部专家输出。
- [x] CLI、Server和Benchmark增加`--moe_parallel_mode tp|ep`。
- [x] 增加权重布局和两Rank局部输出求和测试。

## 发布

- [x] 更新`VERSION`、`CHANGELOG.md`、README和版本报告。
- [x] 运行CPU测试、静态编译和`git diff --check`。
- [ ] 推送`release/0.0.5rc1`并创建`v0.0.5rc1`Tag。

## Atlas验证

本地环境没有Ascend NPU。服务器必须补充：

1. Routed-GEMV与Eager/GMM数值对齐；
2. `--moe_parallel_mode ep`双卡完整模型正确性；
3. Continuous Batching并发请求正确性；
4. NPU Graph capture/replay统计；
5. TP与EP在相同Benchmark口径下的性能和显存对比。
