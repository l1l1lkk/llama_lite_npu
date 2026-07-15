# Changelog

## [0.0.14rc1] - 2026-07-15

Generic Qwen3 MoE runtime boundaries and reference-correctness release.

### Added

- Added the tuple-compatible `RoutingResult` contract and generic `SoftmaxTopKRouter`, while retaining `Qwen3MoeTopKRouter` as the Qwen3 compatibility type.
- Added immutable `ExpertPlacement` metadata for the existing contiguous Tensor Parallel intermediate shards and Expert Parallel expert slices.
- Added the generic `RoutedExpertExecutor`, while retaining `Qwen3MoeExperts` as a state- and signature-compatible Qwen3 type.
- Added an independent inference-only CPU FP32 MoE reference and fail-closed tracked evidence checks.

### Changed

- Separated router, placement, and routed-expert execution responsibilities without changing Qwen3 checkpoint layouts, state-dict keys, backend selection, all-reduce placement, or the eager/GMM/routed-GEMV algorithms.
- Preserved the Qwen3 sparse-block output, `last_router_logits`, tuple unpacking, and existing server/backend behavior.
- Protected compact evidence CSV/JSON with a repository-level LF contract so byte-exact manifests reproduce across Windows and Linux checkouts.

### Tests

- Ran the explicit six-test NPU MoE suite independently on physical NPU 6 and 7; each device passed 6/6 with zero failures, errors, or skips, including the generic boundary and dynamic NPUGraph capture/replay tests.
- Ran a Qwen3-30B-A3B FP16 TP2/EP2/Graph triangle: TP eager and TP auto+Graph produced identical frozen greedy output, while TP and EP differed only at the final near-tied token.
- D1 captured the layerwise router, expert, collective, decoder, and vocabulary evidence and remained `INCONCLUSIVE`; the independent four-layer FP32 D2 oracle classified both paths as `BOTH_PATHS_REFERENCE_ALIGNED`.
- Kept the CPU release validator separate from the explicit per-device NPU hardware gate and added fail-closed validation for the compact 72/8/128-row evidence tables.

### Documentation

- Added the MoE runtime design, correctness validation, compact evidence manifest, and this Chinese release report.

### Known limitations

- The generic boundary currently covers Qwen3 softmax top-k routing and contiguous TP/EP placement only. DeepSeek grouped or sigmoid routing, shared experts, all-to-all, non-contiguous expert maps, quantized MoE paths, and MLA are not implemented by this release.
- The full-model smoke covers one frozen prompt and 16 generated tokens; the independent FP32 layer oracle covers four MoE layers rather than the complete decoder.
- This release makes no new performance claim and does not compare throughput or latency with vLLM or vLLM-Ascend.

### Docs

- [v0.0.14rc1 release report](docs/releases/v0.0.14rc1.md)
- [Qwen3 MoE runtime design](docs/qwen3_moe_runtime_design.md)
- [Qwen3 MoE runtime correctness validation](docs/qwen3_moe_runtime_validation.md)

## [0.0.13rc3] - 2026-07-13

Fixed-output request control and strict benchmark release.

### Added

- Added the public OpenAI-compatible `min_tokens` request field for chat and completion requests, including boundary validation and per-request propagation through Continuous Batching.
- Added per-row EOS-logit masking before sampling until each request reaches its own `min_tokens` threshold.
- Added `min_tokens` to Tensor Parallel prefill control messages so all ranks make the same EOS-mask decision.

### Changed

- Greedy and Top-P sampling now suppress EOS independently for only the rows that have not reached their minimum output length; `min_tokens=0` preserves the previous behavior and EOS can stop normally after the threshold.

### Tests

- Added schema, streaming, scheduler, sampling, TP-control, boundary, and post-threshold EOS tests.
- Validated fixed 256-token output for Qwen3-32B, TP=2, FP16 on two Atlas 910B3 devices at concurrency 1 and 4, with three strict runs per case and zero failures.
- Added a fully paired NPU Graph on/off campaign with frozen request datasets, per-request token fingerprints, Graph counter checks, and an offline-rebuildable Git bundle.

### Benchmark

- For Qwen3-32B, TP=2, FP16, prompt 128, fixed greedy output 256, Graph on reduced mean E2E latency by 79.43% at concurrency 1 and 78.17% at concurrency 4; output throughput was 4.862x and 4.558x higher respectively. These ratios use only the six exact on/off run pairs in the v0.0.13rc3 bundle and are not a vLLM comparison.

### Known limitations

- The legacy non-Continuous-Batching generation path does not yet consume the public `min_tokens` field.
- High-concurrency TTFT and queue wait require separate root-cause analysis and are intentionally deferred to the next benchmark stage.

### Docs

- [v0.0.13rc3 release report](docs/releases/v0.0.13rc3.md)
- [fixed-output validation report](docs/benchmark_results/20260712_min_tokens_fixed_output.md)
- [strict NPU Graph ablation report](docs/benchmark_results/20260712_qwen3_32b_tp2_fp16_graph_ablation.md)

## [0.0.13rc2] - 2026-07-09

Release-validation compatibility bugfix.

### Fixed

- Marked the optional `tests/test_torch_matmul.py` visualization benchmark as skipped when `matplotlib` is not installed, so container release validation does not fail on an optional plotting dependency.

### Docs

- [v0.0.13rc2 release report](docs/releases/v0.0.13rc2.md)

## [0.0.13rc1] - 2026-07-09

Scheduler stabilization and release-validation release.

### Added

- Added `--decode_priority` / `--no_decode_priority` for Continuous Batching.
- Added a release validation helper: `scripts/validate_release.py`.
- Added scheduler tests for decode-priority admission behavior.

### Changed

- Continuous Batching now prioritizes active decode rows by default. When a decode batch is ready, the scheduler defers new prefill admission to the next tick, reducing streaming ITL/P99 jitter under mixed prefill/decode pressure.
- Existing behavior can still be restored with `--no_decode_priority` for A/B testing.

### Docs

- [v0.0.13rc1 release report](docs/releases/v0.0.13rc1.md)

## [0.0.12rc3] - 2026-07-03

Server validation bugfix release.

### Fixed

- Fixed HCCL `all_gather` failure in the Top-P candidate sampling path by making gathered tensors contiguous at the collective boundary.
- Added a regression test for non-contiguous candidate tensors.

### Docs

- [v0.0.12rc3 release report](docs/releases/v0.0.12rc3.md)

## [0.0.12rc2] - 2026-07-03

Dependency alignment release.

### Fixed

- Aligned `torch==2.7.1` with `torch_npu==2.7.1` in `requirement.txt`.
- Fixed the container dependency installation path exposed during v0.0.12 validation.

### Docs

- [v0.0.12rc2 release report](docs/releases/v0.0.12rc2.md)

## [0.0.12rc1] - 2026-07-03

Adaptive Prefill scheduler release.

### Added

- Added `--chunked_prefill_policy {adaptive,always}`.
- Added `--chunked_prefill_min_tokens`, defaulting to `2048`.
- Added scheduler tests proving short prompts stay on packed prefill while long prompts enter chunked prefill.

### Changed

- Chunked Prefill can now be enabled safely for mixed workloads: the adaptive policy only chunks long contexts and leaves short/mid prompts on the faster packed prefill path.
- The scheduler can mix packed prefill and chunked prefill work in the same tick.

### Docs

- [v0.0.12rc1 release report](docs/releases/v0.0.12rc1.md)

## [0.0.11rc1] - 2026-07-03

Decode sampling hot-path cleanup release.

### Added

- Added batched vocabulary-parallel Top-P candidate exchange.
- Added `--sampling_candidate_k` to control per-rank Top-P candidate count.
- Added Prometheus counters for candidate sampling rows, fallback rows, and full-logit gather batches.

### Changed

- Top-P sampling now exchanges candidate tensors once per batch instead of running the candidate collectives row by row.
- Full vocabulary logits are still gathered when exact nucleus completeness cannot be proven, preserving correctness.

### Tests

- Added tests for batched Top-P candidate exchange, fallback accounting, and sampling metrics.

### Docs

- [v0.0.11rc1 release report](docs/releases/v0.0.11rc1.md)

## [0.0.10rc3] - 2026-06-24

Prometheus observability release for the Continuous Batching serving path.

### Added

- Added `GET /metrics` with Prometheus text exposition.
- Added `GET /debug/stats` with scheduler, KV cache, NPU Graph, request, and
  failure snapshots.
- Added request counters and histograms for latency, queue wait, TTFT, ITL,
  prompt tokens, and generated tokens.
- Added queue-depth, active-request, Prefill, KV-page, preemption, and NPU Graph
  metrics.
- Added stable failure categories without high-cardinality exception labels.
- Added `prometheus-client==0.24.0`.

### Changed

- Continuous Batching requests now carry an endpoint type for metric labels.
- Scheduler lifecycle events update metrics without changing behavior when no
  metrics observer is configured.
- Non-streaming Continuous Batching submissions return HTTP 429 when the
  waiting queue is full.

### Tests

- Added metric lifecycle, failure attribution, scheduler integration, runtime
  snapshot, and Server endpoint contract coverage.

### Docs

- [Observability guide](docs/observability.md)
- [v0.0.10rc3 release report](docs/releases/v0.0.10rc3.md)

## [0.0.10rc2] - 2026-06-24

GitHub presentation and repository documentation release.

### Changed

- Replaced the root README with an English GitHub landing page.
- Added a synchronized Chinese README at `README_CN.md`.
- Added an ASCII architecture overview, Quick Start, benchmark tables, and a
  capability matrix.
- Published selected EvalScope measurements from
  `docs/inference_performance_history.md` in both README files.
- Added concise module-level documentation to the scheduler, model executor,
  NPU Graph, Paged KV, and Qwen3 MoE modules.

### Tests

- Added repository documentation contract tests for README structure, release
  version synchronization, benchmark values, and core module docstrings.

### Docs

- [v0.0.10rc2 release report](docs/releases/v0.0.10rc2.md)

## [0.0.10rc1] - 2026-06-23

Decode hot-path cleanup release before merging the 0.0.10 line toward the main branch.

### Changed

- Removed the default per-token TP worker decode-state host synchronization.
- Added `LLAMA_LITE_NPU_VALIDATE_TP_DECODE_STATE=1` as an explicit debug switch for the previous worker-side host validation.
- Skipped prefix-cache host-token storage on worker ranks when `return_host_tokens=False`; rank 0 behavior remains unchanged.
- Avoided a chunked-prefill incremental fallback host copy on worker ranks when host tokens are not needed.

### Notes

- Chunked Prefill remains available, but it is not the recommended path for the current short/mid prompt Qwen3-32B benchmark shape.
- The main optimized path for current testing is packed prefill plus Decode NPU Graph.

### Tests

- Added regression coverage for debug-gated TP decode-state validation.
- Added regression coverage that worker `return_host_tokens=False` paths do not perform prefix-cache host copies.

### Docs

- [v0.0.10rc1 release report](docs/releases/v0.0.10rc1.md)

## [0.0.9rc5] - 2026-06-23

Bugfix/performance release for paged chunk FlashAttention tile selection.

### Changed

- Replaced the fixed 16x32 paged chunk FA tile with runtime tile selection.
- The kernel now tries larger tiles first: 64x64, 32x64, 32x32, 16x32, 16x16.
- Successful tile choices are cached per process/shape; failed tile choices are remembered and skipped on later calls.
- Error logging now reports a short compiler summary instead of dumping the full Triton compiler trace repeatedly.

### Notes

- The later-chunk input path already projects only current chunk tokens; `max_q_len` is the chunk length, not the full context length.
- The fallback remains correctness-first and is only used when all Triton tile candidates fail.

### Docs

- [v0.0.9rc5 release report](docs/releases/v0.0.9rc5.md)

## [0.0.9rc4] - 2026-06-23

Bugfix release for paged chunk FlashAttention Triton Ascend compilation failure.

### Fixed

- Reduced the `paged_chunk_flash_attention` Triton tile from the original 64x64 shape to a conservative 16x32 shape to avoid 910B3 BiShengIR UB overflow during Chunked Prefill.
- Added a one-shot safe torch attention fallback when the paged chunk Triton kernel fails to compile or launch, so the service does not crash on unsupported compiler shapes.
- Kept the default Chunked Prefill behavior unchanged: later chunks still try `paged_chunk_flash_attention` first, then fall back only on failure.

### Tests

- Added source contract coverage for the paged chunk FA tile size and fallback path.

### Docs

- [v0.0.9rc4 release report](docs/releases/v0.0.9rc4.md)

## [0.0.9rc3] - 2026-06-23

Bugfix release for TP Chunked Prefill rank desynchronization.

### Fixed

- Fixed a TP path split in Chunked Prefill where rank 0 preflight capacity preparation allocated `model_request_id` for first-chunk requests before worker ranks saw the command.
- Rank 0 and worker ranks now keep first-chunk request state unchanged until `prefill_chunk()` executes on every rank, so both sides enter the same packed prefill / FlashAttention path.
- Existing later-chunk requests still receive rank-0 capacity checks before the command is mirrored.

### Tests

- Added regression coverage that `prepare_prefill_chunk()` does not mutate first-chunk requests.

### Docs

- [v0.0.9rc3 release report](docs/releases/v0.0.9rc3.md)

## [0.0.9rc2] - 2026-06-23

Feature release for paged chunk FlashAttention in Chunked Prefill.

### Added

- Added a Triton `paged_chunk_flash_attention` kernel for later Chunked Prefill chunks.
- Added executor metadata setup for paged chunk prefill: context lengths, chunk lengths, flat Q start locations, and current-chunk KV write indices.
- Routed Qwen3, Qwen2, and Llama text prefill attention through the paged chunk kernel when `atten_info.is_paged_chunk_prefill` is active.

### Changed

- Chunked Prefill now uses:
  - full prefill: `flash_attention2_no_pad`
  - packed/mixed prefill: `flash_attention2_no_pad`
  - first chunk: packed prefill / `flash_attention2_no_pad`
  - later chunks: Triton `paged_chunk_flash_attention` with safe incremental fallback

### Tests

- Added regression coverage that later chunked-prefill chunks use the paged chunk fast path when the executor exposes it.
- Added ModelExecutor contract coverage for paged chunk prefill metadata.

### Docs

- [v0.0.9rc2 release report](docs/releases/v0.0.9rc2.md)

## [0.0.9rc1] - 2026-06-23

Feature release for chunked-prefill attention routing.

### Changed

- Added a chunked-prefill first-chunk fast path that uses packed prefill, therefore reusing the existing `flash_attention2_no_pad` prefill attention path.
- Kept later chunked-prefill chunks on the existing incremental replay fallback because the current no-pad full-context FlashAttention kernel cannot attend to historical paged KV.
- Added one-shot startup/runtime logging that reports the effective prefill attention paths:
  - full prefill: `flash_attention2_no_pad`
  - packed prefill: `flash_attention2_no_pad`
  - chunked first chunk: `flash_attention2_no_pad`
  - chunked later chunks: paged chunk FA if available, otherwise incremental fallback

### Tests

- Added regression coverage that the first chunk of chunked prefill uses packed prefill instead of decode micro-batch replay.
- Updated later-chunk coverage to ensure non-first chunks still use the safe incremental replay fallback.

### Docs

- [v0.0.9rc1 release report](docs/releases/v0.0.9rc1.md)

## [0.0.8rc9] - 2026-06-22

Bugfix release for a `v0.0.8rc8` `ModelExecutor` initialization regression.

### Fixed

- Moved `_infer_safe_prefill_tokens()` out of the middle of `ModelExecutor.__init__`.
- Restored initialization of request manager, attention metadata, prefix cache, block prefix cache, and NPU Graph runner state.
- Added a source-level regression test to ensure the safe-prefill helper does not split constructor initialization again.

### Docs

- [v0.0.8rc9 release report](docs/releases/v0.0.8rc9.md)

## [0.0.8rc8] - 2026-06-22

Bugfix release for packed-prefill Triton grid overflow in continuous batching.

### Fixed

- Added a vLLM-style default prefill token budget: when `--max_prefill_tokens` is omitted, the scheduler reads the backend/model-derived safe packed-prefill budget.
- Derived the default safe prefill budget from the model's local Q-head count to avoid Q/K RMSNorm Triton launches exceeding the 65535 grid-row limit.
- Added a backend pre-forward guard that splits oversized packed prefill into smaller safe micro-batches.
- Added a slow but safe fallback for a single prompt that exceeds the safe packed-prefill budget.
- Exposed the effective auto `max_prefill_tokens` in server startup logs.

### Tests

- Added scheduler regression coverage for backend-derived default prefill token budget.
- Added backend regression coverage for splitting mixed-length packed prefill by token budget.

### Docs

- [v0.0.8rc8 release report](docs/releases/v0.0.8rc8.md)

## [0.0.8rc7] - 2026-06-22

Bugfix release for server-side `max_seq_len` propagation.

### Fixed

- Exposed `server.py --max_seq_len`.
- Passed `max_seq_len` from server CLI into `GenerateStreamText` and `Qwen3VLGeneratorStream`.
- Printed `Max seq len` during server startup so benchmark logs show the actual context length.
- Added server contract coverage to prevent future CLI/config drift.

### Docs

- [v0.0.8rc7 release report](docs/releases/v0.0.8rc7.md)

## [0.0.8rc6] - 2026-06-22

Bugfix release for continuous-batching context-length admission.

### Fixed

- Added scheduler admission validation for `prompt_tokens + max_tokens > max_seq_len`.
- Added a decode-time context-full guard so active requests finish with `length` before entering backend decode when KV context capacity is exhausted.
- Exposed `max_context_tokens` through the continuous-batching backend and TP coordinator.
- Converted continuous-batching context-capacity submission failures into HTTP 400 errors.

### Tests

- Added regression coverage for prompt-at-capacity rejection.
- Added regression coverage for prompt-plus-generation over-capacity rejection.
- Added regression coverage that context-full active requests are released without calling backend decode.

### Docs

- [v0.0.8rc6 release report](docs/releases/v0.0.8rc6.md)

## [0.0.8rc5] - 2026-06-22

Bugfix release for transactional TP continuous-batching control.

### Fixed

- Added a TCPStore command acknowledgement protocol for TP continuous batching so rank 0 only advances mirrored worker state after worker ranks finish the command.
- Replaced decode-only `control_id` messages with `decode_state` snapshots carrying expected logical sequence lengths.
- Added worker-side decode-state validation before entering decode collectives, turning rank drift into an explicit control-plane error instead of a late NPU/HCCL failure.
- Made TP worker shutdown acknowledge the final command before rank 0 exits the control loop.

### Tests

- Added command-channel acknowledgement protocol coverage.
- Added decode-state codec coverage.
- Added server contract coverage for transactional decode-state dispatch.

### Docs

- [v0.0.8rc5 release report](docs/releases/v0.0.8rc5.md)

## [0.0.8rc4] - 2026-06-22

Bugfix release for TP continuous-batching control-id synchronization.

### Fixed

- Added rank-0 worker-known control-id tracking for TP continuous batching.
- Prevented rank 0 from sending decode commands for requests that were not mirrored to worker ranks.
- Rolled back newly allocated local request ids when chunked-prefill preparation fails before worker dispatch.
- Made worker ranks tolerate release commands for already-unknown control ids, keeping release cleanup idempotent.

### Tests

- Added server contract tests for worker-known control-id tracking and idempotent worker release handling.

### Docs

- [v0.0.8rc4 release report](docs/releases/v0.0.8rc4.md)

## [0.0.8rc3] - 2026-06-22

Bugfix release for TP chunked-prefill capacity handling.

### Fixed

- Added a rank-0 `prepare_prefill_chunk(...)` phase before TP worker dispatch so KV capacity failures are detected by the scheduler before worker ranks enter mirrored model execution.
- Added per-chunk KV capacity checks up to `target_end + 1` before replaying any token in that chunk.
- Removed the unsafe silent clamp of chunked-prefill reserved capacity to `max_seq_len`; over-limit or exhausted-capacity cases now produce explicit capacity errors.
- Improved Paged KV allocation failure diagnostics with current token count, max sequence length, and free page count.

### Tests

- Extended continuous-batching fake executor coverage for reserved-length request allocation and prefill-chunk capacity preparation.

### Docs

- [v0.0.8rc3 release report](docs/releases/v0.0.8rc3.md)

## [0.0.8rc2] - 2026-06-22

Bugfix release for v0.0.8 chunked prefill.

### Fixed

- Fixed chunked prefill Paged KV allocation failure during incremental prompt replay.
- Separated logical sequence length from physical KV page reservation in Paged KV request allocation.
- Chunked prefill now reserves full prompt replay capacity plus the first generated-token slot while exposing only the currently replayed logical length to attention.
- Partial Prefix Cache suffix replay now ensures full prompt KV capacity before replaying uncached suffix tokens.

### Tests

- Added CPU regression coverage for reserved KV capacity without advancing logical sequence length.

### Docs

- Added bug record for the chunked prefill mid-replay allocation failure.
- Recorded v0.0.8rc1 fixed-length and mixed-length EvalScope measurements.
- [v0.0.8rc2 release report](docs/releases/v0.0.8rc2.md)

## [0.0.8rc1] - 2026-06-18

Packed-prefill release for the scheduler/KV-engine line.

### Core changes

- Added live mixed-length packed prefill for continuous batching cache misses. Requests with different prompt lengths can now share one flattened prefill forward instead of being split into equal-length groups.
- Added `ModelExecutor.activate_paged_packed_prefill_batch(...)` to build no-padding PagedAttention metadata: `b_start_loc`, `b_seq_len`, flattened `cur_select_index`, flat position ids, and per-request sample indices.
- Updated chunked prefill replay to process active prefill requests as decode micro-batches instead of looping request-by-request.
- Preserved exact Prefix Cache default behavior and kept page-aligned partial Prefix Cache opt-in via `--partial_prefix_cache`.

### Expected test-visible benefit

- Mixed prompt-length concurrency should reduce TTFT versus equal-length grouping because the scheduler can run heterogeneous prefill work in fewer model forwards.
- `--chunked_prefill` should show lower Python overhead when multiple long prompts are prefilling concurrently.
- Single-request decode speed is not expected to change materially.

### Compatibility and limitations

- This release does not add a dedicated suffix-prefill attention kernel. Chunk replay still uses decode-style KV replay for correctness.
- New Atlas performance numbers are not recorded yet; use EvalScope and profiler runs before updating the performance table.

### Tests

- Added packed prefill backend and executor contract tests.
- Added chunked prefill micro-batch replay regression coverage.

### Docs

- [v0.0.8rc1 release report](docs/releases/v0.0.8rc1.md)


所有触发版本升级的变更按发布时间倒序记录。详细规则见[版本管理与发布规范](docs/versioning.md)。

## [0.0.7rc6] - 2026-06-18

Bugfix release for the v0.0.7 KV-cache line.

### Fixed

- Changed page-aligned partial Prefix Cache reuse from default-on to explicit opt-in via `--partial_prefix_cache`.
- Reworked opt-in partial Prefix Cache to use block-level cache-map lookup over complete KV pages, avoiding full-prompt cache scans.
- Preserved exact Prefix Cache as the default greedy repeated-prompt optimization.
- Avoided EvalScope random-prompt TTFT regression caused by conservative token-by-token suffix replay on shared chat-template prefixes.

### Docs

- [v0.0.7rc6 release report](docs/releases/v0.0.7rc6.md)

## [0.0.7rc5] - 2026-06-18

KV-cache optimization closeout for the v0.0.7 line.

### Core changes

- Added live page-aligned partial prefix reuse for greedy requests. Exact prompt hits still skip prefill entirely; prefix-extension prompts now share cached pages and replay only the uncached suffix.
- Changed chunked prefill from admission-only scheduling into a real multi-tick execution path using safe incremental prompt replay.
- Added TP continuous-batching `prefill_chunk` control messages so worker ranks mirror chunked prefill state correctly.
- Kept mixed-length packed prefill as a tested metadata contract; no-padding packed prefill kernels remain future work.

### Compatibility and limitations

- Prefix reuse remains disabled for sampling requests (`temperature>0`).
- Partial suffix replay is correctness-first and token-by-token; expected benefit is TTFT reduction on repeated prefixes, not maximum raw prefill throughput.
- No Atlas performance number is recorded yet.

### Docs

- [v0.0.7rc5 release report](docs/releases/v0.0.7rc5.md)

## [0.0.7rc4] - 2026-06-17

Runtime-benefit release for the v0.0.7 scheduler/KV-engine line.

### Core changes

- Added exact-prompt live Prefix Cache for greedy requests (`temperature=0`). Repeated identical prompts can skip the full prefill forward and share cached Paged KV pages plus the first sampled token.
- Prefix Cache is intentionally disabled for sampling requests (`temperature>0`) to avoid changing stochastic generation semantics.
- Added Paged KV request sharing APIs backed by page refcounts. Shared pages are released only after all request/cache references are gone.
- Added bounded LRU ownership for cached prompt pages to avoid unbounded KV retention.
- Improved chunked-prefill scheduling behavior: long prompts accumulate chunk credit and can be deferred while shorter prompts are admitted, improving mixed long/short prompt responsiveness without unsafe suffix-prefill execution.

### Expected test-visible benefit

- Repeated exact greedy prompts should show lower TTFT because prefill forward is skipped on cache hits.
- Mixed long/short prompt concurrency should show better short-request responsiveness when `--chunked_prefill --prefill_chunk_size` and a prefill token budget are enabled.
- Random datasets with no repeated prompts should not show Prefix Cache gains.

### Limitations

- This is exact full-prompt caching, not arbitrary partial-prefix reuse yet.
- Chunked prefill is scheduler interleaving, not true suffix-prefill kernel execution.
- Prefix Cache currently targets greedy correctness; stochastic Top-P requests stay on the uncached path.

### Tests

- Added regression tests for shared Paged KV pages, exact Prefix Cache hits, sampling-cache bypass, and chunked long-prompt deferral.

### Docs

- [v0.0.7rc4 release report](docs/releases/v0.0.7rc4.md)

## [0.0.7rc3] - 2026-06-17

Complete the safe runtime pieces of the v0.0.7 scheduler/KV-engine refactor.

### Core changes

- Added scheduler-level KV-pressure preemption. When prefill/decode reports KV capacity pressure, the scheduler can release one active request, requeue it, and rebuild its context from `prompt_tokens + generated_token_ids` without duplicating streamed tokens.
- Added `--max_preemptions` server option for continuous batching; default is `1`, and `0` disables preemption.
- Added live PagedAttention page reference counts, so shared/future prefix-cache pages are not returned to the free pool until the last reference is released.
- Added request page introspection for Paged KV debugging and future prefix-cache integration.
- Added mixed-length no-padding prefill packing metadata (`MixedLengthPrefillPacker`) as the stable contract for future packed prefill kernels.
- Existing chunked-prefill planner remains the explicit chunk contract; runtime execution stays conservative until the Attention path supports suffix-prefill safely.

### Compatibility and limitations

- Default behavior remains unchanged unless KV capacity pressure occurs or `--max_preemptions` is changed.
- True live prefix-cache reuse and no-padding/chunked prefill execution still require Attention/KV writer changes and are not falsely enabled in this release.
- No Atlas performance numbers are recorded for this release.

### Tests

- Added tests for KV page refcounts, mixed-length prefill packing, preempted-request context rebuild, and scheduler KV-pressure recovery.

### Docs

- [v0.0.7rc3 release report](docs/releases/v0.0.7rc3.md)

## [0.0.7rc2] - 2026-06-17

Bugfix release for TP continuous batching idle stability on Ascend.

### Bug fix

- Replaced continuous-batching TP control-plane HCCL tensor broadcast with CPU `StoreCommandChannel` backed by `torch.distributed.TCPStore`.
- Fixed rank 1 idle-time watchdog failure: `ACL stream synchronize failed, error code:507048` / `fftsplus timeout`.
- Worker ranks now block on CPU store metadata while idle and only enter NPU/HCCL for actual model execution.
- Added regression tests to prevent server continuous batching from using `TensorCommandChannel` again.

### Docs

- [v0.0.7rc2 release report](docs/releases/v0.0.7rc2.md)
- [Bug records](docs/bug_records.md)

## [0.0.7rc1] - 2026-06-17

Scheduler and KV-engine refactor foundation release.

### Core changes

- Continuous Batching adds `max_prefill_tokens` for prefill token-budget admission per scheduler tick.
- Oversized prompts can be admitted alone to avoid long-prompt starvation.
- Continuous Batching adds `max_decode_tokens` for active decode-row budgeting per scheduler tick.
- Added CPU-side `KVBlockRefCounter` for logical KV block refcount metadata.
- Added `PrefixCache` for block-aligned longest-prefix matching metadata.
- Added `ChunkedPrefillPlanner` as the planning entry for future chunked prefill execution.
- Server CLI adds `--max_prefill_tokens`, `--max_decode_tokens`, `--chunked_prefill`, and `--prefill_chunk_size`.

### Compatibility and limitations

- Defaults remain compatible: when token budgets are omitted, scheduling still follows `max_batch_size`.
- Prefix cache is metadata-only and is not wired into live PagedAttention KV reuse yet.
- Chunked prefill is a planning/configuration entry and does not change model execution semantics yet.
- No predicted Atlas 910B3 performance numbers are recorded in this release.

### Docs

- [v0.0.7rc1 release report](docs/releases/v0.0.7rc1.md)

## [0.0.6rc2] - 2026-06-12

修复v0.0.6rc1首版Vocab Parallel Greedy按Batch逐行发起小Collective导致的性能回归。

### Bug修复

- Greedy对整个Batch一次性计算局部最大Logit和Token ID；
- 将最大值和精确float32 Token ID打包为`[batch, 2]`，每个Decode Step只执行一次
  AllGather；
- 保持全局Greedy选择与旧版完整词表Argmax语义一致；
- 移除Batch=4时每Token八次小AllGather产生的HCCL启动开销；
- Benchmark在`temperature=0`时打印`Top-p: inactive (temperature=0)`，避免把默认
  `top_p=0.9`误解为实际启用。

### 验证状态

- 采样单元测试覆盖Batch级单Collective、跨Rank全局Token选择和Top-P状态显示；
- 本地CPU回归和Python静态编译通过；
- Atlas 910B3需要复测是否消除v0.0.6rc1相对v0.0.5rc2约7.5%的Greedy回归。

### 文档

- [v0.0.6rc2完整版本报告](docs/releases/v0.0.6rc2.md)

## [0.0.6rc1] - 2026-06-12

优化Qwen3 TP与Continuous Batching的Decode热路径，减少每Token的全词表通信、
Host同步、重复反分词和Python对象广播。

### 核心能力

- Qwen3 Dense、Qwen3 MoE和Qwen3-VL在TP模式下保留本地LM Head词表分片；
- Greedy采样仅交换各Rank局部最大值和全局Token ID；
- Top-P采样先交换有界候选集，并在无法证明候选集覆盖精确nucleus时自动回退完整
  Logits Gather，保证采样语义不变；
- Continuous Batching将最新Token和Decode Position保留在NPU；
- Rank 0每个模型Step只执行一次批量Token D2H，worker Rank不再复制Token到Host；
- 流式输出使用有界后缀增量反分词，边界不稳定时自动回退完整解码；
- TP Continuous Batching控制面由`broadcast_object_list`改为固定头部和张量Payload。

### 兼容性与验证

- 不改变现有`.pth`权重、PagedAttention、NPU Graph Bucket和OpenAI API；
- 完整Logprobs API仍按需Gather全词表Logits；
- Legacy单请求与多模态请求初始化仍可使用对象广播，它们不位于逐Token热路径；
- 本地相关CPU单元测试和Python静态编译通过；
- Atlas 910B3 TP=2吞吐与输出一致性需要服务器实测，本版本不填写预测性能。

### 文档

- [v0.0.6rc1完整版本报告](docs/releases/v0.0.6rc1.md)

## [0.0.5rc3] - 2026-06-12

同步最近版本的文档、Atlas实测结果和当前功能边界；相对v0.0.5rc2不修改推理执行逻辑。

### 文档与实测

- README新增模型、并行、Continuous Batching和NPU Graph支持矩阵；
- MoE启动示例明确区分TP Graph与EP Eager；
- 补录Qwen3-30B-A3B双卡EP Eager结果：5.5 tok/s、Batch 22.1 tok/s、
  181.09ms/token；
- Profiler示例更新为EP/TP Eager通信对照采集；
- 明确EP当前使用本地专家计算加AllReduce，并非Token All-to-All；
- 明确当前只支持单机多卡，尚未实现多机TP × EP二维并行；
- README.zh与主README同步，避免继续展示上游CUDA/ROCm旧说明。

### 文档

- [v0.0.5rc3完整版本报告](docs/releases/v0.0.5rc3.md)

## [0.0.5rc2] - 2026-06-12

修复Qwen3 MoE Expert Parallel启动Decode NPU Graph时因`aclnnNonzero`导致进程退出的问题。

### Bug修复

- EP路由需要使用`torch.nonzero`压缩本地专家assignment；
- Ascend `aclnnNonzero`会同步执行stream，不能进入NPU Graph Capture；
- EP模式现在启动时直接禁用Decode Graph并明确记录Eager回退；
- MoE TP模式和Dense模型继续保留现有NPU Graph路径；
- 避免尝试失败的Capture污染stream，不能仅依赖异常捕获后继续执行。

### 验证状态

- 新增EP禁用Graph、TP保留Graph的回归测试；
- 相关CPU测试和静态编译通过；
- Atlas服务器需确认EP能够完成Warmup与正式Benchmark。

### 文档

- [v0.0.5rc2完整版本报告](docs/releases/v0.0.5rc2.md)

## [0.0.5rc1] - 2026-06-11

增加文本服务Continuous Batching、MoE Decode小Batch专家内核和单机Expert Parallel。

### 核心能力

- OpenAI兼容文本Server由单请求串行执行升级为共享Continuous Batching调度器；
- 每个请求独立管理Paged KV request ID、序列长度、输出队列和结束释放；
- TP进程使用Prefill、Decode、Release步骤级命令保持动态Batch一致；
- 新增Triton Routed-GEMV专家后端，小Decode批次跳过通用专家排序与Gather/Scatter；
- `auto`后端按`tokens * top_k`在Routed-GEMV和Ascend GMM间选择；
- 新增`--moe_parallel_mode ep`，每卡持有部分完整专家并通过HCCL AllReduce合并局部输出；
- CLI、Server和Benchmark均可选择MoE TP或EP执行模式。

### 验证状态

- 47项Continuous Batching、Paged KV、NPU Graph和Qwen3 MoE CPU测试通过；
- 5项Atlas NPU测试入口在无NPU本地环境中按预期跳过；
- Python静态编译通过；
- 本地无Atlas NPU，Triton Ascend内核、EP双卡完整模型、服务并发与Graph Replay需要在910B3验证；
- 本版本不填写预测性能，实测结果后续写入性能历史记录。

### 文档

- [v0.0.5rc1完整版本报告](docs/releases/v0.0.5rc1.md)

## [0.0.4rc1] - 2026-06-11

完成Qwen3 MoE Decode热路径Host同步清理，并开放带安全回退的NPU Graph Capture/Replay。

### 核心能力

- PagedAttention在Prefill阶段缓存CPU请求ID，Decode不再每个Token读取NPU请求Tensor；
- 流式生成复用Token解码时已有的D2H结果判断EOS，删除额外的`eos_reached.all()`同步；
- Qwen3 MoE允许按`(batch_size, 128-token bucket)`尝试NPU Graph Capture；
- Graph Capture失败的Key只尝试一次，后续稳定回退Eager；
- Benchmark输出Graph attempts、captured、replays和fallbacks计数；
- 新增动态专家路由与GMM `group_list` Graph Replay的Atlas NPU测试。

### 验证状态

- Windows CPU契约与回归测试通过；
- Atlas 910B3需运行新增NPU测试确认当前CANN/torch_npu组合支持GMM、Triton路由和HCCL Graph Replay；
- 未填写预测性能，实测后写入性能历史记录。

### 文档

- [v0.0.4rc1完整版本报告](docs/releases/v0.0.4rc1.md)

## [0.0.3rc2] - 2026-06-11

修复Qwen3 MoE Triton路由Gather在Ascend Triton 3.2编译阶段失败的问题。

### Bug修复

- 不再读取`tl.atomic_add`返回的旧值作为专家分组写入位置；
- 改为在NPU上使用`torch.argsort`生成专家顺序，再由Triton融合Gather与路由元数据写入；
- 保持路由过程无CPU同步、无`.tolist()`和无逐专家Python循环；
- 移除Ascend Triton不建议手动传入的`num_warps`参数。

### 验证状态

- 新增Ascend Triton原子返回值兼容性回归测试；
- 19项Qwen3 MoE CPU单元与契约测试通过；
- Atlas 910B3需重新执行NPU GMM测试和双卡端到端启动。

### 文档

- [v0.0.3rc2完整版本报告](docs/releases/v0.0.3rc2.md)

## [0.0.3rc1] - 2026-06-11

将Qwen3-30B-A3B MoE专家执行从动态Python循环升级为Ascend Grouped MatMul与Triton设备侧路由。

### 核心能力

- Gate/Up与Down投影分别使用`torch_npu.npu_grouped_matmul`；
- Triton在NPU侧完成专家计数、按专家Gather和routing weight加权Scatter；
- 移除GMM热路径中的`torch.unique(...).tolist()`及逐专家Python循环；
- 模型加载时将现有`.pth`专家权重一次性转换为GMM原生`[expert, input, output]`布局，无需重新转换权重；
- 保留`eager`参考后端，并支持每个Sparse MoE层在TP AllReduce前进行数值对齐。

### 验证状态

- 17项Qwen3 MoE CPU单元与契约测试通过；
- 新增2项Atlas NPU真实GMM数值测试，本地无NPU环境时明确跳过；
- Python静态编译通过；
- Atlas 910B3端到端数值与性能结果需在目标服务器完成后写入，不在本版本文档中填写预测数据。

### 文档

- [v0.0.3rc1完整版本报告](docs/releases/v0.0.3rc1.md)

## [0.0.2rc2] - 2026-06-10

修复Qwen3-30B-A3B MoE在Prefill阶段因SwiGLU错误读取非连续Gate/Up视图而产生无关回答的问题。

### Bug修复

- SwiGLU Triton内核分别接收Gate、Up和输出张量的行跨度；
- 修复融合Gate/Up经过`chunk()`后输入stride大于输出stride时的错误寻址；
- Dense MLP连续张量路径保持兼容。

### 验证状态

- Qwen3 MoE单元测试由9项增加到10项并全部通过；
- Python静态编译和`git diff --check`通过；
- Atlas 910B3端到端回答正确性等待目标服务器验证。

### 文档

- [v0.0.2rc2完整版本报告](docs/releases/v0.0.2rc2.md)

## [0.0.2rc1] - 2026-06-10

新增Qwen3-30B-A3B MoE模型的正确性优先适配。

### 核心能力

- 新增`qwen3_moe`配置、模型注册和独立双卡CLI；
- 复用现有Qwen3 Attention、RoPE、FlashAttention、Flash Decoding和Paged KV链路；
- 支持128专家、TopK=8 Router以及按命中专家执行的SwiGLU专家MLP；
- 支持专家内部Tensor Parallel，Router复制，专家中间维切分并在输出端AllReduce；
- 权重转换器支持官方Qwen3-30B-A3B权重，并严格检查每层专家完整性；
- MoE首版显式关闭Decode NPU Graph，避免动态专家路径被错误Capture。

### 验证状态

- 配置、Router、专家计算、权重堆叠、TP切分和Graph降级单元测试通过；
- Windows CPU开发环境完成静态编译检查；
- Atlas 910B3双卡权重加载、端到端生成和性能数据需要在目标服务器继续验证。

### 文档

- [v0.0.2rc1完整版本报告](docs/releases/v0.0.2rc1.md)

## [0.0.1rc1] - 2026-06-09

首个带版本号的候选版本。

### 核心能力

- 支持Qwen3-32B在2 × Atlas 910B3上的FP16张量并行推理；
- 支持OpenAI兼容接口、真实SSE Streaming及EvalScope token usage统计；
- 支持Paged KV Cache、Flash Decoding和Triton Ascend融合算子；
- NPU Graph使用官方`NPUGraph`接口，并按128-token长度Bucket进行Capture/Replay；
- Paged KV Decode改为增量更新token映射，避免每个Token重建完整映射表；
- Graph与Eager模式的确定性输出验证通过。

### 性能

- Output Throughput：`24.0606 tok/s`
- TPOT：`40.6 ms`
- ITL：`40.9 ms`
- 相对Graph修复前的`5.696 tok/s`，吞吐提升约`322.4%`；
- 对比第三方vLLM-Ascend 0.8.4rc2结果，当前输出吞吐约为`3.15×`。

### 文档

- [v0.0.1rc1完整版本报告](docs/releases/v0.0.1rc1.md)

[0.0.1rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.1rc1
[0.0.2rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.2rc1
[0.0.2rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.2rc2
[0.0.3rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.3rc1
[0.0.3rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.3rc2
[0.0.4rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.4rc1
[0.0.5rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.5rc1
[0.0.5rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.5rc2
[0.0.5rc3]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.5rc3
[0.0.6rc1]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.6rc1
[0.0.6rc2]: https://gitlab.com/l1l1lkk/llama_lite_npu/-/tags/v0.0.6rc2
