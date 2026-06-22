# Bug Records

## 2026-06-22 - TP worker unknown continuous batching control id

### Problem description

After the chunked-prefill KV capacity fixes, TP rank 1 could still crash with:

```text
RuntimeError: unknown continuous batching control_id: 19
server.py -> _tp_continuous_worker_loop()
```

This happened when rank 0 sent a decode command for a request that rank 1 did not have in its `requests_by_id` mirror.

### Investigation process

The new error was no longer a KV allocation failure. It appeared in the TP control-plane reconstruction path before model execution. That means rank 0's scheduler believed the request had completed prefill and entered decode, while the worker rank had not seen or retained the corresponding prefill/chunked-prefill create command.

### Finding

`v0.0.8rc3` added rank-0 `prepare_prefill_chunk(...)` before worker dispatch. That fixed worker-first KV allocation failures, but it introduced/left exposed a control-plane edge case: local request state could be prepared or partially rolled forward before the worker-side mirror was guaranteed to exist.

### Analysis

In TP continuous batching there are two independent states:

- rank 0 scheduler/local model state;
- worker-rank mirrored request state keyed by `control_id`.

Decode commands are only valid if the worker already knows the same `control_id`. If rank 0 sends decode for an unmirrored id, the worker cannot safely reconstruct KV state because doing prefill locally would not match rank 0's collective execution shape.

### Resolution

`v0.0.8rc4` adds worker-known control-id tracking in `_TpCoordinatedContinuousBackend`:

- rank 0 records ids after sending `prefill` or `prefill_chunk`;
- rank 0 refuses to send `decode` for ids not known by workers;
- failed `prepare_prefill_chunk(...)` rolls back newly allocated local request ids before any worker command is sent;
- worker `release` ignores unknown ids so cleanup remains idempotent.

### Prevention

- Treat TP worker request creation as a control-plane contract.
- Do not send decode unless prefill creation was mirrored.
- Release commands should be idempotent and safe for unknown ids.

## 2026-06-22 - Chunked prefill mid-replay Paged KV allocation failure

### Problem description

When running v0.0.8rc1 server with chunked prefill enabled, TP rank 1 crashed during `prefill_chunk`:

```text
RuntimeError: Paged KV allocation failed for request 1
server.py -> _tp_continuous_worker_loop()
continuous_batching.py -> prefill_chunk()
continuous_batching.py -> _run_prefill_chunk_incremental_batch()
model_executor.py -> extend_paged_requests()
```

The server had already captured several decode graphs, for example `batch=3 bucket=896`, so the visible failure was not caused by graph capture.

### Investigation process

The stack trace showed the failure occurred while incrementally replaying prompt tokens. The chunked prefill path initialized a new request with:

```text
reserve_paged_requests((1,))
```

That reserved physical KV pages for only one logical token. Later, every replayed prompt token called `extend_paged_requests()`, which attempted to allocate more pages on demand.

### Finding

Standard prefill reserves the whole prompt KV capacity before running the forward pass, but chunked prefill only reserved one token and grew the page table during replay. Under longer prompts, concurrent requests, prefix-cache-held pages, or TP rank state pressure, this could exhaust pages halfway through prompt replay and crash the worker rank.

### Analysis

Chunked prefill must distinguish two concepts:

- logical sequence length visible to attention;
- physical KV page capacity reserved for the request.

The logical length should advance one chunk/token at a time. The physical capacity should be reserved up front for the full prompt replay plus the first generated-token slot. Otherwise the request can partially mutate KV state and fail in the middle of a TP worker operation, which is unsafe for continuous batching.

Partial prefix cache replay has the same requirement: after sharing cached full blocks, the suffix replay must ensure enough capacity for the final prompt length before running incremental replay.

### Resolution

The Paged KV allocator now supports reserving physical capacity independently from logical token count:

- `PagedReqTokensManager.reserve_req(num_tokens, reserved_tokens)`
- `PagedReqTokensManager.ensure_req_capacity(req_idx, total_tokens)`
- `ModelExecutor.reserve_paged_requests(..., reserved_lengths=...)`
- `ModelExecutor.ensure_paged_request_capacity(...)`

Chunked prefill now reserves `len(prompt_tokens) + 1` tokens of physical capacity while keeping logical token count at `1` and advancing it incrementally. Partial prefix replay also ensures full prompt capacity before suffix replay.

Follow-up in `v0.0.8rc3`: the first fix was not sufficient for TP continuous batching because rank 0 sent `prefill_chunk` commands to worker ranks before it verified the next chunk's KV capacity locally. If rank 1 encountered the allocation failure first, it exited before the rank 0 scheduler could preempt or fail the request cleanly. `v0.0.8rc3` adds `prepare_prefill_chunk(...)` on rank 0 before worker dispatch, checks capacity for each chunk up to `target_end + 1`, removes the unsafe silent `max_seq_len` clamp, and improves allocation diagnostics.

### Prevention

- Do not grow chunked-prefill KV pages token-by-token unless the scheduler has explicitly admitted the capacity.
- Keep logical length and physical reserved capacity separate in KV manager APIs.
- Cover the allocator behavior with CPU unit tests: logical length must remain short while reserved pages cover the full replay capacity.

## 2026-06-17 - TP continuous batching idle HCCL watchdog timeout

### Problem description

When running the v0.0.7rc1 server with TP continuous batching, the terminal could fail after staying idle for a while, while `/health` still returned OK. The visible error was on rank 1:

```text
RuntimeError: ACL stream synchronize failed, error code:507048
ERR02005 DIST internal error
HCCL watchdog thread terminated
fftsplus timeout
```

### Investigation process

The traceback pointed to:

```text
server.py -> _tp_continuous_worker_loop()
tp_control.py -> TensorCommandChannel.receive()
header.cpu().tolist()
```

`/health` was misleading because it only checked rank 0 FastAPI availability. Rank 1 was the worker process waiting for continuous-batching control commands.

### Finding

`header.cpu().tolist()` was not the true cause. It forced stream synchronization and exposed that rank 1 had been blocked in a long-lived NPU/HCCL broadcast receive path.

### Analysis

HCCL collectives are suitable for short, coordinated model-step communication. They are not suitable as a long-idle command queue where rank 1 waits for minutes without rank 0 sending work. On Ascend this can hit runtime watchdog timeout `507048`.

### Resolution

v0.0.7rc2 adds `StoreCommandChannel`, backed by CPU `torch.distributed.TCPStore`. Continuous-batching TP control messages now wait on CPU store keys while idle. Ranks only enter NPU/HCCL for real model compute.

### Prevention

- Do not use NPU/HCCL collectives as idle control queues.
- Keep control-plane waits on CPU when request arrival is asynchronous.
- Add server contract tests that prevent continuous batching from reverting to `TensorCommandChannel`.
