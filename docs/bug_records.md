# Bug Records

## 2026-06-22 - Server max_seq_len was not exposed to model loading

### Problem description

After `v0.0.8rc6`, the server correctly rejected requests whose prompt plus generation exceeded the model context. However, the server CLI did not expose `--max_seq_len`, so users could not raise the context length from the server entrypoint.

Passing this command failed:

```text
server.py: error: unrecognized arguments: --max_seq_len 2048
```

### Investigation process

The model generator already supported `max_seq_len`:

```text
GenerateStreamText(..., max_seq_len=1024)
Qwen3VLGeneratorStream(..., max_seq_len=2048)
```

The missing link was `server.py`:

```text
server CLI -> load_generator(...) -> GenerateStreamText(...)
```

`server.py` accepted `page_size`, `compiled_model`, and batching parameters, but not `max_seq_len`. Therefore all text server runs used the `GenerateStreamText` default of 1024 regardless of benchmark needs.

### Finding

The rc6 context guard was correct, but server configuration was incomplete. The guard used `ModelExecutor.max_seq_len`, and that value was stuck at 1024 because the server never propagated a user-supplied value.

### Analysis

This was a configuration propagation bug. The execution stack was internally self-consistent, but the public server entrypoint did not expose the key parameter controlling the context limit.

### Resolution

`v0.0.8rc7` adds:

- `server.py --max_seq_len`;
- propagation into `load_generator(..., max_seq_len=...)`;
- propagation into `GenerateStreamText` and `Qwen3VLGeneratorStream`;
- startup logging of the effective max sequence length;
- server contract tests for this CLI/config path.

### Prevention

- Any runtime parameter used by `ModelExecutor.build(...)` must be exposed or intentionally fixed at the server entrypoint.
- Startup logs should print effective dynamic benchmark-critical parameters.
- Contract tests should assert CLI arguments are wired to model construction, not merely parsed.

## 2026-06-22 - Paged KV allocation failure at max_seq_len boundary

### Problem description

During continuous-batching chunked-prefill testing, TP rank 1 crashed in decode:

```text
RuntimeError: Paged KV allocation failed for request 3:
current_tokens=1024, max_seq_len=1024, free_pages=10331
```

### Investigation process

The error message showed that free KV pages were still available, so the failure was not global HBM/KV exhaustion. The request had reached its per-request sequence limit: `current_tokens == max_seq_len`.

The stack trace failed at:

```text
ContinuousBatchModelBackend.decode()
  -> ModelExecutor.extend_paged_requests()
```

That happens after a decode step when the backend tries to reserve KV space for the next token.

### Finding

The scheduler allowed impossible requests into the model path:

```text
prompt_tokens + requested_generation_tokens > max_seq_len
```

With `max_seq_len=1024`, a prompt close to 1024 tokens leaves no room for 128 generated tokens. The worker rank eventually hit the per-request context boundary and crashed.

### Analysis

This was not caused by NPU Graph capture, TP command acknowledgement, or `control_id` drift. The failing layer was admission control and decode-boundary handling.

Production inference engines such as vLLM treat `max_model_len` as a hard request contract: the prompt plus requested output must fit the model context. If it does not fit, the request should be rejected before scheduling rather than allowed to fail inside KV allocation.

### Resolution

`v0.0.8rc6` adds context-length guards:

- scheduler submission rejects prompts at or beyond context capacity;
- scheduler submission rejects `prompt_tokens + max_new_tokens > max_seq_len`;
- active requests whose context is already full are finished with `length` before backend decode;
- the server returns HTTP 400 for context-capacity request errors.

### Prevention

- Benchmark commands must set `--max_seq_len >= prompt_tokens + max_tokens` after chat-template/tokenizer expansion.
- For random EvalScope prompts up to about 1024 tokens with `--max-tokens 128`, use `--max_seq_len 1536` or `2048`.
- Keep context-capacity validation in the scheduler, not only in the KV allocator.

## 2026-06-22 - TP command dispatch advanced state without worker acknowledgement

### Problem description

After `v0.0.8rc4`, TP continuous batching could still fail around chunked prefill / decode transitions. The visible symptom was usually a worker-side control-plane crash:

```text
RuntimeError: unknown continuous batching control_id: 19
```

or a later NPU/HCCL failure caused by rank 0 and worker ranks entering decode with different request state.

### Investigation process

The failure was not in attention, KV page allocation, or NPU Graph capture itself. The failing layer was the TP continuous-batching control protocol:

```text
rank0 scheduler -> StoreCommandChannel -> worker requests_by_id mirror -> backend decode/prefill
```

`v0.0.8rc4` tracked which `control_id`s rank 0 believed workers knew, but that tracking was still optimistic: rank 0 updated local mirror state after sending a command, not after the worker rank had successfully executed it.

### Finding

The protocol did not have a worker acknowledgement or a state snapshot. A command send and a command execution were treated as equivalent. They are not equivalent in a distributed scheduler.

For decode, rank 0 sent only `control_id`s. That was insufficient to prove the worker had the same logical sequence length / decode position as rank 0 before entering the next collective model step.

### Analysis

This is a control-plane consistency bug. TP ranks must enter every model forward with the same batch shape and compatible per-request positions. If rank 0 and rank 1 disagree about:

- whether a request exists;
- whether chunked prefill finished;
- the logical sequence length before decode;

then the next collective path is unsafe. The correct invariant is: rank 0 may advance mirrored worker state only after workers acknowledge the exact command that created or updated that state.

### Resolution

`v0.0.8rc5` makes TP continuous-batching dispatch transactional:

- `StoreCommandChannel.send(...)` now returns a sequence id.
- Workers call `ack(sequence)` only after successful command execution.
- Workers call `ack(sequence, ok=False, message=...)` before re-raising failures.
- Rank 0 calls `wait_ack(sequence)` before marking worker state as mirrored.
- Decode commands now use `decode_state` and carry expected logical sequence lengths.
- Workers validate decode state before entering model decode and report `TP worker decode_state mismatch` if ranks diverge.

### Prevention

- Treat scheduler commands as execution plans, not fire-and-forget notifications.
- Make worker execution observable before rank 0 mutates mirrored-state assumptions.
- Carry enough state in decode commands to validate shape/position invariants before collectives.
- Keep release idempotent and failure messages explicit.

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
