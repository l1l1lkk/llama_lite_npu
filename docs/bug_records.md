# Bug Records

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
