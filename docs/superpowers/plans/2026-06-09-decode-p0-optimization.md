# Decode P0 Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make NPU Graph reusable across Decode steps and remove full Paged KV token-table rebuilds from every generated token.

**Architecture:** Cache one NPU graph per `(batch_size, 128-token sequence-length bucket)`. Graph inputs use stable tensors while actual sequence lengths, positions, and KV write indices are copied before replay. Paged KV allocation keeps page-level ownership but appends only newly created logical-to-physical token mappings.

**Tech Stack:** Python, PyTorch, torch_npu graph API, unittest.

---

### Task 1: Paged KV incremental extension

**Files:**
- Modify: `lite_llama/executor/paged_attention.py`
- Test: `tests/test_decode_p0.py`

- [x] Add a test proving `extend_req()` does not call `build_token_table()` after initial allocation.
- [x] Add a test proving extension across a page boundary allocates a new page and maps only the new token.
- [x] Replace full table rebuild with vectorized mapping of `[old_token_count:new_token_count]`.
- [x] Run `python -m unittest tests.test_decode_p0 -v`.

### Task 2: NPU Graph length buckets

**Files:**
- Modify: `lite_llama/executor/npu_graph.py`
- Modify: `lite_llama/executor/model_executor.py`
- Test: `tests/test_decode_p0.py`

- [x] Add tests for 128-token bucket selection and graph-key reuse.
- [x] Cache captured graphs by batch size and bucket.
- [x] Capture against stable cloned input tensors and a shallow-copied attention metadata object.
- [x] Copy dynamic tensors before replay and remove replay-time global synchronization.
- [x] Blacklist failed graph keys and fall back to eager execution without repeated capture attempts.
- [x] Run unit tests and `python -m py_compile` for changed modules.

### Task 3: Verification and delivery

**Files:**
- Verify all files above.

- [x] Run `git diff --check`.
- [x] Run focused unit tests.
- [x] Confirm unrelated profiler files are not staged.
- [ ] Commit and push `master` to GitLab.
