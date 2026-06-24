# Prometheus Observability 0.0.10rc3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-oriented Prometheus metrics and a JSON debug snapshot for request latency, queue depth, token generation, failures, KV usage, and NPU Graph activity.

**Architecture:** A dedicated `InferenceMetrics` object owns a private Prometheus registry and low-cardinality labels. The continuous-batching scheduler emits lifecycle events without depending on HTTP details. Rank 0 exposes `/metrics` and `/debug/stats`, while runtime KV and Graph counters are synchronized at scrape time.

**Tech Stack:** Python 3.10, `prometheus-client`, FastAPI, unittest.

---

### Task 1: Define metrics and failure taxonomy

**Files:**
- Create: `lite_llama/observability.py`
- Create: `tests/test_observability.py`

- [x] Test stable failure categories.
- [x] Test request, queue, token, TTFT, ITL, and latency metrics.
- [x] Test scheduler, KV, and NPU Graph snapshots.
- [x] Verify the tests fail before implementation.

### Task 2: Instrument continuous batching

**Files:**
- Modify: `lite_llama/continuous_batching.py`
- Modify: `tests/test_continuous_batching.py`

- [x] Add an optional metrics observer to the scheduler.
- [x] Emit submit, admit, token, finish, failure, rejection, and preemption events.
- [x] Keep waiting, prefilling, and running gauges synchronized.
- [x] Preserve scheduler behavior when no observer is supplied.

### Task 3: Expose service endpoints

**Files:**
- Modify: `server.py`
- Modify: `tests/test_server_batching.py`

- [x] Inject the rank-0 metrics object into the scheduler.
- [x] Add `GET /metrics` using the Prometheus content type.
- [x] Add `GET /debug/stats` for a human-readable snapshot.
- [x] Synchronize KV page and NPU Graph state before each scrape.

### Task 4: Document and release

**Files:**
- Modify: `requirement.txt`
- Modify: `VERSION`
- Modify: `CHANGELOG.md`
- Modify: `README.md`
- Modify: `README_CN.md`
- Modify: `docs/README.md`
- Create: `docs/observability.md`
- Create: `docs/releases/v0.0.10rc3.md`

- [x] Pin `prometheus-client`.
- [x] Document metric names, meanings, labels, and Prometheus scrape config.
- [x] Record the rc3 release without claiming unmeasured performance changes.

### Task 5: Verify and publish

- [x] Run observability, scheduler, server, and Decode regression tests.
- [x] Run Python syntax compilation.
- [x] Validate Markdown encoding and links.
- [ ] Commit and push branch/tag to GitLab; allow the configured GitHub mirror to synchronize.
