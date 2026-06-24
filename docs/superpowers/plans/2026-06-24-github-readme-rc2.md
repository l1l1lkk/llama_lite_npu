# GitHub README 0.0.10rc2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish a GitHub-oriented `0.0.10rc2` release with an English landing page, a synchronized Chinese README, visible architecture and benchmark evidence, and concise module documentation.

**Architecture:** Keep `README.md` as the English GitHub landing page and add `README_CN.md` as the complete Chinese counterpart. Both documents share the same section order and factual benchmark data. Add module-level docstrings only to core execution files that currently lack sufficient context, without changing runtime behavior.

**Tech Stack:** Markdown, Python 3.10, Git, existing unittest suite.

---

### Task 1: Add documentation contract tests

**Files:**
- Create: `tests/test_repository_docs.py`

- [ ] Assert the root README is English and links to `README_CN.md`.
- [ ] Assert both README files contain architecture, quick start, benchmark, and feature matrix sections.
- [ ] Assert both README files expose the selected EvalScope results from `docs/inference_performance_history.md`.
- [ ] Assert `VERSION`, README badges, and release links use `0.0.10rc2`.
- [ ] Assert core execution modules expose module-level docstrings.
- [ ] Run the test and confirm it fails before implementation.

### Task 2: Rewrite the GitHub landing pages

**Files:**
- Modify: `README.md`
- Create: `README_CN.md`

- [ ] Write an English-first project positioning section.
- [ ] Add an ASCII architecture diagram.
- [ ] Add a tested two-card Quick Start.
- [ ] Add a capability matrix covering models, serving, scheduling, KV management, graph execution, MoE, profiling, and current limitations.
- [ ] Add selected EvalScope tables for fixed-length concurrency 1, concurrency 4, Top-P concurrency 4, and mixed-length concurrency 4.
- [ ] Add explicit benchmark comparability notes.
- [ ] Keep the Chinese README structurally synchronized.

### Task 3: Add core module docstrings

**Files:**
- Modify: `lite_llama/executor/model_executor.py`
- Modify: `lite_llama/continuous_batching.py`
- Modify: `lite_llama/executor/npu_graph.py`
- Modify: `lite_llama/executor/paged_attention.py`
- Modify: `lite_llama/models/qwen3_moe.py`

- [ ] Document each module's responsibility, main data flow, and important correctness constraints.
- [ ] Do not change runtime behavior.

### Task 4: Publish 0.0.10rc2 metadata

**Files:**
- Modify: `VERSION`
- Modify: `CHANGELOG.md`
- Create: `docs/releases/v0.0.10rc2.md`
- Modify: `docs/README.md`

- [ ] Update version metadata to `0.0.10rc2`.
- [ ] Record the GitHub presentation and documentation changes.
- [ ] Link the release report from the documentation index.

### Task 5: Verify and publish

- [ ] Run `python -m unittest tests.test_repository_docs`.
- [ ] Run `python -m py_compile` for all modified Python modules.
- [ ] Run the existing focused scheduler, server, and decode tests.
- [ ] Scan Markdown for replacement characters and broken local links.
- [ ] Commit the release.
- [ ] Push `release/0.0.10rc2` and tag `v0.0.10rc2` to GitHub and GitLab.
