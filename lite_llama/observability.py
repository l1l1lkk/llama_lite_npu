"""Prometheus metrics and failure attribution for lite_llama serving.

The module deliberately keeps labels low-cardinality. Request IDs, exception
messages, model paths, and sequence lengths are never used as Prometheus labels;
they belong in structured logs instead.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any, Callable

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    GCCollector,
    Histogram,
    PlatformCollector,
    ProcessCollector,
    generate_latest,
)


FAILURE_REASONS = (
    "queue_full",
    "invalid_request",
    "context_limit",
    "kv_capacity",
    "client_cancelled",
    "model_execution",
    "hccl",
    "npu_runtime",
    "graph_failure",
    "internal",
)
ENDPOINTS = frozenset(("chat", "completion"))


def classify_failure(error: BaseException | str) -> str:
    """Map an exception to a stable, low-cardinality failure category."""

    message = str(error).lower()
    if "queue" in message and "full" in message:
        return "queue_full"
    if "cancel" in message or "disconnect" in message:
        return "client_cancelled"
    if (
        "context capacity" in message
        or "max_seq_len" in message
        or "prompt length exceeds" in message
        or "prompt plus generation" in message
    ):
        return "context_limit"
    if (
        ("kv" in message and ("capacity" in message or "allocation" in message))
        or "out of memory" in message
        or "oom" in message
    ):
        return "kv_capacity"
    if (
        "hccl" in message
        or "watchdog" in message
        or "distributed" in message
        or "communication_error" in message
    ):
        return "hccl"
    if "graph" in message and (
        "capture" in message
        or "replay" in message
        or "fallback" in message
        or "failed" in message
    ):
        return "graph_failure"
    if (
        "acl" in message
        or "npu" in message
        or "ascend" in message
        or "fftsplus" in message
    ):
        return "npu_runtime"
    if isinstance(error, ValueError):
        return "invalid_request"
    if isinstance(error, RuntimeError):
        return "model_execution"
    return "internal"


class _RequestTiming:
    def __init__(self, endpoint: str, started_at: float) -> None:
        self.endpoint = endpoint
        self.started_at = started_at
        self.admitted_at: float | None = None
        self.first_token_at: float | None = None
        self.last_token_at: float | None = None


class InferenceMetrics:
    """Own Prometheus collectors and request-lifecycle timing state."""

    LATENCY_BUCKETS = (
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        30.0,
        60.0,
        120.0,
        300.0,
    )
    TOKEN_INTERVAL_BUCKETS = (
        0.001,
        0.0025,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
    )

    def __init__(
        self,
        registry: CollectorRegistry | None = None,
        clock: Callable[[], float] = time.perf_counter,
        include_process_metrics: bool = True,
    ) -> None:
        self.registry = registry or CollectorRegistry(auto_describe=True)
        self.clock = clock
        self._lock = threading.Lock()
        self._request_timings: dict[str, _RequestTiming] = {}
        self._completed = defaultdict(int)
        self._failures = defaultdict(int)
        self._scheduler_snapshot = {
            "waiting": 0,
            "prefilling": 0,
            "running": 0,
        }
        self._kv_snapshot = {"used_pages": 0, "free_pages": 0}
        self._graph_snapshot = {
            "capture_attempts": 0,
            "captures": 0,
            "replays": 0,
            "fallbacks": 0,
        }
        self._graph_last = dict(self._graph_snapshot)

        if include_process_metrics:
            ProcessCollector(registry=self.registry)
            PlatformCollector(registry=self.registry)
            GCCollector(registry=self.registry)

        self.requests_total = Counter(
            "lite_llama_requests_total",
            "Completed inference requests.",
            ("endpoint", "status"),
            registry=self.registry,
        )
        self.request_failures_total = Counter(
            "lite_llama_request_failures_total",
            "Inference request failures by stable root-cause category.",
            ("endpoint", "reason"),
            registry=self.registry,
        )
        self.request_latency_seconds = Histogram(
            "lite_llama_request_latency_seconds",
            "Time from scheduler submission to terminal request state.",
            ("endpoint",),
            buckets=self.LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.queue_wait_seconds = Histogram(
            "lite_llama_queue_wait_seconds",
            "Time from scheduler submission to first admission.",
            ("endpoint",),
            buckets=self.LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.time_to_first_token_seconds = Histogram(
            "lite_llama_time_to_first_token_seconds",
            "Time from scheduler submission to first sampled token.",
            ("endpoint",),
            buckets=self.LATENCY_BUCKETS,
            registry=self.registry,
        )
        self.inter_token_latency_seconds = Histogram(
            "lite_llama_inter_token_latency_seconds",
            "Time between consecutive sampled tokens.",
            ("endpoint",),
            buckets=self.TOKEN_INTERVAL_BUCKETS,
            registry=self.registry,
        )
        self.prompt_tokens_total = Counter(
            "lite_llama_prompt_tokens_total",
            "Prompt tokens accepted by the scheduler.",
            ("endpoint",),
            registry=self.registry,
        )
        self.generated_tokens_total = Counter(
            "lite_llama_generated_tokens_total",
            "Tokens sampled by the inference engine.",
            ("endpoint",),
            registry=self.registry,
        )
        self.waiting_requests = Gauge(
            "lite_llama_waiting_requests",
            "Requests waiting for scheduler admission.",
            registry=self.registry,
        )
        self.prefilling_requests = Gauge(
            "lite_llama_prefilling_requests",
            "Requests currently in Chunked Prefill.",
            registry=self.registry,
        )
        self.running_requests = Gauge(
            "lite_llama_running_requests",
            "Requests currently active in Decode.",
            registry=self.registry,
        )
        self.preemptions_total = Counter(
            "lite_llama_preemptions_total",
            "Requests preempted because of KV pressure.",
            registry=self.registry,
        )
        self.kv_pages_used = Gauge(
            "lite_llama_kv_pages_used",
            "Allocated Paged KV cache pages.",
            registry=self.registry,
        )
        self.kv_pages_free = Gauge(
            "lite_llama_kv_pages_free",
            "Free Paged KV cache pages.",
            registry=self.registry,
        )
        self.graph_capture_attempts_total = Counter(
            "lite_llama_graph_capture_attempts_total",
            "NPU Graph capture attempts.",
            registry=self.registry,
        )
        self.graph_captures_total = Counter(
            "lite_llama_graph_captures_total",
            "Successful NPU Graph captures.",
            registry=self.registry,
        )
        self.graph_replays_total = Counter(
            "lite_llama_graph_replays_total",
            "Successful NPU Graph replays.",
            registry=self.registry,
        )
        self.graph_fallbacks_total = Counter(
            "lite_llama_graph_fallbacks_total",
            "NPU Graph calls that fell back to eager execution.",
            registry=self.registry,
        )
        self.sampling_candidate_rows_total = Counter(
            "lite_llama_sampling_candidate_rows_total",
            "Rows sampled through the vocabulary-parallel candidate path.",
            registry=self.registry,
        )
        self.sampling_fallback_rows_total = Counter(
            "lite_llama_sampling_fallback_rows_total",
            "Rows that fell back to full-logit gather for exact sampling.",
            registry=self.registry,
        )
        self.sampling_full_logit_gather_batches_total = Counter(
            "lite_llama_sampling_full_logit_gather_batches_total",
            "Batches that required a full vocabulary gather during sampling.",
            registry=self.registry,
        )

    @staticmethod
    def _normalize_endpoint(endpoint: Any) -> str:
        endpoint = str(endpoint or "unknown")
        return endpoint if endpoint in ENDPOINTS else "unknown"

    @classmethod
    def _endpoint(cls, request: Any) -> str:
        return cls._normalize_endpoint(
            getattr(request, "endpoint", "unknown")
        )

    def on_request_submitted(self, request: Any) -> None:
        endpoint = self._endpoint(request)
        timing = _RequestTiming(endpoint, self.clock())
        with self._lock:
            self._request_timings[str(request.request_id)] = timing
        self.prompt_tokens_total.labels(endpoint=endpoint).inc(
            len(getattr(request, "prompt_tokens", ()))
        )

    def on_request_admitted(self, request: Any) -> None:
        with self._lock:
            timing = self._request_timings.get(str(request.request_id))
            if timing is None or timing.admitted_at is not None:
                return
            timing.admitted_at = self.clock()
            elapsed = max(0.0, timing.admitted_at - timing.started_at)
        self.queue_wait_seconds.labels(endpoint=timing.endpoint).observe(elapsed)

    def on_token(self, request: Any) -> None:
        now = self.clock()
        endpoint = self._endpoint(request)
        first_token_elapsed = None
        inter_token_elapsed = None
        with self._lock:
            timing = self._request_timings.get(str(request.request_id))
            if timing is not None:
                endpoint = timing.endpoint
                if timing.first_token_at is None:
                    timing.first_token_at = now
                    first_token_elapsed = max(0.0, now - timing.started_at)
                elif timing.last_token_at is not None:
                    inter_token_elapsed = max(0.0, now - timing.last_token_at)
                timing.last_token_at = now
        self.generated_tokens_total.labels(endpoint=endpoint).inc()
        if first_token_elapsed is not None:
            self.time_to_first_token_seconds.labels(
                endpoint=endpoint
            ).observe(first_token_elapsed)
        if inter_token_elapsed is not None:
            self.inter_token_latency_seconds.labels(
                endpoint=endpoint
            ).observe(inter_token_elapsed)

    def on_request_finished(self, request: Any) -> None:
        reason = str(getattr(request, "finish_reason", "") or "")
        status = "cancelled" if reason == "cancelled" else "success"
        self._finish_request(request, status=status)

    def on_request_failed(
        self, request: Any, error: BaseException | str
    ) -> None:
        reason = classify_failure(error)
        endpoint = self._endpoint(request)
        self.request_failures_total.labels(
            endpoint=endpoint, reason=reason
        ).inc()
        with self._lock:
            self._failures[(endpoint, reason)] += 1
        self._finish_request(request, status="error")

    def on_request_rejected(
        self, endpoint: str, error: BaseException | str
    ) -> None:
        endpoint = self._normalize_endpoint(endpoint)
        reason = classify_failure(error)
        self.request_failures_total.labels(
            endpoint=endpoint, reason=reason
        ).inc()
        self.requests_total.labels(endpoint=endpoint, status="error").inc()
        with self._lock:
            self._failures[(endpoint, reason)] += 1
            self._completed[(endpoint, "error")] += 1

    def _finish_request(self, request: Any, status: str) -> None:
        request_id = str(request.request_id)
        endpoint = self._endpoint(request)
        elapsed = None
        with self._lock:
            timing = self._request_timings.pop(request_id, None)
            if timing is not None:
                endpoint = timing.endpoint
                elapsed = max(0.0, self.clock() - timing.started_at)
            self._completed[(endpoint, status)] += 1
        self.requests_total.labels(endpoint=endpoint, status=status).inc()
        if elapsed is not None:
            self.request_latency_seconds.labels(
                endpoint=endpoint
            ).observe(elapsed)

    def on_preemption(self) -> None:
        self.preemptions_total.inc()

    def on_sampling_stats(self, stats: Any) -> None:
        candidate_rows = int(getattr(stats, "candidate_rows", 0) or 0)
        fallback_rows = int(getattr(stats, "fallback_rows", 0) or 0)
        gather_batches = int(
            getattr(stats, "full_logit_gather_batches", 0) or 0
        )
        if candidate_rows > 0:
            self.sampling_candidate_rows_total.inc(candidate_rows)
        if fallback_rows > 0:
            self.sampling_fallback_rows_total.inc(fallback_rows)
        if gather_batches > 0:
            self.sampling_full_logit_gather_batches_total.inc(gather_batches)

    def update_scheduler(
        self, *, waiting: int, prefilling: int, running: int
    ) -> None:
        snapshot = {
            "waiting": int(waiting),
            "prefilling": int(prefilling),
            "running": int(running),
        }
        self.waiting_requests.set(snapshot["waiting"])
        self.prefilling_requests.set(snapshot["prefilling"])
        self.running_requests.set(snapshot["running"])
        with self._lock:
            self._scheduler_snapshot = snapshot

    def sync_runtime(self, executor: Any | None) -> None:
        if executor is None:
            return
        request_manager = getattr(executor, "req_tokens_manager", None)
        page_manager = getattr(request_manager, "page_mgr", None)
        if page_manager is not None:
            free_pages = int(getattr(page_manager, "num_free_pages", 0))
            total_pages = int(getattr(page_manager, "num_pages", 0))
            used_pages = max(0, total_pages - free_pages)
            self.kv_pages_used.set(used_pages)
            self.kv_pages_free.set(free_pages)
            with self._lock:
                self._kv_snapshot = {
                    "used_pages": used_pages,
                    "free_pages": free_pages,
                }

        graph_runner = getattr(executor, "graph_runner", None)
        if graph_runner is None:
            return
        current = {
            "capture_attempts": int(
                getattr(graph_runner, "capture_attempt_count", 0)
            ),
            "captures": int(getattr(graph_runner, "capture_count", 0)),
            "replays": int(getattr(graph_runner, "replay_count", 0)),
            "fallbacks": int(getattr(graph_runner, "fallback_count", 0)),
        }
        collectors = {
            "capture_attempts": self.graph_capture_attempts_total,
            "captures": self.graph_captures_total,
            "replays": self.graph_replays_total,
            "fallbacks": self.graph_fallbacks_total,
        }
        with self._lock:
            for name, value in current.items():
                previous = self._graph_last[name]
                delta = value - previous if value >= previous else value
                if delta > 0:
                    collectors[name].inc(delta)
                self._graph_last[name] = value
            self._graph_snapshot = current

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "scheduler": dict(self._scheduler_snapshot),
                "kv_cache": dict(self._kv_snapshot),
                "npu_graph": dict(self._graph_snapshot),
                "requests": {
                    f"{endpoint}:{status}": count
                    for (endpoint, status), count in self._completed.items()
                },
                "failures": {
                    f"{endpoint}:{reason}": count
                    for (endpoint, reason), count in self._failures.items()
                },
            }

    def render(self) -> bytes:
        """Return the registry in Prometheus text exposition format."""

        return generate_latest(self.registry)

    @property
    def content_type(self) -> str:
        return CONTENT_TYPE_LATEST
