"""Low-overhead request and model-layer tracing for Lite Llama NPU.

The tracing path is intentionally separate from Prometheus metrics. Trace
events are high-cardinality diagnostic records, while metrics must keep a
stable and bounded label set.
"""

from __future__ import annotations

import contextvars
import json
import logging
import os
import queue
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


logger = logging.getLogger(__name__)

TRACE_LEVELS = {
    "request": 0,
    "scheduler": 1,
    "layer": 2,
}
TRACE_SCHEMA_VERSION = 1


def _normalize_json(value: Any) -> Any:
    """Convert trace metadata to JSON-safe, bounded host-side values."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {
            str(key): _normalize_json(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_normalize_json(item) for item in value]
    return str(value)


@dataclass(frozen=True)
class TraceBatchContext:
    """Correlation metadata for one prefill or decode backend call."""

    batch_id: int
    operation: str
    request_ids: tuple[str, ...]
    control_ids: tuple[int | None, ...]
    rank: int
    execution_mode: str


_CURRENT_BATCH: contextvars.ContextVar[TraceBatchContext | None] = (
    contextvars.ContextVar("lite_llama_trace_batch", default=None)
)


def current_batch_context() -> TraceBatchContext | None:
    """Return the active model batch context in the current execution flow."""

    return _CURRENT_BATCH.get()


class TraceManager:
    """Own a bounded live event ring and an optional asynchronous JSONL sink."""

    def __init__(
        self,
        *,
        enabled: bool = False,
        level: str = "request",
        max_events: int = 50_000,
        rank: int = 0,
        output_path: str | Path | None = None,
        clock_ns: Callable[[], int] = time.perf_counter_ns,
        wall_clock_ns: Callable[[], int] = time.time_ns,
    ) -> None:
        self._clock_ns = clock_ns
        self._wall_clock_ns = wall_clock_ns
        self._condition = threading.Condition()
        self._writer_lock = threading.Lock()
        self._writer_queue: queue.Queue[dict[str, Any] | None] | None = None
        self._writer_thread: threading.Thread | None = None
        self._output_path: Path | None = None
        self._events: deque[dict[str, Any]] = deque(
            maxlen=max(1, int(max_events))
        )
        self._seq = 0
        self._batch_seq = 0
        self._dropped_events = 0
        self._writer_dropped_events = 0
        self.enabled = False
        self.level = "request"
        self.rank = int(rank)
        self.session_id = self._new_session_id()
        if enabled:
            self.configure(
                enabled=True,
                level=level,
                max_events=max_events,
                rank=rank,
                output_path=output_path,
            )

    @staticmethod
    def _new_session_id() -> str:
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        return (
            f"trace-{timestamp}-{os.getpid()}-"
            f"{uuid.uuid4().hex[:8]}"
        )

    @property
    def output_path(self) -> Path | None:
        return self._output_path

    @property
    def max_events(self) -> int:
        return int(self._events.maxlen or 0)

    def configure(
        self,
        *,
        enabled: bool,
        level: str = "request",
        max_events: int = 50_000,
        rank: int = 0,
        output_path: str | Path | None = None,
        session_id: str | None = None,
    ) -> None:
        """Start a fresh trace session with the requested storage policy."""

        if level not in TRACE_LEVELS:
            raise ValueError(
                f"unsupported trace level {level!r}; "
                f"expected one of {tuple(TRACE_LEVELS)}"
            )
        if int(max_events) < 1:
            raise ValueError("max_events must be positive")

        self.close()
        with self._condition:
            self.enabled = bool(enabled)
            self.level = level
            self.rank = int(rank)
            self.session_id = session_id or self._new_session_id()
            self._events = deque(maxlen=int(max_events))
            self._seq = 0
            self._batch_seq = 0
            self._dropped_events = 0
            self._writer_dropped_events = 0

        self._output_path = Path(output_path).resolve() if output_path else None
        if self.enabled and self._output_path is not None:
            self._start_writer(self._output_path)
        if self.enabled:
            self.emit(
                "trace_started",
                event_level="request",
                trace_level=self.level,
                max_events=self.max_events,
                output_path=(
                    str(self._output_path)
                    if self._output_path is not None
                    else None
                ),
            )

    def allows(self, event_level: str) -> bool:
        if not self.enabled:
            return False
        if event_level not in TRACE_LEVELS:
            raise ValueError(f"unknown trace event level: {event_level}")
        return TRACE_LEVELS[event_level] <= TRACE_LEVELS[self.level]

    def next_batch_id(self) -> int:
        with self._condition:
            self._batch_seq += 1
            return self._batch_seq

    def emit(
        self,
        event: str,
        *,
        event_level: str = "request",
        **fields: Any,
    ) -> dict[str, Any] | None:
        """Append one event without waiting for disk or network consumers."""

        if not self.allows(event_level):
            return None
        with self._condition:
            self._seq += 1
            if len(self._events) == self.max_events:
                self._dropped_events += 1
            record = {
                "schema_version": TRACE_SCHEMA_VERSION,
                "session_id": self.session_id,
                "seq": self._seq,
                "timestamp_ns": int(self._clock_ns()),
                "wall_time_ns": int(self._wall_clock_ns()),
                "event": str(event),
                "level": event_level,
                "rank": self.rank,
            }
            record.update(
                {
                    str(key): _normalize_json(value)
                    for key, value in fields.items()
                }
            )
            self._events.append(record)
            self._condition.notify_all()

        writer_queue = self._writer_queue
        if writer_queue is not None:
            try:
                writer_queue.put_nowait(record)
            except queue.Full:
                with self._condition:
                    self._writer_dropped_events += 1
        return record

    def events_after(
        self, after_seq: int = 0, limit: int = 1_000
    ) -> list[dict[str, Any]]:
        if limit < 1:
            return []
        with self._condition:
            return self._events_after_unlocked(after_seq, limit)

    def _events_after_unlocked(
        self, after_seq: int, limit: int
    ) -> list[dict[str, Any]]:
        events = [
            event for event in self._events
            if int(event["seq"]) > int(after_seq)
        ]
        return events[: int(limit)]

    def wait_for_events(
        self,
        after_seq: int,
        timeout: float = 1.0,
        limit: int = 256,
    ) -> list[dict[str, Any]]:
        """Wait until a newer event exists, then return a bounded batch."""

        with self._condition:
            if self._seq <= int(after_seq):
                self._condition.wait(timeout=max(0.0, float(timeout)))
            return self._events_after_unlocked(after_seq, limit)

    def snapshot(
        self, *, after_seq: int = 0, limit: int = 1_000
    ) -> dict[str, Any]:
        with self._condition:
            events = self._events_after_unlocked(after_seq, limit)
            oldest_seq = (
                int(self._events[0]["seq"]) if self._events else None
            )
            newest_seq = (
                int(self._events[-1]["seq"]) if self._events else None
            )
            return {
                "enabled": self.enabled,
                "session_id": self.session_id,
                "level": self.level,
                "rank": self.rank,
                "max_events": self.max_events,
                "oldest_seq": oldest_seq,
                "newest_seq": newest_seq,
                "dropped_events": self._dropped_events,
                "writer_dropped_events": self._writer_dropped_events,
                "output_path": (
                    str(self._output_path)
                    if self._output_path is not None
                    else None
                ),
                "events": events,
            }

    def _start_writer(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        writer_queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=max(1_024, min(self.max_events, 65_536))
        )
        self._writer_queue = writer_queue

        def write_events() -> None:
            with path.open("w", encoding="utf-8", newline="\n") as handle:
                while True:
                    record = writer_queue.get()
                    if record is None:
                        break
                    handle.write(
                        json.dumps(
                            record,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    if writer_queue.empty():
                        handle.flush()
                handle.flush()

        self._writer_thread = threading.Thread(
            target=write_events,
            name=f"lite-llama-trace-writer-rank-{self.rank}",
            daemon=True,
        )
        self._writer_thread.start()

    def close(self) -> None:
        """Flush and stop the optional writer without clearing live events."""

        with self._writer_lock:
            writer_queue = self._writer_queue
            writer_thread = self._writer_thread
            self._writer_queue = None
            self._writer_thread = None
            if writer_queue is not None:
                try:
                    writer_queue.put(None, timeout=5.0)
                except queue.Full:
                    logger.warning("trace writer queue did not drain on close")
            if writer_thread is not None:
                writer_thread.join(timeout=10.0)
                if writer_thread.is_alive():
                    logger.warning("trace writer thread did not stop cleanly")


class ObserverHub:
    """Fan out the scheduler's observability callbacks to multiple observers."""

    def __init__(self, *observers: Any) -> None:
        self.observers = tuple(
            observer for observer in observers if observer is not None
        )

    def __getattr__(self, method: str) -> Callable[..., None]:
        callbacks = [
            getattr(observer, method)
            for observer in self.observers
            if hasattr(observer, method)
        ]
        if not callbacks:
            raise AttributeError(method)

        def dispatch(*args: Any, **kwargs: Any) -> None:
            for callback in callbacks:
                try:
                    callback(*args, **kwargs)
                except Exception:
                    logger.exception(
                        "observability callback failed: %s", method
                    )

        return dispatch


class TraceLifecycleObserver:
    """Translate scheduler lifecycle callbacks into redacted trace events."""

    def __init__(self, manager: TraceManager) -> None:
        self.manager = manager
        self._scheduler_snapshot: tuple[int, int, int] | None = None
        self._lock = threading.Lock()

    @staticmethod
    def _request_fields(request: Any) -> dict[str, Any]:
        generated = getattr(request, "generated_token_ids", ())
        return {
            "request_id": str(getattr(request, "request_id", "unknown")),
            "control_id": getattr(request, "control_id", None),
            "endpoint": str(getattr(request, "endpoint", "unknown")),
            "prompt_tokens": len(getattr(request, "prompt_tokens", ())),
            "generated_tokens": len(generated),
            "max_new_tokens": getattr(request, "max_new_tokens", None),
            "prefill_cursor": getattr(request, "prefill_cursor", None),
            "preemptions": getattr(request, "preemptions", 0),
        }

    def on_request_submitted(self, request: Any) -> None:
        self.manager.emit(
            "request_submitted",
            event_level="request",
            **self._request_fields(request),
        )

    def on_request_admitted(self, request: Any) -> None:
        self.manager.emit(
            "request_admitted",
            event_level="request",
            **self._request_fields(request),
        )

    def on_token(self, request: Any) -> None:
        self.manager.emit(
            "token_sampled",
            event_level="request",
            **self._request_fields(request),
        )

    def on_request_finished(self, request: Any) -> None:
        self.manager.emit(
            "request_finished",
            event_level="request",
            finish_reason=getattr(request, "finish_reason", None),
            **self._request_fields(request),
        )

    def on_request_failed(self, request: Any, error: BaseException | str) -> None:
        self.manager.emit(
            "request_failed",
            event_level="request",
            error_type=type(error).__name__,
            **self._request_fields(request),
        )

    def on_request_rejected(
        self, endpoint: str, error: BaseException | str
    ) -> None:
        self.manager.emit(
            "request_rejected",
            event_level="request",
            endpoint=str(endpoint),
            error_type=type(error).__name__,
        )

    def on_preemption(self) -> None:
        self.manager.emit("request_preempted", event_level="request")

    def on_sampling_stats(self, stats: Any) -> None:
        self.manager.emit(
            "sampling_stats",
            event_level="scheduler",
            candidate_rows=getattr(stats, "candidate_rows", 0),
            fallback_rows=getattr(stats, "fallback_rows", 0),
            full_logit_gather_batches=getattr(
                stats, "full_logit_gather_batches", 0
            ),
        )

    def update_scheduler(
        self, *, waiting: int, prefilling: int, running: int
    ) -> None:
        snapshot = (int(waiting), int(prefilling), int(running))
        with self._lock:
            if snapshot == self._scheduler_snapshot:
                return
            self._scheduler_snapshot = snapshot
        self.manager.emit(
            "scheduler_state",
            event_level="scheduler",
            waiting=snapshot[0],
            prefilling=snapshot[1],
            running=snapshot[2],
        )


def _find_executor(backend: Any) -> Any | None:
    current = backend
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        executor = getattr(current, "executor", None)
        if executor is not None:
            return executor
        current = getattr(current, "local_backend", None)
    return None


class TracingBackend:
    """Decorate a continuous-batching backend with correlated span events."""

    def __init__(
        self, backend: Any, manager: TraceManager, *, rank: int = 0
    ) -> None:
        self.backend = backend
        self.manager = manager
        self.rank = int(rank)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.backend, name)

    def _execution_mode(self, operation: str) -> str:
        executor = _find_executor(self.backend)
        graph_runner = (
            getattr(executor, "graph_runner", None)
            if executor is not None
            else None
        )
        if operation == "decode" and graph_runner is not None:
            return "npu_graph"
        return "eager"

    @staticmethod
    def _request_metadata(requests: Sequence[Any]) -> dict[str, Any]:
        return {
            "request_ids": [
                str(getattr(request, "request_id", "unknown"))
                for request in requests
            ],
            "control_ids": [
                getattr(request, "control_id", None)
                for request in requests
            ],
            "batch_size": len(requests),
            "prompt_lengths": [
                len(getattr(request, "prompt_tokens", ()))
                for request in requests
            ],
            "generated_tokens": [
                len(getattr(request, "generated_token_ids", ()))
                for request in requests
            ],
            "prefill_cursors": [
                getattr(request, "prefill_cursor", 0)
                for request in requests
            ],
        }

    def _call(
        self,
        operation: str,
        requests: Sequence[Any],
        callback: Callable[[], Any],
        **fields: Any,
    ) -> Any:
        requests = tuple(requests)
        batch_id = self.manager.next_batch_id()
        execution_mode = self._execution_mode(operation)
        metadata = self._request_metadata(requests)
        context = TraceBatchContext(
            batch_id=batch_id,
            operation=operation,
            request_ids=tuple(metadata["request_ids"]),
            control_ids=tuple(metadata["control_ids"]),
            rank=self.rank,
            execution_mode=execution_mode,
        )
        token = _CURRENT_BATCH.set(context)
        started_ns = time.perf_counter_ns()
        self.manager.emit(
            f"{operation}_start",
            event_level="scheduler",
            batch_id=batch_id,
            operation=operation,
            execution_mode=execution_mode,
            **metadata,
            **fields,
        )
        try:
            result = callback()
        except BaseException as error:
            self.manager.emit(
                f"{operation}_error",
                event_level="scheduler",
                batch_id=batch_id,
                operation=operation,
                execution_mode=execution_mode,
                error_type=type(error).__name__,
                host_duration_us=(
                    time.perf_counter_ns() - started_ns
                ) / 1_000.0,
                **metadata,
                **fields,
            )
            raise
        else:
            self.manager.emit(
                f"{operation}_end",
                event_level="scheduler",
                batch_id=batch_id,
                operation=operation,
                execution_mode=execution_mode,
                host_duration_us=(
                    time.perf_counter_ns() - started_ns
                ) / 1_000.0,
                **metadata,
                **fields,
            )
            return result
        finally:
            _CURRENT_BATCH.reset(token)

    def prefill(self, requests: Sequence[Any]) -> Any:
        return self._call(
            "prefill",
            requests,
            lambda: self.backend.prefill(requests),
        )

    def prefill_chunk(
        self, requests: Sequence[Any], chunk_size: int
    ) -> Any:
        return self._call(
            "prefill_chunk",
            requests,
            lambda: self.backend.prefill_chunk(requests, chunk_size),
            chunk_size=int(chunk_size),
        )

    def decode(self, requests: Sequence[Any]) -> Any:
        return self._call(
            "decode",
            requests,
            lambda: self.backend.decode(requests),
        )

    def release(self, requests: Sequence[Any]) -> Any:
        return self._call(
            "release",
            requests,
            lambda: self.backend.release(requests),
        )

    def preempt(self, requests: Sequence[Any]) -> Any:
        return self._call(
            "preempt",
            requests,
            lambda: self.backend.preempt(requests),
        )


def _shape_of(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    try:
        return [int(dimension) for dimension in shape]
    except (TypeError, ValueError):
        return None


class ModelLayerTracer:
    """Install host-side progress hooks on decoder and vision layer lists."""

    CONTAINER_NAMES = frozenset(("layers", "blocks"))

    def __init__(self, model: Any, manager: TraceManager) -> None:
        self.model = model
        self.manager = manager
        self._handles: list[Any] = []
        self._starts = threading.local()
        self.layer_names: list[str] = []

    def install(self) -> int:
        seen_modules: set[int] = set()
        for container_name, container in self.model.named_modules():
            if not container_name:
                continue
            if container_name.rsplit(".", 1)[-1] not in self.CONTAINER_NAMES:
                continue
            for child_name, layer in container.named_children():
                if id(layer) in seen_modules:
                    continue
                seen_modules.add(id(layer))
                layer_name = f"{container_name}.{child_name}"
                self.layer_names.append(layer_name)
                self._handles.extend(
                    self._register_layer(layer, layer_name, child_name)
                )
        return len(self.layer_names)

    def _register_layer(
        self, layer: Any, layer_name: str, child_name: str
    ) -> tuple[Any, Any]:
        try:
            layer_index: int | str = int(child_name)
        except ValueError:
            layer_index = child_name

        def before(_module: Any, inputs: tuple[Any, ...]) -> None:
            context = current_batch_context()
            if context is None or not self.manager.allows("layer"):
                return
            starts = getattr(self._starts, "values", None)
            if starts is None:
                starts = {}
                self._starts.values = starts
            starts[(context.batch_id, layer_name)] = time.perf_counter_ns()
            self.manager.emit(
                "layer_start",
                event_level="layer",
                batch_id=context.batch_id,
                operation=context.operation,
                execution_mode=context.execution_mode,
                request_ids=context.request_ids,
                control_ids=context.control_ids,
                layer_index=layer_index,
                module=layer_name,
                input_shape=_shape_of(inputs[0]) if inputs else None,
            )

        def after(
            _module: Any, inputs: tuple[Any, ...], output: Any
        ) -> None:
            context = current_batch_context()
            if context is None or not self.manager.allows("layer"):
                return
            starts = getattr(self._starts, "values", {})
            started_ns = starts.pop(
                (context.batch_id, layer_name), time.perf_counter_ns()
            )
            output_value = output[0] if isinstance(output, tuple) else output
            self.manager.emit(
                "layer_end",
                event_level="layer",
                batch_id=context.batch_id,
                operation=context.operation,
                execution_mode=context.execution_mode,
                request_ids=context.request_ids,
                control_ids=context.control_ids,
                layer_index=layer_index,
                module=layer_name,
                input_shape=_shape_of(inputs[0]) if inputs else None,
                output_shape=_shape_of(output_value),
                host_duration_us=(
                    time.perf_counter_ns() - started_ns
                ) / 1_000.0,
            )

        return (
            layer.register_forward_pre_hook(before),
            layer.register_forward_hook(after),
        )

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                logger.exception("failed to remove model trace hook")
        self._handles = []


def install_generator_layer_hooks(
    generator: Any, manager: TraceManager
) -> ModelLayerTracer | None:
    """Find a generator's model and install generic layer-list hooks."""

    executor = getattr(generator, "model_executor", None)
    model = getattr(executor, "model", None)
    if model is None:
        logger.warning("trace layer hooks skipped: model executor not found")
        return None
    tracer = ModelLayerTracer(model, manager)
    count = tracer.install()
    if count == 0:
        logger.warning("trace layer hooks skipped: no layer containers found")
        return None
    logger.info("installed %d model layer trace hooks", count)
    return tracer


def rank_output_path(
    output_path: str | Path | None, *, rank: int, tensor_parallel: bool
) -> Path | None:
    """Return a deterministic per-rank trace path for TP runs."""

    if output_path is None:
        return None
    path = Path(output_path)
    if not tensor_parallel:
        return path
    suffix = path.suffix or ".jsonl"
    stem = path.name[:-len(path.suffix)] if path.suffix else path.name
    return path.with_name(f"{stem}.rank{int(rank)}{suffix}")


def request_ids(requests: Iterable[Any]) -> tuple[str, ...]:
    """Expose a small helper for tests and external trace integrations."""

    return tuple(
        str(getattr(request, "request_id", "unknown"))
        for request in requests
    )
