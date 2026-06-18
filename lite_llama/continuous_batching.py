"""Request state and scheduling primitives for continuous batching.

The scheduler is deliberately model-agnostic. A backend owns the model and KV
cache and exposes prefill/decode/release operations. HTTP handlers may submit
requests concurrently, while one scheduler thread remains the sole model owner.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from queue import Queue
from threading import Event, Lock
from typing import Callable, Protocol, Sequence


class KVCacheCapacityError(RuntimeError):
    """Raised when the live KV cache cannot admit or extend a request."""


class IncrementalTokenDecoder:
    """Decode a bounded suffix and fall back when token boundaries move."""

    def __init__(
        self,
        decode_tokens: Callable[[Sequence[int]], str],
        suffix_window: int = 8,
    ) -> None:
        self.decode_tokens = decode_tokens
        self.suffix_window = max(1, int(suffix_window))
        self.token_ids: list[int] = []
        self.text = ""

    def push(self, token_id: int) -> str:
        previous_suffix_ids = self.token_ids[-self.suffix_window :]
        previous_suffix = (
            self.decode_tokens(previous_suffix_ids)
            if previous_suffix_ids
            else ""
        )
        self.token_ids.append(int(token_id))
        current_suffix_ids = [*previous_suffix_ids, int(token_id)]
        current_suffix = self.decode_tokens(current_suffix_ids)

        if current_suffix.startswith(previous_suffix):
            delta = current_suffix[len(previous_suffix) :]
            self.text += delta
            return delta

        decoded_text = self.decode_tokens(self.token_ids)
        delta = (
            decoded_text[len(self.text) :]
            if decoded_text.startswith(self.text)
            else ""
        )
        self.text = decoded_text
        return delta


@dataclass(frozen=True)
class BatchOutput:
    delta: str = ""
    token_id: int | None = None
    finished: bool = False
    finish_reason: str | None = None
    error: str | None = None


class BatchRequest:
    def __init__(
        self,
        request_id: str,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        control_id: int | None = None,
    ) -> None:
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        self.request_id = request_id
        self.control_id = control_id
        self.prompt_tokens = list(prompt_tokens)
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generated_token_ids: list[int] = []
        self.preemptions = 0
        self.prefill_cursor = 0
        self.prefill_credit_tokens = 0
        self.decoded_text = ""
        self.model_request_id: int | None = None
        self.finished = False
        self.finish_reason: str | None = None
        self.cancelled = False
        self.outputs: Queue[BatchOutput] = Queue()
        self.done = Event()
        self._incremental_decoder: IncrementalTokenDecoder | None = None

    @property
    def last_token_id(self) -> int:
        if not self.generated_token_ids:
            raise RuntimeError("request has no generated token")
        return self.generated_token_ids[-1]

    @property
    def model_context_tokens(self) -> list[int]:
        """Tokens that must exist in KV before sampling the next token."""
        return [*self.prompt_tokens, *self.generated_token_ids]

    def mark_preempted(self) -> None:
        self.preemptions += 1
        self.model_request_id = None

    def accept_token(
        self,
        token_id: int,
        eos_token_id: int,
        decode_tokens: Callable[[Sequence[int]], str],
    ) -> None:
        if self.finished:
            return
        token_id = int(token_id)
        self.generated_token_ids.append(token_id)
        reached_eos = token_id == eos_token_id
        reached_limit = len(self.generated_token_ids) >= self.max_new_tokens
        if self._incremental_decoder is None:
            self._incremental_decoder = IncrementalTokenDecoder(decode_tokens)
        delta = (
            ""
            if reached_eos
            else self._incremental_decoder.push(token_id)
        )
        self.decoded_text = self._incremental_decoder.text
        if delta:
            self.outputs.put(BatchOutput(delta=delta, token_id=token_id))
        if reached_eos or reached_limit:
            self.finish("stop" if reached_eos else "length")

    def finish(self, reason: str) -> None:
        if self.finished:
            return
        self.finished = True
        self.finish_reason = reason
        self.outputs.put(
            BatchOutput(finished=True, finish_reason=reason)
        )
        self.done.set()

    def fail(self, error: BaseException) -> None:
        if self.finished:
            return
        self.finished = True
        self.finish_reason = "error"
        self.outputs.put(
            BatchOutput(
                finished=True,
                finish_reason="error",
                error=str(error),
            )
        )
        self.done.set()

    def cancel(self) -> None:
        self.cancelled = True


class ContinuousBatchBackend(Protocol):
    def prefill(self, requests: Sequence[BatchRequest]) -> Sequence[int]:
        ...

    def decode(self, requests: Sequence[BatchRequest]) -> Sequence[int]:
        ...

    def release(self, requests: Sequence[BatchRequest]) -> None:
        ...

    def preempt(self, requests: Sequence[BatchRequest]) -> None:
        ...


class ContinuousBatchScheduler:
    """Admit waiting requests and execute one decode step per scheduling tick."""

    def __init__(
        self,
        backend: ContinuousBatchBackend,
        max_batch_size: int,
        eos_token_id: int,
        decode_tokens: Callable[[Sequence[int]], str],
        max_waiting_requests: int = 1024,
        max_prefill_tokens: int | None = None,
        max_decode_tokens: int | None = None,
        chunked_prefill: bool = False,
        prefill_chunk_size: int | None = None,
        max_preemptions: int = 1,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self.backend = backend
        self.max_batch_size = max_batch_size
        self.eos_token_id = int(eos_token_id)
        self.decode_tokens = decode_tokens
        self.max_waiting_requests = max_waiting_requests
        self.max_prefill_tokens = self._validate_optional_positive(
            max_prefill_tokens, "max_prefill_tokens"
        )
        self.max_decode_tokens = self._validate_optional_positive(
            max_decode_tokens, "max_decode_tokens"
        )
        self.chunked_prefill = bool(chunked_prefill)
        self.prefill_chunk_size = self._validate_optional_positive(
            prefill_chunk_size, "prefill_chunk_size"
        )
        if self.chunked_prefill and self.prefill_chunk_size is None:
            raise ValueError(
                "prefill_chunk_size must be set when chunked_prefill is enabled"
            )
        self.max_preemptions = int(max_preemptions)
        if self.max_preemptions < 0:
            raise ValueError("max_preemptions must be non-negative")
        self._pending: deque[BatchRequest] = deque()
        self._prefilling: list[BatchRequest] = []
        self._active: list[BatchRequest] = []
        self._lock = Lock()
        self._next_control_id = 0

    @staticmethod
    def _validate_optional_positive(
        value: int | None, name: str
    ) -> int | None:
        if value is None:
            return None
        value = int(value)
        if value < 1:
            raise ValueError(f"{name} must be positive")
        return value

    @property
    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    @property
    def active_count(self) -> int:
        return len(self._active)

    @property
    def prefilling_count(self) -> int:
        return len(self._prefilling)

    def submit(
        self,
        request_id: str,
        prompt_tokens: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
    ) -> BatchRequest:
        with self._lock:
            if len(self._pending) >= self.max_waiting_requests:
                raise RuntimeError("continuous batching waiting queue is full")
            request = BatchRequest(
                request_id=request_id,
                prompt_tokens=prompt_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                control_id=self._next_control_id,
            )
            self._next_control_id += 1
            self._pending.append(request)
        return request

    def _admit(self, capacity: int) -> list[BatchRequest]:
        if self.chunked_prefill and self.prefill_chunk_size:
            return self._admit_chunked(capacity)

        admitted: list[BatchRequest] = []
        remaining_prefill_tokens = self.max_prefill_tokens
        with self._lock:
            while capacity > 0 and self._pending:
                request = self._pending[0]
                if request.cancelled:
                    self._pending.popleft()
                    request.finish("cancelled")
                    continue

                prompt_tokens = len(request.model_context_tokens)
                if self.chunked_prefill and self.prefill_chunk_size:
                    prompt_tokens = min(prompt_tokens, self.prefill_chunk_size)
                if remaining_prefill_tokens is not None:
                    if prompt_tokens > remaining_prefill_tokens:
                        if admitted:
                            break
                        # Avoid starvation: one oversized request may run alone.
                        remaining_prefill_tokens = 0
                    else:
                        remaining_prefill_tokens -= prompt_tokens

                admitted.append(self._pending.popleft())
                capacity -= 1
                if remaining_prefill_tokens == 0:
                    break
        return admitted

    def _admit_chunked(self, capacity: int) -> list[BatchRequest]:
        admitted: list[BatchRequest] = []
        remaining_prefill_tokens = self.max_prefill_tokens
        with self._lock:
            scan_budget = len(self._pending)
            while capacity > 0 and self._pending and scan_budget > 0:
                scan_budget -= 1
                request = self._pending[0]
                if request.cancelled:
                    self._pending.popleft()
                    request.finish("cancelled")
                    continue

                remaining_tokens = max(
                    0, len(request.model_context_tokens) - request.prefill_cursor
                )
                token_cost = min(remaining_tokens, self.prefill_chunk_size)
                if remaining_prefill_tokens is not None:
                    if token_cost > remaining_prefill_tokens:
                        if admitted:
                            break
                        # Avoid starvation: one oversized chunk may run alone.
                        remaining_prefill_tokens = 0
                    else:
                        remaining_prefill_tokens -= token_cost

                admitted.append(self._pending.popleft())
                capacity -= 1
                if remaining_prefill_tokens == 0:
                    break
        return admitted

    def _select_prefill_chunk_requests(
        self, requests: Sequence[BatchRequest]
    ) -> tuple[list[BatchRequest], list[BatchRequest]]:
        if not self.chunked_prefill or self.prefill_chunk_size is None:
            return list(requests), []
        if self.max_prefill_tokens is None:
            return list(requests), []
        selected: list[BatchRequest] = []
        deferred: list[BatchRequest] = []
        remaining_budget = self.max_prefill_tokens
        for index, request in enumerate(requests):
            token_cost = min(
                max(0, len(request.model_context_tokens) - request.prefill_cursor),
                self.prefill_chunk_size,
            )
            if token_cost > remaining_budget and selected:
                deferred.extend(requests[index:])
                break
            if token_cost > remaining_budget:
                selected.append(request)
                deferred.extend(requests[index + 1:])
                break
            selected.append(request)
            remaining_budget = max(0, remaining_budget - token_cost)
            if remaining_budget == 0:
                deferred.extend(requests[index + 1:])
                break
        return selected, deferred

    def _select_decode_requests(
        self, requests: Sequence[BatchRequest]
    ) -> tuple[list[BatchRequest], list[BatchRequest]]:
        if self.max_decode_tokens is None:
            return list(requests), []
        limit = min(len(requests), int(self.max_decode_tokens))
        return list(requests[:limit]), list(requests[limit:])

    def _apply_tokens(
        self,
        requests: Sequence[BatchRequest],
        token_ids: Sequence[int],
    ) -> None:
        if len(requests) != len(token_ids):
            raise RuntimeError(
                "backend returned a token count that does not match the batch"
            )
        for request, token_id in zip(requests, token_ids):
            request.accept_token(
                token_id,
                eos_token_id=self.eos_token_id,
                decode_tokens=self.decode_tokens,
            )

    def _release_finished(
        self, requests: Sequence[BatchRequest]
    ) -> list[BatchRequest]:
        finished = [
            request
            for request in requests
            if request.finished or request.cancelled
        ]
        if finished:
            self.backend.release(finished)
            for request in finished:
                if request.cancelled and not request.finished:
                    request.finish("cancelled")
        return [
            request
            for request in requests
            if request not in finished
        ]

    @staticmethod
    def _is_kv_capacity_error(error: BaseException) -> bool:
        if isinstance(error, KVCacheCapacityError):
            return True
        message = str(error).lower()
        return (
            "kv" in message
            and (
                "capacity" in message
                or "exhaust" in message
                or "allocation failed" in message
                or "out of memory" in message
                or "oom" in message
            )
        )

    def _requeue_front(self, requests: Sequence[BatchRequest]) -> None:
        with self._lock:
            for request in reversed(list(requests)):
                if not request.finished and not request.cancelled:
                    self._pending.appendleft(request)

    def _preempt_one(self, candidates: Sequence[BatchRequest]) -> BatchRequest | None:
        eligible = [
            request
            for request in candidates
            if (
                not request.finished
                and not request.cancelled
                and request.preemptions < self.max_preemptions
            )
        ]
        if not eligible:
            return None
        # Prefer preempting the longest live context. That frees the most KV
        # pages and mirrors common decode-scheduler pressure handling.
        victim = max(eligible, key=lambda request: len(request.model_context_tokens))
        if hasattr(self.backend, "preempt"):
            self.backend.preempt([victim])
        else:
            self.backend.release([victim])
        victim.mark_preempted()
        self._requeue_front([victim])
        return victim

    def step(self) -> bool:
        """Run one scheduler tick. Return whether any work was performed."""
        prior_active = self._release_finished(self._active)
        self._active = prior_active
        decode_active, deferred_active = self._select_decode_requests(
            prior_active
        )
        admitted = self._admit(
            self.max_batch_size - len(prior_active) - len(self._prefilling)
        )
        prefilling_candidates = self._prefilling + admitted
        prefill_work: list[BatchRequest] = []
        deferred_prefilling: list[BatchRequest] = []
        if self.chunked_prefill and self.prefill_chunk_size is not None:
            prefill_work, deferred_prefilling = self._select_prefill_chunk_requests(
                prefilling_candidates
            )
        did_work = bool(
            decode_active or admitted or deferred_active or self._prefilling
        )
        uncommitted_admitted = list(admitted)

        try:
            if self.chunked_prefill and self.prefill_chunk_size is not None:
                if decode_active:
                    decode_tokens = self.backend.decode(decode_active)
                    self._apply_tokens(decode_active, decode_tokens)
                    decode_active = self._release_finished(decode_active)

                completed_prefill: list[BatchRequest] = []
                if prefill_work:
                    prefill_tokens = self.backend.prefill_chunk(
                        prefill_work, self.prefill_chunk_size
                    )
                    for request, token_id in zip(prefill_work, prefill_tokens):
                        if token_id is None:
                            continue
                        self._apply_tokens([request], [int(token_id)])
                        completed_prefill.append(request)
                    completed_prefill = self._release_finished(completed_prefill)
                    uncommitted_admitted = []

                incomplete_prefill = [
                    request
                    for request in prefill_work
                    if request not in completed_prefill
                    and not request.finished
                    and not request.cancelled
                ]
                self._prefilling = deferred_prefilling + incomplete_prefill
                self._active = deferred_active + decode_active + completed_prefill
                return did_work

            if admitted:
                prefill_tokens = self.backend.prefill(admitted)
                self._apply_tokens(admitted, prefill_tokens)
                admitted = self._release_finished(admitted)
                uncommitted_admitted = []

            if decode_active:
                decode_tokens = self.backend.decode(decode_active)
                self._apply_tokens(decode_active, decode_tokens)
                decode_active = self._release_finished(decode_active)
        except BaseException as error:
            if self._is_kv_capacity_error(error):
                if uncommitted_admitted:
                    for request in uncommitted_admitted:
                        request.model_request_id = None
                    self._requeue_front(uncommitted_admitted)
                victim = self._preempt_one(prior_active)
                if victim is not None:
                    self._active = [
                        request for request in prior_active if request is not victim
                    ]
                    return True
            affected = list(prior_active) + list(admitted)
            for request in affected:
                request.fail(error)
            if affected:
                try:
                    self.backend.release(affected)
                except BaseException:
                    pass
            self._active = []
            return did_work

        self._active = deferred_active + decode_active + admitted
        return did_work

    def shutdown(self) -> None:
        with self._lock:
            pending = list(self._pending)
            self._pending.clear()
        for request in pending:
            request.finish("cancelled")
        if self._active:
            self.backend.release(self._active)
            for request in self._active:
                request.finish("cancelled")
            self._active = []
        if self._prefilling:
            self.backend.release(self._prefilling)
            for request in self._prefilling:
                request.finish("cancelled")
            self._prefilling = []


class ContinuousBatchModelBackend:
    """Paged-KV model adapter used by :class:`ContinuousBatchScheduler`."""

    def __init__(
        self,
        generator,
        return_host_tokens: bool = True,
        enable_partial_prefix_cache: bool = False,
    ) -> None:
        self.generator = generator
        self.executor = generator.model_executor
        self.tokenizer = generator.tokenizer
        self.return_host_tokens = bool(return_host_tokens)
        self.enable_partial_prefix_cache = bool(enable_partial_prefix_cache)
        self._device_tokens: dict[int, object] = {}
        self._device_positions: dict[int, object] = {}
        if not self.executor.use_paged_attn:
            raise RuntimeError(
                "continuous batching requires --page_size greater than zero"
            )

    @property
    def eos_token_id(self) -> int:
        return int(self.tokenizer.eos_token_id)

    def tokenize(self, prompt: str) -> list[int]:
        return self.tokenizer.encode(prompt, add_special_tokens=True)

    def decode_tokens(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(
            [int(token_id) for token_id in token_ids],
            skip_special_tokens=True,
        )

    def _sample_device(self, logits, requests: Sequence[BatchRequest]):
        from lite_llama.sampling import sample_next_token

        return sample_next_token(
            logits,
            temperature=[request.temperature for request in requests],
            top_p=[request.top_p for request in requests],
            vocab_parallel=bool(
                getattr(self.executor, "logits_are_sharded", False)
            ),
        )

    def _remember_sampled_tokens(
        self,
        requests: Sequence[BatchRequest],
        sampled,
        *,
        initial_positions: Sequence[int] | None = None,
    ) -> None:
        import torch

        for row, request in enumerate(requests):
            req_idx = int(request.model_request_id)
            self._device_tokens[req_idx] = sampled[row].reshape(())
            if initial_positions is not None:
                self._device_positions[req_idx] = torch.tensor(
                    int(initial_positions[row]),
                    dtype=torch.long,
                    device=self.executor.device,
                )

    def _tokens_to_host(self, sampled) -> list[int]:
        if not self.return_host_tokens:
            return []
        return sampled.detach().cpu().tolist()

    def _cache_lookup(
        self, request: BatchRequest
    ) -> tuple[int, int, int | None] | None:
        context_tokens = request.model_context_tokens
        if request.temperature != 0:
            return None
        if self.enable_partial_prefix_cache and hasattr(
            self.executor, "share_paged_prefix_from_cache"
        ):
            cached = self.executor.share_paged_prefix_from_cache(context_tokens)
            if cached is not None:
                req_idx, matched_tokens, token_id = cached
                return int(req_idx), int(matched_tokens), (
                    None if token_id is None else int(token_id)
                )
        if hasattr(self.executor, "share_paged_request_from_cache"):
            cached = self.executor.share_paged_request_from_cache(context_tokens)
            if cached is not None:
                req_idx, token_id = cached
                return int(req_idx), len(context_tokens), int(token_id)
        return None

    def _ensure_incremental_request(self, request: BatchRequest) -> None:
        if request.model_request_id is not None:
            return
        request_ids = self.executor.reserve_paged_requests((1,))
        request.model_request_id = int(request_ids[0])
        request.prefill_cursor = 0

    def _run_prefill_chunk_incremental(
        self,
        request: BatchRequest,
        chunk_size: int | None,
    ) -> int | None:
        import torch

        context_tokens = request.model_context_tokens
        if not context_tokens:
            raise RuntimeError("empty prompts are not supported by chunked prefill")
        self._ensure_incremental_request(request)
        req_idx = int(request.model_request_id)
        end = len(context_tokens)
        if chunk_size is not None:
            end = min(end, request.prefill_cursor + int(chunk_size))
        logits = None
        while request.prefill_cursor < end:
            position = request.prefill_cursor
            while self.executor.req_tokens_manager.req_token_count[req_idx] <= position:
                self.executor.extend_paged_requests((req_idx,))
            self.executor.activate_paged_decode_batch((req_idx,))
            input_ids = torch.tensor(
                [[int(context_tokens[position])]],
                dtype=torch.long,
                device=self.executor.device,
            )
            position_ids = torch.tensor(
                [[int(position)]],
                dtype=torch.long,
                device=self.executor.device,
            )
            logits = self.executor.forward(input_ids, position_ids)
            request.prefill_cursor += 1
        if request.prefill_cursor < len(context_tokens):
            return None
        if logits is None:
            raise RuntimeError("prefill produced no logits")
        sampled = self._sample_device(logits, [request])
        self._remember_sampled_tokens(
            [request],
            sampled,
            initial_positions=[len(context_tokens)],
        )
        self.executor.extend_paged_requests((req_idx,))
        host_tokens = self._tokens_to_host(sampled)
        token_id = (
            int(sampled.detach().cpu().tolist()[0])
            if not self.return_host_tokens
            else int(host_tokens[0])
        )
        if hasattr(self.executor, "store_paged_request_prefix") and request.temperature == 0:
            self.executor.store_paged_request_prefix(
                context_tokens,
                req_idx,
                token_id,
            )
        return None if not self.return_host_tokens else token_id

    def prefill(self, requests: Sequence[BatchRequest]) -> Sequence[int]:
        import torch

        if not requests:
            return []
        results: list[int | None] = [None] * len(requests)
        cache_misses: list[tuple[int, BatchRequest]] = []
        for index, request in enumerate(requests):
            context_tokens = request.model_context_tokens
            cached = self._cache_lookup(request)
            if cached is None:
                cache_misses.append((index, request))
                continue
            req_idx, matched_tokens, token_id = cached
            request.model_request_id = int(req_idx)
            request.prefill_cursor = int(matched_tokens)
            if token_id is None:
                token_id = self._run_prefill_chunk_incremental(request, None)
                if token_id is None and self.return_host_tokens:
                    raise RuntimeError("partial prefix replay did not produce a token")
                results[index] = token_id
                continue
            sampled = torch.tensor(
                [int(token_id)], dtype=torch.long, device=self.executor.device
            )
            self._remember_sampled_tokens(
                [request],
                sampled,
                initial_positions=[len(context_tokens)],
            )
            results[index] = int(token_id)

        if not cache_misses:
            if not self.return_host_tokens:
                return []
            return [int(token_id) for token_id in results]

        miss_requests = [request for _, request in cache_misses]
        context_lengths = [len(request.model_context_tokens) for request in miss_requests]
        request_ids = self.executor.reserve_paged_requests(
            context_lengths
        )
        for request, req_idx in zip(miss_requests, request_ids):
            request.model_request_id = req_idx

        groups: dict[int, list[tuple[int, BatchRequest]]] = {}
        for index, request in cache_misses:
            groups.setdefault(len(request.model_context_tokens), []).append(
                (index, request)
            )

        try:
            for prompt_length, indexed_requests in groups.items():
                group_requests = [
                    request for _, request in indexed_requests
                ]
                group_ids = tuple(
                    int(request.model_request_id)
                    for request in group_requests
                )
                self.executor.activate_paged_prefill_batch(
                    group_ids, prompt_length
                )
                input_ids = torch.tensor(
                    [request.model_context_tokens for request in group_requests],
                    dtype=torch.long,
                    device=self.executor.device,
                )
                position_ids = torch.arange(
                    prompt_length,
                    dtype=torch.long,
                    device=self.executor.device,
                ).unsqueeze(0).expand(len(group_requests), -1)
                logits = self.executor.forward(input_ids, position_ids)
                sampled = self._sample_device(logits, group_requests)
                self._remember_sampled_tokens(
                    group_requests,
                    sampled,
                    initial_positions=[prompt_length] * len(group_requests),
                )
                group_tokens = self._tokens_to_host(sampled)
                self.executor.extend_paged_requests(group_ids)
                if self.return_host_tokens:
                    for (original_index, _), token_id in zip(
                        indexed_requests, group_tokens
                    ):
                        results[original_index] = token_id
                if hasattr(self.executor, "store_paged_request_prefix"):
                    host_tokens = (
                        group_tokens
                        if self.return_host_tokens
                        else sampled.detach().cpu().tolist()
                    )
                    for request, token_id in zip(group_requests, host_tokens):
                        if request.temperature == 0:
                            self.executor.store_paged_request_prefix(
                                request.model_context_tokens,
                                int(request.model_request_id),
                                int(token_id),
                            )
        except BaseException:
            self.executor.release_paged_request_ids(request_ids)
            for request in miss_requests:
                request.model_request_id = None
            raise

        if not self.return_host_tokens:
            return []
        return [int(token_id) for token_id in results]

    def prefill_chunk(
        self,
        requests: Sequence[BatchRequest],
        chunk_size: int,
    ) -> Sequence[int | None]:
        results: list[int | None] = []
        for request in requests:
            cached = None
            if request.model_request_id is None:
                cached = self._cache_lookup(request)
            if cached is not None:
                req_idx, matched_tokens, token_id = cached
                request.model_request_id = int(req_idx)
                request.prefill_cursor = int(matched_tokens)
                if token_id is not None:
                    import torch

                    sampled = torch.tensor(
                        [int(token_id)], dtype=torch.long, device=self.executor.device
                    )
                    self._remember_sampled_tokens(
                        [request],
                        sampled,
                        initial_positions=[len(request.model_context_tokens)],
                    )
                    results.append(int(token_id) if self.return_host_tokens else None)
                    continue
            results.append(
                self._run_prefill_chunk_incremental(request, int(chunk_size))
            )
        return results

    def decode(self, requests: Sequence[BatchRequest]) -> Sequence[int]:
        import torch

        if not requests:
            return []
        request_ids = tuple(
            int(request.model_request_id) for request in requests
        )
        self.executor.activate_paged_decode_batch(request_ids)
        input_ids = torch.stack(
            [self._device_tokens[req_idx] for req_idx in request_ids]
        ).reshape(-1, 1)
        position_ids = torch.stack(
            [self._device_positions[req_idx] for req_idx in request_ids]
        ).reshape(-1, 1)
        logits = self.executor.forward(input_ids, position_ids)
        sampled = self._sample_device(logits, requests)
        self._remember_sampled_tokens(requests, sampled)
        for req_idx in request_ids:
            self._device_positions[req_idx].add_(1)
        self.executor.extend_paged_requests(request_ids)
        return self._tokens_to_host(sampled)

    def release(self, requests: Sequence[BatchRequest]) -> None:
        request_ids = tuple(
            int(request.model_request_id)
            for request in requests
            if request.model_request_id is not None
        )
        if request_ids:
            self.executor.release_paged_request_ids(request_ids)
            for req_idx in request_ids:
                self._device_tokens.pop(req_idx, None)
                self._device_positions.pop(req_idx, None)
        for request in requests:
            request.model_request_id = None

    def preempt(self, requests: Sequence[BatchRequest]) -> None:
        self.release(requests)
