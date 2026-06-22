"""OpenAI-compatible API server for lite_llama.

Usage (single card):
  python server.py --checkpoints_dir my_weight/Qwen3-32B/ --port 8000

Usage (TP multi-card):
  ASCEND_RT_VISIBLE_DEVICES=4,5 python -m torch.distributed.run \
      --nproc_per_node=2 --master_addr=127.0.0.1 --master_port=29500 \
      server.py --checkpoints_dir my_weight/Qwen3-32B/ --port 8000

Endpoints:
  POST /v1/chat/completions    — OpenAI-compatible chat (with streaming)
  POST /v1/completions          — OpenAI-compatible text completion
  GET  /v1/models               — list available models
  GET  /health                  — health check
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional, List, Union, AsyncGenerator

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from PIL import Image

from lite_llama.utils.device import get_device

# ---------------------------------------------------------------------------
# Pydantic models (OpenAI-compatible schemas)
# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: Union[str, List[dict]]


class StreamOptions(BaseModel):
    include_usage: bool = False


class ChatCompletionRequest(BaseModel):
    model: str = "default"
    messages: List[ChatMessage]
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    stream: bool = False
    stream_options: Optional[StreamOptions] = None
    enable_thinking: bool = True


class CompletionRequest(BaseModel):
    model: str = "default"
    prompt: Union[str, List[str]]
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
    top_p: float = Field(default=0.9, ge=0.0, le=1.0)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    stream: bool = False


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    owned_by: str = "lite_llama"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# ---------------------------------------------------------------------------
# Global model handle
# ---------------------------------------------------------------------------
_generator = None
_is_vl = False
_model_name = "default"
_rank = 0
_is_tp = False
_tp_lock = None  # threading.Lock for single-request-at-a-time in TP mode
_continuous_batching = False
_continuous_backend = None
_continuous_scheduler = None
_partial_prefix_cache = False
_scheduler_thread = None
_scheduler_stop = None
_scheduler_poll_seconds = 0.001


def load_generator(
    checkpoints_dir: str,
    device: str,
    *,
    max_seq_len: int = 1024,
    page_size: int = 16,
    compiled_model: bool = True,
    moe_parallel_mode: str = "tp",
):
    global _generator, _is_vl, _model_name
    import json
    from pathlib import Path

    cfg_path = Path(checkpoints_dir) / "config.json"
    with open(cfg_path) as f:
        config = json.load(f)
    model_type = config.get("model_type", "").lower()
    _is_vl = model_type in ("qwen3_vl", "llava")
    _model_name = Path(checkpoints_dir).name

    if _is_vl:
        from lite_llama.qwen3vl_generate_stream import Qwen3VLGeneratorStream
        _generator = Qwen3VLGeneratorStream(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_seq_len=max_seq_len,
            device=device,
        )
    else:
        from lite_llama.generate_stream import GenerateStreamText
        _generator = GenerateStreamText(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            page_size=page_size,
            moe_parallel_mode=moe_parallel_mode,
            device=device,
        )


# ---------------------------------------------------------------------------
# TP worker sync (rank 1 mirrors rank 0's generation calls)
# ---------------------------------------------------------------------------
def _tp_run_generation(
    prompt: str, images: list,
    temperature: float, top_p: float, max_tokens: int,
) -> str:
    """Called by rank 0. Broadcasts request to rank 1, then both run generation."""
    import threading

    # Broadcast request params
    params = [prompt, len(images), temperature, top_p, max_tokens]
    torch.distributed.broadcast_object_list(params, src=0)
    prompt, n_images, temperature, top_p, max_tokens = params

    images = images if _is_vl and n_images > 0 else []

    # Both ranks run generation
    if _is_vl:
        stream = _generator.text_completion_stream(
            [prompt], images,
            temperature=temperature, top_p=top_p, max_gen_len=max_tokens,
        )
    else:
        stream = _generator.text_completion_stream(
            [prompt],
            temperature=temperature, top_p=top_p, max_gen_len=max_tokens,
        )

    completion = ""
    for batch in stream:
        completion = batch[0].get("generation", "") if isinstance(batch[0], dict) else batch[0]
    return completion


def _tp_worker_loop():
    """Rank 1 loop: wait for broadcast from rank 0, then run generation."""
    import torch
    while True:
        # Wait for request
        params = ["", 0, 0.6, 0.9, 256]
        torch.distributed.broadcast_object_list(params, src=0)
        prompt, n_images, temperature, top_p, max_tokens = params

        if prompt == "__SHUTDOWN__":
            break

        # Run generation (output discarded on rank 1)
        if _is_vl:
            stream = _generator.text_completion_stream(
                [prompt], [],
                temperature=temperature, top_p=top_p, max_gen_len=max_tokens,
            )
        else:
            stream = _generator.text_completion_stream(
                [prompt],
                temperature=temperature, top_p=top_p, max_gen_len=max_tokens,
            )
        for _ in stream:
            pass


class _TpCoordinatedContinuousBackend:
    """Mirror scheduler operations with a CPU-side TP control channel."""

    def __init__(self, local_backend):
        from lite_llama.executor.tp_control import StoreCommandChannel

        self.local_backend = local_backend
        self.channel = StoreCommandChannel()
        self._worker_known_control_ids = set()

    @property
    def eos_token_id(self):
        return self.local_backend.eos_token_id

    @property
    def max_context_tokens(self):
        return getattr(self.local_backend, "max_context_tokens", None)

    @property
    def max_prefill_tokens(self):
        return getattr(self.local_backend, "max_prefill_tokens", None)

    def tokenize(self, prompt):
        return self.local_backend.tokenize(prompt)

    def decode_tokens(self, token_ids):
        return self.local_backend.decode_tokens(token_ids)

    def prefill(self, requests):
        from lite_llama.executor.tp_control import encode_prefill

        sequence = self.channel.send(encode_prefill(requests))
        result = self.local_backend.prefill(requests)
        self.channel.wait_ack(sequence)
        self._worker_known_control_ids.update(
            int(request.control_id) for request in requests
        )
        return result

    def prefill_chunk(self, requests, chunk_size):
        from lite_llama.executor.tp_control import encode_prefill_chunk

        previous_model_request_ids = {
            request: request.model_request_id for request in requests
        }
        try:
            if hasattr(self.local_backend, "prepare_prefill_chunk"):
                self.local_backend.prepare_prefill_chunk(requests, chunk_size)
        except BaseException:
            newly_allocated = [
                request
                for request, previous_id in previous_model_request_ids.items()
                if previous_id is None and request.model_request_id is not None
            ]
            if newly_allocated:
                try:
                    self.local_backend.release(newly_allocated)
                except BaseException:
                    pass
            raise
        sequence = self.channel.send(encode_prefill_chunk(requests, chunk_size))
        result = self.local_backend.prefill_chunk(requests, chunk_size)
        self.channel.wait_ack(sequence)
        self._worker_known_control_ids.update(
            int(request.control_id) for request in requests
        )
        return result

    def decode(self, requests):
        from lite_llama.executor.tp_control import encode_decode_state

        unknown = [
            int(request.control_id)
            for request in requests
            if int(request.control_id) not in self._worker_known_control_ids
        ]
        if unknown:
            raise RuntimeError(
                "TP worker decode requested for unknown control_id(s): "
                f"{unknown}. This indicates prefill state was not mirrored."
            )
        sequence = self.channel.send(encode_decode_state(requests))
        result = self.local_backend.decode(requests)
        self.channel.wait_ack(sequence)
        return result

    def release(self, requests):
        from lite_llama.executor.tp_control import encode_release

        sequence = self.channel.send(
            encode_release([request.control_id for request in requests])
        )
        try:
            result = self.local_backend.release(requests)
            self.channel.wait_ack(sequence)
            return result
        finally:
            for request in requests:
                self._worker_known_control_ids.discard(int(request.control_id))

    def preempt(self, requests):
        return self.release(requests)

    def shutdown_workers(self):
        from lite_llama.executor.tp_control import encode_shutdown

        sequence = self.channel.send(encode_shutdown())
        self.channel.wait_ack(sequence)


def _tp_continuous_worker_loop():
    """Mirror rank-0 scheduler operations at model-step granularity."""
    from lite_llama.continuous_batching import (
        BatchRequest,
        ContinuousBatchModelBackend,
    )
    from lite_llama.executor.tp_control import StoreCommandChannel

    backend = ContinuousBatchModelBackend(
        _generator,
        return_host_tokens=False,
        enable_partial_prefix_cache=_partial_prefix_cache,
    )
    channel = StoreCommandChannel()
    requests_by_id = {}
    while True:
        sequence, command = channel.receive_with_sequence()
        if command.operation == "shutdown":
            channel.ack(sequence)
            break

        try:
            worker_requests = []
            if command.operation in ("prefill", "prefill_chunk"):
                for index, control_id in enumerate(command.control_ids):
                    request = requests_by_id.get(control_id)
                    if request is None:
                        request = BatchRequest(
                            request_id=f"tp-worker-{control_id}",
                            control_id=control_id,
                            prompt_tokens=command.prompt_tokens[index],
                            max_new_tokens=command.max_new_tokens[index],
                            temperature=command.temperatures[index],
                            top_p=command.top_ps[index],
                        )
                        requests_by_id[control_id] = request
                    if command.operation == "prefill_chunk":
                        request.prefill_cursor = command.prefill_cursors[index]
                    worker_requests.append(request)
            else:
                for control_id in command.control_ids:
                    request = requests_by_id.get(control_id)
                    if request is None:
                        if command.operation == "release":
                            continue
                        raise RuntimeError(
                            f"unknown continuous batching control_id: "
                            f"{control_id}"
                        )
                    worker_requests.append(request)

            if command.operation == "prefill":
                backend.prefill(worker_requests)
            elif command.operation == "prefill_chunk":
                backend.prefill_chunk(worker_requests, command.chunk_size)
            elif command.operation in ("decode", "decode_state"):
                if command.operation == "decode_state":
                    for request, expected_seq_len in zip(
                        worker_requests, command.expected_seq_lens
                    ):
                        req_idx = int(request.model_request_id)
                        expected_input_position = int(expected_seq_len) - 1
                        actual_input_position = int(
                            backend._device_positions[req_idx]
                            .detach()
                            .cpu()
                            .item()
                        )
                        if actual_input_position != expected_input_position:
                            raise RuntimeError(
                                "TP worker decode_state mismatch: "
                                f"control_id={request.control_id}, "
                                f"model_request_id={req_idx}, "
                                f"expected_seq_len={expected_seq_len}, "
                                f"actual_input_position={actual_input_position}"
                            )
                backend.decode(worker_requests)
            elif command.operation == "release":
                backend.release(worker_requests)
                for request in worker_requests:
                    requests_by_id.pop(request.control_id, None)
            else:
                raise RuntimeError(
                    f"unknown continuous batching TP operation: "
                    f"{command.operation}"
                )
            channel.ack(sequence)
        except BaseException as error:
            channel.ack(sequence, ok=False, message=str(error))
            raise


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
def _start_continuous_scheduler(
    max_batch_size: int,
    max_waiting_requests: int,
    scheduler_poll_ms: float,
    max_prefill_tokens: int | None = None,
    max_decode_tokens: int | None = None,
    chunked_prefill: bool = False,
    prefill_chunk_size: int | None = None,
    max_preemptions: int = 1,
    partial_prefix_cache: bool = False,
) -> None:
    global _continuous_backend, _continuous_scheduler
    global _scheduler_thread, _scheduler_stop, _scheduler_poll_seconds

    from lite_llama.continuous_batching import (
        ContinuousBatchModelBackend,
        ContinuousBatchScheduler,
    )

    local_backend = ContinuousBatchModelBackend(
        _generator,
        enable_partial_prefix_cache=partial_prefix_cache,
    )
    _continuous_backend = (
        _TpCoordinatedContinuousBackend(local_backend)
        if _is_tp
        else local_backend
    )
    _continuous_scheduler = ContinuousBatchScheduler(
        backend=_continuous_backend,
        max_batch_size=max_batch_size,
        eos_token_id=_continuous_backend.eos_token_id,
        decode_tokens=_continuous_backend.decode_tokens,
        max_waiting_requests=max_waiting_requests,
        max_prefill_tokens=max_prefill_tokens,
        max_decode_tokens=max_decode_tokens,
        chunked_prefill=chunked_prefill,
        prefill_chunk_size=prefill_chunk_size,
        max_preemptions=max_preemptions,
    )
    _scheduler_poll_seconds = max(0.0001, scheduler_poll_ms / 1000.0)
    _scheduler_stop = threading.Event()

    def scheduler_loop():
        while not _scheduler_stop.is_set():
            did_work = _continuous_scheduler.step()
            if not did_work:
                _scheduler_stop.wait(_scheduler_poll_seconds)

    _scheduler_thread = threading.Thread(
        target=scheduler_loop,
        name="lite-llama-continuous-batching",
        daemon=True,
    )
    _scheduler_thread.start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: generator is loaded by main() before uvicorn
    yield
    # Shutdown
    global _generator, _continuous_scheduler
    if _scheduler_stop is not None:
        _scheduler_stop.set()
    if _scheduler_thread is not None:
        _scheduler_thread.join(timeout=10)
    if _continuous_scheduler is not None:
        _continuous_scheduler.shutdown()
    if _is_tp and _continuous_batching:
        _continuous_backend.shutdown_workers()
    _continuous_scheduler = None
    _generator = None


app = FastAPI(title="lite_llama API", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_images(messages: List[ChatMessage]) -> List[Image.Image]:
    """Extract base64 or URL images from chat messages."""
    images = []
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    img_data = part.get("image_url", {}).get("url", "") or part.get("image", "")
                    if img_data.startswith("data:"):
                        # base64: data:image/png;base64,iVBOR...
                        b64 = img_data.split(",", 1)[-1]
                        img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                        images.append(img)
                    elif img_data.startswith(("http://", "https://")):
                        import requests as req
                        resp = req.get(img_data, timeout=10)
                        img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                        images.append(img)
                    elif img_data:
                        img = Image.open(img_data).convert("RGB")
                        images.append(img)
    return images


def _build_prompt(messages: List[ChatMessage], has_images: bool) -> str:
    """Extract text prompt from the last user message."""
    for msg in reversed(messages):
        if msg.role == "user":
            if isinstance(msg.content, str):
                return msg.content
            # Complex content: extract text parts
            texts = [p["text"] for p in msg.content
                     if isinstance(p, dict) and p.get("type") == "text"]
            return " ".join(texts) if texts else ""
    return ""


def _count_tokens(text: str) -> int:
    if _generator is None or not hasattr(_generator, "tokenizer"):
        return 0
    return len(_generator.tokenizer.encode(text, add_special_tokens=False))


def _submit_continuous_request(
    request_id: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
):
    if _continuous_scheduler is None or _continuous_backend is None:
        raise RuntimeError("continuous batching scheduler is not running")
    prompt_tokens = _continuous_backend.tokenize(prompt)
    try:
        return _continuous_scheduler.submit(
            request_id=request_id,
            prompt_tokens=prompt_tokens,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    except ValueError as error:
        raise HTTPException(400, str(error)) from error


async def _collect_continuous_request(batch_request):
    completion = ""
    while True:
        event = await asyncio.to_thread(batch_request.outputs.get)
        if event.error:
            raise RuntimeError(event.error)
        completion += event.delta
        if event.finished:
            return completion, event.finish_reason


async def _wait_continuous_chat(
    prompt: str, req: ChatCompletionRequest, rid: str
):
    batch_request = _submit_continuous_request(
        rid, prompt, req.temperature, req.top_p, req.max_tokens
    )
    try:
        completion, finish_reason = await _collect_continuous_request(
            batch_request
        )
    except Exception as error:
        raise HTTPException(500, str(error))

    prompt_tokens = _count_tokens(prompt)
    completion_tokens = len(batch_request.generated_token_ids)
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": completion},
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


async def _wait_continuous_completion(
    prompt: str, req: CompletionRequest, rid: str
):
    batch_request = _submit_continuous_request(
        rid, prompt, req.temperature, req.top_p, req.max_tokens
    )
    try:
        completion, finish_reason = await _collect_continuous_request(
            batch_request
        )
    except Exception as error:
        raise HTTPException(500, str(error))

    prompt_tokens = _count_tokens(prompt)
    completion_tokens = len(batch_request.generated_token_ids)
    return {
        "id": rid,
        "object": "text_completion",
        "created": int(time.time()),
        "model": _model_name,
        "choices": [{
            "index": 0,
            "text": completion,
            "finish_reason": finish_reason,
        }],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {"status": "ok", "model": _model_name, "vl": _is_vl}


@app.get("/v1/models")
async def list_models():
    return ModelList(data=[ModelInfo(id=_model_name)])


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest, raw: Request):
    if _generator is None:
        raise HTTPException(503, "Model not loaded")

    images = _extract_images(req.messages) if _is_vl else []
    prompt = _build_prompt(req.messages, bool(images))

    # Build ChatML prompt for Qwen3-instruct
    if not _is_vl:
        from lite_llama.utils.prompt_templates import get_prompter
        prompter = get_prompter("qwen3", "", enable_thinking=req.enable_thinking)
        prompter.insert_prompt(prompt)
        prompt = prompter.model_input

    request_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

    if _continuous_batching and not _is_vl:
        if req.stream:
            return StreamingResponse(
                _stream_continuous_chat(prompt, req, request_id),
                media_type="text/event-stream",
            )
        return await _wait_continuous_chat(prompt, req, request_id)

    if req.stream:
        return StreamingResponse(
            _stream_chat(prompt, images, req, request_id),
            media_type="text/event-stream",
        )
    else:
        return _sync_chat(prompt, images, req, request_id)


@app.post("/v1/completions")
async def completions(req: CompletionRequest, raw: Request):
    if _generator is None:
        raise HTTPException(503, "Model not loaded")

    prompts = [req.prompt] if isinstance(req.prompt, str) else req.prompt
    request_id = f"cmpl-{uuid.uuid4().hex[:12]}"

    if _continuous_batching and not _is_vl:
        if req.stream:
            return StreamingResponse(
                _stream_continuous_completion(
                    prompts[0], req, request_id
                ),
                media_type="text/event-stream",
            )
        return await _wait_continuous_completion(
            prompts[0], req, request_id
        )

    if req.stream:
        return StreamingResponse(
            _stream_completion(prompts[0], req, request_id),
            media_type="text/event-stream",
        )
    else:
        return _sync_completion(prompts[0], req, request_id)


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------
async def _stream_continuous_chat(
    prompt: str, req: ChatCompletionRequest, rid: str
) -> AsyncGenerator[str, None]:
    batch_request = _submit_continuous_request(
        rid, prompt, req.temperature, req.top_p, req.max_tokens
    )
    completion = ""
    finish_reason = "stop"
    try:
        while True:
            event = await asyncio.to_thread(batch_request.outputs.get)
            if event.error:
                raise RuntimeError(event.error)
            if event.delta:
                completion += event.delta
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": _model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": event.delta},
                        "finish_reason": None,
                    }],
                }
                yield (
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                )
            if event.finished:
                finish_reason = event.finish_reason or "stop"
                break

        final_chunk = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _model_name,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": finish_reason,
            }],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

        if req.stream_options and req.stream_options.include_usage:
            prompt_tokens = _count_tokens(prompt)
            completion_tokens = len(batch_request.generated_token_ids)
            usage_chunk = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": _model_name,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield (
                f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"
            )
        yield "data: [DONE]\n\n"
    except Exception as error:
        yield (
            f"data: {json.dumps({'error': str(error)}, ensure_ascii=False)}"
            "\n\n"
        )
    finally:
        if not batch_request.finished:
            batch_request.cancel()


async def _stream_continuous_completion(
    prompt: str, req: CompletionRequest, rid: str
) -> AsyncGenerator[str, None]:
    batch_request = _submit_continuous_request(
        rid, prompt, req.temperature, req.top_p, req.max_tokens
    )
    try:
        while True:
            event = await asyncio.to_thread(batch_request.outputs.get)
            if event.error:
                raise RuntimeError(event.error)
            if event.delta:
                chunk = {
                    "id": rid,
                    "object": "text_completion.chunk",
                    "created": int(time.time()),
                    "model": _model_name,
                    "choices": [{
                        "index": 0,
                        "text": event.delta,
                        "finish_reason": None,
                    }],
                }
                yield (
                    f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                )
            if event.finished:
                break
        yield "data: [DONE]\n\n"
    except Exception as error:
        yield (
            f"data: {json.dumps({'error': str(error)}, ensure_ascii=False)}"
            "\n\n"
        )
    finally:
        if not batch_request.finished:
            batch_request.cancel()


def _sync_chat(prompt: str, images: list, req: ChatCompletionRequest, rid: str):
    t0 = time.time()
    try:
        if _is_tp:
            completion = _tp_run_generation(
                prompt, images,
                temperature=req.temperature, top_p=req.top_p,
                max_tokens=req.max_tokens,
            )
        elif _is_vl:
            stream = _generator.text_completion_stream(
                [prompt], images,
                temperature=req.temperature, top_p=req.top_p,
                max_gen_len=req.max_tokens,
            )
            completion = ""
            for batch in stream:
                completion = batch[0].get("generation", "") if isinstance(batch[0], dict) else batch[0]
        else:
            stream = _generator.text_completion_stream(
                [prompt],
                temperature=req.temperature, top_p=req.top_p,
                max_gen_len=req.max_tokens,
            )
            completion = ""
            for batch in stream:
                completion = batch[0].get("generation", "") if isinstance(batch[0], dict) else batch[0]

        prompt_tokens = _count_tokens(prompt)
        completion_tokens = _count_tokens(completion)
        return {
            "id": rid,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": completion},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


def _tp_run_generation_stream(
    prompt: str, images: list,
    temperature: float, top_p: float, max_tokens: int,
):
    """TP streaming: broadcast params, both ranks generate, rank 0 yields tokens."""
    import threading

    params = [prompt, len(images), temperature, top_p, max_tokens]
    torch.distributed.broadcast_object_list(params, src=0)
    prompt, n_images, temperature, top_p, max_tokens = params
    images = images if _is_vl and n_images > 0 else []

    if _is_vl:
        stream = _generator.text_completion_stream(
            [prompt], images,
            temperature=temperature, top_p=top_p, max_gen_len=max_tokens,
        )
    else:
        stream = _generator.text_completion_stream(
            [prompt],
            temperature=temperature, top_p=top_p, max_gen_len=max_tokens,
        )

    for batch in stream:
        yield batch


async def _stream_chat(
    prompt: str, images: list, req: ChatCompletionRequest, rid: str
) -> AsyncGenerator[str, None]:
    t0 = time.time()
    try:
        if _is_tp:
            stream = _tp_run_generation_stream(
                prompt, images,
                temperature=req.temperature, top_p=req.top_p,
                max_tokens=req.max_tokens,
            )
        elif _is_vl:
            stream = _generator.text_completion_stream(
                [prompt], images,
                temperature=req.temperature, top_p=req.top_p,
                max_gen_len=req.max_tokens,
            )
        else:
            stream = _generator.text_completion_stream(
                [prompt],
                temperature=req.temperature, top_p=req.top_p,
                max_gen_len=req.max_tokens,
            )

        completion = ""
        for batch in stream:
            text = batch[0].get("generation", "") if isinstance(batch[0], dict) else batch[0]
            delta = text[len(completion):]
            completion = text

            chunk = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": _model_name,
                "choices": [{"index": 0, "delta": {"content": delta}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"

        # Final chunk
        final_chunk = {
            "id": rid,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": _model_name,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(final_chunk, ensure_ascii=False)}\n\n"

        if req.stream_options and req.stream_options.include_usage:
            prompt_tokens = _count_tokens(prompt)
            completion_tokens = _count_tokens(completion)
            usage_chunk = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": _model_name,
                "choices": [],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            }
            yield f"data: {json.dumps(usage_chunk, ensure_ascii=False)}\n\n"

        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


def _sync_completion(prompt: str, req: CompletionRequest, rid: str):
    try:
        stream = _generator.text_completion_stream(
            [prompt],
            temperature=req.temperature, top_p=req.top_p,
            max_gen_len=req.max_tokens,
        )
        completion = ""
        for batch in stream:
            completion = batch[0].get("generation", "") if isinstance(batch[0], dict) else batch[0]

        prompt_tokens = _count_tokens(prompt)
        completion_tokens = _count_tokens(completion)
        return {
            "id": rid,
            "object": "text_completion",
            "created": int(time.time()),
            "model": _model_name,
            "choices": [{"index": 0, "text": completion, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            },
        }
    except Exception as e:
        raise HTTPException(500, str(e))


async def _stream_completion(
    prompt: str, req: CompletionRequest, rid: str
) -> AsyncGenerator[str, None]:
    try:
        stream = _generator.text_completion_stream(
            [prompt],
            temperature=req.temperature, top_p=req.top_p,
            max_gen_len=req.max_tokens,
        )
        completion = ""
        for batch in stream:
            text = batch[0].get("generation", "") if isinstance(batch[0], dict) else batch[0]
            delta = text[len(completion):]
            completion = text
            chunk = {
                "id": rid,
                "object": "text_completion.chunk",
                "created": int(time.time()),
                "model": _model_name,
                "choices": [{"index": 0, "text": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="lite_llama OpenAI-compatible API server")
    parser.add_argument("--checkpoints_dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default=None,
                        help="Device override (auto-detected if not set). "
                             "For TP, this is auto-set by torchrun.")
    parser.add_argument("--page_size", type=int, default=16,
                        help="PagedAttention page size; use 0 to disable.")
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help=(
            "Maximum per-request model context length. This must cover "
            "prompt tokens plus generated tokens after chat-template "
            "expansion."
        ),
    )
    parser.add_argument(
        "--compiled_model",
        dest="compiled_model",
        action="store_true",
        help="Enable NPU Graph path (default).",
    )
    parser.add_argument(
        "--no_compiled_model",
        dest="compiled_model",
        action="store_false",
        help="Disable NPU Graph path.",
    )
    parser.add_argument(
        "--continuous_batching",
        dest="continuous_batching",
        action="store_true",
        help="Enable text continuous batching (default).",
    )
    parser.add_argument(
        "--no_continuous_batching",
        dest="continuous_batching",
        action="store_false",
        help="Use the legacy request-at-a-time generation path.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=32,
        help="Maximum number of active continuous-batching requests.",
    )
    parser.add_argument(
        "--max_waiting_requests",
        type=int,
        default=1024,
        help="Maximum number of queued requests.",
    )
    parser.add_argument(
        "--scheduler_poll_ms",
        type=float,
        default=1.0,
        help="Idle scheduler polling interval in milliseconds.",
    )
    parser.add_argument(
        "--max_prefill_tokens",
        type=int,
        default=None,
        help=(
            "Optional continuous-batching prefill token budget per "
            "scheduler tick. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--max_decode_tokens",
        type=int,
        default=None,
        help=(
            "Optional continuous-batching decode row budget per "
            "scheduler tick. Disabled when omitted."
        ),
    )
    parser.add_argument(
        "--partial_prefix_cache",
        action="store_true",
        help=(
            "Enable page-aligned partial Prefix Cache reuse. Disabled by "
            "default because suffix replay is correctness-first and can slow "
            "random prompt benchmarks."
        ),
    )
    parser.add_argument(
        "--chunked_prefill",
        action="store_true",
        help=(
            "Enable chunked prefill execution across scheduler ticks. "
            "v0.0.8 batches chunk replay across active prefill requests; "
            "full prompt-cache attention kernels remain future work."
        ),
    )
    parser.add_argument(
        "--prefill_chunk_size",
        type=int,
        default=None,
        help="Chunk size used when --chunked_prefill is enabled.",
    )
    parser.add_argument(
        "--max_preemptions",
        type=int,
        default=1,
        help=(
            "Maximum KV-pressure preemptions per request in continuous "
            "batching. Set 0 to fail instead of preempting."
        ),
    )
    parser.add_argument(
        "--moe_parallel_mode",
        choices=("tp", "ep"),
        default="tp",
        help=(
            "MoE expert execution mode. Ignored by dense and VL models."
        ),
    )
    parser.set_defaults(compiled_model=True)
    parser.set_defaults(continuous_batching=True)
    args = parser.parse_args()

    # Detect TP
    from lite_llama.executor.tp_utils import detect_tp_env
    global _rank, _is_tp, _continuous_batching, _partial_prefix_cache
    tp = detect_tp_env()
    _rank = tp.rank if tp else 0
    _is_tp = tp is not None and tp.enabled
    _continuous_batching = args.continuous_batching
    _partial_prefix_cache = bool(args.partial_prefix_cache)

    device = f"npu:{_rank}" if _is_tp else get_device(args.device)
    if _rank == 0:
        print(f"Loading model from {args.checkpoints_dir}")
        print(f"Device: {device}, TP: world_size={tp.world_size if _is_tp else 1}")
        print(f"Max seq len: {args.max_seq_len}")
        print(f"PagedAttention page_size: {args.page_size}")
        print(f"NPU Graph: {'on' if args.compiled_model else 'off'}")
        print(f"MoE parallel mode: {args.moe_parallel_mode.upper()}")
        print(f"Partial Prefix Cache: {'on' if args.partial_prefix_cache else 'off'}")

    load_generator(
        args.checkpoints_dir,
        device,
        max_seq_len=args.max_seq_len,
        page_size=args.page_size,
        compiled_model=args.compiled_model,
        moe_parallel_mode=args.moe_parallel_mode,
    )

    if _is_vl and _continuous_batching:
        if _rank == 0:
            print(
                "Continuous batching is unavailable for vision models; "
                "using the legacy request path."
            )
        _continuous_batching = False
    if _continuous_batching and args.page_size <= 0:
        raise RuntimeError(
            "Continuous batching requires PagedAttention; set --page_size "
            "to a positive value."
        )

    if _rank == 0:
        if _continuous_batching:
            _start_continuous_scheduler(
                max_batch_size=args.max_batch_size,
                max_waiting_requests=args.max_waiting_requests,
                scheduler_poll_ms=args.scheduler_poll_ms,
                max_prefill_tokens=args.max_prefill_tokens,
                max_decode_tokens=args.max_decode_tokens,
                chunked_prefill=args.chunked_prefill,
                prefill_chunk_size=args.prefill_chunk_size,
                max_preemptions=args.max_preemptions,
                partial_prefix_cache=args.partial_prefix_cache,
            )
        print(f"Server starting on http://{args.host}:{args.port}")
        print(f"Endpoints:")
        print(f"  POST /v1/chat/completions")
        print(f"  POST /v1/completions")
        print(f"  GET  /v1/models")
        print(f"  GET  /health")
        if _continuous_batching:
            effective_max_prefill_tokens = getattr(
                _continuous_scheduler, "max_prefill_tokens", args.max_prefill_tokens
            )
            print(
                "  [Continuous batching: "
                f"max_batch_size={args.max_batch_size}, "
                f"max_prefill_tokens={effective_max_prefill_tokens}, "
                f"max_decode_tokens={args.max_decode_tokens}, "
                f"chunked_prefill={args.chunked_prefill}, "
                f"max_preemptions={args.max_preemptions}]"
            )
        elif _is_tp:
            print(f"  [TP mode: single-request-at-a-time]")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    else:
        # Non-rank-0: TP worker loop — mirrors rank 0 generation
        if _continuous_batching:
            _tp_continuous_worker_loop()
        else:
            _tp_worker_loop()


if __name__ == "__main__":
    main()
