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
import base64
import io
import json
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


def load_generator(
    checkpoints_dir: str,
    device: str,
    *,
    page_size: int = 16,
    compiled_model: bool = True,
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
            device=device,
        )
    else:
        from lite_llama.generate_stream import GenerateStreamText
        _generator = GenerateStreamText(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            compiled_model=compiled_model,
            page_size=page_size,
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


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: generator is loaded by main() before uvicorn
    yield
    # Shutdown
    global _generator
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
    parser.set_defaults(compiled_model=True)
    args = parser.parse_args()

    # Detect TP
    from lite_llama.executor.tp_utils import detect_tp_env
    global _rank, _is_tp
    tp = detect_tp_env()
    _rank = tp.rank if tp else 0
    _is_tp = tp is not None and tp.enabled

    device = f"npu:{_rank}" if _is_tp else get_device(args.device)
    if _rank == 0:
        print(f"Loading model from {args.checkpoints_dir}")
        print(f"Device: {device}, TP: world_size={tp.world_size if _is_tp else 1}")
        print(f"PagedAttention page_size: {args.page_size}")
        print(f"NPU Graph: {'on' if args.compiled_model else 'off'}")

    load_generator(
        args.checkpoints_dir,
        device,
        page_size=args.page_size,
        compiled_model=args.compiled_model,
    )

    if _rank == 0:
        print(f"Server starting on http://{args.host}:{args.port}")
        print(f"Endpoints:")
        print(f"  POST /v1/chat/completions")
        print(f"  POST /v1/completions")
        print(f"  GET  /v1/models")
        print(f"  GET  /health")
        if _is_tp:
            print(f"  [TP mode: single-request-at-a-time]")
        uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
    else:
        # Non-rank-0: TP worker loop — mirrors rank 0 generation
        _tp_worker_loop()


if __name__ == "__main__":
    main()
