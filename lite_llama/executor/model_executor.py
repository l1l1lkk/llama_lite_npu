"""Model loading and execution orchestration for lite_llama.

``ModelExecutor`` owns model construction, tensor-parallel weight sharding,
KV-cache allocation, request-to-page metadata, Prefill/Decode activation, and
optional NPU Graph replay. The executor is the boundary between model-agnostic
scheduling and model-specific forward implementations.

Correctness depends on keeping request token counts, page-table state, and TP
rank execution order identical across all workers.
"""

import torch
import torch.nn as nn

import json, time
from collections import OrderedDict
from pathlib import Path

from transformers import LlavaConfig
from accelerate import init_empty_weights, load_checkpoint_and_dispatch

from .mem_manager import ComputeMaxAvailableBlocks, KVCacheMemoryManager
from .req_tokens_manager import ReqTokensManager
from .paged_attention import PagedKVCacheManager, PagedReqTokensManager

from .cuda_graph import ModelRunner
from .executor_struct import AttentionInfo, CONFIG_CLASS_MAP
from ..models.model_config import LlamaConfig, Qwen3MoeConfig, Qwen3VLConfig
from ..kernels import update_kv_index
from ..utils.device import get_device
from ..utils.logger import get_logger
from .tp_utils import (
    TPConfig, init_tp, detect_tp_env, get_tp_config,
    shard_attention_q, shard_attention_kv, shard_attention_o,
    shard_ffn_gate_up, shard_ffn_down, shard_lm_head,
    prepare_moe_gate_up_for_gmm, prepare_moe_down_for_gmm,
    prepare_moe_gate_up_for_ep, prepare_moe_down_for_ep,
)

logger = get_logger(__name__)

# -----------------------------------------------------------------------------
# Registry helpers (avoid long if/elif chains)
# -----------------------------------------------------------------------------

class ModelExecutor:
    # 定义类属性
    model_config = None
    model = None
    atten_info = AttentionInfo

    # 通过静态方法 build 将类属性当作默认配置使用
    @staticmethod
    def build(
        checkpoints_dir: str,
        max_seq_len: int,
        max_gpu_num_blocks: None,
        compiled_model: bool = False,
        page_size: int | None = None,
        moe_parallel_mode: str = "tp",
        device: str = None,
    ):
        """
        构建 ModelExecutor 实例, 加载模型、分词器和初始化推理信息结构体 atten_info。

        参数:
            checkpoints_dir (str): 模型检查点目录路径。
            load_model (bool): 是否加载模型权重。
            max_seq_len (int): 最大序列长度。
            device (str): 设备类型（'npu:6'、'cuda'或'cpu'），None 为自动检测。

        返回:
            ModelExecutor: 初始化后的 ModelExecutor 实例。
        """
        device = get_device(device)

        # --- Tensor Parallelism init ---
        tp_config = detect_tp_env()
        if tp_config is not None:
            logger.info("TP initialized: world_size=%d rank=%d", tp_config.world_size, tp_config.rank)
            device = f"{'npu' if tp_config.is_npu else 'cuda'}:{tp_config.rank}"
        else:
            tp_config = TPConfig()
        if moe_parallel_mode not in {"tp", "ep"}:
            raise ValueError(
                "moe_parallel_mode must be 'tp' or 'ep', got "
                f"{moe_parallel_mode!r}"
            )
        tp_config.moe_parallel_mode = moe_parallel_mode

        # Set as current device before any allocations
        if "npu" in device:
            torch.npu.set_device(device)
        elif "cuda" in device and device != "cuda":
            torch.cuda.set_device(device)

        model_config = ModelExecutor._load_model_config(checkpoints_dir, max_seq_len)
        if page_size is not None:
            if isinstance(model_config, Qwen3VLConfig):
                model_config.text_config.page_size = page_size
            else:
                model_config.page_size = page_size
        model = ModelExecutor._load_model_weight(
            model_config, checkpoints_dir, device=device, tp_config=tp_config,
        )

        return ModelExecutor(
            checkpoints_dir, model_config, model, max_gpu_num_blocks,
            compiled_model, device, tp_config,
        )

    @staticmethod
    def _load_model_config(checkpoints_dir: str, max_seq_len: int):
        cfg_path = Path(checkpoints_dir) / "config.json"
        if not cfg_path.exists():
            raise FileNotFoundError(f"{cfg_path} not found")

        params = json.loads(cfg_path.read_text())
        cfg_cls = CONFIG_CLASS_MAP.get(params["model_type"].lower())
        
        if cfg_cls is None:
            raise ValueError(f"Unsupported model_type {params['model_type']!r}")
        
        model_config = cfg_cls.from_dict(params)
        model_config.max_seq_len = max_seq_len
        return model_config
    
    @staticmethod
    def _accelerate_load_weight(
        model_config,
        checkpoints_dir,
        device=None,
    ):
        device = get_device(device)
        with init_empty_weights():
            model = ModelExecutor._initialize_model(model_config, device=device)
        model = load_checkpoint_and_dispatch(
            model, checkpoints_dir, device_map="auto", dtype=torch.float16
        )

        # 将模型转换为半精度, 并验证抓换
        model.to(device)
        model.half()
        for param in model.parameters():
            assert param.dtype == torch.float16, "Model parameters are not in FP16"
        logger.info("Converted model to half precision (FP16)")

        return model

    @staticmethod
    def _load_model_weight(
        model_config,
        checkpoints_dir,
        device=None,
        tp_config=None,
    ):
        device = get_device(device)
        tp = tp_config or TPConfig()

        # Set as current device before any allocations
        if "npu" in device:
            torch.npu.set_device(device)
        elif "cuda" in device and device != "cuda":
            torch.cuda.set_device(device)

        start_time = time.time()

        # 初始化模型（TP 感知：sharded weight shapes）
        with init_empty_weights():
            model = ModelExecutor._initialize_model(model_config, device=device, tp_config=tp)
            state_dict = None

        checkpoints = sorted(Path(checkpoints_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {checkpoints_dir}"
        ckpt_path = str(checkpoints[0])
        logger.info(f'Loading checkpoint "{ckpt_path}"')
        # Load to CPU first: avoid NPU OOM when full model > single card capacity
        # TP slicing happens on CPU, then each shard is moved to device via load_state_dict(assign=True)
        state_dict = torch.load(ckpt_path, mmap=True, map_location="cpu")

        # --- TP weight sharding (on CPU) ---
        if tp.enabled:
            logger.info("Sharding weights for TP (rank=%d/%d)", tp.rank, tp.world_size)
        if tp.enabled or model_config.model_type.lower() == "qwen3_moe":
            num_layers = _get_num_layers_from_config(model_config)
            state_dict = _shard_state_dict(state_dict, num_layers, tp, model_config)

        # Load sharded weights into model (from CPU → NPU via assign=True)
        model.load_state_dict(state_dict, strict=True, assign=True)
        model.to(device).half()
        model.eval()
        logger.info(f"Loaded state dict in {time.time() - start_time:.2f}s")

        return model

    @staticmethod
    def _initialize_model(model_config, device: str, tp_config: TPConfig = None) -> nn.Module:
        model_type = model_config.model_type.lower()
        logger.info(
            f"Initializing model of type '{model_type}' to device '{device}' "
            f"(TP world_size={tp_config.world_size if tp_config else 1})"
        )
        if model_type == "llama":
            from ..models.llama import LlamaModel
            model = LlamaModel(model_config)
        elif model_type == "qwen2":
            from ..models.qwen2 import Qwen2Model
            model = Qwen2Model(model_config)
        elif model_type == "qwen3":
            from ..models.qwen3 import Qwen3Model
            model = Qwen3Model(model_config, tp_config=tp_config)
        elif model_type == "qwen3_moe":
            from ..models.qwen3_moe import Qwen3MoeModel
            model = Qwen3MoeModel(model_config, tp_config=tp_config)
        elif model_type == "llava":
            from ..models.llava import LlavaLlama
            model = LlavaLlama(model_config)
        elif model_type == "qwen3_vl":
            from ..models.qwen3vl import Qwen3VLModel
            model = Qwen3VLModel(model_config, tp_config=tp_config)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        logger.info(f"Model initialized on device '{device}'")
        return model

    def __init__(
        self,
        checkpoints_dir: str,
        model_config,
        model,
        max_gpu_num_blocks=None,
        compiled_model=False,
        device=None,
        tp_config: TPConfig = None,
    ):
        self.tp = tp_config or TPConfig()
        self.device = get_device(device)
        self.checkpoints_dir = checkpoints_dir
        self.model_config = model_config
        if isinstance(model_config, LlavaConfig):
            self.llm_config = LlamaConfig.from_dict(model_config.text_config.to_dict())
        elif isinstance(model_config, Qwen3VLConfig):
            self.llm_config = model_config.text_config
        else:
            self.llm_config = model_config

        # KV heads are sharded under TP
        self.local_q_heads = self.llm_config.num_heads // self.tp.world_size
        self.local_kv_heads = self.llm_config.num_kv_heads // self.tp.world_size

        self.max_seq_len = self.llm_config.max_seq_len
        self.max_prefill_tokens = self._infer_safe_prefill_tokens()
        self.model_type = model_config.model_type
        self.model = model
        self.logits_are_sharded = (
            self.tp.enabled
            and self.model_type in {"qwen3", "qwen3_moe", "qwen3_vl"}
        )
        self.model_runner = None
        self.compiled_model = compiled_model
        self.page_size = getattr(self.llm_config, "page_size", 0)
        self.use_paged_attn = self.page_size > 0

        if max_gpu_num_blocks:
            self.kv_mem_manager = self._init_mem_manager(max_gpu_num_blocks, device=self.device)
            self.max_gpu_num_tokens = max_gpu_num_blocks
        else:
            max_gpu_num_blocks, self.max_gpu_num_tokens = (
                self._get_max_avaliable_tokens(model,gpu_memory_utilization=0.9, block_size=1)
            )
            self.kv_mem_manager = self._init_mem_manager(
                max_gpu_num_blocks, block_size=1, device=self.device
            )

        self.max_request_num = max_gpu_num_blocks // self.max_seq_len

        if self.use_paged_attn:
            self.req_tokens_manager = PagedReqTokensManager(
                self.max_request_num, self.max_seq_len, self.kv_mem_manager, device=self.device
            )
        else:
            self.req_tokens_manager = ReqTokensManager(
                self.max_request_num, self.max_seq_len, device=self.device
            )
        self.atten_info = AttentionInfo()  # 创建 AttentionInfo 实例
        self.atten_info.kv_buffer = self.kv_mem_manager.gpu_kv_buffer
        self.atten_info.b_req_tokens_table = self.req_tokens_manager.b_req_tokens_table
        # Paged KV allocation is host-managed. Cache request ids once during
        # prefill so decode does not synchronize an NPU tensor via .tolist()
        # for every generated token.
        self._paged_request_ids: tuple[int, ...] = ()
        self._paged_prefix_cache: OrderedDict[
            tuple[int, ...], tuple[object, int, int]
        ] = OrderedDict()
        self._paged_prefix_cache_max_entries = 128
        # Block-level Prefix Cache: keys form a parent-fingerprint chain over
        # full KV pages.  Partial-prefix lookup is O(prompt_blocks) and never
        # scans all cached full prompts. Only complete blocks are cached/reused.
        self._paged_block_prefix_cache: OrderedDict[
            tuple[int | None, tuple[int, ...]], tuple[object, int]
        ] = OrderedDict()
        self._paged_block_prefix_cache_max_entries = 4096

        # --- NPU Graph (decode kernel launch batching) ---
        self.graph_runner = None
        if self.compiled_model:
            from .npu_graph import supports_decode_graph

            if supports_decode_graph(
                self.model_type,
                moe_parallel_mode=self.tp.moe_parallel_mode,
            ):
                self.apply_npu_graph()
            else:
                logger.warning(
                    "NPU Graph is unavailable for model_type=%s "
                    "moe_parallel_mode=%s; using eager decode.",
                    self.model_type,
                    self.tp.moe_parallel_mode,
                )

    def _infer_safe_prefill_tokens(self) -> int | None:
        """Return a conservative packed-prefill token budget.

        The fused Q/K RMSNorm + RoPE kernel launches one program per token,
        instead of one RMSNorm program per token/head plus one RoPE program per
        token.  The grid limit therefore applies directly to packed tokens.
        """
        local_q_heads = int(getattr(self, "local_q_heads", 0) or 0)
        if local_q_heads <= 0:
            return None
        safe_grid_rows = 60000
        return safe_grid_rows

    def _get_max_avaliable_tokens(self,model, gpu_memory_utilization=0.9, block_size=1):
        avaliable_blocks = ComputeMaxAvailableBlocks(
            num_layers=self.llm_config.num_layers,
            hidden_size=self.llm_config.hidden_size,
            num_heads=self.llm_config.num_heads // self.tp.world_size,
            num_kv_heads=self.local_kv_heads,
            head_dim=self.llm_config.head_dim,
            gpu_memory_utilization=gpu_memory_utilization,
            block_size=block_size,
            device=self.device,
        )
        max_gpu_num_blocks = avaliable_blocks.compute_num_available_blocks(model, model_path=self.checkpoints_dir)
        max_gpu_num_tokens = max_gpu_num_blocks * block_size

        return max_gpu_num_blocks, max_gpu_num_tokens

    def _init_mem_manager(
        self, gpu_num_blocks, block_size=1, dtype=torch.float16, device=None
    ):
        if self.use_paged_attn:
            kv_mem_manager = PagedKVCacheManager(
                num_layers=self.llm_config.num_layers,
                num_kv_heads=self.local_kv_heads,
                head_dim=self.llm_config.head_dim,
                num_pages=max(1, gpu_num_blocks // self.page_size),
                page_size=self.page_size,
                dtype=dtype,
                device=device,
            )
        else:
            kv_mem_manager = KVCacheMemoryManager(
                num_layers=self.llm_config.num_layers,
                num_kv_heads=self.local_kv_heads,
                head_dim=self.llm_config.head_dim,
                gpu_num_blocks=gpu_num_blocks,
                block_size=block_size,
                dtype=dtype,
                device=device,
            )

        return kv_mem_manager

    def apply_npu_graph(self):
        """Apply NPU graph for decode phase (kernel launch batching)."""
        from .npu_graph import NpuGraphRunner
        self.graph_runner = NpuGraphRunner(
            self.model, model_type=self.model_type
        )
        logger.info("NPU Graph runner created (available=%s)", self.graph_runner.available)

    def init_req_to_tokens_table(
        self, b_req_tokens_table, b_req_idx, b_seq_len, alloc_mem_index
    ):
        """
        初始化 prefill 阶段已分配的批次请求项的 kv cache 所用 tokens 索引
        """
        # TODO: 性能等待优化
        start_index = 0
        batch_size = len(b_seq_len)
        b_seq_len_numpy = b_seq_len.cpu().numpy()
        b_req_idx_numpy = b_req_idx.cpu().numpy()
        b_start_loc = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)
        for i in range(batch_size):
            if i > 0:
                b_start_loc[i] = start_index
            cur_seq_len = b_seq_len_numpy[i]
            b_req_tokens_table[b_req_idx_numpy[i], :cur_seq_len] = alloc_mem_index[
                start_index : start_index + cur_seq_len
            ]
            start_index += cur_seq_len

        return b_start_loc

    def prefill_alloc_kv_cache(
        self,
        max_prompt_len,
        actual_prompt_lens,
        b_req_idx,
        image_batch_size=None,
        debug_mode=False,
    ):
        """
        start_index:        tensor([  0, 270, 540, 810], device='cuda:0', dtype=torch.int32)
        b_seq_len:          tensor([14, 12, 11, 11], device='cuda:0')
        Prefill Stage, cur_select_index: tensor([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
                                    270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283,
                                    540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 552, 553,
                                    810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 821, 822, 823
                                ], device='cuda:0')
        Decode Stage, 0 step, cur_select_index: tensor([ 14, 282, 551, 821], device='cuda:0'), cur_b_seq_len: tensor([15, 13, 12, 12], device='cuda:0')
        Decode Stage, 1 step, cur_select_index: tensor([ 15, 283, 552, 822], device='cuda:0'), cur_b_seq_len: tensor([16, 14, 13, 13], device='cuda:0')
        """
        num_patch_indexs = None
        batch_size = len(actual_prompt_lens)
        self.atten_info.b_req_idx = b_req_idx

        if image_batch_size is not None:
            image_size = self.model_config.vision_config.image_size
            pathch_size = self.model_config.vision_config.patch_size
            number_patchs = image_size // pathch_size
            num_patch_indexs = number_patchs * number_patchs - 1
            max_prompt_len += num_patch_indexs
            actual_prompt_lens += num_patch_indexs
            print(f"num_patch_indexs: {num_patch_indexs}")

        context_num_tokens = max_prompt_len * batch_size
        if self.use_paged_attn:
            self._paged_request_ids = tuple(
                int(req_idx)
                for req_idx in b_req_idx.detach().cpu().tolist()
            )
            for req_idx in self._paged_request_ids:
                ok = self.req_tokens_manager.alloc_req(req_idx, max_prompt_len)
                if not ok:
                    raise RuntimeError("Paged KV allocation failed during prefill")
            self.atten_info.cur_select_index = torch.cat(
                [
                    self.req_tokens_manager.get_token_indices(req_idx, max_prompt_len)
                    for req_idx in self._paged_request_ids
                ]
            ).to(torch.int32)
        else:
            self._paged_request_ids = ()
            self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(
                context_num_tokens
            )
        # 初始化每个批次项的实际提示词长度
        self.atten_info.b_seq_len = actual_prompt_lens  # 张量, 形状 [batch_size, 1]
        # 初始化批次请求的当前最大序列上下文长度(对应 kv cache 长度)
        self.atten_info.max_actual_seq_len = max_prompt_len  # int 类型

        if self.use_paged_attn:
            self.atten_info.b_start_loc = torch.arange(
                0,
                batch_size * max_prompt_len,
                max_prompt_len,
                dtype=torch.int32,
                device=self.device,
            )
        else:
            self.atten_info.b_start_loc = self.init_req_to_tokens_table(
                self.atten_info.b_req_tokens_table,
                self.atten_info.b_req_idx,
                self.atten_info.b_seq_len,
                self.atten_info.cur_select_index,
            )

        if debug_mode:
            print(
                f"context_num_tokens: {context_num_tokens}, max_prompt_len:{max_prompt_len}, \n \
                    self.atten_info.cur_select_index: {self.atten_info.cur_select_index},\n \
                    self.atten_info.max_actual_seq_len: {self.atten_info.max_actual_seq_len},\n \
                    self.atten_info.b_seq_len: {self.atten_info.b_seq_len}, \n \
                    self.atten_info.b_start_loc: {self.atten_info.b_start_loc}, "
            )

        return self.atten_info.cur_select_index, num_patch_indexs

    def reserve_paged_requests(
        self,
        prompt_lengths: list[int] | tuple[int, ...],
        reserved_lengths: list[int] | tuple[int, ...] | None = None,
    ) -> tuple[int, ...]:
        """Reserve independent request ids for continuous batching."""
        if not self.use_paged_attn:
            raise RuntimeError(
                "continuous batching requires PagedAttention"
            )
        if reserved_lengths is not None and len(reserved_lengths) != len(prompt_lengths):
            raise ValueError("reserved_lengths must match prompt_lengths")
        request_ids: list[int] = []
        try:
            for index, prompt_length in enumerate(prompt_lengths):
                reserved_length = (
                    None
                    if reserved_lengths is None
                    else int(reserved_lengths[index])
                )
                req_idx = self.req_tokens_manager.reserve_req(
                    int(prompt_length), reserved_length
                )
                if req_idx is None:
                    raise RuntimeError(
                        "Paged KV request or page capacity is exhausted"
                    )
                request_ids.append(req_idx)
        except BaseException:
            for req_idx in request_ids:
                self.req_tokens_manager.free_req(req_idx)
            raise
        return tuple(request_ids)

    def ensure_paged_request_capacity(
        self, req_idx: int, total_tokens: int
    ) -> None:
        """Reserve physical KV pages without changing logical seq length."""
        if not self.use_paged_attn:
            return
        if not self.req_tokens_manager.ensure_req_capacity(req_idx, total_tokens):
            raise RuntimeError(
                f"Paged KV capacity reservation failed for request {req_idx}"
            )

    def _reset_paged_chunk_prefill_metadata(self) -> None:
        self.atten_info.is_paged_chunk_prefill = False
        self.atten_info.chunk_context_len = None
        self.atten_info.chunk_q_seq_len = None
        self.atten_info.max_actual_q_seq_len = None

    def share_paged_request_from_cache(
        self, prompt_tokens: list[int] | tuple[int, ...]
    ) -> tuple[int, int] | None:
        """Share a full-prompt cached KV prefix and return (request_id, token).

        The cache intentionally starts with exact full-prompt matches only. It
        avoids unsafe suffix-prefill semantics while still making repeated
        prompts skip prefill forward entirely.
        """
        if not self.use_paged_attn:
            return None
        cached = self._paged_prefix_cache.get(tuple(int(t) for t in prompt_tokens))
        if cached is None:
            return None
        self._paged_prefix_cache.move_to_end(tuple(int(t) for t in prompt_tokens))
        page_indices, num_tokens, first_token_id = cached
        req_idx = self.req_tokens_manager.reserve_shared_req(
            page_indices, num_tokens
        )
        if req_idx is None:
            return None
        return int(req_idx), int(first_token_id)

    def share_paged_prefix_from_cache(
        self, prompt_tokens: list[int] | tuple[int, ...]
    ) -> tuple[int, int, int | None] | None:
        """Share the longest cached page-aligned prefix for a prompt.

        Returns ``(request_id, matched_tokens, first_token_id)``.  For exact
        prompt hits, ``first_token_id`` is the cached first generated token and
        the caller can skip prefill entirely.  For partial prefix hits,
        ``first_token_id`` is ``None`` and the caller must replay only the
        suffix tokens.
        """
        if not self.use_paged_attn:
            return None
        prompt_key = tuple(int(t) for t in prompt_tokens)
        if not prompt_key:
            return None

        exact = self._paged_prefix_cache.get(prompt_key)
        if exact is not None:
            self._paged_prefix_cache.move_to_end(prompt_key)
            page_indices, num_tokens, first_token_id = exact
            req_idx = self.req_tokens_manager.reserve_shared_req(
                page_indices, num_tokens
            )
            if req_idx is None:
                return None
            return int(req_idx), len(prompt_key), int(first_token_id)

        matched_pages = []
        matched_tokens = 0
        parent_fingerprint = None
        page_size = int(self.page_size)
        for start in range(0, len(prompt_key) - page_size + 1, page_size):
            block_tokens = prompt_key[start : start + page_size]
            block_key = self._paged_block_key(parent_fingerprint, block_tokens)
            cached_block = self._paged_block_prefix_cache.get(block_key)
            if cached_block is None:
                break
            self._paged_block_prefix_cache.move_to_end(block_key)
            page_index, token_count = cached_block
            matched_pages.append(page_index.reshape(-1))
            matched_tokens = int(token_count)
            parent_fingerprint = hash(block_key)

        if not matched_pages or matched_tokens <= 0:
            return None
        best_pages = torch.cat(matched_pages).to(device="cpu", dtype=torch.long)
        req_idx = self.req_tokens_manager.reserve_shared_req(best_pages, matched_tokens)
        if req_idx is None:
            return None
        return int(req_idx), int(matched_tokens), None

    def _paged_block_key(
        self, parent_fingerprint: int | None, block_tokens: list[int] | tuple[int, ...]
    ) -> tuple[int | None, tuple[int, ...]]:
        return parent_fingerprint, tuple(int(token) for token in block_tokens)

    def _store_paged_prompt_blocks(
        self, prompt_tokens: tuple[int, ...], page_indices
    ) -> None:
        page_size = int(self.page_size)
        full_blocks = len(prompt_tokens) // page_size
        if full_blocks <= 0:
            return
        parent_fingerprint = None
        for block_index in range(full_blocks):
            start = block_index * page_size
            block_tokens = prompt_tokens[start : start + page_size]
            block_key = self._paged_block_key(parent_fingerprint, block_tokens)
            if block_key in self._paged_block_prefix_cache:
                self._paged_block_prefix_cache.move_to_end(block_key)
                parent_fingerprint = hash(block_key)
                continue
            page_index = page_indices[block_index : block_index + 1].clone()
            self.req_tokens_manager.page_mgr.add_ref(page_index)
            self._paged_block_prefix_cache[block_key] = (
                page_index,
                (block_index + 1) * page_size,
            )
            self._paged_block_prefix_cache.move_to_end(block_key)
            parent_fingerprint = hash(block_key)
            while (
                len(self._paged_block_prefix_cache)
                > self._paged_block_prefix_cache_max_entries
            ):
                _, (old_page, _) = self._paged_block_prefix_cache.popitem(last=False)
                self.req_tokens_manager.page_mgr.free(old_page)

    def store_paged_request_prefix(
        self,
        prompt_tokens: list[int] | tuple[int, ...],
        req_idx: int,
        first_token_id: int,
    ) -> None:
        """Retain one KV page reference for an exact prompt cache entry."""
        if not self.use_paged_attn:
            return
        if req_idx not in self.req_tokens_manager.req_page_table:
            return
        key = tuple(int(t) for t in prompt_tokens)
        if key in self._paged_prefix_cache:
            return
        page_indices = self.req_tokens_manager.req_page_table[int(req_idx)]
        self.req_tokens_manager.page_mgr.add_ref(page_indices)
        self._paged_prefix_cache[key] = (
            page_indices.clone(),
            int(self.req_tokens_manager.req_token_count[int(req_idx)]),
            int(first_token_id),
        )
        self._paged_prefix_cache.move_to_end(key)
        self._store_paged_prompt_blocks(key, page_indices)
        while len(self._paged_prefix_cache) > self._paged_prefix_cache_max_entries:
            _, (old_pages, _, _) = self._paged_prefix_cache.popitem(last=False)
            self.req_tokens_manager.page_mgr.free(old_pages)

    def activate_paged_prefill_batch(
        self, request_ids: tuple[int, ...], prompt_length: int
    ) -> None:
        """Select an equal-length request group for one prefill forward."""
        self._reset_paged_chunk_prefill_metadata()
        if not request_ids:
            raise ValueError("request_ids must not be empty")
        for req_idx in request_ids:
            actual_length = self.req_tokens_manager.req_token_count[req_idx]
            if actual_length != prompt_length:
                raise ValueError(
                    "continuous prefill groups must have equal prompt "
                    f"lengths: expected={prompt_length}, "
                    f"request={req_idx}, actual={actual_length}"
                )

        self._paged_request_ids = request_ids
        self.atten_info.b_req_idx = torch.tensor(
            request_ids, dtype=torch.int32, device=self.device
        )
        self.atten_info.b_seq_len = torch.full(
            (len(request_ids),),
            prompt_length,
            dtype=torch.long,
            device=self.device,
        )
        self.atten_info.cur_select_index = torch.cat(
            [
                self.req_tokens_manager.get_token_indices(
                    req_idx, prompt_length
                )
                for req_idx in request_ids
            ]
        ).to(torch.int32)
        self.atten_info.b_start_loc = torch.arange(
            0,
            len(request_ids) * prompt_length,
            prompt_length,
            dtype=torch.int32,
            device=self.device,
        )
        self.atten_info.max_actual_seq_len = prompt_length

    def activate_paged_packed_prefill_batch(
        self, request_ids: tuple[int, ...], prompt_lengths: list[int] | tuple[int, ...]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select a mixed-length request group for one no-padding prefill.

        The model receives flattened tensors with shape ``[1, total_tokens]``.
        Attention kernels still see the logical request batch through
        ``b_start_loc`` and ``b_seq_len``.  The returned tensors are the flat
        position ids and the flat logits indices that correspond to the last
        prompt token of each logical request.
        """
        self._reset_paged_chunk_prefill_metadata()
        if not request_ids:
            raise ValueError("request_ids must not be empty")
        if len(request_ids) != len(prompt_lengths):
            raise ValueError("request_ids and prompt_lengths must have same length")

        lengths = tuple(int(length) for length in prompt_lengths)
        if any(length <= 0 for length in lengths):
            raise ValueError("prompt_lengths must be positive")
        for req_idx, prompt_length in zip(request_ids, lengths):
            actual_length = self.req_tokens_manager.req_token_count[int(req_idx)]
            if actual_length != prompt_length:
                raise ValueError(
                    "packed prefill request length mismatch: "
                    f"request={req_idx}, expected={prompt_length}, "
                    f"actual={actual_length}"
                )

        starts: list[int] = []
        sample_indices: list[int] = []
        position_ids: list[int] = []
        cursor = 0
        for prompt_length in lengths:
            starts.append(cursor)
            sample_indices.append(cursor + prompt_length - 1)
            position_ids.extend(range(prompt_length))
            cursor += prompt_length

        self._paged_request_ids = tuple(int(req_idx) for req_idx in request_ids)
        self.atten_info.b_req_idx = torch.tensor(
            self._paged_request_ids, dtype=torch.int32, device=self.device
        )
        self.atten_info.b_seq_len = torch.tensor(
            lengths, dtype=torch.long, device=self.device
        )
        self.atten_info.cur_select_index = torch.cat(
            [
                self.req_tokens_manager.get_token_indices(req_idx, prompt_length)
                for req_idx, prompt_length in zip(self._paged_request_ids, lengths)
            ]
        ).to(torch.int32)
        self.atten_info.b_start_loc = torch.tensor(
            starts, dtype=torch.int32, device=self.device
        )
        self.atten_info.max_actual_seq_len = max(lengths)

        return (
            torch.tensor(position_ids, dtype=torch.long, device=self.device),
            torch.tensor(sample_indices, dtype=torch.long, device=self.device),
        )

    def activate_paged_chunk_prefill_batch(
        self,
        request_ids: tuple[int, ...],
        context_lengths: list[int] | tuple[int, ...],
        chunk_lengths: list[int] | tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Select later chunked-prefill requests for paged chunk attention.

        The model receives a flattened current-Q chunk.  K/V for the chunk are
        written to the paged KV cache first, then attention reads
        historical-paged KV plus current chunk KV through the request token
        table.  ``context_lengths`` are the existing logical lengths before
        this chunk; ``chunk_lengths`` are the number of new prompt tokens.
        """
        if not request_ids:
            raise ValueError("request_ids must not be empty")
        if len(request_ids) != len(context_lengths) or len(request_ids) != len(chunk_lengths):
            raise ValueError(
                "request_ids, context_lengths and chunk_lengths must have same length"
            )

        request_ids = tuple(int(req_idx) for req_idx in request_ids)
        contexts = tuple(int(length) for length in context_lengths)
        chunks = tuple(int(length) for length in chunk_lengths)
        if any(length < 0 for length in contexts):
            raise ValueError("context_lengths must be non-negative")
        if any(length <= 0 for length in chunks):
            raise ValueError("chunk_lengths must be positive")

        for req_idx, context_length, chunk_length in zip(request_ids, contexts, chunks):
            actual_length = self.req_tokens_manager.req_token_count[int(req_idx)]
            if actual_length != context_length:
                raise ValueError(
                    "paged chunk prefill context length mismatch: "
                    f"request={req_idx}, expected={context_length}, "
                    f"actual={actual_length}"
                )
            if not self.req_tokens_manager.extend_req(req_idx, chunk_length):
                current_tokens = self.req_tokens_manager.req_token_count.get(
                    req_idx, "unknown"
                )
                max_seq_len = getattr(
                    self.req_tokens_manager, "max_seq_len", "unknown"
                )
                free_pages = getattr(
                    self.req_tokens_manager.page_mgr, "num_free_pages", "unknown"
                )
                raise RuntimeError(
                    "Paged KV allocation failed for chunked prefill request "
                    f"{req_idx}: context_tokens={context_length}, "
                    f"chunk_tokens={chunk_length}, current_tokens={current_tokens}, "
                    f"max_seq_len={max_seq_len}, free_pages={free_pages}"
                )

        starts: list[int] = []
        sample_indices: list[int] = []
        position_ids: list[int] = []
        cursor = 0
        for context_length, chunk_length in zip(contexts, chunks):
            starts.append(cursor)
            sample_indices.append(cursor + chunk_length - 1)
            position_ids.extend(range(context_length, context_length + chunk_length))
            cursor += chunk_length

        self._paged_request_ids = request_ids
        self.atten_info.b_req_idx = torch.tensor(
            request_ids, dtype=torch.int32, device=self.device
        )
        total_lengths = tuple(
            context_length + chunk_length
            for context_length, chunk_length in zip(contexts, chunks)
        )
        self.atten_info.b_seq_len = torch.tensor(
            total_lengths, dtype=torch.long, device=self.device
        )
        self.atten_info.cur_select_index = torch.cat(
            [
                self.req_tokens_manager.get_token_indices(req_idx, total_length)[
                    context_length:total_length
                ]
                for req_idx, context_length, total_length in zip(
                    request_ids, contexts, total_lengths
                )
            ]
        ).to(torch.int32)
        self.atten_info.b_start_loc = torch.tensor(
            starts, dtype=torch.int32, device=self.device
        )
        self.atten_info.max_actual_seq_len = max(total_lengths)
        self.atten_info.max_actual_q_seq_len = max(chunks)
        self.atten_info.chunk_context_len = torch.tensor(
            contexts, dtype=torch.long, device=self.device
        )
        self.atten_info.chunk_q_seq_len = torch.tensor(
            chunks, dtype=torch.long, device=self.device
        )
        self.atten_info.is_paged_chunk_prefill = True

        return (
            torch.tensor(position_ids, dtype=torch.long, device=self.device),
            torch.tensor(sample_indices, dtype=torch.long, device=self.device),
        )

    def activate_paged_decode_batch(
        self, request_ids: tuple[int, ...]
    ) -> None:
        """Rebuild AttentionInfo for the current dynamic decode batch."""
        self._reset_paged_chunk_prefill_metadata()
        req_ids, seq_lens, last_indices = (
            self.req_tokens_manager.batch_metadata(list(request_ids))
        )
        self._paged_request_ids = request_ids
        self.atten_info.b_req_idx = req_ids
        self.atten_info.b_seq_len = seq_lens
        self.atten_info.cur_select_index = last_indices
        self.atten_info.b_start_loc = torch.zeros(
            len(request_ids), dtype=torch.int32, device=self.device
        )
        self.atten_info.max_actual_seq_len = max(
            self.req_tokens_manager.req_token_count[req_idx]
            for req_idx in request_ids
        )

    def extend_paged_requests(
        self, request_ids: tuple[int, ...]
    ) -> None:
        """Allocate the KV position consumed by the next decode input."""
        for req_idx in request_ids:
            if not self.req_tokens_manager.extend_req(req_idx, 1):
                current_tokens = self.req_tokens_manager.req_token_count.get(
                    req_idx, "unknown"
                )
                max_seq_len = getattr(
                    self.req_tokens_manager, "max_seq_len", "unknown"
                )
                free_pages = getattr(
                    self.req_tokens_manager.page_mgr, "num_free_pages", "unknown"
                )
                raise RuntimeError(
                    "Paged KV allocation failed for request "
                    f"{req_idx}: current_tokens={current_tokens}, "
                    f"max_seq_len={max_seq_len}, free_pages={free_pages}"
                )
        self.activate_paged_decode_batch(request_ids)

    def release_paged_request_ids(
        self, request_ids: tuple[int, ...]
    ) -> None:
        for req_idx in request_ids:
            self.req_tokens_manager.free_req(req_idx)
        if request_ids == self._paged_request_ids:
            self._paged_request_ids = ()

    def decode_alloc_kv_cache(self, batch_size):
        if self.use_paged_attn:
            if len(self._paged_request_ids) != batch_size:
                raise RuntimeError(
                    "Paged request id cache is not initialized for the "
                    f"decode batch: cached={len(self._paged_request_ids)}, "
                    f"batch_size={batch_size}"
                )
            new_indices = []
            for req_idx in self._paged_request_ids:
                ok = self.req_tokens_manager.extend_req(req_idx, 1)
                if not ok:
                    raise RuntimeError("Paged KV allocation failed during decode")
                new_indices.append(self.req_tokens_manager.get_token_indices(req_idx)[-1])
            self.atten_info.cur_select_index = torch.stack(new_indices).to(torch.int32)
        else:
            self.atten_info.cur_select_index, _ = self.kv_mem_manager.alloc_kvcache_index(
                batch_size
            )
            update_kv_index(
                self.atten_info.b_req_tokens_table,
                self.atten_info.b_req_idx,
                self.atten_info.b_seq_len,
                self.atten_info.cur_select_index,
            )

        self.atten_info.b_seq_len += 1
        self.atten_info.max_actual_seq_len += 1

        return self.atten_info.cur_select_index  # shape [batch_size,]

    def release_paged_requests(self) -> None:
        """Release host-managed page mappings without reading an NPU tensor."""
        self.release_paged_request_ids(self._paged_request_ids)

    def forward(self, input_ids, position_ids, image_tensor=None, **kwargs):
        if self.model_type in ("llava", "qwen3_vl"):
            logits = self.model.forward(
                input_ids, position_ids, self.atten_info, image_tensor=image_tensor, **kwargs
            )
        elif self.graph_runner is not None and input_ids.shape[1] == 1:
            logits = self.graph_runner(input_ids, position_ids, self.atten_info)
        else:
            logits = self.model.forward(input_ids, position_ids, self.atten_info)
        return logits


# ---------------------------------------------------------------------------
# TP weight sharding helpers (module-level)
# ---------------------------------------------------------------------------
def _get_num_layers_from_config(model_config) -> int:
    if hasattr(model_config, "num_layers"):
        return model_config.num_layers
    if hasattr(model_config, "text_config"):
        return model_config.text_config.num_layers
    return 0


def _shard_state_dict(
    state_dict: dict, num_layers: int, tp: TPConfig, model_config,
) -> dict:
    """Shard loaded state_dict weights for tensor parallelism."""
    is_vl = isinstance(model_config, Qwen3VLConfig)
    prefix = "language_model." if is_vl else ""
    kv_heads = (
        model_config.text_config.num_kv_heads if is_vl
        else model_config.num_kv_heads
    )
    head_dim = (
        model_config.text_config.head_dim if is_vl
        else model_config.head_dim
    )
    if head_dim is None:
        hidden = kv_heads * 8  # fallback, won't be used if none

    for i in range(num_layers):
        p = f"{prefix}layers.{i}.self_attn"

        # Attention: Q, KV (fused), O
        if f"{p}.q_proj_weight" in state_dict:
            state_dict[f"{p}.q_proj_weight"] = shard_attention_q(
                state_dict[f"{p}.q_proj_weight"], tp
            )
        if f"{p}.kv_proj_weight" in state_dict:
            state_dict[f"{p}.kv_proj_weight"] = shard_attention_kv(
                state_dict[f"{p}.kv_proj_weight"], kv_heads, head_dim, tp,
            )
        if f"{p}.o_proj_weight" in state_dict:
            state_dict[f"{p}.o_proj_weight"] = shard_attention_o(
                state_dict[f"{p}.o_proj_weight"], tp
            )

        # FFN: gate, up, down
        for proj in ("gate_proj.weight", "up_proj.weight"):
            k = f"{p.replace('self_attn', 'mlp')}.{proj}"
            if k in state_dict:
                state_dict[k] = shard_ffn_gate_up(state_dict[k], tp)
        k_down = f"{p.replace('self_attn', 'mlp')}.down_proj.weight"
        if k_down in state_dict:
            state_dict[k_down] = shard_ffn_down(state_dict[k_down], tp)

        # MoE experts: router is replicated. Expert weights either shard their
        # intermediate dimension (TP) or shard complete experts (EP).
        moe_prefix = p.replace("self_attn", "mlp")
        gate_up_key = f"{moe_prefix}.experts.gate_up_weight"
        down_key = f"{moe_prefix}.experts.down_weight"
        if gate_up_key in state_dict:
            if tp.moe_parallel_mode == "ep":
                state_dict[gate_up_key] = prepare_moe_gate_up_for_ep(
                    state_dict[gate_up_key], tp
                )
            else:
                state_dict[gate_up_key] = prepare_moe_gate_up_for_gmm(
                    state_dict[gate_up_key],
                    model_config.moe_intermediate_size,
                    tp,
                )
        if down_key in state_dict:
            if tp.moe_parallel_mode == "ep":
                state_dict[down_key] = prepare_moe_down_for_ep(
                    state_dict[down_key], tp
                )
            else:
                state_dict[down_key] = prepare_moe_down_for_gmm(
                    state_dict[down_key], tp
                )

    # lm_head: column-shard along vocab dim
    lm_key = f"{prefix}lm_head_weight"
    if lm_key in state_dict:
        state_dict[lm_key] = shard_lm_head(state_dict[lm_key], tp)

    return state_dict
