from typing import Optional
import torch
from typing import Optional, TypedDict, Generator
from .executor.model_executor import ModelExecutor
from .utils.device import get_device
from .utils.file_interface import get_model_name_from_path
from .kernels.softmax_split import softmax_split
from .sampling import sample_next_token

from transformers import AutoTokenizer


class CompletionPrediction(TypedDict, total=False):
    generation: str
    tokens: list[str]  # not required
    logprobs: list[float]  # not required


@torch.inference_mode()
def sample_top_p(probs, p):
    """
    执行 Top-p (Nucleus) 采样, 从概率分布中采样下一个词。

    参数：
        probs (torch.Tensor): 概率分布张量，形状为 `[batch_size, vocab_size]`。
        p (float): 累积概率阈值，取值范围在 0 到 1 之间。
    返回：
        torch.Tensor: 采样得到的词索引，形状为 `[batch_size, 1]`。

    说明：
        Top-p 采样算法: 选择概率累积和超过阈值 p 的最小集合，将这些词的概率重新归一化后进行采样。
    """
    # 对概率分布进行降序排序。probs_sort: 排序后的概率值，形状与 probs 相同。probs_idx: 排序后的索引，用于映射回原始词汇表。
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    # 计算排序后概率的累积和. 返回的 probs_sum 是累积概率分布。
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    # 保留累积概率未超过阈值 p 的词汇的概率，其余词汇的概率被置为 0.0。
    mask = (
        probs_sum - probs_sort > p
    )  # 创建掩码，对于每个位置，计算累积概率（不包括当前词）是否超过阈值 p。
    probs_sort[mask] = 0.0  # 将累积概率超过阈值 p 的词的概率置零。

    # 对剩余的概率重新归一化, 确保总和为 1。
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    # 从重新归一化的概率分布中采样下一个词. 返回的 next_token 是采样得到的词在排序后概率分布中的索引。
    next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
    # 在 probs_idx 的最后一维（dim=-1）中，使用 next_token_sorted_idx 作为索引，提取对应的值。沿着 dim=1（列）进行索引提取
    # NOTE: torch.gather 函数按照给定的索引张量 index，从输入张量中收集 (获取) 数据，并返回一个与索引张量形状一致的张量。
    next_token = torch.gather(probs_idx, -1, index=next_token_sorted_idx)

    return next_token  # 返回采样得到的下一个词的索引


class GenerateStreamText:
    """
    GenerateText 类用于加载LLaMA模型并执行迭代式生成式推理 (文本生成)。
    """

    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str,
        max_gpu_num_blocks=None,
        max_seq_len=1024,
        compiled_model=False,
        page_size=None,
        moe_parallel_mode="tp",
        device=None,
    ):
        self.device = get_device(device)
        self.checkpoints_dir = checkpoints_dir

        self.model_executor = ModelExecutor.build(
            checkpoints_dir=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            page_size=page_size,
            moe_parallel_mode=moe_parallel_mode,
            device=device,
        )
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        self.model_config = self.model_executor.model_config
        self.device = device

    def load_tokenizer(self, pretrained_model_name_or_path):
        model_name = get_model_name_from_path(pretrained_model_name_or_path)

        if "llava" in model_name.lower():
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_fast=False, trust_remote_code=True
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, use_fast=True, trust_remote_code=True
            )

        return tokenizer

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_tokens: list[list[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
        device=None,
    ) -> Generator[tuple[list[str], Optional[list[float]]], None, None]:
        if device is None:
            device = self.device
        """
        基于提供的 prompt_tokens, 使用语言生成模型逐个生成 token, 并在生成时立即输出。

        参数：
            prompt_tokens (list[list[int]]): 已经进行分词的 prompt, 每个 prompt 是一个整数列表。
            max_gen_len (int): 生成的最大长度。
            temperature (float, optional): 控制采样随机性的温度值。默认为 0.6。
            top_p (float, optional): 用于 nucleus sampling 的概率阈值。默认为 0.9。
            logprobs (bool, optional): 是否计算生成 token 的对数概率。默认为 False。
            echo (bool, optional): 是否在输出中包含 prompt_tokens。默认为 False。

        generator 输出：
            tuple[list[str], Optional[list[float]]]: 包含生成的文本和对应的对数概率(如果 logprobs 为 True)。
        说明：
            该方法在生成循环中，每生成一个新 token, 就立即输出对应的文本和概率(如果需要）。
        """
        bsz = len(prompt_tokens)
        # min_prompt_len = min(len(t) for t in prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= self.model_config.max_seq_len
        total_len = min(self.model_config.max_seq_len, max_gen_len + max_prompt_len)
        actual_prompt_lens = torch.tensor(
            [len(t) for t in prompt_tokens], dtype=torch.long, device=device
        )
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        # 预分配tokens张量
        tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
        input_text_mask = tokens != pad_id
        host_eos_reached = [False] * bsz
        prev_pos = 0
        last_yielded_pos = [
            len(prompt_tokens[i]) if not echo else 0 for i in range(bsz)
        ]  # 初始化每个样本已输出的位置

        # 填充提示词到 tokens 张量
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

        b_req_idx = torch.arange(bsz, device=self.device) # batch_size 个请求索引 [0, 1, 2, ..., bsz-1]
        all_select_index_list = []
        prefill_select_index, _ = self.model_executor.prefill_alloc_kv_cache(
            max_prompt_len, actual_prompt_lens, b_req_idx
        )
        all_select_index_list.append(prefill_select_index)

        input_ids = tokens[:, :max_prompt_len]  # [batch_size, seq_len]
        for cur_pos in range(max_prompt_len, total_len):
            input_ids = tokens[:, prev_pos:cur_pos]
            batch_size, seq_len = input_ids.shape
            position_ids = (
                torch.arange(prev_pos, prev_pos + seq_len, device=input_ids.device)
                .unsqueeze(0)  # shape: [1, seq_len]
                .repeat(batch_size, 1)  # shape: [batch_size, seq_len], 不分配额外内存
            )

            logits = self.model_executor.forward(input_ids, position_ids)
            decode_select_index = self.model_executor.decode_alloc_kv_cache(bsz)
            all_select_index_list.append(decode_select_index)

            next_token = sample_next_token(
                logits,
                temperature=temperature,
                top_p=top_p,
                vocab_parallel=self.model_executor.logits_are_sharded,
            )

            input_ids = next_token  # [batch_size, 1]

            # 仅在需要生成的情况下替换 token
            # NOTE: input_text_mask[:, cur_pos]：获取掩码中当前列的布尔值，表示每个序列在当前位置是否为实际输入词元。
            # NOTE: tokens[:, cur_pos]：获取 tokens 中当前列的值。next_token：包含当前生成的词元 ID。
            mask = ~input_text_mask[:, cur_pos]  # [batch_size]
            tokens[:, cur_pos] = torch.where(
                mask, next_token.reshape(-1), tokens[:, cur_pos]
            )

            prev_pos = cur_pos

            # 为整个批次收集输出
            batch_outputs = []
            for i in range(bsz):
                start = last_yielded_pos[i]
                end = cur_pos + 1
                if start < end:
                    token = tokens[i, start:end].tolist()
                    if token:
                        host_eos_reached[i] = (
                            host_eos_reached[i]
                            or token[-1] == self.tokenizer.eos_token_id
                        )
                    text = self.tokenizer.decode(
                        token, skip_special_tokens=True
                    )  # 解码时跳过特殊标记。
                    batch_outputs.append(text)
                    last_yielded_pos[i] = end
                else:
                    batch_outputs.append("")  # 如果没有新生成的内容，添加空字符串

            # 将整个批次的输出一次性 yield
            yield batch_outputs

            # Token decoding already copied this step's ids to Host. Reuse
            # that data for EOS detection instead of launching a second
            # NPU-to-Host synchronization through eos_reached.all().
            if all(host_eos_reached):
                break

        # 减少 kv cache 内存管理器的引用计数
        if self.model_executor.use_paged_attn:
            self.model_executor.release_paged_requests()
        else:
            all_select_indexs = torch.concat(all_select_index_list)
            self.model_executor.kv_mem_manager.release_ref(all_select_indexs)

    def text_completion_stream(
        self,
        prompts: list[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        echo: bool = False,
    ) -> Generator[list[CompletionPrediction], None, None]:
        if max_gen_len is None:
            max_gen_len = self.model_config.max_seq_len - 1

        prompt_tokens = [
            self.tokenizer.encode(x, add_special_tokens=True) for x in prompts
        ]

        stream = self.generate_stream(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )
        # 初始化每个样本的生成结果
        completions = [{"generation": "", "tokens": []} for _ in prompts]
        for batch_outputs in stream:
            for i, text in enumerate(batch_outputs):
                completions[i]["generation"] += text
            yield completions.copy()
