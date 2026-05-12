"""Qwen3-VL streaming generation.

Usage:
    generator = Qwen3VLGeneratorStream(checkpoints_dir="/path/to/model")
    outputs = generator.text_completion_stream(
        prompts=["describe this image"],
        image_items=["/path/to/image.jpg"],
    )
"""

from typing import Optional, Union, Generator
import torch
from PIL import Image

from .executor.model_executor import ModelExecutor
from .utils.device import get_device
from .utils.file_interface import get_model_name_from_path

from transformers import AutoTokenizer, AutoProcessor


class Qwen3VLGeneratorStream:
    """Streaming text generator for Qwen3-VL multimodal model."""

    def __init__(
        self,
        checkpoints_dir: str,
        tokenizer_path: str,
        max_gpu_num_blocks=None,
        max_seq_len=2048,
        compiled_model=False,
        device=None,
    ):
        self.checkpoints_dir = checkpoints_dir
        self.compiled_model = compiled_model
        self.max_seq_len = max_seq_len
        self.device = get_device(device)

        self.model_executor = ModelExecutor.build(
            checkpoints_dir=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            device=device,
        )
        self.tokenizer = self.load_tokenizer(tokenizer_path)
        try:
            self.processor = AutoProcessor.from_pretrained(checkpoints_dir)
        except ImportError:
            # NPU may lack torchvision for VideoProcessor; load image-only
            from transformers import AutoImageProcessor
            self.processor = AutoImageProcessor.from_pretrained(checkpoints_dir)
        self.image_token_id = self.model_executor.model_config.image_token_id
        self.vision_start_token_id = self.model_executor.model_config.vision_start_token_id
        self.vision_end_token_id = self.model_executor.model_config.vision_end_token_id

    def load_tokenizer(self, pretrained_model_name_or_path):
        model_name = get_model_name_from_path(pretrained_model_name_or_path)
        use_fast = "llava" not in model_name.lower()
        return AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path, use_fast=use_fast
        )

    def encode_images(self, image_items: list[Union[str, Image.Image]]):
        """Process images into pixel_values and grid_thw."""
        images = []
        for item in image_items:
            if isinstance(item, Image.Image):
                image = item
            elif str(item).startswith(("http://", "https://")):
                import requests
                image = Image.open(requests.get(str(item), stream=True).raw)
            else:
                image = Image.open(str(item))
            images.append(image.convert("RGB"))

        # Qwen3VL processor returns pixel_values and image_grid_thw
        inputs = self.processor(
            text=[""] * len(images),
            images=images,
            return_tensors="pt",
        )
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device, dtype=torch.float16)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)

        return pixel_values, image_grid_thw

    @torch.inference_mode()
    def generate_stream(
        self,
        prompt_tokens: list[list[int]],
        mm_token_type_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        max_gen_len: int = 2048,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> Generator[list[str], None, None]:
        """Generate tokens one at a time, yielding batch outputs at each step."""
        bsz = len(prompt_tokens)
        max_prompt_len = max(len(t) for t in prompt_tokens)
        total_seq_len = min(self.max_seq_len, max_gen_len + max_prompt_len)
        actual_prompt_lens = torch.tensor(
            [len(t) for t in prompt_tokens], dtype=torch.long, device=self.device
        )
        pad_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else self.tokenizer.eos_token_id
        )

        tokens = torch.full(
            (bsz, total_seq_len), pad_id, dtype=torch.long, device=self.device
        )
        for seq_id, token_ids in enumerate(prompt_tokens):
            if isinstance(token_ids, list):
                token_ids = torch.tensor(token_ids, dtype=torch.long, device=self.device)
            tokens[seq_id, : len(token_ids)] = token_ids

        input_text_mask = tokens != pad_id
        eos_reached = torch.tensor([False] * bsz, device=self.device)
        b_req_idx = torch.arange(bsz, device=self.device)

        all_select_index_list = []
        prefill_select_index, _ = self.model_executor.prefill_alloc_kv_cache(
            max_prompt_len, actual_prompt_lens, b_req_idx
        )
        all_select_index_list.append(prefill_select_index)

        position_ids = None
        prev_pos = 0
        input_ids = tokens[:, :max_prompt_len]
        for cur_pos in range(max_prompt_len, total_seq_len):
            batch_size, seq_len = input_ids.shape

            # Compute position_ids for this step
            if seq_len > 1:
                # Prefill: let model compute M-RoPE internally
                step_position_ids = None
            else:
                # Decode: pass incrementing 1D position
                step_position_ids = torch.tensor(
                    [[prev_pos]], dtype=torch.long, device=self.device
                )

            # Prefill step: pass pixel_values for vision encoding
            if cur_pos == max_prompt_len and pixel_values is not None:
                logits = self.model_executor.forward(
                    input_ids,
                    step_position_ids,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    mm_token_type_ids=mm_token_type_ids,
                )
            else:
                logits = self.model_executor.forward(input_ids, step_position_ids)

            decode_select_index = self.model_executor.decode_alloc_kv_cache(bsz)
            all_select_index_list.append(decode_select_index)

            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)

            input_ids = next_token
            mask = ~input_text_mask[:, cur_pos]
            tokens[:, cur_pos] = torch.where(
                mask, next_token.reshape(-1), tokens[:, cur_pos]
            )

            eos_reached = eos_reached | (
                mask & (next_token == self.tokenizer.eos_token_id)
            )

            # Yield only the new token for this step (incremental)
            next_tokens = next_token.reshape(-1)
            batch_outputs = [
                self.tokenizer.decode([next_tokens[i].item()], skip_special_tokens=True)
                for i in range(bsz)
            ]
            yield batch_outputs

            prev_pos += 1

            if eos_reached.all():
                break

        all_select_indexs = torch.concat(all_select_index_list)
        self.model_executor.kv_mem_manager.release_ref(all_select_indexs)

    def text_completion_stream(
        self,
        prompts: list[str],
        image_items: list[Union[str, Image.Image]],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        echo: bool = False,
    ) -> Generator[list[dict], None, None]:
        """Streaming text completion with images.

        Args:
            prompts: List of text prompts (may contain <image> placeholder)
            image_items: List of image paths or PIL Images
            temperature: Sampling temperature
            top_p: Nucleus sampling threshold
            max_gen_len: Maximum generation length
            echo: Whether to include prompt in output

        Yields:
            List of completion dicts with 'generation' and 'tokens' keys
        """
        if max_gen_len is None:
            max_gen_len = self.max_seq_len - 1

        # Build multimodal messages and tokenize
        messages_list = []
        for prompt, img in zip(prompts, image_items):
            messages_list.append([
                {"role": "user", "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ]}
            ])

        # Pass messages directly to processor (it applies chat template internally)
        inputs = self.processor(
            text=messages_list,
            images=image_items,
            return_tensors="pt",
            padding=True,
        )

        input_ids = inputs["input_ids"]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")
        # Qwen3VL processor may output token_type_ids under different keys
        mm_token_type_ids = (
            inputs.get("token_type_ids")
            or inputs.get("mm_token_type_ids")
        )
        # Log available keys for debugging
        import logging
        logging.getLogger(__name__).info("Processor output keys: %s", list(inputs.keys()))
        if mm_token_type_ids is not None:
            logging.getLogger(__name__).info("mm_token_type_ids found, shape=%s", mm_token_type_ids.shape)
        else:
            logging.getLogger(__name__).warning("mm_token_type_ids NOT in processor output, will auto-generate")

        if pixel_values is not None:
            pixel_values = pixel_values.to(self.device, dtype=torch.float16)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(self.device)

        prompt_tokens = [row[row != self.tokenizer.pad_token_id].tolist() for row in input_ids]

        stream = self.generate_stream(
            prompt_tokens=prompt_tokens,
            mm_token_type_ids=mm_token_type_ids.to(self.device) if mm_token_type_ids is not None else None,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )

        completions = [{"generation": "", "tokens": []} for _ in prompts]
        for batch_outputs in stream:
            for i, text in enumerate(batch_outputs):
                completions[i]["generation"] += text
            yield [c.copy() for c in completions]


def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token_sorted_idx = torch.multinomial(probs_sort, num_samples=1)
    return torch.gather(probs_idx, -1, index=next_token_sorted_idx)
