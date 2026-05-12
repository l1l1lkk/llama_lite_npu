"""Interactive CLI for Qwen3-VL multimodal model."""

import torch
from typing import Optional

from rich.console import Console
from rich.prompt import Prompt

import sys, os
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

from lite_llama.qwen3vl_generate_stream import Qwen3VLGeneratorStream
from lite_llama.utils.image_process import vis_images

# Update this path to your Qwen3-VL model checkpoint
checkpoints_dir = "/path/Qwen/Qwen3-VL-4B-Instruct"


def main(
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gpu_num_blocks=None,
    max_gen_len: Optional[int] = 512,
    compiled_model: bool = False,
):
    console = Console()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        generator = Qwen3VLGeneratorStream(
            checkpoints_dir=checkpoints_dir,
            tokenizer_path=checkpoints_dir,
            max_gpu_num_blocks=max_gpu_num_blocks,
            max_seq_len=max_seq_len,
            compiled_model=compiled_model,
            device=device,
        )
    except Exception as e:
        console.print(f"[red]Model load failed: {e}[/red]")
        sys.exit(1)

    while True:
        console.print(
            "[bold green]Enter image path or URL (type 'exit' to quit):[/bold green]"
        )
        while True:
            image_input = Prompt.ask("Image")
            if os.path.isfile(image_input):
                break
            elif image_input.strip().lower() == "exit":
                break
            else:
                console.print(f"[red]'{image_input}' is not a valid file path![/red]")

        image_input = image_input.strip()
        if image_input.lower() == "exit":
            break

        image_items = [image_input]
        vis_images(image_items)

        input_prompt = Prompt.ask("[bold green]Prompt[/bold green]").strip()
        if input_prompt.lower() == "exit":
            break

        prompts = [input_prompt]

        try:
            stream = generator.text_completion_stream(
                prompts,
                image_items,
                temperature=temperature,
                top_p=top_p,
                max_gen_len=max_gen_len,
            )
        except Exception as e:
            console.print(f"[red]Generation failed: {e}[/red]")
            continue

        completion = ""
        console.print("ASSISTANT: ", end="")
        for batch_completions in stream:
            next_text = batch_completions[0]["generation"][len(completion):]
            completion = batch_completions[0]["generation"]
            console.print(f"[red]{next_text}[/red]", end="")

        console.print("\n[bold green]==================================[/bold green]\n")


if __name__ == "__main__":
    main()
