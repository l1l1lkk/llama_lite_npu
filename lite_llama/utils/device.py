"""Centralized device configuration.

Usage:
    from lite_llama.utils.device import get_device
    device = get_device()  # respects LITE_LLAMA_DEVICE env var
"""

import os
from typing import Optional


def get_device(device: Optional[str] = None) -> str:
    """Return the device string for the current environment.

    Resolution order:
    1. Explicit `device` parameter (if not None)
    2. Environment variable `LITE_LLAMA_DEVICE`
    3. Auto-detect: NPU > CUDA > CPU

    Returns:
        str: e.g. "npu:6", "npu", "cuda", "cpu"
    """
    if device is not None:
        return device

    env_device = os.environ.get("LITE_LLAMA_DEVICE")
    if env_device:
        return env_device

    return auto_detect_device()


def auto_detect_device() -> str:
    """Auto-detect the best available device."""
    try:
        import torch
        if hasattr(torch, "npu") and torch.npu.is_available():
            # Return the first available NPU
            npu_count = torch.npu.device_count()
            return f"npu:{npu_count - 1}" if npu_count > 1 else "npu:0"
    except Exception:
        pass

    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass

    return "cpu"
