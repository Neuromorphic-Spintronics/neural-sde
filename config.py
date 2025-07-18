from __future__ import annotations

from typing import Final
import torch


def get_device() -> torch.device:
    """
    Select the best available device: MPS (Apple Silicon), CUDA (NVIDIA), or CPU.
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE: Final[torch.device] = get_device()
