"""Device selection utilities for the neural_dynamics package."""

from __future__ import annotations

from typing import Final

import torch


def get_device() -> torch.device:
    """Return the preferred torch device (MPS, CUDA, or CPU)."""

    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE: Final[torch.device] = get_device()


__all__ = ["DEVICE", "get_device"]
