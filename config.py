from __future__ import annotations

try:
    from neural_dynamics.config import DEVICE, get_device  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for editable installs
    from typing import Final
    import torch

    def get_device() -> torch.device:
        """Select the best available device (MPS, CUDA, or CPU)."""

        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    DEVICE: Final[torch.device] = get_device()

__all__ = ["DEVICE", "get_device"]
