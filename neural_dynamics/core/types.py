"""Core typing helpers for trajectory datasets and related structures."""

from __future__ import annotations

from typing import Protocol

from torch import Tensor


class TrajectoryDataset(Protocol):
    """Protocol describing a time-indexed batch of trajectories.

    Implementations must expose a monotonic ``time_grid`` tensor and a
    ``trajectories`` tensor where the first axis indexes the batch of
    trajectories and the second axis aligns with ``time_grid``.
    """

    time_grid: Tensor
    trajectories: Tensor