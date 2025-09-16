"""Compatibility implementations for legacy training loss functions."""

from __future__ import annotations

from torch import Tensor

from neural_dynamics.models.sde import CriticNet
from neural_dynamics.models import sde as _sde

__all__ = ["wasserstein_critic_loss"]


def wasserstein_critic_loss(
    critic: CriticNet,
    real_trajectories: Tensor,
    fake_trajectories: Tensor,
    gradient_penalty_weight: float = 10.0,
) -> Tensor:
    """Alias for ``neural_dynamics.models.sde.compute_critic_cost``.

    This preserves the previous public API that lived in ``training.loss_functions``
    while delegating to the refactored implementation.
    """

    return _sde.compute_critic_cost(
        critic, real_trajectories, fake_trajectories, gradient_penalty_weight
    )
