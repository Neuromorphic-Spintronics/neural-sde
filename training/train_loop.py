"""Simple compatibility wrapper around GAN training utilities."""

from __future__ import annotations

import torch
from torch import Tensor

from config import DEVICE
from neural_dynamics.core.hyperparameters import Hyperparameters
from neural_dynamics.models.sde import CriticNet, NeuralSDE, fit_neural_sde_gan


def train_gan_model(
    neural_sde: NeuralSDE,
    real_trajectory_data: Tensor,
    hyperparameters: Hyperparameters,
) -> None:
    """Train the GAN components of a neural SDE model.

    The implementation delegates to :func:`fit_neural_sde_gan` while providing
    light validation to ensure type checkers and tests can rely on clear
    contracts.
    """

    if real_trajectory_data.ndim != 3:
        raise ValueError("real_trajectory_data must have shape [batch, time, state]")

    if neural_sde.diffusion_net is None:
        raise ValueError("NeuralSDE must include a diffusion network for GAN training")

    critic_arch = hyperparameters.critic_network
    if critic_arch is None:
        raise ValueError("Hyperparameters must include a critic network")

    batch_size, trajectory_length, state_dimension = real_trajectory_data.shape
    expected_input = state_dimension * trajectory_length
    if critic_arch.input_size != expected_input:
        raise ValueError(
            "Critic architecture input size does not match provided trajectories"
        )

    critic_network = CriticNet(critic_arch, trajectory_length)

    time_grid = torch.linspace(
        0.0,
        hyperparameters.timestep * float(trajectory_length - 1),
        trajectory_length,
        device=DEVICE,
        dtype=real_trajectory_data.dtype,
    )

    fit_neural_sde_gan(
        neural_sde.drift_net,
        neural_sde.diffusion_net,
        critic_network,
        time_grid,
        real_trajectory_data.to(DEVICE),
        hyperparameters=hyperparameters,
    )

    # The underlying routine performs optimisation in-place; no return value.
    return None
