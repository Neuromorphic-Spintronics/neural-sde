"""
End-to-end training loop for the Neural SDE GAN.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import optim

from training.loss_functions import wasserstein_critic_loss

if TYPE_CHECKING:
    from models.neural_sde import NeuralSDE
    from torch import Tensor


def train_gan_model(
    neural_sde: NeuralSDE,
    real_trajectory_data: Tensor,
    number_of_epochs: int,
    critic_updates: int,
    gradient_penalty_weight: float,
    generator_learning_rate: float,
    critic_learning_rate: float,
) -> None:
    """Train the Neural SDE GAN model."""
    optimiser_generator = optim.Adam(
        list(neural_sde.drift_net.parameters()) + list(neural_sde.diffusion_net.parameters()),
        lr=generator_learning_rate,
    )
    optimiser_critic = optim.Adam(neural_sde.critic_net.parameters(), lr=critic_learning_rate)

    for epoch in range(number_of_epochs):
        for _ in range(critic_updates):
            # Sample real trajectories
            batch_indices = torch.randperm(real_trajectory_data.size(0))[:64]
            real_trajectory_batch = real_trajectory_data[batch_indices]

            # Generate fake trajectories
            initial_state_batch = real_trajectory_batch[:, 0, :]
            noise_seed = torch.randn(real_trajectory_batch.size(0), neural_sde.input_dimension, real_trajectory_batch.size(1))
            fake_trajectory_batch = neural_sde(noise_seed, initial_state_batch).permute(0, 2, 1)

            # Critic loss
            critic_cost = wasserstein_critic_loss(
                neural_sde.critic_net, real_trajectory_batch, fake_trajectory_batch, gradient_penalty_weight
            )

            # Update critic
            optimiser_critic.zero_grad()
            critic_cost.backward()
            optimiser_critic.step()

        # Train the generator
        initial_state_batch = real_trajectory_batch[:, 0, :]
        noise_seed = torch.randn(real_trajectory_batch.size(0), neural_sde.input_dimension, real_trajectory_batch.size(1))
        fake_trajectory_batch = neural_sde(noise_seed, initial_state_batch).permute(0, 2, 1)

        # Generator loss
        generator_cost = -torch.mean(neural_sde.critic_net.score(fake_trajectory_batch))

        # Update generator
        optimiser_generator.zero_grad()
        generator_cost.backward()
        optimiser_generator.step()