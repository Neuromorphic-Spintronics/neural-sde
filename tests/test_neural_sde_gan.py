"""
Unit tests for the Neural SDE GAN training loop.
"""

import torch
import sys

sys.path.append(".")
from config import DEVICE
from parameters.hyperparameters import NetworkArchitecture, Hyperparameters
from models.neural_sde import NeuralSDE
from training.train_loop import train_gan_model


def create_minimal_hyperparams() -> Hyperparameters:
    """Helper to create minimal valid hyperparameters."""
    state_dim = 2
    input_dim = 1
    noise_dim = 1

    drift_arch = NetworkArchitecture(
        input_size=state_dim + 1 + input_dim,
        hidden_sizes=[16],
        output_size=state_dim,
    )

    diffusion_arch = NetworkArchitecture(
        input_size=state_dim + 1 + input_dim,
        hidden_sizes=[16],
        output_size=state_dim * noise_dim,
    )

    critic_arch = NetworkArchitecture(
        input_size=32 * state_dim,  # 32 timesteps
        hidden_sizes=[16],
        output_size=1,
    )

    return Hyperparameters(
        drift_network=drift_arch,
        diffusion_network=diffusion_arch,
        critic_network=critic_arch,
        state_dimension=state_dim,
        input_dimension=input_dim,
        timestep=0.01,
    )


class TestGANTraining:
    """Test suite for GAN training loop."""

    def test_train_gan_runs_without_errors(self):
        """Test that the GAN training loop runs without raising errors."""
        hyperparams = create_minimal_hyperparams()
        neural_sde = NeuralSDE(hyperparams)

        batch_size = 4
        state_dim = hyperparams.state_dimension
        num_timesteps = 32
        real_trajectory_data = torch.randn(batch_size, num_timesteps, state_dim, device=DEVICE)

        train_gan_model(
            neural_sde=neural_sde,
            real_trajectory_data=real_trajectory_data,
            number_of_epochs=2,
            critic_updates=2,
            gradient_penalty_weight=10.0,
            generator_learning_rate=1e-4,
            critic_learning_rate=1e-4,
        )