"""Unit tests for the critic loss utilities in the neural SDE module."""
from __future__ import annotations

import torch
import pytest

from neural_dynamics.config import DEVICE
from neural_dynamics.core.hyperparameters import NetworkArchitecture
from neural_dynamics.models.sde import CriticNet, compute_critic_cost


class TestWassersteinCriticLoss:
    """Test suite for the Wasserstein critic loss function."""

    @pytest.fixture
    def setup_critic_and_data(self):
        """Sets up a dummy critic and sample data for testing."""
        batch_size = 4
        trajectory_length = 20
        state_dim = 3

        arch = NetworkArchitecture(
            input_size=trajectory_length * state_dim,
            hidden_sizes=[32],
            output_size=1,
        )
        critic = CriticNet(arch, trajectory_length).to(DEVICE)

        real_trajectories = torch.randn(
            batch_size, trajectory_length, state_dim, device=DEVICE
        )
        fake_trajectories = torch.randn(
            batch_size, trajectory_length, state_dim, device=DEVICE
        )

        return critic, real_trajectories, fake_trajectories

    def test_loss_computation_and_shape(self, setup_critic_and_data):
        """Test that the loss is computed correctly and returns a scalar tensor."""
        critic, real_trajectories, fake_trajectories = setup_critic_and_data
        gradient_penalty_weight = 10.0

        loss = compute_critic_cost(
            critic, real_trajectories, fake_trajectories, gradient_penalty_weight
        )

        assert isinstance(loss, torch.Tensor)
        assert loss.shape == torch.Size([])  # Should be a scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_gradient_flow_through_critic(self, setup_critic_and_data):
        """Test that gradients flow back to the critic's parameters."""
        critic, real_trajectories, fake_trajectories = setup_critic_and_data
        gradient_penalty_weight = 10.0

        critic.zero_grad()

        loss = compute_critic_cost(
            critic, real_trajectories, fake_trajectories, gradient_penalty_weight
        )

        # The loss must require gradients for backpropagation
        assert loss.requires_grad

        loss.backward()

        # Check that all critic parameters have received a gradient
        for param in critic.parameters():
            assert param.grad is not None
            assert not torch.any(torch.isnan(param.grad))


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
