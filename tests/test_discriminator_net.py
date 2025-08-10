"""
Unit tests for the DiscriminatorNet implementation in the neural SDE framework.
"""

from __future__ import annotations
import torch
import pytest
import sys

sys.path.append(".")
from parameters.hyperparameters import NetworkArchitecture
from models.neural_sde import DiscriminatorNet
from config import DEVICE


class TestDiscriminatorNet:
    """Test suite for DiscriminatorNet implementation."""

    @pytest.mark.parametrize(
        "arch,activation,trajectory_length",
        [
            (NetworkArchitecture(150, [64, 32], 1), torch.nn.Tanh, 50),
            (NetworkArchitecture(90, [32], 1), torch.nn.ReLU, 30),
            (NetworkArchitecture(60, [16, 8], 1), torch.nn.Sigmoid, 20),
            (NetworkArchitecture(120, [128, 64, 32], 1), torch.nn.LeakyReLU, 40),
        ],
    )
    def test_initialisation_and_device(self, arch, activation, trajectory_length):
        """Test discriminator initialisation with various architectures."""
        disc = DiscriminatorNet(arch, trajectory_length, activation=activation)
        assert disc.architecture == arch
        assert disc.trajectory_length == trajectory_length
        assert isinstance(disc, DiscriminatorNet)
        assert next(disc.parameters()).device.type == DEVICE.type

    @pytest.mark.parametrize("batch_size", [1, 8, 32, 64])
    def test_score_trajectory_various_batch_sizes(self, batch_size):
        """Test trajectory scoring with various batch sizes."""
        trajectory_length = 25
        state_dim = 3
        arch = NetworkArchitecture(
            input_size=trajectory_length * state_dim,
            hidden_sizes=[32, 16],
            output_size=1,
        )
        disc = DiscriminatorNet(arch, trajectory_length)

        # Create test trajectory segment
        trajectory_segment = torch.randn(batch_size, trajectory_length, state_dim).to(
            DEVICE
        )
        scores = disc.score_trajectory(trajectory_segment)

        assert scores.shape == (batch_size, 1)
        assert not torch.any(torch.isnan(scores))
        assert not torch.any(torch.isinf(scores))

    def test_gradient_computation(self):
        """Test that gradients can be computed through the discriminator."""
        trajectory_length = 15
        state_dim = 4
        batch_size = 5

        arch = NetworkArchitecture(
            input_size=trajectory_length * state_dim,
            hidden_sizes=[32, 16],
            output_size=1,
        )
        disc = DiscriminatorNet(arch, trajectory_length)

        # Create trajectory that requires gradients
        trajectory_segment = torch.randn(
            batch_size, trajectory_length, state_dim, requires_grad=True, device=DEVICE
        )

        scores = disc.score_trajectory(trajectory_segment)
        loss = scores.sum()
        loss.backward()

        # Check that gradients were computed
        assert trajectory_segment.grad is not None
        assert not torch.any(torch.isnan(trajectory_segment.grad))

    def test_input_dimension_validation(self):
        """Test that incorrect input dimensions raise appropriate errors."""
        trajectory_length = 20
        state_dim = 3

        arch = NetworkArchitecture(
            input_size=trajectory_length * state_dim,
            hidden_sizes=[32, 16],
            output_size=1,
        )
        disc = DiscriminatorNet(arch, trajectory_length)

        # Wrong trajectory length
        wrong_trajectory = torch.randn(4, trajectory_length + 5, state_dim).to(DEVICE)
        with pytest.raises(ValueError):
            disc.score_trajectory(wrong_trajectory)

        # Wrong state dimension
        wrong_state_dim = torch.randn(4, trajectory_length, state_dim + 1).to(DEVICE)
        with pytest.raises(ValueError):
            disc.score_trajectory(wrong_state_dim)
