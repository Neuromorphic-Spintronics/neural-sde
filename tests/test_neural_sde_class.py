"""
Unit tests for the NeuralSDE class implementation.

This module tests the core functionality of the NeuralSDE class,
including initialisation, forward pass, and network component access.
"""

import pytest
import torch
import sys

sys.path.append(".")
from parameters.hyperparameters import NetworkArchitecture, Hyperparameters
from models.neural_sde import NeuralSDE


class TestNeuralSDEInitialisation:
    """Test NeuralSDE class initialisation."""

    def test_can_create_neural_sde_with_minimal_config(self):
        """Test that NeuralSDE can be created with basic hyperparameters."""
        # Create minimal but valid hyperparameters
        state_dim = 3
        input_dim = 2
        noise_dim = 2

        drift_arch = NetworkArchitecture(
            input_size=state_dim + 1 + input_dim,  # state + time + inputs
            hidden_sizes=[32, 16],
            output_size=state_dim,
        )

        diffusion_arch = NetworkArchitecture(
            input_size=state_dim + 1 + input_dim,  # state + time + inputs
            hidden_sizes=[32, 16],
            output_size=state_dim * noise_dim,  # flattened diffusion matrix
        )

        discriminator_arch = NetworkArchitecture(
            input_size=64 * state_dim,  # 64 timesteps * state_dim
            hidden_sizes=[32, 16],
            output_size=1,
        )

        hyperparams = Hyperparameters(
            drift_network=drift_arch,
            diffusion_network=diffusion_arch,
            discriminator_network=discriminator_arch,
            state_dimension=state_dim,
            input_dimension=input_dim,
            timestep=0.01,
        )

        # This should not raise an exception when implemented
        neural_sde = NeuralSDE(hyperparams)

        # Basic checks
        assert neural_sde.hyperparameters == hyperparams
        assert neural_sde.timestep == 0.01
        assert neural_sde.state_dimension == state_dim
        assert neural_sde.input_dimension == input_dim

    def test_infers_noise_dimension_correctly(self):
        """Test that noise dimension is correctly inferred from diffusion network."""
        state_dim = 3
        noise_dim = 2
        hyperparams = self._create_hyperparams_with_dimensions(state_dim, noise_dim)

        neural_sde = NeuralSDE(hyperparams)

        # Check that the noise dimension was inferred correctly
        assert neural_sde.diffusion_net.noise_dimension == noise_dim

    def test_infers_trajectory_length_correctly(self):
        """Test that trajectory length is correctly inferred from discriminator network."""
        state_dim = 2
        trajectory_length = 50

        discriminator_arch = NetworkArchitecture(
            input_size=trajectory_length * state_dim, hidden_sizes=[32], output_size=1
        )

        hyperparams = self._create_minimal_hyperparams()
        hyperparams = Hyperparameters(
            drift_network=hyperparams.drift_network,
            diffusion_network=hyperparams.diffusion_network,
            discriminator_network=discriminator_arch,
            state_dimension=state_dim,
            input_dimension=hyperparams.input_dimension,
            timestep=hyperparams.timestep,
        )

        neural_sde = NeuralSDE(hyperparams)
        assert neural_sde.discriminator_net is not None
        assert neural_sde.discriminator_net.trajectory_length == trajectory_length

    def test_raises_error_for_invalid_diffusion_dimensions(self):
        """Test that invalid diffusion network dimensions raise an error."""
        state_dim = 3

        # Create a diffusion network with output size that doesn't divide evenly by state_dim
        diffusion_arch = NetworkArchitecture(
            input_size=state_dim + 1 + 1,  # state + time + input
            hidden_sizes=[16],
            output_size=7,  # Not divisible by state_dim=3
        )

        drift_arch = NetworkArchitecture(
            input_size=state_dim + 1 + 1, hidden_sizes=[16], output_size=state_dim
        )

        discriminator_arch = NetworkArchitecture(
            input_size=32 * state_dim, hidden_sizes=[16], output_size=1
        )

        hyperparams = Hyperparameters(
            drift_network=drift_arch,
            diffusion_network=diffusion_arch,
            discriminator_network=discriminator_arch,
            state_dimension=state_dim,
            input_dimension=1,
            timestep=0.01,
        )

        with pytest.raises(ValueError, match="not divisible by state dimension"):
            NeuralSDE(hyperparams)

    def test_raises_error_for_invalid_discriminator_dimensions(self):
        """Test that invalid discriminator dimensions raise appropriate errors."""
        state_dim = 2
        trajectory_length = 50

        # Create discriminator with wrong input size
        discriminator_arch = NetworkArchitecture(
            input_size=trajectory_length * state_dim + 1,  # Wrong size
            hidden_sizes=[32],
            output_size=1,
        )

        hyperparams = self._create_minimal_hyperparams()
        hyperparams = Hyperparameters(
            drift_network=hyperparams.drift_network,
            diffusion_network=hyperparams.diffusion_network,
            discriminator_network=discriminator_arch,
            state_dimension=state_dim,
            input_dimension=hyperparams.input_dimension,
            timestep=hyperparams.timestep,
        )

        with pytest.raises(ValueError, match="not divisible by state dimension"):
            NeuralSDE(hyperparams)

    def test_raises_error_when_diffusion_present_but_no_discriminator(self):
        """Test that discriminator is required when diffusion network is present."""
        # Create hyperparameters with diffusion but no discriminator
        state_dim = 3
        input_dim = 2
        noise_dim = 2

        drift_arch = NetworkArchitecture(
            input_size=state_dim + 1 + input_dim,
            hidden_sizes=[32, 16],
            output_size=state_dim,
        )

        diffusion_arch = NetworkArchitecture(
            input_size=state_dim + 1 + input_dim,
            hidden_sizes=[32, 16],
            output_size=state_dim * noise_dim,
        )

        # Create a dummy discriminator to satisfy the dataclass requirement
        dummy_discriminator_arch = NetworkArchitecture(
            input_size=64 * state_dim, hidden_sizes=[32, 16], output_size=1
        )

        # Create hyperparameters with discriminator (to satisfy dataclass)
        hyperparams = Hyperparameters(
            drift_network=drift_arch,
            diffusion_network=diffusion_arch,
            discriminator_network=dummy_discriminator_arch,
            state_dimension=state_dim,
            input_dimension=input_dim,
            timestep=0.01,
        )

        # Now remove the discriminator to test the constraint (bypass frozen dataclass)
        object.__setattr__(hyperparams, "discriminator_network", None)

        with pytest.raises(
            ValueError,
            match="Discriminator network is required for stochastic Neural SDEs",
        ):
            NeuralSDE(hyperparams)

    def _create_minimal_hyperparams(self) -> Hyperparameters:
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

        discriminator_arch = NetworkArchitecture(
            input_size=32 * state_dim,  # 32 timesteps
            hidden_sizes=[16],
            output_size=1,
        )

        return Hyperparameters(
            drift_network=drift_arch,
            diffusion_network=diffusion_arch,
            discriminator_network=discriminator_arch,
            state_dimension=state_dim,
            input_dimension=input_dim,
            timestep=0.01,
        )

    def _create_hyperparams_with_dimensions(
        self, state_dim: int, noise_dim: int
    ) -> Hyperparameters:
        """Helper to create hyperparameters with specific state and noise dimensions."""
        input_dim = 1

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

        discriminator_arch = NetworkArchitecture(
            input_size=32 * state_dim, hidden_sizes=[16], output_size=1
        )

        return Hyperparameters(
            drift_network=drift_arch,
            diffusion_network=diffusion_arch,
            discriminator_network=discriminator_arch,
            state_dimension=state_dim,
            input_dimension=input_dim,
            timestep=0.01,
        )


class TestNeuralSDEForwardPass:
    """Test the forward pass through NeuralSDE."""

    def test_forward_pass_basic_shapes(self):
        """Test that forward pass produces correct output shapes."""
        hyperparams = self._create_minimal_hyperparams()
        neural_sde = NeuralSDE(hyperparams)

        batch_size = 4
        state_dim = hyperparams.state_dimension
        input_dim = hyperparams.input_dimension
        num_timesteps = 50

        # Create test inputs
        external_inputs = torch.randn(batch_size, input_dim, num_timesteps)
        initial_state = torch.randn(batch_size, state_dim)

        # Forward pass
        trajectory = neural_sde(external_inputs, initial_state=initial_state)

        # Check output shape
        expected_shape = (batch_size, state_dim, num_timesteps)
        assert trajectory.shape == expected_shape

    def test_forward_pass_without_external_inputs(self):
        """Test forward pass when external inputs are not provided."""
        hyperparams = self._create_minimal_hyperparams()
        neural_sde = NeuralSDE(hyperparams)

        batch_size = 2
        state_dim = hyperparams.state_dimension
        num_timesteps = 20

        # Create inputs without external forcing
        initial_state = torch.randn(batch_size, state_dim)
        empty_inputs = torch.zeros(
            batch_size, hyperparams.input_dimension, num_timesteps
        )

        # This should work - networks should handle zero external inputs
        trajectory = neural_sde(empty_inputs, initial_state=initial_state)

        expected_shape = (batch_size, state_dim, num_timesteps)
        assert trajectory.shape == expected_shape

    def test_forward_pass_validates_input_dimensions(self):
        """Test that forward pass validates input dimensions."""
        hyperparams = self._create_minimal_hyperparams()
        neural_sde = NeuralSDE(hyperparams)

        batch_size = 2
        wrong_input_dim = hyperparams.input_dimension + 1
        num_timesteps = 10

        # Wrong input dimension
        wrong_external_inputs = torch.randn(batch_size, wrong_input_dim, num_timesteps)
        initial_state = torch.randn(batch_size, hyperparams.state_dimension)

        with pytest.raises(
            ValueError, match="External input dimension .* does not match"
        ):
            neural_sde(wrong_external_inputs, initial_state=initial_state)

    def test_forward_pass_validates_state_dimensions(self):
        """Test that forward pass validates state dimensions."""
        hyperparams = self._create_minimal_hyperparams()
        neural_sde = NeuralSDE(hyperparams)

        batch_size = 2
        wrong_state_dim = hyperparams.state_dimension + 1
        num_timesteps = 10

        external_inputs = torch.randn(
            batch_size, hyperparams.input_dimension, num_timesteps
        )
        wrong_initial_state = torch.randn(batch_size, wrong_state_dim)

        with pytest.raises(
            ValueError, match="Initial state dimension .* does not match"
        ):
            neural_sde(external_inputs, initial_state=wrong_initial_state)

    def _create_minimal_hyperparams(self) -> Hyperparameters:
        """Helper to create minimal valid hyperparameters."""
        state_dim = 3
        input_dim = 2
        noise_dim = 1

        drift_arch = NetworkArchitecture(
            input_size=state_dim + 1 + input_dim,
            hidden_sizes=[32],
            output_size=state_dim,
        )

        diffusion_arch = NetworkArchitecture(
            input_size=state_dim + 1 + input_dim,
            hidden_sizes=[32],
            output_size=state_dim * noise_dim,
        )

        discriminator_arch = NetworkArchitecture(
            input_size=64 * state_dim, hidden_sizes=[32], output_size=1
        )

        return Hyperparameters(
            drift_network=drift_arch,
            diffusion_network=diffusion_arch,
            discriminator_network=discriminator_arch,
            state_dimension=state_dim,
            input_dimension=input_dim,
            timestep=0.01,
        )


class TestNeuralSDEParameterCounting:
    """Test parameter counting functionality."""

    def test_count_total_parameters(self):
        """Test that parameter counting works correctly."""
        hyperparams = self._create_minimal_hyperparams()
        neural_sde = NeuralSDE(hyperparams)

        param_counts = neural_sde.count_total_parameters()

        # Check that all expected keys are present
        expected_keys = {"drift_net", "diffusion_net", "discriminator_net", "total"}
        assert set(param_counts.keys()) == expected_keys

        # Check that all counts are non-negative integers
        for component, count in param_counts.items():
            assert isinstance(count, int)
            assert count >= 0

        # Check that total is sum of components
        components_sum = (
            param_counts["drift_net"]
            + param_counts["diffusion_net"]
            + param_counts["discriminator_net"]
        )
        assert param_counts["total"] == components_sum

    def test_parameter_counts_are_positive(self):
        """Test that parameter counts are positive for all networks."""
        hyperparams = self._create_minimal_hyperparams()
        neural_sde = NeuralSDE(hyperparams)

        param_counts = neural_sde.count_total_parameters()

        # When stochastic, all networks should have some parameters (not zero)
        assert param_counts["drift_net"] > 0
        assert param_counts["diffusion_net"] > 0
        assert param_counts["discriminator_net"] > 0
        assert param_counts["total"] > 0

    def _create_minimal_hyperparams(self) -> Hyperparameters:
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

        discriminator_arch = NetworkArchitecture(
            input_size=32 * state_dim, hidden_sizes=[16], output_size=1
        )

        return Hyperparameters(
            drift_network=drift_arch,
            diffusion_network=diffusion_arch,
            discriminator_network=discriminator_arch,
            state_dimension=state_dim,
            input_dimension=input_dim,
            timestep=0.01,
        )


# TODO: needs tests for deterministic mode
