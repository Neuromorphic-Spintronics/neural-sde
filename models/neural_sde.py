"""
Neural Stochastic Differential Equation (Neural SDE) model with adversarial training.

This module implements a neural SDE framework that can learn the dynamics of stochastic systems through a combination of neural networks representing drift and diffusion terms, along with adversarial training using a critic.

Key components:
- `DriftNet`: Neural network approximating deterministic dynamics mu(x, t, u)
- `DiffusionNet`: Neural network approximating stochastic dynamics sigma(x, t, u)
- `CriticNet`: Critic network for adversarial training
- `NeuralSDE`: Main class combining networks with SDE integration for simulation
"""

from __future__ import annotations

from typing import Optional, Callable
import torch
from torch import nn, Tensor

from parameters.hyperparameters import NetworkArchitecture, Hyperparameters
from .modules import FeedForwardNetwork
from .protocols import NeuralSDEProtocol
from .integrators import auto_select_integrator, auto_select_matrix_integrator
from config import DEVICE


def _prepare_network_input(
    state: Tensor,
    time: Tensor,
    external_inputs: Optional[Tensor],
    expected_input_size: int,
) -> Tensor:
    """
    Concatenate state, time (as column), and external inputs, padding zeros if needed.
    Ensures time has shape [batch, 1] and calculates the missing external input size
    from the expected input size.
    """
    # Ensure time has column shape
    if time.dim() == 1:
        time = time.unsqueeze(-1)

    batch_size = state.shape[0]
    device = state.device
    state_size = state.shape[1]
    time_size = time.shape[1]

    # Determine how many external input features are expected
    external_input_size = max(0, expected_input_size - state_size - time_size)

    inputs: list[Tensor] = [state, time]
    if external_inputs is not None:
        inputs.append(external_inputs)
    elif external_input_size > 0:
        inputs.append(torch.zeros(batch_size, external_input_size, device=device))

    return torch.cat(inputs, dim=-1)


def _batch_time_tensor(
    batch_size: int, time_value: float, device: torch.device
) -> Tensor:
    """Create a length-`batch_size` time tensor filled with `time_value` on `device`."""
    return torch.full((batch_size,), time_value, device=device)


class DriftNet(FeedForwardNetwork):
    """
    Neural network approximating the drift term mu(x, t, u) in the neural SDE.

    The drift network represents the deterministic component of the system dynamics. It takes as input the current state, time, and optional external forcing to predict the deterministic rate of change.

    The network architecture is specified through the hyperparameters and uses the established FeedForwardNetwork base class for proper initialisation.

    Args:
        architecture: Network architecture specification from hyperparameters
        activation: Activation function (defaults to nn.Tanh)
        device: Computation device (defaults to globally configured device)

    Example:
        >>> from parameters.hyperparameters import NetworkArchitecture
        >>> architecture = NetworkArchitecture(input_size=4, hidden_sizes=[64, 64], output_size=2)
        >>> drift_net = DriftNet(architecture)
        >>> # Input: [batch_size, state_dim + 1 + input_dim] (state + time + inputs)
        >>> # Output: [batch_size, state_dim] (drift vector)
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        activation: Callable[[], nn.Module] = nn.Tanh,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the drift network with specified architecture.

        Args:
            architecture: Network structure specification
            activation: Activation function
            device: Computation device
        """
        super().__init__(architecture, activation, device=device)

    def compute_drift(
        self, state: Tensor, time: Tensor, external_inputs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the drift term for given state, time, and inputs.

        Args:
            state: Current state of shape [batch_size, state_dimension]
            time: Current time of shape [batch_size, 1] or [batch_size]
            external_inputs: External forcing of shape [batch_size, input_dimension]

        Returns:
            Drift vector of shape [batch_size, state_dim]
        """
        network_input = _prepare_network_input(
            state=state,
            time=time,
            external_inputs=external_inputs,
            expected_input_size=self.architecture.input_size,
        )

        # Pass through the neural network
        return self.forward(network_input)


class DiffusionNet(FeedForwardNetwork):
    """
    Neural network approximating the diffusion term sigma(x, t, u) in the neural SDE.

    The diffusion network represents the stochastic component of the system dynamics.

    Args:
        architecture: Network architecture specification from hyperparameters
        state_dimension: Dimension of the system state space
        noise_dimension: Dimension of the noise process
        activation: Activation function (defaults to nn.Tanh)
        device: Computation device

    Example:
        >>> archicture = NetworkArchitecture(input_size=4, hidden_sizes=[32, 32], output_size=6)
        >>> # For 3D state with 2D noise: output_size = 3 * 2 = 6
        >>> diffusion_net = DiffusionNet(architecture, state_dimension=3, noise_dimension=2)
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        state_dimension: int,
        noise_dimension: int,
        activation: Callable[[], nn.Module] = nn.Tanh,
        *,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the diffusion network.

        Args:
            architecture: Network structure specification
            state_dimension: System state dimension
            noise_dimension: Noise process dimension
            activation: Activation function
            device: Computation device
        """
        super().__init__(architecture, activation, device=device)

        self.state_dimension = state_dimension
        self.noise_dimension = noise_dimension

        # Validate that output size matches expected diffusion matrix size
        expected_output_size = state_dimension * noise_dimension
        if self.architecture.output_size != expected_output_size:
            raise ValueError(
                f"Network output size {self.architecture.output_size} does not match "
                f"expected diffusion matrix size {expected_output_size} "
                f"(state_dim={state_dimension} * noise_dim={noise_dimension})"
            )

    def compute_diffusion(
        self, state: Tensor, time: Tensor, external_inputs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the diffusion matrix for given state, time, and inputs.

        Args:
            state: Current state of shape [batch_size, state_dim]
            time: Current time of shape [batch_size, 1] or [batch_size]
            external_inputs: External forcing of shape [batch_size, input_dim]

        Returns:
            Diffusion matrix of shape [batch_size, state_dim, noise_dim]
        """
        network_input = _prepare_network_input(
            state=state,
            time=time,
            external_inputs=external_inputs,
            expected_input_size=self.architecture.input_size,
        )

        # Pass through the neural network
        diffusion_output = self.forward(network_input)
        batch_size = diffusion_output.shape[0]
        diffusion_matrix = diffusion_output.reshape(
            batch_size, self.state_dimension, self.noise_dimension
        )
        return diffusion_matrix


class CriticNet(FeedForwardNetwork):
    """Critic network for adversarial training of neural SDEs.

    This network scores trajectory segments to distinguish between real (from the
    'true' (simulated) system) and synthetic (from the neural SDE) trajectories.
    Used in Wasserstein GAN training with gradient penalty (to ensure
    1-Lipschitz).

    Args:
        architecture: Network architecture specification
        trajectory_length: Length of trajectory segments to evaluate
        activation: Activation function
        device: Computation device

    Example:
        >>> # Critic for trajectory chunks of length 64 with 3D states
        >>> architecture = NetworkArchitecture(input_size=192, hidden_sizes=[64, 32], output_size=1)
        >>> critic = CriticNet(architecture, trajectory_length=64)
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        trajectory_length: int,
        activation: Callable[[], nn.Module] = nn.Tanh,
        *,
        device: torch.device = DEVICE,
    ):
        """Initialise the critic network.

        Args:
            architecture: Network structure specification
            trajectory_length: Length of trajectory segments
            activation: Activation function
            device: Computation device
        """
        super().__init__(architecture, activation=activation, device=device)
        self.trajectory_length = trajectory_length

    def score(self, trajectory_segment: Tensor) -> Tensor:
        """Score a trajectory segment using the critic network.

        This method takes a batch of trajectory segments and flattens them
        to pass through the feedforward network, producing a score for each
        trajectory that indicates how "real" vs "fake" the critic
        believes each trajectory to be.

        Args:
            trajectory_segment: Trajectory of shape [batch_size,
                trajectory_length, state_dim]

        Returns:
            Scores of shape [batch_size, 1] - higher scores indicate more
            "real" trajectories

        Raises:
            ValueError: If trajectory dimensions don't match expected
                architecture
        """
        batch_size, trajectory_length, state_dimension = trajectory_segment.shape
        if trajectory_length != self.trajectory_length:
            raise ValueError(
                f"Trajectory length {trajectory_length} does not match "
                f"critic's expected length {self.trajectory_length}"
            )

        # Flatten trajectory segments for the feedforward network
        flat_trajectories = trajectory_segment.reshape(
            batch_size, trajectory_length * state_dimension
        )
        return self.forward(flat_trajectories)


class NeuralSDE(nn.Module, NeuralSDEProtocol):
    """
    Core Neural SDE model combining drift, diffusion, and integration.

    This is the main class that orchestrates the neural SDE simulation by combining the neural networks with numerical integration. It can operate in both deterministic-only mode (diffusion_net=None) and stochastic mode.

    The system automatically selects the appropriate integration method:
    - Stochastic Heun method for neural SDEs (when diffusion_net is present)
    - Second-order Runge-Kutta (RK2) method for neural ODEs (when diffusion_net is None)

    Args:
        hyperparameters: Complete hyperparameter specification including network architectures, provided by the Hyperparameters class
        timestep: Integration timestep (defaults to hyperparameters value)
        device: Computation device

    Example:
        >>> from parameters.hyperparameters import Hyperparameters
        >>> hyperparams = Hyperparameters(...)
        >>> neural_sde = NeuralSDE(hyperparams)
        >>>
        >>> # Simulate trajectory
        >>> inputs = torch.randn(batch_size, input_dim, num_timesteps)
        >>> initial_state = torch.randn(batch_size, state_dim)
        >>> trajectory = neural_sde(inputs, initial_state=initial_state)
    """

    def __init__(
        self,
        hyperparameters: Hyperparameters,
        timestep: Optional[float] = None,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the neural SDE model.

        Args:
            hyperparameters: Complete configuration specification
            timestep: Integration timestep (uses hyperparameters if None)
            device: Computation device
        """
        super().__init__()

        self.hyperparameters = hyperparameters
        self.timestep = timestep if timestep is not None else hyperparameters.timestep
        self.state_dimension = hyperparameters.state_dimension
        self.input_dimension = hyperparameters.input_dimension
        self.device = device

        # Validate hyperparameters
        self._validate_hyperparameters(hyperparameters)

        # Initialise neural networks
        self.drift_net = DriftNet(
            architecture=hyperparameters.drift_network, device=device
        )

        # Diffusion network is required for stochastic SDEs
        self.diffusion_net = DiffusionNet(
            architecture=hyperparameters.diffusion_network,
            state_dimension=hyperparameters.state_dimension,
            noise_dimension=self._infer_noise_dimension(hyperparameters),
            device=device,
        )

        # Critic network initialisation (validation already enforces presence/output size)
        self.critic_net = CriticNet(
            architecture=hyperparameters.critic_network,
            trajectory_length=self._infer_trajectory_length(hyperparameters),
            device=device,
        )

    def _validate_hyperparameters(self, hyperparameters: Hyperparameters) -> None:
        """
        Validate that hyperparameters are consistent and complete.

        Args:
            hyperparameters: Configuration to validate

        Raises:
            ValueError: If hyperparameters are invalid
        """

        def _require_attr(obj: object, name: str) -> None:
            if not hasattr(obj, name):
                raise ValueError(f"Hyperparameters missing required attribute: {name}")
            if getattr(obj, name) is None:
                raise ValueError(f"Hyperparameters attribute {name} cannot be None")

        def _require_arch_fields(arch: object, label: str) -> None:
            for field in ("input_size", "output_size"):
                if not hasattr(arch, field):
                    raise ValueError(f"{label} architecture missing {field}")

        # Check required attributes exist
        for attr in [
            "drift_network",
            "diffusion_network",
            "state_dimension",
            "input_dimension",
            "timestep",
        ]:
            _require_attr(hyperparameters, attr)

        # Validate network architectures
        _require_arch_fields(hyperparameters.drift_network, "Drift network")
        _require_arch_fields(hyperparameters.diffusion_network, "Diffusion network")

        # Validate timestep is positive
        if hyperparameters.timestep <= 0:
            raise ValueError(
                f"Timestep must be positive, got {hyperparameters.timestep}"
            )

        # Validate dimensions are positive
        if hyperparameters.state_dimension <= 0:
            raise ValueError(
                f"State dimension must be positive, got {hyperparameters.state_dimension}"
            )
        if hyperparameters.input_dimension < 0:
            raise ValueError(
                f"Input dimension must be non-negative, got {hyperparameters.input_dimension}"
            )

        # Validate critic network is provided for stochastic systems
        if (
            not hasattr(hyperparameters, "critic_network")
            or hyperparameters.critic_network is None
        ):
            raise ValueError(
                "Critic network is required for stochastic Neural SDEs "
                "(when diffusion network is present) for adversarial training"
            )

        # Validate critic network architecture if provided
        if (
            hasattr(hyperparameters, "critic_network")
            and hyperparameters.critic_network is not None
        ):
            if not hasattr(hyperparameters.critic_network, "input_size"):
                raise ValueError(
                    "Critic network architecture missing input_size"
                )
            if not hasattr(hyperparameters.critic_network, "output_size"):
                raise ValueError(
                    "Critic network architecture missing output_size"
                )
            if hyperparameters.critic_network.output_size != 1:
                raise ValueError(
                    f"Critic network output size must be 1, got {hyperparameters.critic_network.output_size}"
                )

    def _infer_noise_dimension(self, hyperparameters: Hyperparameters) -> int:
        """
        Infer noise dimension from diffusion network output size. The noise dimension determines the number of independent Brownian motion processes driving the stochastic dynamics.

        Args:
            hyperparameters: Configuration containing network architectures

        Returns:
            Noise dimension (number of independent Brownian motions)

        Examples:
            >>> # Single scalar noise affecting all states
            >>> state_dim, noise_dim = 3, 1  # output_size = 3
            >>>
            >>> # Independent noise per state variable
            >>> state_dim, noise_dim = 3, 3  # output_size = 9
            >>>
            >>> # Five coloured noise sources
            >>> state_dim, noise_dim = 3, 5  # output_size = 15
        """
        state_dim = hyperparameters.state_dimension
        diffusion_output_size = hyperparameters.diffusion_network.output_size

        # Diffusion matrix is flattened: state_dim * noise_dim = output_size
        if diffusion_output_size % state_dim != 0:
            raise ValueError(
                f"Diffusion network output size {diffusion_output_size} is not "
                f"divisible by state dimension {state_dim}. "
                f"Expected output_size = state_dimension * noise_dimension."
            )

        noise_dim = diffusion_output_size // state_dim
        if noise_dim <= 0:
            raise ValueError(f"Inferred noise dimension {noise_dim} must be positive")

        return noise_dim

    def _infer_trajectory_length(self, hyperparameters: Hyperparameters) -> int:
        """
        Infer trajectory length from critic network input size.

        Args:
            hyperparameters: Configuration containing network architectures

        Returns:
            Trajectory length for critic input
        """
        state_dim = hyperparameters.state_dimension
        critic_input_size = hyperparameters.critic_network.input_size

        # Critic input is flattened trajectory: trajectory_length * state_dim
        if critic_input_size % state_dim != 0:
            raise ValueError(
                f"Critic network input size {critic_input_size} is not "
                f"divisible by state dimension {state_dim}"
            )

        trajectory_length = critic_input_size // state_dim  # floor division
        if trajectory_length <= 0:
            raise ValueError(
                f"Inferred trajectory length {trajectory_length} must be positive"
            )

        return trajectory_length

    def forward(
        self, external_inputs: Tensor, initial_state: Tensor, initial_time: float = 0.0
    ) -> Tensor:
        """
        Simulate neural SDE trajectory given external inputs.

        Steps through time using the configured integration method, with the neural networks providing drift and diffusion terms at each step. Gradients are retained for training via backpropagation through time (as per Manneschi et al.) or an adjoint method (to be implemented).

        Args:
            external_inputs: External forcing/control inputs of shape [batch_size, input_dimension, num_timesteps]
            initial_state: Initial state conditions of shape [batch_size, state_dimension]
            initial_time: Starting time for simulation

        Returns:
            Simulated trajectory of shape [batch_size, state_dim, num_timesteps]

        Example:
            >>> batch_size, state_dim, input_dim, num_steps = 32, 3, 2, 100
            >>> inputs = torch.randn(batch_size, input_dim, num_steps)
            >>> x0 = torch.randn(batch_size, state_dim)
            >>> trajectory = neural_sde(inputs, initial_state=x0)
            >>> print(trajectory.shape)  # torch.Size([32, 3, 100])
        """
        # Move inputs to the correct device
        external_inputs = external_inputs.to(self.device)
        initial_state = initial_state.to(self.device)

        batch_size, input_dim, num_timesteps = external_inputs.shape
        state_dim = initial_state.shape[1]

        # Validate input dimensions
        if input_dim != self.input_dimension:
            raise ValueError(
                f"External input dimension {input_dim} does not match "
                f"expected input dimension {self.input_dimension}"
            )
        if state_dim != self.state_dimension:
            raise ValueError(
                f"Initial state dimension {state_dim} does not match "
                f"expected state dimension {self.state_dimension}"
            )

        # Choose a single step function to keep loop simple
        has_diffusion = self.diffusion_net is not None
        step_fn: Callable[[float, Tensor, Tensor], Tensor]

        if has_diffusion:
            matrix_integrator: Callable[..., Tensor] = auto_select_matrix_integrator(
                has_diffusion=True
            )

            def step_fn(
                current_time: float,
                current_state: Tensor,
                external_input_at_step: Tensor,
            ) -> Tensor:
                def drift_func(t: float, x: Tensor) -> Tensor:
                    return self._compute_drift_at_step(x, t, external_input_at_step)

                def diffusion_matrix_func(t: float, x: Tensor) -> Tensor:
                    return self._compute_diffusion_at_step(x, t, external_input_at_step)  # type: ignore[return-value]

                return matrix_integrator(
                    drift_func,
                    diffusion_matrix_func,
                    current_state,
                    current_time,
                    self.timestep,
                )
        else:
            step_integrator: Callable[..., Tensor] = auto_select_integrator(
                has_diffusion=False
            )

            def step_fn(
                current_time: float,
                current_state: Tensor,
                external_input_at_step: Tensor,
            ) -> Tensor:
                def drift_func(t: float, x: Tensor) -> Tensor:
                    return self._compute_drift_at_step(x, t, external_input_at_step)

                return step_integrator(
                    drift_func, current_state, current_time, self.timestep
                )

        # Initialise trajectory storage
        trajectory = torch.zeros(
            batch_size, state_dim, num_timesteps, device=self.device
        )
        current_state = initial_state.clone()
        current_time = initial_time

        # Simulate trajectory step by step
        for step in range(num_timesteps):
            # Store current state
            trajectory[:, :, step] = current_state

            # Get external input at this step
            external_input_at_step = external_inputs[:, :, step]

            # Single-path step
            current_state = step_fn(current_time, current_state, external_input_at_step)

            # Update time
            current_time += self.timestep

        return trajectory

    def _compute_drift_at_step(
        self, state: Tensor, time: float, external_input: Tensor
    ) -> Tensor:
        """
        Compute drift term at a single time step.

        Args:
            state: Current state [batch_size, state_dim]
            time: Current time
            external_input: External inputs at this step [batch_size, input_dim]

        Returns:
            Drift vector [batch_size, state_dim]
        """
        # Convert time to tensor with correct batch size
        batch_size = state.shape[0]
        time_tensor = _batch_time_tensor(batch_size, time, state.device)

        return self.drift_net.compute_drift(state, time_tensor, external_input)

    def _compute_diffusion_at_step(
        self, state: Tensor, time: float, external_input: Tensor
    ) -> Optional[Tensor]:
        """
        Compute diffusion term at a single time step.

        Args:
            state: Current state [batch_size, state_dim]
            time: Current time
            external_input: External inputs at this step [batch_size, input_dim]

        Returns:
            Diffusion matrix [batch_size, state_dim, noise_dim] or None if deterministic
        """
        if self.diffusion_net is None:
            return None

        # Convert time to tensor with correct batch size
        batch_size = state.shape[0]
        time_tensor = _batch_time_tensor(batch_size, time, state.device)

        return self.diffusion_net.compute_diffusion(state, time_tensor, external_input)

    def count_total_parameters(self) -> dict[str, int]:
        """
        Count parameters in each network component.

        Returns:
            Dictionary mapping component names to parameter counts
        """
        from .modules import count_network_parameters

        param_counts = {
            "drift_net": count_network_parameters(self.drift_net),
            "diffusion_net": count_network_parameters(self.diffusion_net)
            if self.diffusion_net
            else 0,
            "critic_net": count_network_parameters(self.critic_net)
            if self.critic_net
            else 0,
        }

        param_counts["total"] = sum(param_counts.values())

        return param_counts