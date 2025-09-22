"""
Base components for neural dynamics models, including protocols, base network
definitions, and common network modules.
"""

from __future__ import annotations

from typing import Callable, Optional, Protocol
import torch
from torch import nn, Tensor

from neural_dynamics.config import DEVICE
from neural_dynamics.core.hyperparameters import NetworkArchitecture


# --- Protocols ---

class StepIntegratorProtocol(Protocol):
    """
    Protocol for single-step ODE/SDE integration methods used in neural SDE training.
    """

    def __call__(
        self,
        drift_function: Callable[[float, Tensor], Tensor],
        diffusion_function: Optional[Callable[[float, Tensor], Tensor]],
        current_state: Tensor,
        current_time: float,
        timestep: float,
    ) -> Tensor:
        """
        Advance the ODE/SDE state by one timestep.

        Args:
            drift_function: Function computing drift term mu(t, x)
            diffusion_function: Function computing diffusion term sigma(t, x).
                                May be None for deterministic systems.
            current_state: Current state vector of shape [batch_size, state_dimension]
            current_time: Current time value
            timestep: Integration timestep Delta t

        Returns:
            Next state vector of shape [batch_size, state_dim]
        """
        raise NotImplementedError()  # type: ignore


class TrajectoryIntegratorProtocol(Protocol):
    """
    Protocol for trajectory-based SDE integration methods.

    Trajectory integrators compute complete solution paths and are used for generating training data and system analysis.
    """

    def __call__(
        self,
        drift_function: Callable[[float, Tensor], Tensor],
        diffusion_function: Optional[Callable[[float, Tensor], Tensor]],
        initial_state: Tensor,
        initial_time: float,
        final_time: float,
        timestep: float,
    ) -> tuple[Tensor, Tensor]:
        """
        Integrate ODE/SDE over specified time interval.

        Args:
            drift_function: Function computing drift term mu(t, x)
            diffusion_function: Function computing diffusion term sigma(t, x).
                                May be None for deterministic systems.
            initial_state: Initial condition of shape [state_dimension]
            initial_time: Start time
            final_time: End time
            timestep: Integration timestep Δt

        Returns:
            Tuple of (time_grid, trajectory) where:
            - time_grid: Time points of shape [num_steps + 1]
            - trajectory: State evolution of shape [num_steps + 1, state_dim]
        """
        raise NotImplementedError()  # type: ignore


class DynamicalSystemProtocol(Protocol):
    """
    Protocol for dynamical systems that can be used with neural SDE framework.

    Systems implementing this protocol can provide drift and diffusion functions for SDE integration and can generate training data.
    """

    def drift_function(self, time: float, state: Tensor) -> Tensor:
        """
        Compute the drift term of the SDE.

        Args:
            time: Current time
            state: Current state vector

        Returns:
            Drift vector of same shape as state
        """
        raise NotImplementedError()  # type: ignore

    def diffusion_function(self, time: float, state: Tensor) -> Tensor:
        """
        Compute the diffusion term of the SDE.

        Args:
            time: Current time
            state: Current state vector

        Returns:
            Diffusion matrix or vector of appropriate shape for noise multiplication
        """
        raise NotImplementedError()  # type: ignore

    def integrate_sde(
        self, initial_state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        Generate a complete trajectory of the system.

        Args:
            initial_state: Initial conditions. If None, use system defaults.

        Returns:
            Tuple of (time_grid, trajectory)
        """
        raise NotImplementedError()  # type: ignore


class NetworkProtocol(Protocol):
    """
    Protocol for neural networks used in the neural SDE framework.

    Networks implementing this protocol can be used as drift, diffusion, or critic components in the neural SDE architecture.
    """

    def forward(self, inputs: Tensor) -> Tensor:
        """
        Forward pass through the network.

        Args:
            inputs: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]
        """
        raise NotImplementedError()  # type: ignore

    def parameters(self) -> Tensor:
        """Return network parameters for optimisation."""
        raise NotImplementedError()  # type: ignore


class NeuralSDEProtocol(Protocol):
    """
    Protocol for neural SDE models that can simulate system dynamics.

    Neural SDE models implementing this protocol can be trained to approximate the dynamics of stochastic systems and generate synthetic trajectories.
    """

    def forward(
        self, external_inputs: Tensor, initial_state: Tensor, initial_time: float = 0.0
    ) -> Tensor:
        """
        Simulate neural SDE trajectory given external inputs.

        Args:
            external_inputs: External forcing/control inputs of shape [batch_size, input_dim, num_timesteps]
            initial_state: Initial state conditions of shape [batch_size, state_dim]
            initial_time: Starting time for simulation

        Returns:
            Simulated trajectory of shape [batch_size, state_dim, num_timesteps]
        """
        raise NotImplementedError()  # type: ignore

    def get_drift_network(self) -> NetworkProtocol:
        """Return the drift network component."""
        raise NotImplementedError()  # type: ignore

    def get_diffusion_network(self) -> Optional[NetworkProtocol]:
        """Return the diffusion network component (may be None for deterministic phase)."""
        raise NotImplementedError()  # type: ignore

    def get_critic_network(self) -> Optional[NetworkProtocol]:
        """Return the critic network component (may be None if not using adversarial training)."""
        raise NotImplementedError()  # type: ignore


# --- Base Network Modules ---

class FeedForwardNetwork(nn.Sequential, NetworkProtocol):
    """
    Fully connected neural network with configurable activation functions and Xavier initialisation.

    This class provides a standard feedforward architecture that integrates with the parameter system's NetworkArchitecture specification.

    The network applies Xavier initialisation in good approximation to all layers and includes configurable activation functions between layers (except the final layer).

    Args:
        architecture: Network architecture specification defining layer dimensions
        activation: Activation function. If None, defaults to nn.Tanh.
                    Should return a new activation instance when called.
        final_activation: Activation function for the final layer. If None, no activation is applied.
        device: Computation device. Defaults to globally configured device.

    Example:
        >>> from neural_dynamics.core.hyperparameters import NetworkArchitecture
        >>> architecture = NetworkArchitecture(input_size=3, hidden_sizes=[64, 32], output_size=2)
        >>> network = FeedForwardNetwork(architecture, activation=lambda: nn.ReLU())
        >>> output = network(torch.randn(10, 3))  # shape: [10, 2]

    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        activation: Callable[[], nn.Module] = nn.Tanh,
        final_activation: Optional[Callable[[], nn.Module]] = None,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the feedforward network with specified architecture.

        Args:
            architecture: Network layer configuration
            activation: Activation function
            final_activation: Optional activation for the final layer
            device: Computation device
        """

        self.architecture = architecture
        self.input_size = architecture.input_size
        self.output_size = architecture.output_size
        self.layer_sizes = architecture.layer_sizes

        layers = self._build_layers(activation, final_activation)
        super().__init__(*layers)

        self._initialise_weights()
        self.to(device)

    def _build_layers(
        self,
        activation: Callable[[], nn.Module],
        final_activation: Optional[Callable[[], nn.Module]],
    ) -> list[nn.Module]:
        """
        Construct the sequential layers for the network.

        Args:
            activation: Activation function
            final_activation: Optional activation for the final layer

        Returns:
            List of layers to be passed to nn.Sequential
        """
        layers = []

        for i in range(len(self.layer_sizes) - 1):
            # Add linear layer
            layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))

            # Add activation function for all layers except the last one
            is_last_layer = i == len(self.layer_sizes) - 2
            if not is_last_layer:
                layers.append(activation())
            elif final_activation is not None:
                layers.append(final_activation())

        return layers

    def _initialise_weights(self) -> None:
        """
        Apply Xavier initialisation to all linear layers.

        This initialisation helps maintain gradient magnitudes through the network.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, input: Tensor) -> Tensor:  # type: ignore[override]
        """
        Forward pass through the network.

        Args:
            input: Input tensor of shape [batch_size, input_dim]

        Returns:
            Output tensor of shape [batch_size, output_dim]

        Raises:
            ValueError: If input dimension doesn't match architecture (maybe this will be raised by PyTorch in any case)
        """
        # Validate input dimensions
        if input.shape[-1] != self.input_size:
            raise ValueError(
                f"Input dimension {input.shape[-1]} does not match "
                f"expected architecture input size {self.input_size}"
            )

        # Use the parent Sequential's forward method
        current_input = input
        for module in self:
            current_input = module(current_input)
        return current_input


    def _prepare_network_input(
        self,
        state: Tensor,
        time: Tensor,
        external_inputs: Optional[Tensor],
        expected_input_size: int,
    ) -> Tensor:
        """
        Concatenate state, time (as column), and external inputs, padding zeros if needed.
        """
        # Ensure time has column shape
        time = time.view(-1, 1)

        batch_size = state.shape[0]
        device = state.device
        state_size = state.shape[1]
        time_size = time.shape[1]

        # Determine how many external input features are expected
        external_input_size = max(0, expected_input_size - state_size - time_size)

        inputs: list[Tensor] = [state, time]
        if external_inputs is not None:
            if external_inputs.dim() == 0:
                external_inputs = external_inputs.expand(batch_size, 1)
            elif external_inputs.dim() == 1:
                external_inputs = external_inputs.unsqueeze(-1)
            inputs.append(external_inputs)
        elif external_input_size > 0:
            inputs.append(torch.zeros(batch_size, external_input_size, device=device))

        return torch.cat(inputs, dim=-1)


def count_network_parameters(network: nn.Module) -> int:
    """
    Count the total number of trainable parameters in a network.

    This utility function provides a consistent way to count parameters across different network architectures for logging and analysis.
    """
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def initialise_network_weights(
    network: nn.Module, initialisation_method: str = "xavier"
) -> None:
    """
    Apply weight initialisation to all linear layers in a network.
    """
    for module in network.modules():
        if isinstance(module, nn.Linear):
            if initialisation_method == "xavier":
                nn.init.xavier_uniform_(module.weight)
            elif initialisation_method == "kaiming":
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            else:
                raise ValueError(
                    f"Unknown initialisation method: {initialisation_method}. "
                    f"Supported methods: 'xavier', 'kaiming'"
                )

            if module.bias is not None:
                nn.init.zeros_(module.bias)


# --- Common Network Components & Helpers ---




def _batch_time_tensor(
    batch_size: int, time_value: float, device: torch.device
) -> Tensor:
    """Create a length-`batch_size` time tensor filled with `time_value` on `device`."""
    return torch.full((batch_size,), time_value, device=device)


class DriftNet(FeedForwardNetwork):
    """
    Neural network approximating the drift term mu(x, t, u) in the neural SDE.
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        activation: Callable[[], nn.Module] = nn.Tanh,
        final_activation: Optional[Callable[[], nn.Module]] = None,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the drift network with specified architecture.
        """
        super().__init__(
            architecture=architecture,
            activation=activation,
            final_activation=final_activation,
            device=device,
        )

    def compute_drift(
        self, state: Tensor, time: Tensor, external_inputs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the drift term for given state, time, and inputs.
        """
        network_input = self._prepare_network_input(
            state=state,
            time=time,
            external_inputs=external_inputs,
            expected_input_size=self.input_size,
        )

        # Pass through the neural network
        return self.forward(network_input)
