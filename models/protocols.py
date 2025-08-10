"""
Type protocols for neural SDE framework components.

This module defines the interfaces that various components of the neural SDE framework must implement, providing strong typing and clear contracts for each of the neural networks and integration methods.
"""

from __future__ import annotations

from typing import Optional, Protocol, Callable
from torch import Tensor


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

    Networks implementing this protocol can be used as drift, diffusion, or discriminator components in the neural SDE architecture.
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

    def get_discriminator_network(self) -> Optional[NetworkProtocol]:
        """Return the discriminator network component (may be None if not using adversarial training)."""
        raise NotImplementedError()  # type: ignore
