"""
Stochastic Duffing Oscillator

This module implements the stochastic Duffing oscillator as a system of first-order
stochastic differential equations (SDEs) with both white and coloured noise.

The dimensionless SDE system is:
    dq(t) = v(t) dt
    dv(t) = (-alpha q(t) - beta q(t)^3 - mu v(t) + gamma cos(Omega t) + xi(t)) dt + sqrt(2 mu Theta) âŠ™ dW(t)
    dxi(t) = -1/t_correl_tilde * xi(t) dt + sqrt(2 D_tilde / t_correl_tilde) âŠ™ dW(t)

where the following are dimensionless parameters:
    q(t): position
    v(t): velocity
    xi(t): auxiliary variable representing Ornstein-Uhlenbeck coloured noise
    alpha, beta: linear and nonlinear stiffness parameters
    mu: damping coefficient
    gamma: forcing amplitude
    Omega: forcing frequency
    Theta: temperature (white noise)
    D_tilde: coloured noise intensity
    t_correl_tilde: correlation time
"""

from __future__ import annotations

import torch
from typing import Tuple, Optional, Callable
from examples.parameters.duffing_oscillator import DuffingOscillatorParameters
from neural_dynamics.core.integrators import stochastic_heun_method
from config import DEVICE


class DuffingOscillator:
    """
    Stochastic Duffing oscillator implementation using PyTorch.

    Implements the dimensionless SDE system with state vector [q, v, xi] where:
    - q(t) is dimensionless position
    - v(t) is dimensionless velocity
    - xi(t) is the Ornstein-Uhlenbeck process for coloured noise
    """

    def __init__(self, params: DuffingOscillatorParameters):
        """
        Initialise the stochastic Duffing oscillator.

        Args:
            params: Dimensionless system parameters
        """
        self.params = params

        # Derived numerical properties
        self.num_steps: int = int(self.params.total_time / self.params.timestep)
        self.time_grid: torch.Tensor = torch.linspace(
            0.0,
            self.params.total_time,
            self.num_steps + 1,
            device=DEVICE,
        )

        # Check if system has stochastic components
        self.has_white_noise = self.params.Theta > 0
        self.has_coloured_noise = self.params.D_tilde > 0
        self.has_noise = self.has_white_noise or self.has_coloured_noise

    def get_state_dimension(self) -> int:
        """
        Get the dimension of the state vector.

        Returns:
            State dimension (always 3 for [q, v, xi])
        """
        return 3

    def get_initial_state(self, batch_size: int = 1) -> torch.Tensor:
        """
        Get the initial state vector [q_0, v_0, xi_0], batched for efficiency.

        Args:
            batch_size: The number of initial states to generate.

        Returns:
            A batch of initial state tensors of shape [batch_size, 3]
        """
        initial_state = torch.tensor(
            [
                self.params.initial_position,
                self.params.initial_velocity,
                self.params.initial_coloured_noise,
            ],
            device=DEVICE,
            dtype=torch.float32,
        )
        return initial_state.repeat(batch_size, 1)

    def external_forcing(self, t: float) -> torch.Tensor:
        """
        External periodic forcing function gamma cos(Omega t).

        Args:
            t: Dimensionless time (scalar)

        Returns:
            Forcing value at time t
        """
        t_tensor = torch.tensor(t, device=DEVICE, dtype=torch.float32)
        return self.params.gamma * torch.cos(self.params.Omega * t_tensor)

    def drift_function(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the drift term for the SDE system. This function is vectorized
        to handle batches of states efficiently.

        Implements the drift vector field:
            [dq/dt, dv/dt, dxi/dt] = [v, -alpha q - beta q^3 - mu v + gamma cos(Omega t) + xi, -xi/t_correl_tilde]

        Args:
            t: Current dimensionless time
            state: Current state [q, v, xi] of shape [batch_size, 3] or [3]

        Returns:
            Drift term vector of the same shape as state.
        """
        state = state.to(DEVICE)

        # Handle both batched and single trajectory cases
        if state.ndim == 1:
            # Single trajectory: state = [q, v, xi]
            q, v, xi = state[0], state[1], state[2]
        else:
            # Batched: state = [batch_size, 3]
            q, v, xi = state[:, 0], state[:, 1], state[:, 2]

        # Position derivative: dq/dt = v
        dq_dt = v

        # Velocity derivative: dv/dt = -alpha q - beta q^3 - mu v + gamma cos(Omega t) + xi
        acceleration = (
            -self.params.alpha * q
            - self.params.beta * torch.pow(q, 3)
            - self.params.mu * v
            + self.external_forcing(t)
            + xi
        )

        # Coloured noise derivative: dxi/dt = -xi/t_correl_tilde
        dxi_dt = -xi / self.params.t_correl_tilde

        # Combine drift components
        if state.ndim == 1:
            return torch.stack([dq_dt, acceleration, dxi_dt]).to(DEVICE)
        else:
            return torch.stack([dq_dt, acceleration, dxi_dt], dim=1).to(DEVICE)

    def diffusion_function(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffusion term for the SDE system. This function is vectorized
        to handle batches of states efficiently.

        Implements the diffusion matrix:
            [0, sqrt(2 mu Theta), sqrt(2 D_tilde / t_correl_tilde)]

        Args:
            t: Current dimensionless time
            state: Current state [q, v, xi] of shape [batch_size, 3] or [3]

        Returns:
            Diffusion term vector of shape [batch_size, 3] or [3]
        """
        state = state.to(DEVICE)

        # Diffusion coefficients
        position_noise = 0.0  # No noise on position equation
        velocity_noise = self.params.white_noise_strength  # sqrt(2 mu Theta)
        coloured_noise_driving = (
            self.params.coloured_noise_strength
        )  # sqrt(2 D_tilde / t_correl_tilde)

        diffusion_vector = torch.tensor(
            [position_noise, velocity_noise, coloured_noise_driving],
            device=DEVICE,
            dtype=torch.float32,
        )

        if state.ndim == 1:
            return diffusion_vector
        else:
            batch_size = state.shape[0]
            return diffusion_vector.expand(batch_size, -1)

    def integrate_sde(
        self,
        initial_state: Optional[torch.Tensor] = None,
        batch_size: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the stochastic Duffing oscillator SDE system for a batch of trajectories.

        This method was refactored to generate multiple trajectories in a batched manner
        for significantly improved performance, as the underlying drift and diffusion
        functions are vectorized. Generating trajectories one-by-one in a Python loop
        is inefficient.

        Args:
            initial_state: Initial conditions of shape [batch_size, 3]. If None, uses default parameters.
            batch_size: The number of trajectories to generate. Ignored if initial_state is provided.

        Returns:
            Tuple of (time_grid, state_trajectory)
            state_trajectory has shape [batch_size, num_steps+1, 3]
        """
        if initial_state is None:
            initial_state = self.get_initial_state(batch_size=batch_size)
        else:
            if initial_state.ndim == 1:
                initial_state = initial_state.unsqueeze(0)
            initial_state = initial_state.to(DEVICE)

        time_grid, trajectory = stochastic_heun_method(
            drift=self.drift_function,
            diffusion=self.diffusion_function,
            y0=initial_state,
            t0=0.0,
            tN_t=self.params.total_time,
            dt=self.params.timestep,
        )

        # Integrator returns [time, batch, dim], so permute to [batch, time, dim]
        if trajectory.ndim == 3:
            trajectory = trajectory.permute(1, 0, 2)

        return time_grid, trajectory

    def get_position_velocity(
        self, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract position and velocity from state trajectory.

        Args:
            trajectory: State trajectory from integrate_sde of shape [batch_size, num_steps+1, 3]

        Returns:
            Tuple of (position, velocity) tensors
        """
        return trajectory[..., 0], trajectory[..., 1]

    def get_coloured_noise(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extract coloured noise component from state trajectory.

        Args:
            trajectory: State trajectory from integrate_sde

        Returns:
            Coloured noise time series xi(t)
        """
        return trajectory[..., 2]

    def get_drift_function(self) -> Callable[[float, torch.Tensor], torch.Tensor]:
        """Return the drift function for the system."""
        return self.drift_function

    def potential_energy(self, q: torch.Tensor) -> torch.Tensor:
        """
        Compute the Duffing potential energy.

        V(q) = alpha q^2/2 + beta q^4/4

        Args:
            q: Dimensionless position

        Returns:
            Potential energy V(q)
        """
        return (
            self.params.alpha * torch.pow(q, 2) / 2
            + self.params.beta * torch.pow(q, 4) / 4
        )

    def kinetic_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute kinetic energy.

        T = v^2/2

        Args:
            v: Dimensionless velocity

        Returns:
            Kinetic energy T
        """
        return torch.pow(v, 2) / 2

    def total_energy(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute total mechanical energy.

        E = T + V = v^2/2 + alpha q^2/2 + beta q^4/4

        Args:
            q: Dimensionless position
            v: Dimensionless velocity

        Returns:
            Total energy E = T + V
        """
        return self.kinetic_energy(v) + self.potential_energy(q)

    def effective_potential(self, q: torch.Tensor, xi: torch.Tensor) -> torch.Tensor:
        """
        Compute effective potential including coloured noise contribution.

        This is useful for visualising how coloured noise modifies the potential landscape.

        Args:
            q: Dimensionless position
            xi: Coloured noise value (Ornstein-Uhlenbeck process)

        Returns:
            Effective potential V_eff(q, xi) = V(q) - xiÂ·q
        """
        return self.potential_energy(q) - xi * q