"""
Duffing Oscillator

This module defines the Duffing oscillator, a nonlinear second-order dynamical system with both deterministic and stochastic (coloured noise) components.

The canonical form of the Duffing oscillator is:

    d^2x/dt^2 + delta * dx/dt + alpha * x + beta * x^3 = F(t) + eta(t) + xi(t)

where:
    x: displacement
    t: time
    delta: damping coefficient
    alpha, beta: linear and nonlinear stiffness parameters
    F(t): external forcing
    eta(t): white noise term
    xi(t): noise term (coloured noise in this implementation)
"""

from __future__ import annotations

import torch
import math
from typing import Tuple, Optional
from parameters import DuffingOscillatorParameters
from models.integrators import euler_maruyama
from config import DEVICE


class DuffingOscillator:
    """
    Stochastic Duffing oscillator implementation using PyTorch.

    Implements the second-order stochastic differential equation:
    d^2x/dt^2 + delta * dx/dt + alpha * x + beta * x^3 = F(t) + eta(t) + xi(t)

    Where:
    - eta(t) is white noise
    - xi(t) is coloured noise (Ornstein-Uhlenbeck process)
    - F(t) is external periodic forcing
    """

    def __init__(self, params: DuffingOscillatorParameters):
        """
        Initialise the Duffing oscillator.

        Args:
            params: System parameters
        """
        self.params = params
        self.dim = 3

        # Derived numerical properties
        self.num_steps: int = int(self.params.total_time / self.params.timestep)
        self.time_grid: torch.Tensor = torch.linspace(
            0.0,
            self.params.total_time,
            self.num_steps + 1,
            device=DEVICE,
        )

        # Precompute square root of timestep for noise generation
        self.sqrt_dt = math.sqrt(params.timestep)

    def external_forcing(self, t: float) -> torch.Tensor:
        """
        External periodic forcing function F(t).

        Args:
            t: Time (scalar)

        Returns:
            Forcing value at time t
        """
        # t is a float, so convert to tensor only for torch ops
        t_tensor = torch.tensor(t, device=DEVICE)
        return self.params.forcing_amplitude * torch.cos(
            self.params.forcing_frequency * t_tensor
        )

    def drift_function(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the drift term for the SDE.

        State vector: [x, dx/dt, xi]

        Args:
            t: Current time
            state: Current state [x, v, xi] where v = dx/dt, xi is coloured noise

        Returns:
            Drift term of the SDE
        """
        # Ensure computations occur on the same device as the state tensor
        state = state.to(DEVICE)
        x, v, xi = state[0], state[1], state[2]

        # Duffing oscillator dynamics
        acceleration = (
            -self.params.delta * v
            - self.params.alpha * x
            - self.params.beta * torch.pow(x, 3)
            + self.external_forcing(t)
            + xi  # Coloured noise contribution
        )

        # Ornstein-Uhlenbeck process for coloured noise
        # dxi/dt = -xi/tau
        tau = self.params.coloured_noise_timescale
        dxi_dt = -xi / tau

        return torch.stack([v, acceleration, dxi_dt]).to(DEVICE)

    def diffusion_function(self, t: float, state: torch.Tensor) -> torch.Tensor:
        """
        Compute the diffusion term for the SDE.

        Args:
            t: Current time
            state: Current state [x, v, xi]

        Returns:
            Diffusion term of the SDE
        """
        # White noise only affects velocity (acceleration equation)
        white_noise_strength = self.params.white_noise_strength

        # Coloured noise (OU process) diffusion term
        tau = self.params.coloured_noise_timescale
        sigma = self.params.coloured_noise_strength
        coloured_noise_diffusion = math.sqrt(2 * sigma**2 / tau)

        return torch.tensor(
            [
                0.0,  # No noise on position equation
                white_noise_strength,  # White noise on acceleration
                coloured_noise_diffusion,  # Driving noise for OU process
            ],
            device=DEVICE,
        )

    def integrate_sde(
        self, initial_state: Optional[torch.Tensor] = None, add_white_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Integrate the stochastic Duffing oscillator using Euler-Maruyama method.

        Args:
            initial_state: Initial conditions [x0, v0, xi0]. If None, uses default params
            add_white_noise: Whether to add white noise to the system

        Returns:
            Tuple of (time_grid, state_trajectory)
            state_trajectory has shape [num_steps+1, 3] for [x, v, xi]
        """
        if initial_state is None:
            initial_state = torch.tensor(
                [
                    self.params.initial_position,
                    self.params.initial_velocity,
                    0.0,  # Initial coloured noise state
                ],
                device=DEVICE,
            )
        else:
            initial_state = initial_state.to(DEVICE)

        # Choose diffusion function based on whether to add white noise
        diffusion_fn = self.diffusion_function if add_white_noise else None

        # Integrate using Euler-Maruyama
        time_grid, trajectory = euler_maruyama(
            drift=self.drift_function,
            diffusion=diffusion_fn,
            y0=initial_state,
            t0=0.0,
            tN_t=self.params.total_time,
            dt=self.params.timestep,
            device=DEVICE,
        )

        return time_grid, trajectory

    def get_position_velocity(
        self, trajectory: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract position and velocity from trajectory.

        Args:
            trajectory: State trajectory from integrate_sde

        Returns:
            Tuple of (position, velocity)
        """
        return trajectory[:, 0], trajectory[:, 1]

    def get_coloured_noise(self, trajectory: torch.Tensor) -> torch.Tensor:
        """
        Extract coloured noise component from trajectory.

        Args:
            trajectory: State trajectory from integrate_sde

        Returns:
            Coloured noise time series
        """
        return trajectory[:, 2]

    def potential_energy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the Duffing potential energy.

        Args:
            x: Position

        Returns:
            Potential energy V(x) = α*x²/2 + β*x⁴/4
        """
        return (
            self.params.alpha * torch.pow(x, 2) / 2
            + self.params.beta * torch.pow(x, 4) / 4
        )

    def kinetic_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute kinetic energy.

        Args:
            v: Velocity

        Returns:
            Kinetic energy T = v²/2
        """
        return torch.pow(v, 2) / 2

    def total_energy(self, x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute total mechanical energy.

        Args:
            x: Position
            v: Velocity

        Returns:
            Total energy E = T + V
        """
        return self.kinetic_energy(v) + self.potential_energy(x)
