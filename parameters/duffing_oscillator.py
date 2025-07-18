# This module defines the parameters for the Duffing oscillator system.

from dataclasses import dataclass


@dataclass(frozen=True)
class DuffingOscillatorParameters:
    """
    Parameters for the stochastic Duffing oscillator.
    The equation is: d^2x/dt^2 + delta * dx/dt + alpha * x + beta * x^3 = F(t) + eta(t) + xi(t),
    where:
    - x: displacement, with initial position `initial_position` and initial velocity `initial_velocity`
    - t: time, with total simulation time `total_time` and integration time step `timestep`
    - delta > 0: damping coefficient
    - alpha, beta > 0: linear and nonlinear stiffness parameters
    - F(t): external forcing with amplitude `forcing_amplitude` and frequency `forcing_frequency`
    - eta(t): white noise term with strength `white_noise_strength`
    - xi(t): coloured noise term with strength `coloured_noise_strength` and characteristic timescale `coloured_noise_timescale`.
    """

    # Physical parameters
    delta: float = 0.3
    alpha: float = -1.0
    beta: float = 1.0

    # External forcing
    forcing_amplitude: float = 0.3
    forcing_frequency: float = 1.2

    # White noise parameters
    white_noise_strength: float = 0.1

    # Coloured noise parameters
    coloured_noise_strength: float = 0.05
    coloured_noise_timescale: float = 0.2

    # Numerical parameters
    timestep: float = 0.001
    total_time: float = 100.0

    # Initial conditions
    initial_position: float = 0.1
    initial_velocity: float = 0.0
