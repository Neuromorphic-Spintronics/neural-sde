from __future__ import annotations

from typing import Callable

from torch import Tensor

from neural_dynamics.core.integrators import (
    integrate_trajectory_with_step_method,
    stochastic_heun_step,
    runge_kutta_4_step,
)


def rollout_trajectory(
    *,
    drift_function: Callable[[float, Tensor], Tensor],
    initial_state: Tensor,
    initial_time: float,
    final_time: float,
    timestep: float,
    diffusion_function: Callable[[float, Tensor], Tensor] | None = None,
) -> tuple[Tensor, Tensor]:
    """
    Rolls out a trajectory using a generic drift function and integrator.

    This is a wrapper around `integrate_trajectory_with_step_method` to provide
    a clear API for prediction and evaluation.

    Args:
        drift_function: A function that takes (t, y) and returns the drift dy/dt.
        initial_state: The starting state y(0) of the trajectory.
        initial_time: The starting time t_0.
        final_time: The end time t_N.
        timestep: The integration timestep dt.
        diffusion_function: An optional diffusion function for stochastic rollouts.

    Returns:
        A tuple of (time_grid, predicted_trajectory).
    """
    # Choose an appropriate integrator:
    # - Deterministic ODE: RK4 is accurate and efficient
    # - Stochastic rollouts: Heun method
    step_integrator = runge_kutta_4_step if diffusion_function is None else stochastic_heun_step

    time_grid, trajectory = integrate_trajectory_with_step_method(
        step_integrator=step_integrator,
        drift_function=drift_function,
        diffusion_function=diffusion_function,
        initial_state=initial_state,
        initial_time=initial_time,
        final_time=final_time,
        timestep=timestep,
    )
    return time_grid, trajectory
