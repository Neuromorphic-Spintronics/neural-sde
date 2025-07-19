"""
This module provides numerical integration routines for ordinary differential equations (ODEs) and stochastic differential equations (SDEs).

Implemented integrators include:
- Second-order Runge–Kutta (RK2), suitable for deterministic systems
- Fourth-order Runge–Kutta (RK4), also suitable for deterministic systems
- Adaptive Runge–Kutta–Fehlberg fourth-fifth-order (RKF45) method, suitable for systems with additive noise
- Euler-Maruyama method for stochastic differential equations
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import torch


def rk2(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> torch.Tensor:
    """
    Second-order Runge–Kutta (RK2) method for solving ODEs.
    Args:
        f: Callable[[float, torch.Tensor], torch.Tensor]
            The function to integrate.
        y0: torch.Tensor: The initial condition.
        t0: float: The initial time.
        tN_t: float: The final time.
        dt: float: The time step.

    Returns:
        torch.Tensor: The solution to the ODE at time tN_t.
    """
    num_steps = int((tN_t - t0) / dt)

    y = y0
    t = t0
    for _ in range(num_steps):
        k1 = f(t, y)
        k2 = f(t + dt, y + dt * k1)
        y = y + dt * (k1 + k2) / 2
        t += dt

    return y


def rk4(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> torch.Tensor:
    """
    Fourth-order Runge–Kutta (RK4) method for solving ODEs.
    Args:
        f: Callable[[float, torch.Tensor], torch.Tensor]
            The function to integrate.
        y0: torch.Tensor: The initial condition.
        t0: float: The initial time.
        tN_t: float: The final time.
        dt: float: The time step.

    Returns:
    """
    num_steps = int((tN_t - t0) / dt)

    y = y0
    t = t0
    for _ in range(num_steps):
        k1 = f(t, y)
        k2 = f(t + dt / 2, y + dt * k1 / 2)
        k3 = f(t + dt / 2, y + dt * k2 / 2)
        k4 = f(t + dt, y + dt * k3)
        y = y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        t += dt

    return y


def euler_maruyama(
    drift: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion: Optional[Callable[[float, torch.Tensor], torch.Tensor]],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Euler-Maruyama method for solving stochastic differential equations (SDEs).

    Solves SDEs of the form:
        dX(t) = drift(t, X(t)) dt + diffusion(t, X(t)) dW(t)

    where dW(t) is a Wiener process.

    Args:
        drift: Function computing the drift term drift(t, x)
        diffusion: Function computing the diffusion term diffusion(t, x).
                  If None, treats as deterministic ODE.
        y0: Initial condition
        t0: Initial time
        tN_t: Final time
        dt: Time step

    Returns:
        Tuple of (time_grid, trajectory) where:
        - time_grid: Time points of shape [num_steps + 1]
        - trajectory: Solution trajectory of shape [num_steps + 1, state_dim]
    """
    device = y0.device

    # Ensure y0 is a 1D tensor
    if y0.dim() == 0:
        y0 = y0.unsqueeze(0)

    state_dim = y0.shape[0]
    num_steps = int((tN_t - t0) / dt)

    # Setup time grid
    time_grid = torch.linspace(t0, tN_t, num_steps + 1, device=device)

    # Use a Python list to accumulate states in order to avoid any in-place writes
    # on a pre-allocated tensor that autograd might interpret as version-changing
    # operations. We stack into a single tensor at the end.
    states: list[torch.Tensor] = [y0]

    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

    for i in range(num_steps):
        t = time_grid[i].item()
        y = states[-1]

        # Drift term
        drift_term = drift(t, y)
        if drift_term.dim() == 0:
            drift_term = drift_term.unsqueeze(0)

        # Update with drift
        y_next = y + dt * drift_term

        # Add stochastic term if diffusion is provided
        if diffusion is not None:
            diffusion_term = diffusion(t, y)
            if diffusion_term.dim() == 0:
                diffusion_term = diffusion_term.unsqueeze(0)

            # Generate Wiener increments
            dW = torch.randn(state_dim, device=device) * sqrt_dt
            y_next = y_next + diffusion_term * dW

        states.append(y_next)

    trajectory = torch.stack(states, dim=0)
    return time_grid, trajectory


def stochastic_heun_method(
    drift: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Stochastic Heun method for solving stochastic differential equations (SDEs).

    Solves SDEs of the form:
        dX(t) = drift(t, X(t)) dt + diffusion(t, X(t)) dW(t)

    where dW(t) is a Wiener process.

    This is a second-order scheme that uses a predictor-corrector approach:
    1. Predictor: Y_{n+1} = Y_n + drift(t_n, Y_n) * dt + diffusion(t_n, Y_n) * dW_n
    2. Corrector: X_{n+1} = Y_n + 0.5 * [drift(t_n, Y_n) + drift(t_{n+1}, Y_{n+1})] * dt + diffusion(t_n, Y_n) * dW_n

    Args:
        drift: Function computing the drift term drift(t, x)
        diffusion: Function computing the diffusion term diffusion(t, x).
        y0: Initial condition
        t0: Initial time
        tN_t: Final time
        dt: Time step

    Returns:
        Tuple of (time_grid, trajectory) where:
        - time_grid: Time points of shape [num_steps + 1]
        - trajectory: Solution trajectory of shape [num_steps + 1, state_dim]
    """
    device = y0.device

    # Ensure y0 is a 1D tensor
    if y0.dim() == 0:
        y0 = y0.unsqueeze(0)

    state_dim = y0.shape[0]
    num_steps = int((tN_t - t0) / dt)

    # Setup time grid
    time_grid = torch.linspace(t0, tN_t, num_steps + 1, device=device)

    # Use a Python list to accumulate states in order to avoid any in-place writes
    # on a pre-allocated tensor that autograd might interpret as version-changing
    # operations. We stack into a single tensor at the end.
    states: list[torch.Tensor] = [y0]

    sqrt_dt = torch.sqrt(torch.tensor(dt, device=device))

    for i in range(num_steps):
        t = time_grid[i].item()
        y = states[-1]

        # Generate Wiener increments (same for both predictor and corrector)
        dW = torch.randn(state_dim, device=device) * sqrt_dt

        # Drift and diffusion at current point
        drift_current = drift(t, y)
        if drift_current.dim() == 0:
            drift_current = drift_current.unsqueeze(0)

        diffusion_current = diffusion(t, y)
        if diffusion_current.dim() == 0:
            diffusion_current = diffusion_current.unsqueeze(0)

        # Predictor step (Euler-Maruyama)
        y_predictor = y + dt * drift_current + diffusion_current * dW

        # Drift at predicted point
        t_next = time_grid[i + 1].item()
        drift_predictor = drift(t_next, y_predictor)
        if drift_predictor.dim() == 0:
            drift_predictor = drift_predictor.unsqueeze(0)

        # Corrector step (using average of drift at current and predicted points)
        y_next = (
            y + dt * 0.5 * (drift_current + drift_predictor) + diffusion_current * dW
        )

        states.append(y_next)

    trajectory = torch.stack(states, dim=0)
    return time_grid, trajectory


def rkf45(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> None:
    NotImplementedError("RKF45 not implemented. Please choose another integrator.")
