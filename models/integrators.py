"""
This module provides numerical integration routines for ordinary differential equations (ODEs) and stochastic differential equations (SDEs).

The module includes both trajectory-based integrators for generating training data and single-step integrators for neural SDE training

Implemented trajectory integrators:
- Second-order Runge–Kutta (RK2), suitable for deterministic systems
- Fourth-order Runge–Kutta (RK4), also suitable for deterministic systems  
- Adaptive Runge–Kutta–Fehlberg fourth-fifth-order (RKF45) method, suitable for systems with additive noise
- Euler-Maruyama method for stochastic differential equations
- Stochastic Heun method for improved accuracy SDE integration (predictor-corrector method)

Implemented single-step integrators for neural SDE training:
- Euler-Maruyama single step method (handles both deterministic and stochastic cases)
- Stochastic Heun single step method  
- Runge-Kutta 4th order single step method (deterministic only)

All trajectory integrators are implemented as wrappers around their corresponding single-step methods.
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple
import torch


"""
Single-step integration methods for neural SDE/ODE training.
"""
def euler_maruyama_step(
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion_function: Optional[Callable[[float, torch.Tensor], torch.Tensor]],
    current_state: torch.Tensor,
    current_time: float,
    timestep: float,
) -> torch.Tensor:
    """
    Single step of the Euler-Maruyama method for neural SDE/ODE training.
    
    This function performs one integration step and handles both deterministic (ODE) and stochastic (SDE) cases. For deterministic systems, set diffusion_function=None.
    
    The method approximates:
    - SDE: dX(t) = drift(t, X(t)) dt + diffusion(t, X(t)) dW(t)
    - ODE: dX(t)/dt = drift(t, X(t))  [when diffusion_function=None]
    
    Using the discretisation:
        X_{n+1} = X_n + drift(t_n, X_n) * dt + diffusion(t_n, X_n) * sqrt(dt) * Z_n
    
    where Z_n ~ N(0, I) is standard Gaussian noise (omitted for ODEs).
    
    Args:
        drift_function: Function computing the drift term mu(t, x)
        diffusion_function: Function computing the diffusion term sigma(t, x).
                          If None, treats as deterministic system (no noise).
        current_state: Current state vector of shape [batch_size, state_dimension]
        current_time: Current time value
        timestep: Integration timestep Delta t
        
    Returns:
        Next state vector of shape [batch_size, state_dimension]
        
    Example:
        >>> # Stochastic case
        >>> drift = lambda t, x: -0.5 * x  
        >>> diffusion = lambda t, x: 0.1 * torch.ones_like(x)
        >>> x1 = euler_maruyama_step(drift, diffusion, x0, 0.0, 0.01)
        >>> 
        >>> # Deterministic case  
        >>> x1 = euler_maruyama_step(drift, None, x0, 0.0, 0.01)
    """
    device = current_state.device
    
    # Ensure current_state has proper dimensions
    if current_state.dim() == 0:
        current_state = current_state.unsqueeze(0)
    
    # Compute drift term
    drift_term = drift_function(current_time, current_state)
    if drift_term.dim() == 0:
        drift_term = drift_term.unsqueeze(0)
    
    # Apply drift update
    next_state = current_state + timestep * drift_term
    
    # Add stochastic term if diffusion function is provided
    if diffusion_function is not None:
        diffusion_term = diffusion_function(current_time, current_state)
        if diffusion_term.dim() == 0:
            diffusion_term = diffusion_term.unsqueeze(0)
            
        # Generate Wiener increments: dW = sqrt(dt) * Z, where Z ~ N(0, I)
        sqrt_dt = torch.sqrt(torch.tensor(timestep, device=device))
        state_dim = current_state.shape[-1]
        wiener_increments = torch.randn(state_dim, device=device) * sqrt_dt
        
        # Apply diffusion update
        next_state = next_state + diffusion_term * wiener_increments
    
    return next_state


def stochastic_heun_step(
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion_function: Callable[[float, torch.Tensor], torch.Tensor],
    current_state: torch.Tensor,
    current_time: float,
    timestep: float,
) -> torch.Tensor:
    """
    Single step of the stochastic Heun method for neural SDE training.
    
    This implements a single step of the second-order stochastic Heun scheme, which provides better accuracy than Euler-Maruyama for the drift term while maintaining the same order for the diffusion term.
    
    The method uses a predictor-corrector approach:
    1. Predictor: Y_{n+1} = X_n + drift(t_n, X_n) * dt + diffusion(t_n, X_n) * dW_n
    2. Corrector: X_{n+1} = X_n + 0.5 * [drift(t_n, X_n) + drift(t_{n+1}, Y_{n+1})] * dt + diffusion(t_n, X_n) * dW_n
    
    Args:
        drift_function: Function computing the drift term mu(t, x)
        diffusion_function: Function computing the diffusion term sigma(t, x)
        current_state: Current state vector of shape [batch_size, state_dimension]
        current_time: Current time value
        timestep: Integration timestep Delta t
        
    Returns:
        Next state vector of shape [batch_size, state_dimension]
        
    Example:
        >>> drift = lambda t, x: -x + torch.sin(2 * torch.pi * t)  # Driven oscillator
        >>> diffusion = lambda t, x: 0.2 * torch.ones_like(x)
        >>> x1 = stochastic_heun_step(drift, diffusion, x0, 0.0, 0.001)
    """
    device = current_state.device
    
    # Ensure current_state has proper dimensions
    if current_state.dim() == 0:
        current_state = current_state.unsqueeze(0)
    
    # Generate Wiener increments (same for both predictor and corrector)
    sqrt_dt = torch.sqrt(torch.tensor(timestep, device=device))
    state_dim = current_state.shape[-1]
    wiener_increments = torch.randn(state_dim, device=device) * sqrt_dt
    
    # Evaluate drift and diffusion at current point
    drift_current = drift_function(current_time, current_state)
    if drift_current.dim() == 0:
        drift_current = drift_current.unsqueeze(0)
        
    diffusion_current = diffusion_function(current_time, current_state)
    if diffusion_current.dim() == 0:
        diffusion_current = diffusion_current.unsqueeze(0)
    
    # Predictor step (Euler-Maruyama)
    predictor_state = current_state + timestep * drift_current + diffusion_current * wiener_increments
    
    # Evaluate drift at predicted point
    next_time = current_time + timestep
    drift_predictor = drift_function(next_time, predictor_state)
    if drift_predictor.dim() == 0:
        drift_predictor = drift_predictor.unsqueeze(0)
    
    # Corrector step (average of drift at current and predicted points)
    next_state = (
        current_state 
        + timestep * 0.5 * (drift_current + drift_predictor) 
        + diffusion_current * wiener_increments
    )
    
    return next_state

def runge_kutta_4_step(
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    current_state: torch.Tensor,
    current_time: float,
    timestep: float,
) -> torch.Tensor:
    """
    Single step of the fourth-order Runge-Kutta method for deterministic neural ODE training.
    
    This implements a single step of the classical RK4 method, providing fourth-order accuracy for smooth deterministic systems. This is a deterministic-only method.
    
    The method evaluates the drift function at four points:
        k1 = f(t_n, X_n)
        k2 = f(t_n + dt/2, X_n + dt*k1/2)  
        k3 = f(t_n + dt/2, X_n + dt*k2/2)
        k4 = f(t_n + dt, X_n + dt*k3)
        X_{n+1} = X_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    
    Args:
        drift_function: Function computing the drift term f(t, x)
        current_state: Current state vector of shape [batch_size, state_dimension]
        current_time: Current time value
        timestep: Integration timestep Delta t
        
    Returns:
        Next state vector of shape [batch_size, state_dimension]
        
    Example:
        >>> drift = lambda t, x: x * (1 - x)  # Logistic growth equation
        >>> x1 = runge_kutta_4_step(drift, x0, 0.0, 0.1)
    """
    # Ensure current_state has proper dimensions
    if current_state.dim() == 0:
        current_state = current_state.unsqueeze(0)
    
    # Evaluate drift function at four points
    k1 = drift_function(current_time, current_state)
    if k1.dim() == 0:
        k1 = k1.unsqueeze(0)
    
    k2 = drift_function(current_time + timestep/2, current_state + timestep * k1/2)
    if k2.dim() == 0:
        k2 = k2.unsqueeze(0)
    
    k3 = drift_function(current_time + timestep/2, current_state + timestep * k2/2)
    if k3.dim() == 0:
        k3 = k3.unsqueeze(0)
    
    k4 = drift_function(current_time + timestep, current_state + timestep * k3)
    if k4.dim() == 0:
        k4 = k4.unsqueeze(0)
    
    # Combine weighted contributions
    next_state = current_state + timestep * (k1 + 2*k2 + 2*k3 + k4) / 6
    
    return next_state


def runge_kutta_2_step(
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    current_state: torch.Tensor,
    current_time: float,
    timestep: float,
) -> torch.Tensor:
    """
    Single step of the second-order Runge-Kutta method for deterministic neural ODE training.
    
    This implements a single step of the RK2 method (midpoint method), providing second-order accuracy for deterministic systems.
    
    The method evaluates the drift function at two points:
        k1 = f(t_n, X_n)
        k2 = f(t_n + dt, X_n + dt*k1)
        X_{n+1} = X_n + dt/2 * (k1 + k2)
    
    Args:
        drift_function: Function computing the drift term f(t, x)
        current_state: Current state vector of shape [batch_size, state_dimension]
        current_time: Current time value
        timestep: Integration timestep Delta t
        
    Returns:
        Next state vector of shape [batch_size, state_dimension]
        
    Example:
        >>> drift = lambda t, x: -x + torch.sin(t)  # Damped driven oscillator
        >>> x1 = runge_kutta_2_step(drift, x0, 0.0, 0.01)
    """
    # Ensure current_state has proper dimensions
    if current_state.dim() == 0:
        current_state = current_state.unsqueeze(0)
    
    # Evaluate drift function at two points
    k1 = drift_function(current_time, current_state)
    if k1.dim() == 0:
        k1 = k1.unsqueeze(0)
    
    k2 = drift_function(current_time + timestep, current_state + timestep * k1)
    if k2.dim() == 0:
        k2 = k2.unsqueeze(0)
    
    # Combine with equal weighting (midpoint rule)
    next_state = current_state + timestep * (k1 + k2) / 2
    
    return next_state


def auto_select_integrator(
    has_diffusion: bool,
) -> Callable[..., torch.Tensor]:
    """
    Automatically select the appropriate single-step integrator based on system type.
    
    Args:
        has_diffusion: Whether the system includes diffusion/stochastic terms
        
    Returns:
        Single-step integrator function appropriate for the system type
        
    Example:
        >>> # For a neural SDE with diffusion
        >>> integrator = auto_select_integrator(has_diffusion=True)
        >>> # Returns stochastic_heun_step
        >>> 
        >>> # For a neural ODE without diffusion  
        >>> integrator = auto_select_integrator(has_diffusion=False)
        >>> # Returns runge_kutta_2_step
    """
    if has_diffusion:
        return stochastic_heun_step
    else:
        return runge_kutta_2_step


"""
Generic trajectory integration wrapper
"""
def integrate_trajectory_with_step_method(
    step_integrator: Callable[..., torch.Tensor],
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion_function: Optional[Callable[[float, torch.Tensor], torch.Tensor]],
    initial_state: torch.Tensor,
    initial_time: float,
    final_time: float,
    timestep: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generic trajectory integration using any single-step integrator.
    
    This function provides a common implementation for trajectory-based integration by repeatedly calling a single-step integrator. This eliminates code duplication and automatically handles both deterministic and stochastic cases.
    
    Args:
        step_integrator: Single-step integration function
        drift_function: Function computing drift term
        diffusion_function: Function computing diffusion term (may be None for deterministic)
        initial_state: Initial condition
        initial_time: Start time
        final_time: End time
        timestep: Integration timestep
        
    Returns:
        Tuple of (time_grid, trajectory) where:
        - time_grid: Time points of shape [num_steps + 1]
        - trajectory: State evolution of shape [num_steps + 1, state_dim]
    """
    device = initial_state.device
    
    # Ensure initial_state is a 1D tensor
    if initial_state.dim() == 0:
        initial_state = initial_state.unsqueeze(0)
    
    num_steps = int((final_time - initial_time) / timestep)
    
    # Setup time grid
    time_grid = torch.linspace(initial_time, final_time, num_steps + 1, device=device)
    
    # Accumulate states to avoid in-place operations that could interfere with autograd
    states: list[torch.Tensor] = [initial_state]
    
    # Integrate step by step
    for i in range(num_steps):
        current_time = time_grid[i].item()
        current_state = states[-1]
        
        # Determine if this is a stochastic or deterministic integrator
        # by checking if it accepts diffusion_function
        try:
            # Try calling with diffusion_function (works for stochastic integrators)
            next_state = step_integrator(
                drift_function, diffusion_function, current_state, current_time, timestep
            )
        except TypeError:
            # If that fails, it's a deterministic integrator (like RK4)
            next_state = step_integrator(
                drift_function, current_state, current_time, timestep
            )
        
        states.append(next_state)
    
    # Stack all states into a single trajectory tensor
    trajectory = torch.stack(states, dim=0)
    return time_grid, trajectory


"""
Trajectory-based integration methods (wrappers).
"""
def euler_maruyama(
    drift: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion: Optional[Callable[[float, torch.Tensor], torch.Tensor]],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Euler-Maruyama method for solving stochastic differential equations (SDEs) or deterministic ODEs.

    Solves equations of the form:
    - SDE: dX(t) = drift(t, X(t)) dt + diffusion(t, X(t)) dW(t) [when diffusion provided]
    - ODE: dX(t)/dt = drift(t, X(t)) [when diffusion=None]

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
    return integrate_trajectory_with_step_method(
        step_integrator=euler_maruyama_step,
        drift_function=drift,
        diffusion_function=diffusion,
        initial_state=y0,
        initial_time=t0,
        final_time=tN_t,
        timestep=dt,
    )


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
    return integrate_trajectory_with_step_method(
        step_integrator=stochastic_heun_step,
        drift_function=drift,
        diffusion_function=diffusion,
        initial_state=y0,
        initial_time=t0,
        final_time=tN_t,
        timestep=dt,
    )


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


def rkf45(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> None:
    NotImplementedError("RKF45 not implemented. Please choose another integrator.")


def generate_wiener_increments(
    shape: tuple[int, ...], 
    timestep: float, 
    device: torch.device
) -> torch.Tensor:
    """
    Generate Wiener process increments for stochastic integration.
    
    Creates normally distributed random increments scaled by sqrt(timestep)
    for use in stochastic integration methods.
    
    Args:
        shape: Shape of the increment tensor (batch_size, state_dim)
        timestep: Integration timestep Δt
        device: Device for tensor computation
        
    Returns:
        Wiener increments of shape `shape` with distribution N(0, timestep)
        
    Example:
        >>> increments = generate_wiener_increments((32, 3), 0.01, torch.device('cpu'))
        >>> print(increments.shape)  # torch.Size([32, 3])
        >>> print(increments.std())  # Approximately sqrt(0.01) = 0.1
    """
    raise NotImplementedError()
    