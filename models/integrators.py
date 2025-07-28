"""
This module provides numerical integration routines for ordinary differential equations (ODEs) and stochastic differential equations (SDEs).

The module includes both trajectory-based integrators for generating training data and single-step integrators for neural SDE training

Implemented trajectory integrators:
- Euler-Maruyama method for ordinary or stochastic differential equations (the Euler–Maruyama method reduces to the Euler method for ODEs, where `diffusion_function` is `None`)
- Stochastic Heun method for improved accuracy SDE integration (predictor-corrector method)

Implemented single-step integrators for neural SDE training:
- Euler-Maruyama single step method (handles both deterministic and stochastic cases)
- Stochastic Heun single step method  
- Second-order Runge–Kutta (RK2) single-step method, suitable for deterministic systems and not implemented for stochastic systems.


Yet to be implemented:
- Second-order Runge–Kutta (RK2), suitable for deterministic systems
- Fourth-order Runge–Kutta (RK4), also suitable for deterministic systems  
- Adaptive Runge–Kutta–Fehlberg fourth-fifth-order (RKF45) method, suitable for systems with additive noise

All trajectory integrators are implemented as wrappers around their corresponding single-step methods.
"""

from __future__ import annotations

import inspect
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
    # Track if input was 1D
    input_was_1d = current_state.ndim == 1
    if input_was_1d:
        current_state = current_state.unsqueeze(0)  # [1, dim]

    # Compute drift term
    drift_term = drift_function(current_time, current_state)
    if drift_term.dim() == 0:
        drift_term = drift_term.unsqueeze(0)
    next_state = current_state + timestep * drift_term
    if diffusion_function is not None:
        noise_strength = diffusion_function(current_time, current_state)
        if noise_strength.dim() == 0:
            noise_strength = noise_strength.unsqueeze(0)
        dW = generate_wiener_increments(current_state.shape, timestep, current_state.device)
        next_state = next_state + noise_strength * dW
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
    
    This implements a single step of the second-order stochastic Heun scheme, which is a higher order method than Euler-Maruyama for the drift term while maintaining the same order for the diffusion term.
    
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
    if current_state.ndim == 1:
        current_state = current_state.unsqueeze(0)

    drift = drift_function(current_time, current_state)
    diffusion = diffusion_function(current_time, current_state)
    dW = generate_wiener_increments(current_state.shape, timestep, current_state.device)
    diffusion_update = diffusion * dW
    predictor_state = current_state + drift * timestep + diffusion_update
    drift_term_predictor = drift_function(current_time + timestep, predictor_state)
    next_state = current_state + 0.5 * (drift + drift_term_predictor) * timestep + diffusion_update
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
    raise NotImplementedError("RK4 not implemented. Please choose another integrator.")


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
Generic trajectory integration wrapper & integration routines
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
    
    This function provides a common implementation for trajectory-based integration by repeatedly calling a single-step integrator. This automatically handles both deterministic and stochastic cases.
    
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

    # Ensure at least a batch dimension
    if initial_state.ndim == 0:
        initial_state = initial_state.unsqueeze(0)
    if initial_state.ndim == 1:
        initial_state = initial_state.unsqueeze(0)

    num_steps = int((final_time - initial_time) / timestep)
    time_grid = torch.linspace(initial_time, final_time, num_steps + 1, device=device)
    states: list[torch.Tensor] = [initial_state]

    # Precompute whether this integrator expects a diffusion arg
    signature = inspect.signature(step_integrator)
    is_stochastic = len(signature.parameters) == 5  # drift, diff, state, time, dt

    for i in range(num_steps):
        current_time = time_grid[i].item()
        current_state = states[-1]

        if is_stochastic:
            next_state = step_integrator(
                drift_function, diffusion_function, current_state, current_time, timestep
            )
        else:
            next_state = step_integrator(
                drift_function, current_state, current_time, timestep
            )
        states.append(next_state)

    trajectory = torch.stack(states, dim=0)
    return time_grid, trajectory

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
    raise NotImplementedError("RK2 not implemented for full-trajectory integration, only for single-step integration. Please choose another integrator.")


def rk4(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> None:
    raise NotImplementedError("RK4 not implemented. Please choose another integrator.")


def rkf45(
    f: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
) -> None:
    raise NotImplementedError("RKF45 not implemented. Please choose another integrator.")

# Single-step batch integration methods
def batch_euler_maruyama_step(
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion_function: Optional[Callable[[float, torch.Tensor], torch.Tensor]],
    current_state: torch.Tensor,
    current_time: float,
    timestep: float,
    *,
    precomputed_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batched single step of the Euler-Maruyama method.
    
    This computes the Euler-Maruyama method for a batch of trajectories in an efficient manner (single trajectories should be handled by the single-step method, batching with a single dimension may add unnecessary overhead).
    Key optimisations:
    - Precomputed noise to avoid repeated random generation
    - Optimised tensor operations for large batch dimensions
    - Memory-efficient handling of diffusion matrices
    
    Physically, this integrates multiple independent SDE realisations:
        dX^(i)(t) = drift(t, X^(i)(t)) dt + diffusion(t, X^(i)(t)) dW^(i)(t)
    where i = 1, ..., batch_size represents different independent noise realisations from the ensemble.
    
    Args:
        drift_function: Function computing the drift term mu(t, x)
        diffusion_function: Function computing the diffusion term sigma(t, x)
        current_state: Current state tensor [batch_size, state_dimension]
        current_time: Current time value
        timestep: Integration timestep Delta t
        precomputed_noise: Pre-generated noise [batch_size, noise_dim] or None
        
    Returns:
        Next state tensor [batch_size, state_dimension]
        
    Example:
        >>> # Batch ensemble simulation
        >>> batch_size = 1000
        >>> x0 = torch.randn(batch_size, 3)  # Different initial conditions
        >>> noise = generate_wiener_increments((batch_size, 3), 0.01, device)
        >>> x1 = batch_euler_maruyama_step(drift, diffusion, x0, 0.0, 0.01, 
        ...                               precomputed_noise=noise)
    """
    device = current_state.device
    batch_size, state_dim = current_state.shape

    # Compute drift term - vectorised across batch
    drift_term = drift_function(current_time, current_state)
    if drift_term.dim() == 0:
        drift_term = drift_term.unsqueeze(0).expand(batch_size, -1)
    elif drift_term.dim() == 1:
        drift_term = drift_term.unsqueeze(0).expand(batch_size, -1)

    # Apply drift update
    next_state = current_state + timestep * drift_term

    # Add stochastic term if diffusion function is provided
    if diffusion_function is not None:
        noise_strength = diffusion_function(current_time, current_state)
        
        # Generate or use precomputed noise
        if precomputed_noise is None:
            dW = generate_wiener_increments(current_state.shape, timestep, device)
        else:
            dW = precomputed_noise
        next_state = next_state + noise_strength * dW
    return next_state

def batch_stochastic_heun_step(
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion_function: Callable[[float, torch.Tensor], torch.Tensor],
    current_state: torch.Tensor,
    current_time: float,
    timestep: float,
    *,
    precomputed_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Batched single step of the stochastic Heun method.
    
    Implements the second-order predictor-corrector scheme for batched trajectories:
    1. Predictor: Y_{n+1}^(i) = X_n^(i) + drift(t_n, X_n^(i)) * dt + sigma(t_n, X_n^(i)) * dW_n^(i)
    2. Corrector: X_{n+1}^(i) = X_n^(i) + 0.5 * [drift(t_n, X_n^(i)) + drift(t_{n+1}, Y_{n+1}^(i))] * dt + sigma(t_n, X_n^(i)) * dW_n^(i)
    
    This method provides higher-order accuracy for batched ensemble simulations.
    
    Args:
        drift_function: Function computing the drift term
        diffusion_function: Function computing the diffusion term  
        current_state: Current state tensor [batch_size, state_dimension]
        current_time: Current time value
        timestep: Integration timestep
        precomputed_noise: Pre-generated noise [batch_size, noise_dim] or None
        
    Returns:
        Next state tensor [batch_size, state_dimension]
    """
    device = current_state.device
    
    # Generate or use precomputed noise
    if precomputed_noise is None:
        dW = generate_wiener_increments(current_state.shape, timestep, device)
    else:
        dW = precomputed_noise

    # Current drift and diffusion
    drift_term = drift_function(current_time, current_state)
    noise_strength = diffusion_function(current_time, current_state)
    noise_update = noise_strength * dW
    predictor_state = current_state + timestep * drift_term + noise_update

    # Drift at predictor point
    drift_term_predictor = drift_function(current_time + timestep, predictor_state)

    # Corrector step: use average of drift terms
    next_state = current_state + 0.5 * timestep * (drift_term + drift_term_predictor) + noise_update
    
    return next_state

# Batch integrate entire trajectory
def batch_integrate_trajectory(
    step_integrator: Callable,
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion_function: Optional[Callable[[float, torch.Tensor], torch.Tensor]],
    initial_state: torch.Tensor,
    initial_time: float,
    final_time: float,
    timestep: float,
    *,
    batch_noise_generation: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched trajectory integration for ensemble simulations.
    
    This function integrates multiple trajectories simultaneously with optimisations:
    - Batch noise pre-generation for entire trajectory
    - Vectorised operations across batch dimension
    - Memory-efficient tensor management
    
    Physically, this simulates an ensemble of noise trajectories.
    
    Args:
        step_integrator: Single-step integration function (e.g., batch_euler_maruyama_step)
        drift_function: Drift function mu(t, x)
        diffusion_function: Diffusion function sigma(t, x) or None
        initial_state: Initial conditions [batch_size, state_dimension]
        initial_time: Starting time t0
        final_time: Ending time T
        timestep: Integration step dt
        batch_noise_generation: Whether to pre-generate all noise for the entire trajectory
        
    Returns:
        Tuple of (times, trajectories) where:
        - times: Time grid [num_steps + 1]
        - trajectories: Solution paths [num_steps + 1, batch_size, state_dimension]
        
    Example:
        >>> # Ensemble simulation with 1000 noise realisations
        >>> batch_size = 1000
        >>> x0 = torch.randn(batch_size, 3)  # Random initial conditions
        >>> times, trajs = batch_integrate_trajectory(
        ...     batch_euler_maruyama_step, drift, diffusion, x0, 0.0, 1.0, 0.01
        ... )
        >>> # Compute ensemble statistics
        >>> mean_traj = trajs.mean(dim=1)  # [num_steps + 1, state_dimension]
        >>> std_traj = trajs.std(dim=1)    # [num_steps + 1, state_dimension]
    """
    device = initial_state.device
    batch_size, state_dim = initial_state.shape
    
    # Time grid
    num_steps = int((final_time - initial_time) / timestep)
    times = torch.linspace(initial_time, final_time, num_steps + 1, device=device)
    
    # Trajectory storage [time, batch, state]
    trajectories = torch.zeros(num_steps + 1, batch_size, state_dim, device=device)
    trajectories[0] = initial_state
    
    # Pre-generate noise for entire trajectory if requested
    if batch_noise_generation and diffusion_function is not None:
        # Vector/diagonal diffusion only: [batch, state_dim]
        noise_shape = (num_steps, batch_size, state_dim)
        batch_noise = generate_wiener_increments(noise_shape, timestep, device)
    
    # Integration loop
    current_state = initial_state
    for i in range(num_steps):
        current_time = times[i].item()
        
        # Get noise for this step
        if batch_noise_generation and diffusion_function is not None:
            step_noise = batch_noise[i]
        else:
            step_noise = None
        
        # Integration step
        if diffusion_function is not None:
            # Check if step_integrator supports precomputed_noise
            if 'precomputed_noise' in inspect.signature(step_integrator).parameters:
                current_state = step_integrator(
                    drift_function, diffusion_function, current_state, 
                    current_time, timestep, precomputed_noise=step_noise
                )
            else:
                current_state = step_integrator(
                    drift_function, diffusion_function, current_state, 
                    current_time, timestep
                )
        else:
            # Deterministic case
            current_state = step_integrator(
                drift_function, None, current_state, current_time, timestep
            )
        
        trajectories[i + 1] = current_state
    
    return times, trajectories


def batch_euler_maruyama(
    drift: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion: Optional[Callable[[float, torch.Tensor], torch.Tensor]],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
    *,
    batch_noise_generation: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched Euler-Maruyama method for ensemble SDE simulation.
    
    Args:
        drift: Drift function mu(t, x)
        diffusion: Diffusion function sigma(t, x) or None for deterministic case
        y0: Initial conditions [batch_size, state_dimension]
        t0: Initial time
        tN_t: Final time  
        dt: Time step
        batch_noise_generation: Enable batch noise pre-generation
        
    Returns:
        Tuple of (times, trajectories) with shapes:
        - times: [num_steps + 1]
        - trajectories: [num_steps + 1, batch_size, state_dimension]
    """
    return batch_integrate_trajectory(
        step_integrator=batch_euler_maruyama_step,
        drift_function=drift,
        diffusion_function=diffusion,
        initial_state=y0,
        initial_time=t0,
        final_time=tN_t,
        timestep=dt,
        batch_noise_generation=batch_noise_generation,
    )


def batch_stochastic_heun_method(
    drift: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion: Callable[[float, torch.Tensor], torch.Tensor],
    y0: torch.Tensor,
    t0: float,
    tN_t: float,
    dt: float,
    *,
    batch_noise_generation: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Batched stochastic Heun method for ensemble SDE simulation.
    
    Provides higher-order accuracy compared to Euler-Maruyama for batched
    ensemble simulations with optimised memory and computational efficiency.
    
    Args:
        drift: Drift function mu(t, x)
        diffusion: Diffusion function sigma(t, x)
        y0: Initial conditions [batch_size, state_dimension]
        t0: Initial time
        tN_t: Final time
        dt: Time step
        batch_noise_generation: Enable batch noise pre-generation
        
    Returns:
        Tuple of (times, trajectories) with shapes:
        - times: [num_steps + 1]
        - trajectories: [num_steps + 1, batch_size, state_dimension]
    """
    return batch_integrate_trajectory(
        step_integrator=batch_stochastic_heun_step,
        drift_function=drift,
        diffusion_function=diffusion,
        initial_state=y0,
        initial_time=t0,
        final_time=tN_t,
        timestep=dt,
        batch_noise_generation=batch_noise_generation,
    )

# Helper functions
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
        timestep: Integration timestep \\Delta t
        device: Device for tensor computation
        
    Returns:
        Wiener increments of shape `shape` with distribution N(0, timestep)
        
    Example:
        >>> increments = generate_wiener_increments((32, 3), 0.01, torch.device('cpu'))
        >>> print(increments.shape)  # torch.Size([32, 3])
        >>> print(increments.std())  # Approximately sqrt(0.01) = 0.1
    """
    sqrt_dt = timestep ** 0.5
    return torch.randn(shape, device=device) * sqrt_dt