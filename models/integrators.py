"""
This module provides numerical integration routines for ordinary differential equations (ODEs) and stochastic differential equations (SDEs).

The module includes both trajectory-based integrators for generating training data and single-step integrators for neural SDE training.ArithmeticError

The module also supports batching and matrix-aware integrators for neural SDE training.

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


"""
Matrix-diffusion-aware integrators for Neural SDEs

These integrators handle the general neural SDE form:
    dX(t) = mu(X(t), t, u(t)) dt + sigma(X(t), t, u(t)) dW(t)
    
where:
- mu: drift function returning [batch_size, state_dim]
- sigma: diffusion matrix function returning [batch_size, state_dim, noise_dim]
- dW: noise_dim-dimensional Wiener process
"""

def matrix_euler_maruyama_step(
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion_matrix_function: Callable[[float, torch.Tensor], torch.Tensor],
    current_state: torch.Tensor,
    current_time: float,
    timestep: float,
    *,
    precomputed_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Matrix-diffusion-aware Euler-Maruyama integration step for neural SDEs.
    
    Integrates neural SDEs of the form:
        dX(t) = mu(X(t), t) dt + sigma(X(t), t) dW(t)
    
    where sigma(X(t), t) is a [state_dim × noise_dim] matrix and dW(t) is 
    noise_dim-dimensional Brownian motion.
    
    Discretisation:
        X_{n+1} = X_n + mu(t_n, X_n) * dt + sigma(t_n, X_n) @ dW_n
    
    Args:
        drift_function: Function mu(t, x) returning [batch_size, state_dim]
        diffusion_matrix_function: Function sigma(t, x) returning [batch_size, state_dim, noise_dim]
        current_state: Current state [batch_size, state_dim]
        current_time: Current time t
        timestep: Integration timestep dt
        precomputed_noise: Pre-generated noise [batch_size, noise_dim] or None
        
    Returns:
        Next state [batch_size, state_dim]
        
    Example:
        >>> def drift(t, x): return -0.1 * x  # Damping
        >>> def diffusion_matrix(t, x): 
        ...     # Return 3x2 diffusion matrix for each batch element
        ...     batch_size = x.shape[0]
        ...     return torch.randn(batch_size, 3, 2) * 0.1
        >>> x0 = torch.randn(32, 3)
        >>> x1 = matrix_euler_maruyama_step(drift, diffusion_matrix, x0, 0.0, 0.01)
    """
    device = current_state.device
    batch_size, state_dim = current_state.shape
    
    # Compute drift term: mu(t, x)
    drift_term = drift_function(current_time, current_state)
    if drift_term.dim() == 0:
        drift_term = drift_term.unsqueeze(0).expand(batch_size, -1)
    elif drift_term.dim() == 1:
        drift_term = drift_term.unsqueeze(0).expand(batch_size, -1)
    
    # Apply drift update
    next_state = current_state + timestep * drift_term
    
    # Compute diffusion matrix: sigma(t, x)
    diffusion_matrix = diffusion_matrix_function(current_time, current_state)
    
    if diffusion_matrix is not None:
        # Extract noise dimension from diffusion matrix shape
        batch_size_check, state_dim_check, noise_dim = diffusion_matrix.shape
        assert batch_size_check == batch_size and state_dim_check == state_dim, \
            f"Diffusion matrix shape mismatch: expected [{batch_size}, {state_dim}, noise_dim], got {diffusion_matrix.shape}"
        
        # Generate or use precomputed noise
        if precomputed_noise is None:
            dW = generate_wiener_increments((batch_size, noise_dim), timestep, device)
        else:
            dW = precomputed_noise
            
        # Compute matrix-vector product: sigma @ dW
        # diffusion_matrix: [batch_size, state_dim, noise_dim]
        # dW: [batch_size, noise_dim] -> [batch_size, noise_dim, 1]
        # Result: [batch_size, state_dim, 1] -> [batch_size, state_dim]
        diffusion_update = torch.bmm(diffusion_matrix, dW.unsqueeze(-1)).squeeze(-1)
        next_state = next_state + diffusion_update
        
    return next_state


def matrix_stochastic_heun_step(
    drift_function: Callable[[float, torch.Tensor], torch.Tensor],
    diffusion_matrix_function: Callable[[float, torch.Tensor], torch.Tensor],
    current_state: torch.Tensor,
    current_time: float,
    timestep: float,
    *,
    precomputed_noise: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Matrix-diffusion-aware Stochastic Heun integration step for neural SDEs.
    
    This implements the matrix-aware version of the stochastic Heun method,
    providing second-order accuracy for the drift term while properly handling
    matrix diffusion.
    
    Predictor-corrector scheme:
    1. Predictor: Y_{n+1} = X_n + mu(t_n, X_n) * dt + sigma(t_n, X_n) @ dW_n
    2. Corrector: X_{n+1} = X_n + 0.5 * [mu(t_n, X_n) + mu(t_{n+1}, Y_{n+1})] * dt + sigma(t_n, X_n) @ dW_n
    
    Args:
        drift_function: Function mu(t, x) returning [batch_size, state_dim]
        diffusion_matrix_function: Function sigma(t, x) returning [batch_size, state_dim, noise_dim]
        current_state: Current state [batch_size, state_dim]
        current_time: Current time t
        timestep: Integration timestep dt
        precomputed_noise: Pre-generated noise [batch_size, noise_dim] or None
        
    Returns:
        Next state [batch_size, state_dim]
    """
    device = current_state.device
    batch_size, state_dim = current_state.shape
    
    # Compute current drift: mu(t, x)
    drift_current = drift_function(current_time, current_state)
    if drift_current.dim() == 0:
        drift_current = drift_current.unsqueeze(0).expand(batch_size, -1)
    elif drift_current.dim() == 1:
        drift_current = drift_current.unsqueeze(0).expand(batch_size, -1)
    
    # Compute diffusion matrix: sigma(t, x)
    diffusion_matrix = diffusion_matrix_function(current_time, current_state)
    
    if diffusion_matrix is None:
        # Deterministic case: just second-order Runge-Kutta
        drift_predictor = drift_function(current_time + timestep, current_state + timestep * drift_current)
        return current_state + 0.5 * timestep * (drift_current + drift_predictor)
    
    # Stochastic case with matrix diffusion
    batch_size_check, state_dim_check, noise_dim = diffusion_matrix.shape
    assert batch_size_check == batch_size and state_dim_check == state_dim
    
    # Generate or use precomputed noise (same for predictor and corrector)
    if precomputed_noise is None:
        dW = generate_wiener_increments((batch_size, noise_dim), timestep, device)
    else:
        dW = precomputed_noise
        
    # Compute diffusion update: sigma @ dW (same for both steps)
    diffusion_update = torch.bmm(diffusion_matrix, dW.unsqueeze(-1)).squeeze(-1)
    
    # Predictor step
    predictor_state = current_state + drift_current * timestep + diffusion_update
    
    # Compute drift at predictor point
    drift_predictor = drift_function(current_time + timestep, predictor_state)
    if drift_predictor.dim() == 0:
        drift_predictor = drift_predictor.unsqueeze(0).expand(batch_size, -1)
    elif drift_predictor.dim() == 1:
        drift_predictor = drift_predictor.unsqueeze(0).expand(batch_size, -1)
    
    # Corrector step: average the drift terms
    drift_average = 0.5 * (drift_current + drift_predictor)
    
    return current_state + drift_average * timestep + diffusion_update


def auto_select_matrix_integrator(
    has_diffusion: bool,
) -> Callable[..., torch.Tensor]:
    """
    Automatically select the appropriate matrix-diffusion-aware integrator.
    
    Args:
        has_diffusion: Whether the system includes matrix diffusion terms
        
    Returns:
        Matrix-aware integrator function appropriate for the system type
        
    Example:
        >>> # For neural SDE with matrix diffusion
        >>> integrator = auto_select_matrix_integrator(has_diffusion=True)
        >>> # Returns matrix_stochastic_heun_step
        >>> 
        >>> # For deterministic neural ODE
        >>> integrator = auto_select_matrix_integrator(has_diffusion=False)
        >>> # Returns runge_kutta_2_step (from standard integrators)
    """
    if has_diffusion:
        return matrix_stochastic_heun_step
    else:
        return runge_kutta_2_step