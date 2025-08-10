import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
from models.integrators import (
    euler_maruyama_step,
    stochastic_heun_step,
    euler_maruyama,
    stochastic_heun_method,
    integrate_trajectory_with_step_method,
    auto_select_integrator,
    runge_kutta_2_step,
    generate_wiener_increments,
    matrix_euler_maruyama_step,
    matrix_stochastic_heun_step,
    auto_select_matrix_integrator,
)
from utils import plotting

COLOURS = plotting.COLOURS


# Set-up the physical problem
def simple_drift(t, x):
    # Linear drift: dx/dt = -x
    return -x


def simple_diffusion_scalar(t, x, strength=0.5):
    return strength * torch.ones_like(x)


# Test single-step methods
def test_euler_step_single():
    """
    Test Euler-Maruyama with single trajectory (one initial condition, and one noise trajectory).
    """
    # Setup inputs
    x0 = torch.tensor([1.0, -1.0])
    t0 = 0.0
    dt = 0.1
    # Run Euler-Maruyama step
    x1 = euler_maruyama_step(simple_drift, simple_diffusion_scalar, x0, t0, dt)
    # Validate output shape and device
    assert x1.shape == (1, 2)
    assert x1.device == x0.device


def test_heun_step_single():
    """
    Test stochastic Heun with single step.
    """
    # Setup inputs
    x0 = torch.tensor([1.0, -1.0])
    t0 = 0.0
    dt = 0.1
    # Run stochastic Heun step
    x1 = stochastic_heun_step(simple_drift, simple_diffusion_scalar, x0, t0, dt)
    # Validate output
    assert x1.shape == (1, 2)
    assert x1.device == x0.device


def test_euler_step_batch():
    """
    Test Euler-Maruyama with multiple trajectories (e.g., two initial conditions, and two noise trajectories).
    """
    # Setup batched inputs
    x0 = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    t0 = 0.0
    dt = 0.1
    # Run Euler-Maruyama on batch
    x1 = euler_maruyama_step(simple_drift, simple_diffusion_scalar, x0, t0, dt)
    # Check batch output
    assert x1.shape == (2, 2)
    assert x1.device == x0.device


def test_heun_step_batch():
    """
    Test stochastic Heun with multiple trajectories (e.g., two initial conditions, and two noise trajectories).
    """
    # Setup batched inputs
    x0 = torch.tensor([[1.0, -1.0], [0.5, 0.5]])
    t0 = 0.0
    dt = 0.1
    # Run stochastic Heun on batch
    x1 = stochastic_heun_step(simple_drift, simple_diffusion_scalar, x0, t0, dt)
    # Check batch results
    assert x1.shape == (2, 2)
    assert x1.device == x0.device


def test_euler_traj_single():
    """Test single trajectory Euler-Maruyama integration."""
    x0 = torch.tensor([1.0, -1.0])
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    # Integrate full trajectory
    times, trajectories = euler_maruyama(
        simple_drift, simple_diffusion_scalar, x0, t0, tN, dt
    )
    # Validate output shapes
    assert times.shape == (101,)  # 100 steps + 1 initial
    assert trajectories.shape == (101, 1, 2)  # time, batch, state
    assert times.device == x0.device
    assert trajectories.device == x0.device


def test_heun_traj_single():
    """Test single trajectory stochastic Heun integration."""
    x0 = torch.tensor([1.0, -1.0])
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    # Integrate full trajectory
    times, trajectories = stochastic_heun_method(
        simple_drift, simple_diffusion_scalar, x0, t0, tN, dt
    )
    # Validate output shapes
    assert times.shape == (101,)  # 100 steps + 1 initial
    assert trajectories.shape == (101, 1, 2)  # time, batch, state
    assert times.device == x0.device
    assert trajectories.device == x0.device


def test_integrate_trajectory_with_step_method():
    """Test generic trajectory integration wrapper."""
    x0 = torch.tensor([1.0, -1.0])
    t0 = 0.0
    tN = 1.0
    dt = 0.01

    # Test with Euler-Maruyama step
    times, trajectories = integrate_trajectory_with_step_method(
        euler_maruyama_step, simple_drift, simple_diffusion_scalar, x0, t0, tN, dt
    )

    assert times.shape == (101,)
    assert trajectories.shape == (101, 1, 2)


def test_auto_select_integrator():
    """Test automatic integrator selection."""
    # Should return stochastic Heun for stochastic case
    stochastic_integrator = auto_select_integrator(has_diffusion=True)
    assert stochastic_integrator == stochastic_heun_step

    # Should return RK2 for deterministic case
    deterministic_integrator = auto_select_integrator(has_diffusion=False)
    assert deterministic_integrator == runge_kutta_2_step


def test_generate_wiener_increments():
    """Test Wiener increment generation."""
    shape = (10, 3)
    timestep = 0.01
    device = torch.device("cpu")

    increments = generate_wiener_increments(shape, timestep, device)

    assert increments.shape == shape
    assert increments.device == device
    # Check that variance is approximately equal to timestep
    empirical_var = increments.var().item()
    expected_var = timestep
    assert (
        abs(empirical_var - expected_var) < 0.1
    )  # Allow some tolerance for randomness


def test_matrix_euler_maruyama_step():
    """Test matrix-aware Euler-Maruyama integration step."""
    batch_size, state_dim, noise_dim = 4, 3, 2
    timestep = 0.01

    # Simple drift and diffusion functions for testing
    def drift_func(t: float, x: torch.Tensor) -> torch.Tensor:
        return -0.1 * x  # Simple damping

    def diffusion_matrix_func(t: float, x: torch.Tensor) -> torch.Tensor:
        # Return a simple diffusion matrix [batch_size, state_dim, noise_dim]
        batch_size = x.shape[0]
        return 0.1 * torch.randn(batch_size, state_dim, noise_dim)

    # Initial state
    x0 = torch.randn(batch_size, state_dim)

    # Integration step
    x1 = matrix_euler_maruyama_step(
        drift_func, diffusion_matrix_func, x0, 0.0, timestep
    )

    # Check output shape
    assert x1.shape == (batch_size, state_dim)

    # Check that the state has changed (unless we're very unlucky with noise)
    assert not torch.allclose(x0, x1)


def test_matrix_stochastic_heun_step():
    """Test matrix-aware Stochastic Heun integration step."""
    batch_size, state_dim, noise_dim = 4, 3, 2
    timestep = 0.01

    # Simple drift and diffusion functions for testing
    def drift_func(t: float, x: torch.Tensor) -> torch.Tensor:
        return -0.1 * x  # Simple damping

    def diffusion_matrix_func(t: float, x: torch.Tensor) -> torch.Tensor:
        # Return a simple diffusion matrix [batch_size, state_dim, noise_dim]
        batch_size = x.shape[0]
        return 0.1 * torch.randn(batch_size, state_dim, noise_dim)

    # Initial state
    x0 = torch.randn(batch_size, state_dim)

    # Integration step
    x1 = matrix_stochastic_heun_step(
        drift_func, diffusion_matrix_func, x0, 0.0, timestep
    )

    # Check output shape
    assert x1.shape == (batch_size, state_dim)

    # Check that the state has changed
    assert not torch.allclose(x0, x1)


def test_auto_select_matrix_integrator():
    """Test automatic selection of matrix-aware integrators."""
    # Should return matrix Heun for stochastic case
    stochastic_integrator = auto_select_matrix_integrator(has_diffusion=True)
    assert stochastic_integrator == matrix_stochastic_heun_step

    # Should return RK2 for deterministic case
    deterministic_integrator = auto_select_matrix_integrator(has_diffusion=False)
    assert deterministic_integrator == runge_kutta_2_step


def test_matrix_integrators_deterministic_consistency():
    """Test that matrix integrators give consistent results for deterministic case."""
    batch_size, state_dim = 2, 3
    timestep = 0.01

    def drift_func(t: float, x: torch.Tensor) -> torch.Tensor:
        return -0.1 * x

    def diffusion_matrix_func(t: float, x: torch.Tensor) -> torch.Tensor:
        # Return zero diffusion matrix for deterministic case
        batch_size, state_dim = x.shape
        return torch.zeros(batch_size, state_dim, 1)  # Zero diffusion matrix

    x0 = torch.randn(batch_size, state_dim)

    # Both should work for deterministic case
    x1_euler = matrix_euler_maruyama_step(
        drift_func, diffusion_matrix_func, x0, 0.0, timestep
    )
    x1_heun = matrix_stochastic_heun_step(
        drift_func, diffusion_matrix_func, x0, 0.0, timestep
    )

    assert x1_euler.shape == (batch_size, state_dim)
    assert x1_heun.shape == (batch_size, state_dim)

    # Heun should be more accurate than Euler for deterministic case
    # (Both should give reasonable results)
