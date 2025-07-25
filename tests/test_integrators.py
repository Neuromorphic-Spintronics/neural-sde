import sys
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).resolve().parents[1]))

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import pytest
from models.integrators import (
    euler_maruyama_step, stochastic_heun_step, euler_maruyama, stochastic_heun_method,
    batch_euler_maruyama_step, batch_euler_maruyama, 
    batch_stochastic_heun_method, batch_integrate_trajectory,
    integrate_trajectory_with_step_method, auto_select_integrator, runge_kutta_2_step,
    generate_wiener_increments
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

def test_euler_traj_batch():
    """Test batch Euler-Maruyama trajectory integration."""
    batch_size = 100
    state_dim = 2
    x0 = torch.randn(batch_size, state_dim)
    t0 = 0.0
    tN = 1.0
    dt = 0.01
    # Integrate full trajectory for batch
    times, trajectories = batch_euler_maruyama(
        simple_drift, simple_diffusion_scalar, x0, t0, tN, dt
    )
    num_steps = int((tN - t0) / dt) + 1
    # Validate time grid
    assert times.shape == (num_steps,)
    # Validate trajectory shape
    assert trajectories.shape == (num_steps, batch_size, state_dim)
    assert torch.allclose(trajectories[0], x0, atol=1e-6)

def test_heun_traj_batch():
    """Test batch stochastic Heun trajectory integration."""
    batch_size = 50
    state_dim = 3
    x0 = torch.randn(batch_size, state_dim)
    t0 = 0.0
    tN = 0.5
    dt = 0.01
    # Batch stochastic Heun trajectory integration
    times, trajectories = batch_stochastic_heun_method(
        simple_drift, simple_diffusion_scalar, x0, t0, tN, dt
    )
    num_steps = int((tN - t0) / dt) + 1
    # Check time steps
    assert times.shape == (num_steps,)
    # Check trajectory dimensions
    assert trajectories.shape == (num_steps, batch_size, state_dim)
    assert torch.allclose(trajectories[0], x0, atol=1e-6)


def test_euler_traj_single():
    """Test trajectory-wide integration using euler_maruyama (non-batch wrapper)."""
    batch_size = 20
    state_dim = 2
    x0 = torch.randn(batch_size, state_dim)
    t0 = 0.0
    tN = 0.5
    dt = 0.01
    # Non‑batch Euler–Maruyama trajectory integration
    times, trajectories = euler_maruyama(simple_drift, simple_diffusion_scalar, x0, t0, tN, dt)
    num_steps = int((tN - t0) / dt) + 1
    # Validate time axis
    assert times.shape == (num_steps,)
    # Validate trajectory shape
    assert trajectories.shape == (num_steps, batch_size, state_dim)
    # Check initial state
    assert torch.allclose(trajectories[0], x0, atol=1e-6)

def test_heun_traj_single():
    """Test stochastic Heun trajectory integration for a single initial condition."""
    # Setup single initial state
    state_dim = 2
    x0 = torch.randn(1, state_dim)
    t0 = 0.0
    tN = 0.5
    dt = 0.01
    # Single trajectory integration
    times, trajectories = stochastic_heun_method(simple_drift, simple_diffusion_scalar, x0, t0, tN, dt)
    num_steps = int((tN - t0) / dt) + 1
    # Validate time axis
    assert times.shape == (num_steps,)
    # Validate trajectory shape
    assert trajectories.shape == (num_steps, 1, state_dim)
    # Initial state check
    assert torch.allclose(trajectories[0], x0, atol=1e-6)

def test_batch_integrate_trajectory():
    """Test batch_integrate_trajectory for deterministic ODE case."""
    # Setup batch initial states
    batch_size = 50
    state_dim = 1
    x0 = torch.ones(batch_size, state_dim)
    t0 = 0.0
    tN = 1.0
    dt = 0.1
    # Integrate trajectory deterministically
    times, trajectories = batch_integrate_trajectory(
        batch_euler_maruyama_step, simple_drift, None, x0, t0, tN, dt
    )
    num_steps = int((tN - t0) / dt) + 1
    # Validate shapes
    assert times.shape == (num_steps,)
    assert trajectories.shape == (num_steps, batch_size, state_dim)
    # Check initial state
    assert torch.allclose(trajectories[0], x0, atol=1e-6)
    # Check final state approximates analytical solution
    expected_final = x0 * torch.exp(-torch.tensor(tN))
    assert torch.allclose(trajectories[-1], expected_final, atol=3e-2)

def test_integrate_trajectory_with_step_method():
    """Test generic trajectory integrator wrapper with Euler-Maruyama step (deterministic)."""
    # Setup initial state
    x0 = torch.tensor([1.0])
    t0 = 0.0
    tN = 0.5
    dt = 0.25
    # Integrate trajectory using wrapper
    times, traj = integrate_trajectory_with_step_method(
        euler_maruyama_step, simple_drift, None, x0, t0, tN, dt
    )
    num_steps = int((tN - t0) / dt) + 1
    # Validate time axis and trajectory shape
    assert times.shape == (num_steps,)
    # Validate trajectory shape: (num_steps, batch_size, state_dim)
    assert traj.shape[0] == num_steps
    assert traj.shape == (num_steps, 1, 1)  # batch_size=1, state_dim=1
    # Check initial state
    assert torch.allclose(traj[0], x0.unsqueeze(0), atol=1e-6)

def test_auto_select_integrator():
    """Test auto_select_integrator picks correct integrator based on flag."""
    # Stochastic selection
    selected_sto = auto_select_integrator(True)
    assert selected_sto is stochastic_heun_step
    # Deterministic selection
    selected_det = auto_select_integrator(False)
    assert selected_det is runge_kutta_2_step


def test_generate_wiener_increments():
    """Test generate_wiener_increments returns correct shape and statistics."""
    shape = (10000, 3)
    dt = 0.05
    dW = generate_wiener_increments(shape, dt, device=torch.device("cpu"))
    # Expect the same shape
    assert dW.shape == shape
    # Mean should be ~0 and std ~sqrt(dt)
    mean = dW.mean().item()
    std = dW.std().item()
    assert abs(mean) < 0.02
    assert abs(std - np.sqrt(dt)) < 0.02

def plot_batch_integration_summary():
    """
    Create a single row of batch integration plots:
    - Ensemble trajectories
    - Mean/variance evolution
    - Performance comparison
    """
    plotting.set_default_plotting_style(use_tex=True)
    fig = plt.figure(figsize=(18, 5), constrained_layout=True)
    gs = gridspec.GridSpec(1, 3, figure=fig, width_ratios=[1, 1, 1])
    ax_ensemble = fig.add_subplot(gs[0, 0])
    ax_statistics = fig.add_subplot(gs[0, 1])
    ax_performance = fig.add_subplot(gs[0, 2])
    # Ensemble trajectories
    batch_size = 200
    x0 = torch.ones(batch_size, 1)
    t0, tN, dt = 0.0, 2.0, 0.01
    sigma = 0.5
    times, trajectories = batch_euler_maruyama(
        simple_drift, lambda t, x: simple_diffusion_scalar(t, x, strength=sigma), 
        x0, t0, tN, dt
    )
    traj_np = trajectories.squeeze(-1).numpy()
    times_np = times.numpy()
    n_plot = 50
    indices = np.random.choice(batch_size, n_plot, replace=False)
    for i in indices:
        ax_ensemble.plot(times_np, traj_np[:, i], alpha=0.3, color=COLOURS[3], linewidth=0.8)
    mean_traj = traj_np.mean(axis=1)
    std_traj = traj_np.std(axis=1)
    ax_ensemble.plot(times_np, mean_traj, color='k', linewidth=2, label='Ensemble mean')
    ax_ensemble.fill_between(times_np, mean_traj - 2*std_traj, mean_traj + 2*std_traj, 
                            alpha=0.2, color='k', label=r'$\pm 2\sigma$ envelope')
    analytical_mean = np.exp(-times_np)
    ax_ensemble.plot(times_np, analytical_mean, '--', color='red', linewidth=2, 
                    label='Analytical mean')
    ax_ensemble.set_xlabel(r'$t$')
    ax_ensemble.set_ylabel(r'$x(t)$')
    ax_ensemble.set_title('Batch Ensemble Trajectories')
    ax_ensemble.legend()
    plotting.style_axis_clean(ax_ensemble)
    # Mean and variance evolution
    batch_size_large = 2000
    x0_large = torch.ones(batch_size_large, 1)
    times_large, traj_large = batch_stochastic_heun_method(
        simple_drift, lambda t, x: simple_diffusion_scalar(t, x, strength=sigma),
        x0_large, t0, tN, dt
    )
    traj_large_np = traj_large.squeeze(-1).numpy()
    times_large_np = times_large.numpy()
    mean_large = traj_large_np.mean(axis=1)
    var_large = traj_large_np.var(axis=1)
    analytical_mean_large = np.exp(-times_large_np)
    analytical_var = (sigma**2 / 2) * (1 - np.exp(-2 * times_large_np))
    ax_statistics.plot(times_large_np, mean_large, color=COLOURS[0], linewidth=2, 
                      label='Numerical mean')
    ax_statistics.plot(times_large_np, analytical_mean_large, '--', color=COLOURS[0], 
                      linewidth=2, alpha=0.7, label='Analytical mean')
    ax_statistics.plot(times_large_np, var_large, color=COLOURS[1], linewidth=2, 
                      label='Numerical variance')
    ax_statistics.plot(times_large_np, analytical_var, '--', color=COLOURS[1], 
                      linewidth=2, alpha=0.7, label='Analytical variance')
    ax_statistics.set_xlabel(r'$t$')
    ax_statistics.set_ylabel('Value')
    ax_statistics.set_title('Ensemble Statistics')
    ax_statistics.legend()
    plotting.style_axis_clean(ax_statistics)
    # Performance comparison
    batch_sizes = [10, 50, 100, 500, 1000, 2000]
    batch_times = []
    single_times = []
    for bs in batch_sizes:
        x0_test = torch.randn(bs, 2)
        t_test, tN_test, dt_test = 0.0, 0.1, 0.01
        start = time.time()
        _ = batch_euler_maruyama(simple_drift, simple_diffusion_scalar, 
                                x0_test, t_test, tN_test, dt_test)
        batch_time = time.time() - start
        batch_times.append(batch_time)
        start = time.time()
        for _ in range(bs):
            _ = euler_maruyama(simple_drift, simple_diffusion_scalar, 
                             torch.randn(1, 2), t_test, tN_test, dt_test)
        single_time = time.time() - start
        single_times.append(single_time)
    ax_performance.loglog(batch_sizes, batch_times, 'o-', color=COLOURS[2], 
                         linewidth=2, markersize=6, label='Batch method')
    ax_performance.loglog(batch_sizes, single_times, 's-', color=COLOURS[4], 
                         linewidth=2, markersize=6, label='Sequential method')
    ax_performance.set_xlabel('Batch size')
    ax_performance.set_ylabel('Computation time (s)')
    ax_performance.set_title('Performance Comparison')
    ax_performance.legend()
    ax_performance.grid(True, alpha=0.3)
    plotting.style_axis_clean(ax_performance)
    outdir = os.path.join(os.path.dirname(__file__), "figures")
    os.makedirs(outdir, exist_ok=True)
    fig.savefig(os.path.join(outdir, "batch_integration_summary.pdf"), bbox_inches="tight")
    plt.close(fig)


def main():
    print("Running batch integration plots...")
    plot_batch_integration_summary()
    print("Plot saved to tests/figures/batch_integration_summary.pdf.")
    print("Running pytest...")
    pytest.main([__file__])

if __name__ == "__main__":
    main()