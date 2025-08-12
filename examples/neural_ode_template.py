"""
Generic neural ODE template for 1D second-order stochastic systems.

This template provides a framework for training neural ODEs on 1D second-order stochastic differential equations with multiple noisy realisations. It will produce a trained model representing the mean dynamics of the system.

ADAPTATION INSTRUCTIONS:
1. Replace placeholder system parameters and classes with your own
2. Modify the external forcing function (if any) for your system (or set HAS_EXTERNAL_FORCING to False)
3. Update physical units and characteristic scales
4. Adjust visualisation labels
5. Configure system-specific parameters in the configuration section
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple, Callable
from dataclasses import dataclass
import math
import os
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from torch import Tensor  # noqa: E402
from tqdm import tqdm  # noqa: E402
import numpy as np  # noqa: E402

from models.neural_sde import DriftNet  # noqa: E402
from utils.plotting import style_axis_clean, set_default_plotting_style  # noqa: E402
from parameters.hyperparameters import NetworkArchitecture  # noqa: E402
from models.integrators import (
    auto_select_integrator,
    integrate_trajectory_with_step_method,
)
from config import DEVICE  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, MaxNLocator  # noqa: E402

# TODO: Replace these imports with your system-specific classes
from parameters import (
    PhysicalDuffingParameters,
)  # REPLACE: Your parameter class  # noqa: E402
from systems.duffing_oscillator import (
    DuffingOscillator,
)  # REPLACE: Your system class  # noqa: E402


# TODO: Update these settings for your specific system
SYSTEM_NAME = "your_system"

DEFAULT_HIDDEN_LAYERS = [128, 128, 64]

# Set to True if your system has external driving: u(t) = f(t)
# Set to False for autonomous systems (no external forcing)
HAS_EXTERNAL_FORCING = True  # TODO: Set appropriately for your system

STATE_DIMENSION = 2

EXTERNAL_INPUT_DIMENSION = 1 if HAS_EXTERNAL_FORCING else 0
TOTAL_INPUT_DIMENSION = STATE_DIMENSION + 1 + EXTERNAL_INPUT_DIMENSION

# Physical parameters - TODO: Replace with your system defaults
DEFAULT_TOTAL_TIME = 25.0  # Simulation time in dimensionless units
DEFAULT_TIMESTEP = 0.01  # Integration timestep


def runge_kutta_4_step(
    drift_function: Callable[[float | Tensor, Tensor], Tensor],
    current_state: Tensor,
    current_time: float | Tensor,
    timestep: float,
) -> Tensor:
    """Perform one step of the 4th-order Runge-Kutta method for ODEs."""
    k1 = timestep * drift_function(current_time, current_state)
    k2 = timestep * drift_function(current_time + timestep / 2, current_state + k1 / 2)
    k3 = timestep * drift_function(current_time + timestep / 2, current_state + k2 / 2)
    k4 = timestep * drift_function(current_time + timestep, current_state + k3)
    next_state = current_state + (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return next_state


@dataclass(frozen=True)
class NeuralODETrainingConfig:
    """Configuration for neural ODE training."""

    learning_rate: float = 1e-2
    batch_size: int = 8
    num_epochs: int = 50
    max_batches_per_epoch: int | None = None


def build_neural_ode_drift(
    *,
    hidden_layer_sizes: list[int] | None = None,
    system_name: str | None = None,
) -> DriftNet:
    """Construct a `DriftNet` for your system's neural ODE.

    The network learns the drift function f(x, t, u) where:
    - x = [position, velocity] (state vector)
    - t = time
    - u = external forcing (if applicable)

    Args:
        hidden_layer_sizes: Network architecture
        system_name: System identifier

    Returns:
        Configured DriftNet
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = DEFAULT_HIDDEN_LAYERS

    if system_name is None:
        system_name = SYSTEM_NAME

    network_architecture = NetworkArchitecture(
        input_size=TOTAL_INPUT_DIMENSION,
        hidden_sizes=hidden_layer_sizes,
        output_size=STATE_DIMENSION,
    )
    drift_net = DriftNet(architecture=network_architecture, device=DEVICE)
    return drift_net


def _get_default_physical_parameters():
    """TODO: Replace with your system's parameter class and defaults."""
    # TODO: Replace with your parameter class
    return PhysicalDuffingParameters(
        total_time=DEFAULT_TOTAL_TIME,
        timestep=DEFAULT_TIMESTEP,
    )


def _compute_external_forcing(
    amplitude: float | Tensor, frequency: float | Tensor, time: Tensor
) -> Tensor:
    """TODO: Implement your system's external forcing function.

    Examples:
    - Duffing oscillator: amplitude * torch.cos(frequency * time)
    - Driven pendulum: amplitude * torch.sin(frequency * time)
    - Custom periodic: implement your f(t)
    - If HAS_EXTERNAL_FORCING is False, the code auto returns 0.0

    Args:
        amplitude: Forcing strength parameter
        frequency: Forcing frequency parameter
        time: Time value (can be tensor)

    Returns:
        External forcing value u(t)
    """
    if not HAS_EXTERNAL_FORCING:
        return torch.zeros_like(time)

    # TODO: Implement your forcing function
    # Default: cosine forcing
    return amplitude * torch.cos(frequency * time)


def _compute_drift_with_external_input(
    drift_net: DriftNet,
    *,
    amplitude: float,
    frequency: float,
    time: float,
    state: Tensor,
) -> Tensor:
    """Compute neural ODE drift with external forcing."""
    if state.ndim == 1:
        state_batched = state.unsqueeze(0)
    else:
        state_batched = state

    time_column = torch.full_like(state_batched[:, :1], float(time))

    if HAS_EXTERNAL_FORCING:
        external_forcing = _compute_external_forcing(amplitude, frequency, time_column)
        return drift_net.compute_drift(state_batched, time_column, external_forcing)
    else:
        return drift_net.compute_drift(state_batched, time_column)


def generate_stochastic_dataset(
    *,
    num_trajectories: int = 16,
    total_time: float | None = None,
    dt: float | None = None,
    seed: int = 42069,
    with_noise: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Generate dataset of stochastic trajectories (delegates to batched generator)."""
    if total_time is None:
        total_time = DEFAULT_TOTAL_TIME
    if dt is None:
        dt = DEFAULT_TIMESTEP

    time_grid, trajectories = _generate_batch_stochastic_dataset(
        num_trajectories=num_trajectories,
        total_time=total_time,
        dt=dt,
        batch_size=min(num_trajectories, 8),
        seed=seed,
        system_name=SYSTEM_NAME,
        with_noise=with_noise,
    )
    return time_grid, trajectories


def _generate_batch_stochastic_dataset(
    *,
    num_trajectories: int = 16,
    total_time: float = 25.0,
    dt: float = 0.01,
    batch_size: int = 8,
    seed: int = 42069,
    system_name: str = SYSTEM_NAME,
    with_noise: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Optimised trajectory generation.

    - On GPU/MPS: vectorised batches as before
    - On CPU: parallelise across all available cores to generate batches concurrently
    """
    torch.manual_seed(seed)

    # TODO: Replace with your parameter class
    physical_parameters = _get_default_physical_parameters()
    object.__setattr__(physical_parameters, "total_time", total_time)
    object.__setattr__(physical_parameters, "timestep", dt)

    if not with_noise:
        print("   Generating deterministic dataset (noise turned off).")
        object.__setattr__(physical_parameters, "temperature", 0.0)
        object.__setattr__(physical_parameters, "coloured_noise_intensity", 0.0)

    dimensionless_parameters = physical_parameters.to_dimensionless()

    num_steps = int(total_time / dt)
    time_grid = torch.linspace(0, total_time, num_steps + 1, device=DEVICE)

    # TODO: Adjust state dimension if your system has different dimensionality
    state_dim = 3  # [position, velocity, noise_state] - adjust for your system
    all_trajectories = torch.zeros(
        num_trajectories, num_steps + 1, state_dim, device=DEVICE
    )

    num_batches = math.ceil(num_trajectories / batch_size)

    device_str = str(DEVICE).lower()
    running_on_cpu = "cpu" in device_str

    if running_on_cpu:
        num_workers = max(1, min(os.cpu_count() or 1, num_trajectories))
        base = num_trajectories // num_workers
        remainder = num_trajectories % num_workers
        chunk_sizes = [base + (1 if i < remainder else 0) for i in range(num_workers)]

        with tqdm(
            total=num_trajectories,
            desc=f"Generating {system_name} trajectories",
            unit="traj",
        ) as pbar:
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures: list[tuple] = []
                start_offset = 0
                for local_chunk_size in chunk_sizes:
                    if local_chunk_size == 0:
                        continue
                    worker_seed = seed + start_offset
                    fut = executor.submit(
                        _integrate_batch_sde_trajectories,
                        dimensionless_parameters,
                        local_chunk_size,
                        time_grid.cpu(),
                        dt,
                        "cpu",
                        worker_seed,
                    )
                    futures.append((fut, start_offset, local_chunk_size))
                    start_offset += local_chunk_size

                for fut, start_offset, local_chunk_size in futures:
                    batch_trajectories = fut.result()
                    all_trajectories[start_offset : start_offset + local_chunk_size] = (
                        batch_trajectories
                    )
                    pbar.update(local_chunk_size)

    else:
        with tqdm(
            total=num_trajectories,
            desc=f"Generating {system_name} trajectories",
            unit="traj",
        ) as pbar:
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_trajectories)
                current_batch_size = end_idx - start_idx

                batch_trajectories = _integrate_batch_sde_trajectories(
                    dimensionless_parameters, current_batch_size, time_grid, dt
                )
                all_trajectories[start_idx:end_idx] = batch_trajectories
                pbar.update(current_batch_size)

    return time_grid, all_trajectories


def _integrate_batch_sde_trajectories(
    dimensionless_parameters,
    batch_size: int,
    time_grid: Tensor,
    dt: float,
    device: str | torch.device = DEVICE,
    random_seed: int | None = None,
) -> Tensor:
    """Integrate multiple SDE trajectories in parallel on GPU.

    TODO: Replace with your system's integration logic.
    """
    if random_seed is not None:
        torch.manual_seed(random_seed)

    device_obj = (
        torch.device(device) if not isinstance(device, torch.device) else device
    )
    num_steps = len(time_grid) - 1

    # TODO: Adjust initial conditions for your system
    initial_states = torch.zeros(
        batch_size, 3, device=device_obj
    )  # [position, velocity, noise_state]

    # TODO: Replace with your system class
    system = DuffingOscillator(dimensionless_parameters)
    trajectories = torch.zeros(
        batch_size, num_steps + 1, 3, device=device_obj
    )  # Adjust state dimension
    trajectories[:, 0] = initial_states
    current_states = initial_states.clone()

    step_integrator = auto_select_integrator(has_diffusion=True)
    for step in range(num_steps):
        current_time = float(time_grid[step].item())
        next_states = step_integrator(
            lambda t, x: system.drift_function(t, x),
            lambda t, x: system.diffusion_function(t, x),
            current_states,
            current_time,
            dt,
        )
        trajectories[:, step + 1] = next_states
        current_states = next_states

    return trajectories


def simulate_neural_ode_trajectory(
    drift_net: DriftNet,
    *,
    x0: Tensor,
    t0: float,
    tN: float,
    dt: float,
    amplitude: float | None = None,
    frequency: float | None = None,
) -> Tuple[Tensor, Tensor]:
    """Simulate trajectory using the trained neural ODE."""
    if amplitude is None or frequency is None:
        # TODO: Replace with your parameter defaults
        default_parameters = _get_default_physical_parameters().to_dimensionless()
        amplitude = default_parameters.gamma if amplitude is None else amplitude
        frequency = default_parameters.Omega if frequency is None else frequency

    def drift_function(time: float, current_state: Tensor) -> Tensor:
        return _compute_drift_with_external_input(
            drift_net,
            amplitude=float(amplitude),
            frequency=float(frequency),
            time=time,
            state=current_state,
        )

    # Select appropriate integrator (RK4 for deterministic neural ODE)
    selected_integrator = runge_kutta_4_step

    time_grid, trajectory = integrate_trajectory_with_step_method(
        step_integrator=selected_integrator,
        drift_function=drift_function,
        diffusion_function=None,
        initial_state=x0.to(DEVICE),
        initial_time=t0,
        final_time=tN,
        timestep=dt,
    )
    return time_grid, trajectory


def _optimised_fit_neural_ode(
    drift_net: DriftNet,
    time_grid: Tensor,
    stochastic_trajectories_with_noise: Tensor,
    *,
    learning_rate: float = 2e-3,
    batch_size: int | None = None,
    num_epochs: int = 100,
    random_seed: int = 0,
    system_name: str = SYSTEM_NAME,
) -> List[float]:
    """Optimised neural ODE training."""
    torch.manual_seed(random_seed)

    num_trajectories = stochastic_trajectories_with_noise.shape[0]
    if batch_size is None:
        batch_size = min(num_trajectories, 16)

    optimiser = torch.optim.Adam(drift_net.parameters(), lr=learning_rate)
    drift_net.train()

    position_velocity_trajectories = stochastic_trajectories_with_noise[..., :2].to(
        DEVICE
    )
    time_grid = time_grid.to(DEVICE)
    time_step_size = (time_grid[1] - time_grid[0]).item()

    num_trajectories, num_time_points, state_dimension = (
        position_velocity_trajectories.shape
    )

    # TODO: Replace with your parameter class
    default_parameters = _get_default_physical_parameters().to_dimensionless()
    forcing_amplitude = float(default_parameters.gamma) if HAS_EXTERNAL_FORCING else 0.0
    forcing_frequency = float(default_parameters.Omega) if HAS_EXTERNAL_FORCING else 0.0
    integration_times = time_grid[:-1]

    training_losses: List[float] = []

    # Select the integrator for the deterministic neural ODE
    step_integrator = runge_kutta_4_step

    def vectorised_drift(t, states):
        if HAS_EXTERNAL_FORCING:
            forcing = forcing_amplitude * torch.cos(forcing_frequency * t)
            return drift_net.compute_drift(states, t, forcing)
        else:
            return drift_net.compute_drift(states, t)

    with tqdm(
        range(num_epochs), desc=f"Training {system_name} Neural ODE", unit="epoch"
    ) as epoch_pbar:
        for epoch in epoch_pbar:
            trajectory_indices = torch.randperm(num_trajectories, device=DEVICE)
            epoch_losses = []
            num_batches = math.ceil(num_trajectories / batch_size)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, num_trajectories)
                actual_batch_size = end_idx - start_idx

                batch_indices = trajectory_indices[start_idx:end_idx]
                batch_trajectories = position_velocity_trajectories[batch_indices]

                current_states = batch_trajectories[:, :-1].reshape(-1, state_dimension)
                target_states = batch_trajectories[:, 1:].reshape(-1, state_dimension)

                batch_times = (
                    integration_times.unsqueeze(0)
                    .expand(actual_batch_size, -1)
                    .reshape(-1, 1)
                )

                # Use the selected integrator to take one step. This is much faster
                # than looping over time steps due to vectorisation.
                predicted_next_states = step_integrator(
                    vectorised_drift, current_states, batch_times, time_step_size
                )

                loss = torch.mean((predicted_next_states - target_states).pow(2))

                optimiser.zero_grad()
                loss.backward()
                optimiser.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            training_losses.append(avg_loss)
            epoch_pbar.set_postfix({"Loss": f"{avg_loss:.2e}"})

    return training_losses


def plot_trajectory_comparison(
    time_grid: Tensor,
    stochastic_trajectories: Tensor,
    neural_ode_time: Tensor,
    neural_ode_trajectory: Tensor,
    rk_time: Tensor | None = None,
    rk_trajectory: Tensor | None = None,
    *,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot comparison between stochastic trajectories and neural ODE prediction.

    TODO: Update axis labels, units, and conversion factors for your system.
    """
    set_default_plotting_style(use_tex=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7, 5), sharex=True)

    physical_params = _get_default_physical_parameters()
    physical_params.to_dimensionless()  # This computes and sets the characteristic scales

    tau = physical_params.characteristic_time
    t_stoch = time_grid.detach().cpu().numpy() * tau
    t_neural = neural_ode_time.detach().cpu().numpy() * tau

    trajectories = stochastic_trajectories.detach().cpu().numpy()
    neural_traj = neural_ode_trajectory.squeeze(1).detach().cpu().numpy()

    lambda_ = physical_params.characteristic_length

    trajectories[:, :, 0] = trajectories[:, :, 0] * lambda_
    neural_traj[:, 0] = neural_traj[:, 0] * lambda_
    trajectories[:, :, 1] = trajectories[:, :, 1] * (lambda_ / tau)
    neural_traj[:, 1] = neural_traj[:, 1] * (lambda_ / tau)

    if rk_time is not None and rk_trajectory is not None:
        t_rk_phys = rk_time.detach().cpu().numpy() * tau
        rk_traj_phys = rk_trajectory.squeeze(1).detach().cpu().numpy()
        rk_traj_phys[:, 0] *= lambda_
        rk_traj_phys[:, 1] *= lambda_ / tau
    else:
        t_rk_phys, rk_traj_phys = None, None

    for i in range(trajectories.shape[0]):
        label = "Stochastic realisations" if i == 0 else None
        ax1.plot(
            t_stoch,
            trajectories[i, :, 0],
            color="black",
            alpha=0.15,
            linewidth=0.6,
            label=label,
        )

    ax1.plot(
        t_neural, neural_traj[:, 0], color="black", linewidth=1.4, label="Neural ODE"
    )
    # Optional: overlay RK trajectory (position) as dashed
    if t_rk_phys is not None and rk_traj_phys is not None:
        ax1.plot(
            t_rk_phys,
            rk_traj_phys[:, 0],
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="RK (true)",
        )

    ax1.set_ylabel(r"$q(t)$ [m]")
    ax1.legend(frameon=False)
    style_axis_clean(ax1)

    for i in range(trajectories.shape[0]):
        label = "Stochastic realisations" if i == 0 else None
        ax2.plot(
            t_stoch,
            trajectories[i, :, 1],
            color="black",
            alpha=0.15,
            linewidth=0.6,
            label=label,
        )

    ax2.plot(
        t_neural, neural_traj[:, 1], color="black", linewidth=1.4, label="Neural ODE"
    )
    # Optional: overlay RK trajectory (velocity) as dashed
    if t_rk_phys is not None and rk_traj_phys is not None:
        ax2.plot(
            t_rk_phys,
            rk_traj_phys[:, 1],
            color="black",
            linewidth=1.2,
            linestyle="--",
            label="RK (true)",
        )

    ax2.set_xlabel(r"$t$ [s]")
    ax2.set_ylabel(r"$v(t)$ [m s$^{-1}$]")
    ax2.legend(frameon=False)
    style_axis_clean(ax2)

    pos_all = np.concatenate([trajectories[:, :, 0].ravel(), neural_traj[:, 0].ravel()])
    vel_all = np.concatenate([trajectories[:, :, 1].ravel(), neural_traj[:, 1].ravel()])
    pos_max = float(np.max(np.abs(pos_all)))
    vel_max = float(np.max(np.abs(vel_all)))
    pos_lim = 1.05 * pos_max if pos_max > 0 else 1.0
    vel_lim = 1.05 * vel_max if vel_max > 0 else 1.0
    ax1.set_ylim(-pos_lim, pos_lim)
    ax2.set_ylim(-vel_lim, vel_lim)

    ax1.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
    ax2.yaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_training_loss(
    training_losses: List[float],
    *,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot training loss evolution."""
    set_default_plotting_style(use_tex=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    epochs = range(1, len(training_losses) + 1)
    ax.plot(epochs, training_losses, color="black", linewidth=1.4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_yscale("log")

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=(2, 3, 4, 5, 6, 7, 8, 9), numticks=12)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis_clean(ax)

    final_loss = training_losses[-1]
    print(f"Final loss: {final_loss:.2e}")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def plot_phase_space_comparison(
    stochastic_trajectories: Tensor,
    neural_ode_trajectory: Tensor,
    *,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot phase space (position vs velocity) comparison.

    TODO: Update axis labels and units for your system.
    """
    set_default_plotting_style(use_tex=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    trajectories = stochastic_trajectories.detach().cpu().numpy()
    neural_traj = neural_ode_trajectory.squeeze(1).detach().cpu().numpy()

    physical_params = _get_default_physical_parameters()
    physical_params.to_dimensionless()  # This computes and sets the characteristic scales
    lambda_ = physical_params.characteristic_length
    tau = physical_params.characteristic_time

    trajectories[:, :, 0] = trajectories[:, :, 0] * lambda_
    neural_traj[:, 0] = neural_traj[:, 0] * lambda_
    trajectories[:, :, 1] = trajectories[:, :, 1] * (lambda_ / tau)
    neural_traj[:, 1] = neural_traj[:, 1] * (lambda_ / tau)

    for i in range(trajectories.shape[0]):
        label = "Stochastic trajectories" if i == 0 else None
        ax.plot(
            trajectories[i, :, 0],
            trajectories[i, :, 1],
            color="black",
            alpha=0.05,
            linewidth=0.8,
            label=label,
        )

    ax.plot(
        neural_traj[:, 0],
        neural_traj[:, 1],
        color="black",
        linewidth=1.6,
        label="Neural ODE",
    )

    ax.set_xlabel(r"$q(t)$ [m]")
    ax.set_ylabel(r"$v(q(t))$ [m s$^{-1}$]")
    ax.legend(frameon=False)
    ax.set_aspect("equal")  # Equal aspect ratio for phase space
    style_axis_clean(ax)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def create_neural_ode_demonstration(
    *,
    num_trajectories: int = 16,
    total_time: float = 25.0,
    timestep: float = 0.01,
    hidden_layer_sizes: list[int] | None = None,
    num_epochs: int = 100,
    learning_rate: float = 2e-3,
    save_directory: str | Path = "examples/figures",
    show_plots: bool = False,
    trajectory_batch_size: int = 8,
    training_batch_size: int = 16,
    with_noise: bool = True,
    system_name: str | None = None,
) -> Tuple[DriftNet, List[float], dict]:
    """
    Complete neural ODE demonstration.

    This function:
    1. Generates stochastic training data
    2. Builds and trains the neural ODE
    3. Creates publication-quality visualisations

    Args:
        num_trajectories: Number of stochastic trajectories to generate
        total_time: Total simulation time (dimensionless)
        timestep: Integration timestep (dimensionless)
        hidden_layer_sizes: Neural network architecture
        num_epochs: Number of training epochs
        learning_rate: Optimiser learning rate
        save_directory: Directory to save figures
        show_plots: Whether to display plots interactively
        trajectory_batch_size: Batch size for parallel trajectory generation
        training_batch_size: Batch size for training
        with_noise: Whether to include stochastic noise in the training data

    Returns:
        Tuple of (trained_drift_net, training_losses, results_summary)
    """
    if hidden_layer_sizes is None:
        hidden_layer_sizes = DEFAULT_HIDDEN_LAYERS
    if system_name is None:
        system_name = SYSTEM_NAME

    save_dir = Path(save_directory)
    save_dir.mkdir(exist_ok=True)

    print(f"Starting {system_name} Neural ODE Demonstration")
    print("=" * 70)
    print("Configuration:")
    print(f"   System: {system_name}")
    print(f"   Trajectories: {num_trajectories}")
    print(f"   Time span: {total_time} units ({int(total_time / timestep)} steps)")
    print(f"   Architecture: {hidden_layer_sizes}")
    print(f"   Training: {num_epochs} epochs, lr={learning_rate}")
    print(f"   Device: {DEVICE}")
    print(f"   External forcing: {'Yes' if HAS_EXTERNAL_FORCING else 'No'}")
    print(f"   Stochastic Noise: {'Enabled' if with_noise else 'Disabled'}")
    print("=" * 70)

    # 1. Generate training data
    print("Data Generation...")
    time_grid, stochastic_trajectories = _generate_batch_stochastic_dataset(
        num_trajectories=num_trajectories,
        total_time=total_time,
        dt=timestep,
        batch_size=trajectory_batch_size,
        seed=42,
        system_name=system_name,
        with_noise=with_noise,
    )
    print(
        f"   Generated {stochastic_trajectories.shape[0]} trajectories of shape {tuple(stochastic_trajectories.shape)}"
    )

    # 2. Build neural network
    print("Network Construction...")
    drift_network = build_neural_ode_drift(
        hidden_layer_sizes=hidden_layer_sizes, system_name=system_name
    )

    # 3. Training
    print("Neural ODE Training...")
    training_losses = _optimised_fit_neural_ode(
        drift_network,
        time_grid=time_grid,
        stochastic_trajectories_with_noise=stochastic_trajectories,
        learning_rate=learning_rate,
        batch_size=training_batch_size,
        num_epochs=num_epochs,
        random_seed=0,
        system_name=system_name,
    )
    print(f"   Training complete! Final loss: {training_losses[-1]:.2e}")

    # 4. Generate trajectory
    print("Simulating trajectory...")
    initial_state = stochastic_trajectories[0, 0, :2]  # [position, velocity]
    neural_time, neural_trajectory = simulate_neural_ode_trajectory(
        drift_network,
        x0=initial_state,
        t0=time_grid[0].item(),
        tN=time_grid[-1].item(),
        dt=timestep,
    )

    # 5. Visualisation
    print("Visualisation Generation...")
    figures = [
        (
            "Training loss evolution",
            f"{system_name}_training_loss.png",
            lambda: plot_training_loss(
                training_losses,
                save_path=save_dir / f"{system_name}_training_loss.png",
                show_plot=show_plots,
            ),
        ),
        (
            "Trajectory comparison",
            f"{system_name}_trajectory_comparison.png",
            lambda: plot_trajectory_comparison(
                time_grid,
                stochastic_trajectories,
                neural_time,
                neural_trajectory,
                save_path=save_dir / f"{system_name}_trajectory_comparison.png",
                show_plot=show_plots,
            ),
        ),
        (
            "Phase space portrait",
            f"{system_name}_phase_space.png",
            lambda: plot_phase_space_comparison(
                stochastic_trajectories,
                neural_trajectory,
                save_path=save_dir / f"{system_name}_phase_space.png",
                show_plot=show_plots,
            ),
        ),
    ]

    for desc, filename, plot_func in tqdm(
        figures, desc="Creating figures", unit="figure"
    ):
        plot_func()

    print(f"   All figures saved to {save_dir}")

    # 6. Model saving
    model_dir = Path("examples/models")
    model_dir.mkdir(exist_ok=True)
    print(f"Saving trained {system_name} neural ODE model...")
    model_path = model_dir / f"{system_name}_neural_ode.pth"
    torch.save(
        {
            "model_state_dict": drift_network.state_dict(),
            "model_architecture": {
                "hidden_layer_sizes": hidden_layer_sizes,
                "input_size": TOTAL_INPUT_DIMENSION,
                "output_size": STATE_DIMENSION,
                "system_name": system_name,
                "has_external_forcing": HAS_EXTERNAL_FORCING,
            },
            "training_config": {
                "num_trajectories": num_trajectories,
                "total_time": total_time,
                "timestep": timestep,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "final_loss": training_losses[-1],
                "initial_loss": training_losses[0],
            },
            "training_losses": training_losses,
        },
        model_path,
    )

    history_path = model_dir / f"{system_name}_training_history.txt"
    with open(history_path, "w") as f:
        f.write(f"{system_name} Neural ODE Training History\n")
        f.write("=" * 40 + "\n")
        f.write(f"System: {system_name}\n")
        f.write(f"Model Architecture: {hidden_layer_sizes}\n")
        f.write(
            f"Training Dataset: {num_trajectories} trajectories, {total_time} time units\n"
        )
        f.write(f"Training Configuration: {num_epochs} epochs, lr={learning_rate}\n")
        f.write(f"External Forcing: {'Yes' if HAS_EXTERNAL_FORCING else 'No'}\n")
        f.write(f"Final Loss: {training_losses[-1]:.2e}\n")
        f.write(f"Initial Loss: {training_losses[0]:.2e}\n")
        f.write("Loss per Epoch:\n")
        for epoch, loss in enumerate(training_losses, 1):
            f.write(f"Epoch {epoch:3d}: {loss:.6e}\n")

    print(f"   Model and history saved to {model_dir}")

    # 7. Save parameters to JSON
    params_dir = Path("examples/data")
    params_dir.mkdir(exist_ok=True)
    params_path = params_dir / f"{system_name}_parameters.json"
    print(f"Saving parameters to {params_path}")

    physical_params = _get_default_physical_parameters()
    object.__setattr__(physical_params, "total_time", total_time)
    object.__setattr__(physical_params, "timestep", timestep)
    dimensionless_params = physical_params.to_dimensionless()

    from dataclasses import asdict
    import json

    all_parameters = {
        "system_name": system_name,
        "physical_parameters": asdict(physical_params),
        "dimensionless_parameters": asdict(dimensionless_params),
        "ml_parameters": {
            "model_architecture": {
                "hidden_layer_sizes": hidden_layer_sizes,
                "input_size": TOTAL_INPUT_DIMENSION,
                "output_size": STATE_DIMENSION,
                "has_external_forcing": HAS_EXTERNAL_FORCING,
            },
            "training_configuration": {
                "num_trajectories": num_trajectories,
                "total_time": total_time,
                "timestep": timestep,
                "num_epochs": num_epochs,
                "learning_rate": learning_rate,
                "trajectory_batch_size": trajectory_batch_size,
                "training_batch_size": training_batch_size,
            },
            "training_results": {
                "final_loss": training_losses[-1],
                "initial_loss": training_losses[0],
            },
        },
    }

    with open(params_path, "w") as f:
        json.dump(all_parameters, f, indent=4)

    print("   Parameters saved successfully.")

    return drift_network, training_losses, {"final_loss": training_losses[-1]}


if __name__ == "__main__":
    # TODO: Customise parameters for your system
    print(f"Running {SYSTEM_NAME} Neural ODE Template")
    print("=" * 50)
    print("This is a template—modify the system-specific sections!")
    print("=" * 50)

    # Run demonstration
    trained_model, losses, benchmarks = create_neural_ode_demonstration(
        num_trajectories=16,
        total_time=DEFAULT_TOTAL_TIME,
        timestep=DEFAULT_TIMESTEP,
        hidden_layer_sizes=DEFAULT_HIDDEN_LAYERS,
        num_epochs=100,
        learning_rate=2e-3,
        trajectory_batch_size=8,
        training_batch_size=16,
        show_plots=False,
    )

    print(f"{SYSTEM_NAME} demonstration complete.")
    print("TODO: Adapt this template for your own system by:")
    print("   1. Replacing system parameters and classes")
    print("   2. Modifying external forcing function")
    print("   3. Updating physical units and scales")
    print("   4. Adjusting visualisation labels")
    print("   5. Testing with your own data")
