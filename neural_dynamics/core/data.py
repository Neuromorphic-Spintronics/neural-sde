
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, Any, Sequence, Optional, List, Dict

from examples.systems.registry import get_system_info
from neural_dynamics.config import DEVICE
from .integrators import auto_select_integrator

def _integrate_batch_sde_trajectories(
    system_name: str,
    dimensionless_parameters: Any,
    batch_size: int,
    time_grid: Tensor,
    dt: float,
    device: str | torch.device = DEVICE,
    random_seed: int | None = None,
) -> Tensor:
    """Integrate multiple SDE trajectories in parallel."""
    if random_seed is not None:
        torch.manual_seed(random_seed)

    system_info = get_system_info(system_name)
    SystemClass = system_info["system_class"]
    system = SystemClass(dimensionless_parameters)
    device_obj = torch.device(device)

    initial_states = system.get_initial_state().expand(batch_size, -1).to(device_obj)
    num_steps = len(time_grid) - 1
    trajectories = torch.zeros(
        batch_size, num_steps + 1, system.get_state_dimension(), device=device_obj
    )
    trajectories[:, 0] = initial_states
    current_states = initial_states.clone()

    step_integrator = auto_select_integrator(has_diffusion=system.has_noise)
    for step in range(num_steps):
        current_time = float(time_grid[step].item())
        if system.has_noise:
            next_states = step_integrator(
                system.drift_function,
                system.diffusion_function,
                current_states,
                current_time,
                dt,
            )
        else:
            next_states = step_integrator(
                system.drift_function, current_states, current_time, dt
            )
        trajectories[:, step + 1] = next_states
        current_states = next_states

    return trajectories


def generate_stochastic_dataset(
    *,
    system_name: str,
    num_trajectories: int,
    total_time: float,
    dt: float,
    seed: int = 42069,
    with_noise: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Optimised trajectory generation for a given system."""
    torch.manual_seed(seed)
    system_info = get_system_info(system_name)
    ParameterClass = system_info["parameter_class"]
    physical_parameters = ParameterClass(total_time=total_time, timestep=dt)

    if not with_noise:
        print("   Generating deterministic dataset (noise turned off).")
        if hasattr(physical_parameters, "temperature"):
            object.__setattr__(physical_parameters, "temperature", 0.0)
        if hasattr(physical_parameters, "coloured_noise_intensity"):
            object.__setattr__(physical_parameters, "coloured_noise_intensity", 0.0)

    dimensionless_parameters = physical_parameters.to_dimensionless()
    num_steps = int(total_time / dt)
    time_grid = torch.linspace(0, total_time, num_steps + 1, device=DEVICE)

    print(f"Generating {num_trajectories} trajectories in a single batch...")
    all_trajectories = _integrate_batch_sde_trajectories(
        system_name,
        dimensionless_parameters,
        num_trajectories,  # Integrate all trajectories at once
        time_grid,
        dt,
    )
    print("Trajectory generation complete.")

    return time_grid, all_trajectories


def apply_keep_fraction(
    trajectories: Tensor,
    time_grid: Tensor,
    fraction: float,
) -> Tuple[Tensor, Tensor, int, Sequence[int]]:
    """Slice each trajectory into fixed-length windows covering ``fraction`` of the run."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1].")
    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape [num_runs, total_steps, state_dim].")

    total_steps = trajectories.shape[1]
    keep_steps = max(2, int(round(total_steps * fraction)))
    keep_steps = min(total_steps, keep_steps)

    starts = list(range(0, total_steps - keep_steps + 1, keep_steps))
    tail_start = total_steps - keep_steps
    if tail_start >= 0 and (not starts or starts[-1] != tail_start):
        starts.append(tail_start)

    windows: List[Tensor] = []
    windows_per_run: List[int] = []
    for run_idx in range(trajectories.shape[0]):
        run = trajectories[run_idx]
        run_windows = []
        for start in starts:
            end = start + keep_steps
            if end > total_steps:
                continue
            run_windows.append(run[start:end].clone())
        if not run_windows:
            raise ValueError(
                f"Could not create windows for run {run_idx}; keep_steps={keep_steps}, total_steps={total_steps}."
            )
        windows.extend(run_windows)
        windows_per_run.append(len(run_windows))

    windowed = torch.stack(windows, dim=0)
    base_time = time_grid[:keep_steps].clone()
    if base_time.numel() > 0:
        base_time = base_time - base_time[0].item()

    return windowed, base_time, keep_steps, windows_per_run


def resample_to_target_timesteps(
    trajectories: Tensor,
    time_grid: Tensor,
    target_steps: int,
) -> Tuple[Tensor, Tensor]:
    """Resample trajectories and time grid to ``target_steps`` via linear interpolation."""

    if target_steps <= 0:
        raise ValueError("target_steps must be a positive integer.")

    current_steps = trajectories.shape[1]
    if current_steps == target_steps or current_steps <= 1:
        return trajectories, time_grid

    start_time = float(time_grid[0].item())
    end_time = float(time_grid[-1].item())
    resampled_time = torch.linspace(
        start_time,
        end_time,
        target_steps,
        dtype=time_grid.dtype,
        device=time_grid.device,
    )

    traj_channels_first = trajectories.permute(0, 2, 1)
    resampled_traj = F.interpolate(
        traj_channels_first,
        size=target_steps,
        mode="linear",
        align_corners=False,
    ).permute(0, 2, 1).contiguous()

    return resampled_traj, resampled_time


def replicate_trajectories(
    trajectories: Tensor,
    factor: int,
) -> Tensor:
    """Repeat trajectories along the batch axis ``factor`` times."""

    if factor < 1:
        raise ValueError("replication_factor must be a positive integer.")
    if factor == 1:
        return trajectories
    return trajectories.repeat((factor, 1, 1))


def normalise_dataset(
    trajectories: Tensor,
    h_signal: Tensor,
    *,
    enabled: bool,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Optional[Dict[str, float]]]:
    """Mean-centre and scale AMR/H channels alongside the reference ``h_signal``."""

    if not enabled:
        return trajectories, h_signal, None

    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape [batch, steps, features].")

    trajectories = trajectories.clone()
    h_signal = h_signal.clone()

    amr = trajectories[..., 0]
    h = trajectories[..., 1]

    amr_centered = amr - amr.mean(dim=1, keepdim=True)
    h_centered = h - h.mean(dim=1, keepdim=True)

    amr_std = amr_centered.std(unbiased=False).clamp_min(eps)
    h_std = h_centered.std(unbiased=False).clamp_min(eps)

    trajectories[..., 0] = amr_centered / amr_std
    trajectories[..., 1] = h_centered / h_std

    h_signal_centered = h_signal - h_signal.mean()
    h_signal = h_signal_centered / h_std

    stats = {
        "amr_mean": float(amr_centered.mean().item()),
        "amr_std": float(amr_std.item()),
        "h_mean": float(h_signal_centered.mean().item()),
        "h_std": float(h_std.item()),
    }
    return trajectories, h_signal, stats
