
from __future__ import annotations

import torch
from torch import Tensor
from typing import Tuple, Any

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
