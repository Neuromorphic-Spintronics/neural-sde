from dataclasses import dataclass
import math
from typing import List, Tuple, Callable

import torch
from torch import Tensor
from tqdm import tqdm

from config import DEVICE
from examples.systems.registry import get_system_info
from neural_dynamics.core.integrators import (
    integrate_trajectory_with_step_method,
    runge_kutta_4_step,
)
from neural_dynamics.models.base import DriftNet


@dataclass(frozen=True)
class NeuralODETrainingConfig:
    """Configuration for neural ODE training."""

    learning_rate: float = 1e-2
    batch_size: int = 8
    num_epochs: int = 50


def simulate_neural_ode_trajectory(
    drift_net: DriftNet,
    drift_function: Callable[[float, Tensor], Tensor],
    *,
    x0: Tensor,
    t0: float,
    tN: float,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """Simulate trajectory using the trained neural ODE."""
    return integrate_trajectory_with_step_method(
        step_integrator=runge_kutta_4_step,
        drift_function=drift_function,
        diffusion_function=None,
        initial_state=x0.to(DEVICE),
        initial_time=t0,
        final_time=tN,
        timestep=dt,
    )


def fit_neural_ode(
    drift_net: DriftNet,
    vectorised_drift: Callable[[Tensor, Tensor], Tensor],
    time_grid: Tensor,
    stochastic_trajectories: Tensor,
    *,
    system_name: str,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    random_seed: int = 0,
) -> List[float]:
    """Trains the drift network using an efficient one-step-ahead prediction method."""
    torch.manual_seed(random_seed)
    system_info = get_system_info(system_name)
    state_dimension = system_info["state_dimension"]

    optimiser = torch.optim.Adam(drift_net.parameters(), lr=learning_rate)
    criterion = torch.nn.MSELoss()
    drift_net.train()

    core_state_trajectories = stochastic_trajectories[..., :state_dimension].to(DEVICE)
    time_step_size = (time_grid[1] - time_grid[0]).item()
    integration_times = time_grid[:-1]

    training_losses: List[float] = []

    with tqdm(
        range(num_epochs), desc=f"Training {system_name} Neural ODE", unit="epoch"
    ) as pbar:
        for epoch in pbar:
            trajectory_indices = torch.randperm(
                core_state_trajectories.shape[0], device=DEVICE
            )
            epoch_losses = []
            num_batches = math.ceil(core_state_trajectories.shape[0] / batch_size)

            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(
                    start_idx + batch_size, core_state_trajectories.shape[0]
                )
                actual_batch_size = end_idx - start_idx

                batch_indices = trajectory_indices[start_idx:end_idx]
                batch_trajectories = core_state_trajectories[batch_indices]

                optimiser.zero_grad()

                current_states = batch_trajectories[:, :-1]
                target_states = batch_trajectories[:, 1:]

                times = integration_times.unsqueeze(0).expand(actual_batch_size, -1)

                # Reshape for batch processing
                current_states_flat = current_states.reshape(-1, state_dimension)
                target_states_flat = target_states.reshape(-1, state_dimension)
                times_flat = times.reshape(-1, 1)

                # Predict the derivative from the current state and time
                predicted_derivatives = vectorised_drift(
                    times_flat, current_states_flat
                )

                # Use Euler step to get the next state prediction
                predicted_next_states_flat = (
                    current_states_flat + time_step_size * predicted_derivatives
                )

                loss = criterion(predicted_next_states_flat, target_states_flat)
                loss.backward()
                optimiser.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            training_losses.append(avg_loss)
            pbar.set_postfix({"Loss": f"{avg_loss:.2e}"})

    return training_losses
