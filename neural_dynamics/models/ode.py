from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from neural_dynamics.core.hyperparameters import Hyperparameters
from neural_dynamics.core.integrators import (
    integrate_trajectory_with_step_method,
    runge_kutta_4_step,
)
from neural_dynamics.models.base import DriftNet
from neural_dynamics.training.base import train_with_validation


class NeuralODE(nn.Module):
    """Neural ODE model backed by a trained drift network."""

    def __init__(
        self,
        *,
        drift_net: DriftNet,
        hyperparameters: Hyperparameters,
        time_grid: Optional[Tensor] = None,
        training_losses: Optional[list[float]] = None,
        validation_losses: Optional[list[float]] = None,
    ) -> None:
        super().__init__()
        self.drift_net = drift_net
        self.hyperparameters = hyperparameters
        self.time_grid = time_grid
        self.training_losses = list(training_losses or [])
        self.validation_losses = list(validation_losses or [])
        self.device = next(self.parameters()).device

    def forward(self, initial_state: Tensor, t_span: Tensor) -> Tensor:
        model_device = self.device
        _, trajectory = integrate_trajectory_with_step_method(
            step_integrator=runge_kutta_4_step,
            drift_function=self.drift_function,
            diffusion_function=None,
            initial_state=initial_state.to(model_device),
            initial_time=t_span[0].item(),
            final_time=t_span[-1].item(),
            timestep=self.hyperparameters.timestep,
        )
        return trajectory

    def drift_function(self, t: float, state: Tensor) -> Tensor:
        time_tensor = torch.full((state.shape[0],), t, device=state.device)
        return self.drift_net.compute_drift(state, time_tensor)

    @classmethod
    def train(
        cls,
        *,
        hyperparameters: Hyperparameters,
        trajectories: Tensor,
        time_grid: Tensor,
        device: torch.device,
        validation_split: float = 0.2,
        early_stopping_patience: Optional[int] = None,
    ) -> "NeuralODE":
        """Train the drift network using batched trajectory supervision."""

        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [batch, time, state]")
        if time_grid.ndim != 1:
            raise ValueError("time_grid must be a 1D tensor")
        if trajectories.shape[1] != time_grid.shape[0]:
            raise ValueError("trajectory length must match time grid length")
        if not 0.0 < validation_split < 1.0:
            raise ValueError("validation_split must be in (0, 1)")

        state_dim = hyperparameters.state_dimension
        if trajectories.shape[2] < state_dim:
            raise ValueError(
                "trajectory state dimension is smaller than configured state_dimension"
            )

        drift_net = DriftNet(hyperparameters.drift_network).to(device)

        total_trajectories = trajectories.shape[0]
        val_size = max(1, int(total_trajectories * validation_split))
        train_size = total_trajectories - val_size
        if train_size <= 0:
            raise ValueError("validation_split leaves no data for training")

        train_trajs = trajectories[:train_size].to(device)
        val_trajs = trajectories[train_size:].to(device)

        train_loader = DataLoader(
            TensorDataset(train_trajs),
            batch_size=hyperparameters.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            TensorDataset(val_trajs),
            batch_size=hyperparameters.batch_size,
            shuffle=False,
            num_workers=0,
        )

        dt = float((time_grid[1] - time_grid[0]).item())
        early_stop = early_stopping_patience or hyperparameters.number_of_epochs

        time_grid_device = time_grid.to(device)

        def batch_preparation_fn(raw_batch, target_device):
            (batch_trajs,) = raw_batch
            current = batch_trajs[:, :-1, :state_dim]
            target = batch_trajs[:, 1:, :state_dim]

            times = time_grid_device[:-1].view(1, -1, 1).expand_as(current[..., :1])
            net_input = torch.cat([current, times], dim=-1).reshape(-1, state_dim + 1)
            true_derivatives = ((target - current) / dt).reshape(-1, state_dim)

            return (
                net_input.to(target_device, non_blocking=True),
                true_derivatives.to(target_device, non_blocking=True),
            )

        trained_drift, train_losses, val_losses = train_with_validation(
            model=drift_net,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=hyperparameters.number_of_epochs,
            learning_rate=hyperparameters.learning_rates.drift,
            early_stopping_patience=early_stop,
            device=device,
            batch_preparation_fn=batch_preparation_fn,
        )

        trained_drift.eval()

        return cls(
            drift_net=trained_drift,
            hyperparameters=hyperparameters,
            time_grid=time_grid.detach().cpu(),
            training_losses=train_losses,
            validation_losses=val_losses,
        )
