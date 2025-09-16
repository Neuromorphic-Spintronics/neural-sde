import torch
from torch import nn, Tensor

from .base import DriftNet
from neural_dynamics.core.hyperparameters import Hyperparameters
from neural_dynamics.core.integrators import integrate_trajectory_with_step_method, runge_kutta_4_step
from config import DEVICE

class NeuralODE(nn.Module):
    def __init__(self, drift_net: DriftNet, hyperparameters: Hyperparameters):
        super().__init__()
        self.drift_net = drift_net
        self.hyperparameters = hyperparameters

    def forward(self, initial_state: Tensor, t_span: Tensor) -> Tensor:
        t, trajectory = integrate_trajectory_with_step_method(
            step_integrator=runge_kutta_4_step,
            drift_function=self.drift_function,
            diffusion_function=None,
            initial_state=initial_state.to(DEVICE),
            initial_time=t_span[0].item(),
            final_time=t_span[-1].item(),
            timestep=self.hyperparameters.timestep,
        )
        return trajectory

    def drift_function(self, t: float, state: Tensor) -> Tensor:
        time_tensor = torch.full((state.shape[0],), t, device=state.device)
        return self.drift_net.compute_drift(state, time_tensor)