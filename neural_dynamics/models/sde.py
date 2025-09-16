from __future__ import annotations
import torch
from torch import nn, optim, Tensor
from typing import Callable, Optional, Tuple, List
from tqdm import tqdm
import math

from .base import FeedForwardNetwork, DriftNet
from neural_dynamics.core.hyperparameters import NetworkArchitecture, Hyperparameters
from neural_dynamics.core.integrators import integrate_trajectory_with_step_method, stochastic_heun_step
from config import DEVICE

class DiffusionNet(FeedForwardNetwork):
    """
    Neural network approximating the diffusion term sigma(x, t, u) in the neural SDE.
    """
    def __init__(
        self,
        architecture: NetworkArchitecture,
        activation: Callable[[], nn.Module] = nn.Tanh,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(architecture, activation, device=device)

    @torch.jit.script_method
    def compute_diffusion(
        self, state: Tensor, time: Tensor, external_inputs: Optional[Tensor] = None
    ) -> Tensor:
        network_input = self._prepare_network_input(
            state=state,
            time=time,
            external_inputs=external_inputs,
            expected_input_size=self.input_size,
        )
        return self.forward(network_input)

class CriticNet(FeedForwardNetwork):
    """
    Neural network that scores trajectories for the WGAN-GP critic.
    """
    def __init__(
        self,
        architecture: NetworkArchitecture,
        activation: Callable[[], nn.Module] = nn.Tanh,
        device: torch.device = DEVICE,
    ) -> None:
        super().__init__(architecture, activation, device=device)

class NeuralSDE(nn.Module):
    def __init__(self, drift_net: DriftNet, diffusion_net: DiffusionNet, hyperparameters: Hyperparameters):
        super().__init__()
        self.drift_net = drift_net
        self.diffusion_net = diffusion_net
        self.hyperparameters = hyperparameters

    def forward(self, initial_state: Tensor, t_span: Tensor) -> Tensor:
        initial_time = t_span[0].item()
        final_time = t_span[-1].item()
        timestep = self.hyperparameters.timestep

        _, trajectory = integrate_trajectory_with_step_method(
            step_integrator=stochastic_heun_step,
            drift_function=self.drift_function,
            diffusion_function=self.diffusion_function,
            initial_state=initial_state.to(DEVICE),
            initial_time=initial_time,
            final_time=final_time,
            timestep=timestep,
        )
        return trajectory

    def drift_function(self, t: float, state: Tensor) -> Tensor:
        time_tensor = torch.full((state.shape[0],), t, device=state.device)
        return self.drift_net.compute_drift(state, time_tensor)

    def diffusion_function(self, t: float, state: Tensor) -> Tensor:
        time_tensor = torch.full((state.shape[0],), t, device=state.device)
        return self.diffusion_net.compute_diffusion(state, time_tensor)

# ---------------------------------------------------------------------------- #
#                           WGAN-GP Training Logic                             #
# ---------------------------------------------------------------------------- #

def _compute_gradient_penalty(
    critic_network: CriticNet, real_trajectories: Tensor, fake_trajectories: Tensor
) -> Tensor:
    """
    Calculates the gradient penalty for the WGAN-GP critic.
    This enforces the 1-Lipschitz constraint on the critic, which is crucial
    for stable training. The penalty is computed on interpolated trajectories
    between real and fake data.
    """
    batch_size, _, _ = real_trajectories.shape
    alpha = torch.rand(batch_size, 1, 1, device=DEVICE)
    alpha = alpha.expand_as(real_trajectories)

    interpolated_trajectories = (alpha * real_trajectories) + ((1 - alpha) * fake_trajectories)
    interpolated_trajectories = interpolated_trajectories.requires_grad_(True)

    scores = critic_network(interpolated_trajectories.reshape(batch_size, -1))

    gradients = torch.autograd.grad(
        outputs=scores,
        inputs=interpolated_trajectories,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()
    return gradient_penalty


def compute_critic_cost(
    critic_network: CriticNet,
    real_trajectories: Tensor,
    fake_trajectories: Tensor,
    gradient_penalty_weight: float,
) -> Tensor:
    """
    Calculate the critic's loss, which includes the Wasserstein distance
    and a gradient penalty.
    """
    real_scores = critic_network(real_trajectories.reshape(real_trajectories.shape[0], -1))
    fake_scores = critic_network(fake_trajectories.reshape(fake_trajectories.shape[0], -1))

    wasserstein_distance = fake_scores.mean() - real_scores.mean()
    gradient_penalty = _compute_gradient_penalty(
        critic_network, real_trajectories, fake_trajectories
    )

    return wasserstein_distance + gradient_penalty_weight * gradient_penalty


def compute_generator_cost(
    critic_network: CriticNet,
    fake_trajectories: Tensor,
) -> Tensor:
    """
    Calculate the generator's loss, which aims to maximise the critic's score
    for fake trajectories.
    """
    fake_scores = critic_network(fake_trajectories.reshape(fake_trajectories.shape[0], -1))
    return -fake_scores.mean()


def train_critic_step(
    critic_optimiser: optim.Optimizer,
    critic_network: CriticNet,
    real_trajectories: Tensor,
    fake_trajectories: Tensor,
    gradient_penalty_weight: float,
) -> float:
    """
    Perform a single training step for the critic network.
    """
    critic_optimiser.zero_grad()
    critic_cost = compute_critic_cost(
        critic_network, real_trajectories, fake_trajectories.detach(), gradient_penalty_weight
    )
    critic_cost.backward()
    critic_optimiser.step()
    return critic_cost.item()


def train_generator_step(
    generator_optimiser: optim.Optimizer,
    critic_network: CriticNet,
    fake_trajectories: Tensor,
) -> float:
    """
    Perform a single training step for the generator networks (drift and diffusion).
    """
    generator_optimiser.zero_grad()
    generator_cost = compute_generator_cost(critic_network, fake_trajectories)
    generator_cost.backward()
    generator_optimiser.step()
    return generator_cost.item()


def fit_neural_sde_gan(
    drift_network: DriftNet,
    diffusion_network: DiffusionNet,
    critic_network: CriticNet,
    time_grid: Tensor,
    stochastic_trajectories: Tensor,
    *,
    hyperparameters: Hyperparameters,
    random_seed: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    Train the Neural SDE GAN using the WGAN-GP algorithm.
    """
    torch.manual_seed(random_seed)

    # Freeze the drift network by setting it to evaluation mode and excluding its parameters from the optimizer
    drift_network.eval()
    generator_optimiser = optim.Adam(diffusion_network.parameters(), lr=hyperparameters.learning_rates.generator)
    critic_optimiser = optim.Adam(critic_network.parameters(), lr=hyperparameters.learning_rates.critic)

    generator_losses = []
    critic_losses = []

    num_batches = math.ceil(stochastic_trajectories.shape[0] / hyperparameters.batch_size)

    with tqdm(range(hyperparameters.number_of_epochs), desc="Training Neural SDE GAN") as pbar:
        for epoch in pbar:
            epoch_critic_losses = []
            epoch_generator_losses = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * hyperparameters.batch_size
                end_idx = min(start_idx + hyperparameters.batch_size, stochastic_trajectories.shape[0])
                real_trajectories = stochastic_trajectories[start_idx:end_idx].to(DEVICE)
                
                if real_trajectories.shape[0] == 0:
                    continue

                initial_states = real_trajectories[:, 0, :]
                sde = NeuralSDE(drift_network, diffusion_network, hyperparameters)

                with torch.no_grad():
                    fake_trajectories_for_critic = sde(initial_states, time_grid).permute(1, 0, 2)

                # Train critic
                for _ in range(hyperparameters.critic_updates):
                    critic_loss = train_critic_step(
                        critic_optimiser,
                        critic_network,
                        real_trajectories,
                        fake_trajectories_for_critic,
                        hyperparameters.gradient_penalty_weight,
                    )
                    epoch_critic_losses.append(critic_loss)

                # Train generator
                fake_trajectories_for_generator = sde(initial_states, time_grid).permute(1, 0, 2)

                generator_loss = train_generator_step(
                    generator_optimiser, critic_network, fake_trajectories_for_generator
                )
                epoch_generator_losses.append(generator_loss)

            generator_losses.append(sum(epoch_generator_losses) / len(epoch_generator_losses))
            critic_losses.append(sum(epoch_critic_losses) / len(epoch_critic_losses))
            pbar.set_postfix({"Gen Loss": generator_losses[-1], "Critic Loss": critic_losses[-1]})

    return generator_losses, critic_losses
