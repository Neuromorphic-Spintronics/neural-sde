from __future__ import annotations
import math
from typing import Any, Callable, List, Optional, Tuple, overload

import torch
from torch import Tensor, nn, optim
from tqdm import tqdm

from .base import DriftNet, FeedForwardNetwork, count_network_parameters
from neural_dynamics.core.hyperparameters import Hyperparameters, NetworkArchitecture
from neural_dynamics.models.ode import NeuralODE
from neural_dynamics.config import DEVICE

class DiffusionNet(FeedForwardNetwork):
    """Neural network approximating the diffusion term sigma(x, t, u)."""

    def __init__(
        self,
        architecture: NetworkArchitecture,
        *,
        state_dimension: int,
        noise_dimension: int,
        activation: Callable[[], nn.Module] = nn.Tanh,
        device: torch.device = DEVICE,
    ) -> None:
        """Initialise the diffusion network with dimensionality checks."""
        if state_dimension <= 0:
            raise ValueError("state_dimension must be a positive integer")
        if noise_dimension <= 0:
            raise ValueError("noise_dimension must be a positive integer")

        expected_output = state_dimension * noise_dimension
        if architecture.output_size != expected_output:
            raise ValueError(
                "Diffusion network output size must equal "
                "state_dimension * noise_dimension"
            )

        super().__init__(architecture, activation, device=device)
        self.state_dimension = state_dimension
        self.noise_dimension = noise_dimension

    def compute_diffusion(
        self, state: Tensor, time: Tensor, external_inputs: Optional[Tensor] = None
    ) -> Tensor:
        """Compute diffusion matrices for given state, time, and inputs."""
        network_input = self._prepare_network_input(
            state=state,
            time=time,
            external_inputs=external_inputs,
            expected_input_size=self.input_size,
        )
        batch_size = state.shape[0]
        diffusion_flat = super().forward(network_input)
        return diffusion_flat.view(batch_size, self.state_dimension, self.noise_dimension)

class CriticNet(FeedForwardNetwork):
    """Neural network that scores trajectories for the WGAN-GP critic."""

    def __init__(
        self,
        architecture: NetworkArchitecture,
        trajectory_length: int,
        *,
        activation: Callable[[], nn.Module] = nn.Tanh,
        device: torch.device = DEVICE,
    ) -> None:
        """Initialise the critic with trajectory-specific validation.

        Args:
            architecture: Network architecture describing the critic network.
            trajectory_length: Number of timesteps in each trajectory segment.
            activation: Factory returning the activation module used between layers.
            device: Device where the network parameters are stored.

        Raises:
            ValueError: If the provided configuration is inconsistent.
        """

        if trajectory_length <= 0:
            raise ValueError("trajectory_length must be a positive integer")

        if architecture.input_size % trajectory_length != 0:
            raise ValueError(
                "architecture.input_size must be divisible by trajectory_length"
            )

        self.architecture = architecture
        self.trajectory_length = trajectory_length
        self._state_dimension = architecture.input_size // trajectory_length

        super().__init__(architecture, activation, device=device)

    def score(self, trajectory_segment: Tensor) -> Tensor:
        """Score a batch of trajectories.

        Args:
            trajectory_segment: Tensor of shape ``[batch, time, state_dim]``.

        Returns:
            Critic scores with shape ``[batch, 1]``.

        Raises:
            ValueError: If the input tensor shape does not match configuration.
        """

        if trajectory_segment.ndim != 3:
            raise ValueError("trajectory_segment must be a 3D tensor")

        batch_size, trajectory_length, state_dim = trajectory_segment.shape

        if trajectory_length != self.trajectory_length:
            raise ValueError(
                "trajectory length does not match expected value "
                f"{self.trajectory_length}"
            )

        if state_dim != self._state_dimension:
            raise ValueError(
                "state dimension does not match expected value "
                f"{self._state_dimension}"
            )

        critic_input = trajectory_segment.reshape(batch_size, -1)
        return super().forward(critic_input)

class NeuralSDE(nn.Module):
    """Neural SDE model composed of drift, diffusion, and critic networks."""

    def __init__(
        self,
        drift_or_hyperparameters: DriftNet | Hyperparameters,
        diffusion_net: DiffusionNet | None = None,
        hyperparameters: Hyperparameters | None = None,
        *,
        critic_net: CriticNet | None = None,
    ) -> None:
        super().__init__()

        if isinstance(drift_or_hyperparameters, Hyperparameters):
            hyperparams = drift_or_hyperparameters
            self.hyperparameters = hyperparams
            self.state_dimension = hyperparams.state_dimension
            self.input_dimension = hyperparams.input_dimension
            self.timestep = hyperparams.timestep

            self.drift_net = DriftNet(hyperparams.drift_network)

            diffusion_arch = hyperparams.diffusion_network
            noise_dimension = self._infer_noise_dimension(
                diffusion_arch, self.state_dimension
            )
            self.noise_dimension = noise_dimension
            self.diffusion_net = (
                None
                if diffusion_arch is None
                else DiffusionNet(
                    diffusion_arch,
                    state_dimension=self.state_dimension,
                    noise_dimension=noise_dimension,
                )
            )

            critic_arch = hyperparams.critic_network
            if self.diffusion_net is not None and critic_arch is None:
                raise ValueError("Critic network is required for stochastic Neural SDEs")

            trajectory_length = self._infer_trajectory_length(
                critic_arch, self.state_dimension
            )
            self.critic_net = (
                None
                if critic_arch is None
                else CriticNet(critic_arch, trajectory_length)
            )
            self.trajectory_length = trajectory_length
        else:
            if diffusion_net is None:
                raise ValueError(
                    "diffusion_net must be provided when supplying explicit networks"
                )
            if hyperparameters is None:
                raise ValueError(
                    "hyperparameters must be provided when supplying explicit networks"
                )

            self.hyperparameters = hyperparameters
            self.state_dimension = hyperparameters.state_dimension
            self.input_dimension = hyperparameters.input_dimension
            self.timestep = hyperparameters.timestep

            self.drift_net = drift_or_hyperparameters
            self.diffusion_net = diffusion_net
            self.noise_dimension = diffusion_net.noise_dimension

            if critic_net is not None:
                self.critic_net = critic_net
                self.trajectory_length = critic_net.trajectory_length
            else:
                critic_arch = hyperparameters.critic_network
                trajectory_length = self._infer_trajectory_length(
                    critic_arch, self.state_dimension
                )
                self.critic_net = (
                    None
                    if critic_arch is None
                    else CriticNet(critic_arch, trajectory_length)
                )
                self.trajectory_length = trajectory_length

        self._sqrt_timestep = math.sqrt(max(self.timestep, 1e-12))
        self.generator_losses: list[float] = []
        self.critic_losses: list[float] = []

    @classmethod
    def train(
        cls,
        *,
        hyperparameters: Hyperparameters,
        neural_ode: NeuralODE,
        trajectories: Tensor,
        time_grid: Tensor,
        device: torch.device,
        enable_adversarial: bool = False,
        random_seed: int = 0,
    ) -> "NeuralSDE":
        """Train a neural SDE starting from a pretrained neural ODE."""

        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [batch, time, state]")
        if time_grid.ndim != 1:
            raise ValueError("time_grid must be a 1D tensor")
        if trajectories.shape[1] != time_grid.shape[0]:
            raise ValueError("trajectory length must match time grid length")

        model = cls(hyperparameters).to(device)
        model.drift_net.load_state_dict(neural_ode.drift_net.state_dict())
        model.drift_net.to(device)
        model.drift_net.eval()

        trajectories_device = trajectories.to(device)
        time_grid_device = time_grid.to(device)

        if enable_adversarial:
            if model.diffusion_net is None or model.critic_net is None:
                raise ValueError(
                    "Adversarial training requires diffusion and critic networks"
                )

            critic_window = model.critic_net.trajectory_length
            if critic_window is None:
                raise ValueError("Critic network must define a trajectory window length")
            if trajectories_device.shape[1] < critic_window:
                raise ValueError(
                    "Provided trajectories are shorter than the critic window length"
                )

            gan_trajectories = trajectories_device[:, :critic_window, : model.state_dimension]
            gan_time_grid = time_grid_device[:critic_window]

            generator_losses, critic_losses = fit_neural_sde_gan(
                drift_network=model.drift_net,
                diffusion_network=model.diffusion_net,
                critic_network=model.critic_net,
                time_grid=gan_time_grid,
                stochastic_trajectories=gan_trajectories,
                hyperparameters=hyperparameters,
                random_seed=random_seed,
            )
            model.generator_losses = generator_losses
            model.critic_losses = critic_losses
        else:
            if model.diffusion_net is not None:
                # In non-adversarial training, the diffusion network is not used.
                # We zero out its parameters to effectively disable it, ensuring that
                # it does not contribute to the model's output. This is the intended
                # behavior for non-adversarial SDE training.
                with torch.no_grad():
                    for param in model.diffusion_net.parameters():
                        param.zero_()

        return model

    @staticmethod
    def _infer_noise_dimension(
        architecture: NetworkArchitecture | None, state_dimension: int
    ) -> int:
        if architecture is None:
            return 0
        if architecture.output_size % state_dimension != 0:
            raise ValueError(
                "Diffusion network output size is not divisible by state dimension"
            )
        return architecture.output_size // state_dimension

    @staticmethod
    def _infer_trajectory_length(
        architecture: NetworkArchitecture | None, state_dimension: int
    ) -> int | None:
        if architecture is None:
            return None
        if architecture.input_size % state_dimension != 0:
            raise ValueError(
                "Critic network input size is not divisible by state dimension"
            )
        return architecture.input_size // state_dimension

    @overload
    def forward(self, initial_state: Tensor, time_grid: Tensor) -> Tensor:  # type: ignore[override]
        ...

    @overload
    def forward(
        self,
        external_inputs: Tensor,
        *,
        initial_state: Tensor,
        initial_time: float = 0.0,
    ) -> Tensor:
        ...

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:  # type: ignore[override]
        if len(args) == 2 and not kwargs:
            initial_state, time_grid = args
            return self._simulate_with_time_grid(initial_state, time_grid)

        if len(args) == 1 and "initial_state" in kwargs:
            external_inputs = args[0]
            initial_state = kwargs["initial_state"]
            initial_time = float(kwargs.get("initial_time", 0.0))
            return self._simulate_with_inputs(
                external_inputs, initial_state, initial_time
            )

        raise TypeError(
            "forward expects either (initial_state, time_grid) or "
            "(external_inputs, *, initial_state, initial_time)"
        )

    def _simulate_with_time_grid(self, initial_state: Tensor, time_grid: Tensor) -> Tensor:
        if time_grid.ndim != 1:
            raise ValueError("time_grid must be a 1D tensor")

        device = next(self.parameters()).device
        dtype = initial_state.dtype

        state = initial_state.to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0)

        batch_size = state.shape[0]
        trajectory = torch.empty(
            time_grid.shape[0], batch_size, self.state_dimension, device=device, dtype=dtype
        )
        trajectory[0] = state

        for idx in range(1, time_grid.shape[0]):
            previous_time = float(time_grid[idx - 1].item())
            dt = float(time_grid[idx].item() - time_grid[idx - 1].item())
            time_tensor = torch.full((batch_size, 1), previous_time, device=device, dtype=dtype)

            drift = self.drift_net.compute_drift(state, time_tensor, external_inputs=None)
            next_state = state + drift * dt

            if self.diffusion_net is not None and self.noise_dimension > 0:
                diffusion = self.diffusion_net.compute_diffusion(
                    state, time_tensor, external_inputs=None
                )
                noise = torch.randn(
                    batch_size, self.noise_dimension, device=device, dtype=dtype
                )
                diffusion_update = torch.bmm(diffusion, noise.unsqueeze(-1)).squeeze(-1)
                next_state = next_state + diffusion_update * math.sqrt(max(dt, 1e-12))

            trajectory[idx] = next_state
            state = next_state

        return trajectory

    def _simulate_with_inputs(
        self, external_inputs: Tensor, initial_state: Tensor, initial_time: float
    ) -> Tensor:
        if external_inputs.ndim != 3:
            raise ValueError("external_inputs must have shape [batch, inputs, time]")

        batch_size, input_dim, num_timesteps = external_inputs.shape
        if input_dim != self.input_dimension:
            raise ValueError(
                "External input dimension "
                f"{external_inputs.shape[1]} does not match configured input_dimension "
                f"{self.input_dimension}"
            )

        state = initial_state
        if state.ndim == 1:
            state = state.unsqueeze(0)
        if state.shape[0] != batch_size:
            raise ValueError(
                "Initial state batch size does not match external input batch size"
            )
        if state.shape[1] != self.state_dimension:
            raise ValueError(
                "Initial state dimension "
                f"{state.shape[1]} does not match configured state_dimension "
                f"{self.state_dimension}"
            )

        device = next(self.parameters()).device
        dtype = state.dtype

        state = state.to(device=device, dtype=dtype)
        inputs = external_inputs.to(device=device, dtype=dtype)

        trajectories = torch.empty(
            batch_size,
            self.state_dimension,
            num_timesteps,
            device=device,
            dtype=dtype,
        )

        current_state = state
        current_time = initial_time
        for step in range(num_timesteps):
            time_tensor = torch.full(
                (batch_size, 1), current_time, device=device, dtype=dtype
            )
            step_inputs = inputs[:, :, step]

            drift = self.drift_net.compute_drift(current_state, time_tensor, step_inputs)
            next_state = current_state + drift * self.timestep

            if self.diffusion_net is not None and self.noise_dimension > 0:
                diffusion = self.diffusion_net.compute_diffusion(
                    current_state, time_tensor, step_inputs
                )
                noise = torch.randn(
                    batch_size, self.noise_dimension, device=device, dtype=dtype
                )
                diffusion_update = torch.bmm(diffusion, noise.unsqueeze(-1)).squeeze(-1)
                next_state = next_state + diffusion_update * self._sqrt_timestep

            trajectories[:, :, step] = next_state
            current_state = next_state
            current_time += self.timestep

        return trajectories

    def count_total_parameters(self) -> dict[str, int]:
        """Return the number of trainable parameters per network component."""
        drift_params = count_network_parameters(self.drift_net)
        diffusion_params = (
            count_network_parameters(self.diffusion_net)
            if self.diffusion_net is not None
            else 0
        )
        critic_params = (
            count_network_parameters(self.critic_net)
            if self.critic_net is not None
            else 0
        )
        total = drift_params + diffusion_params + critic_params
        return {
            "drift_net": drift_params,
            "diffusion_net": diffusion_params,
            "critic_net": critic_params,
            "total": total,
        }

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
