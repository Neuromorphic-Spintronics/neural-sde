from __future__ import annotations
import dataclasses
import math
from typing import Any, Callable, List, Optional, Tuple, overload, TYPE_CHECKING

import torch
from torch import Tensor, nn, optim
from tqdm import tqdm

from .base import DriftNet, FeedForwardNetwork, count_network_parameters
from neural_dynamics.core.hyperparameters import Hyperparameters, NetworkArchitecture
from neural_dynamics.config import DEVICE

if TYPE_CHECKING:
    from neural_dynamics.models.ode import NeuralODE

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
        self.state_dimension = state_dimension  # type: ignore
        self.noise_dimension = noise_dimension  # type: ignore

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

    trajectory_length: int
    _state_dimension: int

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

        _state_dimension = architecture.input_size // trajectory_length
        new_input_size = trajectory_length * _state_dimension
        architecture = dataclasses.replace(architecture, input_size=new_input_size)

        self.trajectory_length = trajectory_length  # type: ignore
        self._state_dimension = _state_dimension  # type: ignore

        super().__init__(architecture, activation, device=device)

    def forward(self, trajectory_segment: Tensor) -> Tensor:
        if trajectory_segment.ndim == 2:
            # Assume it's already flattened [batch_size, trajectory_length * state_dim]
            critic_input = trajectory_segment
        elif trajectory_segment.ndim == 3:
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
        else:
            raise ValueError("trajectory_segment must be 2D or 3D tensor")

        return super().forward(critic_input)

class NeuralSDE(nn.Module):
    """Neural SDE model composed of drift, diffusion, and critic networks."""

    hyperparameters: Hyperparameters
    state_dimension: int
    input_dimension: int
    timestep: float
    drift_net: DriftNet
    diffusion_net: Optional[DiffusionNet]
    noise_dimension: int
    critic_net: Optional[CriticNet]
    trajectory_length: Optional[int]
    _sqrt_timestep: float
    generator_losses: list[float]
    critic_losses: list[float]
    device: torch.device

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
            self.hyperparameters = hyperparams  # type: ignore
            self.state_dimension = hyperparams.state_dimension  # type: ignore
            self.input_dimension = hyperparams.input_dimension  # type: ignore
            self.timestep = hyperparams.timestep  # type: ignore

            self.drift_net = DriftNet(hyperparams.drift_network)

            diffusion_arch = hyperparams.diffusion_network
            noise_dimension = self._infer_noise_dimension(
                diffusion_arch, self.state_dimension
            )
            self.noise_dimension = noise_dimension  # type: ignore
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
                else CriticNet(critic_arch, trajectory_length)  # type: ignore
            )
            self.trajectory_length = trajectory_length  # type: ignore
        else:
            if diffusion_net is None:
                raise ValueError(
                    "diffusion_net must be provided when supplying explicit networks"
                )
            if hyperparameters is None:
                raise ValueError(
                    "hyperparameters must be provided when supplying explicit networks"
                )

            self.hyperparameters = hyperparameters  # type: ignore
            self.state_dimension = hyperparameters.state_dimension  # type: ignore
            self.input_dimension = hyperparameters.input_dimension  # type: ignore
            self.timestep = hyperparameters.timestep  # type: ignore

            self.drift_net = drift_or_hyperparameters
            self.diffusion_net = diffusion_net
            self.noise_dimension = diffusion_net.noise_dimension  # type: ignore

            if critic_net is not None:
                self.critic_net = critic_net
                self.trajectory_length = critic_net.trajectory_length  # type: ignore
            else:
                critic_arch = hyperparameters.critic_network
                trajectory_length = self._infer_trajectory_length(
                    critic_arch, self.state_dimension
                )
                self.critic_net = (
                    None
                    if critic_arch is None
                    else CriticNet(critic_arch, trajectory_length)  # type: ignore
                )
                self.trajectory_length = trajectory_length  # type: ignore

        self._sqrt_timestep = math.sqrt(max(self.timestep, 1e-12))  # type: ignore
        self.generator_losses: list[float] = []
        self.critic_losses: list[float] = []
        self.drift_losses: list[float] = []  # Drift network SmoothL1 loss during joint training
        self.device = next(self.parameters()).device

    @classmethod
    def train(
        cls,
        *,
        hyperparameters: Hyperparameters,
        neural_ode: "NeuralODE",
        trajectories: Tensor,
        time_grid: Tensor,
        device: torch.device,
        enable_adversarial: bool = False,
        random_seed: int = 0,
        wandb_run: Optional[Any] = None,
    ) -> "NeuralSDE":
        """Train a neural SDE starting from a pretrained neural ODE.

        Args:
            hyperparameters: Training hyperparameters
            neural_ode: Pretrained neural ODE to initialize drift network
            trajectories: Training trajectories [batch, time, state]
            time_grid: Time grid for trajectories
            device: Device for training
            enable_adversarial: Whether to enable adversarial training
            random_seed: Random seed for reproducibility
            wandb_run: Optional W&B run object for logging training progress

        Returns:
            Trained NeuralSDE instance
        """

        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [batch, time, state]")
        if time_grid.ndim != 1:
            raise ValueError("time_grid must be a 1D tensor")
        if trajectories.shape[1] != time_grid.shape[0]:
            raise ValueError("trajectory length must match time grid length")

        model = cls(hyperparameters).to(device)
        model.device = device
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

            # Pass full trajectories - random windows will be sampled internally
            gan_trajectories = trajectories_device[:, :, : model.state_dimension]
            gan_time_grid = time_grid_device

            # Get actual number of ODE epochs trained (accounting for early stopping)
            actual_ode_epochs = len(neural_ode.training_losses) if hasattr(neural_ode, 'training_losses') else None

            generator_losses, critic_losses, drift_losses = fit_neural_sde_gan(
                drift_network=model.drift_net,
                diffusion_network=model.diffusion_net,
                critic_network=model.critic_net,
                time_grid=gan_time_grid,
                stochastic_trajectories=gan_trajectories,
                hyperparameters=hyperparameters,
                random_seed=random_seed,
                wandb_run=wandb_run,
                actual_ode_epochs=actual_ode_epochs,
            )
            model.generator_losses = generator_losses
            model.critic_losses = critic_losses
            model.drift_losses = drift_losses  # Store drift losses for plotting
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

        device = self.device
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

        return trajectory.permute(1, 0, 2)  # [batch, time, state]

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

        device = self.device
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

    def sample_trajectory(
        self,
        initial_state: Tensor,
        time_grid: Tensor,
        drift_function: Callable[[float, Tensor], Tensor],
        diffusion_function: Callable[[float, Tensor], Tensor],
        device: torch.device,
    ) -> Tensor:
        """Sample a single stochastic trajectory using custom drift/diffusion functions.
        
        This method allows evaluating the SDE with time-varying external inputs by
        providing custom drift and diffusion functions that capture the full dynamics.
        
        Args:
            initial_state: Initial state [batch_size, state_dim] or [state_dim]
            time_grid: Time points for integration [num_steps]
            drift_function: Custom drift function f(t, y) -> drift
            diffusion_function: Custom diffusion function g(t, y) -> diffusion matrix
            device: Device for computation
            
        Returns:
            Trajectory tensor [batch_size, num_steps, state_dim]
        """
        dtype = initial_state.dtype
        state = initial_state.to(device=device, dtype=dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        
        batch_size = state.shape[0]
        num_steps = time_grid.shape[0]
        
        trajectory = torch.empty(
            num_steps, batch_size, self.state_dimension, device=device, dtype=dtype
        )
        trajectory[0] = state
        
        for idx in range(1, num_steps):
            current_time = float(time_grid[idx - 1].item())
            dt = float(time_grid[idx].item() - time_grid[idx - 1].item())
            
            # Compute drift and diffusion with custom functions
            drift = drift_function(current_time, state)
            next_state = state + drift * dt
            
            if self.diffusion_net is not None and self.noise_dimension > 0:
                diffusion = diffusion_function(current_time, state)
                noise = torch.randn(
                    batch_size, self.noise_dimension, device=device, dtype=dtype
                )
                diffusion_update = torch.bmm(diffusion, noise.unsqueeze(-1)).squeeze(-1)
                next_state = next_state + diffusion_update * math.sqrt(max(dt, 1e-12))
            
            trajectory[idx] = next_state
            state = next_state
        
        return trajectory.permute(1, 0, 2)  # [batch, time, state]

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
    real_trajectories: Tensor | None = None,
    moment_matching_weight: float = 0.0,
    moment_matching_enabled: bool = False,
) -> Tensor:
    """
    Calculate the generator's loss, which aims to maximise the critic's score
    for fake trajectories and optionally match statistical moments.
    
    Args:
        critic_network: The critic network
        fake_trajectories: Generated trajectories [batch, time, state]
        real_trajectories: Real trajectories for moment matching [batch, time, state]
        moment_matching_weight: Weight for moment matching loss
        moment_matching_enabled: Whether to enable moment matching
    """
    fake_scores = critic_network(fake_trajectories.reshape(fake_trajectories.shape[0], -1))
    adversarial_loss = -fake_scores.mean()
    
    if moment_matching_enabled and real_trajectories is not None:
        # Compute mean and variance along batch dimension for each timestep
        fake_mean = torch.mean(fake_trajectories, dim=0)  # [time, state]
        real_mean = torch.mean(real_trajectories, dim=0)  # [time, state]
        
        fake_var = torch.var(fake_trajectories, dim=0, unbiased=False)  # [time, state]
        real_var = torch.var(real_trajectories, dim=0, unbiased=False)  # [time, state]
        
        # Mean matching loss
        mean_loss = torch.mean((fake_mean - real_mean).pow(2))
        
        # Variance matching loss
        var_loss = torch.mean((fake_var - real_var).pow(2))
        
        # Combined moment matching loss
        moment_loss = mean_loss + var_loss
        
        return adversarial_loss + moment_matching_weight * moment_loss
    
    return adversarial_loss


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
    # Gradient clipping for stability
    torch.nn.utils.clip_grad_norm_(critic_network.parameters(), max_norm=1.0)
    critic_optimiser.step()
    return critic_cost.item()


def train_generator_step(
    generator_optimiser: optim.Optimizer,
    critic_network: CriticNet,
    fake_trajectories: Tensor,
    real_trajectories: Tensor,
    sde_l1_weight: float,
    moment_matching_weight: float = 0.0,
    moment_matching_enabled: bool = False,
    diffusion_network: Optional[DiffusionNet] = None,
    drift_only_loss: Optional[Tensor] = None,
    drift_l1_weight: float = 0.0,
) -> Tuple[float, dict]:
    """
    Perform a single training step for the generator networks (drift and diffusion).

    Args:
        generator_optimiser: Optimizer for generator networks
        critic_network: The critic network
        fake_trajectories: Generated trajectories (drift + diffusion)
        real_trajectories: Ground truth trajectories
        sde_l1_weight: Weight for L1 pathwise loss on full SDE trajectories
        moment_matching_weight: Weight for statistical moment matching
        moment_matching_enabled: Whether to enable moment matching
        diffusion_network: Optional diffusion network for gradient clipping
        drift_only_loss: Optional SmoothL1 loss for drift-only predictions (requires gradients)
        drift_l1_weight: Weight for drift-only SmoothL1 constraint during adversarial training

    Returns:
        Tuple of (total_loss, loss_components_dict)
    """
    generator_optimiser.zero_grad()

    # Adversarial loss with optional moment matching
    adversarial_loss = compute_generator_cost(
        critic_network, fake_trajectories, real_trajectories, moment_matching_weight, moment_matching_enabled
    )

    # Pathwise L1 loss on full SDE trajectories (drift + diffusion)
    l1_loss = nn.functional.smooth_l1_loss(
        fake_trajectories.contiguous(), real_trajectories.contiguous()
    )

    # Combined loss with optional drift-only constraint
    generator_cost = adversarial_loss + sde_l1_weight * l1_loss

    # Track individual loss components for analysis
    loss_components = {
        'adversarial': adversarial_loss.item(),
        'sde_l1': l1_loss.item(),
        'sde_l1_weighted': (sde_l1_weight * l1_loss).item(),
        'drift_only': 0.0,
        'drift_only_weighted': 0.0,
    }

    # Add explicit drift-only SmoothL1 constraint if provided
    # This ensures the deterministic component (drift) continues to improve
    if drift_only_loss is not None and drift_l1_weight > 0:
        generator_cost = generator_cost + drift_l1_weight * drift_only_loss
        loss_components['drift_only'] = drift_only_loss.item()
        loss_components['drift_only_weighted'] = (drift_l1_weight * drift_only_loss).item()

    generator_cost.backward()

    # Gradient clipping for stability (especially important for diffusion network)
    if diffusion_network is not None:
        torch.nn.utils.clip_grad_norm_(diffusion_network.parameters(), max_norm=0.5)

    generator_optimiser.step()
    return generator_cost.item(), loss_components


def fit_neural_sde_gan(
    drift_network: DriftNet,
    diffusion_network: DiffusionNet,
    critic_network: CriticNet,
    time_grid: Tensor,
    stochastic_trajectories: Tensor,
    *,
    hyperparameters: Hyperparameters,
    random_seed: int = 0,
    wandb_run: Optional[Any] = None,
    actual_ode_epochs: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """
    Train the Neural SDE GAN using the WGAN-GP algorithm.

    Samples random windows of size critic_network.trajectory_length from the full
    trajectories to ensure the GAN learns diffusion dynamics across different
    time regions and varying exogenous inputs (e.g., H field). This enables
    backpropagation through time (BPTT) over the entire trajectory domain.

    Args:
        drift_network: The drift network
        diffusion_network: The diffusion network
        critic_network: The critic network
        time_grid: Time grid for trajectories
        stochastic_trajectories: Training trajectories [batch, time, state]
        hyperparameters: Training hyperparameters
        random_seed: Random seed for reproducibility
        wandb_run: Optional W&B run object for logging training progress
        actual_ode_epochs: Actual number of ODE epochs trained (for W&B step offset)

    Returns:
        Tuple of (generator_losses, critic_losses, drift_losses).
        drift_losses will be populated only when train_ode_with_sde=True.
    """
    torch.manual_seed(random_seed)

    # Configure drift network training mode based on hyperparameter
    if hyperparameters.train_ode_with_sde:
        drift_network.train()
        generator_params = list(drift_network.parameters()) + list(diffusion_network.parameters())
    else:
        # Freeze the drift network by setting it to evaluation mode
        drift_network.eval()
        generator_params = diffusion_network.parameters()
    
    generator_optimiser = optim.Adam(
        generator_params, lr=hyperparameters.learning_rates.generator
    )
    critic_optimiser = optim.Adam(
        critic_network.parameters(), lr=hyperparameters.learning_rates.critic
    )

    generator_losses = []
    critic_losses = []
    drift_losses = []  # Track drift network SmoothL1 loss when training jointly

    stochastic_trajectories = stochastic_trajectories.to(DEVICE)
    time_grid = time_grid.to(DEVICE)
    
    # Get critic window size and trajectory length
    critic_window = critic_network.trajectory_length
    full_trajectory_length = stochastic_trajectories.shape[1]
    
    if full_trajectory_length < critic_window:
        raise ValueError(
            f"Trajectory length {full_trajectory_length} is shorter than "
            f"critic window {critic_window}"
        )
    
    num_batches = math.ceil(stochastic_trajectories.shape[0] / hyperparameters.batch_size)

    # Create SDE model outside the loop for efficiency
    sde = NeuralSDE(drift_network, diffusion_network, hyperparameters).to(DEVICE)
    num_epochs = hyperparameters.number_of_gan_epochs

    with tqdm(range(num_epochs), desc="Training Neural SDE GAN") as pbar:
        for epoch in pbar:
            epoch_critic_losses = []
            epoch_generator_losses = []
            epoch_drift_losses = []  # Track drift-only loss when joint training

            # Accumulate loss components for detailed analysis
            epoch_loss_components = {
                'adversarial': [],
                'sde_l1': [],
                'sde_l1_weighted': [],
                'drift_only': [],
                'drift_only_weighted': [],
            }
            
            # Shuffle data each epoch for better generalization
            perm = torch.randperm(stochastic_trajectories.shape[0], device=DEVICE)
            shuffled_trajectories = stochastic_trajectories[perm]
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * hyperparameters.batch_size
                end_idx = min(start_idx + hyperparameters.batch_size, shuffled_trajectories.shape[0])
                
                if start_idx >= end_idx:
                    continue

                # Sample RANDOM windows from full trajectories to ensure critic sees entire time domain
                # and varying exogenous inputs (e.g., different H field values across time)
                batch_trajectories = shuffled_trajectories[start_idx:end_idx]  # [batch, full_time, state]
                batch_size_actual = batch_trajectories.shape[0]
                
                # For each trajectory in batch, sample a random starting point for the window
                max_start_idx = full_trajectory_length - critic_window
                if max_start_idx > 0:
                    # Random start indices for each trajectory in the batch
                    start_indices = torch.randint(
                        0, max_start_idx + 1, (batch_size_actual,), device=DEVICE
                    )
                else:
                    # If trajectory is exactly critic_window length, start at 0
                    start_indices = torch.zeros(batch_size_actual, dtype=torch.long, device=DEVICE)
                
                # Extract random windows for each trajectory
                real_trajectories = torch.stack([
                    batch_trajectories[i, start_indices[i]:start_indices[i] + critic_window, :]
                    for i in range(batch_size_actual)
                ]).contiguous()
                
                # Extract corresponding time windows
                window_time_grids = torch.stack([
                    time_grid[start_indices[i]:start_indices[i] + critic_window]
                    for i in range(batch_size_actual)
                ])
                
                # Use the time grid from the first trajectory (all should be identical structure)
                window_time_grid = window_time_grids[0]
                initial_states = real_trajectories[:, 0, :]

                # Generate fake trajectories once and reuse for critic updates (efficiency)
                with torch.no_grad():
                    fake_trajectories_detached = sde(initial_states, window_time_grid).detach()

                # Train critic multiple times with the same fake batch (WGAN-GP best practice)
                for _ in range(hyperparameters.critic_updates):
                    critic_loss = train_critic_step(
                        critic_optimiser,
                        critic_network,
                        real_trajectories,
                        fake_trajectories_detached,
                        hyperparameters.gradient_penalty_weight,
                    )
                    epoch_critic_losses.append(critic_loss)

                # Train generator once with fresh gradient-enabled trajectories
                fake_trajectories = sde(initial_states, window_time_grid)

                # Compute drift-only loss WITH gradients if training drift jointly
                drift_only_loss_tensor = None
                drift_l1_weight = getattr(hyperparameters, 'drift_l1_weight', 0.0)
                if hyperparameters.train_ode_with_sde and drift_l1_weight > 0:
                    # Compute predicted derivatives using drift network (matching ODE training)
                    # This ensures the loss is comparable to the ODE training phase
                    predicted_derivatives = []
                    current_state = real_trajectories[:, :-1, :]  # All states except last

                    for step_idx in range(len(window_time_grid) - 1):
                        t = float(window_time_grid[step_idx])
                        time_tensor = torch.full((current_state.shape[0], 1), t, device=DEVICE, dtype=current_state.dtype)
                        drift = drift_network.compute_drift(current_state[:, step_idx, :], time_tensor, external_inputs=None)
                        predicted_derivatives.append(drift)

                    predicted_derivatives = torch.stack(predicted_derivatives, dim=1)  # [batch, time-1, state]

                    # Compute true derivatives from trajectory (matching ODE training data preparation)
                    dt = float(window_time_grid[1] - window_time_grid[0])
                    current_positions = real_trajectories[:, :-1, :]
                    next_positions = real_trajectories[:, 1:, :]
                    true_derivatives = (next_positions - current_positions) / dt

                    # Compute HuberLoss on derivatives (matching ODE training criterion)
                    # This makes the loss values directly comparable between ODE and SDE phases
                    drift_only_loss_tensor = nn.functional.huber_loss(
                        predicted_derivatives, true_derivatives, delta=0.5
                    )

                generator_loss, loss_components = train_generator_step(
                    generator_optimiser,
                    critic_network,
                    fake_trajectories,
                    real_trajectories,
                    hyperparameters.sde_l1_weight,
                    moment_matching_weight=getattr(hyperparameters, 'moment_matching_weight', 0.0),
                    moment_matching_enabled=getattr(hyperparameters, 'moment_matching_enabled', False),
                    diffusion_network=diffusion_network,
                    drift_only_loss=drift_only_loss_tensor,
                    drift_l1_weight=drift_l1_weight,
                )
                epoch_generator_losses.append(generator_loss)

                # Accumulate loss components for epoch-level analysis
                for key, value in loss_components.items():
                    epoch_loss_components[key].append(value)

                # Detailed logging for first 5 epochs to understand loss magnitudes
                if epoch < 5 and batch_idx == 0:
                    print(f"\n{'='*80}")
                    print(f"EPOCH {epoch} - BATCH {batch_idx} - LOSS COMPONENT ANALYSIS")
                    print(f"{'='*80}")
                    print("\n[GENERATOR LOSS COMPONENTS]")
                    print(f"  1. Adversarial Loss (from critic):        {loss_components['adversarial']:>12.6f}")
                    print(f"  2. SDE L1 Loss (pathwise, unweighted):    {loss_components['sde_l1']:>12.6f}")
                    print(f"     → Weighted (× {hyperparameters.sde_l1_weight:.1f}):            {loss_components['sde_l1_weighted']:>12.6f}")
                    print(f"  3. Drift-only Loss (derivatives):         {loss_components['drift_only']:>12.6f}")
                    print(f"     → Weighted (× {drift_l1_weight:.1f}):            {loss_components['drift_only_weighted']:>12.6f}")
                    print(f"\n  TOTAL GENERATOR LOSS:                     {generator_loss:>12.6f}")
                    print("\n[RELATIVE MAGNITUDES]")
                    total = generator_loss
                    if total > 0:
                        print(f"  Adversarial:        {100*loss_components['adversarial']/total:>6.2f}%")
                        print(f"  SDE L1 (weighted):  {100*loss_components['sde_l1_weighted']/total:>6.2f}%")
                        print(f"  Drift (weighted):   {100*loss_components['drift_only_weighted']/total:>6.2f}%")
                    print("\n[WHERE EACH TERM APPEARS IN TRAINING]")
                    print("  • Adversarial Loss: Computed by critic network evaluating")
                    print("                      fake vs real trajectory distributions")
                    print("  • SDE L1 Loss:      Pathwise constraint on FULL SDE trajectories")
                    print("                      (drift + diffusion), ensures trajectory matching")
                    print("  • Drift-only Loss:  Derivative constraint on drift network ONLY")
                    print("                      (ignoring diffusion), ensures ODE accuracy")
                    print(f"{'='*80}\n")

                # Track drift-only loss for plotting (can reuse or recompute without gradients)
                if hyperparameters.train_ode_with_sde:
                    if drift_only_loss_tensor is not None:
                        # Reuse the computed loss value
                        epoch_drift_losses.append(drift_only_loss_tensor.item())
                    else:
                        # Compute for monitoring only (when drift_l1_weight=0)
                        with torch.no_grad():
                            drift_predictions = []
                            current_state = initial_states
                            for step_idx in range(1, len(window_time_grid)):
                                dt = float(window_time_grid[step_idx] - window_time_grid[step_idx - 1])
                                t = float(window_time_grid[step_idx - 1])
                                time_tensor = torch.full((current_state.shape[0], 1), t, device=DEVICE, dtype=current_state.dtype)
                                drift = drift_network.compute_drift(current_state, time_tensor, external_inputs=None)
                                next_state = current_state + drift * dt
                                drift_predictions.append(next_state)
                                current_state = next_state
                            drift_predictions = torch.stack(drift_predictions, dim=1)
                            drift_only_loss = nn.functional.smooth_l1_loss(
                                drift_predictions, real_trajectories[:, 1:, :]
                            )
                            epoch_drift_losses.append(drift_only_loss.item())

            if epoch_generator_losses:
                generator_losses.append(
                    sum(epoch_generator_losses) / len(epoch_generator_losses)
                )
            else:
                generator_losses.append(0.0)

            if epoch_critic_losses:
                critic_losses.append(sum(epoch_critic_losses) / len(epoch_critic_losses))
            else:
                critic_losses.append(0.0)
            
            # Track drift loss for plotting continuity when training jointly
            if hyperparameters.train_ode_with_sde and epoch_drift_losses:
                drift_losses.append(sum(epoch_drift_losses) / len(epoch_drift_losses))
            else:
                drift_losses.append(0.0)

            # Print epoch-level loss component summary at key checkpoints
            checkpoint_epochs = [0, 1, 2, 3, 4, 10, 25, 50, 100, 200, 300]
            if epoch in checkpoint_epochs or (epoch + 1) == num_epochs:
                # Compute mean of each component across all batches this epoch
                if epoch_loss_components['adversarial']:
                    mean_components = {
                        key: sum(values) / len(values)
                        for key, values in epoch_loss_components.items()
                    }
                    total_loss = generator_losses[-1]

                    print(f"\n{'='*80}")
                    print(f"EPOCH {epoch} SUMMARY - LOSS COMPONENT ANALYSIS")
                    print(f"{'='*80}")
                    print("\n[MEAN LOSS COMPONENTS ACROSS ALL BATCHES]")
                    print(f"  1. Adversarial Loss:               {mean_components['adversarial']:>12.6f}")
                    print(f"  2. SDE L1 Loss (unweighted):       {mean_components['sde_l1']:>12.6f}")
                    print(f"     → Weighted (× {hyperparameters.sde_l1_weight:.1f}):         {mean_components['sde_l1_weighted']:>12.6f}")
                    print(f"  3. Drift-only Loss (unweighted):   {mean_components['drift_only']:>12.6f}")
                    print(f"     → Weighted (× {getattr(hyperparameters, 'drift_l1_weight', 0.0):.1f}):         {mean_components['drift_only_weighted']:>12.6f}")
                    print(f"\n  TOTAL GENERATOR LOSS:              {total_loss:>12.6f}")
                    print("\n[RELATIVE CONTRIBUTION TO TOTAL LOSS]")
                    if total_loss > 0:
                        adv_pct = 100 * mean_components['adversarial'] / total_loss
                        sde_pct = 100 * mean_components['sde_l1_weighted'] / total_loss
                        drift_pct = 100 * mean_components['drift_only_weighted'] / total_loss
                        print(f"  Adversarial:        {adv_pct:>6.2f}%")
                        print(f"  SDE L1 (weighted):  {sde_pct:>6.2f}%")
                        print(f"  Drift (weighted):   {drift_pct:>6.2f}%")

                        # Issue warnings if one term dominates
                        if drift_pct > 70:
                            print(f"\n  ⚠️  WARNING: Drift loss dominates ({drift_pct:.1f}%)")
                            print("      This may prevent the diffusion network from learning!")
                            print(f"      Consider reducing DRIFT_L1_WEIGHT from {getattr(hyperparameters, 'drift_l1_weight', 0.0)}")
                        elif adv_pct > 80:
                            print(f"\n  ⚠️  WARNING: Adversarial loss dominates ({adv_pct:.1f}%)")
                            print("      Pathwise constraints may be too weak!")
                    print(f"{'='*80}\n")

            # Log to W&B if available (offset by actual ODE epochs trained)
            if wandb_run is not None:
                # Use actual ODE epochs if provided, otherwise fall back to configured value
                ode_epochs = actual_ode_epochs if actual_ode_epochs is not None else getattr(hyperparameters, 'number_of_epochs', 0)
                step = ode_epochs + epoch
                log_data = {
                    "sde/generator_loss": generator_losses[-1],
                    "sde/critic_loss": critic_losses[-1]
                }
                
                # Log drift loss as continuation of ODE training (same metric name for continuity)
                if drift_losses[-1] > 0:  # Only log if drift loss was computed
                    log_data["ode/train_loss"] = drift_losses[-1]  # Continue the ODE plot
                    log_data["sde/drift_loss"] = drift_losses[-1]  # Also keep separate metric
                
                wandb_run.log(log_data, step=step)
                
                # Log detailed checkpoint every 100 epochs
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    checkpoint_data = {
                        "sde/checkpoint/epoch": epoch + 1,
                        "sde/checkpoint/generator_loss": generator_losses[-1],
                        "sde/checkpoint/critic_loss": critic_losses[-1],
                    }
                    if drift_losses[-1] > 0:
                        checkpoint_data["sde/checkpoint/drift_loss"] = drift_losses[-1]
                    
                    # Add statistics about batch losses if available
                    if epoch_generator_losses:
                        checkpoint_data["sde/checkpoint/generator_loss_mean"] = sum(epoch_generator_losses) / len(epoch_generator_losses)
                        checkpoint_data["sde/checkpoint/generator_loss_std"] = (
                            sum((x - checkpoint_data["sde/checkpoint/generator_loss_mean"]) ** 2 for x in epoch_generator_losses) / len(epoch_generator_losses)
                        ) ** 0.5 if len(epoch_generator_losses) > 1 else 0.0
                    if epoch_critic_losses:
                        checkpoint_data["sde/checkpoint/critic_loss_mean"] = sum(epoch_critic_losses) / len(epoch_critic_losses)
                        checkpoint_data["sde/checkpoint/critic_loss_std"] = (
                            sum((x - checkpoint_data["sde/checkpoint/critic_loss_mean"]) ** 2 for x in epoch_critic_losses) / len(epoch_critic_losses)
                        ) ** 0.5 if len(epoch_critic_losses) > 1 else 0.0
                    if epoch_drift_losses:
                        checkpoint_data["sde/checkpoint/drift_loss_mean"] = sum(epoch_drift_losses) / len(epoch_drift_losses)
                        checkpoint_data["sde/checkpoint/drift_loss_std"] = (
                            sum((x - checkpoint_data["sde/checkpoint/drift_loss_mean"]) ** 2 for x in epoch_drift_losses) / len(epoch_drift_losses)
                        ) ** 0.5 if len(epoch_drift_losses) > 1 else 0.0
                    
                    wandb_run.log(checkpoint_data, step=step)

            pbar.set_postfix(
                {"Gen Loss": generator_losses[-1], "Critic Loss": critic_losses[-1]}
            )

    return generator_losses, critic_losses, drift_losses
