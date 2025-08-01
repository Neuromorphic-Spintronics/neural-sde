"""
Neural Stochastic Differential Equation (Neural SDE) model with adversarial training.

This module implements a neural SDE framework that can learn the dynamics of stochastic systems through a combination of neural networks representing drift and diffusion terms, along with adversarial training using a discriminator.

Key components:
- `DriftNet`: Neural network approximating deterministic dynamics mu(x, t, u)
- `DiffusionNet`: Neural network approximating stochastic dynamics sigma(x, t, u)  
- `DiscriminatorNet`: Critic network for adversarial training
- `NeuralSDE`: Main class combining networks with SDE integration for simulation
"""

from __future__ import annotations

from typing import Optional, Callable
import torch
from torch import nn, Tensor

from parameters.hyperparameters import NetworkArchitecture, Hyperparameters
from .modules import FeedForwardNetwork
from .protocols import NeuralSDEProtocol
from .integrators import auto_select_integrator, auto_select_matrix_integrator

# Device selection
def get_device() -> torch.device:
    """Select the best available device: MPS (Apple Silicon), CUDA (NVIDIA), or CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()

class DriftNet(FeedForwardNetwork):
    """
    Neural network approximating the drift term mu(x, t, u) in the neural SDE.
    
    The drift network represents the deterministic component of the system dynamics. It takes as input the current state, time, and optional external forcing to predict the deterministic rate of change.
    
    The network architecture is specified through the hyperparameters and uses the established FeedForwardNetwork base class for proper initialisation.
    
    Args:
        architecture: Network architecture specification from hyperparameters
        activation: Activation function (defaults to nn.Tanh)
        device: Computation device (defaults to globally configured device)
        
    Example:
        >>> from parameters.hyperparameters import NetworkArchitecture
        >>> architecture = NetworkArchitecture(input_size=4, hidden_sizes=[64, 64], output_size=2)
        >>> drift_net = DriftNet(architecture)
        >>> # Input: [batch_size, state_dim + 1 + input_dim] (state + time + inputs)
        >>> # Output: [batch_size, state_dim] (drift vector)
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        activation: Callable[[], nn.Module] = nn.Tanh,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the drift network with specified architecture.
        
        Args:
            architecture: Network structure specification
            activation: Activation function
            device: Computation device
        """
        super().__init__(architecture, activation, device=device)

    def compute_drift(
        self, state: Tensor, time: Tensor, external_inputs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the drift term for given state, time, and inputs.
        
        Args:
            state: Current state of shape [batch_size, state_dimension]
            time: Current time of shape [batch_size, 1] or [batch_size]
            external_inputs: External forcing of shape [batch_size, input_dimension]
            
        Returns:
            Drift vector of shape [batch_size, state_dim]
        """
        # Ensure time has the correct shape
        if time.dim() == 1:
            time = time.unsqueeze(-1)  # Convert [batch_size] to [batch_size, 1]

        # Prepare input tensor by concatenating state, time, and external inputs
        input_components = [state, time]

        if external_inputs is not None:
            input_components.append(external_inputs)
        else:
            # Create zero external inputs if none provided to match expected input size
            batch_size = state.shape[0]
            device = state.device
            # Calculate expected external input dimension
            expected_input_size = self.architecture.input_size
            state_time_size = state.shape[1] + time.shape[1]
            external_input_size = expected_input_size - state_time_size

            if external_input_size > 0:
                zero_external_inputs = torch.zeros(
                    batch_size, external_input_size, device=device
                )
                input_components.append(zero_external_inputs)

        # Concatenate all components along the feature dimension
        network_input = torch.cat(input_components, dim=-1)

        # Pass through the neural network
        return self.forward(network_input)


class DiffusionNet(FeedForwardNetwork):
    """
    Neural network approximating the diffusion term sigma(x, t, u) in the neural SDE.
    
    The diffusion network represents the stochastic component of the system dynamics.
    
    Args:
        architecture: Network architecture specification from hyperparameters
        state_dimension: Dimension of the system state space  
        noise_dimension: Dimension of the noise process
        activation: Activation function (defaults to nn.Tanh)
        device: Computation device
        
    Example:
        >>> archicture = NetworkArchitecture(input_size=4, hidden_sizes=[32, 32], output_size=6)
        >>> # For 3D state with 2D noise: output_size = 3 * 2 = 6
        >>> diffusion_net = DiffusionNet(architecture, state_dimension=3, noise_dimension=2)
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        state_dimension: int,
        noise_dimension: int,
        activation: Callable[[], nn.Module] = nn.Tanh,
        *,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the diffusion network.
        
        Args:
            architecture: Network structure specification
            state_dimension: System state dimension
            noise_dimension: Noise process dimension  
            activation: Activation function
            device: Computation device
        """ 
        super().__init__(architecture, activation, device=device)
        
        self.state_dimension = state_dimension
        self.noise_dimension = noise_dimension
        
        # Validate that output size matches expected diffusion matrix size
        expected_output_size = state_dimension * noise_dimension
        if self.architecture.output_size != expected_output_size:
            raise ValueError(
                f"Network output size {self.architecture.output_size} does not match "
                f"expected diffusion matrix size {expected_output_size} "
                f"(state_dim={state_dimension} * noise_dim={noise_dimension})"
            )

    def compute_diffusion(
        self, 
        state: Tensor, 
        time: Tensor, 
        external_inputs: Optional[Tensor] = None
    ) -> Tensor:
        """
        Compute the diffusion matrix for given state, time, and inputs.
        
        Args:
            state: Current state of shape [batch_size, state_dim]
            time: Current time of shape [batch_size, 1] or [batch_size]
            external_inputs: External forcing of shape [batch_size, input_dim]
            
        Returns:
            Diffusion matrix of shape [batch_size, state_dim, noise_dim]
        """
        # Ensure time has the correct shape
        if time.dim() == 1:
            time = time.unsqueeze(-1)  # Convert [batch_size] to [batch_size, 1]

        # Calculate the correct external input dimension
        time_size = time.shape[1]
        state_size = state.shape[1]
        expected_input_size = self.architecture.input_size
        # The input to the network is constructed as [state, time, external_inputs],
        # so the total input size is: state_size + time_size + external_input_size.
        # To determine the size for external_inputs, subtract the space used by state and time.
        external_input_size = expected_input_size - state_size - time_size

        # Prepare input tensor by concatenating state, time, and external inputs
        input_components = [state, time]
        if external_inputs is not None:
            input_components.append(external_inputs)
        else:
            batch_size = state.shape[0]
            device = state.device
            if external_input_size > 0:
                zero_external_inputs = torch.zeros(
                    batch_size, external_input_size, device=device
                )
                input_components.append(zero_external_inputs)
        # Concatenate all components along the feature dimension
        network_input = torch.cat(input_components, dim=-1)

        # Pass through the neural network
        diffusion_output = self.forward(network_input)
        batch_size = diffusion_output.shape[0]
        diffusion_matrix = diffusion_output.reshape(
            batch_size, self.state_dimension, self.noise_dimension
        )
        return diffusion_matrix


class DiscriminatorNet(FeedForwardNetwork):
    """
    Discriminator network for adversarial training of neural SDEs.
    
    This network acts as a critic that scores trajectory segments to distinguish between real (from the 'true' (simulated) system) and synthetic (from the neural SDE) trajectories. Used in Wasserstein GAN training with gradient penalty (to ensure 1-Lipschitz).

    Args:
        architecture: Network architecture specification
        trajectory_length: Length of trajectory segments to evaluate
        activation: Activation function
        device: Computation device
        
    Example:
        >>> # Discriminator for trajectory chunks of length 64 with 3D states
        >>> architecture = NetworkArchitecture(input_size=192, hidden_sizes=[64, 32], output_size=1)
        >>> disc = DiscriminatorNet(architecture, trajectory_length=64)
    """

    def __init__(
        self,
        architecture: NetworkArchitecture,
        trajectory_length: int,
        activation: Callable[[], nn.Module] = nn.Tanh,
        *,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the discriminator network.
        
        Args:
            architecture: Network structure specification
            trajectory_length: Length of trajectory segments
            activation: Activation function
            device: Computation device
        """
        super().__init__(architecture, activation, device=device)
        self.trajectory_length = trajectory_length
        
        # Validate architecture matches expected input size
        # Input should be trajectory_length * state_dim
        if architecture.output_size != 1:
            raise ValueError(
                f"Discriminator output size must be 1, got {architecture.output_size}"
            )

    def score_trajectory(self, trajectory_segment: Tensor) -> Tensor:
        """
        Score a trajectory segment using the discriminator network.
        
        This method takes a batch of trajectory segments and flattens them
        to pass through the feedforward network, producing a score for each
        trajectory that indicates how "real" vs "fake" the discriminator
        believes each trajectory to be.
        
        Args:
            trajectory_segment: Trajectory of shape [batch_size, trajectory_length, state_dim]
            
        Returns:
            Scores of shape [batch_size, 1] - higher scores indicate more "real" trajectories
            
        Raises:
            ValueError: If trajectory dimensions don't match expected architecture
        """
        batch_size, traj_len, state_dim = trajectory_segment.shape
        
        # Validate input dimensions
        if traj_len != self.trajectory_length:
            raise ValueError(
                f"Expected trajectory length {self.trajectory_length}, "
                f"got {traj_len}"
            )
        
        expected_input_size = self.trajectory_length * state_dim
        if expected_input_size != self.architecture.input_size:
            raise ValueError(
                f"Trajectory dimensions ({traj_len} × {state_dim} = {expected_input_size}) "
                f"don't match architecture input size {self.architecture.input_size}"
            )
        
        # Flatten trajectory for feedforward network: [batch_size, trajectory_length * state_dim]
        flattened_trajectory = trajectory_segment.view(batch_size, -1)
        
        # Pass through the discriminator network
        scores = self.forward(flattened_trajectory)
        
        return scores

class NeuralSDE(nn.Module, NeuralSDEProtocol):
    """
    Core Neural SDE model combining drift, diffusion, and integration.
    
    This is the main class that orchestrates the neural SDE simulation by combining the neural networks with numerical integration. It can operate in both deterministic-only mode (diffusion_net=None) and stochastic mode.
    
    The system automatically selects the appropriate integration method:
    - Stochastic Heun method for neural SDEs (when diffusion_net is present)
    - Second-order Runge-Kutta (RK2) method for neural ODEs (when diffusion_net is None)
    
    **Important Constraint**: For stochastic systems (when diffusion_net is present), a discriminator network is required for adversarial training. For deterministic systems, the discriminator is optional.
    
    Args:
        hyperparameters: Complete hyperparameter specification including network architectures, provided by the Hyperparameters class
        timestep: Integration timestep (defaults to hyperparameters value)
        device: Computation device
        
    Example:
        >>> from parameters.hyperparameters import Hyperparameters
        >>> hyperparams = Hyperparameters(...)
        >>> neural_sde = NeuralSDE(hyperparams)
        >>> 
        >>> # Simulate trajectory
        >>> inputs = torch.randn(batch_size, input_dim, num_timesteps)
        >>> initial_state = torch.randn(batch_size, state_dim)
        >>> trajectory = neural_sde(inputs, initial_state=initial_state)
    """

    def __init__(
        self,
        hyperparameters: Hyperparameters,
        timestep: Optional[float] = None,
        device: torch.device = DEVICE,
    ) -> None:
        """
        Initialise the neural SDE model.
        
        Args:
            hyperparameters: Complete configuration specification
            timestep: Integration timestep (uses hyperparameters if None)
            device: Computation device
        """
        super().__init__()
        
        self.hyperparameters = hyperparameters
        self.timestep = timestep if timestep is not None else hyperparameters.timestep
        self.state_dimension = hyperparameters.state_dimension
        self.input_dimension = hyperparameters.input_dimension
        self.device = device
        
        # Validate hyperparameters
        self._validate_hyperparameters(hyperparameters)
        
        # Initialise neural networks
        self.drift_net = DriftNet(
            architecture=hyperparameters.drift_network,
            device=device
        )
        
        # Diffusion network is required for stochastic SDEs
        self.diffusion_net = DiffusionNet(
            architecture=hyperparameters.diffusion_network,
            state_dimension=hyperparameters.state_dimension,
            noise_dimension=self._infer_noise_dimension(hyperparameters),
            device=device
        )
        
        # For stochastic systems (with diffusion), discriminator is required for adversarial training
        # For deterministic systems (no diffusion), discriminator is optional
        if self.diffusion_net is not None:
            # Stochastic case: discriminator is required
            if not hasattr(hyperparameters, 'discriminator_network') or hyperparameters.discriminator_network is None:
                raise ValueError(
                    "Discriminator network is required for stochastic Neural SDEs "
                    "(when diffusion network is present) for adversarial training"
                )
            self.discriminator_net = DiscriminatorNet(
                architecture=hyperparameters.discriminator_network,
                trajectory_length=self._infer_trajectory_length(hyperparameters),
                device=device
            )
        else:
            # Deterministic case: discriminator is optional
            if hasattr(hyperparameters, 'discriminator_network') and hyperparameters.discriminator_network is not None:
                self.discriminator_net = DiscriminatorNet(
                    architecture=hyperparameters.discriminator_network,
                    trajectory_length=self._infer_trajectory_length(hyperparameters),
                    device=device
                )
            else:
                self.discriminator_net = None

    def _validate_hyperparameters(self, hyperparameters: Hyperparameters) -> None:
        """
        Validate that hyperparameters are consistent and complete.
        
        Args:
            hyperparameters: Configuration to validate
            
        Raises:
            ValueError: If hyperparameters are invalid
        """
        # Check required attributes exist
        required_attrs = ['drift_network', 'diffusion_network', 'state_dimension', 'input_dimension', 'timestep']
        for attr in required_attrs:
            if not hasattr(hyperparameters, attr):
                raise ValueError(f"Hyperparameters missing required attribute: {attr}")
            if getattr(hyperparameters, attr) is None:
                raise ValueError(f"Hyperparameters attribute {attr} cannot be None")
        
        # Validate network architectures
        if not hasattr(hyperparameters.drift_network, 'input_size'):
            raise ValueError("Drift network architecture missing input_size")
        if not hasattr(hyperparameters.drift_network, 'output_size'):
            raise ValueError("Drift network architecture missing output_size")
        if not hasattr(hyperparameters.diffusion_network, 'input_size'):
            raise ValueError("Diffusion network architecture missing input_size")
        if not hasattr(hyperparameters.diffusion_network, 'output_size'):
            raise ValueError("Diffusion network architecture missing output_size")
        
        # Validate timestep is positive
        if hyperparameters.timestep <= 0:
            raise ValueError(f"Timestep must be positive, got {hyperparameters.timestep}")
        
        # Validate dimensions are positive
        if hyperparameters.state_dimension <= 0:
            raise ValueError(f"State dimension must be positive, got {hyperparameters.state_dimension}")
        if hyperparameters.input_dimension < 0:
            raise ValueError(f"Input dimension must be non-negative, got {hyperparameters.input_dimension}")
        
        # Validate discriminator network is provided for stochastic systems
        if not hasattr(hyperparameters, 'discriminator_network') or hyperparameters.discriminator_network is None:
            raise ValueError(
                "Discriminator network is required for stochastic Neural SDEs "
                "(when diffusion network is present) for adversarial training"
            )
        
        # Validate discriminator network architecture if provided
        if hasattr(hyperparameters, 'discriminator_network') and hyperparameters.discriminator_network is not None:
            if not hasattr(hyperparameters.discriminator_network, 'input_size'):
                raise ValueError("Discriminator network architecture missing input_size")
            if not hasattr(hyperparameters.discriminator_network, 'output_size'):
                raise ValueError("Discriminator network architecture missing output_size")
            if hyperparameters.discriminator_network.output_size != 1:
                raise ValueError(
                    f"Discriminator network output size must be 1, got {hyperparameters.discriminator_network.output_size}"
                )

    def _infer_noise_dimension(self, hyperparameters: Hyperparameters) -> int:
        """
        Infer noise dimension from diffusion network output size. The noise dimension determines the number of independent Brownian motion processes driving the stochastic dynamics.
        
        Args:
            hyperparameters: Configuration containing network architectures
            
        Returns:
            Noise dimension (number of independent Brownian motions)
            
        Examples:
            >>> # Single scalar noise affecting all states
            >>> state_dim, noise_dim = 3, 1  # output_size = 3
            >>> 
            >>> # Independent noise per state variable  
            >>> state_dim, noise_dim = 3, 3  # output_size = 9
            >>>
            >>> # Five coloured noise sources
            >>> state_dim, noise_dim = 3, 5  # output_size = 15
        """
        state_dim = hyperparameters.state_dimension
        diffusion_output_size = hyperparameters.diffusion_network.output_size
        
        # Diffusion matrix is flattened: state_dim * noise_dim = output_size
        if diffusion_output_size % state_dim != 0:
            raise ValueError(
                f"Diffusion network output size {diffusion_output_size} is not "
                f"divisible by state dimension {state_dim}. "
                f"Expected output_size = state_dimension * noise_dimension."
            )
        
        noise_dim = diffusion_output_size // state_dim
        if noise_dim <= 0:
            raise ValueError(f"Inferred noise dimension {noise_dim} must be positive")
        
        return noise_dim

    def _infer_trajectory_length(self, hyperparameters: Hyperparameters) -> int:
        """
        Infer trajectory length from discriminator network input size.
        
        Args:
            hyperparameters: Configuration containing network architectures
            
        Returns:
            Trajectory length for discriminator input
        """
        state_dim = hyperparameters.state_dimension
        discriminator_input_size = hyperparameters.discriminator_network.input_size
        
        # Discriminator input is flattened trajectory: trajectory_length * state_dim
        if discriminator_input_size % state_dim != 0:
            raise ValueError(
                f"Discriminator network input size {discriminator_input_size} is not "
                f"divisible by state dimension {state_dim}"
            )
        
        trajectory_length = discriminator_input_size // state_dim # floor division
        if trajectory_length <= 0:
            raise ValueError(f"Inferred trajectory length {trajectory_length} must be positive")
        
        return trajectory_length

    def forward(
        self, 
        external_inputs: Tensor, 
        initial_state: Tensor,
        initial_time: float = 0.0
    ) -> Tensor:
        """
        Simulate neural SDE trajectory given external inputs.
        
        Steps through time using the configured integration method, with the neural networks providing drift and diffusion terms at each step. Gradients are retained for training via backpropagation through time (as per original paper by Manneschi et al.) or an adjoint method (to be implemented).
        
        Args:
            external_inputs: External forcing/control inputs of shape [batch_size, input_dimension, num_timesteps]
            initial_state: Initial state conditions of shape [batch_size, state_dimension]  
            initial_time: Starting time for simulation
            
        Returns:
            Simulated trajectory of shape [batch_size, state_dim, num_timesteps]
            
        Example:
            >>> batch_size, state_dim, input_dim, num_steps = 32, 3, 2, 100
            >>> inputs = torch.randn(batch_size, input_dim, num_steps)
            >>> x0 = torch.randn(batch_size, state_dim)
            >>> trajectory = neural_sde(inputs, initial_state=x0)
            >>> print(trajectory.shape)  # torch.Size([32, 3, 100])
        """
        # Move inputs to the correct device
        external_inputs = external_inputs.to(self.device)
        initial_state = initial_state.to(self.device)
        
        batch_size, input_dim, num_timesteps = external_inputs.shape
        state_dim = initial_state.shape[1]
        
        # Validate input dimensions
        if input_dim != self.input_dimension:
            raise ValueError(
                f"External input dimension {input_dim} does not match "
                f"expected input dimension {self.input_dimension}"
            )
        if state_dim != self.state_dimension:
            raise ValueError(
                f"Initial state dimension {state_dim} does not match "
                f"expected state dimension {self.state_dimension}"
            )

        # Automatically select integration method based on whether diffusion is present
        step_integrator: Callable[..., Tensor] = auto_select_integrator(has_diffusion=self.diffusion_net is not None)
        matrix_integrator: Callable[..., Tensor] = auto_select_matrix_integrator(has_diffusion=self.diffusion_net is not None)

        # Initialise trajectory storage
        trajectory = torch.zeros(batch_size, state_dim, num_timesteps, device=self.device)
        current_state = initial_state.clone()
        current_time = initial_time

        # Simulate trajectory step by step
        for step in range(num_timesteps):
            # Store current state
            trajectory[:, :, step] = current_state
            
            # Get external input at this step
            external_input_at_step = external_inputs[:, :, step]
            
            # Create drift function for this step
            def drift_func(time: float, state: Tensor) -> Tensor:
                return self._compute_drift_at_step(state, time, external_input_at_step)
            
            # Create diffusion matrix function for this step
            def diffusion_matrix_func(time: float, state: Tensor) -> Tensor:
                return self._compute_diffusion_at_step(state, time, external_input_at_step)
            
            # Integrate one step forward using matrix-aware integration
            if self.diffusion_net is not None:
                # Stochastic case - use matrix-aware integrator
                current_state = matrix_integrator(
                    drift_func, diffusion_matrix_func, current_state, current_time, self.timestep
                )
            else:
                # Deterministic case - use standard integrator
                current_state = step_integrator(
                    drift_func, current_state, current_time, self.timestep
                )
            
            # Update time
            current_time += self.timestep

        return trajectory

    def _compute_drift_at_step(
        self, 
        state: Tensor, 
        time: float, 
        external_input: Tensor
    ) -> Tensor:
        """
        Compute drift term at a single time step.
        
        Args:
            state: Current state [batch_size, state_dim]
            time: Current time
            external_input: External inputs at this step [batch_size, input_dim]
            
        Returns:
            Drift vector [batch_size, state_dim]
        """
        # Convert time to tensor with correct batch size
        batch_size = state.shape[0]
        time_tensor = torch.full((batch_size,), time, device=state.device)
        
        return self.drift_net.compute_drift(state, time_tensor, external_input)

    def _compute_diffusion_at_step(
        self, 
        state: Tensor, 
        time: float, 
        external_input: Tensor
    ) -> Optional[Tensor]:
        """
        Compute diffusion term at a single time step.
        
        Args:
            state: Current state [batch_size, state_dim]
            time: Current time
            external_input: External inputs at this step [batch_size, input_dim]
            
        Returns:
            Diffusion matrix [batch_size, state_dim, noise_dim] or None if deterministic
        """
        if self.diffusion_net is None:
            return None
            
        # Convert time to tensor with correct batch size
        batch_size = state.shape[0]
        time_tensor = torch.full((batch_size,), time, device=state.device)
        
        return self.diffusion_net.compute_diffusion(state, time_tensor, external_input)

    def _generate_wiener_increments(
        self, 
        batch_size: int, 
        noise_dim: int, 
        device: torch.device, 
        timestep: float
    ) -> Tensor:
        """
        Generate Wiener process increments for the noise dimension.
        
        Args:
            batch_size: Number of trajectories
            noise_dim: Dimension of the noise process
            device: Device for tensor creation
            timestep: Integration timestep
            
        Returns:
            Wiener increments of shape [batch_size, noise_dim]
        """
        # Generate standard normal random variables and scale by sqrt(timestep)
        standard_normal = torch.randn(batch_size, noise_dim, device=device)
        return standard_normal * torch.sqrt(torch.tensor(timestep, device=device))

    def get_drift_network(self) -> DriftNet:
        """Return the drift network component."""
        return self.drift_net

    def get_diffusion_network(self) -> Optional[DiffusionNet]:
        """Return the diffusion network component (may be None for deterministic phase)."""
        return self.diffusion_net

    def get_discriminator_network(self) -> Optional[DiscriminatorNet]:
        """Return the discriminator network component (may be None if not using adversarial training)."""
        return self.discriminator_net

    def count_total_parameters(self) -> dict[str, int]:
        """
        Count parameters in each network component.
        
        Returns:
            Dictionary mapping component names to parameter counts
        """
        from .modules import count_network_parameters
        
        param_counts = {
            'drift_net': count_network_parameters(self.drift_net),
            'diffusion_net': count_network_parameters(self.diffusion_net) if self.diffusion_net else 0,
            'discriminator_net': count_network_parameters(self.discriminator_net) if self.discriminator_net else 0,
        }
        
        param_counts['total'] = sum(param_counts.values())
        
        return param_counts