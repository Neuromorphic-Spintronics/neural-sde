from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class LearningRates:
    """Learning rates for different network components."""

    drift: float
    diffusion: float
    critic: float
    generator: float

    def __post_init__(self):
        """Validate learning rates."""
        if self.drift <= 0:
            raise ValueError("Drift learning rate must be greater than 0")
        if self.diffusion <= 0:
            raise ValueError("Diffusion learning rate must be greater than 0")
        if self.critic <= 0:
            raise ValueError("Critic learning rate must be greater than 0")
        if self.generator <= 0:
            raise ValueError("Generator learning rate must be greater than 0")


@dataclass(frozen=True)
class NetworkArchitecture:
    """Network architecture specification with validation."""

    input_size: int
    hidden_sizes: List[int]
    output_size: int

    def __post_init__(self):
        """Validate network architecture."""
        if self.input_size <= 0:
            raise ValueError("Input size must be greater than 0")
        if self.output_size <= 0:
            raise ValueError("Output size must be greater than 0")
        if not all(size > 0 for size in self.hidden_sizes):
            raise ValueError("All hidden layer sizes must be greater than 0")

    @property
    def layer_sizes(self) -> List[int]:
        """Get the complete list of layer sizes including input and output."""
        return [self.input_size] + self.hidden_sizes + [self.output_size]

    @property
    def num_layers(self) -> int:
        """Get the total number of layers (including input and output)."""
        return len(self.layer_sizes)

    @property
    def num_hidden_layers(self) -> int:
        """Get the number of hidden layers."""
        return len(self.hidden_sizes)


@dataclass(frozen=True)
class Hyperparameters:
    """
    Hyperparameters for neural SDE/ODE training.

    This dataclass contains all the parameters needed to configure and train
    a neural SDE/ODE model using the NeuralSDE class.
    """

    # Network architecture parameters
    drift_network: NetworkArchitecture
    diffusion_network: NetworkArchitecture
    critic_network: NetworkArchitecture

    # System dimension parameters
    state_dimension: int  # Dimension of the system state variables
    input_dimension: int  # Dimension of external input signals

    # Auxiliary variable parameters
    hidden_variables: int = 0  # Number of hidden/latent variables in the system
    external_hidden_variables: int = 0  # Number of external hidden variables

    # Training parameters
    timestep: float = 0.01  # Integration timestep for numerical methods, this perhaps could be adjusted dynamically to satisfy the CFL condition for PDEs
    learning_rates: LearningRates = LearningRates(
        drift=0.001,
        diffusion=0.001,
        critic=0.001,
        generator=0.001,
    )  # Learning rates for drift, diffusion, and critic networks
    number_of_epochs: int = 100
    number_of_gan_epochs: int = 100
    critic_updates: int = 5 # Number of critic updates per generator update (Arjovsky et al., 2017)
    gradient_penalty_weight: float = 10.0 # Weight for WGAN-GP gradient penalty (Arjovsky et al., 2017)
    batch_size: int = 64
    sde_l1_weight: float = 0.1 
    moment_matching_weight: float = 0.0  # Weight for statistical moment matching in generator loss
    moment_matching_enabled: bool = False  # Whether to enable statistical moment matching

    @classmethod
    def defaults(cls) -> "Hyperparameters":
        """Return the canonical hyperparameter configuration for the library.

        The defaults mirror the settings used in the Duffing oscillator notebook
        and provide a well-tested starting point for other systems. They include
        a three-layer drift network with 256 hidden units, matching diffusion and
        critic architectures, and conservative learning rates for WGAN-GP.
        """

        state_dim = 3  # [q, v, xi]
        input_dim = 0

        drift_architecture = NetworkArchitecture(
            input_size=state_dim + 1,  # state concatenated with time
            hidden_sizes=[256, 256, 256],
            output_size=state_dim,
        )

        noise_dimension = 1
        diffusion_architecture = NetworkArchitecture(
            input_size=state_dim + 1,
            hidden_sizes=[256, 256, 256],
            output_size=state_dim * noise_dimension,
        )

        critic_window = 64
        critic_architecture = NetworkArchitecture(
            input_size=critic_window * state_dim,
            hidden_sizes=[256, 256],
            output_size=1,
        )

        learning_rates = LearningRates(
            drift=5.0e-4,
            diffusion=5.0e-4,
            critic=2.0e-4,
            generator=3.0e-4,
        )

        return cls(
            state_dimension=state_dim,
            input_dimension=input_dim,
            timestep=25.0 / 255,
            drift_network=drift_architecture,
            diffusion_network=diffusion_architecture,
            critic_network=critic_architecture,
            learning_rates=learning_rates,
            batch_size=32,
            number_of_epochs=192,
            number_of_gan_epochs=320,
            critic_updates=3,
            gradient_penalty_weight=5.0,
            sde_l1_weight=0.25,
            moment_matching_weight=0.1,
            moment_matching_enabled=False,
        )



