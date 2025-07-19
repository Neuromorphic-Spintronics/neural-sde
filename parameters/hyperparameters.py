from dataclasses import dataclass
from typing import List, Tuple


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
    discriminator_network: NetworkArchitecture

    # System dimension parameters
    state_dimension: int  # Dimension of the system state variables
    input_dimension: int  # Dimension of external input signals

    # Auxiliary variable parameters
    hidden_variables: int = 0  # Number of hidden/latent variables in the system
    external_hidden_variables: int = 0  # Number of external hidden variables

    # Training parameters
    timestep: float = 0.01  # Integration timestep for numerical methods, this perhaps could be adjusted dynamically to satisfy the CFL condition
    learning_rates: LearningRates = LearningRates(
        drift=0.001,
        diffusion=0.001,
        discriminator=0.001,
    )  # Learning rates for drift, diffusion, and discriminator networks
