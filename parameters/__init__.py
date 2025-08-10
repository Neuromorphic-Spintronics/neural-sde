# This file initialises the parameters package.

from .duffing_oscillator import DuffingOscillatorParameters, PhysicalDuffingParameters
from .hyperparameters import NetworkArchitecture, Hyperparameters
from typing import Final, Union

HYPERPARAMETERS: Final[Hyperparameters] = Hyperparameters(
    drift_network=NetworkArchitecture(
        input_size=1, hidden_sizes=[100, 100], output_size=1
    ),
    diffusion_network=NetworkArchitecture(
        input_size=1, hidden_sizes=[100, 100], output_size=1
    ),
    discriminator_network=NetworkArchitecture(
        input_size=1, hidden_sizes=[100, 100], output_size=1
    ),
    state_dimension=1,
    input_dimension=1,
)

DUFFING_OSCILLATOR_PARAMETERS: Final[DuffingOscillatorParameters] = (
    PhysicalDuffingParameters().to_dimensionless()
)


def get_parameters(
    system: str,
) -> Union[Hyperparameters, DuffingOscillatorParameters]:
    if system == "duffing_oscillator":
        return DUFFING_OSCILLATOR_PARAMETERS
    else:
        raise ValueError(f"Invalid system: {system}")
