# This file initialises the parameters package.

from .duffing_oscillator import DuffingOscillatorParameters
from .leaky_integrator import LeakyIntegratorParameters
from .hyperparameters import Hyperparameters
from typing import Final, Union

HYPERPARAMETERS: Final[Hyperparameters] = Hyperparameters()
LEAKY_INTEGRATOR_PARAMETERS: Final[LeakyIntegratorParameters] = LeakyIntegratorParameters()
DUFFING_OSCILLATOR_PARAMETERS: Final[DuffingOscillatorParameters] = DuffingOscillatorParameters()

def get_parameters(system: str) -> Union[Hyperparameters, LeakyIntegratorParameters, DuffingOscillatorParameters]:
    if system == "leaky_integrator":
        return LEAKY_INTEGRATOR_PARAMETERS
    elif system == "duffing_oscillator":
        return DUFFING_OSCILLATOR_PARAMETERS
    else:
        raise ValueError(f"Invalid system: {system}") 