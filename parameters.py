# Initialise the physical parameters for the systems of choice and the hyperparameters for the neural network

from __future__ import annotations

from typing import Final, Union

from dataclasses import dataclass

@dataclass(frozen=True)
class Hyperparameters:
    NotImplementedError("Hyperparameters not implemented")


@dataclass(frozen=True)
class LeakyIntegratorParameters:
    noise_standard_deviation: float = 0.0
    auxiliary_variables: int = 2
    timestep: float = 0.1
    NotImplementedError("LeakyIntegratorParameters not implemented")

@dataclass(frozen=True)
class DuffingOscillatorParameters:
    
    NotImplementedError("DuffingOscillatorParameters not implemented")

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