from __future__ import annotations

from typing import Any, Dict, List

# Import system-specific classes for implemented systems
from systems.duffing_oscillator import DuffingOscillator
from parameters.duffing_oscillator import PhysicalDuffingParameters

# Leaky integrator is not implemented yet, so we don't import its classes.
# When it is implemented, uncomment the following lines:
# from systems.leaky_integrator import LeakyIntegrator
# from parameters.leaky_integrator import PhysicalLeakyIntegratorParameters


SYSTEM_REGISTRY: Dict[str, Dict[str, Any]] = {
    "duffing": {
        "system_class": DuffingOscillator,
        "parameter_class": PhysicalDuffingParameters,
        "description": "A forced, damped oscillator with a nonlinear restoring force.",
        "is_implemented": True,
        "state_variables": ["q", "v", "xi"],
        "state_units": ["m", "m/s", "N/A"],
        "has_external_forcing": True,
        "default_hidden_layers": [128, 128, 128],
        "state_dimension": 2,  # Position and velocity are the core state
        "full_state_dimension": 3, # Includes auxiliary noise state
        "get_drift_function": DuffingOscillator.get_drift_function,
    },
    "leaky_integrator": {
        "system_class": None,  # LeakyIntegrator,
        "parameter_class": None,  # PhysicalLeakyIntegratorParameters,
        "description": "A simple first-order linear system (Not Implemented).",
        "is_implemented": False,
        "state_variables": ["V"],
        "state_units": ["V"],
        "has_external_forcing": False,
        "default_hidden_layers": [32, 32],
        "state_dimension": 1,
        "full_state_dimension": 1,
        "get_drift_function": None,
    },
}


def get_system_info(system_name: str) -> Dict[str, Any]:
    """
    Retrieve information about a system from the registry.

    Args:
        system_name: The name of the system.

    Returns:
        A dictionary containing system information.

    Raises:
        ValueError: If the system is not found in the registry.
    """
    system_info = SYSTEM_REGISTRY.get(system_name)
    if system_info is None:
        raise ValueError(f"System '{system_name}' not found in the registry. Available systems: {get_available_systems()}")
    if not system_info["is_implemented"]:
        raise NotImplementedError(f"System '{system_name}' is not implemented yet.")
    return system_info


def get_available_systems() -> List[str]:
    """
    Get a list of all available system names.

    Returns:
        A list of strings with the names of the systems.
    """
    return list(SYSTEM_REGISTRY.keys())


def get_implemented_systems() -> List[str]:
    """
    Get a list of implemented system names.

    Returns:
        A list of strings with the names of the implemented systems.
    """
    return [name for name, info in SYSTEM_REGISTRY.items() if info["is_implemented"]]
