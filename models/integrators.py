"""Compatibility wrapper exposing core integrators under ``models.integrators``."""

from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Any

_core_integrators: ModuleType = import_module("neural_dynamics.core.integrators")

__all__ = [
    name
    for name in getattr(_core_integrators, "__all__", [])
    if not name.startswith("_")
]

if not __all__:
    __all__ = [
        name
        for name, value in vars(_core_integrators).items()
        if not name.startswith("_") and not isinstance(value, ModuleType)
    ]

for _name in __all__:
    globals()[_name] = getattr(_core_integrators, _name)


def __getattr__(name: str) -> Any:
    """Forward attribute access to the core integrators module."""
    try:
        return getattr(_core_integrators, name)
    except AttributeError as exc:  # pragma: no cover - mirrors import behaviour
        raise AttributeError(name) from exc
