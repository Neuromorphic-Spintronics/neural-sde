"""Shared helpers for loading and saving training artefacts."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Sequence, Tuple, cast

import torch
from torch.nn import Module

from neural_dynamics.config import DEVICE
from neural_dynamics.core.hyperparameters import Hyperparameters
from neural_dynamics.core.utils import (
    generate_output_directory_name,
    load_model_checkpoints,
    save_model_checkpoints,
)
from neural_dynamics.models.base import DriftNet
from neural_dynamics.models.ode import NeuralODE
from neural_dynamics.models.sde import NeuralSDE


DEFAULT_TRAINING_OUTPUT_DIR: Path = Path("examples/output")

DEFAULT_MODEL_FILENAMES: Mapping[str, str] = {
    "neural_ode": "neural_ode_model.pth",
    "neural_sde": "neural_sde_model.pth",
}

DEFAULT_METRIC_FILENAMES: Mapping[str, str] = {
    "train_losses": "train_losses.pth",
    "val_losses": "val_losses.pth",
    "generator_losses": "generator_losses.pth",
    "critic_losses": "critic_losses.pth",
    "drift_losses_sde": "drift_losses_sde.pth",
}


def _build_model_factories(
    hyperparameters: Hyperparameters,
) -> dict[str, Callable[[], Module]]:
    return {
        "neural_ode": lambda: NeuralODE(
            drift_net=DriftNet(hyperparameters.drift_network),
            hyperparameters=hyperparameters,
        ),
        "neural_sde": lambda: NeuralSDE(hyperparameters),
    }


def _load_models_internal(
    output_dir: Path | str,
    *,
    hyperparameters: Hyperparameters,
    model_filenames: Mapping[str, str],
    metric_filenames: Optional[Mapping[str, str]] = None,
    device: torch.device = DEVICE,
) -> Optional[Tuple[NeuralODE, NeuralSDE, Mapping[str, Sequence[float]]]]:
    """Load saved NeuralODE/NeuralSDE models and metrics if present."""

    loaded = load_model_checkpoints(
        output_dir,
        model_factories=_build_model_factories(hyperparameters),
        model_filenames=model_filenames,
        metric_filenames=metric_filenames,
        device=device,
    )

    if loaded is None:
        return None

    models, metrics = loaded
    return (
        cast(NeuralODE, models["neural_ode"]),
        cast(NeuralSDE, models["neural_sde"]),
        metrics,
    )


def _save_models_internal(
    output_dir: Path | str,
    *,
    neural_ode: NeuralODE,
    neural_sde: NeuralSDE,
    model_filenames: Mapping[str, str],
    metrics: Optional[Mapping[str, Sequence[float]]] = None,
    metric_filenames: Optional[Mapping[str, str]] = None,
) -> None:
    """Persist NeuralODE/NeuralSDE state dictionaries and optional metrics."""

    models = {
        "neural_ode": neural_ode,
        "neural_sde": neural_sde,
    }

    save_model_checkpoints(
        output_dir,
        models=models,
        model_filenames=model_filenames,
        metrics=metrics,
        metric_filenames=metric_filenames,
    )


def load_models(
    output_dir: Path | str,
    *,
    hyperparameters: Hyperparameters,
    model_filenames: Mapping[str, str],
    metric_filenames: Optional[Mapping[str, str]] = None,
    device: torch.device = DEVICE,
) -> Optional[Tuple[NeuralODE, NeuralSDE, Mapping[str, Sequence[float]]]]:
    """Load persisted models and metrics for a given training run."""

    return _load_models_internal(
        output_dir,
        hyperparameters=hyperparameters,
        model_filenames=model_filenames,
        metric_filenames=metric_filenames,
        device=device,
    )


def save_models(
    output_dir: Path | str,
    *,
    neural_ode: NeuralODE,
    neural_sde: NeuralSDE,
    model_filenames: Mapping[str, str],
    metrics: Optional[Mapping[str, Sequence[float]]] = None,
    metric_filenames: Optional[Mapping[str, str]] = None,
) -> None:
    """Persist models and optional training metrics in the output directory."""

    _save_models_internal(
        output_dir,
        neural_ode=neural_ode,
        neural_sde=neural_sde,
        model_filenames=model_filenames,
        metrics=metrics,
        metric_filenames=metric_filenames,
    )


@dataclass(frozen=True)
class ModellingConfig:
    """Configuration for training and loading neural dynamics models."""

    system_name: str
    hyperparameters: Hyperparameters
    enable_adversarial: bool = True
    sde_sample_count: int = 512

    @classmethod
    def with_defaults(
        cls,
        system_name: str,
        *,
        critic_window_size: Optional[int] = None,
        moment_matching_enabled: Optional[bool] = None,
        enable_adversarial: bool = True,
        sde_sample_count: int = 512,
    ) -> "ModellingConfig":
        hyperparameters = Hyperparameters.defaults()

        if critic_window_size is not None:
            critic_network = dataclasses.replace(
                hyperparameters.critic_network,
                input_size=critic_window_size * hyperparameters.state_dimension,
            )
            hyperparameters = dataclasses.replace(
                hyperparameters,
                critic_network=critic_network,
            )

        if moment_matching_enabled is not None:
            hyperparameters = dataclasses.replace(
                hyperparameters,
                moment_matching_enabled=moment_matching_enabled,
            )

        return cls(
            system_name=system_name,
            hyperparameters=hyperparameters,
            enable_adversarial=enable_adversarial,
            sde_sample_count=sde_sample_count,
        )

    def output_directory(
        self,
        *,
        base_dir: Path | str,
        prefix: Optional[str] = None,
    ) -> Path:
        directory = Path(
            generate_output_directory_name(
                Path(base_dir).as_posix(),
                self.hyperparameters,
                prefix or self.system_name,
            )
        )
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def load_models(
        self,
        output_dir: Path | str,
        *,
        model_filenames: Mapping[str, str],
        metric_filenames: Optional[Mapping[str, str]] = None,
        device: torch.device = DEVICE,
    ) -> Optional[Tuple[NeuralODE, NeuralSDE, Mapping[str, Sequence[float]]]]:
        """Load models associated with this configuration from ``output_dir``."""

        return _load_models_internal(
            output_dir,
            hyperparameters=self.hyperparameters,
            model_filenames=model_filenames,
            metric_filenames=metric_filenames,
            device=device,
        )

    def save_models(
        self,
        output_dir: Path | str,
        *,
        neural_ode: NeuralODE,
        neural_sde: NeuralSDE,
        model_filenames: Mapping[str, str],
        metrics: Optional[Mapping[str, Sequence[float]]] = None,
        metric_filenames: Optional[Mapping[str, str]] = None,
    ) -> None:
        """Save models and optional metrics tied to this configuration."""

        _save_models_internal(
            output_dir,
            neural_ode=neural_ode,
            neural_sde=neural_sde,
            model_filenames=model_filenames,
            metrics=metrics,
            metric_filenames=metric_filenames,
        )


__all__ = [
    "DEFAULT_TRAINING_OUTPUT_DIR",
    "DEFAULT_MODEL_FILENAMES",
    "DEFAULT_METRIC_FILENAMES",
    "ModellingConfig",
    "load_models",
    "save_models",
]
