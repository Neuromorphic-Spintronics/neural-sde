#!/usr/bin/env python3
"""Train and evaluate the nanorings neural ODE/SDE models from the terminal."""

from __future__ import annotations

import sys
from pathlib import Path

import argparse
import logging
import os
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor
import torch.optim as optim
from tqdm.auto import tqdm
import math

try:
    from examples.systems.parameters.nanorings import NanoringsHyperparameters
    from examples.utils.plotting import finalise_plot, setup_matplotlib_style
    from neural_dynamics.config import DEVICE
    from neural_dynamics.core.data import (
        apply_keep_fraction,
        normalise_dataset,
        replicate_trajectories,
        resample_to_target_timesteps,
    )
    from neural_dynamics.core.hyperparameters import (
        Hyperparameters,
        LearningRates,
        NetworkArchitecture,
    )
    from neural_dynamics.core.utils import (
        COLOURS,
        compare_statistics,
        compute_statistics,
        sample_sde_rollouts,
    )
    from neural_dynamics.models.ode import NeuralODE
    from neural_dynamics.models.sde import NeuralSDE, train_critic_step, train_generator_step
    from neural_dynamics.models import sde as sde_module
    from neural_dynamics.utils.training_helpers import (
        DEFAULT_METRIC_FILENAMES,
        DEFAULT_MODEL_FILENAMES,
        DEFAULT_TRAINING_OUTPUT_DIR,
        ModellingConfig,
    )
except ModuleNotFoundError:  # pragma: no cover - fallback when run as script
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from examples.systems.parameters.nanorings import NanoringsHyperparameters
    from examples.utils.plotting import finalise_plot, setup_matplotlib_style
    from neural_dynamics.config import DEVICE
    from neural_dynamics.core.data import (
        apply_keep_fraction,
        normalise_dataset,
        replicate_trajectories,
        resample_to_target_timesteps,
    )
    from neural_dynamics.core.hyperparameters import (
        Hyperparameters,
        LearningRates,
        NetworkArchitecture,
    )
    from neural_dynamics.core.utils import (
        COLOURS,
        compare_statistics,
        compute_statistics,
        sample_sde_rollouts,
    )
    from neural_dynamics.models.ode import NeuralODE
    from neural_dynamics.models.sde import NeuralSDE, train_critic_step, train_generator_step
    from neural_dynamics.models import sde as sde_module
    from neural_dynamics.utils.training_helpers import (
        DEFAULT_METRIC_FILENAMES,
        DEFAULT_MODEL_FILENAMES,
        DEFAULT_TRAINING_OUTPUT_DIR,
        ModellingConfig,
    )

try:  # pragma: no cover - best-effort optional dependency
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

from dataclasses import replace

LOGGER = logging.getLogger("nanorings")

TARGET_TIMESTEPS: int = 256


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or evaluate neural ODE/SDE models on the nanorings dataset."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=Path("examples/data/nanorings_first_set.pt"),
        help="Path to the cached nanorings dataset (.pt file).",
    )
    parser.add_argument(
        "--additional-dataset-paths",
        type=Path,
        nargs="*",
        default=(),
        help=(
            "Optional extra dataset (.pt) files to concatenate with --dataset-path. "
            "Use this to include multiple measurement signals."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device override (e.g. 'cuda', 'cpu'). Defaults to neural_dynamics.config.DEVICE.",
    )
    parser.add_argument(
        "--disable-wandb",
        action="store_true",
        help="Disable Weights & Biases logging even when training.",
    )
    parser.add_argument(
        "--disable-normalisation",
        action="store_true",
        help="Skip baseline removal and normalisation of trajectories.",
    )
    parser.add_argument(
        "--keep-fraction",
        type=float,
        default=0.1,
        help="Fraction of each trajectory to retain after transient removal (0 < f <= 1).",
    )
    parser.add_argument(
        "--replication-factor",
        type=int,
        default=1,
        help="Repeat the trajectory set this many times to augment the dataset.",
    )
    parser.add_argument(
        "--stats-checkpoint-interval",
        type=int,
        default=0,
        help=(
            "Save intermediate statistics plots every N GAN epochs during SDE "
            "training. Set to 0 to disable checkpointing."
        ),
    )
    parser.add_argument(
        "--stats-checkpoint-limit",
        type=int,
        default=5,
        help="Maximum number of intermediate statistics checkpoints to save (ignored when interval is 0).",
    )
    parser.add_argument(
        "--stats-checkpoint-samples",
        type=int,
        default=32,
        help="Number of rollouts to draw when producing statistics checkpoints.",
    )
    return parser.parse_args(argv)


def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(dataset_path: Path) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    LOGGER.info("Loading dataset from %s", dataset_path)
    cached: MutableMapping[str, Any] = torch.load(dataset_path)

    trajectories = cached["trajectories"].float()
    time_grid = cached["time_grid"].float()
    h_signal = cached["h_signal"].float()
    metadata = dict(cached.get("metadata", {}))

    LOGGER.info("Trajectories shape: %s", tuple(trajectories.shape))
    LOGGER.info("Time grid shape: %s", tuple(time_grid.shape))
    LOGGER.info("H-field shape: %s", tuple(h_signal.shape))
    LOGGER.info("Metadata: %s", metadata)

    if trajectories.shape[-1] > 1:
        h_from_trajectories = trajectories[..., 1]
        expected_h = h_signal.unsqueeze(0).expand_as(h_from_trajectories)
        if torch.allclose(h_from_trajectories, expected_h):
            LOGGER.info("Verified all trajectories share the same H-field sequence.")
        else:
            max_diff = torch.max(torch.abs(h_from_trajectories - expected_h))
            LOGGER.warning(
                "Detected multiple H-field sequences across trajectories (max |Δ|=%f).",
                float(max_diff),
            )

    return trajectories, time_grid, h_signal, metadata


def load_combined_dataset(
    dataset_path: Path,
    extra_paths: Sequence[Path],
) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    """Load one or more cached datasets and concatenate them along the run axis."""

    all_paths = [dataset_path, *extra_paths]
    if not all_paths:
        raise ValueError("At least one dataset path must be provided.")

    combined_trajectories: list[Tensor] = []
    time_grid_ref: Optional[Tensor] = None
    h_signal_ref: Optional[Tensor] = None
    combined_metadata: Dict[str, Any] = {}

    for path in all_paths:
        trajectories, time_grid, h_signal, metadata = load_dataset(path)
        if time_grid_ref is None:
            time_grid_ref = time_grid
        else:
            if time_grid.shape != time_grid_ref.shape or not torch.allclose(
                time_grid, time_grid_ref
            ):
                raise ValueError(
                    "All datasets must share the same time grid to be concatenated."
                )
        if h_signal_ref is None:
            h_signal_ref = h_signal
        else:
            if h_signal.shape != h_signal_ref.shape or not torch.allclose(
                h_signal, h_signal_ref
            ):
                raise ValueError(
                    "All datasets must share the same external H-field signal."
                )
        combined_trajectories.append(trajectories)
        if not combined_metadata:
            combined_metadata = dict(metadata)
            combined_metadata["loaded_dataset_paths"] = [path.as_posix()]
            combined_metadata["combined_signal_indices"] = [
                metadata.get("signal_index")
            ]
        else:
            combined_metadata["loaded_dataset_paths"].append(path.as_posix())
            combined_metadata.setdefault("combined_signal_indices", []).append(
                metadata.get("signal_index")
            )

    assert time_grid_ref is not None  # for type checkers
    assert h_signal_ref is not None

    stacked_trajectories = torch.cat(combined_trajectories, dim=0)
    combined_metadata["num_component_datasets"] = len(all_paths)
    combined_metadata["num_runs_combined"] = int(stacked_trajectories.shape[0])

    return stacked_trajectories, time_grid_ref, h_signal_ref, combined_metadata


def build_hyperparameters(
    config: NanoringsHyperparameters,
    trajectories: Tensor,
    timestep: float,
) -> Hyperparameters:
    state_dimension = trajectories.shape[-1]

    drift_network = NetworkArchitecture(
        input_size=state_dimension + 1,
        hidden_sizes=config.DRIFT_NET_ARCH.hidden_sizes,
        output_size=state_dimension,
    )
    diffusion_network = NetworkArchitecture(
        input_size=state_dimension + 1,
        hidden_sizes=config.DIFFUSION_NET_ARCH.hidden_sizes,
        output_size=state_dimension,
    )
    critic_network = NetworkArchitecture(
        input_size=config.CRITIC_WINDOW_SIZE * state_dimension,
        hidden_sizes=config.CRITIC_NET_ARCH.hidden_sizes,
        output_size=config.CRITIC_NET_ARCH.output_size,
    )

    return Hyperparameters(
        drift_network=drift_network,
        diffusion_network=diffusion_network,
        critic_network=critic_network,
        state_dimension=state_dimension,
        input_dimension=0,
        timestep=timestep,
        learning_rates=LearningRates(
            drift=config.DRIFT_LEARNING_RATE,
            diffusion=config.DIFFUSION_LEARNING_RATE,
            critic=config.CRITIC_LEARNING_RATE,
            generator=config.GENERATOR_LEARNING_RATE,
        ),
        number_of_epochs=config.NUMBER_OF_EPOCHS,
        number_of_gan_epochs=config.NUMBER_OF_GAN_EPOCHS,
        critic_updates=config.CRITIC_UPDATES,
        gradient_penalty_weight=config.GRADIENT_PENALTY_WEIGHT,
        batch_size=config.BATCH_SIZE,
        sde_l1_weight=config.SDE_L1_WEIGHT,
        drift_l1_weight=config.DRIFT_L1_WEIGHT,
        moment_matching_weight=config.MOMENT_MATCHING_WEIGHT,
        moment_matching_enabled=config.MOMENT_MATCHING_ENABLED,
        train_ode_with_sde=config.TRAIN_ODE_WITH_SDE,
    )


def _format_run_descriptor(metadata: Mapping[str, Any], default_runs: int) -> str:
    """Helper to format run/window descriptor for plot titles."""
    original_runs = metadata.get("num_runs_original")
    num_windows = metadata.get("num_windows")
    if original_runs is not None and num_windows is not None:
        return f"{original_runs} runs -> {num_windows} windows"
    return f"{metadata.get('num_runs', default_runs)} runs"


def plot_signal_overlays(
    trajectories: Tensor,
    time_grid: Tensor,
    h_signal: Tensor,
    output_directory: Path,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
    normalised: bool = False,
) -> None:
    """Plot stacked AMR trajectories alongside the shared H-field."""

    setup_matplotlib_style()
    fig, (ax_amr, ax_h) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    time_axis = time_grid.detach().cpu().numpy()
    amr_responses = trajectories.detach().cpu().numpy()[..., 0]
    h_series = h_signal.detach().cpu().numpy()

    amr_colour = "black" if normalised else "gray"
    amr_label = "Normalised AMR" if normalised else "AMR responses"
    h_colour = COLOURS[2] if normalised else COLOURS[1]
    h_label = "H-field (normalised)" if normalised else "H-field"
    filename = (
        "nanorings_normalised_signals.pdf"
        if normalised
        else "nanorings_raw_signals.pdf"
    )

    opacity = max(0.08, 3.0 / max(1, amr_responses.shape[0]))
    lines = ax_amr.plot(
        time_axis,
        amr_responses.T,
        color=amr_colour,
        alpha=opacity,
        linewidth=0.6,
    )
    if lines:
        lines[0].set_label(amr_label)

    ax_amr.set_ylabel("AMR response" if not normalised else "AMR (normalised)")
    ax_amr.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax_amr.grid(False)

    ax_h.plot(time_axis, h_series, color=h_colour, linewidth=1.0, label=h_label)
    ax_h.set_xlabel("Time [s]")
    ax_h.set_ylabel("Exogenous H" if not normalised else "H (normalised)")
    ax_h.grid(False)
    ax_h.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")

    if metadata is not None and not normalised:
        run_descriptor = _format_run_descriptor(metadata, amr_responses.shape[0])
        ax_amr.set_title(
            "Loaded nanorings subset: "
            f"{run_descriptor}, "
            f"{metadata.get('num_steps_retained', amr_responses.shape[1])} steps"
        )

    fig.tight_layout()
    finalise_plot(fig, filename, str(output_directory), os)


def plot_ode_preview(
    trajectories: Tensor,
    time_grid: Tensor,
    ode_prediction: Tensor,
    output_directory: Path,
) -> None:
    setup_matplotlib_style()
    time_axis = time_grid.detach().cpu().numpy()

    fig_preview, ax_preview = plt.subplots(figsize=(8, 4))
    ax_preview.plot(
        time_axis,
        trajectories[0, :, 0].cpu(),
        color="gray",
        alpha=0.6,
        linewidth=0.8,
        label="Training sample",
    )
    ax_preview.plot(
        time_axis,
        ode_prediction[:, 0].cpu(),
        color=COLOURS[2],
        linewidth=1.5,
        label="Neural ODE",
    )
    ax_preview.set_title("Neural ODE preview after training")
    ax_preview.set_xlabel("Time [s]")
    ax_preview.set_ylabel("AMR response")
    ax_preview.grid(False)
    ax_preview.legend(loc="upper right")
    fig_preview.tight_layout()
    finalise_plot(fig_preview, "nanorings_neural_ode_preview.pdf", str(output_directory), os)


class StatisticsCheckpointManager:
    """Manage intermediate statistics snapshots during SDE training."""

    def __init__(
        self,
        *,
        interval: int,
        output_directory: Path,
        device: torch.device,
        sample_count: int,
        max_snapshots: Optional[int] = None,
    ) -> None:
        if interval <= 0:
            raise ValueError("interval must be a positive integer when checkpointing.")
        if sample_count <= 0:
            raise ValueError("sample_count must be a positive integer.")

        self.interval = interval
        self.output_directory = output_directory
        self.device = device
        self.sample_count = sample_count
        self.max_snapshots = max_snapshots

        self.snapshots_taken = 0
        self.training_trajectories: Optional[Tensor] = None
        self.time_grid_device: Optional[Tensor] = None
        self.time_grid_cpu: Optional[Tensor] = None
        self._limit_logged = False

    def clear(self) -> None:
        """Reset stored training context."""

        self.training_trajectories = None
        self.time_grid_device = None
        self.time_grid_cpu = None
        self._limit_logged = False

    def update_training_context(self, trajectories: Tensor, time_grid: Tensor) -> None:
        """Update the training trajectories and time grid used for checkpoints."""

        self.training_trajectories = trajectories.detach()
        self.time_grid_device = time_grid.detach()
        self.time_grid_cpu = time_grid.detach().cpu()

    def maybe_checkpoint(
        self,
        *,
        epoch: int,
        model: NeuralSDE,
        generator_loss: float,
        critic_loss: float,
    ) -> None:
        """Save a statistics snapshot if the interval condition holds."""

        if (
            self.training_trajectories is None
            or self.time_grid_device is None
            or self.time_grid_cpu is None
        ):
            return

        if (epoch + 1) % self.interval != 0:
            return

        if self.max_snapshots is not None and self.snapshots_taken >= self.max_snapshots:
            if not self._limit_logged:
                LOGGER.info(
                    "Statistics checkpoint limit (%d) reached; skipping further snapshots.",
                    self.max_snapshots,
                )
                self._limit_logged = True
            return

        num_available = self.training_trajectories.shape[0]
        if num_available == 0:
            return

        samples = min(self.sample_count, num_available)

        previous_mode = model.training
        model.eval()
        try:
            rollout = sample_sde_rollouts(
                model=model,
                trajectories=self.training_trajectories,
                time_grid=self.time_grid_device,
                num_samples=samples,
                device=self.device,
            )
        finally:
            super(NeuralSDE, model).train(previous_mode)

        sde_stats, training_stats = compute_and_compare_statistics(rollout)
        suffix = f"epoch-{epoch + 1:04d}"
        plot_statistics(
            sde_stats,
            training_stats,
            self.time_grid_cpu,
            self.output_directory,
            suffix=suffix,
        )
        LOGGER.info(
            "Checkpoint %s | generator_loss=%.6f | critic_loss=%.6f",
            suffix,
            generator_loss,
            critic_loss,
        )
        self.snapshots_taken += 1


_STATS_CHECKPOINT_MANAGER: Optional[StatisticsCheckpointManager] = None
_ORIGINAL_FIT_NEURAL_SDE_GAN: Optional[Callable[..., Tuple[List[float], List[float], List[float]]]] = None


def _fit_neural_sde_gan_with_checkpoints(
    drift_network,  # type: ignore
    diffusion_network,  # type: ignore
    critic_network,  # type: ignore
    time_grid: Tensor,
    stochastic_trajectories: Tensor,
    *,
    hyperparameters: Hyperparameters,
    random_seed: int = 0,
    wandb_run: Optional[Any] = None,
    actual_ode_epochs: Optional[int] = None,
) -> Tuple[List[float], List[float], List[float]]:
    """Wrapped GAN trainer that emits statistics checkpoints."""

    manager = _STATS_CHECKPOINT_MANAGER

    def _ensure_finite(tensor: Tensor, label: str, clamp: float = 10.0) -> Tensor:
        mask = ~torch.isfinite(tensor)
        if mask.any():
            LOGGER.warning(
                "Detected %d non-finite values in %s; replacing with zeros.",
                int(mask.sum().item()),
                label,
            )
            tensor = torch.nan_to_num(tensor, nan=0.0, posinf=clamp, neginf=-clamp)
        if clamp is not None:
            tensor = torch.clamp(tensor, -clamp, clamp)
        return tensor

    torch.manual_seed(random_seed)

    if hyperparameters.train_ode_with_sde:
        drift_network.train()
        generator_params = list(drift_network.parameters()) + list(diffusion_network.parameters())
    else:
        drift_network.eval()
        generator_params = diffusion_network.parameters()

    generator_optimiser = optim.Adam(
        generator_params, lr=hyperparameters.learning_rates.generator
    )
    critic_optimiser = optim.Adam(
        critic_network.parameters(), lr=hyperparameters.learning_rates.critic
    )

    generator_losses: List[float] = []
    critic_losses: List[float] = []
    drift_losses: List[float] = []

    stochastic_trajectories = stochastic_trajectories.to(DEVICE)
    time_grid = time_grid.to(DEVICE)

    critic_window = critic_network.trajectory_length
    full_trajectory_length = stochastic_trajectories.shape[1]

    if full_trajectory_length < critic_window:
        raise ValueError(
            f"Trajectory length {full_trajectory_length} is shorter than critic window {critic_window}"
        )

    max_window_start = full_trajectory_length - critic_window
    num_batches = math.ceil(
        stochastic_trajectories.shape[0] / hyperparameters.batch_size
    )

    sde = NeuralSDE(drift_network, diffusion_network, hyperparameters).to(DEVICE)
    num_epochs = hyperparameters.number_of_gan_epochs

    if manager is not None:
        manager.update_training_context(
            stochastic_trajectories[:, :, : sde.state_dimension], time_grid
        )

    with tqdm(range(num_epochs), desc="Training Neural SDE GAN") as pbar:
        for epoch in pbar:
            epoch_critic_losses: List[float] = []
            epoch_generator_losses: List[float] = []
            epoch_drift_losses: List[float] = []

            for batch_idx in range(num_batches):
                start_idx = batch_idx * hyperparameters.batch_size
                end_idx = min(
                    start_idx + hyperparameters.batch_size,
                    stochastic_trajectories.shape[0],
                )

                if start_idx >= end_idx:
                    continue

                window_start = torch.randint(0, max_window_start + 1, (1,)).item()
                window_end = window_start + critic_window

                real_trajectories = stochastic_trajectories[start_idx:end_idx, window_start:window_end, :].contiguous()
                window_time_grid = time_grid[window_start:window_end]
                initial_states = real_trajectories[:, 0, :]

                fake_trajectories = sde(initial_states, window_time_grid)
                real_trajectories = _ensure_finite(real_trajectories, "real trajectories")
                fake_trajectories = _ensure_finite(fake_trajectories, "fake trajectories")
                fake_for_critic = fake_trajectories.detach()

                for _ in range(hyperparameters.critic_updates):
                    critic_loss = train_critic_step(
                        critic_optimiser,
                        critic_network,
                        real_trajectories,
                        fake_for_critic,
                        hyperparameters.gradient_penalty_weight,
                    )
                    epoch_critic_losses.append(float(critic_loss))

                drift_l1_weight = getattr(hyperparameters, "drift_l1_weight", 0.0)
                drift_only_loss_tensor = None

                generator_loss, loss_components = train_generator_step(
                    generator_optimiser=generator_optimiser,
                    critic_network=critic_network,
                    fake_trajectories=fake_trajectories,
                    real_trajectories=real_trajectories,
                    sde_l1_weight=hyperparameters.sde_l1_weight,
                    moment_matching_weight=getattr(hyperparameters, "moment_matching_weight", 0.0),
                    moment_matching_enabled=getattr(hyperparameters, "moment_matching_enabled", False),
                    diffusion_network=diffusion_network,
                    drift_only_loss=drift_only_loss_tensor,
                    drift_l1_weight=drift_l1_weight,
                )
                epoch_generator_losses.append(float(generator_loss))

            generator_losses.append(float(np.mean(epoch_generator_losses)) if epoch_generator_losses else 0.0)
            critic_losses.append(float(np.mean(epoch_critic_losses)) if epoch_critic_losses else 0.0)
            drift_losses.append(float(np.mean(epoch_drift_losses)) if epoch_drift_losses else 0.0)

            if manager is not None:
                manager.maybe_checkpoint(
                    epoch=epoch,
                    model=sde,
                    generator_loss=generator_losses[-1],
                    critic_loss=critic_losses[-1],
                )

    return generator_losses, critic_losses, drift_losses


def _install_sde_checkpointing(manager: Optional[StatisticsCheckpointManager]) -> None:
    """Install the GAN training wrapper with the provided manager."""

    global _ORIGINAL_FIT_NEURAL_SDE_GAN, _STATS_CHECKPOINT_MANAGER

    if _ORIGINAL_FIT_NEURAL_SDE_GAN is None:
        _ORIGINAL_FIT_NEURAL_SDE_GAN = sde_module.fit_neural_sde_gan  # type: ignore
        sde_module.fit_neural_sde_gan = _fit_neural_sde_gan_with_checkpoints  # type: ignore

    _STATS_CHECKPOINT_MANAGER = manager


def _restore_sde_checkpointing() -> None:
    """Restore the original GAN trainer and clear global state."""

    global _ORIGINAL_FIT_NEURAL_SDE_GAN, _STATS_CHECKPOINT_MANAGER
    if _ORIGINAL_FIT_NEURAL_SDE_GAN is not None:
        sde_module.fit_neural_sde_gan = _ORIGINAL_FIT_NEURAL_SDE_GAN  # type: ignore
        _ORIGINAL_FIT_NEURAL_SDE_GAN = None
    _STATS_CHECKPOINT_MANAGER = None


def train_or_load_models(
    trajectories: Tensor,
    time_grid: Tensor,
    device: torch.device,
    hyperparameters: Hyperparameters,
    modelling_config: ModellingConfig,
    config: NanoringsHyperparameters,
    output_directory: Path,
    enable_wandb: bool,
    validation_split: float,
    early_stopping_patience: int,
    stats_checkpoint_manager: Optional[StatisticsCheckpointManager] = None,
    sde_hyperparameters: Optional[Hyperparameters] = None,
) -> Tuple[
    NeuralODE,
    NeuralSDE,
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float],
    Sequence[float],
]:
    trajectories_device = trajectories.to(device)
    time_grid_device = time_grid.to(device)

    loaded = modelling_config.load_models(
        output_directory,
        model_filenames=DEFAULT_MODEL_FILENAMES,
        metric_filenames=DEFAULT_METRIC_FILENAMES,
        device=device,
    )

    train_losses: Sequence[float]
    val_losses: Sequence[float]
    generator_losses: Sequence[float]
    critic_losses: Sequence[float]

    wandb_run = None

    if loaded is not None:
        neural_ode, neural_sde, loaded_metrics = loaded
        train_losses = list(loaded_metrics.get("train_losses", []))
        val_losses = list(loaded_metrics.get("val_losses", []))
        generator_losses = list(loaded_metrics.get("generator_losses", []))
        critic_losses = list(loaded_metrics.get("critic_losses", []))
        drift_losses_sde = list(loaded_metrics.get("drift_losses_sde", []))  # Load drift losses from SDE phase
        LOGGER.info("Loaded existing checkpoints from %s.", output_directory)
        if stats_checkpoint_manager is not None:
            stats_checkpoint_manager.clear()
            LOGGER.info("Statistics checkpointing skipped (existing checkpoints loaded).")
    else:
        LOGGER.info("Training neural ODE...")
        if wandb is not None and enable_wandb:
            # Use the output directory name as the W&B run name
            run_name = output_directory.name
            wandb_run = wandb.init(
                entity="neuromorphic-spintronics",
                project="neural-sde",
                name=run_name,
                reinit=True,
            )
            wandb_run.config.update({
                "output_directory": output_directory.as_posix(),
                "number_of_epochs": config.NUMBER_OF_EPOCHS,
                "number_of_gan_epochs": config.NUMBER_OF_GAN_EPOCHS,
                "moment_matching_weight": config.MOMENT_MATCHING_WEIGHT,
                "moment_matching_enabled": config.MOMENT_MATCHING_ENABLED,
                "sde_l1_weight": config.SDE_L1_WEIGHT,
                "train_ode_with_sde": config.TRAIN_ODE_WITH_SDE,
                "batch_size": config.BATCH_SIZE,
                "gan_batch_size": config.GAN_BATCH_SIZE,
                "critic_window_size": config.CRITIC_WINDOW_SIZE,
            })

        neural_ode = NeuralODE.train(
            hyperparameters=hyperparameters,
            trajectories=trajectories_device,
            time_grid=time_grid_device,
            device=device,
            validation_split=validation_split,
            early_stopping_patience=early_stopping_patience,
            wandb_run=wandb_run,
        )
        train_losses = list(neural_ode.training_losses)
        val_losses = list(neural_ode.validation_losses)

        # Plot ODE preview immediately after training
        LOGGER.info("Training neural SDE...")
        # Use separate batch size for GAN training if provided
        sde_hparams = sde_hyperparameters if sde_hyperparameters is not None else hyperparameters
        LOGGER.info("Using batch size %d for SDE training", sde_hparams.batch_size)
        try:
            _install_sde_checkpointing(stats_checkpoint_manager)
            neural_sde = NeuralSDE.train(
                hyperparameters=sde_hparams,
                neural_ode=neural_ode,
                trajectories=trajectories_device,
                time_grid=time_grid_device,
                device=device,
                enable_adversarial=modelling_config.enable_adversarial,
                random_seed=0,
                wandb_run=wandb_run,
            )
        finally:
            _restore_sde_checkpointing()
            if stats_checkpoint_manager is not None:
                stats_checkpoint_manager.clear()

        generator_losses = list(getattr(neural_sde, "generator_losses", []))
        critic_losses = list(getattr(neural_sde, "critic_losses", []))
        drift_losses_sde = list(getattr(neural_sde, "drift_losses", []))  # Drift loss during SDE training

        training_metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "generator_losses": generator_losses,
            "critic_losses": critic_losses,
            "drift_losses_sde": drift_losses_sde,  # Save drift losses from SDE phase
        }

        modelling_config.save_models(
            output_directory,
            neural_ode=neural_ode,
            neural_sde=neural_sde,
            model_filenames=DEFAULT_MODEL_FILENAMES,
            metrics=training_metrics,
            metric_filenames=DEFAULT_METRIC_FILENAMES,
        )

        if wandb_run is not None:
            wandb_run.finish()

    neural_ode = neural_ode.to(device)
    neural_sde = neural_sde.to(device)
    super(NeuralSDE, neural_sde).train(False)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return neural_ode, neural_sde, train_losses, val_losses, generator_losses, critic_losses, drift_losses_sde


def predict_with_neural_ode(
    neural_ode: NeuralODE,
    trajectories: Tensor,
    time_grid: Tensor,
    device: torch.device,
) -> Tensor:
    reference_state = trajectories[0:1, 0, :]
    with torch.no_grad():
        prediction = neural_ode(reference_state.to(device), time_grid.to(device))[0]
    return prediction.detach().cpu()


def generate_rollouts(
    neural_sde: NeuralSDE,
    trajectories: Tensor,
    time_grid: Tensor,
    sample_count: int,
    device: torch.device,
):
    return sample_sde_rollouts(
        model=neural_sde,
        trajectories=trajectories.to(device),
        time_grid=time_grid.to(device),
        num_samples=sample_count,
        device=device,
    )


def compute_and_compare_statistics(
    rollout,
) -> Tuple[Any, Any]:
    training_stats = compute_statistics(rollout.training_subset)
    sde_stats = compute_statistics(rollout.rollout_tensor)
    differences = compare_statistics(training_stats, sde_stats)
    LOGGER.info(
        "Statistic differences | mean: %.6f | variance: %.6f",
        differences["mean_abs_difference"],
        differences["variance_abs_difference"],
    )
    return sde_stats, training_stats


def plot_statistics(
    sde_stats,
    training_stats,
    time_grid: Tensor,
    output_directory: Path,
    *,
    suffix: Optional[str] = None,
) -> None:
    if sde_stats.mean.numel() == 0:
        LOGGER.warning("Skipping statistics plot; no SDE rollouts were generated.")
        return

    setup_matplotlib_style()
    stats_time_axis = time_grid.detach().cpu().numpy().ravel()
    training_mean = training_stats.mean.detach().cpu().numpy()
    training_var = training_stats.variance.detach().cpu().numpy()
    sde_mean = sde_stats.mean.detach().cpu().numpy()
    sde_var = sde_stats.variance.detach().cpu().numpy()

    train_envelope = np.sqrt(np.maximum(training_var[:, 0], 0.0))
    sde_envelope = np.sqrt(np.maximum(sde_var[:, 0], 0.0))

    fig_stats, ax_stats = plt.subplots(1, 1, figsize=(9, 4))
    ax_stats.fill_between(
        stats_time_axis,
        training_mean[:, 0] - train_envelope,
        training_mean[:, 0] + train_envelope,
        alpha=0.25,
        color="black",
        label=r"Training mean $\pm \sigma$",
    )
    ax_stats.plot(
        stats_time_axis,
        training_mean[:, 0],
        color="black",
        linewidth=1.8,
        label="Training mean",
    )
    ax_stats.fill_between(
        stats_time_axis,
        sde_mean[:, 0] - sde_envelope,
        sde_mean[:, 0] + sde_envelope,
        alpha=0.25,
        color=COLOURS[3],
        label=r"Neural SDE mean $\pm \sigma$",
    )
    ax_stats.plot(
        stats_time_axis,
        sde_mean[:, 0],
        color=COLOURS[3],
        linewidth=1.8,
        label="Neural SDE mean",
    )

    ax_stats.set_xlabel("Time [s]")
    ax_stats.set_ylabel("AMR response")
    ax_stats.grid(False)
    ax_stats.legend(bbox_to_anchor=(0.5, 1.02), loc='lower center', ncol=2, frameon=False)

    for spine in ax_stats.spines.values():
        spine.set_linewidth(1.0)

    fig_stats.tight_layout()
    filename = "nanorings_statistics"
    if suffix:
        filename = f"{filename}_{suffix}"
    filename = f"{filename}.pdf"
    finalise_plot(fig_stats, filename, str(output_directory), os)


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    generator_losses: Sequence[float],
    critic_losses: Sequence[float],
    modelling_config: ModellingConfig,
    output_directory: Path,
    drift_losses_sde: Optional[Sequence[float]] = None,
) -> None:
    setup_matplotlib_style()
    fig_metrics, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    # Use ACTUAL number of ODE epochs trained (accounting for early stopping)
    actual_ode_epochs = len(train_losses)
    actual_gan_epochs = len(generator_losses)
    actual_total_epochs = actual_ode_epochs + actual_gan_epochs

    # Plot ODE training phase
    ode_epoch_range = np.arange(actual_ode_epochs)
    ax1.plot(ode_epoch_range, train_losses, label="Training", color=COLOURS[0], linewidth=1.5)
    if len(val_losses) == len(train_losses):
        ax1.plot(
            ode_epoch_range,
            val_losses,
            label="Validation",
            color=COLOURS[1],
            linewidth=1.5,
        )

    # Plot continuing drift loss during SDE training (if available and training jointly)
    if drift_losses_sde and len(drift_losses_sde) > 0:
        # Check if any non-zero losses (indicates joint training was enabled)
        if any(loss > 0 for loss in drift_losses_sde):
            # Start SDE drift plot immediately after ODE training ends
            sde_drift_epochs = np.arange(actual_ode_epochs, actual_ode_epochs + len(drift_losses_sde))
            ax1.plot(
                sde_drift_epochs,
                drift_losses_sde,
                label="Drift-only (SDE phase)",
                color=COLOURS[0],
                linewidth=1.5,
                linestyle="--",
                alpha=0.7,
            )

    # Mark the transition from ODE to SDE training
    ax1.axvline(
        x=max(actual_ode_epochs - 1, 0),
        color="black",
        linestyle="--",
        alpha=0.7,
        linewidth=1.0,
    )
    ax1.text(
        max(actual_ode_epochs - 1, 0),
        ax1.get_ylim()[1] * 0.9 if train_losses else 0.0,
        "SDE Training Start",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
    )
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.set_ylabel("SmoothL1 Loss")
    ax1.grid(False)

    # Plot generator loss starting from end of ODE training
    sde_epochs = np.arange(actual_ode_epochs, actual_ode_epochs + len(generator_losses))
    if len(sde_epochs) == len(generator_losses):
        ax2.plot(sde_epochs, generator_losses, color=COLOURS[2], linewidth=1.5)
    ax2.set_ylabel("Generator Loss")
    ax2.grid(False)

    # Plot critic loss
    if len(sde_epochs) == len(critic_losses):
        ax3.plot(sde_epochs, critic_losses, color=COLOURS[3], linewidth=1.5)
    ax3.set_ylabel("Critic Loss")
    ax3.set_xlabel("Epoch")
    ax3.grid(False)

    # Set x-axis limits to show ACTUAL trained epochs (not configured max epochs)
    if actual_total_epochs > 1:
        for ax in [ax1, ax2, ax3]:
            ax.set_xlim(0, actual_total_epochs - 1)

    fig_metrics.tight_layout()
    finalise_plot(fig_metrics, "training_losses.pdf", str(output_directory), os)


def plot_model_comparison(
    trajectories: Tensor,
    time_grid: Tensor,
    h_signal: Tensor,
    ode_prediction: Tensor,
    rollout,
    metadata: Mapping[str, Any],
    output_directory: Path,
) -> None:
    setup_matplotlib_style()
    fig_comparison, (ax_amr, ax_h) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    time_axis = time_grid.detach().cpu().numpy().ravel()
    amr_training = trajectories.detach().cpu().numpy()[..., 0]
    h_series = h_signal.detach().cpu().numpy()
    ode_series = ode_prediction.detach().cpu().numpy()[:, 0]

    n_show = min(50, amr_training.shape[0])
    if n_show:
        train_lines = ax_amr.plot(
            time_axis,
            amr_training[:n_show].T,
            color="gray",
            alpha=0.2,
            linewidth=0.6,
        )
        if train_lines:
            train_lines[0].set_label("Training")

    ax_amr.plot(
        time_axis,
        ode_series,
        color=COLOURS[2],
        linewidth=1.2,
        label="Neural ODE",
    )

    if rollout.rollout_tensor.numel() > 0:
        sde_series = rollout.rollout_tensor[0].detach().cpu().numpy()[:, 0]
        ax_amr.plot(
            time_axis,
            sde_series,
            color=COLOURS[3],
            linewidth=1.0,
            alpha=0.9,
            label="Neural SDE",
        )

    ax_amr.set_ylabel("AMR response")
    ax_amr.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax_amr.grid(False)

    ax_h.plot(time_axis, h_series, color=COLOURS[1], linewidth=1.0, label="H-field")
    ax_h.set_xlabel("Time [s]")
    ax_h.set_ylabel("Exogenous H")
    ax_h.grid(False)
    ax_h.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")

    run_descriptor = _format_run_descriptor(metadata, amr_training.shape[0])
    ax_amr.set_title(
        f"Nanorings subset: {run_descriptor}, "
        f"{metadata.get('num_steps_retained', amr_training.shape[1])} steps"
    )

    fig_comparison.tight_layout()
    finalise_plot(fig_comparison, "nanorings_signals.pdf", str(output_directory), os)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    _configure_logging()

    device = torch.device(args.device) if args.device is not None else DEVICE
    LOGGER.info("Using device: %s", device)

    trajectories, time_grid, h_signal, metadata = load_combined_dataset(
        args.dataset_path,
        args.additional_dataset_paths,
    )
    if args.additional_dataset_paths:
        LOGGER.info(
            "Loaded %d dataset files -> %d runs.",
            1 + len(args.additional_dataset_paths),
            trajectories.shape[0],
        )
    original_steps = trajectories.shape[1]
    original_duration = (
        float(time_grid[-1] - time_grid[0]) if time_grid.numel() > 1 else 0.0
    )
    (
        trajectories,
        time_grid,
        kept_steps,
        windows_per_run,
    ) = apply_keep_fraction(
        trajectories,
        time_grid,
        args.keep_fraction,
    )
    h_signal = trajectories[0, :, 1].clone()
    total_windows = trajectories.shape[0]
    total_runs = len(windows_per_run)
    total_samples = total_windows * kept_steps
    metadata["num_steps_retained"] = kept_steps
    metadata["keep_fraction_used"] = args.keep_fraction
    metadata["total_steps_original"] = original_steps
    metadata["num_windows_before_replication"] = total_windows
    metadata["num_windows"] = total_windows
    metadata["windows_per_run"] = list(windows_per_run)
    metadata["num_runs_original"] = total_runs
    metadata["total_samples"] = total_samples
    LOGGER.info(
        "Keeping %.2f%% of each trajectory per window (%d/%d steps).",
        args.keep_fraction * 100.0,
        kept_steps,
        original_steps,
    )
    if total_runs > 0:
        avg_windows = total_windows / float(total_runs)
    else:
        avg_windows = 0.0
    LOGGER.info(
        "Constructed %d training windows from %d runs (%.2f windows/run).",
        total_windows,
        total_runs,
        avg_windows,
    )
    LOGGER.info(
        "Total samples available for training: %d (window length %.4fs of %.4fs original).",
        total_samples,
        float(time_grid[-1] - time_grid[0]) if time_grid.numel() > 1 else 0.0,
        original_duration,
    )
    # Provide quick glimpse of first few runs
    if total_runs and total_runs <= 10:
        windows_repr = ", ".join(str(count) for count in windows_per_run)
        LOGGER.info("Windows per run: [%s]", windows_repr)
    elif total_runs:
        first_five = ", ".join(str(count) for count in windows_per_run[:5])
        LOGGER.info(
            "Windows per run (first 5 of %d): [%s] ...", total_runs, first_five
        )

    if trajectories.shape[1] != TARGET_TIMESTEPS:
        trajectories, time_grid = resample_to_target_timesteps(
            trajectories,
            time_grid,
            TARGET_TIMESTEPS,
        )
        kept_steps = trajectories.shape[1]
        h_signal = trajectories[0, :, 1].clone()
        metadata["num_steps_retained"] = kept_steps
        metadata["resampled_to_timesteps"] = TARGET_TIMESTEPS
        LOGGER.info("Resampled trajectories to %d timesteps for training.", kept_steps)
        total_samples = trajectories.shape[0] * kept_steps
        metadata["total_samples"] = total_samples
        LOGGER.info(
            "Total samples after resampling: %d (per-window steps %d).",
            total_samples,
            kept_steps,
        )

    trajectories = replicate_trajectories(trajectories, args.replication_factor)
    if args.replication_factor > 1:
        metadata["replication_factor"] = args.replication_factor
        LOGGER.info(
            "Replicated trajectories %d× -> total windows: %d (%d samples).",
            args.replication_factor,
            trajectories.shape[0],
            trajectories.shape[0] * trajectories.shape[1],
        )
    metadata["num_runs"] = trajectories.shape[0]
    metadata["num_windows"] = trajectories.shape[0]
    metadata["total_samples"] = trajectories.shape[0] * trajectories.shape[1]

    trajectories, h_signal, normalisation_stats = normalise_dataset(
        trajectories,
        h_signal,
        enabled=not args.disable_normalisation,
    )
    if normalisation_stats is not None:
        LOGGER.info(
            "Applied normalisation | AMR mean %.6f std %.6f | H mean %.6f std %.6f",
            normalisation_stats["amr_mean"],
            normalisation_stats["amr_std"],
            normalisation_stats["h_mean"],
            normalisation_stats["h_std"],
        )

    config = NanoringsHyperparameters()
    hyperparameters = build_hyperparameters(
        config, trajectories, timestep=float(time_grid[1] - time_grid[0])
    )
    # Create separate hyperparameters for SDE with different batch size
    sde_config = replace(config, BATCH_SIZE=config.GAN_BATCH_SIZE)
    sde_hyperparameters = build_hyperparameters(
        sde_config, trajectories, timestep=float(time_grid[1] - time_grid[0])
    )
    modelling_config = ModellingConfig(
        system_name="nanorings",
        hyperparameters=hyperparameters,
        enable_adversarial=True,
        sde_sample_count=config.NUMBER_OF_SDE_ROLLOUT_SAMPLES,
    )

    output_directory = modelling_config.output_directory(
        base_dir=DEFAULT_TRAINING_OUTPUT_DIR
    )
    LOGGER.info("Outputs will be saved to %s", output_directory)

    stats_checkpoint_manager: Optional[StatisticsCheckpointManager] = None
    if args.stats_checkpoint_interval > 0:
        max_snapshots = (
            args.stats_checkpoint_limit if args.stats_checkpoint_limit > 0 else None
        )
        stats_checkpoint_manager = StatisticsCheckpointManager(
            interval=args.stats_checkpoint_interval,
            output_directory=output_directory,
            device=device,
            sample_count=args.stats_checkpoint_samples,
            max_snapshots=max_snapshots,
        )
        LOGGER.info(
            "Intermediate statistics checkpointing enabled every %d epoch(s) (limit=%s, samples=%d).",
            args.stats_checkpoint_interval,
            "∞" if max_snapshots is None else str(max_snapshots),
            args.stats_checkpoint_samples,
        )

    plot_is_normalised = normalisation_stats is not None
    plot_signal_overlays(
        trajectories,
        time_grid,
        h_signal,
        output_directory,
        normalised=plot_is_normalised,
    )
    plot_filename = (
        "nanorings_normalised_signals.pdf"
        if plot_is_normalised
        else "nanorings_raw_signals.pdf"
    )
    LOGGER.info(
        "Saved training signal plot to %s",
        (output_directory / plot_filename).as_posix(),
    )

    neural_ode, neural_sde, train_losses, val_losses, generator_losses, critic_losses, drift_losses_sde = (
        train_or_load_models(
            trajectories=trajectories,
            time_grid=time_grid,
            device=device,
            hyperparameters=hyperparameters,
            modelling_config=modelling_config,
            config=config,
            output_directory=output_directory,
            enable_wandb=not args.disable_wandb,
            validation_split=config.VALIDATION_SPLIT,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            stats_checkpoint_manager=stats_checkpoint_manager,
            sde_hyperparameters=sde_hyperparameters,
        )
    )

    ode_prediction = predict_with_neural_ode(neural_ode, trajectories, time_grid, device)
    plot_ode_preview(trajectories, time_grid, ode_prediction, output_directory)
    LOGGER.info(
        "Saved Neural ODE preview plot to %s",
        (output_directory / "nanorings_neural_ode_preview.pdf").as_posix(),
    )

    rollout = generate_rollouts(
        neural_sde,
        trajectories,
        time_grid,
        sample_count=modelling_config.sde_sample_count,
        device=device,
    )

    sde_stats, training_stats = compute_and_compare_statistics(rollout)

    plot_statistics(sde_stats, training_stats, time_grid, output_directory)
    plot_training_curves(
        train_losses, val_losses, generator_losses, critic_losses, modelling_config, output_directory,
        drift_losses_sde=drift_losses_sde
    )
    plot_model_comparison(
        trajectories,
        time_grid,
        h_signal,
        ode_prediction,
        rollout,
        metadata,
        output_directory,
    )

    LOGGER.info("Nanorings experiment complete.")


if __name__ == "__main__":
    main()
