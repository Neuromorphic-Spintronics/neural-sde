#!/usr/bin/env python3
"""Train and evaluate Neural ODE/SDE models on the nanorings dataset.

This script implements a complete training pipeline for learning dynamics when an exogenous H-field is applied to a nanoring array. The pipeline:

1. Loads preprocessed nanorings data (AMR responses + H-field)
2. Applies windowing, resampling, and standardisation 
3. Trains a Neural ODE to capture deterministic dynamics
4. Trains a Neural SDE (via WGAN-GP) to model the stochastic dynamics
5. Generates predictions, computes statistics, and creates visualisations

Usage:
    uv run examples/nanorings.py

Configuration:
    Edit the `NanoringsHyperparameters` dataclass in `examples/systems/parameters/nanorings.py`.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from tqdm.auto import tqdm

try:  # pragma: no cover - best-effort optional dependency
    import wandb
except ImportError:  # pragma: no cover
    wandb = None  # type: ignore[assignment]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from examples.systems.parameters.nanorings import NanoringsHyperparameters  # noqa: E402
from examples.utils.plotting import finalise_plot, setup_matplotlib_style  # noqa: E402
from neural_dynamics.config import DEVICE  # noqa: E402
from neural_dynamics.core.hyperparameters import (  # noqa: E402
    Hyperparameters,
    LearningRates,
    NetworkArchitecture,
)
from neural_dynamics.core.utils import (  # noqa: E402
    COLOURS,
    compare_statistics,
    compute_statistics,
    sample_sde_rollouts,
)
from neural_dynamics.models.ode import NeuralODE  # noqa: E402
from neural_dynamics.models.sde import (  # noqa: E402
    NeuralSDE,
)
from neural_dynamics.utils.training_helpers import (  # noqa: E402
    DEFAULT_METRIC_FILENAMES,
    DEFAULT_MODEL_FILENAMES,
    DEFAULT_TRAINING_OUTPUT_DIR,
    ModellingConfig,
)

LOGGER = logging.getLogger("nanorings")


# ============================================================================
# Utility Functions
# ============================================================================

def _configure_logging() -> None:
    """Set up logging format for experiment output."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def load_dataset(dataset_path: Path) -> Tuple[Tensor, Tensor, Tensor, Dict[str, Any]]:
    """Load the nanorings dataset from a .pt file.
    
    Args:
        dataset_path: Path to the cached dataset file.
        
    Returns:
        Tuple of (trajectories, time_grid, h_signal, metadata) where:
        - trajectories: [num_runs, num_steps, state_dim] AMR responses
        - time_grid: [num_steps] time axis
        - h_signal: [num_steps] exogenous H-field forcing
        - metadata: Dataset information (empty dict if not present in .pt file)
    """
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

    return trajectories, time_grid, h_signal, metadata


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


def plot_raw_signals(
    trajectories: Tensor,
    time_grid: Tensor,
    h_signal: Tensor,
    metadata: Mapping[str, Any],
    output_directory: Path,
) -> None:
    setup_matplotlib_style()
    fig_raw, (ax_raw_amr, ax_raw_h) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    time_axis = time_grid.detach().cpu().numpy()
    amr_responses = trajectories.detach().cpu().numpy()[..., 0]
    h_series = h_signal.detach().cpu().numpy()

    opacity = max(0.08, 3.0 / max(1, amr_responses.shape[0]))
    for idx in tqdm(
        range(amr_responses.shape[0]),
        desc="Plotting raw AMR responses",
        leave=False,
    ):
        ax_raw_amr.plot(
            time_axis,
            amr_responses[idx],
            color="gray",
            alpha=opacity,
            linewidth=0.6,
            label="AMR responses" if idx == 0 else "",
        )

    ax_raw_amr.set_ylabel("AMR response")
    ax_raw_amr.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax_raw_amr.grid(False)

    ax_raw_h.plot(time_axis, h_series, color=COLOURS[1], linewidth=1.0, label="H-field")
    ax_raw_h.set_xlabel("Time [s]")
    ax_raw_h.set_ylabel("Exogenous H")
    ax_raw_h.grid(False)
    ax_raw_h.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")

    run_descriptor = _format_run_descriptor(metadata, amr_responses.shape[0])
    ax_raw_amr.set_title(
        "Loaded nanorings subset: "
        f"{run_descriptor}, "
        f"{metadata.get('num_steps_retained', amr_responses.shape[1])} steps"
    )

    fig_raw.tight_layout()
    finalise_plot(fig_raw, "nanorings_raw_signals.pdf", str(output_directory), os)

def plot_normalised_signals(
    trajectories: Tensor,
    time_grid: Tensor,
    h_signal: Tensor,
    output_directory: Path,
) -> None:
    setup_matplotlib_style()
    fig_norm, (ax_norm_amr, ax_norm_h) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    time_axis = time_grid.detach().cpu().numpy()
    amr_responses = trajectories.detach().cpu().numpy()[..., 0]
    h_series = h_signal.detach().cpu().numpy()

    opacity = max(0.08, 3.0 / max(1, amr_responses.shape[0]))
    for idx in tqdm(
        range(amr_responses.shape[0]),
        desc="Plotting normalised AMR responses",
        leave=False,
    ):
        ax_norm_amr.plot(
            time_axis,
            amr_responses[idx],
            color="black",
            alpha=opacity,
            linewidth=0.6,
            label="Normalised AMR" if idx == 0 else "",
        )

    ax_norm_amr.set_ylabel("AMR (normalised)")
    ax_norm_amr.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")
    ax_norm_amr.grid(False)

    ax_norm_h.plot(time_axis, h_series, color=COLOURS[2], linewidth=1.0, label="H-field (normalised)")
    ax_norm_h.set_xlabel("Time [s]")
    ax_norm_h.set_ylabel("H (normalised)")
    ax_norm_h.grid(False)
    ax_norm_h.legend(bbox_to_anchor=(1.0, 1.0), loc="upper left")

    fig_norm.tight_layout()
    finalise_plot(fig_norm, "nanorings_normalised_signals.pdf", str(output_directory), os)


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


def apply_keep_fraction(
    trajectories: Tensor,
    time_grid: Tensor,
    h_signal: Tensor,
    fraction: float,
) -> Tuple[Tensor, Tensor, Tensor, int, Sequence[int]]:
    """Convert full trajectories into fixed-length windows based on keep_fraction.

    The original nanorings dataset contains long trajectories for each H-field
    sweep. When ``fraction`` is small (e.g. 0.05), training on a single cropped
    trajectory wastes the remaining repetitions. This helper slices every run
    into non-overlapping windows of length ``round(fraction * total_steps)`` and
    includes a tail window so the final portion of each run is still seen.

    Args:
        trajectories: Tensor ``[num_runs, total_steps, state_dim]``.
        time_grid: Shared time grid ``[total_steps]``.
        h_signal: Shared exogenous forcing ``[total_steps]``.
        fraction: Fraction of a trajectory to keep per window, ``(0, 1]``.

    Returns:
        windowed_trajectories: Tensor ``[num_windows, keep_steps, state_dim]``.
        window_time_grid: Time grid aligned with each window.
        representative_h_signal: The H signal aligned with ``window_time_grid``.
        keep_steps: Number of steps retained per window.
        windows_per_run: Count of windows generated from each original run.
    """

    if not 0.0 < fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1].")

    total_steps = trajectories.shape[1]
    if trajectories.ndim != 3:
        raise ValueError("trajectories must have shape [num_runs, total_steps, state_dim].")
    if h_signal.ndim != 1 or h_signal.shape[0] != total_steps:
        raise ValueError("h_signal must have shape [total_steps].")
    keep_steps = max(2, int(round(total_steps * fraction)))
    keep_steps = min(total_steps, keep_steps)

    starts: list[int]
    starts = list(range(0, total_steps - keep_steps + 1, keep_steps))
    tail_start = total_steps - keep_steps
    if tail_start >= 0 and (not starts or starts[-1] != tail_start):
        starts.append(tail_start)

    window_list: list[Tensor] = []
    windows_per_run: list[int] = []
    for run_idx in range(trajectories.shape[0]):
        run = trajectories[run_idx]
        run_windows = []
        for start in starts:
            end = start + keep_steps
            if end > total_steps:
                continue
            run_windows.append(run[start:end].clone())
        if not run_windows:
            raise ValueError(
                f"Could not create windows for run {run_idx}; "
                f"keep_steps={keep_steps}, total_steps={total_steps}."
            )
        window_list.extend(run_windows)
        windows_per_run.append(len(run_windows))

    windowed_trajectories = torch.stack(window_list, dim=0)

    base_time = time_grid[:keep_steps].clone()
    if base_time.numel() > 0:
        base_time = base_time - base_time[0].item()

    representative_h = windowed_trajectories[0, :, 1].clone()
    if representative_h.numel() != keep_steps:
        raise ValueError("Representative H signal length mismatch after windowing.")

    return windowed_trajectories, base_time, representative_h, keep_steps, windows_per_run


def resample_to_target_timesteps(
    trajectories: Tensor,
    time_grid: Tensor,
    h_signal: Tensor,
    target_steps: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    """Resample trajectories to a fixed number of timesteps using linear interpolation."""

    if target_steps <= 0:
        raise ValueError("target_steps must be a positive integer.")

    current_steps = trajectories.shape[1]
    if current_steps == target_steps or current_steps <= 1:
        return trajectories, time_grid, h_signal

    start_time = float(time_grid[0].item())
    end_time = float(time_grid[-1].item())
    resampled_time = torch.linspace(
        start_time,
        end_time,
        target_steps,
        dtype=time_grid.dtype,
        device=time_grid.device,
    )

    # Interpolate trajectories: reshape to [batch, channels, time] for F.interpolate.
    traj_channels_first = trajectories.permute(0, 2, 1)
    resampled_traj = F.interpolate(
        traj_channels_first,
        size=target_steps,
        mode="linear",
        align_corners=False,
    ).permute(0, 2, 1).contiguous()

    # H signal interpolation (shape [time]) -> treat as batch=1, channel=1
    h_signal_batch = h_signal.view(1, 1, -1)
    resampled_h = F.interpolate(
        h_signal_batch,
        size=target_steps,
        mode="linear",
        align_corners=False,
    ).view(-1).contiguous()

    return resampled_traj, resampled_time, resampled_h


def normalise_dataset(
    trajectories: Tensor,
    h_signal: Tensor,
    *,
    enabled: bool,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Optional[Dict[str, float]]]:
    """Apply standardisation: baseline removal and normalisation to unit variance."""
    if not enabled:
        return trajectories, h_signal, None

    trajectories = trajectories.clone()
    h_signal = h_signal.clone()

    amr = trajectories[..., 0]
    h = trajectories[..., 1]

    amr_centered = amr - amr.mean(dim=1, keepdim=True)
    h_centered = h - h.mean(dim=1, keepdim=True)

    amr_std = amr_centered.std(unbiased=False).clamp_min(eps)
    h_std = h_centered.std(unbiased=False).clamp_min(eps)

    trajectories[..., 0] = amr_centered / amr_std
    trajectories[..., 1] = h_centered / h_std

    h_signal_centered = h_signal - h_signal.mean()
    h_signal = h_signal_centered / h_std

    stats = {
        "amr_mean": float(amr_centered.mean().item()),
        "amr_std": float(amr_std.item()),
        "h_mean": float(h_signal_centered.mean().item()),
        "h_std": float(h_std.item()),
    }
    return trajectories, h_signal, stats


def preprocess_nanorings_trajectories(
    trajectories: Tensor,
    time_grid: Tensor,
    h_signal: Tensor,
    config: "NanoringsHyperparameters",
    metadata: Dict[str, Any],
) -> Tuple[Tensor, Tensor, Tensor, Optional[Dict[str, float]]]:
    """Apply preprocessing steps 3-5: windowing, resampling, and standardisation.

    This wrapper function orchestrates the complete preprocessing pipeline:
    1. Apply keep_fraction windowing to create fixed-length training windows
    2. Resample trajectories to a consistent number of timesteps
    3. Apply standardisation (baseline removal and normalisation)

    Args:
        trajectories: Tensor ``[num_runs, total_steps, state_dim]``.
        time_grid: Time grid ``[total_steps]``.
        h_signal: Exogenous forcing signal ``[total_steps]``.
        config: NanoringsHyperparameters instance with keep_fraction, TARGET_TIMESTEPS,
                and enable_standardisation settings.
        metadata: Dictionary to update with preprocessing statistics.

    Returns:
        windowed_trajectories: Windowed and resampled ``[num_windows, target_steps, state_dim]``.
        aligned_time_grid: Time grid aligned with windowed trajectories.
        aligned_h_signal: H signal aligned with windowed trajectories.
        standardisation_stats: Optional dict with mean/std values if standardisation enabled.
    """

    # Step 3: Apply keep_fraction windowing
    original_steps = trajectories.shape[1]
    original_duration = float(time_grid[-1] - time_grid[0]) if time_grid.numel() > 1 else 0.0
    LOGGER.info("Step 3: Applying keep_fraction windowing (%.2f%% of trajectory).", config.keep_fraction * 100.0)

    trajectories, time_grid, h_signal, kept_steps, windows_per_run = apply_keep_fraction(
        trajectories,
        time_grid,
        h_signal,
        config.keep_fraction,
    )

    total_windows = trajectories.shape[0]
    total_runs = len(windows_per_run)
    total_samples = total_windows * kept_steps
    metadata["num_steps_retained"] = kept_steps
    metadata["keep_fraction_used"] = config.keep_fraction
    metadata["total_steps_original"] = original_steps
    metadata["num_windows_before_replication"] = total_windows
    metadata["num_windows"] = total_windows
    metadata["windows_per_run"] = list(windows_per_run)
    metadata["num_runs_original"] = total_runs
    metadata["total_samples"] = total_samples

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

    # Step 4: Resample to target timesteps if needed
    LOGGER.info("Step 4: Resampling trajectories to %d timesteps.", config.TARGET_TIMESTEPS)

    if trajectories.shape[1] != config.TARGET_TIMESTEPS:
        trajectories, time_grid, h_signal = resample_to_target_timesteps(
            trajectories,
            time_grid,
            h_signal,
            config.TARGET_TIMESTEPS,
        )
        resampled_steps = trajectories.shape[1]
        metadata["num_steps_retained"] = resampled_steps
        metadata["resampled_to_timesteps"] = config.TARGET_TIMESTEPS
        LOGGER.info("Resampled trajectories to %d timesteps for training.", resampled_steps)
        total_samples = trajectories.shape[0] * resampled_steps
        metadata["total_samples"] = total_samples
        LOGGER.info(
            "Total samples after resampling: %d (per-window steps %d).",
            total_samples,
            resampled_steps,
        )

    # Step 5: Standardisation
    LOGGER.info("Step 5: Applying standardisation (baseline removal and normalisation).")

    trajectories, h_signal, standardisation_stats = normalise_dataset(
        trajectories,
        h_signal,
        enabled=config.enable_standardisation,
    )

    if standardisation_stats is not None:
        LOGGER.info(
            "Applied standardisation | AMR mean %.6f std %.6f | H mean %.6f std %.6f",
            standardisation_stats["amr_mean"],
            standardisation_stats["amr_std"],
            standardisation_stats["h_mean"],
            standardisation_stats["h_std"],
        )
    else:
        LOGGER.info("Standardisation disabled; trajectories remain unnormalised.")

    return trajectories, time_grid, h_signal, standardisation_stats


def predict_with_neural_ode(
    neural_ode: NeuralODE,
    trajectories: Tensor,
    time_grid: Tensor,
) -> Tensor:
    reference_state = trajectories[0:1, 0, :]
    with torch.no_grad():
        prediction = neural_ode(reference_state.to(DEVICE), time_grid.to(DEVICE))[0]
    return prediction.detach().cpu()


def generate_rollouts(
    neural_sde: NeuralSDE,
    trajectories: Tensor,
    time_grid: Tensor,
    sample_count: int,
):
    return sample_sde_rollouts(
        model=neural_sde,
        trajectories=trajectories.to(DEVICE),
        time_grid=time_grid.to(DEVICE),
        num_samples=sample_count,
        device=DEVICE,
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
    for idx in range(n_show):
        ax_amr.plot(
            time_axis,
            amr_training[idx],
            color="gray",
            alpha=0.2,
            linewidth=0.6,
            label="Training" if idx == 0 else "",
        )

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


# ============================================================================
# Training Pipeline
# ============================================================================
_configure_logging()

# Instantiate configuration from hyperparameters
config = NanoringsHyperparameters()

# 1. Device setup
LOGGER.info("Using device: %s", DEVICE)

# 2. Load and prepare raw data
trajectories, time_grid, h_signal, metadata = load_dataset(config.dataset_path)

# 3-5. Preprocessing: windowing, resampling, standardisation
trajectories, time_grid, h_signal, standardisation_stats = preprocess_nanorings_trajectories(
    trajectories,
    time_grid,
    h_signal,
    config,
    metadata,
)

# 6. Build hyperparameters and configuration
# Instantiate experiment configuration and convert to training hyperparameters.
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

# 7. Generate initial visualisations
plot_raw_signals(trajectories, time_grid, h_signal, metadata, output_directory)
LOGGER.info(
    "Saved raw signal plot to %s",
    (output_directory / "nanorings_raw_signals.pdf").as_posix(),
)
if standardisation_stats is not None:
    plot_normalised_signals(trajectories, time_grid, h_signal, output_directory)

# 9. Load or train models
# Check if pre-trained models exist; if not, train from scratch.
loaded = modelling_config.load_models(
    output_directory,
    model_filenames=DEFAULT_MODEL_FILENAMES,
    metric_filenames=DEFAULT_METRIC_FILENAMES,
    device=DEVICE,
)

if loaded is not None:
    # Models already trained - load them
    neural_ode, neural_sde, loaded_metrics = loaded
    train_losses = list(loaded_metrics.get("train_losses", []))
    val_losses = list(loaded_metrics.get("val_losses", []))
    generator_losses = list(loaded_metrics.get("generator_losses", []))
    critic_losses = list(loaded_metrics.get("critic_losses", []))
    drift_losses_sde = list(loaded_metrics.get("drift_losses_sde", []))
    LOGGER.info("Loaded existing checkpoints from %s.", output_directory)
else:
    # Train models from scratch
    LOGGER.info("Training neural ODE...")
    if wandb is not None and config.enable_wandb:
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
    else:
        wandb_run = None

    neural_ode = NeuralODE.train(
        hyperparameters=hyperparameters,
        trajectories=trajectories.to(DEVICE),
        time_grid=time_grid.to(DEVICE),
        device=DEVICE,
        validation_split=config.VALIDATION_SPLIT,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        wandb_run=wandb_run,
    )
    train_losses = list(neural_ode.training_losses)
    val_losses = list(neural_ode.validation_losses)

    # Plot ODE preview immediately after training to verify convergence
    ode_prediction = predict_with_neural_ode(neural_ode, trajectories, time_grid)
    plot_ode_preview(trajectories, time_grid, ode_prediction, output_directory)
    LOGGER.info(
        "Saved Neural ODE preview plot to %s",
        (output_directory / "nanorings_neural_ode_preview.pdf").as_posix(),
    )

    # Train the SDE (stochastic component) on top of the ODE
    LOGGER.info("Training neural SDE...")
    # Use separate batch size for GAN training if provided
    sde_hparams = sde_hyperparameters if sde_hyperparameters is not None else hyperparameters
    LOGGER.info("Using batch size %d for SDE training", sde_hparams.batch_size)
    neural_sde = NeuralSDE.train(
        hyperparameters=sde_hparams,
        neural_ode=neural_ode,
        trajectories=trajectories.to(DEVICE),
        time_grid=time_grid.to(DEVICE),
        device=DEVICE,
        enable_adversarial=modelling_config.enable_adversarial,
        random_seed=0,
        wandb_run=wandb_run,
    )

    generator_losses = list(getattr(neural_sde, "generator_losses", []))
    critic_losses = list(getattr(neural_sde, "critic_losses", []))
    drift_losses_sde = list(getattr(neural_sde, "drift_losses", []))

    training_metrics = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "generator_losses": generator_losses,
        "critic_losses": critic_losses,
        "drift_losses_sde": drift_losses_sde,
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

neural_ode = neural_ode.to(DEVICE)
neural_sde = neural_sde.to(DEVICE)
super(NeuralSDE, neural_sde).train(False)  # Set to evaluation mode

# Clear CUDA cache if available
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# 10. Generate predictions and compute statistics
# Run the trained models to generate predictions and analyze performance.
ode_prediction = predict_with_neural_ode(neural_ode, trajectories, time_grid)

rollout = generate_rollouts(
    neural_sde,
    trajectories,
    time_grid,
    sample_count=modelling_config.sde_sample_count,
)
sde_stats, training_stats = compute_and_compare_statistics(rollout)

# 11. Create final plots and visualizations
# Generate comprehensive plots showing model performance and training curves.
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
