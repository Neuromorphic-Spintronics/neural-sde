"""Compare teacher-forced and horizon-based Neural ODE training on the Duffing system."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence, cast
import sys
import time
import resource
import os
import argparse
import hashlib
import json
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from examples.systems.parameters.duffing_oscillator import PhysicalDuffingParameters
    from examples.systems.duffing_oscillator import DuffingOscillator
except ModuleNotFoundError:  # pragma: no cover - fallback for script execution
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from examples.systems.parameters.duffing_oscillator import PhysicalDuffingParameters
    from examples.systems.duffing_oscillator import DuffingOscillator
from neural_dynamics.config import DEVICE
from neural_dynamics.core.hyperparameters import Hyperparameters, LearningRates
from neural_dynamics.core.utils import (
    COLOURS,
    set_default_plotting_style,
)
from neural_dynamics.core.utils_deprecated import hash_system_parameters  # type: ignore
from neural_dynamics.models.base import DriftNet
from neural_dynamics.models.evaluation import evaluate_rollout_mse  # type: ignore
from neural_dynamics.models.ode import HorizonNeuralODETrainer, NeuralODE  # type: ignore


@dataclass
class TrainingResult:
    """Container for rollout quality metrics."""

    label: str
    horizon: int
    mse: float
    predicted: torch.Tensor
    training_losses: Sequence[float]
    validation_losses: Sequence[float] | None = None
    drift_state_dict: dict[str, torch.Tensor] | None = None
    # New fields for multiple attempts
    mse_mean: float | None = None
    mse_std: float | None = None
    mse_all_attempts: Sequence[float] | None = None
    predictions_all_attempts: Sequence[torch.Tensor] | None = None


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def apply_study_plot_style() -> None:
    """Apply consistent matplotlib styling for the Duffing study."""
    set_default_plotting_style(use_tex=True)
    plt.rcParams.update({"font.size": 14, "axes.linewidth": 1.0})


def generate_duffing_dataset(
    num_trajectories: int,
    physical_params: PhysicalDuffingParameters,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate Duffing oscillator trajectories without down-sampling."""
    system = DuffingOscillator(physical_params.to_dimensionless())
    time_grid, trajectories = system.integrate_sde(batch_size=num_trajectories)

    return time_grid.to(torch.float32), trajectories.to(torch.float32)


def configure_hyperparameters(
    *,
    base: Hyperparameters,
    dt: float,
    number_of_epochs: int,
    batch_size: int,
    drift_learning_rate: float,
) -> Hyperparameters:
    """Return a copy of hyperparameters with study-specific overrides."""
    learning_rates = LearningRates(
        drift=drift_learning_rate,
        diffusion=base.learning_rates.diffusion,
        critic=base.learning_rates.critic,
        generator=base.learning_rates.generator,
    )
    return replace(
        base,
        timestep=dt,
        number_of_epochs=number_of_epochs,
        batch_size=batch_size,
        learning_rates=learning_rates,
    )


def train_teacher_forcing(
    *,
    train_trajs: torch.Tensor,
    eval_trajs: torch.Tensor,
    time_grid: torch.Tensor,
    hyperparameters: Hyperparameters,
    device: torch.device,
    max_attempts: int,
    pretrained_state_dict: dict[str, torch.Tensor] | None = None,
) -> TrainingResult:
    """Train NeuralODE with teacher forcing, with multiple attempts and MSE averaging."""
    if pretrained_state_dict is not None:
        drift_net = DriftNet(hyperparameters.drift_network)
        drift_net.load_state_dict(pretrained_state_dict)
        model = NeuralODE(drift_net=drift_net, hyperparameters=hyperparameters)
        model = model.to(device)
        mse, prediction = evaluate_rollout_mse(
            model=model,
            trajectories=eval_trajs,
            time_grid=time_grid,
            device=device,
        )
        return TrainingResult(
            label="Teacher Forcing",
            horizon=1,
            mse=mse,
            predicted=prediction,
            training_losses=[],
            drift_state_dict=model.drift_net.state_dict(),
            mse_mean=mse,
            mse_std=0.0,
            mse_all_attempts=[mse],
            predictions_all_attempts=[prediction.cpu()],
        )

    best_result: TrainingResult | None = None
    all_mses: list[float] = []
    all_predictions: list[torch.Tensor] = []

    for attempt in range(max_attempts):
        attempt_seed = 1234 + attempt
        set_global_seed(attempt_seed)

        model = NeuralODE.train(
            hyperparameters=hyperparameters,
            trajectories=train_trajs,
            time_grid=time_grid,
            device=device,
            validation_split=0.2,
            early_stopping_patience=hyperparameters.number_of_epochs,
        )

        mse, prediction = evaluate_rollout_mse(
            model=model,
            trajectories=eval_trajs,
            time_grid=time_grid,
            device=device,
        )

        all_mses.append(mse)
        all_predictions.append(prediction.cpu())

        result = TrainingResult(
            label="Teacher Forcing",
            horizon=1,
            mse=mse,
            predicted=prediction,
            training_losses=model.training_losses,
            validation_losses=model.validation_losses
            if model.validation_losses
            else None,
            drift_state_dict=model.drift_net.state_dict(),
        )

        if best_result is None or result.mse < best_result.mse:
            best_result = result

        print(f"  Attempt {attempt + 1}/{max_attempts}: MSE = {result.mse:.4e}")

    if best_result is None:
        raise RuntimeError("Teacher forcing training did not produce a result.")

    # Compute statistics across all attempts
    mse_array = np.array(all_mses)
    mse_mean = float(np.mean(mse_array))
    mse_std = float(np.std(mse_array))

    # Update best_result with aggregated statistics
    best_result.mse_mean = mse_mean
    best_result.mse_std = mse_std
    best_result.mse_all_attempts = all_mses
    best_result.predictions_all_attempts = all_predictions

    print(
        f"Teacher forcing: MSE mean={mse_mean:.4e}, std={mse_std:.4e} (n={len(all_mses)})"
    )

    return best_result


def train_horizon_model(
    *,
    train_trajs: torch.Tensor,
    eval_trajs: torch.Tensor,
    time_grid: torch.Tensor,
    hyperparameters: Hyperparameters,
    device: torch.device,
    max_attempts: int,
    horizon: int,
    initial_state_dict: dict[str, torch.Tensor] | None = None,
    step_method: str = "rk4",
    steps_per_epoch: int | None = 128,
    num_epochs: int | None = None,
    early_stopping_patience: int | None = None,
) -> TrainingResult:
    """Train the horizon-based NeuralODE trainer with multiple attempts and MSE averaging.

    Args:
        train_trajs: Training trajectories.
        eval_trajs: Evaluation trajectories.
        time_grid: Time grid for trajectories.
        hyperparameters: Model hyperparameters.
        device: Torch device.
        max_attempts: Number of training attempts to run and average.
        horizon: Horizon length.
        initial_state_dict: Optional pre-trained weights.
        step_method: Integration method ('rk2' or 'rk4').
        steps_per_epoch: Number of random windows per epoch.
        num_epochs: Number of training epochs.
        early_stopping_patience: Early stopping patience.

    Returns:
        TrainingResult with best single result and statistics across all attempts.
    """
    best_result: TrainingResult | None = None
    all_mses: list[float] = []
    all_predictions: list[torch.Tensor] = []

    for attempt in range(max_attempts):
        attempt_seed = 5678 + attempt
        set_global_seed(attempt_seed)

        trainer = HorizonNeuralODETrainer(
            hyperparameters=hyperparameters,
            horizon_length=horizon,
            batch_size=hyperparameters.batch_size,
            learning_rate=hyperparameters.learning_rates.drift,
            num_epochs=(
                num_epochs
                if num_epochs is not None
                else hyperparameters.number_of_epochs
            ),
            random_seed=attempt_seed,
            initial_state_dict=initial_state_dict,
            step_method=cast(Literal["rk2", "rk4"], step_method),
            steps_per_epoch=steps_per_epoch,
            early_stopping_patience=early_stopping_patience,
        )

        model = trainer.train(
            trajectories=train_trajs,
            time_grid=time_grid,
            device=device,
            progress_desc=f"Horizon N={horizon} (attempt {attempt + 1}/{max_attempts})",
        )

        print(
            f"\nEvaluating horizon N={horizon} (attempt {attempt + 1}/{max_attempts})..."
        )
        try:
            mse, prediction = evaluate_rollout_mse(
                model=model,
                trajectories=eval_trajs,
                time_grid=time_grid,
                device=device,
            )
            print(f"Evaluation complete. MSE={mse:.4e}")
        except Exception as e:
            print(f"Evaluation failed for N={horizon}: {e}. Using dummy result.")
            mse = 1e6
            prediction = eval_trajs[0, :, :2].cpu()

        all_mses.append(mse)
        all_predictions.append(prediction.cpu())

        result = TrainingResult(
            label=f"Horizon $N={horizon}$",
            horizon=horizon,
            mse=mse,
            predicted=prediction,
            training_losses=list(model.training_losses),
            validation_losses=list(model.validation_losses)
            if model.validation_losses
            else None,
            drift_state_dict=model.drift_net.state_dict(),
        )

        if best_result is None or result.mse < best_result.mse:
            best_result = result

        print(f"  Attempt {attempt + 1}/{max_attempts}: MSE = {result.mse:.4e}")

    if best_result is None:
        raise RuntimeError(
            f"Horizon training for N={horizon} did not produce a result."
        )

    # Compute statistics across all attempts
    mse_array = np.array(all_mses)
    mse_mean = float(np.mean(mse_array))
    mse_std = float(np.std(mse_array))

    # Update best_result with aggregated statistics
    best_result.mse_mean = mse_mean
    best_result.mse_std = mse_std
    best_result.mse_all_attempts = all_mses
    best_result.predictions_all_attempts = all_predictions

    print(
        f"Horizon N={horizon}: MSE mean={mse_mean:.4e}, std={mse_std:.4e} (n={len(all_mses)})"
    )

    return best_result


def plot_prediction_rollout(
    *,
    time_grid: torch.Tensor,
    train_trajs: torch.Tensor,
    val_trajs: torch.Tensor,
    test_trajs: torch.Tensor | None = None,
    reference_idx_in_val: int,
    result: TrainingResult,
    output_path: Path,
    title: str,
    color_pred: str,
) -> None:
    """Plot rollout + residual with clear train/validation segment backgrounds.

    Args:
        time_grid: Time grid for the trajectories.
        train_trajs: Training trajectories [N_train, T, D].
        val_trajs: Validation trajectories [N_val, T, D] (used for training reference).
        test_trajs: Test trajectories [N_test, T, D] (used for validation domain; if None, uses empty space).
        reference_idx_in_val: Index within val_trajs to use as the reference trajectory.
        result: TrainingResult containing the prediction.
        output_path: Path to save the figure.
        title: Figure title (unused; kept for compatibility).
        color_pred: Color for the prediction line.
    """
    apply_study_plot_style()

    base_time = time_grid.detach().cpu().numpy()
    if base_time.size < 2:
        raise ValueError(
            "time_grid must contain at least two time points for plotting."
        )
    original_len = base_time.shape[0]
    dt = float(base_time[1] - base_time[0])

    if test_trajs is not None and test_trajs.shape[0] > 0:
        target_trajs = test_trajs
    else:
        target_trajs = val_trajs
    if target_trajs is None or target_trajs.shape[0] == 0:
        raise ValueError(
            "At least one validation or test trajectory is required for plotting."
        )
    target_index = reference_idx_in_val % target_trajs.shape[0]

    pred_eval = result.predicted.detach().cpu().numpy()
    reference_eval = target_trajs[target_index, :, :].detach().cpu().numpy()

    pred_eval = np.where(np.isfinite(pred_eval), pred_eval, np.nan)
    reference_eval = np.where(np.isfinite(reference_eval), reference_eval, np.nan)

    train_np = train_trajs[..., :original_len, :].detach().cpu().numpy()
    target_np = target_trajs[..., :original_len, :].detach().cpu().numpy()

    train_np = np.where(np.isfinite(train_np), train_np, np.nan)
    target_np = np.where(np.isfinite(target_np), target_np, np.nan)

    # For teacher forcing (horizon <= 1), use entire trajectory as training (no validation segment in plot)
    if result.horizon <= 1:
        observed_steps = original_len
    elif result.horizon >= original_len:
        observed_steps = original_len - 1
    else:
        observed_steps = max(1, min(result.horizon, original_len - 1))

    # For horizon training, ensure at least 60% is shown as training
    if result.horizon > 1:
        min_train_steps = max(2, int(original_len * 0.6))
        observed_steps = max(observed_steps, min_train_steps)
        observed_steps = min(observed_steps, original_len - 1)

    t_train = base_time[:observed_steps]
    has_validation_segment = observed_steps < original_len
    if has_validation_segment:
        raw_val_time = base_time[observed_steps - 1 :]
        time_shift = t_train[-1] + dt - raw_val_time[0]
        t_val_shifted = raw_val_time + time_shift
        # Drop the overlapping boundary element so validation starts right after training.
        t_val_plot = t_val_shifted[1:]
        reference_train = reference_eval[:observed_steps, :]
        reference_val = reference_eval[observed_steps - 1 :, :]
        reference_val_plot = reference_val[1:, :]
        pred_train = pred_eval[:observed_steps, :]
        pred_val = pred_eval[observed_steps - 1 :, :]
        pred_val_plot = pred_val[1:, :]
        val_time_with_boundary = np.concatenate(
            (np.array([t_train[-1]]), t_val_plot), axis=0
        )
        val_truth_with_boundary = np.concatenate(
            (reference_train[-1:, :], reference_val_plot), axis=0
        )
        val_pred_with_boundary = np.concatenate(
            (pred_train[-1:, :], pred_val_plot), axis=0
        )
    else:
        t_val_shifted = np.empty(0, dtype=base_time.dtype)
        t_val_plot = t_val_shifted
        reference_train = reference_eval
        reference_val = np.empty_like(reference_train[:0])
        reference_val_plot = reference_val
        pred_train = pred_eval
        pred_val = np.empty_like(pred_train[:0])
        pred_val_plot = pred_val
        val_time_with_boundary = np.empty(0, dtype=base_time.dtype)
        val_truth_with_boundary = np.empty_like(reference_train[:0])
        val_pred_with_boundary = np.empty_like(pred_train[:0])
    train_truth = reference_train

    fig = plt.figure(figsize=(8.5, 5.3))
    from matplotlib.gridspec import GridSpec as _GS

    gs = _GS(
        nrows=2,
        ncols=2,
        figure=fig,
        height_ratios=[1.0, 0.22],
        hspace=0.25,
        wspace=0.35,
    )
    ax_q = fig.add_subplot(gs[0, 0])
    ax_v = fig.add_subplot(gs[0, 1])
    ax_q_res = fig.add_subplot(gs[1, 0], sharex=ax_q)
    ax_v_res = fig.add_subplot(gs[1, 1], sharex=ax_v)

    t_min = float(t_train[0])
    if has_validation_segment and t_val_shifted.size > 0:
        t_max = float(t_val_shifted[-1])
    else:
        t_max = float(t_train[-1])
    t_boundary = float(t_train[-1])

    # Color scheme: validation region only (COLOURS[3] = purple)
    val_color = COLOURS[3]

    # Only add background shading for validation region
    if has_validation_segment:
        # Add shaded validation region to all axes (will be added to legend from ax_q only)
        ax_q.axvspan(
            t_boundary, t_max, alpha=0.1, color=val_color, zorder=0, label="Validation"
        )
        for ax in (ax_v, ax_q_res, ax_v_res):
            ax.axvspan(t_boundary, t_max, alpha=0.1, color=val_color, zorder=0)

    # Training region: black trajectories (no colors, no background)
    n_train_show = min(train_np.shape[0], 100)
    if observed_steps > 0:
        for i in range(n_train_show):
            ax_q.plot(
                t_train,
                train_np[i, :observed_steps, 0],
                color="black",
                linewidth=0.4,
                alpha=0.3,
                zorder=1,
            )
            ax_v.plot(
                t_train,
                train_np[i, :observed_steps, 1],
                color="black",
                linewidth=0.4,
                alpha=0.3,
                zorder=1,
            )

    # Validation region: purple trajectories
    if has_validation_segment:
        n_val_show = min(target_np.shape[0], 60)
        for i in range(n_val_show):
            ax_q.plot(
                t_val_plot,
                target_np[i, observed_steps:, 0],
                color=val_color,
                linewidth=0.38,
                alpha=0.35,
                zorder=1,
            )
            ax_v.plot(
                t_val_plot,
                target_np[i, observed_steps:, 1],
                color=val_color,
                linewidth=0.38,
                alpha=0.35,
                zorder=1,
            )

    # Ground truth: black with different alphas for train/val
    ax_q.plot(
        t_train,
        reference_train[:, 0],
        color="black",
        linewidth=1.4,
        alpha=0.25,
        label="Ground truth (train)",
        zorder=10,
    )
    ax_v.plot(
        t_train,
        reference_train[:, 1],
        color="black",
        linewidth=1.4,
        alpha=0.25,
        zorder=10,
    )

    # Prediction on training: black with alpha=1
    ax_q.plot(
        t_train,
        pred_train[:, 0],
        color="black",
        linewidth=1.25,
        alpha=1.0,
        zorder=11,
        label="Prediction (train)",
    )
    ax_v.plot(
        t_train, pred_train[:, 1], color="black", linewidth=1.25, alpha=1.0, zorder=11
    )

    if has_validation_segment:
        # Ground truth on validation: purple
        ax_q.plot(
            val_time_with_boundary,
            val_truth_with_boundary[:, 0],
            color=val_color,
            linewidth=1.4,
            alpha=0.8,
            label="Ground truth (val)",
            zorder=10,
        )
        ax_v.plot(
            val_time_with_boundary,
            val_truth_with_boundary[:, 1],
            color=val_color,
            linewidth=1.4,
            alpha=0.8,
            zorder=10,
        )
        # Prediction on validation: purple
        ax_q.plot(
            val_time_with_boundary,
            val_pred_with_boundary[:, 0],
            color=val_color,
            linewidth=1.4,
            alpha=1.0,
            zorder=12,
            label="Prediction (val)",
        )
        ax_v.plot(
            val_time_with_boundary,
            val_pred_with_boundary[:, 1],
            color=val_color,
            linewidth=1.4,
            alpha=1.0,
            zorder=12,
        )

    # Residuals: black for training, purple for validation
    res_train = pred_train - reference_train
    res_train = np.where(np.isfinite(res_train), res_train, np.nan)
    ax_q_res.plot(
        t_train, res_train[:, 0], color="black", linewidth=1.1, alpha=1.0, zorder=2
    )
    ax_v_res.plot(
        t_train, res_train[:, 1], color="black", linewidth=1.1, alpha=1.0, zorder=2
    )

    if has_validation_segment:
        res_val_plot = val_pred_with_boundary - val_truth_with_boundary
        res_val_plot = np.where(np.isfinite(res_val_plot), res_val_plot, np.nan)
        ax_q_res.plot(
            val_time_with_boundary,
            res_val_plot[:, 0],
            color=val_color,
            linewidth=1.1,
            alpha=1.0,
            zorder=3,
        )
        ax_v_res.plot(
            val_time_with_boundary,
            res_val_plot[:, 1],
            color=val_color,
            linewidth=1.1,
            alpha=1.0,
            zorder=3,
        )

    ax_q_res.axhline(
        0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1
    )
    ax_v_res.axhline(
        0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1
    )

    # Symmetric 3-tick labels centred about 0 for residuals
    # Collect all residual data for proper scaling
    if has_validation_segment:
        finite_res_q = np.concatenate(
            [
                res_train[:, 0][np.isfinite(res_train[:, 0])],
                res_val_plot[:, 0][np.isfinite(res_val_plot[:, 0])],
            ]
        )
        finite_res_v = np.concatenate(
            [
                res_train[:, 1][np.isfinite(res_train[:, 1])],
                res_val_plot[:, 1][np.isfinite(res_val_plot[:, 1])],
            ]
        )
    else:
        finite_res_q = res_train[:, 0][np.isfinite(res_train[:, 0])]
        finite_res_v = res_train[:, 1][np.isfinite(res_train[:, 1])]

    if finite_res_q.size == 0:
        finite_res_q = np.array([0.0])
    if finite_res_v.size == 0:
        finite_res_v = np.array([0.0])

    # Set symmetric limits about 0 based on max magnitude
    max_abs_q = np.max(np.abs(finite_res_q)) if finite_res_q.size > 0 else 1e-6
    max_abs_v = np.max(np.abs(finite_res_v)) if finite_res_v.size > 0 else 1e-6

    # Ceil to 1 significant figure with padding to ensure no clipping
    def ceil_to_1_sig_fig_with_padding(x):
        if x == 0:
            return 0
        from math import log10, floor, ceil

        # Apply padding first
        x_padded = x * 1.2  # 20% padding to ensure no clipping
        magnitude = floor(log10(abs(x_padded)))
        scale = 10**magnitude
        return ceil(x_padded / scale) * scale

    q_lim_rounded = ceil_to_1_sig_fig_with_padding(max_abs_q)
    v_lim_rounded = ceil_to_1_sig_fig_with_padding(max_abs_v)

    # Set limits first
    ax_q_res.set_ylim(-q_lim_rounded, q_lim_rounded)
    ax_v_res.set_ylim(-v_lim_rounded, v_lim_rounded)

    # Set symmetric 3 ticks: -max, 0, +max
    ax_q_res.set_yticks([-q_lim_rounded, 0.0, q_lim_rounded])
    ax_v_res.set_yticks([-v_lim_rounded, 0.0, v_lim_rounded])

    ax_q.set_ylabel("$q(t)$ [m]")
    ax_v.set_ylabel("$v(t)$ [m s$^{-1}]$")
    ax_q_res.set_ylabel(r"$e_q$")
    ax_v_res.set_ylabel(r"$e_v$")
    ax_q_res.set_xlabel(r"$t$ [s]")
    ax_v_res.set_xlabel(r"$t$ [s]")
    plt.setp(ax_q.get_xticklabels(), visible=False)
    plt.setp(ax_v.get_xticklabels(), visible=False)

    ax_q.set_xlim(t_min, t_max)
    ax_v.set_xlim(t_min, t_max)
    ax_q_res.set_xlim(t_min, t_max)
    ax_v_res.set_xlim(t_min, t_max)

    q_components: list[np.ndarray] = [
        train_truth[:, 0],
        pred_train[:, 0],
        train_np[:, :observed_steps, 0].reshape(-1),
    ]
    v_components: list[np.ndarray] = [
        train_truth[:, 1],
        pred_train[:, 1],
        train_np[:, :observed_steps, 1].reshape(-1),
    ]
    if has_validation_segment:
        q_components.extend(
            [
                val_truth_with_boundary[:, 0],
                val_pred_with_boundary[:, 0],
                target_np[:, observed_steps:, 0].reshape(-1),
            ]
        )
        v_components.extend(
            [
                val_truth_with_boundary[:, 1],
                val_pred_with_boundary[:, 1],
                target_np[:, observed_steps:, 1].reshape(-1),
            ]
        )

    q_data = np.concatenate(
        [comp[np.isfinite(comp)] for comp in q_components if comp.size > 0]
    )
    v_data = np.concatenate(
        [comp[np.isfinite(comp)] for comp in v_components if comp.size > 0]
    )

    if q_data.size > 0:
        q_margin = (q_data.max() - q_data.min()) * 0.05
        ax_q.set_ylim(q_data.min() - q_margin, q_data.max() + q_margin)
    if v_data.size > 0:
        v_margin = (v_data.max() - v_data.min()) * 0.05
        ax_v.set_ylim(v_data.min() - v_margin, v_data.max() + v_margin)

    # Create a single legend, filtering out validation items if no validation segment
    handles_q, labels_q = ax_q.get_legend_handles_labels()
    if handles_q:
        # For teacher forcing (no validation), exclude validation-related legend items
        if not has_validation_segment:
            filtered_handles = []
            filtered_labels = []
            for handle, label in zip(handles_q, labels_q):
                if "val" not in label.lower():
                    filtered_handles.append(handle)
                    filtered_labels.append(label)
            handles_q, labels_q = filtered_handles, filtered_labels

        fig.legend(
            handles_q,
            labels_q,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.02),
            ncol=3,
            frameon=False,
        )
    ax_q.grid(False)
    ax_v.grid(False)
    ax_q_res.grid(False)
    ax_v_res.grid(False)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_training_curve(
    losses: Sequence[float],
    output_path: Path,
    title: str,
    val_losses: Sequence[float] | None = None,
) -> None:
    """Plot training and optional validation loss curves.

    Args:
        losses: Training losses (one per epoch).
        output_path: Path to save the figure.
        title: Figure title (unused; kept for compatibility).
        val_losses: Optional validation losses (one per epoch).
    """
    if not losses:
        return
    apply_study_plot_style()
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot training loss: black
    ax.plot(
        np.arange(len(losses)),
        losses,
        color="black",
        linewidth=1.5,
        label="Training",
        alpha=1.0,
    )

    # Plot validation loss: purple (COLOURS[3])
    if val_losses is not None and len(val_losses) > 0:
        ax.plot(
            np.arange(len(val_losses)),
            val_losses,
            color=COLOURS[3],
            linewidth=1.5,
            label="Validation",
            alpha=1.0,
        )

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (SmoothL1)")
    ax.set_yscale("log")
    ax.grid(False)

    if val_losses is not None and len(val_losses) > 0:
        ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_mse_vs_horizon(
    results: Sequence[TrainingResult], output_path: Path, use_loglog: bool = False
) -> None:
    """Plot mean square error as a function of horizon (2^N on x-axis) with error bars.

    Args:
        results: Sequence of TrainingResult objects.
        output_path: Path to save the figure.
        use_loglog: If True, use log-log scale; otherwise use semilogy (log y-axis only).
    """
    apply_study_plot_style()
    sorted_results = sorted(results, key=lambda item: item.horizon)
    horizons = [res.horizon for res in sorted_results]
    mses = [
        res.mse_mean if res.mse_mean is not None else res.mse for res in sorted_results
    ]
    mse_stds = [
        res.mse_std if res.mse_std is not None else 0.0 for res in sorted_results
    ]

    fig, ax = plt.subplots(figsize=(6, 4))

    # Equally spaced x positions for clear labeling
    unique_horizons = sorted(set(horizons))
    x_pos = np.arange(len(unique_horizons))
    horizon_to_x = {h: x for x, h in zip(x_pos, unique_horizons)}
    x_vals = np.array([horizon_to_x[int(h)] for h in horizons])

    mses = np.asarray(mses, dtype=float)
    mse_stds = np.asarray(mse_stds, dtype=float)

    # Prepare asymmetric errorbars, clipping the lower part to avoid non-positive values
    lower_err = mse_stds.copy()
    eps_frac = 1e-6
    min_allowed = np.maximum(mses * eps_frac, 1e-16)
    lower_err = np.minimum(lower_err, mses - min_allowed)
    lower_err = np.maximum(lower_err, 0.0)
    upper_err = mse_stds.copy()

    ax.errorbar(
        x_vals,
        mses,
        yerr=[lower_err, upper_err],
        fmt="o",
        color="black",
        markersize=6,
        capsize=5,
        linewidth=0,
        elinewidth=1.5,
    )

    ax.set_yscale("log")
    ax.set_xlabel("Horizon")
    ax.set_ylabel("MSE")

    # Compute safe y-limits
    upper = np.max(mses + upper_err)
    lowers = mses - lower_err
    pos_lowers = lowers[lowers > 0]
    if pos_lowers.size > 0:
        lower = np.min(pos_lowers)
    else:
        lower = np.min(mses[mses > 0]) * 1e-3 if np.any(mses > 0) else 1e-12
    y_min = max(lower * 0.8, 1e-16)
    y_max = upper * 1.2
    ax.set_ylim(y_min, y_max)

    # Label x-ticks with raw horizon numbers, equally spaced
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(int(h)) for h in unique_horizons])

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_cpu_mem_vs_horizon(
    *,
    horizons: Sequence[int],
    cpu_seconds: Sequence[float],
    memory_mib: Sequence[float],
    output_path: Path,
) -> None:
    """Plot CPU time and memory vs horizon (2^N on x-axis)."""
    apply_study_plot_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    ax1, ax2 = axes

    # CPU time
    cpu_arr = np.asarray(cpu_seconds, dtype=float)
    cpu_arr[cpu_arr <= 0] = (
        np.min(cpu_arr[cpu_arr > 0]) if np.any(cpu_arr > 0) else 1e-9
    )
    ax1.plot(horizons, cpu_arr, marker="o", color="black", linewidth=2.0)
    ax1.set_xlabel("Horizon")
    ax1.set_ylabel("CPU time [s]")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")

    # Set x-ticks
    unique_horizons = sorted(set(horizons))
    ax1.set_xticks(unique_horizons)
    x_labels = []
    for h in unique_horizons:
        if h == 1:
            x_labels.append(r"$2^{0}$")
        else:
            log_h = np.log2(h)
            if np.isclose(log_h, round(log_h)):
                n = int(round(log_h))
                x_labels.append(rf"$2^{{{n}}}$")
            else:
                x_labels.append(str(h))
    ax1.set_xticklabels(x_labels)
    ax1.grid(False)

    # Memory
    mem_arr = np.asarray(memory_mib, dtype=float)
    mem_arr[mem_arr <= 0] = (
        np.min(mem_arr[mem_arr > 0]) if np.any(mem_arr > 0) else 1e-9
    )
    ax2.plot(horizons, mem_arr, marker="o", color="black", linewidth=2.0)
    ax2.set_xlabel("Horizon")
    ax2.set_ylabel("Memory [MiB]")
    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xticks(unique_horizons)
    ax2.set_xticklabels(x_labels)
    ax2.grid(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_ops_vs_horizon(
    *,
    horizons: Sequence[int],
    operations: Sequence[float],
    output_path: Path,
) -> None:
    """Plot estimated operation count vs horizon (2^N on x-axis).

    steps_per_epoch sets how many random windows you train on each epoch (and thus optimiser steps/epoch).

    The horizon N is how many time steps you integrate per window, scaling work linearly with N.

    The integrator adds a per‑step multiplier (drift_evals_per_step: 2 for RK2, 4 for RK4).

    Finally, epochs multiplies the total passes over those windows.

    A useful proxy is ops_estimate = steps_per_epoch × N × drift_evals_per_step × epochs (e.g., 98,304 × N for RK2 with 96 steps/epoch and 512 epochs). This counts drift forward calls for relative comparison.

    Teacher forcing uses a different proxy tied to per‑time‑step supervision: (T − 1) × num_train × epochs.

    """
    apply_study_plot_style()

    fig, ax = plt.subplots(figsize=(6, 4))
    ops_arr = np.asarray(operations, dtype=float)
    ops_arr[ops_arr <= 0] = (
        np.min(ops_arr[ops_arr > 0]) if np.any(ops_arr > 0) else 1e-9
    )
    ax.plot(horizons, ops_arr, marker="o", color="black", linewidth=2.0)
    ax.set_xlabel("Horizon")
    ax.set_ylabel("Estimated operations [arb units]")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")

    # Set x-ticks
    unique_horizons = sorted(set(horizons))
    ax.set_xticks(unique_horizons)
    x_labels = []
    for h in unique_horizons:
        if h == 1:
            x_labels.append(r"$2^{0}$")
        else:
            log_h = np.log2(h)
            if np.isclose(log_h, round(log_h)):
                n = int(round(log_h))
                x_labels.append(rf"$2^{{{n}}}$")
            else:
                x_labels.append(str(h))
    ax.set_xticklabels(x_labels)

    ax.grid(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def estimate_phase_lag(
    *,
    time_grid: torch.Tensor,
    reference: torch.Tensor,
    predicted: torch.Tensor,
    max_lag: int,
) -> tuple[int, float]:
    """Estimate discrete phase lag by maximising cross-correlation of q(t).

    Args:
        time_grid: 1D tensor of times [T].
        reference: [T, D] reference trajectory (at least q in dim 0).
        predicted: [T, D] predicted trajectory.
        max_lag: maximum shift (in steps) to search, symmetric around 0.

    Returns:
        (best_lag_steps, best_lag_seconds)
    """
    q_ref = reference[:, 0].detach().cpu().numpy()
    q_pred = predicted[:, 0].detach().cpu().numpy()
    q_ref = q_ref - q_ref.mean()
    q_pred = q_pred - q_pred.mean()
    T = len(q_ref)
    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a = q_ref[: T + lag]
            b = q_pred[-lag:]
        elif lag > 0:
            a = q_ref[lag:]
            b = q_pred[: T - lag]
        else:
            a = q_ref
            b = q_pred
        if len(a) == 0 or len(b) == 0:
            continue
        corr = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    dt = float((time_grid[1] - time_grid[0]).item())
    return best_lag, best_lag * dt


def save_trajectories_to_csv(
    *,
    output_dir: Path,
    time_grid: torch.Tensor,
    reference_trajs: torch.Tensor,
    result: TrainingResult,
    prefix: str,
) -> None:
    """Save reference and predicted trajectories to CSV files.

    Args:
        output_dir: Directory to save CSV files.
        time_grid: Time grid [T].
        reference_trajs: Reference trajectories [N, T, D].
        result: TrainingResult containing predictions.
        prefix: Prefix for CSV filenames (e.g., 'teacher_forcing' or 'horizon_32').
    """
    time_np = time_grid.detach().cpu().numpy()

    # Save best prediction (only q and v, first 2 dimensions)
    pred_np = result.predicted.detach().cpu().numpy()[:, :2]
    pred_data = np.column_stack([time_np, pred_np])
    pred_header = "time,q,v"
    np.savetxt(
        output_dir / f"{prefix}_prediction_best.csv",
        pred_data,
        delimiter=",",
        header=pred_header,
        comments="",
    )

    # Save all predictions from all attempts (if available)
    if result.predictions_all_attempts is not None:
        for attempt_idx, pred_tensor in enumerate(result.predictions_all_attempts):
            pred_attempt_np = pred_tensor.detach().cpu().numpy()[:, :2]
            pred_attempt_data = np.column_stack([time_np, pred_attempt_np])
            np.savetxt(
                output_dir / f"{prefix}_prediction_attempt_{attempt_idx + 1}.csv",
                pred_attempt_data,
                delimiter=",",
                header=pred_header,
                comments="",
            )

    # Save reference trajectories (first 10 for brevity)
    n_save = min(10, reference_trajs.shape[0])
    for traj_idx in range(n_save):
        ref_np = reference_trajs[traj_idx, :, :2].detach().cpu().numpy()
        ref_data = np.column_stack([time_np, ref_np])
        np.savetxt(
            output_dir / f"{prefix}_reference_traj_{traj_idx + 1}.csv",
            ref_data,
            delimiter=",",
            header=pred_header,
            comments="",
        )

    # Save MSE statistics if available
    if result.mse_all_attempts is not None:
        mse_data = np.array(
            [[i + 1, mse] for i, mse in enumerate(result.mse_all_attempts)]
        )
        np.savetxt(
            output_dir / f"{prefix}_mse_per_attempt.csv",
            mse_data,
            delimiter=",",
            header="attempt,mse",
            comments="",
        )


def main() -> None:
    """Entry point for the horizon comparison study."""
    parser = argparse.ArgumentParser(
        description="Duffing horizon vs teacher-forcing study"
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="*",
        help="Horizon sizes to run. Include 1 to run teacher forcing only (no horizons if 1 is the only value).",
    )
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Skip teacher-forcing training/evaluation.",
    )
    parser.add_argument(
        "--step-method",
        choices=["rk2", "rk4"],
        default="rk2",
        help="Single-step integrator for horizon training (default: rk2).",
    )
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=96,
        help="Number of random horizon windows per epoch (default: 96).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=512,
        help="Number of training epochs for horizon runs (default: 512).",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="examples/output/duffing_horizon_comparison",
        help="Base output directory. A unique subdirectory will be created to avoid overwrites.",
    )
    parser.add_argument(
        "--tag",
        type=str,
        default=None,
        help="Optional tag for this run. If omitted, a timestamp is used.",
    )
    parser.add_argument(
        "--num-attempts",
        type=int,
        default=5,
        help="Number of training attempts to run and average for MSE statistics (default: 5).",
    )
    args, _ = parser.parse_known_args()

    set_global_seed(42)

    # Create a descriptive per-run output directory to avoid overwriting CSVs/figures.
    args_for_hash = vars(args).copy()
    output_root_str = args_for_hash.pop("output_root")
    tag_override = args_for_hash.pop("tag", None)

    # Build a descriptive name from key parameters
    horizons_arg = args_for_hash.get("horizons", [1, 2, 4, 8, 16, 32, 64])
    step_method = args_for_hash.get("step_method", "rk2")
    steps_per_epoch = args_for_hash.get("steps_per_epoch", 96)
    epochs = args_for_hash.get("epochs", 512)

    # Create horizon range description (e.g., "h2-64" or "h32-512")
    if horizons_arg and len(horizons_arg) > 0:
        h_min = min(horizons_arg)
        h_max = max(horizons_arg)
        horizon_desc = f"h{h_min}-{h_max}"
    else:
        horizon_desc = "h_default"

    # Create parameter string
    param_parts = [
        horizon_desc,
        step_method,
        f"s{steps_per_epoch}",
        f"e{epochs}",
    ]
    param_desc = "_".join(param_parts)

    # Hash the full args for uniqueness (in case of identical param descriptions)
    args_serialised = json.dumps(args_for_hash, sort_keys=True, default=str)
    arg_hash = hashlib.sha256(args_serialised.encode("utf-8")).hexdigest()[:6]

    # Use custom tag or timestamp
    if tag_override:
        run_tag = f"{param_desc}_{tag_override}_{arg_hash}"
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_tag = f"{param_desc}_{timestamp}_{arg_hash}"
    output_dir = Path(output_root_str) / run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs under: {output_dir}")
    print(f"Argument hash suffix: {arg_hash}")

    physical_params = PhysicalDuffingParameters(
        temperature=0.0, coloured_noise_intensity=0.0, timestep=0.01
    )
    # Use all available CPU cores for Torch ops
    try:
        n_threads = max(1, os.cpu_count() or 1)
        torch.set_num_threads(n_threads)
        torch.set_num_interop_threads(min(n_threads, 4))
        print(f"Torch threads set to {n_threads}")
    except Exception:
        pass
    time_grid, trajectories = generate_duffing_dataset(
        num_trajectories=512,
        physical_params=physical_params,
    )
    split_idx = int(trajectories.size(0) * 0.8)
    train_trajs = trajectories[:split_idx]
    eval_trajs = trajectories[split_idx:]

    if eval_trajs.size(0) == 0:
        raise ValueError(
            "Insufficient evaluation trajectories. Increase num_trajectories."
        )

    base_hyperparameters = Hyperparameters.defaults()
    dt = float((time_grid[1] - time_grid[0]).item())
    study_hyperparameters = configure_hyperparameters(
        base=base_hyperparameters,
        dt=dt,
        number_of_epochs=1024,
        batch_size=16,
        drift_learning_rate=3e-4,
    )
    state_dim = study_hyperparameters.state_dimension

    # Split eval_trajs into val (for loss curves during training) and test (for final evaluation)
    # 50/50 split: first half = validation during training, second half = test for final metrics
    split_idx_val = eval_trajs.shape[0] // 2
    val_trajs = eval_trajs[:split_idx_val]
    test_trajs = eval_trajs[split_idx_val:]

    # Use first validation trajectory as reference for plotting
    reference_idx_in_val = 0
    reference_state = val_trajs[reference_idx_in_val, :, :state_dim].cpu()

    # Locate a pre-trained teacher drift checkpoint from the Duffing notebook.
    # Primary expected path uses the parameter hash; if absent, fall back to scanning examples/output.
    teacher_source_hash = hash_system_parameters(physical_params)
    expected_dir = Path("examples/output") / f"duffing_{teacher_source_hash}"
    expected_ckpt = expected_dir / "neural_ode_state_dict.pt"
    pretrained_state: dict[str, torch.Tensor] | None = None
    loaded_from: Path | None = None
    if expected_ckpt.exists():
        loaded_from = expected_ckpt
    else:
        # Fallback: search any duffing_* directory for neural_ode_state_dict.pt and pick the newest
        candidates = list(
            Path("examples/output").glob("duffing_*/neural_ode_state_dict.pt")
        )
        if candidates:
            loaded_from = max(candidates, key=lambda p: p.stat().st_mtime)
    if loaded_from is not None and loaded_from.exists():
        try:
            pretrained_state = torch.load(loaded_from, map_location=DEVICE)
            print(f"Loaded pre-trained teacher drift from {loaded_from}")
        except Exception as e:
            print(
                f"Failed to load checkpoint at {loaded_from}: {e}. Will train from scratch."
            )
            pretrained_state = None
    else:
        print(
            "No pre-trained teacher drift found; looked for",
            expected_ckpt,
            "or any examples/output/duffing_*/neural_ode_state_dict.pt. Will train from scratch.",
        )

    # Measure teacher resource usage
    cpu_before = time.process_time()
    wall_before = time.perf_counter()
    rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

    # Determine whether to run teacher forcing
    horizons_arg = (
        args.horizons if args.horizons is not None else [1, 2, 4, 8, 16, 32, 64]
    )
    run_teacher = (not args.skip_teacher) and (1 in horizons_arg)

    teacher_result: TrainingResult | None = None
    if run_teacher:
        teacher_result = train_teacher_forcing(
            train_trajs=train_trajs,
            eval_trajs=test_trajs,
            time_grid=time_grid,
            hyperparameters=study_hyperparameters,
            device=DEVICE,
            max_attempts=args.num_attempts,
            pretrained_state_dict=pretrained_state,
        )

    cpu_after = time.process_time()
    wall_after = time.perf_counter()
    rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    teacher_cpu_sec = max(0.0, cpu_after - cpu_before)
    teacher_wall_sec = max(0.0, wall_after - wall_before)
    # ru_maxrss: on macOS bytes, on Linux kilobytes; convert heuristically to MiB
    teacher_mem_mib = max(0.0, (rss_after_kb - rss_before_kb)) / (
        1024.0 if rss_after_kb < 10**7 else (1024.0 * 1024.0)
    )

    if teacher_result is not None:
        teacher_state_path = output_dir / "teacher_forcing_drift.pt"
        torch.save(teacher_result.drift_state_dict, teacher_state_path)

        teacher_loss_path = output_dir / "teacher_forcing_losses.png"
        plot_training_curve(
            losses=teacher_result.training_losses,
            output_path=teacher_loss_path,
            title="Teacher Forcing Training Loss",
            val_losses=teacher_result.validation_losses,
        )
        if teacher_result.training_losses:
            np.savetxt(
                output_dir / "teacher_forcing_losses.txt",
                np.asarray(teacher_result.training_losses),
            )

        teacher_rollout_path = output_dir / "teacher_forcing_rollout.png"
        plot_prediction_rollout(
            time_grid=time_grid,
            train_trajs=train_trajs,
            val_trajs=val_trajs,
            test_trajs=None,
            reference_idx_in_val=reference_idx_in_val,
            result=teacher_result,
            output_path=teacher_rollout_path,
            title="Teacher Forcing Rollout",
            color_pred=COLOURS[2],
        )

        # Save teacher forcing trajectory data to CSV
        save_trajectories_to_csv(
            output_dir=output_dir,
            time_grid=time_grid,
            reference_trajs=test_trajs,
            result=teacher_result,
            prefix="teacher_forcing",
        )

    # Prepare study horizons
    horizon_lengths = [h for h in horizons_arg if h >= 2]
    horizon_results: list[TrainingResult] = []
    horizons_for_metrics: list[int] = [1] if teacher_result is not None else []
    cpu_secs: list[float] = [teacher_cpu_sec] if teacher_result is not None else []
    wall_secs: list[float] = [teacher_wall_sec] if teacher_result is not None else []
    mem_mib: list[float] = [teacher_mem_mib] if teacher_result is not None else []
    ops_estimate: list[float] = []

    # Estimate teacher operations: per epoch we evaluate (T-1) derivatives per trajectory
    T = int(time_grid.shape[0])
    num_train = int(train_trajs.shape[0])
    teacher_epochs = int(study_hyperparameters.number_of_epochs)
    if teacher_result is not None:
        teacher_ops = float((T - 1) * num_train * teacher_epochs)
        ops_estimate.append(teacher_ops)
    for idx, horizon in enumerate(horizon_lengths):
        cpu_before = time.process_time()
        wall_before = time.perf_counter()
        rss_before_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result = train_horizon_model(
            train_trajs=train_trajs,
            eval_trajs=test_trajs,
            time_grid=time_grid,
            hyperparameters=study_hyperparameters,
            device=DEVICE,
            max_attempts=args.num_attempts,
            horizon=horizon,
            initial_state_dict=(
                teacher_result.drift_state_dict if teacher_result is not None else None
            ),
            step_method=args.step_method,
            steps_per_epoch=int(args.steps_per_epoch),
            num_epochs=args.epochs,
            early_stopping_patience=15,  # Stop if validation loss doesn't improve for 15 epochs
        )
        horizon_results.append(result)
        cpu_after = time.process_time()
        wall_after = time.perf_counter()
        rss_after_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        horizons_for_metrics.append(horizon)
        cpu_secs.append(max(0.0, cpu_after - cpu_before))
        wall_secs.append(max(0.0, wall_after - wall_before))
        mem_mib.append(
            max(0.0, rss_after_kb - rss_before_kb)
            / (1024.0 if rss_after_kb < 10**7 else (1024.0 * 1024.0))
        )

        # Save the rollout image immediately after finishing each horizon
        horizon_rollout_path = output_dir / f"horizon_{horizon}_rollout.png"
        color = COLOURS[(idx + 3) % len(COLOURS)]
        print(f"Plotting horizon N={horizon} rollout...")
        plot_prediction_rollout(
            time_grid=time_grid,
            train_trajs=train_trajs,
            val_trajs=val_trajs,
            test_trajs=test_trajs,
            reference_idx_in_val=reference_idx_in_val,
            result=result,
            output_path=horizon_rollout_path,
            title=f"Horizon Rollout (N={horizon})",
            color_pred=color,
        )
        print(f"Saved: {horizon_rollout_path}")
        # Save horizon checkpoint and losses
        horizon_state_path = output_dir / f"horizon_{horizon}_drift.pt"
        torch.save(result.drift_state_dict, horizon_state_path)
        horizon_loss_path = output_dir / f"horizon_{horizon}_losses.png"
        plot_training_curve(
            losses=result.training_losses,
            output_path=horizon_loss_path,
            title=f"Horizon N={horizon} Training Loss",
            val_losses=result.validation_losses,
        )
        if result.training_losses:
            np.savetxt(
                output_dir / f"horizon_{horizon}_losses.txt",
                np.asarray(result.training_losses),
            )

        # Save horizon trajectory data to CSV
        save_trajectories_to_csv(
            output_dir=output_dir,
            time_grid=time_grid,
            reference_trajs=test_trajs,
            result=result,
            prefix=f"horizon_{horizon}",
        )

        # Estimate ops for horizon: batches_per_epoch * horizon_length * drift_evals_per_step * epochs
        drift_evals_per_step = 2 if args.step_method == "rk2" else 4
        batches_per_epoch = int(args.steps_per_epoch)
        ops = float(batches_per_epoch * horizon * drift_evals_per_step * args.epochs)
        ops_estimate.append(ops)

        # Phase lag diagnostics (q only)
        try:
            # Recompute rollout vs reference for lag estimate using the saved result
            best_lag_steps, best_lag_sec = estimate_phase_lag(
                time_grid=time_grid,
                reference=reference_state,
                predicted=result.predicted,
                max_lag=min(horizon, int(0.1 * len(time_grid))),
            )
            with open(output_dir / f"horizon_{horizon}_phase_lag.txt", "w") as f:
                f.write(f"best_lag_steps,{best_lag_steps}\n")
                f.write(f"best_lag_seconds,{best_lag_sec}\n")
        except Exception as e:
            print(f"Phase lag estimate failed for N={horizon}: {e}")

    summary_path = output_dir / "mse_vs_horizon.png"
    all_results = (
        [teacher_result] if teacher_result is not None else []
    ) + horizon_results
    plot_mse_vs_horizon(all_results, summary_path, use_loglog=True)

    # Save numeric summaries (MSE and resource metrics) with mean and std
    mse_data = []
    for res in all_results:
        mse_mean = res.mse_mean if res.mse_mean is not None else res.mse
        mse_std = res.mse_std if res.mse_std is not None else 0.0
        mse_data.append([res.horizon, res.mse, mse_mean, mse_std])
    mses = np.array(mse_data, dtype=float)
    np.savetxt(
        output_dir / "mse_vs_horizon.csv",
        mses,
        delimiter=",",
        header="horizon,mse_best,mse_mean,mse_std",
        comments="",
    )

    # Plot CPU/memory vs horizon and operations vs horizon (include N=1)
    cpu_mem_path = output_dir / "cpu_memory_vs_horizon.png"
    plot_cpu_mem_vs_horizon(
        horizons=horizons_for_metrics,
        cpu_seconds=cpu_secs,
        memory_mib=mem_mib,
        output_path=cpu_mem_path,
    )
    ops_path = output_dir / "operations_vs_horizon.png"
    plot_ops_vs_horizon(
        horizons=horizons_for_metrics,
        operations=ops_estimate,
        output_path=ops_path,
    )

    # Also save raw metrics CSV
    metrics = np.column_stack(
        [
            np.array(horizons_for_metrics, dtype=float),
            np.array(cpu_secs, dtype=float),
            np.array(wall_secs, dtype=float),
            np.array(mem_mib, dtype=float),
            np.array(ops_estimate, dtype=float),
        ]
    )
    np.savetxt(
        output_dir / "metrics_vs_horizon.csv",
        metrics,
        delimiter=",",
        header="horizon,cpu_seconds,wall_seconds,memory_mib,ops_estimate",
        comments="",
    )

    print("Results")
    print("-------")
    if teacher_result is not None:
        print(f"Teacher forcing (N=1) MSE: {teacher_result.mse:.4e}")
    else:
        print("Teacher forcing (N=1) was not executed.")
    for result in horizon_results:
        print(f"{result.label} MSE: {result.mse:.4e}")
    print(f"Figures saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
