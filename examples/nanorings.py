#!/usr/bin/env python3
"""Train and evaluate the nanorings neural ODE/SDE models from the terminal."""

from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch import Tensor
from tqdm.auto import tqdm

from examples.systems.parameters.nanorings import NanoringsHyperparameters
from examples.utils.plotting import finalise_plot, setup_matplotlib_style
from neural_dynamics.config import DEVICE
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
from neural_dynamics.models.sde import (
    NeuralSDE,
    train_critic_step,
    train_generator_step,
)
from neural_dynamics.utils.training_helpers import (
    DEFAULT_METRIC_FILENAMES,
    DEFAULT_MODEL_FILENAMES,
    DEFAULT_TRAINING_OUTPUT_DIR,
    ModellingConfig,
)
from neural_dynamics.models import sde as sde_module

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

        if self.training_trajectories is None or self.time_grid_device is None or self.time_grid_cpu is None:
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
            if previous_mode:
                model.train()  # type: ignore

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
    
    # Configure drift network training mode based on hyperparameter
    if hyperparameters.train_ode_with_sde:
        drift_network.train()
        generator_params = list(drift_network.parameters()) + list(diffusion_network.parameters())
    else:
        # Freeze the drift network by setting it to evaluation mode
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
    drift_losses: List[float] = []  # Track drift losses when training jointly

    stochastic_trajectories = stochastic_trajectories.to(DEVICE)
    time_grid = time_grid.to(DEVICE)
    
    # Get critic window size and trajectory length for random window sampling
    critic_window = critic_network.trajectory_length
    full_trajectory_length = stochastic_trajectories.shape[1]
    
    if full_trajectory_length < critic_window:
        raise ValueError(
            f"Trajectory length {full_trajectory_length} is shorter than "
            f"critic window {critic_window}"
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
            epoch_drift_losses: List[float] = []  # Track drift-only loss when joint training

            for batch_idx in range(num_batches):
                start_idx = batch_idx * hyperparameters.batch_size
                end_idx = min(
                    start_idx + hyperparameters.batch_size,
                    stochastic_trajectories.shape[0],
                )

                if start_idx >= end_idx:
                    continue

                # Sample a random window position for this batch
                window_start = torch.randint(0, max_window_start + 1, (1,)).item()
                window_end = window_start + critic_window
                
                # Extract windowed segments directly (data already on DEVICE)
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
                    epoch_critic_losses.append(critic_loss)

                # Compute drift-only loss WITH gradients if training drift jointly
                drift_only_loss_tensor = None
                drift_l1_weight = getattr(hyperparameters, 'drift_l1_weight', 0.0)
                if hyperparameters.train_ode_with_sde and drift_l1_weight > 0:
                    # Compute deterministic drift-only predictions WITH gradients
                    # This adds an explicit SmoothL1 constraint to keep drift network accurate
                    drift_predictions = []
                    current_state = initial_states
                    for step_idx in range(1, len(window_time_grid)):
                        dt = float(window_time_grid[step_idx] - window_time_grid[step_idx - 1])
                        t = float(window_time_grid[step_idx - 1])
                        time_tensor = torch.full((current_state.shape[0], 1), t, device=DEVICE, dtype=current_state.dtype)
                        drift = drift_network.compute_drift(current_state, time_tensor, external_inputs=None)
                        next_state = current_state + drift * dt
                        drift_predictions.append(next_state)
                        current_state = next_state
                    drift_predictions = torch.stack(drift_predictions, dim=1).contiguous()
                    
                    # Compute SmoothL1 loss (WITH gradients for backprop)
                    drift_only_loss_tensor = F.smooth_l1_loss(
                        drift_predictions, real_trajectories[:, 1:, :].contiguous()
                    )
                    
                    # Debug: check if loss is being computed correctly
                    if epoch == 0 and batch_idx == 0:
                        print("\n[DEBUG] First batch of SDE training:")
                        print(f"  drift_predictions shape: {drift_predictions.shape}")
                        print(f"  real_trajectories[:, 1:, :] shape: {real_trajectories[:, 1:, :].shape}")
                        print(f"  drift_only_loss_tensor: {drift_only_loss_tensor.item():.6f}")
                        print(f"  drift_l1_weight: {drift_l1_weight}")
                        print(f"  drift_predictions mean: {drift_predictions.mean().item():.6f}, std: {drift_predictions.std().item():.6f}")
                        print(f"  real_trajectories mean: {real_trajectories[:, 1:, :].mean().item():.6f}, std: {real_trajectories[:, 1:, :].std().item():.6f}")
                        print(f"  drift_predictions.requires_grad: {drift_predictions.requires_grad}")
                        print(f"  drift_only_loss_tensor.requires_grad: {drift_only_loss_tensor.requires_grad}")
                        print(f"  drift_network.training: {drift_network.training}")

                # Store initial drift network parameters for gradient verification
                if epoch == 0 and batch_idx == 0:
                    first_param_before = next(drift_network.parameters()).clone().detach()
                
                generator_loss, loss_components = train_generator_step(
                    generator_optimiser,
                    critic_network,
                    fake_trajectories,
                    real_trajectories,
                    hyperparameters.sde_l1_weight,
                    moment_matching_weight=getattr(
                        hyperparameters, "moment_matching_weight", 0.0
                    ),
                    moment_matching_enabled=getattr(
                        hyperparameters, "moment_matching_enabled", False
                    ),
                    diffusion_network=diffusion_network,
                    drift_only_loss=drift_only_loss_tensor,
                    drift_l1_weight=drift_l1_weight,
                )
                epoch_generator_losses.append(generator_loss)

                # Print loss component breakdown for first few epochs
                if epoch < 3 and batch_idx == 0:
                    print(f"\n{'='*80}")
                    print(f"NANORINGS WRAPPER - EPOCH {epoch} BATCH {batch_idx}")
                    print(f"{'='*80}")
                    print(f"  Adversarial:     {loss_components['adversarial']:>10.6f}")
                    print(f"  SDE L1 (×{hyperparameters.sde_l1_weight}):    {loss_components['sde_l1_weighted']:>10.6f}")
                    print(f"  Drift (×{drift_l1_weight}):     {loss_components['drift_only_weighted']:>10.6f}")
                    print(f"  TOTAL:           {generator_loss:>10.6f}")
                    print(f"{'='*80}\n")
                
                # Verify drift network gradients were computed and parameters updated
                if epoch == 0 and batch_idx == 0:
                    first_param = next(drift_network.parameters())
                    first_param_after = first_param.clone().detach()
                    param_changed = not torch.allclose(first_param_before, first_param_after, atol=1e-10)
                    has_grad = first_param.grad is not None
                    grad_norm = first_param.grad.norm().item() if has_grad else 0.0
                    print("  [GRADIENT CHECK]")
                    print(f"    Drift network param has gradient: {has_grad}")
                    print(f"    Gradient norm: {grad_norm:.6f}")
                    print(f"    Parameters changed after optimizer step: {param_changed}")
                    print(f"    Parameter change magnitude: {(first_param_after - first_param_before).abs().max().item():.6e}")
                
                # Track drift-only loss for logging (reuse computed value)
                if hyperparameters.train_ode_with_sde and drift_only_loss_tensor is not None:
                    epoch_drift_losses.append(drift_only_loss_tensor.item())

            generator_losses.append(
                sum(epoch_generator_losses) / len(epoch_generator_losses)
                if epoch_generator_losses
                else 0.0
            )
            critic_losses.append(
                sum(epoch_critic_losses) / len(epoch_critic_losses)
                if epoch_critic_losses
                else 0.0
            )
            
            # Track drift loss for plotting continuity when training jointly
            if hyperparameters.train_ode_with_sde and epoch_drift_losses:
                drift_losses.append(sum(epoch_drift_losses) / len(epoch_drift_losses))
            else:
                drift_losses.append(0.0)

            # Log to W&B if available (offset by actual ODE epochs trained)
            if wandb_run is not None:
                # Use actual ODE epochs if provided, otherwise fall back to configured value
                ode_epochs = actual_ode_epochs if actual_ode_epochs is not None else getattr(hyperparameters, 'number_of_epochs', 0)
                step = ode_epochs + epoch
                log_data = {
                    "sde/generator_loss": generator_losses[-1],
                    "sde/critic_loss": critic_losses[-1]
                }
                
                # Log drift loss as continuation of ODE training (same metric name for continuity)
                if drift_losses[-1] > 0:  # Only log if drift loss was computed
                    log_data["ode/train_loss"] = drift_losses[-1]  # Continue the ODE plot
                    log_data["sde/drift_loss"] = drift_losses[-1]  # Also keep separate metric
                
                # Debug: print first few epochs to understand what's happening
                if epoch < 5:
                    print(f"\n[DEBUG] Epoch {epoch}: drift_loss={drift_losses[-1]:.4f}, generator_loss={generator_losses[-1]:.4f}, critic_loss={critic_losses[-1]:.4f}")
                
                wandb_run.log(log_data, step=step)
                
                # Log detailed checkpoint every 100 epochs
                if (epoch + 1) % 100 == 0 or epoch == 0:
                    checkpoint_data = {
                        "sde/checkpoint/epoch": epoch + 1,
                        "sde/checkpoint/generator_loss": generator_losses[-1],
                        "sde/checkpoint/critic_loss": critic_losses[-1],
                    }
                    if drift_losses[-1] > 0:
                        checkpoint_data["sde/checkpoint/drift_loss"] = drift_losses[-1]
                    
                    # Add statistics about batch losses if available
                    if epoch_generator_losses:
                        checkpoint_data["sde/checkpoint/generator_loss_mean"] = sum(epoch_generator_losses) / len(epoch_generator_losses)
                        checkpoint_data["sde/checkpoint/generator_loss_std"] = (
                            sum((x - checkpoint_data["sde/checkpoint/generator_loss_mean"]) ** 2 for x in epoch_generator_losses) / len(epoch_generator_losses)
                        ) ** 0.5 if len(epoch_generator_losses) > 1 else 0.0
                    if epoch_critic_losses:
                        checkpoint_data["sde/checkpoint/critic_loss_mean"] = sum(epoch_critic_losses) / len(epoch_critic_losses)
                        checkpoint_data["sde/checkpoint/critic_loss_std"] = (
                            sum((x - checkpoint_data["sde/checkpoint/critic_loss_mean"]) ** 2 for x in epoch_critic_losses) / len(epoch_critic_losses)
                        ) ** 0.5 if len(epoch_critic_losses) > 1 else 0.0
                    if epoch_drift_losses:
                        checkpoint_data["sde/checkpoint/drift_loss_mean"] = sum(epoch_drift_losses) / len(epoch_drift_losses)
                        checkpoint_data["sde/checkpoint/drift_loss_std"] = (
                            sum((x - checkpoint_data["sde/checkpoint/drift_loss_mean"]) ** 2 for x in epoch_drift_losses) / len(epoch_drift_losses)
                        ) ** 0.5 if len(epoch_drift_losses) > 1 else 0.0
                    
                    wandb_run.log(checkpoint_data, step=step)

            pbar.set_postfix(
                {"Gen Loss": generator_losses[-1], "Critic Loss": critic_losses[-1]}
            )

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
        _ORIGINAL_FIT_NEURAL_SDE_GAN = sde_module.fit_neural_sde_gan
        sde_module.fit_neural_sde_gan = _fit_neural_sde_gan_with_checkpoints  # type: ignore

    _STATS_CHECKPOINT_MANAGER = manager


def _restore_sde_checkpointing() -> None:
    """Restore the original GAN trainer and clear global state."""

    global _ORIGINAL_FIT_NEURAL_SDE_GAN, _STATS_CHECKPOINT_MANAGER
    if _ORIGINAL_FIT_NEURAL_SDE_GAN is not None:
        sde_module.fit_neural_sde_gan = _ORIGINAL_FIT_NEURAL_SDE_GAN  # type: ignore
        _ORIGINAL_FIT_NEURAL_SDE_GAN = None
    _STATS_CHECKPOINT_MANAGER = None


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


def replicate_trajectories(
    trajectories: Tensor,
    factor: int,
) -> Tensor:
    """Replicate trajectories to augment dataset size."""
    if factor < 1:
        raise ValueError("replication_factor must be a positive integer.")
    if factor == 1:
        return trajectories
    return trajectories.repeat((factor, 1, 1))


def normalise_dataset(
    trajectories: Tensor,
    h_signal: Tensor,
    *,
    enabled: bool,
    eps: float = 1e-8,
) -> Tuple[Tensor, Tensor, Optional[Dict[str, float]]]:
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
            LOGGER.info(
                "Statistics checkpointing skipped (existing checkpoints loaded)."
            )
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
        ode_prediction = predict_with_neural_ode(neural_ode, trajectories_device, time_grid_device, device)
        plot_ode_preview(trajectories_device, time_grid_device, ode_prediction, output_directory)
        LOGGER.info(
            "Saved Neural ODE preview plot to %s",
            (output_directory / "nanorings_neural_ode_preview.pdf").as_posix(),
        )

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


def main(argv: Optional[Sequence[str]] = None) -> None:
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    args = parse_args(argv)
    _configure_logging()

    device = torch.device(args.device) if args.device is not None else DEVICE
    LOGGER.info("Using device: %s", device)

    trajectories, time_grid, h_signal, metadata = load_dataset(args.dataset_path)
    original_steps = trajectories.shape[1]
    original_duration = (
        float(time_grid[-1] - time_grid[0]) if time_grid.numel() > 1 else 0.0
    )
    (
        trajectories,
        time_grid,
        h_signal,
        kept_steps,
        windows_per_run,
    ) = apply_keep_fraction(
        trajectories,
        time_grid,
        h_signal,
        args.keep_fraction,
    )
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
        trajectories, time_grid, h_signal = resample_to_target_timesteps(
            trajectories,
            time_grid,
            h_signal,
            TARGET_TIMESTEPS,
        )
        kept_steps = trajectories.shape[1]
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
            metadata["total_samples"],
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
            "Intermediate statistics checkpointing enabled every %d epoch(s) "
            "(limit=%s, samples=%d).",
            args.stats_checkpoint_interval,
            "∞" if max_snapshots is None else str(max_snapshots),
            args.stats_checkpoint_samples,
        )

    plot_raw_signals(trajectories, time_grid, h_signal, metadata, output_directory)
    LOGGER.info(
        "Saved raw signal plot to %s",
        (output_directory / "nanorings_raw_signals.pdf").as_posix(),
    )
    if normalisation_stats is not None:
        plot_normalised_signals(trajectories, time_grid, h_signal, output_directory)

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
