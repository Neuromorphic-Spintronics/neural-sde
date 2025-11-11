from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, Tuple, List, Final, Optional, Callable, Mapping, Sequence

import torch
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter
from torch import Tensor
from torch.nn import Module

from ..models.base import DriftNet
from examples.systems.registry import get_system_info
from neural_dynamics.core.hyperparameters import NetworkArchitecture
from neural_dynamics.config import DEVICE


def generate_output_directory_name(
    base_path: str, 
    hyperparameters, 
    prefix: str = ""
) -> str:
    """
    Generate a descriptive output directory name based on hyperparameters.
    
    Args:
        base_path: Base directory path
        hyperparameters: Hyperparameters object
        prefix: Optional prefix for the directory name (typically system name)
        
    Returns:
        Full path to the output directory
    """
    import hashlib
    import json
    from dataclasses import asdict
    
    # Create a hash of the hyperparameters for uniqueness
    hyper_dict = asdict(hyperparameters)
    # Sort keys for consistent hashing
    hyper_json = json.dumps(hyper_dict, sort_keys=True, default=str)
    hyper_hash = hashlib.sha256(hyper_json.encode("utf-8")).hexdigest()[:8]
    
    # Use prefix (system name) + hash
    directory_name = f"{prefix}_{hyper_hash}"
    return f"{base_path}/{directory_name}"


def _generate_hyperparameter_string(
    hidden_layer_sizes: list[int],
    num_trajectories: int,
    num_epochs: int,
    learning_rate: float,
    with_noise: bool,
) -> str:
    """Generate a string representation of hyperparameters for directory naming."""
    arch_str = f"arch-{'-'.join(map(str, hidden_layer_sizes))}"
    traj_str = f"traj-{num_trajectories}"
    epoch_str = f"ep-{num_epochs}"
    lr_str = f"lr-{learning_rate:.1e}"
    noise_str = f"noise-{'yes' if with_noise else 'no'}"
    return f"{arch_str}_{traj_str}_{epoch_str}_{lr_str}_{noise_str}"


def load_trained_model(
    model_path: str | Path,
    system_name: str
) -> Tuple[DriftNet, Dict[str, Any]]:
    """Load a trained neural ODE model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading trained model from {model_path}")
    checkpoint = torch.load(model_path, map_location=DEVICE)
    
    arch_info = checkpoint["model_architecture"]
    # Be tolerant to possible key naming differences
    hidden_layers = arch_info.get("hidden_layer_sizes") or arch_info.get("hidden_layers")
    if hidden_layers is None:
        raise KeyError("Checkpoint missing 'hidden_layer_sizes'/'hidden_layers' in 'model_architecture'.")

    loaded_system = arch_info.get("system_name")
    if loaded_system and loaded_system != system_name:
        raise ValueError(f"Model was trained on '{loaded_system}', but trying to load for '{system_name}'.")

    # Build a DriftNet compatible with the saved architecture. If input/output
    # sizes are present in the checkpoint, prefer them; otherwise infer from the
    # system registry (state_dimension + time input -> output = state_dimension).
    system_info = get_system_info(system_name)
    state_dim = int(system_info["state_dimension"])  # core state dimension
    input_size = int(arch_info.get("input_size", state_dim + 1))
    output_size = int(arch_info.get("output_size", state_dim))

    architecture = NetworkArchitecture(
        input_size=input_size,
        hidden_sizes=list(map(int, hidden_layers)),
        output_size=output_size,
    )
    model = DriftNet(architecture)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    info = {k: v for k, v in checkpoint.items() if k != 'model_state_dict'}
    
    return model, info


def print_model_summary(model_info: Dict[str, Any]):
    """Print a detailed summary of the loaded model."""
    print("" + "=" * 50)
    print("MODEL SUMMARY")
    print("=" * 50)

    arch = model_info.get("model_architecture", {})
    config = model_info.get("training_config", {})

    print(f"System: {arch.get('system_name', 'N/A')}")
    print("🏗️  ARCHITECTURE:")
    print(f"   Input size:  {arch.get('input_size', 'N/A')}")
    hidden_layers = arch.get('hidden_layers') or arch.get('hidden_layer_sizes')
    print(f"   Hidden layers: {hidden_layers if hidden_layers is not None else 'N/A'}")
    print(f"   Output size: {arch.get('output_size', 'N/A')}")

    print("TRAINING CONFIG:")
    print(f"   Trajectories: {config.get('num_trajectories', 'N/A')}")
    print(f"   Time span:    {config.get('total_time', 'N/A')} dimensionless units")
    print(f"   Timestep:     {config.get('timestep', 'N/A')}")
    print(f"   Epochs:       {config.get('num_epochs', 'N/A')}")
    print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"   Initial loss: {config.get('initial_loss', 'N/A'):.2e}")
    print(f"   Final loss:   {config.get('final_loss', 'N/A'):.2e}")


def get_physical_parameters(system_name: str, **kwargs) -> Any:
    """Get an instance of the physical parameters class for a given system."""
    system_info = get_system_info(system_name)
    ParameterClass = system_info["parameter_class"]
    return ParameterClass(**kwargs)


COLOURS: Final[list[str]] = [
    "#ffbe0b",
    "#fb5607",
    "#ff006e",
    "#8338ec",
    "#390099",
    "#61E8E1",
    "#00D4FF"
]


def set_default_plotting_style(use_tex: bool = True) -> None:
    """Set default matplotlib style parameters."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "DejaVu Serif"],
            "font.size": 11,
            "axes.labelsize": 14,
            "axes.titlesize": 14,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "legend.fontsize": 14,
            "figure.titlesize": 14,
            "text.usetex": use_tex,
            "mathtext.fontset": "custom",
            "mathtext.rm": "Times",
            "mathtext.it": "Times:italic",
            "mathtext.bf": "Times:bold",
            "mathtext.default": "it",
        }
    )


def set_symmetric_three_ticks(ax, data, axis="y"):
    import numpy as np

    max_val = np.nanmax(np.abs(data))
    if max_val == 0:
        ticks = [-1, 0, 1]
    else:
        tick_val = np.ceil(max_val * 2.0) / 2.0
        ticks = [-tick_val, 0.0, tick_val]

    if axis == "y":
        ax.set_ylim(ticks[0], ticks[-1])
        ax.set_yticks(ticks)
    elif axis == "x":
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_xticks(ticks)
    elif axis == "both":
        ax.set_xlim(ticks[0], ticks[-1])
        ax.set_xticks(ticks)
        ax.set_ylim(ticks[0], ticks[-1])
        ax.set_yticks(ticks)


@dataclass
class RolloutBatch:
    """Rollout trajectories sampled from a dynamical model."""

    rollouts: list[Tensor]
    rollout_tensor: Tensor
    sample_indices: Tensor
    training_subset: Tensor


@dataclass
class TrajectoryStatistics:
    """Mean and variance statistics computed from a batch of trajectories."""

    mean: Tensor
    variance: Tensor


def compute_statistics(
    trajectories: Tensor,
    *,
    dim: int = 0,
    unbiased: bool = False,
) -> TrajectoryStatistics:
    """Compute basic statistics for a batch of trajectories.

    Args:
        trajectories: Tensor containing trajectories. The reduction dimension is
            controlled by ``dim`` and defaults to the batch axis.
        dim: Dimension along which the statistics are computed.
        unbiased: Whether to use the unbiased estimator for the variance.

    Returns:
        TrajectoryStatistics with mean and variance tensors.
    """

    if trajectories.numel() == 0:
        raise ValueError("Cannot compute statistics for an empty tensor.")

    mean = torch.mean(trajectories, dim=dim)
    variance = torch.var(trajectories, dim=dim, unbiased=unbiased)
    return TrajectoryStatistics(mean=mean, variance=variance)


def compare_statistics(
    training: TrajectoryStatistics,
    generated: TrajectoryStatistics,
    *,
    reduction: str = "mean",
) -> dict[str, Any]:
    """Compare two sets of trajectory statistics.

    Args:
        training: Reference statistics, typically computed from training data.
        generated: Statistics computed from generated trajectories.
        reduction: Reduction to apply to the absolute differences. Supported
            options are ``"mean"``, ``"max"``, and ``"none"``.

    Returns:
        Dictionary containing absolute differences of the statistics. The
        returned values are floats when a reduction is applied, or tensors when
        ``reduction`` is set to ``"none"``.
    """

    if training.mean.shape != generated.mean.shape:
        raise ValueError("Mean tensors must share the same shape for comparison.")
    if training.variance.shape != generated.variance.shape:
        raise ValueError(
            "Variance tensors must share the same shape for comparison."
        )

    mean_diff = torch.abs(training.mean - generated.mean)
    var_diff = torch.abs(training.variance - generated.variance)

    if reduction == "none":
        return {
            "mean_abs_difference": mean_diff,
            "variance_abs_difference": var_diff,
        }

    if reduction == "mean":
        reducer = torch.mean
    elif reduction == "max":
        reducer = torch.max
    else:
        raise ValueError(
            "Reduction must be one of {'mean', 'max', 'none'}, "
            f"received '{reduction}'."
        )

    return {
        "mean_abs_difference": reducer(mean_diff).item(),
        "variance_abs_difference": reducer(var_diff).item(),
    }


def sample_sde_rollouts(
    *,
    model: Callable[[Tensor, Tensor], Tensor],
    trajectories: Tensor,
    time_grid: Tensor,
    num_samples: int,
    device: torch.device = DEVICE,
) -> RolloutBatch:
    """Sample rollouts from a dynamical model for comparison with data.

    Args:
        model: Callable that evolves a batch of initial states along ``time_grid``.
        trajectories: Reference trajectories used to select initial conditions.
        time_grid: Time grid for integration.
        num_samples: Maximum number of rollouts to generate.
        device: Device on which sampling indices are generated.

    Returns:
        RolloutBatch containing the generated trajectories and bookkeeping
        information.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if trajectories.dim() < 3:
        raise ValueError(
            "trajectories tensor must have shape (batch, time, features)."
        )

    dataset_size = trajectories.shape[0]
    if dataset_size == 0:
        raise ValueError("Cannot sample rollouts without reference trajectories.")

    actual_samples = min(num_samples, dataset_size)
    if actual_samples == dataset_size:
        subset_indices = torch.arange(dataset_size, device=device)
    else:
        subset_indices = torch.randperm(dataset_size, device=device)[:actual_samples]

    initial_state_batch = trajectories[subset_indices, 0, :]

    with torch.no_grad():
        rollout_tensor = model(initial_state_batch, time_grid)

    training_subset = trajectories[subset_indices].detach().cpu()
    rollout_cpu = rollout_tensor.detach().cpu()

    return RolloutBatch(
        rollouts=[rollout_cpu[i] for i in range(rollout_cpu.size(0))],
        rollout_tensor=rollout_cpu,
        sample_indices=subset_indices.detach().cpu(),
        training_subset=training_subset,
    )


def sample_sde_rollouts_with_inputs(
    *,
    model: Callable,
    trajectories: Tensor,
    external_inputs: Tensor,
    time_grid: Tensor,
    num_samples: int,
    device: torch.device = DEVICE,
) -> RolloutBatch:
    """Sample rollouts from a dynamical model with external inputs.
    
    This version supports models that require exogenous signals (e.g., H field,
    forcing functions, time features) by passing them through to the model.

    Args:
        model: Callable that evolves states with external inputs.
               Expected signature: model(external_inputs, initial_state=..., initial_time=...)
        trajectories: Reference trajectories [batch, time, state_dim].
        external_inputs: Exogenous signals [batch, input_dim, time].
        time_grid: Time grid for integration [num_steps].
        num_samples: Maximum number of rollouts to generate.
        device: Device on which sampling indices are generated.

    Returns:
        RolloutBatch containing the generated trajectories and bookkeeping.
    """

    if num_samples <= 0:
        raise ValueError("num_samples must be a positive integer.")
    if trajectories.dim() != 3:
        raise ValueError(
            "trajectories tensor must have shape (batch, time, state_dim)."
        )
    if external_inputs.dim() != 3:
        raise ValueError(
            "external_inputs tensor must have shape (batch, input_dim, time)."
        )
    if trajectories.shape[0] != external_inputs.shape[0]:
        raise ValueError(
            "trajectories and external_inputs must have the same batch size."
        )
    if trajectories.shape[1] != external_inputs.shape[2]:
        raise ValueError(
            "trajectories time dimension must match external_inputs time dimension."
        )

    dataset_size = trajectories.shape[0]
    if dataset_size == 0:
        raise ValueError("Cannot sample rollouts without reference trajectories.")

    actual_samples = min(num_samples, dataset_size)
    if actual_samples == dataset_size:
        subset_indices = torch.arange(dataset_size, device=trajectories.device)
    else:
        subset_indices = torch.randperm(dataset_size, device=trajectories.device)[:actual_samples]

    initial_state_batch = trajectories[subset_indices, 0, :]
    external_inputs_batch = external_inputs[subset_indices]

    initial_time = float(time_grid[0].item())

    with torch.no_grad():
        rollout_tensor = model(
            external_inputs_batch,
            initial_state=initial_state_batch,
            initial_time=initial_time,
        )

    training_subset = trajectories[subset_indices].detach().cpu()
    rollout_cpu = rollout_tensor.detach().cpu()

    # Permute from [batch, state_dim, time] to [batch, time, state_dim]
    if rollout_cpu.shape[1] != training_subset.shape[1]:
        rollout_cpu = rollout_cpu.permute(0, 2, 1)

    return RolloutBatch(
        rollouts=[rollout_cpu[i] for i in range(rollout_cpu.size(0))],
        rollout_tensor=rollout_cpu,
        sample_indices=subset_indices.detach().cpu(),
        training_subset=training_subset,
    )


def save_statistics(
    *,
    output_dir: Path | str,
    training: TrajectoryStatistics,
    generated: TrajectoryStatistics,
    dimension_labels: Optional[Sequence[str]] = None,
    training_filename: str = "training_statistics.txt",
    generated_filename: str = "generated_statistics.txt",
) -> None:
    """Persist trajectory statistics to disk in a tabular format.

    Args:
        output_dir: Directory where the statistics files are written.
        training: Statistics computed from the reference data.
        generated: Statistics computed from generated trajectories.
        dimension_labels: Optional labels for the dynamical dimensions used to
            annotate the saved files.
        training_filename: Output filename for the training statistics.
        generated_filename: Output filename for the generated statistics.
    """

    if training.mean.shape != generated.mean.shape:
        raise ValueError("Training and generated means must share the same shape.")
    if training.variance.shape != generated.variance.shape:
        raise ValueError(
            "Training and generated variances must share the same shape."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    def _prepare_stats(stats: TrajectoryStatistics) -> Tensor:
        mean_tensor = stats.mean.detach().cpu()
        var_tensor = stats.variance.detach().cpu()
        if mean_tensor.dim() == 1:
            mean_tensor = mean_tensor.unsqueeze(0)
        if var_tensor.dim() == 1:
            var_tensor = var_tensor.unsqueeze(0)
        return torch.cat([mean_tensor, var_tensor], dim=-1)

    prepared_training = _prepare_stats(training)
    prepared_generated = _prepare_stats(generated)

    num_dimensions = prepared_training.shape[-1] // 2
    if dimension_labels is None:
        labels = [f"dim{index}" for index in range(num_dimensions)]
    else:
        labels = list(dimension_labels)
        if len(labels) != num_dimensions:
            raise ValueError(
                "dimension_labels length must match the number of state dimensions."
            )

    header = ",".join(
        [f"mean_{label}" for label in labels]
        + [f"variance_{label}" for label in labels]
    )

    np.savetxt(
        output_path / training_filename,
        prepared_training.numpy(),
        header=header,
    )
    np.savetxt(
        output_path / generated_filename,
        prepared_generated.numpy(),
        header=header,
    )


def save_model_checkpoints(
    output_dir: Path | str,
    *,
    models: Mapping[str, Module],
    model_filenames: Mapping[str, str],
    metrics: Optional[Mapping[str, Sequence[float]]] = None,
    metric_filenames: Optional[Mapping[str, str]] = None,
) -> None:
    """Save model state dictionaries and scalar histories to disk.

    Args:
        output_dir: Directory used to store the checkpoints.
        models: Mapping of logical names to models whose ``state_dict`` is saved.
        model_filenames: Mapping of model names to filenames.
        metrics: Optional mapping of metric names to sequences of scalar values.
        metric_filenames: Mapping of metric names to filenames.
    """

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for name, model in models.items():
        if name not in model_filenames:
            raise KeyError(f"Missing filename for model '{name}'.")
        torch.save(model.state_dict(), output_path / model_filenames[name])

    if metrics is None:
        return

    if metric_filenames is None:
        raise ValueError("metric_filenames must be provided when metrics are saved.")

    for name, values in metrics.items():
        if name not in metric_filenames:
            raise KeyError(f"Missing filename for metric '{name}'.")
        torch.save(list(values), output_path / metric_filenames[name])


def load_model_checkpoints(
    output_dir: Path | str,
    *,
    model_factories: Mapping[str, Callable[[], Module]],
    model_filenames: Mapping[str, str],
    metric_filenames: Optional[Mapping[str, str]] = None,
    device: torch.device = DEVICE,
) -> Optional[tuple[dict[str, Module], dict[str, list[float]]]]:
    """Load model checkpoints and associated scalar histories if available.

    Args:
        output_dir: Directory that potentially contains the checkpoints.
        model_factories: Mapping from names to zero-argument callables that
            construct models before loading their state dictionaries.
        model_filenames: Mapping from model names to filenames.
        metric_filenames: Optional mapping of metric names to filenames.
        device: Device used when loading the checkpoints.

    Returns:
        ``None`` if any required file is missing. Otherwise, a tuple containing a
        mapping of model names to loaded models and a mapping of metric names to
        lists of scalar values.
    """

    output_path = Path(output_dir)

    required_files = [output_path / model_filenames[name] for name in model_factories]
    if metric_filenames is not None:
        required_files.extend(output_path / fname for fname in metric_filenames.values())

    if not all(path.exists() for path in required_files):
        return None

    models: dict[str, Module] = {}
    for name, factory in model_factories.items():
        if name not in model_filenames:
            raise KeyError(f"Missing filename for model '{name}'.")
        model = factory()
        state_dict = torch.load(
            output_path / model_filenames[name],
            map_location=device,
        )
        model.load_state_dict(state_dict)
        model.to(device)
        models[name] = model

    metrics: dict[str, list[float]] = {}
    if metric_filenames is not None:
        for name, filename in metric_filenames.items():
            metrics[name] = list(torch.load(output_path / filename))

    return models, metrics



def style_axis_clean(ax: matplotlib.axes.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def plot_training_loss(
    training_losses: List[float],
    *,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot training loss evolution."""
    set_default_plotting_style(use_tex=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    epochs = range(1, len(training_losses) + 1)
    ax.plot(epochs, training_losses, color="black", linewidth=1.4)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_yscale("log")

    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=12))
    ax.yaxis.set_minor_locator(
        LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=12)
    )
    ax.yaxis.set_minor_formatter(NullFormatter())
    style_axis_clean(ax)

    final_loss = training_losses[-1]
    print(f"Final loss: {final_loss:.2e}")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    if show_plot:
        plt.show()
    plt.close()


def plot_trajectory_comparison(
    time_grid: Tensor,
    stochastic_trajectories: Tensor,
    neural_ode_time: Tensor,
    neural_ode_trajectory: Tensor,
    *,
    system_name: str,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot comparison between stochastic trajectories and neural ODE prediction."""
    set_default_plotting_style(use_tex=True)
    system_info = get_system_info(system_name)
    state_vars = system_info["state_variables"]
    state_units = system_info["state_units"]
    num_core_states = system_info["state_dimension"]

    fig, axes = plt.subplots(num_core_states, 1, figsize=(7, 2.5 * num_core_states), sharex=True)
    if num_core_states == 1:
        axes = [axes]

    physical_params = get_physical_parameters(system_name)
    physical_params.to_dimensionless()
    tau = getattr(physical_params, "characteristic_time", 1.0)
    lambda_ = getattr(physical_params, "characteristic_length", 1.0)
    scales = [lambda_, lambda_ / tau]  # Position, Velocity - extend if more states

    t_stoch = time_grid.detach().cpu().numpy() * tau
    t_neural = neural_ode_time.detach().cpu().numpy() * tau
    trajectories = stochastic_trajectories.detach().cpu().numpy()
    neural_traj = neural_ode_trajectory.squeeze(1).detach().cpu().numpy()

    for i in range(num_core_states):
        ax = axes[i]
        stoch_phys = trajectories[:, :, i] * scales[i]
        neural_phys = neural_traj[:, i] * scales[i]

        for j in range(trajectories.shape[0]):
            label = "Stochastic realisations" if j == 0 else None
            ax.plot(t_stoch, stoch_phys[j, :], color="black", alpha=0.15, linewidth=0.6, label=label)

        ax.plot(t_neural, neural_phys, color="black", linewidth=1.4, label="Neural ODE")
        ax.set_ylabel(f"${state_vars[i]}(t)$ [{state_units[i]}]")
        ax.legend(frameon=False)
        style_axis_clean(ax)

    axes[-1].set_xlabel(r"$t$ [s]")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    if show_plot:
        plt.show()
    plt.close()


def plot_phase_space_comparison(
    stochastic_trajectories: Tensor,
    neural_ode_trajectory: Tensor,
    *,
    system_name: str,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot phase space (position vs velocity) comparison."""
    system_info = get_system_info(system_name)
    if system_info["state_dimension"] < 2:
        print("Phase space plot requires at least 2 dimensions.")
        return

    set_default_plotting_style(use_tex=True)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    physical_params = get_physical_parameters(system_name)
    physical_params.to_dimensionless()
    lambda_ = getattr(physical_params, "characteristic_length", 1.0)
    tau = getattr(physical_params, "characteristic_time", 1.0)
    scales = [lambda_, lambda_ / tau]

    trajectories = stochastic_trajectories.detach().cpu().numpy()
    neural_traj = neural_ode_trajectory.squeeze(1).detach().cpu().numpy()

    pos_phys = trajectories[:, :, 0] * scales[0]
    vel_phys = trajectories[:, :, 1] * scales[1]
    neural_pos_phys = neural_traj[:, 0] * scales[0]
    neural_vel_phys = neural_traj[:, 1] * scales[1]

    for i in range(trajectories.shape[0]):
        label = "Stochastic trajectories" if i == 0 else None
        ax.plot(pos_phys[i, :], vel_phys[i, :], color="black", alpha=0.05, linewidth=0.8, label=label)

    ax.plot(neural_pos_phys, neural_vel_phys, color="black", linewidth=1.6, label="Neural ODE")

    state_vars = system_info["state_variables"]
    state_units = system_info["state_units"]
    ax.set_xlabel(f"${state_vars[0]}(t)$ [{state_units[0]}]")
    ax.set_ylabel(f"${state_vars[1]}({state_vars[0]}(t))$ [{state_units[1]}]")
    ax.legend(frameon=False)
    ax.set_aspect("equal")
    style_axis_clean(ax)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Plot saved to {save_path}")
    if show_plot:
        plt.show()
    plt.close()