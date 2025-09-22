from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, List, Final

import torch
import matplotlib.axes
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, NullFormatter
from torch import Tensor

from ..models.base import DriftNet
from examples.systems.registry import get_system_info
from neural_dynamics.core.hyperparameters import NetworkArchitecture
from neural_dynamics.config import DEVICE


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


class MinMaxScaler:
    """
    A PyTorch-based min-max scaler to normalise a tensor to a given range,
    typically [0, 1].
    """
    def __init__(self):
        self.min_val = None
        self.max_val = None

    def fit_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Fits the scaler to the data and returns the transformed tensor.
        
        Args:
            tensor: The input tensor to scale.
            
        Returns:
            The tensor scaled to the [0, 1] range.
        """
        self.min_val = torch.min(tensor)
        self.max_val = torch.max(tensor)
        # Add a small epsilon to avoid division by zero if max == min
        return (tensor - self.min_val) / (self.max_val - self.min_val + 1e-8)

    def inverse_transform(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse transformation to a scaled tensor.
        
        Args:
            tensor: The scaled tensor.
            F
        Returns:
            The tensor scaled back to its original range.
        """
        if self.min_val is None or self.max_val is None:
            raise RuntimeError("Scaler has not been fitted yet. Call fit_transform first.")
        return tensor * (self.max_val - self.min_val + 1e-8) + self.min_val
