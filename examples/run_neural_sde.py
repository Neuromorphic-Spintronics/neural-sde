"""
Generic neural SDE script for 1D second-order stochastic systems.

This script provides a framework for training neural SDEs on 1D second-order
stochastic differential equations. It uses a system registry (systems/registry.py)
to dynamically load system-specific configurations, allowing it to be used
for any implemented system. The training is based on a Wasserstein GAN with
gradient penalty (WGAN-GP) approach.

Usage:
    # Train a model for the Duffing oscillator (default system)
    python examples/run_neural_sde.py --train

    # Train a model for a different system
    python examples/run_neural_sde.py --system <system_name> --train

    # Load and simulate a trained model
    python examples/run_neural_sde.py --load path/to/model.pth --simulate
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import List, Tuple, Callable, Dict, Any
from dataclasses import dataclass
# import math
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import torch
from torch import Tensor, optim
# from tqdm import tqdm
import numpy as np

from models.neural_sde import DriftNet, DiffusionNet, CriticNet
from utils.plotting import style_axis_clean, set_default_plotting_style
# from parameters.hyperparameters import NetworkArchitecture
from models.integrators import (
    auto_select_integrator,
    #integrate_trajectory_with_step_method,
    #runge_kutta_4_step,
)
from config import DEVICE
from matplotlib.ticker import LogLocator, NullFormatter

from systems.registry import get_system_info, get_implemented_systems


@dataclass(frozen=True)
class NeuralSDETrainingConfig:
    """Configuration for neural SDE GAN training."""
    generator_learning_rate: float = 1e-4
    critic_learning_rate: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 100
    critic_steps: int = 5
    gradient_penalty_weight: float = 10.0

# ---------------------------------------------------------------------------- #
#                           Neural network components                          #
# ---------------------------------------------------------------------------- #
def build_neural_sde_components(
    *,
    system_name: str,
    generator_hidden_layer_sizes: list[int] | None = None,
    critic_hidden_layer_sizes: list[int] | None = None,
) -> Tuple[DriftNet, DiffusionNet, CriticNet]:
    """
    Construct the generator (DriftNet, DiffusionNet) and CriticNet for the
    specified system's neural SDE.
    """
    raise NotImplementedError("Function build_neural_sde_components is not implemented.")


def compute_critic_cost(
    critic_network: CriticNet,
    real_trajectories: Tensor,
    fake_trajectories: Tensor,
    gradient_penalty_weight: float,
) -> Tensor:
    """
    Calculate the critic's loss, which includes the Wasserstein distance
    and a gradient penalty.
    """
    raise NotImplementedError("Function compute_critic_cost is not implemented.")


def compute_generator_cost(
    critic_network: CriticNet,
    fake_trajectories: Tensor,
) -> Tensor:
    """
    Calculate the generator's loss, which aims to maximise the critic's score
    for fake trajectories.
    """
    raise NotImplementedError("Function compute_generator_cost is not implemented.")


def train_critic_step(
    critic_optimiser: optim.Optimizer,
    critic_network: CriticNet,
    real_trajectories: Tensor,
    fake_trajectories: Tensor,
    gradient_penalty_weight: float,
) -> None:
    """
    Perform a single training step for the critic network.
    """
    raise NotImplementedError("Function train_critic_step is not implemented.")


def train_generator_step(
    generator_optimiser: optim.Optimizer,
    critic_network: CriticNet,
    fake_trajectories: Tensor,
) -> None:
    """
    Perform a single training step for the generator networks (drift and diffusion).
    """
    raise NotImplementedError("Function train_generator_step is not implemented.")

# ---------------------------------------------------------------------------- #
#                     Get physical parameters of the system                    #
# ---------------------------------------------------------------------------- #
def get_physical_parameters(system_name: str, **kwargs) -> Any:
    """Get an instance of the physical parameters class for a given system."""
    system_info = get_system_info(system_name)
    ParameterClass = system_info["parameter_class"]
    return ParameterClass(**kwargs)

# ---------------------------------------------------------------------------- #
#                  Generate the training ("real") trajectories                 #
# ---------------------------------------------------------------------------- #
def _integrate_batch_sde_trajectories(
    system_name: str,
    dimensionless_parameters: Any,
    batch_size: int,
    time_grid: Tensor,
    dt: float,
    device: str | torch.device = DEVICE,
    random_seed: int | None = None,
) -> Tensor:
    """Integrate multiple SDE trajectories in parallel."""
    if random_seed is not None:
        torch.manual_seed(random_seed)

    system_info = get_system_info(system_name)
    SystemClass = system_info["system_class"]
    system = SystemClass(dimensionless_parameters)
    device_obj = torch.device(device)

    initial_states = system.get_initial_state().expand(batch_size, -1).to(device_obj)
    num_steps = len(time_grid) - 1
    trajectories = torch.zeros(
        batch_size, num_steps + 1, system.get_state_dimension(), device=device_obj
    )
    trajectories[:, 0] = initial_states
    current_states = initial_states.clone()

    step_integrator = auto_select_integrator(has_diffusion=system.has_noise)
    for step in range(num_steps):
        current_time = float(time_grid[step].item())
        if system.has_noise:
            next_states = step_integrator(
                system.drift_function,
                system.diffusion_function,
                current_states,
                current_time,
                dt,
            )
        else:
            next_states = step_integrator(
                system.drift_function, current_states, current_time, dt
            )
        trajectories[:, step + 1] = next_states
        current_states = next_states

    return trajectories


def generate_stochastic_dataset(
    *,
    system_name: str,
    num_trajectories: int,
    total_time: float,
    dt: float,
    seed: int = 42069,
    with_noise: bool = True,
) -> Tuple[Tensor, Tensor]:
    """Optimised trajectory generation for a given system."""
    torch.manual_seed(seed)
    system_info = get_system_info(system_name)
    ParameterClass = system_info["parameter_class"]
    physical_parameters = ParameterClass(total_time=total_time, timestep=dt)

    if not with_noise:
        print("   Generating deterministic dataset (noise turned off).")
        if hasattr(physical_parameters, "temperature"):
            object.__setattr__(physical_parameters, "temperature", 0.0)
        if hasattr(physical_parameters, "coloured_noise_intensity"):
            object.__setattr__(physical_parameters, "coloured_noise_intensity", 0.0)

    dimensionless_parameters = physical_parameters.to_dimensionless()
    num_steps = int(total_time / dt)
    time_grid = torch.linspace(0, total_time, num_steps + 1, device=DEVICE)

    print(f"Generating {num_trajectories} trajectories in a single batch...")
    all_trajectories = _integrate_batch_sde_trajectories(
        system_name,
        dimensionless_parameters,
        num_trajectories,  # Integrate all trajectories at once
        time_grid,
        dt,
    )
    print("Trajectory generation complete.")

    return time_grid, all_trajectories

# ---------------------------------------------------------------------------- #
#                                   Inference                                  #
# ---------------------------------------------------------------------------- #
def simulate_neural_sde_trajectory(
    drift_net: DriftNet,
    diffusion_net: DiffusionNet,
    drift_function: Callable[[float, Tensor], Tensor],
    diffusion_function: Callable[[float, Tensor], Tensor],
    *,
    x0: Tensor,
    t0: float,
    tN: float,
    dt: float,
) -> Tuple[Tensor, Tensor]:
    """Simulate trajectory using the trained neural SDE."""
    raise NotImplementedError("Function simulate_neural_sde_trajectory is not implemented.")


# ---------------------------------------------------------------------------- #
#                              Plotting functions                              #
# ---------------------------------------------------------------------------- #
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
    neural_sde_time: Tensor,
    neural_sde_trajectory: Tensor,
    *,
    system_name: str,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot comparison between stochastic trajectories and neural SDE prediction."""
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
    t_neural = neural_sde_time.detach().cpu().numpy() * tau
    trajectories = stochastic_trajectories.detach().cpu().numpy()
    neural_traj = neural_sde_trajectory.squeeze(1).detach().cpu().numpy()

    for i in range(num_core_states):
        ax = axes[i]
        stoch_phys = trajectories[:, :, i] * scales[i]
        neural_phys = neural_traj[:, i] * scales[i]

        for j in range(trajectories.shape[0]):
            label = "Stochastic realisations" if j == 0 else None
            ax.plot(t_stoch, stoch_phys[j, :], color="black", alpha=0.15, linewidth=0.6, label=label)

        ax.plot(t_neural, neural_phys, color="black", linewidth=1.4, label="Neural SDE")
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
    neural_sde_trajectory: Tensor,
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
    neural_traj = neural_sde_trajectory.squeeze(1).detach().cpu().numpy()

    pos_phys = trajectories[:, :, 0] * scales[0]
    vel_phys = trajectories[:, :, 1] * scales[1]
    neural_pos_phys = neural_traj[:, 0] * scales[0]
    neural_vel_phys = neural_traj[:, 1] * scales[1]

    for i in range(trajectories.shape[0]):
        label = "Stochastic trajectories" if i == 0 else None
        ax.plot(pos_phys[i, :], vel_phys[i, :], color="black", alpha=0.05, linewidth=0.8, label=label)

    ax.plot(neural_pos_phys, neural_vel_phys, color="black", linewidth=1.6, label="Neural SDE")

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

# ---------------------------------------------------------------------------- #
#                           CLI and helper functions                           #
# ---------------------------------------------------------------------------- #
def _generate_hyperparameter_string(
    hidden_layer_sizes: list[int],
    num_trajectories: int,
    num_epochs: int,
    learning_rate: float,
    with_noise: bool,
) -> str:
    """Generate a string representation of hyperparameters for directory naming."""
    arch_str = f"arch-{''.join(map(str, hidden_layer_sizes))}"
    traj_str = f"traj-{num_trajectories}"
    epoch_str = f"ep-{num_epochs}"
    lr_str = f"lr-{learning_rate:.1e}"
    noise_str = f"noise-{'yes' if with_noise else 'no'}"
    return f"{arch_str}_{traj_str}_{epoch_str}_{lr_str}_{noise_str}"


def load_trained_model(
    model_path: str | Path,
    system_name: str
) -> Tuple[DriftNet, DiffusionNet, CriticNet, Dict[str, Any]]:
    """Load a trained neural SDE model from disk."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading trained model from {model_path}")
    # checkpoint = torch.load(model_path, map_location=DEVICE)
    
    raise NotImplementedError("Function load_trained_model is not implemented.")


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
    print(f"   Hidden layers: {arch.get('hidden_layer_sizes', 'N/A')}")
    print(f"   Output size: {arch.get('output_size', 'N/A')}")

    print("TRAINING CONFIG:")
    print(f"   Trajectories: {config.get('num_trajectories', 'N/A')}")
    print(f"   Time span:    {config.get('total_time', 'N/A')} dimensionless units")
    print(f"   Timestep:     {config.get('timestep', 'N/A')}")
    print(f"   Epochs:       {config.get('num_epochs', 'N/A')}")
    print(f"   Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"   Initial loss: {config.get('initial_loss', 'N/A'):.2e}")
    print(f"   Final loss:   {config.get('final_loss', 'N/A'):.2e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Neural SDE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--system",
        type=str,
        default="duffing",
        choices=get_implemented_systems(),
        help="The system to model (default: duffing)",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--load",
        type=str,
        metavar="MODEL_PATH",
        help="Load and evaluate a trained model from the specified path",
    )
    mode_group.add_argument(
        "--train",
        action="store_true",
        help="Train a new model",
    )

    sim_group = parser.add_argument_group("Simulation Parameters (used with --load)")
    sim_group.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation with a loaded model.",
    )
    sim_group.add_argument(
        "--initial-state",
        nargs='+',
        type=float,
        default=None,
        help="Initial conditions for simulation (e.g., 0.1 0.0). Uses system default if not provided.",
    )
    
    training_group = parser.add_argument_group("Training Parameters (used with --train)")
    training_group.add_argument("--trajectories", type=int, default=256)
    training_group.add_argument("--epochs", type=int, default=512)
    training_group.add_argument("--generator-learning-rate", type=float, default=1e-4)
    training_group.add_argument("--critic-learning-rate", type=float, default=1e-4)
    training_group.add_argument("--critic-steps", type=int, default=5)
    training_group.add_argument("--gradient-penalty-weight", type=float, default=10.0)
    training_group.add_argument("--generator-architecture", nargs="+", type=int, default=None)
    training_group.add_argument("--critic-architecture", nargs="+", type=int, default=None)
    training_group.add_argument("--no-noise", action="store_true")
    training_group.add_argument("--training-batch-size", type=int, default=64)
    
    general_group = parser.add_argument_group("General Parameters")
    general_group.add_argument("--time", type=float, default=25.0, help="Total simulation time (dimensionless)")
    general_group.add_argument("--timestep", type=float, default=0.01, help="Integration timestep (dimensionless)")
    general_group.add_argument("--show-plots", action="store_true", help="Display plots interactively")
    general_group.add_argument("--save-dir", type=str, default="examples", help="Base directory to save all outputs")

    return parser

# ---------------------------------------------------------------------------- #
#                               Train the WGAN-GP                              #
# ---------------------------------------------------------------------------- #
def fit_neural_sde_gan(
    drift_network: DriftNet,
    diffusion_network: DiffusionNet,
    critic_network: CriticNet,
    time_grid: Tensor,
    stochastic_trajectories: Tensor,
    *,
    system_name: str,
    config: NeuralSDETrainingConfig,
    random_seed: int = 0,
) -> Tuple[List[float], List[float]]:
    """
    Train the Neural SDE GAN using the WGAN-GP algorithm.
    """
    torch.manual_seed(random_seed)
    raise NotImplementedError("Function fit_neural_sde_gan is not implemented.")


def run_training_mode(args: argparse.Namespace):
    """Run the full training and evaluation pipeline."""
    system_name = args.system
    system_info = get_system_info(system_name)
    gen_hidden_sizes = args.generator_architecture or system_info["default_hidden_layers"]
    crit_hidden_sizes = args.critic_architecture or system_info["default_hidden_layers"]

    hyperparam_string = _generate_hyperparameter_string(
        hidden_layer_sizes=gen_hidden_sizes,
        num_trajectories=args.trajectories,
        num_epochs=args.epochs,
        learning_rate=args.generator_learning_rate,
        with_noise=not args.no_noise,
    )

    base_save_dir = Path(args.save_dir)
    figure_dir = base_save_dir / "figures" / system_name / hyperparam_string
    model_dir = base_save_dir / "models" / system_name / hyperparam_string
    data_dir = base_save_dir / "data" / system_name / hyperparam_string
    figure_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    print(f"--- Training {system_name} ---")
    print(f"Results will be saved in: {base_save_dir}/<type>/{system_name}/{hyperparam_string}")

    time_grid, stochastic_trajectories = generate_stochastic_dataset(
        system_name=system_name,
        num_trajectories=args.trajectories,
        total_time=args.time,
        dt=args.timestep,
        with_noise=not args.no_noise,
    )

    drift_network, diffusion_network, critic_network = build_neural_sde_components(
        system_name=system_name,
        generator_hidden_layer_sizes=gen_hidden_sizes,
        critic_hidden_layer_sizes=crit_hidden_sizes,
    )

    training_config = NeuralSDETrainingConfig(
        generator_learning_rate=args.generator_learning_rate,
        critic_learning_rate=args.critic_learning_rate,
        batch_size=args.training_batch_size,
        num_epochs=args.epochs,
        critic_steps=args.critic_steps,
        gradient_penalty_weight=args.gradient_penalty_weight,
    )

    critic_losses, generator_losses = fit_neural_sde_gan(
        drift_network,
        diffusion_network,
        critic_network,
        time_grid,
        stochastic_trajectories,
        system_name=system_name,
        config=training_config,
    )

    initial_state = stochastic_trajectories[0, 0, :system_info["state_dimension"]]
    
    # system = system_info["system_class"](get_physical_parameters(system_name).to_dimensionless())

    neural_time, neural_trajectory = simulate_neural_sde_trajectory(
        drift_network,
        diffusion_network,
        drift_function=lambda t, y: drift_network(torch.cat([y, t.expand(y.size(0), 1)], dim=1)),
        diffusion_function=lambda t, y: diffusion_network(torch.cat([y, t.expand(y.size(0), 1)], dim=1)),
        x0=initial_state,
        t0=time_grid[0].item(),
        tN=time_grid[-1].item(),
        dt=args.timestep,
    )

    print("--- Generating Visualisations ---")
    plot_training_loss(
        critic_losses, save_path=figure_dir / f"{system_name}_critic_loss.pdf", show_plot=args.show_plots
    )
    plot_training_loss(
        generator_losses, save_path=figure_dir / f"{system_name}_generator_loss.pdf", show_plot=args.show_plots
    )
    plot_trajectory_comparison(
        time_grid, stochastic_trajectories, neural_time, neural_trajectory,
        system_name=system_name, save_path=figure_dir / f"{system_name}_trajectory_comparison.pdf", show_plot=args.show_plots
    )
    plot_phase_space_comparison(
        stochastic_trajectories, neural_trajectory,
        system_name=system_name, save_path=figure_dir / f"{system_name}_phase_space.pdf", show_plot=args.show_plots
    )

    print("--- Saving Model and Results ---")
    model_path = model_dir / f"{system_name}_neural_sde.pth"
    torch.save(
        {
            "drift_net_state_dict": drift_network.state_dict(),
            "diffusion_net_state_dict": diffusion_network.state_dict(),
            "critic_net_state_dict": critic_network.state_dict(),
            "model_architecture": {
                "generator_hidden_layer_sizes": gen_hidden_sizes,
                "critic_hidden_layer_sizes": crit_hidden_sizes,
                "system_name": system_name,
            },
            "training_config": vars(training_config),
            "critic_losses": critic_losses,
            "generator_losses": generator_losses,
        },
        model_path,
    )
    print(f"  Model saved to {model_path}")

    history_path = model_dir / f"{system_name}_training_history.txt"
    with open(history_path, "w") as f:
        f.write(f"Training History for {system_name} ({hyperparam_string})")
        f.write(json.dumps(vars(args), indent=4))
        f.write("Critic Loss per Epoch:")
        for epoch, loss in enumerate(critic_losses, 1):
            f.write(f"Epoch {epoch:3d}: {loss:.6e}")
        f.write("Generator Loss per Epoch:")
        for epoch, loss in enumerate(generator_losses, 1):
            f.write(f"Epoch {epoch:3d}: {loss:.6e}")
    print(f"  History saved to {history_path}")
    
    print("--- Training Complete ---")

# ---------------------------------------------------------------------------- #
#                           Load the model (optional)                          #
# ---------------------------------------------------------------------------- #
def run_loading_mode(args: argparse.Namespace):
    """Run the loading and simulation pipeline."""
    print(f"--- Loading {args.system} model from {args.load} ---")
    drift_net, diffusion_net, critic_net, info = load_trained_model(args.load, system_name=args.system)
    print_model_summary(info)

    if args.simulate:
        print("--- Simulating Trajectory ---")
        system_info = get_system_info(args.system)
        
        if args.initial_state:
            if len(args.initial_state) != system_info["state_dimension"]:
                raise ValueError(f"Initial state for {args.system} requires {system_info['state_dimension']} values, but got {len(args.initial_state)}")
            initial_state = torch.tensor(args.initial_state, device=DEVICE, dtype=torch.float32)
        else:
            phys_params = get_physical_parameters(args.system)
            dimless_params = phys_params.to_dimensionless()
            initial_state = torch.tensor([
                dimless_params.initial_position, dimless_params.initial_velocity
            ], device=DEVICE, dtype=torch.float32)

        print(f"Simulating from initial state: {initial_state.tolist()}")
        
        time_grid, trajectory = simulate_neural_sde_trajectory(
            drift_net,
            diffusion_net,
            drift_function=lambda t, y: drift_net(torch.cat([y, t.expand(y.size(0), 1)], dim=1)),
            diffusion_function=lambda t, y: diffusion_net(torch.cat([y, t.expand(y.size(0), 1)], dim=1)),
            x0=initial_state,
            t0=0.0,
            tN=args.time,
            dt=args.timestep,
        )
        print(f"Simulation complete. Trajectory shape: {trajectory.shape}")

        if args.show_plots or args.save_dir:
            save_path = Path(args.save_dir) / "figures" / f"{args.system}_loaded_simulation.pdf"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            dummy_stoch = trajectory.squeeze(1).unsqueeze(0)
            plot_trajectory_comparison(
                time_grid, dummy_stoch, time_grid, trajectory,
                system_name=args.system, save_path=save_path, show_plot=args.show_plots
            )

    print("--- Loading Complete ---")

# ---------------------------------------------------------------------------- #
#                                  Entry point                                 #
# ---------------------------------------------------------------------------- #
def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if args.load:
        run_loading_mode(args)
    elif args.train:
        run_training_mode(args)


if __name__ == "__main__":
    main()
