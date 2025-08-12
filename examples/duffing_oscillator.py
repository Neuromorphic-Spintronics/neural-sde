r"""
Neural ODE for the Duffing oscillator using the neural ODE template.

This example demonstrates a minimal implementation of the Duffing oscillator neural ODE by using the functions provided in neural_ode_template.py.

Typical usage:
    >>> from examples.duffing_oscillator import create_duffing_neural_ode_demonstration
    >>> trained_model, losses, benchmarks = create_duffing_neural_ode_demonstration()
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any, cast

from .neural_ode_template import (
    create_neural_ode_demonstration,
    build_neural_ode_drift,
    simulate_neural_ode_trajectory,
    plot_trajectory_comparison,
    generate_stochastic_dataset,
    NeuralODETrainingConfig,
)
from models.neural_sde import DriftNet
import torch
from torch import Tensor

SYSTEM_NAME = "duffing"


def build_duffing_neural_ode_drift(
    *,
    hidden_layer_sizes: list[int] | None = None,
) -> DriftNet:
    """Build a neural ODE drift network for the Duffing oscillator."""
    return build_neural_ode_drift(
        hidden_layer_sizes=hidden_layer_sizes,
        system_name=SYSTEM_NAME,
    )


def generate_stochastic_duffing_dataset(
    *,
    num_trajectories: int = 16,
    total_time: float | None = None,
    dt: float | None = None,
    seed: int = 42069,
) -> Tuple[Tensor, Tensor]:
    """Generate stochastic Duffing dataset using template function."""
    return generate_stochastic_dataset(
        num_trajectories=num_trajectories,
        total_time=total_time,
        dt=dt,
        seed=seed,
    )


def simulate_duffing_neural_ode_trajectory(
    drift_net: DriftNet,
    *,
    x0: Tensor,
    t0: float,
    tN: float,
    dt: float,
    gamma: float | None = None,
    Omega: float | None = None,
) -> Tuple[Tensor, Tensor]:
    """Simulate Duffing neural ODE trajectory using template function."""
    return simulate_neural_ode_trajectory(
        drift_net,
        x0=x0,
        t0=t0,
        tN=tN,
        dt=dt,
        amplitude=gamma,
        frequency=Omega,
    )


def create_duffing_neural_ode_demonstration(
    **kwargs,
) -> Tuple[DriftNet, List[float], Dict[str, Any]]:
    """Create Duffing neural ODE demonstration using the template."""
    kwargs.setdefault("save_directory", "examples/figures")
    kwargs.setdefault("system_name", SYSTEM_NAME)
    return create_neural_ode_demonstration(**kwargs)


def plot_duffing_trajectories_vs_neural_ode(
    time_grid: Tensor,
    stochastic_trajectories: Tensor,
    neural_ode_time: Tensor,
    neural_ode_trajectory: Tensor,
    *,
    save_path: str | Path | None = None,
    show_plot: bool = False,
) -> None:
    """Plot Duffing trajectories with proper units and labels."""
    return plot_trajectory_comparison(
        time_grid,
        stochastic_trajectories,
        neural_ode_time,
        neural_ode_trajectory,
        save_path=save_path,
        show_plot=show_plot,
    )


def fit_neural_ode_to_duffing_mean(
    drift_net: DriftNet,
    *,
    time_grid: Tensor,
    stochastic_trajectories_with_noise: Tensor,
    training_config: NeuralODETrainingConfig,
    random_seed: int = 0,
) -> List[float]:
    """Public wrapper to fit the neural ODE to Duffing mean dynamics."""
    from examples.neural_ode_template import _optimised_fit_neural_ode

    return _optimised_fit_neural_ode(
        drift_net,
        time_grid=time_grid,
        stochastic_trajectories_with_noise=stochastic_trajectories_with_noise,
        learning_rate=training_config.learning_rate,
        batch_size=training_config.batch_size,
        num_epochs=training_config.num_epochs,
        random_seed=random_seed,
        system_name=SYSTEM_NAME,
    )


def load_trained_duffing_model(
    model_path: str | Path,
) -> Tuple[DriftNet, Dict[str, Any]]:
    """Load a trained Duffing neural ODE model from disk."""
    from parameters.hyperparameters import NetworkArchitecture
    from config import DEVICE

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading trained model from {model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)

    arch_info = checkpoint["model_architecture"]
    architecture = NetworkArchitecture(
        input_size=arch_info["input_size"],
        hidden_sizes=arch_info["hidden_layer_sizes"],
        output_size=arch_info["output_size"],
    )

    model = DriftNet(architecture=architecture, device=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    model_info = {
        "architecture": arch_info,
        "training_config": checkpoint["training_config"],
        "training_losses": checkpoint["training_losses"],
        "performance": {
            "final_loss": checkpoint["training_config"].get("final_loss"),
            "initial_loss": checkpoint["training_config"].get("initial_loss"),
            "improvement_factor": checkpoint["training_config"].get(
                "improvement_factor"
            ),
        },
    }

    print("Model loaded successfully.")
    print(f"   Architecture: {arch_info['hidden_layer_sizes']}")
    print(f"   Training: {checkpoint['training_config']['num_epochs']} epochs")
    print(f"   Final loss: {checkpoint['training_config']['final_loss']:.2e}")

    return model, model_info


def simulate_with_loaded_model(
    model: DriftNet,
    initial_state: list[float] | Tensor,
    total_time: float,
    timestep: float = 0.01,
    gamma: float | None = None,
    omega: float | None = None,
) -> Tuple[Tensor, Tensor]:
    """Simulate a trajectory using a loaded neural ODE model."""
    from config import DEVICE

    if isinstance(initial_state, list):
        initial_state_tensor: Tensor = torch.tensor(
            initial_state, device=DEVICE, dtype=torch.float32
        )
    else:
        initial_state_tensor = cast(Tensor, initial_state)

    print("Simulating trajectory from", initial_state_tensor.tolist())
    print(f"   Time span: 0 to {total_time} (dimensionless)")
    print(f"   Timestep: {timestep}")

    time_grid, trajectory = simulate_duffing_neural_ode_trajectory(
        model,
        x0=initial_state_tensor,
        t0=0.0,
        tN=total_time,
        dt=timestep,
        gamma=gamma,
        Omega=omega,
    )

    print(f"Simulation complete: {len(time_grid)} time points")

    return time_grid, trajectory


def print_model_summary(model_info: Dict[str, Any]) -> None:
    """Print a detailed summary of the loaded model."""
    print("\n" + "=" * 50)
    print("DUFFING NEURAL ODE MODEL SUMMARY")
    print("=" * 50)

    arch = model_info["architecture"]
    config = model_info["training_config"]

    print("\n🏗️  ARCHITECTURE:")
    print(f"   Input size:  {arch['input_size']} (position, velocity, time, forcing)")
    print(f"   Hidden layers: {arch['hidden_layer_sizes']}")
    print(f"   Output size: {arch['output_size']} (d_position/dt, d_velocity/dt)")

    print("\nTRAINING DATA:")
    print(f"   Trajectories: {config['num_trajectories']}")
    print(f"   Time span:    {config['total_time']} dimensionless units")
    print(f"   Timestep:     {config['timestep']}")
    print(f"   Final loss:    {config['final_loss']:.2e}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser for different modes of operation."""
    parser = argparse.ArgumentParser(
        description="Duffing Neural ODE",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model (default)
  python examples/duffing_oscillator.py
  
  # Train without noise
  python examples/duffing_oscillator.py --no-noise

  # Load and test a trained model
  python examples/duffing_oscillator.py --load examples/models/duffing_neural_ode.pth
  
  # Load model and simulate custom trajectory
  python examples/duffing_oscillator.py --load examples/models/duffing_neural_ode.pth --simulate --initial-state 0.5 0.0 --time 10.0
  
  # Train with custom parameters
  python examples/duffing_oscillator.py --epochs 500 --learning-rate 1e-3 --trajectories 32
        """,
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--load",
        type=str,
        metavar="MODEL_PATH",
        help="Load and evaluate a trained model from the specified path",
    )
    mode_group.add_argument(
        "--train",
        action="store_true",
        default=True,
        help="Train a new model (default mode)",
    )

    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Run simulation with loaded model (requires --load)",
    )
    parser.add_argument(
        "--initial-state",
        nargs=2,
        type=float,
        default=[0.1, 0.0],
        metavar=("POS", "VEL"),
        help="Initial conditions [position, velocity] for simulation (default: 0.1 0.0)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=5.0,
        help="Simulation time for loaded model (default: 5.0)",
    )

    training_group = parser.add_argument_group("Training Parameters")
    training_group.add_argument(
        "--trajectories",
        type=int,
        default=256,
        help="Number of training trajectories (default: 1024)",
    )
    training_group.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of training epochs (default: 300)",
    )
    training_group.add_argument(
        "--learning-rate",
        type=float,
        default=3.4e-4,
        help="Learning rate for training (default: 0.001)",
    )
    training_group.add_argument(
        "--architecture",
        nargs="+",
        type=int,
        default=[128, 128, 128],
        help="Hidden layer sizes (default: 256 256 128)",
    )
    training_group.add_argument(
        "--no-noise",
        action="store_true",
        help="Generate training data without stochastic noise.",
    )

    parser.add_argument(
        "--show-plots", action="store_true", help="Display plots interactively"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="examples/figures",
        help="Directory to save figures (default: examples/figures)",
    )

    return parser


def run_training_mode(args: argparse.Namespace) -> None:
    """Run the training mode with specified parameters."""

    print("TRAINING MODE")
    print(f"   Parameters: {args.trajectories} trajectories, {args.epochs} epochs")
    print(f"   Architecture: {args.architecture}")
    print(f"   Learning rate: {args.learning_rate}")
    learning_rate = args.learning_rate
    epochs = args.epochs
    architecture = args.architecture
    trajectories = args.trajectories
    training_batch_size = 16

    create_duffing_neural_ode_demonstration(
        num_trajectories=trajectories,
        total_time=25.0,
        timestep=0.01,
        hidden_layer_sizes=architecture,
        num_epochs=epochs,
        learning_rate=learning_rate,
        save_directory=args.save_dir,
        show_plots=args.show_plots,
        trajectory_batch_size=128,
        training_batch_size=training_batch_size,
        with_noise=not args.no_noise,
    )

    print("\nTraining complete.")


def run_loading_mode(args: argparse.Namespace) -> None:
    """Run the model loading and evaluation mode."""
    print("LOADING MODE")
    print(f"   Model path: {args.load}")

    model, info = load_trained_duffing_model(args.load)

    print_model_summary(info)

    if args.simulate:
        print("Running trajectory simulation...")
        time_grid, trajectory = simulate_with_loaded_model(
            model, initial_state=args.initial_state, total_time=args.time, timestep=0.02
        )

        print(f"Trajectory shape: {trajectory.shape}")
        print(f"   Final position: {trajectory[-1, 0, 0].item():.3f}")
        print(f"   Final velocity: {trajectory[-1, 0, 1].item():.3f}")

        if args.show_plots or args.save_dir:
            print("Creating trajectory visualisation...")
            stoch_for_plot = trajectory.squeeze(1).unsqueeze(0)
            save_path = (
                Path(args.save_dir) / "loaded_model_trajectory.png"
                if args.save_dir
                else None
            )
            plot_duffing_trajectories_vs_neural_ode(
                time_grid,
                stoch_for_plot,
                time_grid,
                trajectory,
                save_path=save_path,
                show_plot=args.show_plots,
            )

    print("\nModel evaluation complete.")


def main():
    """Main entry point with command line argument parsing."""
    parser = create_argument_parser()
    args = parser.parse_args()

    if args.simulate and not args.load:
        parser.error("--simulate requires --load to be specified")

    if args.load:
        run_loading_mode(args)
    else:
        run_training_mode(args)


if __name__ == "__main__":
    main()
