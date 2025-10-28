"""Checkpoint utilities for Neural SDE/ODE models."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch

from neural_dynamics.core.hyperparameters import Hyperparameters
from neural_dynamics.models.ode import NeuralODE
from neural_dynamics.models.sde import NeuralSDE


def train_or_load_models(
    hyperparameters: Hyperparameters,
    trajectories_tensor: torch.Tensor,
    time_grid_tensor: torch.Tensor,
    device: torch.device,
    output_dir: str,
) -> Tuple[NeuralODE, NeuralSDE, List[float], List[float], List[float], List[float]]:
    """Train models or load them from disk if artefacts already exist.

    Args:
        hyperparameters: Training hyperparameters.
        trajectories_tensor: Training trajectories on the target device.
        time_grid_tensor: Time grid on the target device.
        device: Device to use for training/inference.
        output_dir: Directory for saving/loading models and losses.

    Returns:
        Tuple containing the trained NeuralODE, NeuralSDE and loss histories.
    """
    from neural_dynamics.training.train_loop import train_loop  # Lazy import to avoid cycles

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ode_path = output_path / "neural_ode_model.pth"
    sde_path = output_path / "neural_sde_model.pth"
    train_losses_path = output_path / "train_losses.pth"
    val_losses_path = output_path / "val_losses.pth"
    gen_losses_path = output_path / "generator_losses.pth"
    crit_losses_path = output_path / "critic_losses.pth"

    artefacts_exist = all(
        path.exists()
        for path in (
            ode_path,
            sde_path,
            train_losses_path,
            val_losses_path,
            gen_losses_path,
            crit_losses_path,
        )
    )

    if artefacts_exist:
        print("Loading existing models and loss histories...")
        neural_sde = NeuralSDE(hyperparameters)
        neural_sde.load_state_dict(torch.load(sde_path, map_location=device))
        neural_sde.to(device)
        neural_ode = NeuralODE(
            drift_net=neural_sde.drift_net, hyperparameters=hyperparameters
        )
        super(NeuralSDE, neural_sde).train(False)
        neural_ode.drift_net.eval()

        train_losses = torch.load(train_losses_path)
        val_losses = torch.load(val_losses_path)
        generator_losses = torch.load(gen_losses_path)
        critic_losses = torch.load(crit_losses_path)
        return (
            neural_ode,
            neural_sde,
            train_losses,
            val_losses,
            generator_losses,
            critic_losses,
        )

    print("Training Neural SDE model...")
    neural_ode, neural_sde, train_losses, val_losses = train_loop(
        trajectories_tensor, time_grid_tensor, hyperparameters, device
    )

    torch.save(neural_ode.state_dict(), ode_path)
    torch.save(neural_sde.state_dict(), sde_path)
    torch.save(train_losses, train_losses_path)
    torch.save(val_losses, val_losses_path)

    generator_losses = neural_sde.generator_losses
    critic_losses = neural_sde.critic_losses
    torch.save(generator_losses, gen_losses_path)
    torch.save(critic_losses, crit_losses_path)

    return (
        neural_ode,
        neural_sde,
        train_losses,
        val_losses,
        generator_losses,
        critic_losses,
    )
