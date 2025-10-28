from typing import Tuple, TYPE_CHECKING
import torch
from torch import Tensor

from neural_dynamics.models.sde import NeuralSDE
from neural_dynamics.core.hyperparameters import Hyperparameters

if TYPE_CHECKING:
    from neural_dynamics.models.ode import NeuralODE


def train_loop(
    trajectories: Tensor,
    time_grid: Tensor,
    hyperparameters: Hyperparameters,
    device: torch.device,
    enable_adversarial: bool = True,
) -> Tuple["NeuralODE", NeuralSDE, list[float], list[float]]:
    """
    Train a Neural SDE model by first training a Neural ODE, then using it to initialise the SDE.

    Args:
        neural_sde_placeholder: Placeholder argument for compatibility, ignored.
        trajectories: Training trajectories tensor of shape [batch, time, state]
        time_grid: Time grid tensor
        hyperparameters: Hyperparameters for training
        device: Device to train on
        enable_adversarial: Whether to enable adversarial training for the SDE

    Returns:
        Trained NeuralODE, Trained NeuralSDE, training losses, validation losses
    """
    # Import here to avoid circular import
    from neural_dynamics.models.ode import NeuralODE
    
    # First train the Neural ODE
    print("Training Neural ODE...")
    neural_ode = NeuralODE.train(
        hyperparameters=hyperparameters,
        trajectories=trajectories,
        time_grid=time_grid,
        device=device,
        validation_split=0.2,
        early_stopping_patience=hyperparameters.number_of_epochs // 4,
    )

    # Train the Neural SDE using the pretrained Neural ODE
    print("Training Neural SDE...")
    neural_sde = NeuralSDE.train(
        hyperparameters=hyperparameters,
        neural_ode=neural_ode,
        trajectories=trajectories,
        time_grid=time_grid,
        device=device,
        enable_adversarial=enable_adversarial,
        random_seed=42,
    )

    # Return the trained ODE, SDE and the ODE's losses
    return neural_ode, neural_sde, neural_ode.training_losses, neural_ode.validation_losses