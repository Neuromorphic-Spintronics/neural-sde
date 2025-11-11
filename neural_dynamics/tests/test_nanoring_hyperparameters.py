import torch

from examples.nanorings import build_hyperparameters
from examples.systems.parameters.nanorings import NanoringsHyperparameters


def test_build_hyperparameters_uses_sde_learning_rates() -> None:
    """Ensure nanorings SDE training uses the dedicated learning rates."""
    config = NanoringsHyperparameters()
    dummy_trajectories = torch.zeros(4, 256, 2)

    hyperparameters = build_hyperparameters(
        config, dummy_trajectories, timestep=0.01
    )

    assert hyperparameters.learning_rates.drift == config.DRIFT_LEARNING_RATE
    assert hyperparameters.learning_rates.diffusion == config.DIFFUSION_LEARNING_RATE
    assert hyperparameters.learning_rates.generator == config.GENERATOR_LEARNING_RATE
    assert hyperparameters.learning_rates.critic == config.CRITIC_LEARNING_RATE
