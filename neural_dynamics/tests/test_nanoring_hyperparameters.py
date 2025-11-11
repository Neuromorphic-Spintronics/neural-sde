import sys
from pathlib import Path

import torch

try:
    from examples.nanorings import build_hyperparameters
    from examples.systems.parameters.nanorings import NanoringsHyperparameters
except ModuleNotFoundError:  # pragma: no cover - script fallback
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
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
