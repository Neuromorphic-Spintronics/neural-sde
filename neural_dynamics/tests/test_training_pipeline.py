from __future__ import annotations

from dataclasses import replace

import torch

from neural_dynamics.core.hyperparameters import Hyperparameters, LearningRates
from neural_dynamics.models.ode import NeuralODE
from neural_dynamics.models.sde import NeuralSDE


def test_defaults_are_consistent() -> None:
    defaults = Hyperparameters.defaults()

    state_dim = defaults.state_dimension
    assert defaults.drift_network.output_size == state_dim
    assert defaults.diffusion_network.output_size % state_dim == 0
    assert defaults.critic_network.input_size % state_dim == 0


def _make_reduced_hyperparameters(defaults: Hyperparameters) -> Hyperparameters:
    reduced_learning_rates = LearningRates(
        drift=defaults.learning_rates.drift,
        diffusion=defaults.learning_rates.diffusion,
        critic=defaults.learning_rates.critic,
        generator=defaults.learning_rates.generator,
    )

    return replace(
        defaults,
        number_of_epochs=2,
        batch_size=2,
        learning_rates=reduced_learning_rates,
    )


def _synthetic_trajectories(batch_size: int, time_steps: int, state_dim: int) -> torch.Tensor:
    time_grid = torch.linspace(0.0, 0.2, time_steps)
    trajectories = []
    for idx in range(batch_size):
        phase = 0.1 * idx
        components = [torch.sin(time_grid + phase), torch.cos(time_grid + phase)]
        extra = torch.sin(0.5 * time_grid + 0.2 * idx)
        components.append(extra)
        state = torch.stack(components[:state_dim], dim=-1)
        trajectories.append(state)
    return torch.stack(trajectories, dim=0)


def test_training_pipeline_runs() -> None:
    defaults = Hyperparameters.defaults()
    test_hparams = _make_reduced_hyperparameters(defaults)

    device = torch.device("cpu")
    time_steps = 16
    trajectories = _synthetic_trajectories(batch_size=4, time_steps=time_steps, state_dim=test_hparams.state_dimension)
    time_grid = torch.linspace(0.0, 0.2, time_steps)

    neural_ode = NeuralODE.train(
        hyperparameters=test_hparams,
        trajectories=trajectories,
        time_grid=time_grid,
        device=device,
        validation_split=0.5,
        early_stopping_patience=2,
    )

    assert neural_ode.device == device, "NeuralODE device should be cached correctly"
    assert neural_ode.training_losses, "Expected training losses to be recorded"
    assert all(torch.isfinite(torch.tensor(neural_ode.training_losses)))

    neural_sde = NeuralSDE.train(
        hyperparameters=test_hparams,
        neural_ode=neural_ode,
        trajectories=trajectories,
        time_grid=time_grid,
        device=device,
        enable_adversarial=False,
    )

    assert neural_sde.device == device, "NeuralSDE device should be cached correctly"

    for name, parameter in neural_ode.drift_net.state_dict().items():
        sde_parameter = neural_sde.drift_net.state_dict()[name]
        assert torch.allclose(parameter, sde_parameter)

    if neural_sde.diffusion_net is not None:
        for param in neural_sde.diffusion_net.parameters():
            assert torch.allclose(param, torch.zeros_like(param))

    assert neural_sde.generator_losses == []
    assert neural_sde.critic_losses == []
