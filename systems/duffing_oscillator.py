"""Compatibility wrapper for the Duffing oscillator example system."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from examples.systems.duffing_oscillator import DuffingOscillator as _ExampleDuffingOscillator


class DuffingOscillator(_ExampleDuffingOscillator):
    """Extend the example Duffing oscillator with the legacy API semantics used in tests."""

    def get_initial_state(self, batch_size: Optional[int] = None) -> torch.Tensor:  # type: ignore[override]
        """Return a 1D state vector when no batch size is specified."""
        if batch_size is None:
            return super().get_initial_state(batch_size=1).squeeze(0)
        return super().get_initial_state(batch_size=batch_size)

    def integrate_sde(
        self,
        initial_state: Optional[torch.Tensor] = None,
        batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:  # type: ignore[override]
        """Preserve the legacy return shape of ``[num_steps + 1, state_dim]`` for single trajectories."""

        squeeze_output = False
        effective_batch_size = batch_size

        if initial_state is None and batch_size is None:
            effective_batch_size = 1
            squeeze_output = True
        elif batch_size is None and initial_state is not None and initial_state.ndim == 1:
            effective_batch_size = 1
            squeeze_output = True
        elif batch_size is None and initial_state is not None:
            effective_batch_size = initial_state.shape[0]

        time_grid, trajectory = super().integrate_sde(
            initial_state=initial_state,
            batch_size=effective_batch_size or 1,
        )

        if squeeze_output:
            trajectory = trajectory.squeeze(0)

        return time_grid, trajectory


__all__ = ["DuffingOscillator"]
