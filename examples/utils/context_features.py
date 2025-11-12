"""Feature builders for single-state dynamical systems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import MutableMapping, Tuple

import torch
from torch import Tensor


def _build_context_windows(series: Tensor, *, window_size: int) -> Tensor:
    """Return lagged copies of ``series`` with zero padding for early steps."""

    if window_size <= 0:
        raise ValueError("window_size must be a positive integer.")

    batch, time = series.shape
    contexts = []
    for lag in range(1, window_size + 1):
        shifted = torch.zeros(batch, time, dtype=series.dtype, device=series.device)
        shifted[:, lag:] = series[:, :-lag]
        contexts.append(shifted.unsqueeze(-1))
    return torch.cat(contexts, dim=-1)


@dataclass
class SingleStateContextEmbedding:
    """Construct auxiliary features for single-state trajectories.

    The nanorings dataset only contains an AMR channel (and a shared H-field
    drive). This helper reproduces the manual feature engineering we explored
    earlier, but keeps the original state untouched: we expose the augmented
    features separately so callers can provide them as *inputs* to the drift or
    diffusion networks without inflating the state dimension.

    Example usage::

        builder = SingleStateContextEmbedding(context_points=5)
        state, inputs = builder.build(trajectories, h_signal, time_grid, metadata)

    Here ``state`` is the original AMR/H tensor, while ``inputs`` contains the
    sinusoidal harmonics and context windows described below.
    """

    context_points: int = 5
    include_sinusoids: bool = True

    def build(
        self,
        trajectories: Tensor,
        h_signal: Tensor,
        time_grid: Tensor,
        metadata: MutableMapping[str, object],
    ) -> Tuple[Tensor, Tensor]:
        """Return ``(state, inputs)`` tensors with additional context features."""

        if trajectories.ndim != 3:
            raise ValueError("trajectories must have shape [batch, time, features].")
        if trajectories.shape[-1] < 1:
            raise ValueError("trajectories must contain at least one state dimension.")

        batch, time_steps, _ = trajectories.shape
        device = trajectories.device
        dtype = trajectories.dtype

        amr = trajectories[..., 0]

        # Shared H signal broadcast to match batch orientation.
        h_broadcast = h_signal.to(device=device, dtype=dtype).unsqueeze(0).expand(batch, -1)

        feature_tensors: list[Tensor] = []
        feature_names: list[str] = []

        if self.include_sinusoids:
            freq_hz = float(metadata.get("derived_h_frequency_hz", 0.0))
            if freq_hz > 0.0:
                omega = 2.0 * torch.pi * freq_hz
                sin_wt = torch.sin(omega * time_grid.to(device=device, dtype=dtype))
                sin_2wt = torch.sin(2.0 * omega * time_grid.to(device=device, dtype=dtype))
            else:
                sin_wt = torch.zeros(time_steps, device=device, dtype=dtype)
                sin_2wt = torch.zeros_like(sin_wt)

            feature_tensors.extend(
                [sin_wt.unsqueeze(0).expand(batch, -1), sin_2wt.unsqueeze(0).expand(batch, -1)]
            )
            feature_names.extend(["sin_wt", "sin_2wt"])

        if self.context_points > 0:
            amr_ctx = _build_context_windows(amr, window_size=self.context_points)
            h_ctx = _build_context_windows(h_broadcast, window_size=self.context_points)
            feature_tensors.extend([amr_ctx.reshape(batch, time_steps, -1), h_ctx.reshape(batch, time_steps, -1)])
            feature_names.extend(
                [f"amr_ctx_{idx}" for idx in range(1, self.context_points + 1)]
                + [f"h_ctx_{idx}" for idx in range(1, self.context_points + 1)]
            )

        if not feature_tensors:
            inputs = torch.empty(batch, time_steps, 0, device=device, dtype=dtype)
        else:
            # Ensure every feature tensor is 3-D [batch, time, feat].
            normalised = [tensor.unsqueeze(-1) if tensor.ndim == 2 else tensor for tensor in feature_tensors]
            inputs = torch.cat(normalised, dim=-1)

        metadata["context_feature_names"] = feature_names
        metadata["context_points"] = self.context_points

        return trajectories, inputs


__all__ = ["SingleStateContextEmbedding"]
