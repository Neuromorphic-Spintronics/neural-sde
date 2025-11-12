#!/usr/bin/env python3
"""Create a compact nanorings dataset for the neural ODE/SDE examples.

This script selects a pair of nanorings input and measurement files (defaulting
to the first available pair), trims the initial transient portion, keeps 10% of
the remaining trajectory, reconstructs the exogenous H-field signal, and saves
the result as Torch tensors under ``examples/data/``.

The resulting artefact contains:

``trajectories`` (torch.Tensor)
    Shape ``[num_runs, num_steps, 2]`` with columns ``[AMR, H]``.
``time_grid`` (torch.Tensor)
    Monotonic time grid aligned with the retained samples.
``dt`` (torch.Tensor)
    Scalar timestep corresponding to the provided sample rate.
``h_signal`` (torch.Tensor)
    Shared exogenous forcing signal of length ``num_steps``.
``metadata`` (dict[str, Any])
    Auxiliary information about the preprocessing decisions.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch


@dataclass(frozen=True)
class NanoringsSubsetConfig:
    """Configuration for building the reduced nanorings dataset."""

    raw_data_dir: Path = Path("data/nanorings/raw")
    output_path: Path = Path("examples/data/nanorings_first_set.pt")
    sample_rate: int = 3200
    target_channel: int = 0
    transient_length: int = 300
    keep_fraction: float = 1.0
    signal_index: Optional[int] = None


def _signal_identifier(path: Path, *, prefix: str) -> int:
    """Extract the integer identifier from a nanorings filename."""

    stem = path.stem
    if prefix not in stem:
        raise ValueError(f"Unexpected file name format: {path.name}")
    identifier = stem.split(prefix)[-1]
    try:
        return int(identifier)
    except ValueError as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"Could not parse identifier from {path.name}") from exc


def _find_first_pair(
    raw_data_dir: Path,
    *,
    signal_index: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Locate the requested (or first available) Input/Measured nanorings pair."""

    input_files = sorted(raw_data_dir.glob("Input_Fields_sig*.npy"))
    input_by_id = {
        _signal_identifier(path, prefix="Input_Fields_sig"): path for path in input_files
    }
    measured_files = {
        _signal_identifier(path, prefix="Measured_signals_sig"): path
        for path in raw_data_dir.glob("Measured_signals_sig*.npy")
    }

    if not input_files or not measured_files:
        raise FileNotFoundError(
            "Could not locate both Input_Fields_sig*.npy and Measured_signals_sig*.npy files."
        )

    if signal_index is not None:
        if signal_index <= 0:
            raise ValueError("signal_index must be a positive integer when provided.")
        input_path = input_by_id.get(signal_index)
        measured_path = measured_files.get(signal_index)
        if input_path is None or measured_path is None:
            raise FileNotFoundError(
                f"No matching nanorings pair found for signal_index={signal_index}."
            )
        return input_path, measured_path

    for input_path in input_files:
        signal_id = _signal_identifier(input_path, prefix="Input_Fields_sig")
        measured_path = measured_files.get(signal_id)
        if measured_path is not None:
            return input_path, measured_path

    raise FileNotFoundError("No matching nanorings input/measurement file pair found.")


def _load_raw_arrays(
    input_path: Path,
    measured_path: Path,
    *,
    target_channel: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw numpy arrays for the chosen signal pair."""

    h_field = np.load(input_path)  # Expected shape: (1, 100)
    measured = np.load(measured_path)  # Expected shape: (1, 100, 7, 5300)

    if h_field.ndim != 2 or h_field.shape[0] != 1:
        raise ValueError(
            f"Unexpected H-field shape {h_field.shape}; expected (1, N_steps)."
        )

    if measured.ndim != 4 or measured.shape[0] != 1:
        raise ValueError(
            f"Unexpected measured signal shape {measured.shape}; expected (1, N_runs, 7, T)."
        )

    if not 0 <= target_channel < measured.shape[2]:
        valid_upper = measured.shape[2] - 1
        raise ValueError(
            f"target_channel={target_channel} is outside valid range [0, {valid_upper}]."
        )

    h_field_sequence = np.asarray(h_field[0], dtype=np.float32)  # Shape: (N_steps,)
    amr_responses = np.asarray(
        measured[0, :, target_channel, :], dtype=np.float32
    )  # Shape: (N_runs, T)

    return h_field_sequence, amr_responses


def _reconstruct_h_signal(
    h_field_sequence: np.ndarray,
    total_time_steps: int,
    *,
    sample_rate: int,
) -> Tuple[np.ndarray, float]:
    """Convert tabulated amplitudes into a sinusoidal H-field signal."""

    steps = h_field_sequence.shape[0]
    if steps == 0:
        raise ValueError("H-field sequence is empty.")

    samples_per_step = total_time_steps // steps
    if samples_per_step <= 0:
        raise ValueError("Insufficient samples to reconstruct H-field waveform.")

    base_frequency_hz = sample_rate / float(samples_per_step)
    omega = 2.0 * np.pi * base_frequency_hz
    time_grid = np.arange(total_time_steps, dtype=np.float32) / float(sample_rate)

    amplitude_envelope = np.repeat(h_field_sequence, samples_per_step)
    if amplitude_envelope.shape[0] < total_time_steps:
        pad_length = total_time_steps - amplitude_envelope.shape[0]
        amplitude_envelope = np.concatenate(
            [amplitude_envelope, np.full(pad_length, h_field_sequence[-1], dtype=np.float32)]
        )
    amplitude_envelope = amplitude_envelope[:total_time_steps].astype(np.float32)

    h_signal = amplitude_envelope * np.sin(omega * time_grid)
    return h_signal.astype(np.float32), base_frequency_hz


def _trim_and_subsample(
    h_signal: np.ndarray,
    amr_responses: np.ndarray,
    *,
    transient_length: int,
    keep_fraction: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove the initial transient and keep a fraction of the remaining samples."""

    if not 0.0 < keep_fraction <= 1.0:
        raise ValueError("keep_fraction must lie in (0, 1].")
    if transient_length < 0:
        raise ValueError("transient_length must be non-negative.")
    if transient_length >= h_signal.shape[0]:
        raise ValueError(
            "Transient length is greater than or equal to the available samples."
        )

    trimmed_h = h_signal[transient_length:]
    trimmed_amr = amr_responses[:, transient_length:]

    retained_steps = trimmed_h.shape[0]
    keep_steps = max(1, int(math.floor(retained_steps * keep_fraction)))

    return trimmed_h[:keep_steps].copy(), trimmed_amr[:, :keep_steps].copy()


def _build_dataset(
    h_signal: np.ndarray,
    amr_responses: np.ndarray,
    *,
    sample_rate: int,
    base_frequency_hz: float,
    input_path: Path,
    measured_path: Path,
    transient_length: int,
    keep_fraction: float,
) -> Dict[str, Any]:
    """Assemble tensors and metadata for persistence."""

    if amr_responses.ndim != 2:
        raise ValueError("AMR responses must have shape [num_runs, num_steps].")
    if h_signal.ndim != 1:
        raise ValueError("H signal must be one-dimensional.")

    num_runs, num_steps = amr_responses.shape
    if h_signal.shape[0] != num_steps:
        raise ValueError(
            f"H signal length {h_signal.shape[0]} does not match AMR steps {num_steps}."
        )

    dt = 1.0 / float(sample_rate)
    time_grid = torch.arange(num_steps, dtype=torch.float32) * dt

    h_broadcast = np.tile(h_signal, (num_runs, 1))
    stacked = np.stack((amr_responses, h_broadcast), axis=-1).astype(np.float32)
    signal_identifier = _signal_identifier(
        input_path,
        prefix="Input_Fields_sig",
    )

    dataset: Dict[str, Any] = {
        "trajectories": torch.from_numpy(stacked.copy()),  # [runs, steps, 2]
        "time_grid": time_grid,
        "dt": torch.tensor(dt, dtype=torch.float32),
        "h_signal": torch.from_numpy(h_signal.copy()),
        "metadata": {
            "source_files": {
                "input_path": input_path.as_posix(),
                "measured_path": measured_path.as_posix(),
            },
            "signal_index": signal_identifier,
            "sample_rate": sample_rate,
            "transient_length": transient_length,
            "keep_fraction": keep_fraction,
            "num_runs": int(num_runs),
            "num_steps_retained": int(num_steps),
            "derived_h_frequency_hz": base_frequency_hz,
        },
    }
    return dataset


def preprocess(config: NanoringsSubsetConfig) -> Dict[str, Any]:
    """Execute the preprocessing pipeline and return the dataset dictionary."""

    input_path, measured_path = _find_first_pair(
        config.raw_data_dir,
        signal_index=config.signal_index,
    )
    h_field_sequence, amr_responses = _load_raw_arrays(
        input_path,
        measured_path,
        target_channel=config.target_channel,
    )

    expanded_h, base_frequency_hz = _reconstruct_h_signal(
        h_field_sequence,
        total_time_steps=amr_responses.shape[-1],
        sample_rate=config.sample_rate,
    )

    retained_h, retained_amr = _trim_and_subsample(
        expanded_h,
        amr_responses,
        transient_length=config.transient_length,
        keep_fraction=config.keep_fraction,
    )

    return _build_dataset(
        retained_h,
        retained_amr,
        sample_rate=config.sample_rate,
        base_frequency_hz=base_frequency_hz,
        input_path=input_path,
        measured_path=measured_path,
        transient_length=config.transient_length,
        keep_fraction=config.keep_fraction,
    )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments for the preprocessing script."""

    parser = argparse.ArgumentParser(
        description="Create a compact nanorings dataset for the neural ODE/SDE examples.",
    )
    parser.add_argument(
        "--signal-index",
        type=int,
        default=None,
        help=(
            "Select a specific nanorings signal pair by its numeric identifier "
            "(e.g. '--signal-index 3' loads Input_Fields_sig3.npy). "
            "Defaults to the first available pair."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    """Entry point used when running the script directly."""

    args = parse_args(argv)

    config = NanoringsSubsetConfig(signal_index=args.signal_index)

    dataset = preprocess(config)
    output_path = config.output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)

    metadata = dataset["metadata"]
    print("Saved nanorings subset dataset:", output_path.as_posix())
    print(f" - signal index: {metadata['signal_index']}")
    print(f" - runs retained: {metadata['num_runs']}")
    print(f" - steps retained: {metadata['num_steps_retained']}")
    print(f" - timestep (dt): {dataset['dt'].item():.6e}")


if __name__ == "__main__":
    main()
