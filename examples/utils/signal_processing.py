"""Signal processing utilities for preprocessing."""

import numpy as np
import torch
from scipy import signal


def normalize_h_field(h_traj, return_params=False):
    """Normalize H-field trajectory."""
    mean = h_traj.mean()
    std = h_traj.std()
    if std < 1e-8:
        std = 1.0
    norm = (h_traj - mean) / std
    if return_params:
        return norm, (mean, std)
    return norm


def process_amr_trajectory(amr_traj, sample_rate, highpass_cutoff=0.1, return_params=False):
    """Process AMR trajectory with highpass filter and normalization."""
    # Highpass filter
    sos = signal.butter(4, highpass_cutoff, 'highpass', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, amr_traj)
    
    # Normalize
    mean = filtered.mean()
    std = filtered.std()
    if std < 1e-8:
        std = 1.0
    norm = (filtered - mean) / std
    
    if return_params:
        return norm, (mean, std)
    return norm


def create_time_features(time_grid, frequencies):
    """Create sinusoidal time features."""
    features = {}
    for i, freq in enumerate(frequencies, 1):
        features[f'sin_time_{i}'] = torch.sin(2 * np.pi * freq * time_grid)
    return features