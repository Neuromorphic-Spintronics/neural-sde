import os
import sys
import glob
import torch
import numpy as np
from scipy import signal
from examples.systems.parameters.nanorings import NanoringsHyperparameters
from config import DEVICE

# Add project root to path to allow imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def normalize_h_field(h_traj, return_params=False):
    """Normalize H-field trajectory to zero mean and unit std."""
    h_mean = h_traj.mean()
    h_std = h_traj.std()
    h_norm = (h_traj - h_mean) / (h_std + 1e-8)
    if return_params:
        return h_norm, {"mean": h_mean, "std": h_std}
    return h_norm


def process_amr_trajectory(amr_traj, sample_rate, highpass_cutoff=0.1, return_params=False):
    """Apply high-pass filter and standardization to AMR trajectory."""
    # High-pass filter
    sos = signal.butter(4, highpass_cutoff, btype='high', fs=sample_rate, output='sos')
    amr_filtered = signal.sosfilt(sos, amr_traj)
    
    # Standardize
    amr_mean = amr_filtered.mean()
    amr_std = amr_filtered.std()
    amr_norm = (amr_filtered - amr_mean) / (amr_std + 1e-8)
    
    if return_params:
        return amr_norm, {"mean": amr_mean, "std": amr_std, "cutoff": highpass_cutoff}
    return amr_norm


def create_time_features(time_grid, frequencies):
    """Create sinusoidal time features at specified frequencies."""
    features = {}
    for i, freq in enumerate(frequencies, 1):
        features[f'sin_time_{i}'] = torch.sin(2 * np.pi * freq * time_grid)
    return features


def preprocess_and_save_nanorings_data():
    """
    Loads raw nanoring data, applies high-pass filtering and standardisation,
    and saves the final dataset.
    """
    config = NanoringsHyperparameters()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_path = os.path.join(project_root, "data.tmp/nanorings/raw/")
    output_dir = os.path.join(project_root, "examples/data/processed/")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Raw Data ---
    all_target_files = sorted(glob.glob(os.path.join(data_path, "Measured_signals_sig*.npy")))
    all_input_files = sorted(glob.glob(os.path.join(data_path, "Input_Fields_sig*.npy")))

    num_files_to_load = int(len(all_target_files) * config.PERCENTAGE_OF_FILES_TO_LOAD)
    if num_files_to_load == 0:
        num_files_to_load = 1

    target_files_to_load = all_target_files[:num_files_to_load]
    input_files_to_load = all_input_files[:num_files_to_load]

    list_of_amr_signals, list_of_h_fields = [], []
    print(f"Loading {len(target_files_to_load)} of {len(all_target_files)} signal file(s)...")

    _first_target_squeezed = torch.from_numpy(np.load(target_files_to_load[0], allow_pickle=True)).float().squeeze()
    total_original_length = _first_target_squeezed.shape[2]
    dt = 1.0 / config.SAMPLE_RATE
    total_trajectories_before_filtering = 0

    for target_file, input_file in zip(target_files_to_load, input_files_to_load):
        amr_signals_one_file = torch.from_numpy(np.load(target_file, allow_pickle=True)).float().squeeze().to(DEVICE)
        h_amplitudes_one_file = torch.from_numpy(np.load(input_file, allow_pickle=True)).float().squeeze().to(DEVICE)

        if amr_signals_one_file.dim() == 2:
            amr_signals_one_file = amr_signals_one_file.unsqueeze(0)
        num_repetitions = amr_signals_one_file.shape[0]
        total_trajectories_before_filtering += num_repetitions

        amplitudes_repeated = h_amplitudes_one_file.repeat_interleave(config.H_FIELD_AMPLITUDE_UPDATE_RATE)
        h_field_template = amplitudes_repeated[:total_original_length]
        h_fields_one_file = h_field_template.unsqueeze(0).repeat(num_repetitions, 1)

        # No hard mean filter — keep all trajectories
        list_of_amr_signals.append(amr_signals_one_file)
        list_of_h_fields.append(h_fields_one_file)

    amr_signals_full = torch.cat(list_of_amr_signals, dim=0)
    H_field_full = torch.cat(list_of_h_fields, dim=0)
    num_total_trajectories = amr_signals_full.shape[0]

    print(f"Kept {num_total_trajectories} out of {total_trajectories_before_filtering} trajectories.")

    # --- 2. Preprocess: High-pass filter + Standardise ---
    amr_target_chan_full = amr_signals_full[:, config.TARGET_CHANNEL, :]
    amr_main_norm_list, amr_context_norm_list = [], []
    h_main_norm_list, h_context_norm_list = [], []
    scalers, h_scalers = [], []

    for i in range(num_total_trajectories):
        # Process H-field using utility function
        h_traj = H_field_full[i]
        h_traj_norm, h_params = normalize_h_field(h_traj, return_params=True)
        h_scalers.append(h_params)

        # Process AMR signal using utility function
        amr_traj = amr_target_chan_full[i].cpu().numpy()
        amr_traj_norm, amr_params = process_amr_trajectory(
            amr_traj, 
            config.SAMPLE_RATE, 
            highpass_cutoff=0.1, 
            return_params=True
        )
        scalers.append(amr_params)

        # Convert back to torch
        amr_traj_norm = torch.from_numpy(amr_traj_norm).float().to(DEVICE)

        # Context / main splits
        h_transient = h_traj_norm[:config.TRANSIENT_LENGTH]
        h_main_part = h_traj_norm[config.TRANSIENT_LENGTH:]
        amr_transient = amr_traj_norm[:config.TRANSIENT_LENGTH]
        amr_main_part = amr_traj_norm[config.TRANSIENT_LENGTH:]

        h_context_norm_list.append(h_transient[-config.CONTEXT_POINTS:])
        amr_context_norm_list.append(amr_transient[-config.CONTEXT_POINTS:])

        clipped_len = int(h_main_part.shape[0] * config.PERCENTAGE_OF_MAIN_SEQUENCE_TO_USE)
        h_main_norm_list.append(h_main_part[:clipped_len])
        amr_main_norm_list.append(amr_main_part[:clipped_len])

    h_main = torch.stack(h_main_norm_list)
    amr_main_norm = torch.stack(amr_main_norm_list)
    h_context = torch.stack(h_context_norm_list)
    amr_context = torch.stack(amr_context_norm_list)
    clipped_main_len = h_main.shape[1]

    # --- 3. Create Time Features ---
    time_grid = torch.linspace(0.0, dt * (clipped_main_len - 1), clipped_main_len).to(DEVICE)
    t_min, t_max = time_grid[0], time_grid[-1]  # Keep for metadata
    
    # Use utility function for time features
    time_features = create_time_features(time_grid, frequencies=[2.0, 4.0])
    sin_time_1_full = time_features['sin_time_1'].unsqueeze(0).repeat(num_total_trajectories, 1)
    sin_time_2_full = time_features['sin_time_2'].unsqueeze(0).repeat(num_total_trajectories, 1)

    # --- 4. Train/Val Split ---
    num_val = int(num_total_trajectories * config.VALIDATION_SPLIT)
    if num_val == 0 and num_total_trajectories > 1:
        num_val = 1
    num_train = num_total_trajectories - num_val
    indices = torch.randperm(num_total_trajectories)
    train_idx, val_idx = indices[:num_train], indices[num_train:]

    # --- 5. Package Data ---
    processed_data = {
        "train_h_context": h_context[train_idx].cpu(),
        "train_amr_context": amr_context[train_idx].cpu(),
        "train_h_main": h_main[train_idx].cpu(),
        "train_amr_main_norm": amr_main_norm[train_idx].cpu(),
        "train_sin_time_1": sin_time_1_full[train_idx].cpu(),
        "train_sin_time_2": sin_time_2_full[train_idx].cpu(),

        "val_h_context": h_context[val_idx].cpu(),
        "val_amr_context": amr_context[val_idx].cpu(),
        "val_h_main": h_main[val_idx].cpu(),
        "val_amr_main_norm": amr_main_norm[val_idx].cpu(),
        "val_sin_time_1": sin_time_1_full[val_idx].cpu(),
        "val_sin_time_2": sin_time_2_full[val_idx].cpu(),

        # Scalers for invertibility
        "val_scalers": [scalers[i] for i in val_idx],
        "val_h_scalers": [h_scalers[i] for i in val_idx],
        "train_scalers": [scalers[i] for i in train_idx],

        "dt": dt,
        "t_min": t_min.cpu(),
        "t_max": t_max.cpu(),
        "time_grid": time_grid.cpu(),
    }

    # --- 6. Save ---
    output_path = os.path.join(output_dir, "nanorings_all_trajectories_highpass_standardised.pt")
    torch.save(processed_data, output_path)
    print(f"Saved high-pass + standardised dataset (all trajectories) to {output_path}")

if __name__ == "__main__":
    preprocess_and_save_nanorings_data()
