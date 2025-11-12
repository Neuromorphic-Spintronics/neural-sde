import os
import sys
import glob
import torch
import numpy as np

# Add project root to path to allow imports
sys.path.insert(0, '/Users/jack/Library/Mobile Documents/com~apple~CloudDocs/Research/neural-sde')
from examples.systems.parameters.nanorings import NanoringsHyperparameters

def preprocess_and_save_nanorings_data():
    """
    Loads raw nanoring data, performs all preprocessing steps, and saves the
    final processed data required for training and evaluation into a single file.
    """
    config = NanoringsHyperparameters()
    # Dynamically determine the project root directory
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
    data_path = os.path.join(project_root, "data/nanorings/raw/")
    output_dir = os.path.join(project_root, "examples/data/")
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load Raw Data from Files ---
    all_target_files = sorted(glob.glob(os.path.join(data_path, "Measured_signals_sig*.npy")))
    all_input_files = sorted(glob.glob(os.path.join(data_path, "Input_Fields_sig*.npy")))

    # Load multiple files (each file has 100 repetitions of the same H-field sweep)
    num_files_to_load = min(config.SIGNALS, int(len(all_target_files) * config.PERCENTAGE_OF_FILES_TO_LOAD))
    if num_files_to_load == 0:
        num_files_to_load = 1
    
    print("Each file contains ~100 repetitions of the same signal")
    print("Standardization will be done per-signal (across all 100 repetitions)")

    target_files_to_load = all_target_files[:num_files_to_load]
    input_files_to_load = all_input_files[:num_files_to_load]

    list_of_amr_signals, list_of_h_fields = [], []
    print(f"Loading ALL signals from {len(target_files_to_load)} of {len(all_target_files)} file(s) (no filtering)...")

    _first_target_squeezed = torch.from_numpy(np.load(target_files_to_load[0], allow_pickle=True)).float().squeeze()
    total_original_length = _first_target_squeezed.shape[2]
    dt = 1.0 / config.SAMPLE_RATE

    for target_file, input_file in zip(target_files_to_load, input_files_to_load):
        # Process on CPU to avoid GPU memory issues with large dataset
        amr_signals_one_file = torch.from_numpy(np.load(target_file, allow_pickle=True)).float().squeeze()
        h_amplitudes_one_file = torch.from_numpy(np.load(input_file, allow_pickle=True)).float().squeeze()

        if amr_signals_one_file.dim() == 2:
            amr_signals_one_file = amr_signals_one_file.unsqueeze(0)
        num_repetitions = amr_signals_one_file.shape[0]

        amplitudes_repeated = h_amplitudes_one_file.repeat_interleave(config.H_FIELD_AMPLITUDE_UPDATE_RATE)
        h_field_template = amplitudes_repeated[:total_original_length]
        h_fields_one_file = h_field_template.unsqueeze(0).repeat(num_repetitions, 1)

        # No filtering - keep ALL trajectories regardless of mean AMR values
        list_of_amr_signals.append(amr_signals_one_file)
        list_of_h_fields.append(h_fields_one_file)

    amr_signals_full = torch.cat(list_of_amr_signals, dim=0)
    H_field_full = torch.cat(list_of_h_fields, dim=0)
    num_total_trajectories = amr_signals_full.shape[0]

    if num_total_trajectories == 0:
        raise ValueError("No trajectories found in the data files.")

    print(f"Loaded ALL trajectories: {num_total_trajectories} total (no filtering applied).")

    # --- 2. Standardize Data PER-SIGNAL (across all 100 repetitions of each file) ---
    amr_target_chan_full = amr_signals_full[:, config.TARGET_CHANNEL, :]
    amr_main_norm_list = []
    h_main_norm_list = []
    scalers, h_scalers = [], []
    window_signal_ids = []
    windows_per_signal = []
    window_length = None
    
    # Each file has ~100 repetitions, so process in groups
    repetitions_per_file = amr_signals_one_file.shape[0]  # Should be ~100
    trajectories_per_file = repetitions_per_file
    
    print(f"\nStandardizing per-signal (across {repetitions_per_file} repetitions per file)...")
    
    for file_idx in range(len(list_of_amr_signals)):
        start_idx = file_idx * trajectories_per_file
        end_idx = start_idx + trajectories_per_file
        
        # Get all repetitions for this signal
        h_signal_reps = H_field_full[start_idx:end_idx]
        amr_signal_reps = amr_target_chan_full[start_idx:end_idx]
        
        signal_h_windows = []
        signal_amr_windows = []
        
        for i in range(trajectories_per_file):
            h_traj = h_signal_reps[i]
            amr_traj = amr_signal_reps[i]
            
            h_main_part = h_traj[config.TRANSIENT_LENGTH:]
            amr_main_part = amr_traj[config.TRANSIENT_LENGTH:]
            
            clipped_len = max(
                1, int(h_main_part.shape[0] * config.PERCENTAGE_OF_MAIN_SEQUENCE_TO_USE)
            )
            if window_length is None:
                window_length = clipped_len
            elif window_length != clipped_len:
                raise ValueError(
                    f"Inconsistent window length detected: expected {window_length}, got {clipped_len}"
                )

            total_steps = h_main_part.shape[0]
            num_full_windows = max(1, total_steps // clipped_len)
            window_starts = [idx * clipped_len for idx in range(num_full_windows)]
            if total_steps % clipped_len != 0:
                tail_start = total_steps - clipped_len
                if tail_start not in window_starts:
                    window_starts.append(tail_start)

            for start in window_starts:
                end = start + clipped_len
                if end > total_steps:
                    continue
                signal_h_windows.append(h_main_part[start:end])
                signal_amr_windows.append(amr_main_part[start:end])
        
        if not signal_h_windows:
            continue

        windows_per_signal.append(len(signal_h_windows))
        
        h_main_stacked = torch.stack(signal_h_windows)
        amr_main_stacked = torch.stack(signal_amr_windows)
        
        # First, remove per-repetition baseline (each rep centered on its own mean)
        h_main_baselined = h_main_stacked - h_main_stacked.mean(dim=1, keepdim=True)
        amr_main_baselined = amr_main_stacked - amr_main_stacked.mean(dim=1, keepdim=True)
        
        # Compute mean (should be ~0 after baseline removal) and std ACROSS all 100 repetitions
        h_mean = h_main_baselined.mean()
        h_std = h_main_baselined.std()
        if h_std < 1e-8:
            h_std = 1.0
            
        amr_mean = amr_main_baselined.mean()
        amr_std = amr_main_baselined.std()
        if amr_std < 1e-8:
            amr_std = 1.0
        
        # Apply same standardization to ALL repetitions of this signal
        for h_window, amr_window in zip(signal_h_windows, signal_amr_windows):
            h_main_baselined = h_window - h_window.mean()
            amr_main_baselined = amr_window - amr_window.mean()

            h_main_norm = (h_main_baselined - h_mean) / h_std
            amr_main_norm = (amr_main_baselined - amr_mean) / amr_std

            h_main_norm_list.append(h_main_norm)
            amr_main_norm_list.append(amr_main_norm)
            h_scalers.append((h_mean, h_std))
            scalers.append((amr_mean, amr_std))
            window_signal_ids.append(file_idx)

    if not h_main_norm_list:
        raise ValueError("No windowed trajectories were generated; check preprocessing settings.")

    h_main = torch.stack(h_main_norm_list)
    amr_main_norm = torch.stack(amr_main_norm_list)
    clipped_main_len = h_main.shape[1]

    # --- 3. Create Time Features ---
    time_grid = torch.linspace(0.0, dt * (clipped_main_len - 1), clipped_main_len)
    t_min, t_max = time_grid[0], time_grid[-1]
    time_main_norm = (time_grid - t_min) / (t_max - t_min)

    sin_time_1 = torch.sin(2 * np.pi * time_main_norm)
    sin_time_2 = torch.sin(4 * np.pi * time_main_norm)
    sin_time_1_full = sin_time_1.unsqueeze(0).repeat(h_main.shape[0], 1)
    sin_time_2_full = sin_time_2.unsqueeze(0).repeat(h_main.shape[0], 1)

    total_windowed_points = h_main.shape[0] * clipped_main_len
    print(f"\nWindowed trajectory length: {clipped_main_len} steps (~{clipped_main_len * dt:.4f} s)")
    print(f"Total windowed trajectories: {h_main.shape[0]}")
    print(f"Total samples seen during training: {total_windowed_points} points")
    print("Windowed samples per signal:")
    for sig_idx, count in enumerate(windows_per_signal):
        print(f"  Signal {sig_idx}: {count} windows")

    # --- 4. Create Train/Val Splits (by SIGNAL, not by trajectory) ---
    # Keep signal groups together - split by files, not individual trajectories
    num_signals = len(list_of_amr_signals)
    num_val_signals = int(num_signals * config.VALIDATION_SPLIT)
    if num_val_signals == 0 and num_signals > 1:
        num_val_signals = 1
    num_train_signals = num_signals - num_val_signals
    
    # Random permutation of SIGNAL indices (not trajectory indices)
    signal_indices = torch.randperm(num_signals)
    train_signal_idx = signal_indices[:num_train_signals]
    val_signal_idx = signal_indices[num_train_signals:]
    
    train_signals_set = set(train_signal_idx.tolist())
    val_signals_set = set(val_signal_idx.tolist())

    train_idx = torch.tensor(
        [idx for idx, sig in enumerate(window_signal_ids) if sig in train_signals_set],
        dtype=torch.long,
    )
    val_idx = torch.tensor(
        [idx for idx, sig in enumerate(window_signal_ids) if sig in val_signals_set],
        dtype=torch.long,
    )

    print(
        f"\nTrain/val split: {num_train_signals} signals ({len(train_idx)} windows) for train, "
        f"{num_val_signals} signals ({len(val_idx)} windows) for val"
    )

    # --- 5. Prepare Final Tensors for Training Loop ---
    # The data is kept in trajectory form. The training loop will be responsible
    # for creating the one-step-ahead predictions.

    # --- 6. Package Data for Saving ---
    processed_data = {
        # Data for DataLoaders (split into train/val)
        "train_h_main": h_main[train_idx].cpu(),
        "train_amr_main_norm": amr_main_norm[train_idx].cpu(),
        "train_sin_time_1": sin_time_1_full[train_idx].cpu(),
        "train_sin_time_2": sin_time_2_full[train_idx].cpu(),

        "val_h_main": h_main[val_idx].cpu(),
        "val_amr_main_norm": amr_main_norm[val_idx].cpu(),
        "val_sin_time_1": sin_time_1_full[val_idx].cpu(),
        "val_sin_time_2": sin_time_2_full[val_idx].cpu(),

        # Data for evaluation and plotting
        "val_scalers": [scalers[i] for i in val_idx],
        "val_h_scalers": [h_scalers[i] for i in val_idx],
        "all_train_amr_main_norm": amr_main_norm[train_idx].cpu(),
        "train_scalers": [scalers[i] for i in train_idx],

        # Metadata
        "dt": dt,
        "t_min": t_min.cpu(),
        "t_max": t_max.cpu(),
        "time_grid": time_grid.cpu(),
        "window_length": clipped_main_len,
        "window_signal_ids": torch.tensor(window_signal_ids),
    }

    # --- 7. Save to File ---
    output_path = os.path.join(output_dir, "nanorings_dataset.pt")
    torch.save(processed_data, output_path)
    print(f"Successfully preprocessed and saved data to {output_path}")

if __name__ == "__main__":
    preprocess_and_save_nanorings_data()
