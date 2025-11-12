import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import os
    import numpy as np
    import torch

    from neural_dynamics.config import DEVICE
    from neural_dynamics.core.hyperparameters import (
        NetworkArchitecture,
    )
    from neural_dynamics.models.base import DriftNet
    from neural_dynamics.training.base import train_with_validation
    from neural_dynamics.training.evaluation import rollout_trajectory
    from examples.systems.parameters.nanorings import NanoringsHyperparameters
    from torch.utils.data import DataLoader, TensorDataset
    # from examples.utils.data_preprocessing import (
    #     baseline_shift_and_normalise,
    #     inverse_baseline_shift_and_normalise,
    # )

    def baseline_shift_and_normalise(dataset):
        return dataset

    def inverse_baseline_shift_and_normalise(sequence, _):
        return sequence

    from examples.utils.plotting import plot_nanoring_results
    
    return (
        DataLoader,
        DriftNet,
        NetworkArchitecture,
        NanoringsHyperparameters,
        TensorDataset,
        DEVICE,
        np,
        os,
        plot_nanoring_results,
        rollout_trajectory,
        torch,
        train_with_validation,
    )


@app.cell
def _(NanoringsHyperparameters, Path, torch):
    config = NanoringsHyperparameters()  # Use default 1024 epochs
    data_path = Path("examples/data/nanorings_dataset.pt")
    raw_processed_data = torch.load(data_path)

    # The current dataset is min-max normalized (values ~0.907 with tiny variance)
    # We need to properly standardize it for neural network training
    # Store original scalers for denormalization later
    processed_data = {}

    for data_key in raw_processed_data:
        if isinstance(raw_processed_data[data_key], torch.Tensor):
            processed_data[data_key] = raw_processed_data[data_key].to("cpu")
        else:
            processed_data[data_key] = raw_processed_data[data_key]

    # Standardize AMR training data (mean=0, std=1)
    amr_train_raw = processed_data["train_amr_main_norm"]
    amr_val_raw = processed_data["val_amr_main_norm"]

    # Compute global statistics from training data only
    amr_mean = amr_train_raw.mean()
    amr_std = amr_train_raw.std()

    print(f"Original AMR stats - mean: {amr_mean:.6f}, std: {amr_std:.6f}")

    # Standardize both train and val using training statistics
    processed_data["train_amr_main_norm"] = (amr_train_raw - amr_mean) / amr_std
    processed_data["val_amr_main_norm"] = (amr_val_raw - amr_mean) / amr_std
    processed_data["all_train_amr_main_norm"] = processed_data["train_amr_main_norm"]

    # Update scalers to account for this standardization
    # New scalers need to reverse: x_orig = x_std * amr_std + amr_mean
    # Then apply old scalers: x_final = x_orig * (old_max - old_min) + old_min
    def _augment_scalers(scalers):
        augmented = []
        for scaler in scalers:
            if len(scaler) == 2:
                extra_1, extra_2 = scaler
            else:
                extra_1, extra_2 = scaler
            augmented.append((amr_mean, amr_std, extra_1, extra_2))
        return augmented

    processed_data["train_scalers"] = _augment_scalers(processed_data["train_scalers"])
    processed_data["val_scalers"] = _augment_scalers(processed_data["val_scalers"])

    print(f"Standardized AMR stats - mean: {processed_data['train_amr_main_norm'].mean():.6f}, std: {processed_data['train_amr_main_norm'].std():.6f}")

    dt_value = processed_data.get("dt")
    if isinstance(dt_value, torch.Tensor):
        dt = float(dt_value.item())
    else:
        dt = float(dt_value)
    return amr_mean, amr_std, config, dt, processed_data


@app.cell
def _(DEVICE, DriftNet, NetworkArchitecture, config, os):
    # Match the working code architecture exactly
    # Input: [AMR, H, sin(ωt), sin(2ωt)] = 4 features
    drift_net_arch = NetworkArchitecture(
        input_size=1 + 1 + 2,
        hidden_sizes=[128, 128, 128],
        output_size=1,
    )
    
    initial_drift_net = DriftNet(drift_net_arch).to(DEVICE)
    
    output_directory = os.path.join(
        "examples",
        "output",
        f"nanorings_ode_epochs-{config.NUM_EPOCHS}_bs-{config.BATCH_SIZE}_lr-{config.LEARNING_RATE}",
    )
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory: {output_directory}")
    
    return drift_net_arch, initial_drift_net, output_directory


@app.cell
def _(
    DEVICE,
    DataLoader,
    TensorDataset,
    config,
    initial_drift_net,
    np,
    os,
    output_directory,
    processed_data,
    torch,
    train_with_validation,
):
    # Move all data to DEVICE to avoid per-batch transfers
    for tensor_key in processed_data:
        if isinstance(processed_data[tensor_key], torch.Tensor):
            processed_data[tensor_key] = processed_data[tensor_key].to(DEVICE)

    # Create trajectory-level datasets
    train_dataset = TensorDataset(
        processed_data["train_h_main"],
        processed_data["train_amr_main_norm"],
        processed_data["train_sin_time_1"],
        processed_data["train_sin_time_2"],
    )
    val_dataset = TensorDataset(
        processed_data["val_h_main"],
        processed_data["val_amr_main_norm"],
        processed_data["val_sin_time_1"],
        processed_data["val_sin_time_2"],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0
    )

    def batch_preparation_fn(raw_batch, device):
        """Convert trajectories into per-step supervised pairs using finite differences.
        
        Args:
            raw_batch: Tuple of tensors with shapes:
                - h_main:      [B, T]    H time series
                - amr_main:    [B, T]    AMR time series (target signal y)
                - sin_t1:      [B, T]    sin(2π t) feature
                - sin_t2:      [B, T]    sin(4π t) feature
            device: Torch device (already on this device).
            
        Returns:
            - net_input:       [B*(T-1), F] concatenated features per time step
            - true_derivatives:[B*(T-1), 1] finite-difference dy/dt target
        """
        h_main, amr_main, sin_t1, sin_t2 = raw_batch

        # Current and next AMR values: shapes [B, T-1]
        current = amr_main[:, :-1]  # y[t]
        target = amr_main[:, 1:]    # y[t+1]
        B, T = current.shape

        # Stack per-time-step features then flatten from [B, T, 4] -> [B*T, 4]
        step_features = torch.stack(
            [current, h_main[:, :-1], sin_t1[:, :-1], sin_t2[:, :-1]], dim=-1
        ).reshape(B * T, -1).contiguous()

        # Concatenate contexts [B, C_h+C_y], broadcast to [B, T, C], then flatten
        net_input = step_features

        # Supervision: finite difference dy/dt (teacher forcing)
        dt = processed_data["dt"]
        if isinstance(dt, torch.Tensor):
            dt = dt.item()
        true_derivatives = ((target - current).reshape(-1, 1) / float(dt)).contiguous()

        return net_input, true_derivatives

    # Run training
    trained_drift_net, training_losses, validation_losses = train_with_validation(
        model=initial_drift_net,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        device=DEVICE,
        batch_preparation_fn=batch_preparation_fn,
    )

    # Save losses
    np.savetxt(
        os.path.join(output_directory, "losses.txt"),
        np.column_stack([training_losses, validation_losses]),
        header="Training_Loss,Validation_Loss",
    )
    print(f"✓ Training complete. Losses saved to {output_directory}/losses.txt")
    return trained_drift_net, training_losses, validation_losses


@app.cell
def _(
    DEVICE,
    np,
    os,
    output_directory,
    plot_nanoring_results,
    processed_data,
    rollout_trajectory,
    torch,
    trained_drift_net,
    training_losses,
    validation_losses,
):
    # Evaluate on a validation trajectory
    trained_drift_net.eval()
    
    traj_idx_to_plot = 0
    # New scaler format: (amr_mean, amr_std, extra_1, extra_2)
    h_mean_val, h_std_val = processed_data["val_h_scalers"][traj_idx_to_plot]

    with torch.no_grad():
        # Get initial state
        y0_eval = (
            processed_data["val_amr_main_norm"][traj_idx_to_plot, 0]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        val_h_main_traj = processed_data["val_h_main"][traj_idx_to_plot].to(DEVICE)

        # Define trajectory-specific drift function
        def drift_func_eval(t, y):
            idx = min(
                int((t - processed_data["t_min"].item()) / processed_data["dt"]),
                val_h_main_traj.shape[0] - 1,
            )
            h_t = val_h_main_traj[idx].unsqueeze(0).unsqueeze(0)

            t_norm = (t - processed_data["t_min"].item()) / (
                processed_data["t_max"].item() - processed_data["t_min"].item()
            )
            t_norm_tensor = torch.full((1, 1), t_norm, device=y.device, dtype=y.dtype)
            sin_t_1 = torch.sin(2 * torch.pi * t_norm_tensor)
            sin_t_2 = torch.sin(4 * torch.pi * t_norm_tensor)

            net_input = torch.cat([y, h_t, sin_t_1, sin_t_2], dim=1)
            return trained_drift_net(net_input)

        # Rollout trajectory
        eval_num_steps_required = processed_data["val_amr_main_norm"].shape[1] - 1
        eval_final_time = (
            processed_data["t_min"].item() + eval_num_steps_required * processed_data["dt"]
        )

        _, pred_traj_norm = rollout_trajectory(
            drift_function=drift_func_eval,
            initial_state=y0_eval,
            initial_time=processed_data["t_min"].item(),
            final_time=eval_final_time,
            timestep=processed_data["dt"],
        )

        # Denormalize predictions
        # Scaler format: (amr_mean, amr_std, old_min, old_max)
        # First undo standardization: x = x_std * amr_std + amr_mean
        amr_mean_val, amr_std_val, _, _ = processed_data["val_scalers"][traj_idx_to_plot]
        amr_mean_val = float(amr_mean_val)
        amr_std_val = float(amr_std_val)
        predicted_sequence = pred_traj_norm.squeeze() * amr_std_val + amr_mean_val
        true_val_sequence = (
            processed_data["val_amr_main_norm"][traj_idx_to_plot]
            * amr_std_val
            + amr_mean_val
        )
        # Denormalize H-field using stored mean/std
        h_field_unnorm = val_h_main_traj * float(h_std_val) + float(h_mean_val)

        # Denormalize training sequences
        all_train_seqs = []
        for i, scaler_tuple in enumerate(processed_data["train_scalers"]):
            # New format: (amr_mean, amr_std, old_min, old_max)
            # First undo standardization: x = x_std * amr_std + amr_mean
            amr_mean_tr, amr_std_tr, _, _ = scaler_tuple
            amr_mean_tr = float(amr_mean_tr)
            amr_std_tr = float(amr_std_tr)
            train_seq_norm = processed_data["all_train_amr_main_norm"][i]
            train_sequence = train_seq_norm * amr_std_tr + amr_mean_tr
            all_train_seqs.append(train_sequence.cpu().numpy())

        predicted_sequence_cpu = predicted_sequence.cpu()
        true_val_sequence_cpu = true_val_sequence.cpu()
        h_field_cpu = h_field_unnorm.cpu()

        # Save results
        np.savetxt(
            os.path.join(output_directory, "predicted_trajectory.txt"),
            predicted_sequence_cpu.numpy(),
        )
        np.savetxt(
            os.path.join(output_directory, "true_trajectory.txt"),
            true_val_sequence_cpu.numpy(),
        )

    # Plot results - use current function signature (expects ODE/SDE predictions)
    # For now we only have ODE predictions, so use them for both
    plot_nanoring_results(
        time_axis=processed_data["time_grid"].cpu().numpy(),
        true_sequence=true_val_sequence_cpu,
        predicted_sequence_ode=predicted_sequence_cpu,
        predicted_sequence_sde=predicted_sequence_cpu,  # Use same for now since we don't have SDE
        h_field=h_field_cpu,
        all_train_sequences=np.array(all_train_seqs),
        output_dir=output_directory,
    )
    print(f"✓ Results plotted and saved to {output_directory}")


if __name__ == "__main__":
    app.run()
