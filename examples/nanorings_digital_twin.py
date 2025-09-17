import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Neural ODE for nanoring array dynamics

    This notebook trains a **Neural Ordinary Differential Equation (Neural ODE)** to model the dynamics of a nanoring array's anisotropic magnetoresistance (AMR) in response to an external magnetic field (H).

    The pipeline has two stages: a one-time preprocessing script and this training notebook.

    ### Preprocessing Script (`examples/preprocess_nanorings_data.py`)

    Raw AMR and H-field trajectories are first loaded and filtered so that the mean AMR signal lies within a narrow, consistent range (though this could be extended to the full signal range). Each trajectory is then split into a transient part and a main sequence, with AMR and H normalised independently.  

    From the transient part, the last few points of the AMR and H signals are extracted as contexts (`c_y`, `c_H`, which are akin to but strictly not initial conditions). These serve as trajectory-specific embeddings that condition the model; they are not initial conditions in the strict sense.  

    To help capture periodicity, heuristic sinusoidal time features $\sin(2\pi t)$ and $\sin(4\pi t)$ are added.

    Finally, all training and validation data, along with contexts, are saved into a single file: `data/processed/nanorings_dataset.pt`. This file is then loaded by the training notebook.

    ### Training Notebook (This File)

    The training notebook loads the processed dataset and trains a `DriftNet` model using the `train_with_validation` function from the `neural_dynamics` library, which handles training, validation, and early stopping.  

    For inference, a single validation trajectory is selected. The model is conditioned on its contexts, and the `rollout_trajectory` function integrates the learned $\frac{dy}{dt}$ to produce a full prediction. The predicted trajectory is then compared to the ground truth.

    ### The Neural ODE Model

    The model learns the function $f$ in the ODE

    $$
    \frac{dy}{dt} = f\Big(y(t), H(t), \sin(\omega t), \sin(2\omega t), c_H, c_y\Big)
    $$

    where:
    - $y(t)$ is the AMR signal (the state being predicted),  
    - $H(t)$ is the external driving magnetic field,  
    - $\sin(\omega t), \sin(2\omega t)$ are periodic time features,  
    - $c_H, c_y$ are the context vectors conditioning predictions on a given trajectory.
    """
    )
    return


@app.cell
def _():
    # --- 1. Setup: Imports and Configuration ---
    import marimo as mo
    import os
    import sys
    from pathlib import Path
    import torch
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from neural_dynamics.config import DEVICE
    from neural_dynamics.core.hyperparameters import NetworkArchitecture
    from neural_dynamics.models.base import DriftNet
    from neural_dynamics.training.base import train_with_validation
    from neural_dynamics.training.evaluation import rollout_trajectory
    from examples.utils.plotting import plot_nanoring_results

    # --- 2. Load Config and Preprocessed Data ---
    class NanoringsHyperparameters:
        """Hyperparameters for the nanorings digital twin experiment."""

        # Training parameters
        LEARNING_RATE: float = 3.4e-4
        NUM_EPOCHS: int = 1024
        BATCH_SIZE: int = 16
        VALIDATION_SPLIT: float = 0.2
        EARLY_STOPPING_PATIENCE: int = NUM_EPOCHS # in principle this could be reduced

        # Data sampling parameters
        PERCENTAGE_OF_FILES_TO_LOAD: float = 1.0
        PERCENTAGE_OF_MAIN_SEQUENCE_TO_USE: float = 0.1
        TARGET_CHANNEL: int = 0

        # System dynamics parameters
        SAMPLE_RATE: int = 3200  # Hz
        H_FIELD_AMPLITUDE_UPDATE_RATE: int = 100  # Timesteps
        TRANSIENT_LENGTH: int = 100
        CONTEXT_POINTS: int = 5

        # Model architecture
        DRIFT_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
            # amr, h, sin(wt), sin(2wt), h_ctx, amr_ctx
            input_size=1 + 1 + 2 + CONTEXT_POINTS + CONTEXT_POINTS,
            hidden_sizes=[128, 128, 128],
            output_size=1,
        )


    config = NanoringsHyperparameters()

    output_dir = os.path.join(
        "examples",
        "output",
        f"epochs-{config.NUM_EPOCHS}_batch_size-{config.BATCH_SIZE}_lr-{config.LEARNING_RATE}",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    data_path = os.path.join("examples", "data", "nanorings_dataset.pt")
    processed_data = torch.load(data_path)
    print(f"Loaded preprocessed data from {data_path}")

    # --- 3. Build Model ---
    initial_drift_net = DriftNet(config.DRIFT_NET_ARCH).to(DEVICE)
    return (
        DEVICE,
        DataLoader,
        TensorDataset,
        config,
        initial_drift_net,
        mo,
        np,
        os,
        output_dir,
        plot_nanoring_results,
        processed_data,
        rollout_trajectory,
        torch,
        train_with_validation,
    )


@app.cell
def _(
    DEVICE,
    DataLoader,
    TensorDataset,
    config,
    initial_drift_net,
    mo,
    np,
    os,
    output_dir,
    processed_data,
    torch,
    train_with_validation,
):
    # --- 4. Train the Model ---
    # Move data tensors to DEVICE once to avoid per-batch transfers
    for key in processed_data:
        if isinstance(processed_data[key], torch.Tensor):
            processed_data[key] = processed_data[key].to(DEVICE)

    # Trajectory-level datasets; per-batch vectorisation flattens (B, T-1) -> (B*(T-1), ...)
    train_dataset = TensorDataset(
        processed_data["train_h_context"],
        processed_data["train_amr_context"],
        processed_data["train_h_main"],
        processed_data["train_amr_main_norm"],
        processed_data["train_sin_time_1"],
        processed_data["train_sin_time_2"],
    )
    val_dataset = TensorDataset(
        processed_data["val_h_context"],
        processed_data["val_amr_context"],
        processed_data["val_h_main"],
        processed_data["val_amr_main_norm"],
        processed_data["val_sin_time_1"],
        processed_data["val_sin_time_2"],
    )

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    def batch_preparation_fn(raw_batch, device):
        """Convert a batch of full trajectories into per-step supervised pairs.

        This prepares inputs for a drift network f that predicts dy/dt from
        features at time t. We construct one training example per time step
        and trajectory (horizon-of-one), and use finite differences as the
        supervision target: (y[t+1] - y[t]) / dt.

        Args:
            raw_batch: Tuple of tensors with shapes:
                - h_ctx:       [B, C_h]  trajectory-level H contexts
                - amr_ctx:     [B, C_y]  trajectory-level AMR contexts
                - h_main:      [B, T]    H time series
                - amr_main:    [B, T]    AMR time series (target signal y)
                - sin_t1:      [B, T]    sin(2π t) feature
                - sin_t2:      [B, T]    sin(4π t) feature
            device: Torch device; tensors are already on this device.

        Returns:
            - net_input:       [B*(T-1), F] concatenated features per time step
            - true_derivatives:[B*(T-1), 1] finite-difference dy/dt target
        """
        # Unpack batch; all tensors are on DEVICE already.
        h_ctx, amr_ctx, h_main, amr_main, sin_t1, sin_t2 = raw_batch

        # Current and next AMR values: shapes [B, T-1]
        current = amr_main[:, :-1]  # y[t]
        target = amr_main[:, 1:]    # y[t+1]
        B, T = current.shape        # B: trajectories in batch, T: steps per trajectory minus one

        # Stack per-time-step features then flatten from [B, T, 4] -> [B*T, 4]
        step_features = torch.stack(
            [current, h_main[:, :-1], sin_t1[:, :-1], sin_t2[:, :-1]], dim=-1
        ).reshape(B * T, -1).contiguous()

        # Concatenate contexts once [B, C_h+C_y], broadcast across time to [B, T, C], then flatten
        ctx = torch.cat([h_ctx, amr_ctx], dim=1)
        ctx_flat = ctx.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1).contiguous()

        # Final model input per step: [amr, h, sin1, sin2, h_ctx..., amr_ctx...]
        net_input = torch.cat([step_features, ctx_flat], dim=1)

        # Supervision: true dy/dt via finite difference (teacher forcing)
        # This trains the drift network directly on derivatives, avoiding running an
        # ODE solver inside the training loop and improving stability and throughput.
        dt = processed_data["dt"]
        if isinstance(dt, torch.Tensor):
            dt = dt.item()
        true_derivatives = ((target - current).reshape(-1, 1) / float(dt)).contiguous()

        return net_input, true_derivatives

    # Run the training loop
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

    # --- Save Losses for Debugging ---
    np.savetxt(
        os.path.join(output_dir, "losses.txt"),
        np.column_stack([training_losses, validation_losses]),
        header="Training_Loss,Validation_Loss",
    )

    mo.show_code()
    return trained_drift_net, training_losses, validation_losses


@app.cell
def _(
    DEVICE,
    np,
    os,
    output_dir,
    plot_nanoring_results,
    processed_data,
    rollout_trajectory,
    torch,
    trained_drift_net,
    training_losses,
    validation_losses,
):
    # --- 5. Evaluate and Visualize ---
    trained_drift_net.eval()

    # Get one sample from the validation set for inference
    traj_idx_to_plot = 0
    val_min, val_max = processed_data["val_scalers"][traj_idx_to_plot]
    h_min_val, h_max_val = processed_data["val_h_scalers"][traj_idx_to_plot]

    with torch.no_grad():
        # Get the necessary data for this trajectory from the processed file
        y0_eval = (
            processed_data["val_amr_main_norm"][traj_idx_to_plot, 0]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        h_context_eval = processed_data["val_h_context"][traj_idx_to_plot].unsqueeze(0).to(DEVICE)
        amr_context_eval = processed_data["val_amr_context"][traj_idx_to_plot].unsqueeze(0).to(DEVICE)
        val_h_main_traj = processed_data["val_h_main"][traj_idx_to_plot].to(DEVICE)

        # Define the example-specific drift function that captures context
        def drift_func_eval(t, y):
            idx = min(
                int((t - processed_data["t_min"].item()) / processed_data["dt"]),
                val_h_main_traj.shape[0] - 1,
            )
            h_t = val_h_main_traj[idx].unsqueeze(0).unsqueeze(0)

            t_norm = (
                t
                - processed_data["t_min"].item()
            ) / (processed_data["t_max"].item() - processed_data["t_min"].item())
            t_norm_tensor = torch.full((1, 1), t_norm, device=y.device, dtype=y.dtype)
            sin_t_1 = torch.sin(2 * torch.pi * t_norm_tensor)
            sin_t_2 = torch.sin(4 * torch.pi * t_norm_tensor)

            net_input = torch.cat(
                [y, h_t, sin_t_1, sin_t_2, h_context_eval, amr_context_eval], dim=1
            )
            return trained_drift_net(net_input)

        # Use the generic rollout function to get the predicted trajectory
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

        predicted_sequence = pred_traj_norm.squeeze() * (val_max - val_min) + val_min
        true_val_sequence = (
            processed_data["val_amr_main_norm"][traj_idx_to_plot]
            * (val_max - val_min)
            + val_min
        )

        h_field_unnorm = (
            val_h_main_traj * (h_max_val - h_min_val + 1e-8) + h_min_val
        )

        all_train_seqs = []
        for i, (train_min, train_max) in enumerate(processed_data["train_scalers"]):
            train_seq_norm = processed_data["all_train_amr_main_norm"][i]
            train_sequence = train_seq_norm * (train_max - train_min) + train_min
            all_train_seqs.append(train_sequence.cpu().numpy())

        np.savetxt(
            os.path.join(output_dir, "predicted_trajectory.txt"),
            predicted_sequence.cpu().numpy(),
        )
        np.savetxt(
            os.path.join(output_dir, "true_trajectory.txt"),
            true_val_sequence.cpu().numpy(),
        )

    plot_nanoring_results(
        training_losses=training_losses,
        validation_losses=validation_losses,
        time_axis=processed_data["time_grid"].cpu().numpy(),
        true_sequence=true_val_sequence,
        predicted_sequence=predicted_sequence,
        h_field=h_field_unnorm,
        all_train_sequences=np.array(all_train_seqs),
        output_dir=output_dir,
    )
    return


if __name__ == "__main__":
    app.run()
