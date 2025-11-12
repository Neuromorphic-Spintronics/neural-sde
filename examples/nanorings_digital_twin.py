import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    """Import dependencies and configure project paths."""
    import sys
    from pathlib import Path

    # Ensure project root is in path for local imports
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    import os
    import numpy as np
    import torch
    import matplotlib.pyplot as plt

    from neural_dynamics.config import DEVICE
    from neural_dynamics.core.utils import (
        COLOURS,
        compare_statistics,
        compute_statistics,
        sample_sde_rollouts,
    )
    from neural_dynamics.core.hyperparameters import NetworkArchitecture
    from neural_dynamics.models.base import DriftNet
    from neural_dynamics.models.ode import NeuralODE
    from neural_dynamics.models.sde import NeuralSDE
    from neural_dynamics.training.base import train_with_validation
    from neural_dynamics.training.evaluation import rollout_trajectory
    from neural_dynamics.utils.training_helpers import (
        DEFAULT_METRIC_FILENAMES,
        DEFAULT_MODEL_FILENAMES,
    )
    from examples.systems.parameters.nanorings import NanoringsHyperparameters
    from examples.utils.plotting import plot_nanoring_results, setup_matplotlib_style, finalise_plot
    from torch.utils.data import DataLoader, TensorDataset
    return (
        COLOURS,
        DEFAULT_METRIC_FILENAMES,
        DEFAULT_MODEL_FILENAMES,
        DEVICE,
        DataLoader,
        DriftNet,
        NanoringsHyperparameters,
        NetworkArchitecture,
        NeuralODE,
        NeuralSDE,
        Path,
        TensorDataset,
        compare_statistics,
        compute_statistics,
        finalise_plot,
        np,
        os,
        plot_nanoring_results,
        plt,
        rollout_trajectory,
        sample_sde_rollouts,
        setup_matplotlib_style,
        torch,
        train_with_validation,
    )


@app.cell
def _(NanoringsHyperparameters, Path, torch):
    """
    Load preprocessed data (already standardized per-trajectory during preprocessing).
    Data is standardized: zero mean, unit variance per trajectory.
    """
    # Load hyperparameters and dataset
    config = NanoringsHyperparameters(NUMBER_OF_GAN_EPOCHS=1024, NUMBER_OF_EPOCHS=1024)
    data_path = Path("examples/data/nanorings_dataset.pt")
    raw_data = torch.load(data_path)

    processed_data = {}
    for key, value in raw_data.items():
        if isinstance(value, torch.Tensor):
            processed_data[key] = value.to("cpu")
        else:
            processed_data[key] = value

    print(f"\nLoaded dataset with {processed_data['train_amr_main_norm'].shape[0]} training and {processed_data['val_amr_main_norm'].shape[0]} validation trajectories")
    print(f"Sequence length: {processed_data['train_amr_main_norm'].shape[1]} timesteps")
    print("\nData is already standardized per-trajectory (zero mean, unit variance)")
    print(f"Training AMR - mean: {processed_data['train_amr_main_norm'].mean():.4f}, std: {processed_data['train_amr_main_norm'].std():.4f}")
    print(f"Training H-field - mean: {processed_data['train_h_main'].mean():.4f}, std: {processed_data['train_h_main'].std():.4f}")
    total_windows = (
        processed_data["train_amr_main_norm"].shape[0]
        + processed_data["val_amr_main_norm"].shape[0]
    )
    window_length = processed_data.get(
        "window_length", processed_data["train_amr_main_norm"].shape[1]
    )
    dt = (
        processed_data["dt"].item()
        if hasattr(processed_data["dt"], "item")
        else float(processed_data["dt"])
    )
    print(f"Window length: {window_length} steps (~{window_length * dt:.4f}s)")
    print(f"Total windows considered (train+val): {total_windows}")
    print(
        f"Total samples seen: {total_windows * window_length} "
        f"(train+val across all signals)"
    )
    window_signal_ids = processed_data.get("window_signal_ids")
    if window_signal_ids is not None:
        unique_signals = window_signal_ids.unique()
        print(f"Distinct signals contributing windows: {unique_signals.numel()}")
    return config, processed_data


@app.cell
def _(COLOURS, plt, processed_data, setup_matplotlib_style):
    """Visualise per-signal standardisation with separate subplots."""
    setup_matplotlib_style()

    print("\n" + "="*60)
    print("Visualising Per-Signal Standardisation")
    print("="*60)

    train_amr = processed_data["train_amr_main_norm"].numpy()
    time_grid_viz = processed_data["time_grid"].numpy()

    # Each signal has 100 repetitions
    reps_per_signal = 100
    num_signals = train_amr.shape[0] // reps_per_signal

    print(f"Total trajectories: {train_amr.shape[0]}")
    print(f"Number of signals: {num_signals}")
    print(f"Repetitions per signal: {reps_per_signal}")

    # Create square subplots: one per signal
    fig_height = 4 * num_signals
    fig, axes = plt.subplots(num_signals, 1, figsize=(8, fig_height), sharex=True)
    if num_signals == 1:
        axes = [axes]

    for sig_idx in range(num_signals):
        ax = axes[sig_idx]
        start_idx = sig_idx * reps_per_signal
        end_idx = start_idx + reps_per_signal
        signal_data = train_amr[start_idx:end_idx]

        # Plot all 100 repetitions with transparency
        for i in range(reps_per_signal):
            ax.plot(time_grid_viz, signal_data[i], alpha=0.1, color='black', linewidth=0.5)

        # Plot mean and std envelope for this signal
        mean_signal = signal_data.mean(axis=0)
        std_signal = signal_data.std(axis=0)
        ax.plot(time_grid_viz, mean_signal, color=COLOURS[3], linewidth=2, label='Mean', zorder=10)
        ax.fill_between(time_grid_viz, 
                         mean_signal - std_signal, 
                         mean_signal + std_signal, 
                         alpha=0.3, color=COLOURS[3], label=r'$\pm 1\sigma$')

        ax.set_ylabel(r'Standardised AMR')
        ax.legend(loc='upper right')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    axes[-1].set_xlabel(r'Time [s]')
    plt.tight_layout()
    plt.show()

    print("\nOverall Dataset Statistics:")
    print(f"  Mean across all points: {train_amr.mean():.4f} (expected: ~0.0)")
    print(f"  Std across all points: {train_amr.std():.4f} (expected: ~1.0)")

    print("\nPer-Signal Statistics:")
    for sig_idx in range(num_signals):
        start_idx = sig_idx * reps_per_signal
        end_idx = start_idx + reps_per_signal
        signal_data = train_amr[start_idx:end_idx]
        print(f"  Signal {sig_idx+1}: μ={signal_data.mean():.6f}, σ={signal_data.std():.6f}")

    return


@app.cell
def _(DEVICE, DriftNet, NetworkArchitecture, config, os):
    """Define drift network architecture for Neural ODE.

    The network learns the deterministic dynamics: dx/dt = f(x, t, u)
    Input features: AMR(1), H-field(1), time encoding(2)
    """
    drift_net_arch = NetworkArchitecture(
        input_size=1 + 1 + 2,  # amr, h, sin(wt), sin(2wt)
        hidden_sizes=[512, 512, 512, 256],  # Large capacity for complex AMR dynamics
        output_size=1,  # Single state: AMR signal
    )

    initial_drift_net = DriftNet(drift_net_arch).to(DEVICE)
    print(f"Drift network: {sum(p.numel() for p in initial_drift_net.parameters()):,} parameters")

    output_directory = os.path.join(
        "examples/output",
        f"nanorings_ode_epochs-{config.NUMBER_OF_EPOCHS}_bs-{config.BATCH_SIZE}_lr-{config.LEARNING_RATE}",
    )
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output directory: {output_directory}")
    return drift_net_arch, initial_drift_net, output_directory


@app.cell
def _(
    DEFAULT_METRIC_FILENAMES,
    DEFAULT_MODEL_FILENAMES,
    DEVICE,
    DataLoader,
    TensorDataset,
    config,
    initial_drift_net,
    os,
    output_directory,
    processed_data,
    torch,
    train_with_validation,
):
    """Train Neural ODE.

    The ODE learns: dx/dt = f(x, t, u).
    """
    # Check for saved model checkpoint
    ode_model_path = os.path.join(output_directory, DEFAULT_MODEL_FILENAMES["neural_ode"])
    train_losses_path = os.path.join(output_directory, DEFAULT_METRIC_FILENAMES["train_losses"])
    val_losses_path = os.path.join(output_directory, DEFAULT_METRIC_FILENAMES["val_losses"])

    if os.path.exists(ode_model_path) and os.path.exists(train_losses_path) and os.path.exists(val_losses_path):
        print("Loading existing Neural ODE from checkpoint...")
        trained_drift_net = initial_drift_net
        trained_drift_net.load_state_dict(torch.load(ode_model_path, map_location=DEVICE))
        trained_drift_net.to(DEVICE)

        # Load training history
        training_losses = list(torch.load(train_losses_path, map_location="cpu"))
        validation_losses = list(torch.load(val_losses_path, map_location="cpu"))
        print(f"Loaded Neural ODE - final train loss: {training_losses[-1]:.6f}, val loss: {validation_losses[-1]:.6f}")
    else:
        print("Training Neural ODE from scratch...")
        # Move all tensors to device once
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

            # Final model input: [amr, h, sin1, sin2]
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
            num_epochs=config.NUMBER_OF_EPOCHS,
            learning_rate=config.LEARNING_RATE,
            device=DEVICE,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            batch_preparation_fn=batch_preparation_fn,
        )

        # Save the trained model and losses
        os.makedirs(output_directory, exist_ok=True)
        torch.save(trained_drift_net.state_dict(), ode_model_path)
        torch.save(training_losses, train_losses_path)
        torch.save(validation_losses, val_losses_path)
        print(f"✓ Neural ODE saved to {ode_model_path}")
        print(f"✓ Metrics saved to {train_losses_path} and {val_losses_path}")
    return trained_drift_net, training_losses, validation_losses


@app.cell
def _(
    COLOURS,
    DEVICE,
    np,
    plt,
    processed_data,
    rollout_trajectory,
    setup_matplotlib_style,
    torch,
    trained_drift_net,
):
    """Evaluate Neural ODE on validation data (intermediate check)."""
    setup_matplotlib_style()
    
    print("\n" + "="*60)
    print("Neural ODE Evaluation (Standardised Space)")
    print("="*60)

    trained_drift_net.eval()
    traj_idx_ode_eval = 0

    with torch.no_grad():
        # Extract initial state (standardised space)
        y0_ode = (
            processed_data["val_amr_main_norm"][traj_idx_ode_eval, 0]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        val_h_main_ode = processed_data["val_h_main"][traj_idx_ode_eval].to(DEVICE)

        # Define trajectory-specific drift function
        def drift_func_ode(t, y):
            idx = min(
                int((t - processed_data["t_min"].item()) / processed_data["dt"]),
                val_h_main_ode.shape[0] - 1,
            )
            h_t = val_h_main_ode[idx].unsqueeze(0).unsqueeze(0)
            
            # Compute time encoding on the fly (same as training)
            t_norm = (t - processed_data["t_min"].item()) / (
                processed_data["t_max"].item() - processed_data["t_min"].item()
            )
            t_norm_tensor = torch.full((1, 1), t_norm, device=y.device, dtype=y.dtype)
            sin_t_1 = torch.sin(2 * torch.pi * t_norm_tensor)
            sin_t_2 = torch.sin(4 * torch.pi * t_norm_tensor)
            
            net_input = torch.cat([y, h_t, sin_t_1, sin_t_2], dim=1)
            return trained_drift_net(net_input)

        # Rollout Neural ODE prediction
        time_grid_ode = processed_data["time_grid"].to(DEVICE)
        _, predicted_ode_traj = rollout_trajectory(
            drift_function=drift_func_ode,
            initial_state=y0_ode,
            initial_time=time_grid_ode[0].item(),
            final_time=time_grid_ode[-1].item(),
            timestep=processed_data["dt"],
        )
        predicted_ode = predicted_ode_traj.squeeze()

    # Plot ODE results
    time_np_ode_plot = processed_data["time_grid"].cpu().numpy()
    true_amr_np_ode_plot = processed_data["val_amr_main_norm"][traj_idx_ode_eval].cpu().numpy()
    pred_ode_np_plot = predicted_ode.cpu().numpy()
    h_field_np_ode_plot = val_h_main_ode.cpu().numpy()
    
    # Handle potential length mismatch (rollout may return N-1 or N+1 points)
    min_len = min(len(time_np_ode_plot), len(pred_ode_np_plot), len(true_amr_np_ode_plot), len(h_field_np_ode_plot))
    time_np_ode_plot = time_np_ode_plot[:min_len]
    true_amr_np_ode_plot = true_amr_np_ode_plot[:min_len]
    pred_ode_np_plot = pred_ode_np_plot[:min_len]
    h_field_np_ode_plot = h_field_np_ode_plot[:min_len]

    fig_ode, (ax1_ode, ax2_ode) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot H-field
    ax1_ode.plot(time_np_ode_plot, h_field_np_ode_plot, color=COLOURS[1], linewidth=1.5, label=r'$H(t)$')
    ax1_ode.set_ylabel(r'$H$ (Standardised)')
    ax1_ode.legend(loc='upper right')
    ax1_ode.spines['top'].set_visible(False)
    ax1_ode.spines['right'].set_visible(False)

    # Plot AMR: ground truth and ODE prediction
    ax2_ode.plot(time_np_ode_plot, true_amr_np_ode_plot, color=COLOURS[3], linewidth=2, label='Ground Truth', alpha=0.8)
    ax2_ode.plot(time_np_ode_plot, pred_ode_np_plot, color=COLOURS[0], linewidth=2, linestyle='--', label='Neural ODE')
    ax2_ode.set_xlabel(r'Time [s]')
    ax2_ode.set_ylabel(r'AMR (Standardised)')
    ax2_ode.legend(loc='upper right')
    ax2_ode.spines['top'].set_visible(False)
    ax2_ode.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    
    print("✓ Neural ODE prediction complete")
    # Use the trimmed version for MSE calculation
    predicted_ode_trimmed = predicted_ode[:min_len]
    true_ode_trimmed = processed_data['val_amr_main_norm'][traj_idx_ode_eval, :min_len].to(DEVICE)
    print(f"  MSE: {((predicted_ode_trimmed - true_ode_trimmed)**2).mean().item():.6f}")
    
    return predicted_ode, traj_idx_ode_eval


@app.cell
def _(
    DEFAULT_METRIC_FILENAMES,
    DEFAULT_MODEL_FILENAMES,
    DEVICE,
    NetworkArchitecture,
    NeuralODE,
    NeuralSDE,
    config,
    drift_net_arch,
    os,
    output_directory,
    processed_data,
    torch,
    trained_drift_net,
    training_losses,
    validation_losses,
):
    """Train Neural SDE

    Uses WGAN-GP to learn diffusion term: dx = f(x,t,u)dt + sigma(x,t,u) circ dW
    The critic discriminates between real and generated trajectory windows.
    """
    from neural_dynamics.models.sde import DiffusionNet, CriticNet, fit_neural_sde_gan

    print("\n" + "="*60)
    print("Training Neural SDE (Stochastic Diffusion)")
    print("="*60)

    # Check for saved SDE model checkpoint
    sde_model_path = os.path.join(output_directory, DEFAULT_MODEL_FILENAMES["neural_sde"])
    generator_losses_path = os.path.join(output_directory, DEFAULT_METRIC_FILENAMES["generator_losses"])
    critic_losses_path = os.path.join(output_directory, DEFAULT_METRIC_FILENAMES["critic_losses"])
    drift_losses_path = os.path.join(output_directory, DEFAULT_METRIC_FILENAMES["drift_losses_sde"])
    sde_metrics_exist = os.path.exists(generator_losses_path) and os.path.exists(critic_losses_path) and os.path.exists(drift_losses_path)

    # Freeze drift network (only train diffusion term, keep ODE fixed)
    trained_drift_net.eval()
    for param in trained_drift_net.parameters():
        param.requires_grad = False

    # Prepare standardised trajectories for SDE training
    train_amr_for_sde = processed_data["train_amr_main_norm"].to(DEVICE)
    time_grid_for_sde = processed_data["time_grid"].to(DEVICE)

    # Reshape for SDE: [num_trajectories, num_timesteps, state_dim]
    trajectories_for_sde = train_amr_for_sde.unsqueeze(-1)

    print(f"Training on {trajectories_for_sde.shape[0]} trajectories")
    print(f"Time grid: {time_grid_for_sde.shape[0]} timesteps")
    print(f"Trajectory shape: {trajectories_for_sde.shape}")

    # Convert config to unified Hyperparameters object
    hyperparameters = config.to_hyperparameters()
    print(f"\nCritic window size: {config.CRITIC_WINDOW_SIZE} timesteps")
    print(f"Critic updates per generator update: {config.CRITIC_UPDATES}")

    # Wrap trained drift network in NeuralODE container
    neural_ode = NeuralODE(
        drift_net=trained_drift_net,
        hyperparameters=hyperparameters,
        time_grid=time_grid_for_sde,
        training_losses=training_losses,
        validation_losses=validation_losses,
    )
    print(f"Neural ODE wrapper: {sum(p.numel() for p in neural_ode.drift_net.parameters()):,} parameters")

    # Create diffusion network (learns stochastic term: sigma(x,t,u) circ dW)
    # Matches drift architecture for balanced capacity
    diffusion_arch = NetworkArchitecture(
        input_size=drift_net_arch.input_size,  # Same inputs as drift: 14 features
        hidden_sizes=drift_net_arch.hidden_sizes,  # Match drift: [512, 512, 512, 256]
        output_size=1,  # state_dim × noise_dim = 1 × 1
    )

    diffusion_net = DiffusionNet(
        diffusion_arch,
        state_dimension=1,  # 1D AMR signal
        noise_dimension=1,  # 1D Noise motion
        device=DEVICE,
    )
    print(f"Diffusion network: {sum(p.numel() for p in diffusion_net.parameters()):,} parameters")

    # Create smaller critic network to prevent collapse
    # Perhaps the critic's capacity must be much smaller than what it discriminates to avoid rapid overfitting (unsure)
    critic_window = config.CRITIC_WINDOW_SIZE

    critic_net = CriticNet(
        config.CRITIC_NET_ARCH,  # Small architecture: [64, 32] hidden units
        trajectory_length=critic_window,
        device=DEVICE,
    )
    print(f"Critic network: {sum(p.numel() for p in critic_net.parameters()):,} parameters (deliberately small!)")
    print(f"  Input: {critic_window} timesteps → Output: scalar score")

    # Manually instantiate Neural SDE with the trained drift network from the ODE
    neural_sde = NeuralSDE(
        neural_ode.drift_net,  # Use drift from the NeuralODE object
        diffusion_net,
        hyperparameters,
        critic_net=critic_net,
    ).to(DEVICE)

    # Check if we can load existing SDE model
    if os.path.exists(sde_model_path) and sde_metrics_exist:
        print(f"Loading existing Neural SDE from {sde_model_path}...")
        neural_sde.load_state_dict(torch.load(sde_model_path, map_location=DEVICE))
        neural_sde.to(DEVICE)

        # Load SDE metrics
        generator_losses = list(torch.load(generator_losses_path, map_location="cpu"))
        critic_losses = list(torch.load(critic_losses_path, map_location="cpu"))
        drift_losses = list(torch.load(drift_losses_path, map_location="cpu"))
        print(f"Loaded Neural SDE with {len(generator_losses)} generator epochs")
    else:
        print("Training Neural SDE from scratch...")
        # Run GAN training to fit the diffusion term
        # Pass full trajectories - random windows will be sampled internally
        generator_losses, critic_losses, drift_losses = fit_neural_sde_gan(
            drift_network=neural_ode.drift_net,  # Use drift from the NeuralODE object
            diffusion_network=diffusion_net,
            critic_network=critic_net,
            time_grid=time_grid_for_sde,  # Pass full time grid
            stochastic_trajectories=trajectories_for_sde,  # Pass full trajectories
            hyperparameters=hyperparameters,
            random_seed=42069,
        )

        # Store losses in the model
        neural_sde.generator_losses = generator_losses
        neural_sde.critic_losses = critic_losses
        neural_sde.drift_losses = drift_losses

        # Save the trained SDE model
        torch.save(neural_sde.state_dict(), sde_model_path)
        torch.save(generator_losses, generator_losses_path)
        torch.save(critic_losses, critic_losses_path)
        torch.save(drift_losses, drift_losses_path)
        print(f"Neural SDE saved to {sde_model_path}")
        print(f"Metrics saved to {generator_losses_path}, {critic_losses_path}, and {drift_losses_path}")

    print("Neural SDE training complete")
    print(f" Generator epochs: {len(generator_losses)}")
    print(f" Critic epochs: {len(critic_losses)}")
    print(f" Drift epochs: {len(drift_losses)}")
    return critic_losses, generator_losses, drift_losses, neural_sde


@app.cell
def _(DEVICE, config, neural_sde, processed_data, sample_sde_rollouts_with_inputs, torch):
    """Generate SDE rollouts for evaluation.

    Samples multiple stochastic trajectories from the trained SDE to assess
    whether the learnt dynamics match the true data distribution.
    
    Now uses sample_sde_rollouts_with_inputs to properly include exogenous signals
    (H field, sin/cos time features) that the nanorings system depends on.
    """
    # Prepare standardised trajectories for rollout generation
    train_amr_for_rollout = processed_data["train_amr_main_norm"].to(DEVICE)
    trajectories_for_rollout = train_amr_for_rollout.unsqueeze(-1)
    time_grid_sde = processed_data["time_grid"].to(DEVICE)
    
    # Prepare external inputs: [batch, input_dim, time]
    # input_dim = 3: H field + sin(wt) + sin(2wt)
    num_trajectories = train_amr_for_rollout.shape[0]
    
    # Get H field for all training trajectories [batch, time]
    train_h_main = processed_data["train_h_main"].to(DEVICE)
    
    # Compute time features for all timesteps
    t_min = processed_data["t_min"].item()
    t_max = processed_data["t_max"].item()
    t_norm = (time_grid_sde - t_min) / (t_max - t_min)
    sin_t_1 = torch.sin(2 * torch.pi * t_norm)
    sin_t_2 = torch.sin(4 * torch.pi * t_norm)
    
    # Broadcast time features to batch dimension [batch, time]
    sin_t_1_batch = sin_t_1.unsqueeze(0).expand(num_trajectories, -1)
    sin_t_2_batch = sin_t_2.unsqueeze(0).expand(num_trajectories, -1)
    
    # Stack external inputs: [batch, input_dim, time]
    external_inputs = torch.stack([train_h_main, sin_t_1_batch, sin_t_2_batch], dim=1)

    print("\n" + "="*60)
    print("Generating SDE Rollouts with Exogenous Inputs")
    print("="*60)
    print(f"External inputs shape: {external_inputs.shape} [batch, input_dim=3, time]")
    print("  - H field + sin(wt) + sin(2wt)")
    
    rollout = sample_sde_rollouts_with_inputs(
        model=neural_sde,
        trajectories=trajectories_for_rollout,
        external_inputs=external_inputs,
        time_grid=time_grid_sde,
        num_samples=config.NUMBER_OF_SDE_ROLLOUT_SAMPLES,
        device=DEVICE,
    )

    print(f"Generated {rollout.rollout_tensor.shape[0]} stochastic rollouts")
    print(f"  Shape: {rollout.rollout_tensor.shape} [samples, timesteps, state_dim]")
    return (rollout,)


@app.cell
def _(
    COLOURS,
    DEVICE,
    config,
    neural_sde,
    np,
    plt,
    predicted_ode,
    processed_data,
    setup_matplotlib_style,
    torch,
    traj_idx_ode_eval,
):
    """Evaluate Neural SDE on validation data (intermediate check)."""
    setup_matplotlib_style()
    
    print("\n" + "="*60)
    print("Neural SDE Evaluation (Standardised Space)")
    print("="*60)

    neural_sde.eval()
    
    with torch.no_grad():
        # Use same trajectory as ODE evaluation
        y0_sde = (
            processed_data["val_amr_main_norm"][traj_idx_ode_eval, 0]
            .unsqueeze(0)
            .unsqueeze(0)
            .to(DEVICE)
        )
        val_h_main_sde = processed_data["val_h_main"][traj_idx_ode_eval].to(DEVICE)
        time_grid_sde_eval = processed_data["time_grid"].to(DEVICE)

        # Define trajectory-specific functions for SDE
        def drift_func_sde(t, y):
            idx = min(
                int((t - processed_data["t_min"].item()) / processed_data["dt"]),
                val_h_main_sde.shape[0] - 1,
            )
            h_t = val_h_main_sde[idx].unsqueeze(0).unsqueeze(0)
            
            t_norm = (t - processed_data["t_min"].item()) / (
                processed_data["t_max"].item() - processed_data["t_min"].item()
            )
            t_norm_tensor = torch.full((1, 1), t_norm, device=y.device, dtype=y.dtype)
            sin_t_1 = torch.sin(2 * torch.pi * t_norm_tensor)
            sin_t_2 = torch.sin(4 * torch.pi * t_norm_tensor)
            
            net_input = torch.cat([y, h_t, sin_t_1, sin_t_2], dim=1)
            return neural_sde.drift_net(net_input)

        def diffusion_func_sde(t, y):
            idx = min(
                int((t - processed_data["t_min"].item()) / processed_data["dt"]),
                val_h_main_sde.shape[0] - 1,
            )
            h_t = val_h_main_sde[idx].unsqueeze(0).unsqueeze(0)
            
            t_norm = (t - processed_data["t_min"].item()) / (
                processed_data["t_max"].item() - processed_data["t_min"].item()
            )
            t_norm_tensor = torch.full((1, 1), t_norm, device=y.device, dtype=y.dtype)
            sin_t_1 = torch.sin(2 * torch.pi * t_norm_tensor)
            sin_t_2 = torch.sin(4 * torch.pi * t_norm_tensor)
            
            net_input = torch.cat([y, h_t, sin_t_1, sin_t_2], dim=1)
            return neural_sde.diffusion_net(net_input)

        # Sample multiple SDE trajectories
        num_sde_samples = 20
        sde_predictions = []
        for _ in range(num_sde_samples):
            pred_sde_sample = neural_sde.sample_trajectory(
                y0_sde,
                time_grid_sde_eval,
                drift_func_sde,
                diffusion_func_sde,
                device=DEVICE,
            )
            sde_predictions.append(pred_sde_sample.squeeze().cpu())
        
        sde_predictions_stacked = torch.stack(sde_predictions)
        sde_mean = sde_predictions_stacked.mean(dim=0).numpy()
        sde_std = sde_predictions_stacked.std(dim=0).numpy()

    # Plot SDE results compared with ODE
    time_np_sde_plot = processed_data["time_grid"].cpu().numpy()
    true_amr_np_sde_plot = processed_data["val_amr_main_norm"][traj_idx_ode_eval].cpu().numpy()
    pred_ode_np_sde_plot = predicted_ode.cpu().numpy()
    h_field_np_sde_plot = val_h_main_sde.cpu().numpy()
    
    # Handle potential length mismatch
    min_len_sde = min(len(time_np_sde_plot), len(pred_ode_np_sde_plot), len(true_amr_np_sde_plot), 
                      len(h_field_np_sde_plot), len(sde_mean))
    time_np_sde_plot = time_np_sde_plot[:min_len_sde]
    true_amr_np_sde_plot = true_amr_np_sde_plot[:min_len_sde]
    pred_ode_np_sde_plot = pred_ode_np_sde_plot[:min_len_sde]
    h_field_np_sde_plot = h_field_np_sde_plot[:min_len_sde]
    sde_mean = sde_mean[:min_len_sde]
    sde_std = sde_std[:min_len_sde]

    fig_sde, (ax1_sde, ax2_sde) = plt.subplots(2, 1, figsize=(8, 8))

    # Plot H-field
    ax1_sde.plot(time_np_sde_plot, h_field_np_sde_plot, color=COLOURS[1], linewidth=1.5, label=r'$H(t)$')
    ax1_sde.set_ylabel(r'$H$ (Standardised)')
    ax1_sde.legend(loc='upper right')
    ax1_sde.spines['top'].set_visible(False)
    ax1_sde.spines['right'].set_visible(False)

    # Plot AMR: ground truth, ODE, and SDE
    ax2_sde.plot(time_np_sde_plot, true_amr_np_sde_plot, color=COLOURS[3], linewidth=2, label='Ground Truth', alpha=0.8)
    ax2_sde.plot(time_np_sde_plot, pred_ode_np_sde_plot, color=COLOURS[0], linewidth=2, linestyle='--', label='Neural ODE')
    
    # Plot SDE samples with transparency
    for sde_sample_idx in range(num_sde_samples):
        sde_sample_trimmed = sde_predictions_stacked[sde_sample_idx].numpy()[:min_len_sde]
        ax2_sde.plot(time_np_sde_plot, sde_sample_trimmed, 
                color=COLOURS[2], alpha=0.1, linewidth=0.5, label='_nolegend_')
    
    # Plot SDE mean
    ax2_sde.plot(time_np_sde_plot, sde_mean, color=COLOURS[2], linewidth=2, label='Neural SDE (mean)')
    ax2_sde.fill_between(time_np_sde_plot, sde_mean - sde_std, sde_mean + sde_std, 
                     color=COLOURS[2], alpha=0.2, label=r'Neural SDE ($\pm 1\sigma$)')
    
    ax2_sde.set_xlabel(r'Time [s]')
    ax2_sde.set_ylabel(r'AMR (Standardised)')
    ax2_sde.legend(loc='upper right')
    ax2_sde.spines['top'].set_visible(False)
    ax2_sde.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    
    print("✓ Neural SDE prediction complete")
    print(f"  SDE mean MSE: {((torch.from_numpy(sde_mean) - processed_data['val_amr_main_norm'][traj_idx_ode_eval])**2).mean().item():.6f}")
    print(f"  ODE MSE: {((predicted_ode.cpu() - processed_data['val_amr_main_norm'][traj_idx_ode_eval])**2).mean().item():.6f}")
    
    return num_sde_samples, sde_mean, sde_predictions_stacked, sde_std


@app.cell
def _(compare_statistics, compute_statistics, rollout):
    """
    Compare statistical properties of real vs generated trajectories.
    """
    print("\n" + "="*60)
    print("Computing Statistics")
    print("="*60)
    training_stats = compute_statistics(rollout.training_subset)
    sde_stats = compute_statistics(rollout.rollout_tensor)

    print("\nComparing distributions (moment matching):")
    compare_statistics(training_stats, sde_stats)
    return sde_stats, training_stats


@app.cell
def _(
    COLOURS,
    finalise_plot,
    np,
    os,
    output_directory,
    plt,
    processed_data,
    sde_stats,
    setup_matplotlib_style,
    training_stats,
):
    """
    Plot mean(t) and variance(t) comparison between training data and SDE predictions.
    
    This visualization is critical for diagnosing whether:
    1. The SDE captures the correct mean trajectory over time
    2. The learned diffusion matches the data variance (especially at trajectory end)
    3. Variance "blows up" at later times (indicating distribution mismatch)
    """
    if sde_stats.mean.numel() > 0:
        setup_matplotlib_style()
        fig_stats, ax_stats = plt.subplots(1, 1, figsize=(9, 4))
        
        stats_time_axis = processed_data["time_grid"].detach().cpu().numpy().ravel()
        training_mean = training_stats.mean.detach().cpu().numpy()
        training_var = training_stats.variance.detach().cpu().numpy()
        sde_mean = sde_stats.mean.detach().cpu().numpy()
        sde_var = sde_stats.variance.detach().cpu().numpy()

        # Plot mean ± sqrt(variance) envelopes
        train_envelope = np.sqrt(np.maximum(training_var[:, 0], 0.0))
        sde_envelope = np.sqrt(np.maximum(sde_var[:, 0], 0.0))

        ax_stats.fill_between(
            stats_time_axis,
            training_mean[:, 0] - train_envelope,
            training_mean[:, 0] + train_envelope,
            alpha=0.25,
            color="black",
            label="Training mean ± sqrt(var)",
        )
        ax_stats.plot(
            stats_time_axis,
            training_mean[:, 0],
            color="black",
            linewidth=1.8,
            label="Training mean",
        )
        
        ax_stats.fill_between(
            stats_time_axis,
            sde_mean[:, 0] - sde_envelope,
            sde_mean[:, 0] + sde_envelope,
            alpha=0.25,
            color=COLOURS[3],
            label="Neural SDE mean ± sqrt(var)",
        )
        ax_stats.plot(
            stats_time_axis,
            sde_mean[:, 0],
            color=COLOURS[3],
            linewidth=1.8,
            label="Neural SDE mean",
        )

        ax_stats.set_xlabel("Time [s]")
        ax_stats.set_ylabel("AMR (standardized)")
        ax_stats.set_title("Statistical Comparison: Training vs Neural SDE")
        ax_stats.grid(False)
        ax_stats.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left")

        for spine in ax_stats.spines.values():
            spine.set_linewidth(1.0)

        fig_stats.tight_layout()
        finalise_plot(fig_stats, "nanorings_statistics.pdf", str(output_directory), os)
        
        print("\n" + "="*60)
        print("Statistics Plot Saved")
        print("="*60)
        print(f"✓ Saved to: {output_directory}/nanorings_statistics.pdf")
        print("\nCheck this plot to verify:")
        print("  - Mean trajectories align between training and SDE")
        print("  - Variance envelopes match (no blowup at trajectory end)")
        print("  - Diffusion network learned appropriate stochasticity")
    else:
        print("\nSkipping statistics plot - no SDE rollouts generated")
    return


@app.cell
def _(
    DEVICE,
    np,
    os,
    output_directory,
    plot_nanoring_results,
    processed_data,
    rollout,
    rollout_trajectory,
    torch,
    trained_drift_net,
):
    """Evaluate Neural ODE and SDE predictions on validation data.

    All evaluation is done in STANDARDIZED SPACE (μ=0, σ=1) - no denormalization applied.
    """
    print("\n" + "="*60)
    print("Evaluating on Validation Data (Standardized Space)")
    print("="*60)

    trained_drift_net.eval()
    traj_idx_to_plot = 0

    with torch.no_grad():
        # Extract initial state (standardized space)
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

        # Rollout ODE trajectory
        eval_num_steps = processed_data["val_amr_main_norm"].shape[1] - 1
        eval_final_time = processed_data["t_min"].item() + eval_num_steps * processed_data["dt"]

        _, pred_ode_norm = rollout_trajectory(
            drift_function=drift_func_eval,
            initial_state=y0_eval,
            initial_time=processed_data["t_min"].item(),
            final_time=eval_final_time,
            timestep=processed_data["dt"],
        )

        # Keep predictions in standardized space - NO denormalization
        pred_ode = pred_ode_norm.squeeze().cpu()
        true_val = processed_data["val_amr_main_norm"][traj_idx_to_plot].cpu()
        h_field_norm = val_h_main_traj.cpu()

        # Get all training sequences in standardized space
        all_train_seqs = processed_data["train_amr_main_norm"].cpu().numpy()

        # Get SDE prediction in standardized space
        if rollout.rollout_tensor.numel() > 0:
            pred_sde = rollout.rollout_tensor[0, :, 0].cpu()
        else:
            pred_sde = pred_ode

        # Save standardized numerical results
        np.savetxt(
            os.path.join(output_directory, "predicted_trajectory_ode_standardized.txt"),
            pred_ode.numpy(),
        )
        np.savetxt(
            os.path.join(output_directory, "predicted_trajectory_sde_standardized.txt"),
            pred_sde.numpy(),
        )
        np.savetxt(
            os.path.join(output_directory, "true_trajectory_standardized.txt"),
            true_val.numpy(),
        )

    # Plot results in standardized space
    time_axis = processed_data["time_grid"].cpu().numpy()
    plot_nanoring_results(
        time_axis=time_axis,
        true_sequence=true_val,
        predicted_sequence_ode=pred_ode,
        predicted_sequence_sde=pred_sde,
        h_field=h_field_norm,
        all_train_sequences=all_train_seqs,
        output_dir=output_directory,
    )

    print(f"\nResults plotted and saved to {output_directory}")
    print("\nValidation Performance (Standardized Space):")
    print(f"  ODE MSE: {torch.mean((pred_ode - true_val)**2).item():.6f}")
    print(f"  SDE MSE: {torch.mean((pred_sde - true_val)**2).item():.6f}")
    return


@app.cell
def _(
    COLOURS,
    critic_losses,
    drift_losses,
    generator_losses,
    np,
    os,
    output_directory,
    plt,
    training_losses,
    validation_losses,
):
    """Visualise training convergence.

    Shows ODE supervised learning curves and SDE adversarial training dynamics.
    """
    print("\n" + "="*60)
    print("Training Loss Analysis")
    print("="*60)

    # Calculate total epochs and create shared x-axis
    ode_epochs = len(training_losses)
    sde_epochs = len(generator_losses)
    total_epochs = ode_epochs + sde_epochs

    # Create 4-panel plot with shared x-axis
    fig_losses, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(8, 10), sharex=True)

    # Create epoch arrays
    ode_epoch_range = np.arange(ode_epochs)
    sde_epoch_range = np.arange(ode_epochs, total_epochs)

    # Panel 1: Neural ODE supervised learning (finite differences)
    ax1.plot(
        ode_epoch_range,
        training_losses,
        color=COLOURS[0],
        linewidth=1.5,
        label='Training'
    )
    ax1.plot(
        ode_epoch_range,
        validation_losses,
        color=COLOURS[1],
        linewidth=1.5,
        label='Validation'
    )

    # Mark transition to SDE training phase with annotation
    if ode_epochs > 0:
        ax1.axvline(
            x=ode_epochs - 1,
            color="black",
            linestyle="--",
            alpha=0.7,
            linewidth=1.5,
        )
        # Add text annotation
        ax1.text(
            ode_epochs * 1.05,
            ax1.get_ylim()[1] * 0.45,
            "SDE training starts",
            ha="center",
            va="bottom",
            fontsize=14,
            rotation=90,
        )

    ax1.legend(loc="best", framealpha=0.9)
    ax1.set_ylabel("Smooth L1 Loss")

    # Panel 2: SDE Generator Loss (adversarial + L1 + moment matching)
    if len(generator_losses) > 0:
        ax2.plot(sde_epoch_range, generator_losses, color='k', linewidth=1.5)
    ax2.set_ylabel("Generator Loss")

    # Panel 3: Critic Loss (Wasserstein distance + gradient penalty)
    if len(critic_losses) > 0:
        ax3.plot(sde_epoch_range, critic_losses, color='k', linewidth=1.5)
    ax3.set_ylabel("Critic Loss")

    # Panel 4: Drift Loss (drift network fine-tuning during SDE training)
    if len(drift_losses) > 0:
        ax4.plot(sde_epoch_range, drift_losses, color='red', linewidth=1.5)
    ax4.set_ylabel("Drift Loss")
    ax4.set_xlabel("Epoch")

    # Set x-axis limits to show all epochs
    ax3.set_xlim(0, total_epochs - 1)

    plt.tight_layout()
    loss_plot_path = os.path.join(output_directory, "training_losses.pdf")
    fig_losses.savefig(loss_plot_path, dpi=300, bbox_inches="tight")
    print(f"\nLoss curves saved to {loss_plot_path}")
    print("\nTraining Summary:")
    print(f"  ODE epochs: {ode_epochs}")
    print(f"  SDE epochs: {sde_epochs}")
    print(f"  Final ODE train loss: {training_losses[-1]:.6f}")
    print(f"  Final ODE val loss: {validation_losses[-1]:.6f}")
    if len(generator_losses) > 0:
        print(f"  Final generator loss: {generator_losses[-1]:.6f}")
        print(f"  Final critic loss: {critic_losses[-1]:.6f}")
        print(f"  Final drift loss: {drift_losses[-1]:.6f}")
    plt.show()
    plt.close(fig_losses)
    return


if __name__ == "__main__":
    app.run()
