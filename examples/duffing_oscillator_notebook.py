import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    # --- 1. Setup: Imports and Configuration ---
    import os
    import torch
    import matplotlib.pyplot as plt
    import numpy as np
    from torch.utils.data import DataLoader, TensorDataset

    from config import DEVICE
    from neural_dynamics.core.hyperparameters import NetworkArchitecture
    from neural_dynamics.models.base import DriftNet
    from neural_dynamics.training.base import train_with_validation
    from neural_dynamics.training.evaluation import rollout_trajectory
    from neural_dynamics.core.utils import COLOURS
    from neural_dynamics.core.utils import set_symmetric_three_ticks

    from examples.systems.duffing_oscillator import DuffingOscillator
    from examples.parameters.duffing_oscillator import PhysicalDuffingParameters

    class DuffingHyperparameters:
        """Hyperparameters for the Duffing oscillator experiment."""

        # Data generation
        NUM_TRAJECTORIES: int = 512
        NO_NOISE: bool = True

        # Training
        LEARNING_RATE: float = 3e-4
        NUM_EPOCHS: int = 512
        BATCH_SIZE: int = 16
        VALIDATION_SPLIT: float = 0.2
        EARLY_STOPPING_PATIENCE: int = NUM_EPOCHS

        # Model
        STATE_DIM: int = 3  # [q, v, xi]
        DRIFT_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
            input_size=STATE_DIM + 1,  # state + time
            hidden_sizes=[128, 128, 128],
            output_size=STATE_DIM,
        )

    config = DuffingHyperparameters()

    output_dir = os.path.join(
        "examples",
        "output",
        f"duffing_epochs-{config.NUM_EPOCHS}_batch_size-{config.BATCH_SIZE}_lr-{config.LEARNING_RATE}",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- 2. Data Generation (batched and vectorised) ---
    if config.NO_NOISE:
        physical_params = PhysicalDuffingParameters(
            temperature=0.0, coloured_noise_intensity=0.0, timestep=0.01
        )
    else:
        physical_params = PhysicalDuffingParameters(timestep=25.0 / 255)

    dimless_params = physical_params.to_dimensionless()
    system = DuffingOscillator(dimless_params)
    time_grid, trajectories = system.integrate_sde(batch_size=config.NUM_TRAJECTORIES)

    # Move data to device once
    trajectories = trajectories.to(DEVICE)
    time_grid = time_grid.to(DEVICE)
    dt = float((time_grid[1] - time_grid[0]).item())

    # --- 3. Build Model and Dataloaders ---
    drift_net = DriftNet(config.DRIFT_NET_ARCH).to(DEVICE)

    # Split into training/validation sets at the trajectory level
    num_train = int((1.0 - config.VALIDATION_SPLIT) * trajectories.shape[0])
    train_trajs = trajectories[:num_train]
    val_trajs = trajectories[num_train:]

    train_ds = TensorDataset(train_trajs)
    val_ds = TensorDataset(val_trajs)
    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0)

    def batch_preparation_fn(raw_batch, device):
        """Prepare per-step supervised pairs (inputs, dstate/dt) for training.

        Net input is [q, v, xi, t]. Supervision uses finite-difference derivatives.
        Shapes: raw_batch[0] -> [B, T, 3] -> flatten to [B*(T-1), ...].
        """
        (batch_trajs,) = raw_batch  # [B, T, 3]
        current = batch_trajs[:, :-1, :]  # [B, T-1, 3]
        target = batch_trajs[:, 1:, :]    # [B, T-1, 3]
        B, Tm1, D = current.shape

        # Broadcast times and flatten
        times = time_grid[:-1].view(1, Tm1, 1).expand(B, Tm1, 1)
        net_input = torch.cat([current, times], dim=-1).reshape(B * Tm1, D + 1).contiguous()

        true_derivatives = ((target - current) / dt).reshape(B * Tm1, D).contiguous()
        return net_input.to(device, non_blocking=True), true_derivatives.to(device, non_blocking=True)
    return (
        COLOURS,
        DEVICE,
        batch_preparation_fn,
        config,
        drift_net,
        np,
        os,
        output_dir,
        plt,
        rollout_trajectory,
        set_symmetric_three_ticks,
        time_grid,
        torch,
        train_loader,
        train_with_validation,
        trajectories,
        val_loader,
    )


@app.cell
def _(
    DEVICE,
    batch_preparation_fn,
    config,
    drift_net,
    np,
    os,
    output_dir,
    torch,
    train_loader,
    train_with_validation,
    val_loader,
):
    # --- 4. Train the Drift Network (vectorised, with validation) ---
    trained_drift_net, train_losses, val_losses = train_with_validation(
        model=drift_net,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config.NUM_EPOCHS,
        learning_rate=config.LEARNING_RATE,
        early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
        device=DEVICE,
        batch_preparation_fn=batch_preparation_fn,
    )

    # Persist losses and learnt model for downstream SDE/GAN baselines
    np.savetxt(
        os.path.join(output_dir, "losses.txt"),
        np.column_stack([train_losses, val_losses]),
        header="Training_Loss,Validation_Loss",
    )
    torch.save(trained_drift_net.state_dict(), os.path.join(output_dir, "drift_net_state_dict.pt"))
    return train_losses, trained_drift_net, val_losses


@app.cell
def _(
    DEVICE,
    config,
    np,
    os,
    output_dir,
    rollout_trajectory,
    time_grid,
    torch,
    trained_drift_net,
    trajectories,
):
    # --- 5. Simulate a trajectory with the trained drift (Neural ODE) ---
    trained_drift_net.eval()
    with torch.no_grad():
        # Use the first validation (or training if val empty) trajectory as reference
        ref_traj = trajectories[0].to(DEVICE)
        y0 = ref_traj[0, : config.STATE_DIM].unsqueeze(0)

        def drift_func_eval(t, y):
            t_tensor = torch.as_tensor([[t]], device=y.device, dtype=y.dtype)
            net_in = torch.cat([y, t_tensor], dim=1)
            return trained_drift_net(net_in)

        t0 = float(time_grid[0].item())
        dt_step = float((time_grid[1] - time_grid[0]).item())
        # Ensure the rollout uses the exact same number of steps as the reference
        num_steps_required = int(time_grid.shape[0] - 1)
        tN = t0 + num_steps_required * dt_step
        pred_time, pred_traj_batched = rollout_trajectory(
            drift_function=drift_func_eval,
            initial_state=y0,
            initial_time=t0,
            final_time=tN,
            timestep=dt_step,
        )

        # Save prediction artefacts for later analysis
        np.savetxt(os.path.join(output_dir, "time_grid.txt"), pred_time.cpu().numpy())
        pred_traj = pred_traj_batched.squeeze(1)  # remove batch dim -> [T, D]
        np.savetxt(os.path.join(output_dir, "predicted_trajectory.txt"), pred_traj.cpu().numpy())
        np.savetxt(os.path.join(output_dir, "true_trajectory.txt"), ref_traj.cpu().numpy())
    return pred_time, pred_traj, ref_traj


@app.cell
def _(
    COLOURS,
    np,
    os,
    output_dir,
    plt,
    pred_time,
    pred_traj,
    ref_traj,
    set_symmetric_three_ticks,
    train_losses,
    trajectories,
    val_losses,
):
    # --- 6. Three figures: losses, phase space, and pos/vel with residuals ---
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "Times",
        "font.size": 14,
        "axes.linewidth": 1.0,
    })

    os.makedirs(output_dir, exist_ok=True)

    # 1) Losses
    fig1, ax = plt.subplots(figsize=(4.0, 4.0))
    ax.semilogy(np.arange(len(train_losses)), train_losses, color="black", linewidth=0.1)
    ax.semilogy(np.arange(len(val_losses)),   val_losses,   color="black", linewidth=1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.grid(False)
    try:
        ax.set_box_aspect(1)
    except Exception:
        pass
    for s in ax.spines.values():
        s.set_linewidth(1.0)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "losses.pdf"), bbox_inches="tight", dpi=300)
    plt.show()

    # 2) Phase space
    q_true = ref_traj[:, 0].cpu().numpy()
    v_true = ref_traj[:, 1].cpu().numpy()
    q_pred = pred_traj[:, 0].cpu().numpy()
    v_pred = pred_traj[:, 1].cpu().numpy()

    fig2, axp = plt.subplots(figsize=(4.0, 4.0))
    axp.plot(q_true, v_true, color="black", linewidth=0.1)
    axp.plot(q_pred, v_pred, color="black", linewidth=1.0)
    axp.set_ylabel("$v(q(t))$ [ms$^{-1}$]")
    axp.set_xlabel("$q(t)$ [m]")
    axp.grid(False)
    set_symmetric_three_ticks(axp, np.concatenate([q_true, q_pred, v_true, v_pred]), axis="both")

    try:
        axp.set_box_aspect(1)
    except Exception:
        pass
    for s in axp.spines.values():
        s.set_linewidth(1.0)
    fig2.savefig(os.path.join(output_dir, "phase_space.pdf"), bbox_inches="tight", dpi=300)
    plt.show()

    # 3) Position and velocity time series (side-by-side), with residuals beneath
    from matplotlib.gridspec import GridSpec as _GS

    t = pred_time.cpu().numpy().ravel()
    traj = trajectories.detach().cpu().numpy()
    pred = pred_traj.detach().cpu().numpy()
    ref = ref_traj.detach().cpu().numpy()

    fig3 = plt.figure(figsize=(8.5, 5.3))
    gs = _GS(nrows=2, ncols=2, figure=fig3, height_ratios=[1.0, 0.22], hspace=0.0, wspace=0.3)

    axQ = fig3.add_subplot(gs[0, 0])
    axV = fig3.add_subplot(gs[0, 1])
    axQres = fig3.add_subplot(gs[1, 0], sharex=axQ)
    axVres = fig3.add_subplot(gs[1, 1], sharex=axV)

    # Background training trajectories
    n_show = min(traj.shape[0], 200)
    for i in range(n_show):
        axQ.plot(t, traj[i, :, 0], color="black", linewidth=1.0, alpha=0.10)
        axV.plot(t, traj[i, :, 1], color="black", linewidth=1.0, alpha=0.10)

    # Reference (solid) and Predicted (dashed)
    axQ.plot(t, ref[:, 0], color="black", linewidth=0.1)
    axQ.plot(t, pred[:, 0], color=COLOURS[2], linewidth=1.0)
    axV.plot(t, ref[:, 1], color="black", linewidth=0.1)
    axV.plot(t, pred[:, 1], color=COLOURS[2], linewidth=1.0)

    # Residuals
    res_q = pred[:, 0] - ref[:, 0]
    res_v = pred[:, 1] - ref[:, 1]
    axQres.plot(t, res_q, color="black", linewidth=1.0)
    axVres.plot(t, res_v, color="black", linewidth=1.0)
    axQres.axhline(0.0, color="black", linewidth=0.8)
    axVres.axhline(0.0, color="black", linewidth=0.8)
    set_symmetric_three_ticks(axQres, res_q)
    set_symmetric_three_ticks(axVres, res_v)

    # Labels
    axQ.set_ylabel("$q(t)$")
    axV.set_ylabel("$v(t)$")
    axQres.set_ylabel(r"$e_q$")
    axVres.set_ylabel(r"$e_v$")
    axQres.set_xlabel(r"$t$ [s]")
    axVres.set_xlabel(r"$t$ [s]")

    # Hide x tick labels on main axes so residuals carry the x-axis
    plt.setp(axQ.get_xticklabels(), visible=False)
    plt.setp(axV.get_xticklabels(), visible=False)

    for axm in (axQ, axV):
        try:
            axm.set_box_aspect(1)
        except Exception:
            pass

    for axm in (axQ, axV, axQres, axVres):
        axm.grid(False)
        for s in axm.spines.values():
            s.set_linewidth(1.0)

    fig3.savefig(os.path.join(output_dir, "timeseries.pdf"), bbox_inches="tight", dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
