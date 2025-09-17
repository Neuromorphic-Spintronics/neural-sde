import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    # --- 1. Setup: Imports and Configuration ---
    import os
    import sys
    from pathlib import Path

    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    # Ensure the project root is importable when running the notebook standalone.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from neural_dynamics.config import DEVICE
    from neural_dynamics.core.hyperparameters import Hyperparameters
    from neural_dynamics.models.ode import NeuralODE
    from neural_dynamics.models.sde import NeuralSDE
    from neural_dynamics.training.evaluation import rollout_trajectory
    from neural_dynamics.core.utils import COLOURS
    from neural_dynamics.core.utils import set_symmetric_three_ticks

    from examples.systems.duffing_oscillator import DuffingOscillator
    from examples.parameters.duffing_oscillator import PhysicalDuffingParameters

    class NotebookConfig:
        """Top-level configuration for the Duffing notebook."""

        NUM_TRAJECTORIES: int = 512
        NO_NOISE: bool = False
        VALIDATION_SPLIT: float = 0.2
        RUN_GAN: bool = True
        NUM_SDE_SAMPLES: int = 20

    defaults = Hyperparameters.defaults()

    output_dir = os.path.join(
        "examples",
        "output",
        f"duffing_epochs-{defaults.number_of_epochs}_batch_size-{defaults.batch_size}_lr-{defaults.learning_rates.drift}",
    )
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # --- 2. Data Generation (batched and vectorised) ---
    if NotebookConfig.NO_NOISE:
        physical_params = PhysicalDuffingParameters(
            temperature=0.0, coloured_noise_intensity=0.0, timestep=0.01
        )
    else:
        physical_params = PhysicalDuffingParameters(timestep=25.0 / 255)

    dimless_params = physical_params.to_dimensionless()
    system = DuffingOscillator(dimless_params)
    time_grid, trajectories = system.integrate_sde(batch_size=NotebookConfig.NUM_TRAJECTORIES)

    trajectories = trajectories.to(torch.float32)
    time_grid = time_grid.to(torch.float32)
    return (
        COLOURS,
        DEVICE,
        NeuralODE,
        NeuralSDE,
        NotebookConfig,
        defaults,
        np,
        os,
        output_dir,
        plt,
        rollout_trajectory,
        set_symmetric_three_ticks,
        time_grid,
        torch,
        trajectories,
    )


@app.cell
def _(
    DEVICE,
    NeuralODE,
    NotebookConfig,
    defaults,
    np,
    os,
    output_dir,
    time_grid,
    torch,
    trajectories,
):
    # --- 4. Train the drift network via the NeuralODE helper ---
    neural_ode = NeuralODE.train(
        hyperparameters=defaults,
        trajectories=trajectories,
        time_grid=time_grid,
        device=DEVICE,
        validation_split=NotebookConfig.VALIDATION_SPLIT,
        early_stopping_patience=defaults.number_of_epochs,
    )

    train_losses = neural_ode.training_losses
    val_losses = neural_ode.validation_losses

    # Persist losses and learnt model for downstream SDE/GAN baselines
    np.savetxt(
        os.path.join(output_dir, "losses.txt"),
        np.column_stack([train_losses, val_losses]),
        header="Training_Loss,Validation_Loss",
    )
    torch.save(neural_ode.drift_net.state_dict(), os.path.join(output_dir, "drift_net_state_dict.pt"))
    return neural_ode, train_losses, val_losses


@app.cell
def _(
    DEVICE,
    defaults,
    neural_ode,
    np,
    os,
    output_dir,
    rollout_trajectory,
    time_grid,
    torch,
    trajectories,
):
    # --- 5. Simulate a trajectory with the trained drift (Neural ODE) ---
    drift_net = neural_ode.drift_net
    drift_net.eval()
    with torch.no_grad():
        # Use the first validation (or training if val empty) trajectory as reference
        ref_traj = trajectories[0].to(DEVICE)
        y0 = ref_traj[0, : defaults.state_dimension].unsqueeze(0)

        def drift_func_eval(t, y):
            t_tensor = torch.as_tensor([[t]], device=y.device, dtype=y.dtype)
            net_in = torch.cat([y, t_tensor], dim=1)
            return drift_net(net_in)

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
    DEVICE,
    NeuralSDE,
    NotebookConfig,
    defaults,
    neural_ode,
    time_grid,
    trajectories,
):
    neural_sde = NeuralSDE.train(
        hyperparameters=defaults,
        neural_ode=neural_ode,
        trajectories=trajectories,
        time_grid=time_grid,
        device=DEVICE,
        enable_adversarial=NotebookConfig.RUN_GAN,
    )
    return (neural_sde,)


@app.cell
def _(
    DEVICE,
    NotebookConfig,
    defaults,
    neural_sde,
    time_grid,
    torch,
    trajectories,
):
    neural_sde_rollouts = []
    if NotebookConfig.RUN_GAN and NotebookConfig.NUM_SDE_SAMPLES > 0:
        sde_initial_state = (
            trajectories[0, 0, : defaults.state_dimension]
            .unsqueeze(0)
            .to(DEVICE)
        )
        time_grid_device = time_grid.to(DEVICE)

        with torch.no_grad():
            for sample_idx in range(NotebookConfig.NUM_SDE_SAMPLES):
                sample_rollout = neural_sde(sde_initial_state, time_grid_device)
                neural_sde_rollouts.append(
                    sample_rollout[:, 0, : defaults.state_dimension].cpu()
                )
    return (neural_sde_rollouts,)


@app.cell
def _(
    COLOURS,
    neural_sde_rollouts,
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
    traj_np = trajectories.detach().cpu().numpy()
    n_show_phase = min(traj_np.shape[0], 200)

    fig2, axp = plt.subplots(figsize=(4.0, 4.0))
    for i in range(n_show_phase):
        axp.plot(
            traj_np[i, :, 0],
            traj_np[i, :, 1],
            color="black",
            linewidth=0.6,
            alpha=0.05,
        )
    axp.plot(q_true, v_true, color="black", linewidth=0.1)
    axp.plot(q_pred, v_pred, color="black", linewidth=1.0)

    for sde_rollout in neural_sde_rollouts:
        sde_q = sde_rollout[:, 0].numpy()
        sde_v = sde_rollout[:, 1].numpy()
        axp.plot(sde_q, sde_v, color=COLOURS[3], linewidth=1.0, alpha=0.1)
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
        axQ.plot(t, traj[i, :, 0], color="black", linewidth=1.0, alpha=0.05)
        axV.plot(t, traj[i, :, 1], color="black", linewidth=1.0, alpha=0.05)

    for sde_rollout in neural_sde_rollouts:
        rollout_np = sde_rollout.numpy()
        axQ.plot(t, rollout_np[:, 0], color=COLOURS[3], linewidth=1.0, alpha=0.3)
        axV.plot(t, rollout_np[:, 1], color=COLOURS[3], linewidth=1.0, alpha=0.3)

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
