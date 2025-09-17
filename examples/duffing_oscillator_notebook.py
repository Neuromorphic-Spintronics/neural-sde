import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    # --- 1. Setup: Imports and Configuration ---
    import os
    import sys
    from dataclasses import replace
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

    # Any changes to the default hyperparameters can be made here
    defaults = replace(Hyperparameters.defaults(), number_of_epochs=512)

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
    NotebookConfig,
    neural_sde,
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

    # 1) Loss panels (drift + optional GAN losses)
    generator_losses = getattr(neural_sde, "generator_losses", []) or []
    critic_losses = getattr(neural_sde, "critic_losses", []) or []

    loss_panels = []
    loss_panels.append(
        (
            "Drift",
            np.arange(len(train_losses)),
            [
                ("Training", train_losses, "black", 1.2),
                ("Validation", val_losses, "black", 2.0),
            ],
        )
    )
    if generator_losses:
        loss_panels.append(
            (
                "Generator",
                np.arange(len(generator_losses)),
                [("Generator", generator_losses, COLOURS[3], 1.5)],
            )
        )
    if critic_losses:
        loss_panels.append(
            (
                "Critic",
                np.arange(len(critic_losses)),
                [("Critic", critic_losses, COLOURS[2], 1.5)],
            )
        )

    fig1, axes = plt.subplots(1, len(loss_panels), figsize=(4 * len(loss_panels),4))
    if len(loss_panels) == 1:
        axes = [axes]

    for ax, (title, epochs, series) in zip(axes, loss_panels):
        for label, values, color, width in series:
            ax.plot(epochs, values, label=label, color=color, linewidth=width)
            if label == "Validation":
                ax.set_ylabel("Overall Loss")
            else:
                ax.set_ylabel(f"{label} Loss")
        ax.set_xlabel("Epoch")
        ax.grid(False)
        if len(series) > 1:
            ax.legend()
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

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

    phase_fig, axp = plt.subplots(figsize=(4.0, 4.0))
    training_alpha_phase = 1.0 if NotebookConfig.NO_NOISE else 0.1
    for i in range(n_show_phase):
        axp.plot(
            traj_np[i, :, 0],
            traj_np[i, :, 1],
            color="black",
            linewidth=0.6,
            alpha=training_alpha_phase,
        )
    axp.plot(q_true, v_true, color="black", linewidth=0.6, alpha=training_alpha_phase)
    axp.plot(q_pred, v_pred, color=COLOURS[2], linewidth=1.0)

    for sde_rollout in neural_sde_rollouts:
        rollout_np = sde_rollout.numpy()
        axp.plot(rollout_np[:, 0], rollout_np[:, 1], color=COLOURS[3], linewidth=1.0, alpha=0.1)

    axp.set_ylabel("$v(q(t))$ [ms$^{-1}$]")
    axp.set_xlabel("$q(t)$ [m]")
    axp.grid(False)

    # Determine limits from training, deterministic, and SDE trajectories; add padding
    phase_components = [
        q_true,
        v_true,
        q_pred,
        v_pred,
        traj_np[:n_show_phase, :, 0].ravel(),
        traj_np[:n_show_phase, :, 1].ravel(),
    ]
    for sde_rollout in neural_sde_rollouts:
        rollout_np = sde_rollout.numpy()
        phase_components.append(rollout_np[:, 0])
        phase_components.append(rollout_np[:, 1])

    phase_values = np.concatenate(phase_components)
    set_symmetric_three_ticks(axp, phase_values, axis="both")

    x0, x1 = axp.get_xlim()
    ymin_lim, ymax_lim = axp.get_ylim()
    padding = 0.05
    axp.set_xlim(x0 - padding * (x1 - x0), x1 + padding * (x1 - x0))
    axp.set_ylim(ymin_lim - padding * (ymax_lim - ymin_lim), ymax_lim + padding * (ymax_lim - ymin_lim))

    phase_fig.savefig(os.path.join(output_dir, "phase_space.pdf"), bbox_inches="tight", dpi=300)
    plt.show()

    # 3) Position/velocity time series; residuals only when no SDE rollouts
    from matplotlib.gridspec import GridSpec as _GS

    t = pred_time.cpu().numpy().ravel()
    traj = trajectories.detach().cpu().numpy()
    pred = pred_traj.detach().cpu().numpy()
    ref = ref_traj.detach().cpu().numpy()

    show_residuals = len(neural_sde_rollouts) == 0

    fig3 = plt.figure(figsize=(8.5, 5.3))
    if show_residuals:
        gs = _GS(nrows=2, ncols=2, figure=fig3, height_ratios=[1.0, 0.22], hspace=0.0, wspace=0.3)
        axQ = fig3.add_subplot(gs[0, 0])
        axV = fig3.add_subplot(gs[0, 1])
        axQres = fig3.add_subplot(gs[1, 0], sharex=axQ)
        axVres = fig3.add_subplot(gs[1, 1], sharex=axV)
    else:
        gs = _GS(nrows=1, ncols=2, figure=fig3, wspace=0.3)
        axQ = fig3.add_subplot(gs[0, 0])
        axV = fig3.add_subplot(gs[0, 1])

    training_alpha = 1.0 if NotebookConfig.NO_NOISE else 0.1

    # Background training trajectories
    n_show = min(traj.shape[0], 200)
    for i in range(n_show):
        axQ.plot(t, traj[i, :, 0], color="black", linewidth=1.0, alpha=training_alpha)
        axV.plot(t, traj[i, :, 1], color="black", linewidth=1.0, alpha=training_alpha)

    # Neural SDE rollouts
    for sde_rollout in neural_sde_rollouts:
        rollout_np = sde_rollout.numpy()
        axQ.plot(t, rollout_np[:, 0], color=COLOURS[3], linewidth=1.0, alpha=0.1)
        axV.plot(t, rollout_np[:, 1], color=COLOURS[3], linewidth=1.0, alpha=0.1)

    # Reference and prediction
    axQ.plot(t, ref[:, 0], color="black", linewidth=0.6, alpha=training_alpha)
    axQ.plot(t, pred[:, 0], color=COLOURS[2], linewidth=1.0)
    axV.plot(t, ref[:, 1], color="black", linewidth=0.6, alpha=training_alpha)
    axV.plot(t, pred[:, 1], color=COLOURS[2], linewidth=1.0)

    # Residuals (Neural ODE only)
    if show_residuals:
        res_q = pred[:, 0] - ref[:, 0]
        res_v = pred[:, 1] - ref[:, 1]
        axQres.plot(t, res_q, color="black", linewidth=1.0)
        axVres.plot(t, res_v, color="black", linewidth=1.0)
        axQres.axhline(0.0, color="black", linewidth=0.8)
        axVres.axhline(0.0, color="black", linewidth=0.8)
        set_symmetric_three_ticks(axQres, res_q)
        set_symmetric_three_ticks(axVres, res_v)

    # Axis labels
    axQ.set_ylabel("$q(t)$")
    axV.set_ylabel("$v(t)$")
    if show_residuals:
        axQres.set_ylabel(r"$e_q$")
        axVres.set_ylabel(r"$e_v$")
        axQres.set_xlabel(r"$t$ [s]")
        axVres.set_xlabel(r"$t$ [s]")
        plt.setp(axQ.get_xticklabels(), visible=False)
        plt.setp(axV.get_xticklabels(), visible=False)
    else:
        axQ.set_xlabel(r"$t$ [s]")
        axV.set_xlabel(r"$t$ [s]")

    # Styling
    for main_ax in (axQ, axV):
        try:
            main_ax.set_box_aspect(1)
        except Exception:
            pass

    axes_to_style = (axQ, axV)
    if show_residuals:
        axes_to_style = (*axes_to_style, axQres, axVres)
    for axis in axes_to_style:
        axis.grid(False)
        for spine in axis.spines.values():
            spine.set_linewidth(1.0)

    fig3.savefig(os.path.join(output_dir, "timeseries.pdf"), bbox_inches="tight", dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
