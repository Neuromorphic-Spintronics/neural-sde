import marimo

__generated_with = "0.15.2"
app = marimo.App(width="full")


@app.cell
def _():
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import numpy as np
    import os
    import torch
    import matplotlib.pyplot as plt

    from neural_dynamics.config import DEVICE
    from neural_dynamics.core.utils import (
        COLOURS,
        compare_statistics,
        compute_statistics,
        sample_sde_rollouts,
    )
    from examples.systems.parameters.duffing_oscillator import PhysicalDuffingParameters
    from examples.systems.duffing_oscillator import (
        DuffingDataConfig,
        prepare_duffing_training_data,
    )
    from neural_dynamics.models.ode import NeuralODE
    from neural_dynamics.models.sde import NeuralSDE
    from examples.utils.plotting import setup_matplotlib_style, finalise_plot
    from neural_dynamics.utils.training_helpers import (
        DEFAULT_METRIC_FILENAMES,
        DEFAULT_MODEL_FILENAMES,
        DEFAULT_TRAINING_OUTPUT_DIR,
        ModellingConfig,
    )
    return (
        COLOURS,
        DEFAULT_METRIC_FILENAMES,
        DEFAULT_MODEL_FILENAMES,
        DEFAULT_TRAINING_OUTPUT_DIR,
        DEVICE,
        DuffingDataConfig,
        ModellingConfig,
        NeuralODE,
        NeuralSDE,
        PhysicalDuffingParameters,
        compare_statistics,
        compute_statistics,
        finalise_plot,
        np,
        os,
        plt,
        prepare_duffing_training_data,
        sample_sde_rollouts,
        setup_matplotlib_style,
        torch,
    )


@app.cell
def _(
    DEFAULT_TRAINING_OUTPUT_DIR,
    DuffingDataConfig,
    ModellingConfig,
    PhysicalDuffingParameters,
):
    system_name = "duffing"
    data_config = DuffingDataConfig()
    modelling_config = ModellingConfig.with_defaults(
        system_name=system_name,
        critic_window_size=64,
    )
    physical_params = PhysicalDuffingParameters()
    output_directory = modelling_config.output_directory(base_dir=DEFAULT_TRAINING_OUTPUT_DIR)
    return data_config, modelling_config, output_directory, physical_params


@app.cell
def _(DEVICE, data_config, physical_params, prepare_duffing_training_data):
    dataset_path, training_data, system, resolved_params = prepare_duffing_training_data(
        config=data_config,
        physical_params=physical_params,
        device=DEVICE,
    )

    print(f"Using training data cached at {dataset_path}")

    time_grid = training_data.time_grid.to(DEVICE)
    trajectories = training_data.trajectories.to(DEVICE)
    return system, time_grid, trajectories


@app.cell
def _(
    DEFAULT_METRIC_FILENAMES,
    DEFAULT_MODEL_FILENAMES,
    DEVICE,
    NeuralODE,
    NeuralSDE,
    data_config,
    modelling_config,
    output_directory,
    time_grid,
    trajectories,
):
    loaded = modelling_config.load_models(
        output_directory,
        model_filenames=DEFAULT_MODEL_FILENAMES,
        metric_filenames=DEFAULT_METRIC_FILENAMES,
        device=DEVICE,
    )

    if loaded is not None:
        neural_ode, neural_sde, loaded_metrics = loaded
        train_losses = list(loaded_metrics.get("train_losses", []))
        val_losses = list(loaded_metrics.get("val_losses", []))
        generator_losses = list(loaded_metrics.get("generator_losses", []))
        critic_losses = list(loaded_metrics.get("critic_losses", []))
        print(f"Loaded existing checkpoints from {output_directory}.")
    else:
        print("Training Neural ODE...")
        neural_ode = NeuralODE.train(
            hyperparameters=modelling_config.hyperparameters,
            trajectories=trajectories,
            time_grid=time_grid,
            device=DEVICE,
            validation_split=0.2,
            early_stopping_patience=modelling_config.hyperparameters.number_of_epochs // 4,
        )
        train_losses = list(neural_ode.training_losses)
        val_losses = list(neural_ode.validation_losses)

        print("Training Neural SDE...")
        neural_sde = NeuralSDE.train(
            hyperparameters=modelling_config.hyperparameters,
            neural_ode=neural_ode,
            trajectories=trajectories,
            time_grid=time_grid,
            device=DEVICE,
            enable_adversarial=modelling_config.enable_adversarial,
            random_seed=data_config.seed,
        )

        generator_losses = list(getattr(neural_sde, "generator_losses", []))
        critic_losses = list(getattr(neural_sde, "critic_losses", []))

        training_metrics = {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "generator_losses": generator_losses,
            "critic_losses": critic_losses,
        }

        modelling_config.save_models(
            output_directory,
            neural_ode=neural_ode,
            neural_sde=neural_sde,
            model_filenames=DEFAULT_MODEL_FILENAMES,
            metrics=training_metrics,
            metric_filenames=DEFAULT_METRIC_FILENAMES,
        )

    neural_ode = neural_ode.to(DEVICE)
    neural_sde = neural_sde.to(DEVICE)
    super(NeuralSDE, neural_sde).train(False)
    return (
        critic_losses,
        generator_losses,
        neural_ode,
        neural_sde,
        train_losses,
        val_losses,
    )


@app.cell
def _(neural_ode, time_grid, torch, trajectories):
    reference_state = trajectories[0:1, 0, :]
    with torch.no_grad():
        ode_prediction = neural_ode(reference_state, time_grid)[0]
    return (ode_prediction,)


@app.cell
def _(
    DEVICE,
    modelling_config,
    neural_sde,
    sample_sde_rollouts,
    time_grid,
    trajectories,
):
    rollout = sample_sde_rollouts(
        model=neural_sde,
        trajectories=trajectories,
        time_grid=time_grid,
        num_samples=modelling_config.sde_sample_count,
        device=DEVICE,
    )
    return (rollout,)


@app.cell
def _(compare_statistics, compute_statistics, rollout):
    training_stats = compute_statistics(rollout.training_subset)
    sde_stats = compute_statistics(rollout.rollout_tensor)
    _ = compare_statistics(training_stats, sde_stats)
    return sde_stats, training_stats


@app.cell
def _(
    COLOURS,
    critic_losses,
    finalise_plot,
    generator_losses,
    modelling_config,
    np,
    os,
    output_directory,
    plt,
    setup_matplotlib_style,
    train_losses,
    val_losses,
):
    setup_matplotlib_style()
    os.makedirs(output_directory, exist_ok=True)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    ode_epochs = modelling_config.hyperparameters.number_of_epochs
    gan_epochs = modelling_config.hyperparameters.number_of_gan_epochs
    total_epochs = ode_epochs + gan_epochs

    ode_epoch_range = np.arange(len(train_losses))

    ax1.plot(
        ode_epoch_range,
        train_losses,
        label="Training",
        color=COLOURS[0],
        linewidth=1.5,
    )
    ax1.plot(
        ode_epoch_range, val_losses, label="Validation", color=COLOURS[1], linewidth=1.5
    )

    ax1.axvline(
        x=len(train_losses) - 1,
        color="black",
        linestyle="--",
        alpha=0.7,
        linewidth=1.0,
    )
    ax1.text(
        len(train_losses) - 1,
        ax1.get_ylim()[1] * 0.9,
        "SDE Training Start",
        rotation=90,
        verticalalignment="top",
        horizontalalignment="right",
    )

    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.set_ylabel("SmoothL1 Loss")
    ax1.grid(False)

    sde_epochs = np.arange(ode_epochs, ode_epochs + len(generator_losses))

    if len(sde_epochs) == len(generator_losses):
        ax2.plot(sde_epochs, generator_losses, color=COLOURS[2], linewidth=1.5)
    ax2.set_ylabel("Generator Loss")
    ax2.grid(False)

    if len(sde_epochs) == len(critic_losses):
        ax3.plot(sde_epochs, critic_losses, color=COLOURS[3], linewidth=1.5)
    ax3.set_ylabel("Critic Loss")
    ax3.set_xlabel("Epoch")
    ax3.grid(False)

    if total_epochs > 1:
        ax1.set_xlim(0, total_epochs - 1)

    finalise_plot(fig, "training_losses.pdf", str(output_directory), os)
    return


@app.cell
def _(
    COLOURS,
    data_config,
    finalise_plot,
    np,
    ode_prediction,
    os,
    output_directory,
    plt,
    rollout,
    trajectories,
):
    phase_fig, phase_ax = plt.subplots(figsize=(6.0, 4.0))
    q_pred = ode_prediction[:, 0].detach().cpu().numpy()
    v_pred = ode_prediction[:, 1].detach().cpu().numpy()
    trajectories_np = trajectories.detach().cpu().numpy()
    n_show_phase = min(trajectories_np.shape[0], 200)
    training_opacity_phase = 1.0 if data_config.noise_disabled else 0.1
    for phase_idx in range(n_show_phase):
        phase_ax.plot(
            trajectories_np[phase_idx, :, 0],
            trajectories_np[phase_idx, :, 1],
            color="black",
            linewidth=0.6,
            alpha=training_opacity_phase,
            label="Training" if phase_idx == 0 else "",
        )
    phase_ax.plot(
        q_pred, v_pred, color=COLOURS[2], linewidth=1.0, label="Neural ODE"
    )
    if rollout.rollout_tensor.numel() > 0:
        phase_rollout_np = rollout.rollout_tensor[0].numpy()
        phase_ax.plot(
            phase_rollout_np[:, 0],
            phase_rollout_np[:, 1],
            color=COLOURS[3],
            linewidth=1.0,
            alpha=1.0,
            label="Neural SDE",
        )
    phase_ax.set_ylabel(r"$v(q(t))$ [ms$^{-1}$]")
    phase_ax.set_xlabel(r"$q(t)$ [m]")
    phase_ax.grid(False)
    phase_ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)

    if trajectories_np is not None:
        phase_ax.set_xlim(np.min(trajectories_np[:, :, 0]), np.max(trajectories_np[:, :, 0]))
        phase_ax.set_ylim(np.min(trajectories_np[:, :, 1]), np.max(trajectories_np[:, :, 1]))

    for phase_spine in phase_ax.spines.values():
        phase_spine.set_linewidth(1.0)
    finalise_plot(phase_fig, "phase_space.pdf", str(output_directory), os)
    return


@app.cell
def _(
    COLOURS,
    data_config,
    finalise_plot,
    np,
    ode_prediction,
    os,
    output_directory,
    plt,
    rollout,
    time_grid,
    trajectories,
):
    fig3, (axq, axv) = plt.subplots(1, 2, figsize=(10, 4))
    base_time = time_grid.detach().cpu().numpy().ravel()
    trajectories_array = trajectories.detach().cpu().numpy()
    pred_array = ode_prediction.detach().cpu().numpy()

    if base_time.size != 0:
        def _build_time_axis(length: int) -> np.ndarray:
            if length <= 1:
                return np.array([base_time[0]])
            return np.linspace(base_time[0], base_time[-1], length)

        training_time = _build_time_axis(trajectories_array.shape[1])
        ode_time = _build_time_axis(pred_array.shape[0])
        ntrain_show = min(50, trajectories_array.shape[0])
        training_opacity = 1.0 if data_config.noise_disabled else 0.05
        for itrain in range(ntrain_show):
            axq.plot(
                training_time,
                trajectories_array[itrain, :, 0],
                alpha=training_opacity,
                color="gray",
                linewidth=0.6,
                label="Training" if itrain == 0 else "",
            )
            axv.plot(
                training_time,
                trajectories_array[itrain, :, 1],
                alpha=training_opacity,
                color="gray",
                linewidth=0.6,
                label="Training" if itrain == 0 else "",
            )
        if rollout.rollout_tensor.numel() > 0:
            traj_rollout_np = rollout.rollout_tensor[0].numpy()
            sde_time = _build_time_axis(traj_rollout_np.shape[0])
            axq.plot(
                sde_time,
                traj_rollout_np[:, 0],
                color=COLOURS[3],
                linewidth=1.0,
                alpha=1.0,
                label="Neural SDE",
            )
            axv.plot(
                sde_time,
                traj_rollout_np[:, 1],
                color=COLOURS[3],
                linewidth=1.0,
                alpha=1.0,
                label="Neural SDE",
            )
        axq.plot(
            ode_time,
            pred_array[:, 0],
            color=COLOURS[2],
            linewidth=1.0,
            alpha=1.0,
            label="Neural ODE",
        )
        axv.plot(
            ode_time,
            pred_array[:, 1],
            color=COLOURS[2],
            linewidth=1.0,
            alpha=1.0,
            label="Neural ODE",
        )
        axq.set_xlabel(r"$t$ [s]")
        axq.set_ylabel(r"$q(t)$ [m]")
        axq.grid(False)
        axv.set_xlabel(r"$t$ [s]")
        axv.set_ylabel(r"$v(t)$ [m s$^{-1}$]")
        axv.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)
        axv.grid(False)
        for ax_item in [axq, axv]:
            for axis_spine in ax_item.spines.values():
                axis_spine.set_linewidth(1.0)
        finalise_plot(fig3, "trajectories.pdf", str(output_directory), os)
    return


@app.cell
def _(
    COLOURS,
    finalise_plot,
    np,
    os,
    output_directory,
    plt,
    rollout,
    sde_stats,
    time_grid,
    training_stats,
):
    if rollout.rollout_tensor.numel() > 0:
        fig4, (ax_stats_q, ax_stats_v) = plt.subplots(1, 2, figsize=(10, 3.5))
        t_stats = time_grid.detach().cpu().numpy().ravel()
        training_mean_plot = training_stats.mean.detach().cpu().numpy()
        training_var_plot = training_stats.variance.detach().cpu().numpy()
        sde_mean_plot = sde_stats.mean.detach().cpu().numpy()
        sde_var_plot = sde_stats.variance.detach().cpu().numpy()

        training_q_envelope = np.sqrt(training_var_plot[:, 0])
        sde_q_envelope = np.sqrt(sde_var_plot[:, 0])
        training_v_envelope = np.sqrt(training_var_plot[:, 1])
        sde_v_envelope = np.sqrt(sde_var_plot[:, 1])

        ax_stats_q.fill_between(
            t_stats,
            training_mean_plot[:, 0] - training_q_envelope,
            training_mean_plot[:, 0] + training_q_envelope,
            alpha=0.3,
            color="black",
            label="Training mean ± sqrt(var)",
        )
        ax_stats_q.plot(
            t_stats,
            training_mean_plot[:, 0],
            color="black",
            linewidth=2.0,
            label="Training mean",
        )
        ax_stats_q.fill_between(
            t_stats,
            sde_mean_plot[:, 0] - sde_q_envelope,
            sde_mean_plot[:, 0] + sde_q_envelope,
            alpha=0.3,
            color=COLOURS[3],
            label="Neural SDE mean ± sqrt(var)",
        )
        ax_stats_q.plot(
            t_stats,
            sde_mean_plot[:, 0],
            color=COLOURS[3],
            linewidth=2.0,
            label="Neural SDE mean",
        )
        ax_stats_q.set_xlabel(r"$t$ [s]")
        ax_stats_q.set_ylabel(r"$q(t)$ [m]")
        ax_stats_q.grid(False)

        ax_stats_v.fill_between(
            t_stats,
            training_mean_plot[:, 1] - training_v_envelope,
            training_mean_plot[:, 1] + training_v_envelope,
            alpha=0.3,
            color="black",
            label=r"Training mean $\pm\sigma$",
        )
        ax_stats_v.plot(
            t_stats,
            training_mean_plot[:, 1],
            color="black",
            linewidth=2.0,
            label="Training mean",
        )
        ax_stats_v.fill_between(
            t_stats,
            sde_mean_plot[:, 1] - sde_v_envelope,
            sde_mean_plot[:, 1] + sde_v_envelope,
            alpha=0.3,
            color=COLOURS[3],
            label=r"Neural SDE mean $\pm\sigma$",
        )
        ax_stats_v.plot(
            t_stats,
            sde_mean_plot[:, 1],
            color=COLOURS[3],
            linewidth=2.0,
            label="Neural SDE mean",
        )
        ax_stats_v.set_xlabel(r"$t$ [s]")
        ax_stats_v.set_ylabel(r"$v(t)$ [m s$^{-1}$]")
        ax_stats_v.grid(False)
        ax_stats_v.legend(bbox_to_anchor=(1.05, 1), loc="upper left", ncol=1)

        for stats_axis in (ax_stats_q, ax_stats_v):
            for stats_spine in stats_axis.spines.values():
                stats_spine.set_linewidth(1.0)

        finalise_plot(fig4, "statistics.pdf", str(output_directory), os)
    return


@app.cell
def _(
    data_config,
    np,
    rollout,
    sde_stats,
    system,
    time_grid,
    torch,
    training_stats,
):
    if rollout.rollout_tensor.numel() > 0:
        dt_scalar = float(time_grid[1] - time_grid[0])

        training_mean_metrics = training_stats.mean.detach().cpu().numpy()
        sde_mean_metrics = sde_stats.mean.detach().cpu().numpy()

        training_q_gradient = np.gradient(training_mean_metrics[:, 0], dt_scalar)
        sde_q_gradient = np.gradient(sde_mean_metrics[:, 0], dt_scalar)

        training_v_mean = training_mean_metrics[:, 1]
        sde_v_mean = sde_mean_metrics[:, 1]

        gradient_error_training = float(np.mean(np.abs(training_q_gradient - training_v_mean)))
        gradient_error_sde = float(np.mean(np.abs(sde_q_gradient - sde_v_mean)))

        sample_state_tensor = training_stats.mean[0, :].detach().cpu().to(torch.float32)
        diffusion_value = system.diffusion_function(0.0, sample_state_tensor).cpu().numpy()
        velocity_diffusion_squared = float(diffusion_value[1] ** 2)

        # Metrics for analysis (currently not exported/used)
        _ = {
            "gradient_error_training": gradient_error_training,
            "gradient_error_sde": gradient_error_sde,
            "velocity_diffusion_squared": velocity_diffusion_squared,
            "noise_disabled": data_config.noise_disabled,
        }
    return


if __name__ == "__main__":
    app.run()
