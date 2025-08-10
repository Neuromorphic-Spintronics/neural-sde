from examples.duffing_oscillator import create_duffing_neural_ode_demonstration

if __name__ == "__main__":
    create_duffing_neural_ode_demonstration(
        num_trajectories=256,
        hidden_layer_sizes=[256, 256, 128],
        num_epochs=150,
        save_directory="examples/models",
        system_name="duffing_retrained",
    )
