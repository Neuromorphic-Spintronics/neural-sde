from dataclasses import dataclass
from neural_dynamics.core.hyperparameters import NetworkArchitecture


@dataclass(frozen=True)
class NanoringsHyperparameters:
    """Hyperparameters for the Nanorings digital twin experiment."""

    # Training parameters
    LEARNING_RATE: float = 3.4e-4
    NUM_EPOCHS: int = 1024
    BATCH_SIZE: int = 16
    VALIDATION_SPLIT: float = 0.2
    EARLY_STOPPING_PATIENCE: int = 100

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
