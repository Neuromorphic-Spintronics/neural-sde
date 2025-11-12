from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from neural_dynamics.core.hyperparameters import (
    Hyperparameters,
    LearningRates,
    NetworkArchitecture,
)


@dataclass(frozen=True)
class NanoringsHyperparameters:
    """Hyperparameters and configuration for the nanorings digital twin experiment."""

    # Runtime configuration
    dataset_path: Path = Path("examples/data/nanorings_first_set.pt")
    keep_fraction: float = 0.04
    enable_standardisation: bool = True
    device_override: Optional[str] = None
    enable_wandb: bool = True

    # Training parameters
    DRIFT_LEARNING_RATE: float = 3e-3
    DIFFUSION_LEARNING_RATE: float = 5e-4
    GENERATOR_LEARNING_RATE: float = 5e-4
    CRITIC_LEARNING_RATE: float = 5e-4
    NUMBER_OF_EPOCHS: int = 2**12
    BATCH_SIZE: int = 128
    GAN_BATCH_SIZE: int = 128
    VALIDATION_SPLIT: float = 0.2
    EARLY_STOPPING_PATIENCE: int = 150

    # Neural SDE / GAN parameters
    NUMBER_OF_GAN_EPOCHS: int = 2048 
    CRITIC_WINDOW_SIZE: int = 64
    NUMBER_OF_SDE_ROLLOUT_SAMPLES: int = 256
    MOMENT_MATCHING_WEIGHT: float = 100.0
    MOMENT_MATCHING_ENABLED: bool = True
    CRITIC_UPDATES: int = 1
    SDE_L1_WEIGHT: float = 1.0
    DRIFT_L1_WEIGHT: float = 1.0
    GRADIENT_PENALTY_WEIGHT: float = 10.0
    TRAIN_ODE_WITH_SDE: bool = True

    # Data sampling parameters
    PERCENTAGE_OF_FILES_TO_LOAD: float = 1.0
    PERCENTAGE_OF_MAIN_SEQUENCE_TO_USE: float = 0.05
    TARGET_CHANNEL: int = 0
    SIGNALS: int = 1
    TARGET_TIMESTEPS: int = 256  # How many time steps should each split of the data be?

    # System dynamics parameters
    SAMPLE_RATE: int = 3200  # Hz
    H_FIELD_AMPLITUDE_UPDATE_RATE: int = 100  # Timesteps
    TRANSIENT_LENGTH: int = 300
    CONTEXT_POINTS: int = 0

    # Model architecture
    DRIFT_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        # amr, h, sin(wt), sin(2wt)
        input_size=1 + 1 + 2,
        hidden_sizes=[128, 128, 128],
        output_size=1,
    )
    
    # Diffusion network architecture (for Neural SDE)
    DIFFUSION_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        input_size=1 + 1 + 2,
        hidden_sizes=[128, 128, 128],
        output_size=1,
    )
    
    # Critic network architecture (for WGAN-GP)
    CRITIC_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        input_size=CRITIC_WINDOW_SIZE * 1,  # window_size * state_dim (TARGET_TIMESTEPS scalar values)
        hidden_sizes=[128, 128, 128], 
        output_size=1,
    )
    
    def to_hyperparameters(self) -> Hyperparameters:
        """Convert to standard Hyperparameters object for Neural SDE training."""
        return Hyperparameters(
            # Network architectures
            drift_network=self.DRIFT_NET_ARCH,
            diffusion_network=self.DIFFUSION_NET_ARCH,
            critic_network=self.CRITIC_NET_ARCH,
            
            # System dimensions
            state_dimension=1,  # AMR is 1-dimensional
            input_dimension=1 + 2,  # h, sin(wt), sin(2wt)
            
            # Training parameters
            timestep=1.0 / self.SAMPLE_RATE,  # Convert sample rate to timestep
            learning_rates=LearningRates(
                drift=self.DRIFT_LEARNING_RATE,
                diffusion=self.DIFFUSION_LEARNING_RATE,
                critic=self.CRITIC_LEARNING_RATE,
                generator=self.GENERATOR_LEARNING_RATE,
            ),
            number_of_epochs=self.NUMBER_OF_EPOCHS,
            number_of_gan_epochs=self.NUMBER_OF_GAN_EPOCHS,
            batch_size=self.BATCH_SIZE,
            critic_updates=self.CRITIC_UPDATES,
            gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT,
            sde_l1_weight=self.SDE_L1_WEIGHT,
            drift_l1_weight=self.DRIFT_L1_WEIGHT,
            moment_matching_weight=self.MOMENT_MATCHING_WEIGHT,
            moment_matching_enabled=self.MOMENT_MATCHING_ENABLED,
            train_ode_with_sde=self.TRAIN_ODE_WITH_SDE,
        )
