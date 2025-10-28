from dataclasses import dataclass
from neural_dynamics.core.hyperparameters import (
    Hyperparameters,
    LearningRates,
    NetworkArchitecture,
)


@dataclass(frozen=True)
class NanoringsHyperparameters:
    """Hyperparameters for the Nanorings digital twin experiment."""

        # Training parameters
    LEARNING_RATE: float = 3e-3  # Aggressive learning rate for strong ODE convergence
    CRITIC_LEARNING_RATE: float = 1e-5  # Very low for stable critic training (prevent collapse)
    NUMBER_OF_EPOCHS: int = 2048  # Many epochs for thorough optimisation
    BATCH_SIZE: int = 64  # Larger batch for stable, smooth gradients (was 32)
    VALIDATION_SPLIT: float = 0.2
    EARLY_STOPPING_PATIENCE: int = 300  # Much more patience for deep convergence (was 150)
    
    # Neural SDE / GAN parameters
    NUMBER_OF_GAN_EPOCHS: int = 1024  # Match ODE epochs for balanced training
    CRITIC_WINDOW_SIZE: int = 128  # Longer window to see more dynamics
    NUMBER_OF_SDE_ROLLOUT_SAMPLES: int = 500  # More samples for robust statistics
    MOMENT_MATCHING_WEIGHT: float = 0.1  # Reduced to avoid dominating adversarial loss
    MOMENT_MATCHING_ENABLED: bool = True  # Enable moment matching loss
    CRITIC_UPDATES: int = 5  # Number of critic updates per generator update (more stable)
    SDE_L1_WEIGHT: float = 0.5  # Increased pathwise matching (was 0.25)
    GRADIENT_PENALTY_WEIGHT: float = 10.0  # WGAN-GP gradient penalty (standard value)

    # Data sampling parameters
    PERCENTAGE_OF_FILES_TO_LOAD: float = 1.0
    PERCENTAGE_OF_MAIN_SEQUENCE_TO_USE: float = 0.05
    TARGET_CHANNEL: int = 0
    SIGNALS: int = 50

    # System dynamics parameters
    SAMPLE_RATE: int = 3200  # Hz
    H_FIELD_AMPLITUDE_UPDATE_RATE: int = 100  # Timesteps
    TRANSIENT_LENGTH: int = 300
    CONTEXT_POINTS: int = 5

    # Model architecture
    DRIFT_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        # amr, h, sin(wt), sin(2wt), h_ctx, amr_ctx
        input_size=1 + 1 + 2 + CONTEXT_POINTS + CONTEXT_POINTS,
        hidden_sizes=[128, 128, 128],
        output_size=1,
    )
    
    # Diffusion network architecture (for Neural SDE)
    DIFFUSION_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        input_size=1 + 1 + 2 + CONTEXT_POINTS + CONTEXT_POINTS,  # Same as drift
        hidden_sizes=[128, 128, 128],
        output_size=1,  # 1 noise dimension for 1-dimensional state
    )
    
    # Critic network architecture (for WGAN-GP)
    # Much smaller architecture to prevent overfitting on trajectory windows
    CRITIC_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        input_size=CRITIC_WINDOW_SIZE * 1,  # window_size * state_dim (256 scalar values)
        hidden_sizes=[64, 32],  # Small network to prevent rapid collapse
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
            input_dimension=1 + 2 + self.CONTEXT_POINTS + self.CONTEXT_POINTS,  # h, sin(wt), sin(2wt), contexts
            
            # Training parameters
            timestep=1.0 / self.SAMPLE_RATE,  # Convert sample rate to timestep
            learning_rates=LearningRates(
                drift=self.LEARNING_RATE,
                diffusion=self.LEARNING_RATE,
                critic=self.CRITIC_LEARNING_RATE,  # Separate lower rate for critic
                generator=self.LEARNING_RATE,
            ),
            number_of_epochs=self.NUMBER_OF_EPOCHS,
            number_of_gan_epochs=self.NUMBER_OF_GAN_EPOCHS,
            batch_size=self.BATCH_SIZE,
            critic_updates=self.CRITIC_UPDATES,
            gradient_penalty_weight=self.GRADIENT_PENALTY_WEIGHT,
            sde_l1_weight=self.SDE_L1_WEIGHT,
            moment_matching_weight=self.MOMENT_MATCHING_WEIGHT,
            moment_matching_enabled=self.MOMENT_MATCHING_ENABLED,
        )
