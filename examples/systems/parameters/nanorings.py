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
    DRIFT_LEARNING_RATE: float = 3e-3
    DIFFUSION_LEARNING_RATE: float = 5e-4
    GENERATOR_LEARNING_RATE: float = 1e-4  # Lower generator LR for stability
    CRITIC_LEARNING_RATE: float = 5e-4  # Higher critic LR to learn faster
    NUMBER_OF_EPOCHS: int = 2**10
    BATCH_SIZE: int = 128  # Larger batches for efficiency
    GAN_BATCH_SIZE: int = 128  # Larger batches reduce number of iterations
    VALIDATION_SPLIT: float = 0.2
    EARLY_STOPPING_PATIENCE: int = 150

    # Neural SDE / GAN parameters
    NUMBER_OF_GAN_EPOCHS: int = 400  # Reduced for practical training (~90-120 min with keep_fraction=0.1)
    CRITIC_WINDOW_SIZE: int = 64  # Reduced to 64 for faster integration
    NUMBER_OF_SDE_ROLLOUT_SAMPLES: int = 256
    MOMENT_MATCHING_WEIGHT: float = 100.0  # Increased to 10.0 for stronger mean alignment during joint training
    MOMENT_MATCHING_ENABLED: bool = True  # Enable moment matching
    CRITIC_UPDATES: int = 1  # Single critic update since we reuse samples
    SDE_L1_WEIGHT: float = 2.0  # Increased to 2.0 for stronger pathwise supervision when training drift jointly
    DRIFT_L1_WEIGHT: float = 1.0  # Explicit drift-only SmoothL1 constraint to stabilize deterministic component
    GRADIENT_PENALTY_WEIGHT: float = 10.0
    TRAIN_ODE_WITH_SDE: bool = True  # Whether to continue training drift network during adversarial phase

    # Data sampling parameters
    PERCENTAGE_OF_FILES_TO_LOAD: float = 1.0
    PERCENTAGE_OF_MAIN_SEQUENCE_TO_USE: float = 0.05
    TARGET_CHANNEL: int = 0
    SIGNALS: int = 1

    # System dynamics parameters
    SAMPLE_RATE: int = 3200  # Hz
    H_FIELD_AMPLITUDE_UPDATE_RATE: int = 100  # Timesteps
    TRANSIENT_LENGTH: int = 300
    CONTEXT_POINTS: int = 0

    # Model architecture
    DRIFT_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        # amr, h, sin(wt), sin(2wt)
        input_size=1 + 1 + 2,
        hidden_sizes=[128, 128, 128, 128],
        output_size=1,
    )
    
    # Diffusion network architecture (for Neural SDE)
    # Smaller than drift to prevent noise from dominating
    DIFFUSION_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        input_size=1 + 1 + 2,  # Same as drift
        hidden_sizes=[64, 64],  # Smaller network for controlled stochasticity
        output_size=1,  # 1 noise dimension for 1-dimensional state
    )
    
    # Critic network architecture (for WGAN-GP)
    # Stronger critic architecture for better gradient quality
    CRITIC_NET_ARCH: NetworkArchitecture = NetworkArchitecture(
        input_size=CRITIC_WINDOW_SIZE * 1,  # window_size * state_dim (256 scalar values)
        hidden_sizes=[128, 128, 64],  # Stronger critic for better discrimination
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
