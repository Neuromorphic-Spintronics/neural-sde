from dataclasses import dataclass
from typing import Optional
import math


@dataclass(frozen=True)
class DuffingOscillatorParameters:
    """
    Dimensionless parameters for the stochastic Duffing oscillator SDE system.

    This class contains the dimensionless parameters after conversion from physical units.
    It should typically be created from PhysicalDuffingParameters via the to_dimensionless() method.
    All parameters must be explicitly provided - no default values to ensure consistency.
    """

    # System parameters
    alpha: float  # Linear stiffness parameter
    beta: float  # Nonlinear stiffness parameter
    mu: float  # Damping coefficient

    # External forcing
    gamma: float  # Forcing amplitude
    Omega: float  # Forcing frequency

    # White noise parameters
    Theta: float  # Temperature (white noise strength)

    # Coloured noise parameters
    D_tilde: float  # Noise intensity
    t_correl_tilde: float  # Correlation time

    # Numerical integration parameters
    timestep: float  # Integration timestep
    total_time: float  # Total simulation time

    # Initial conditions
    initial_position: float  # Initial position q(0)
    initial_velocity: float  # Initial velocity v(0)
    initial_coloured_noise: float  # Initial coloured noise xi(0)

    def __post_init__(self):
        """Validate parameters after initialisation."""
        if self.alpha >= 0:
            raise ValueError(
                "Linear stiffness alpha must be negative for double-well potential"
            )
        if self.mu < 0:
            raise ValueError("Damping coefficient mu must be non-negative")
        if self.beta <= 0:
            raise ValueError("Nonlinear stiffness beta must be positive")
        if self.Theta < 0:
            raise ValueError("Dimensionless temperature Theta must be non-negative")
        if self.D_tilde < 0:
            raise ValueError(
                "Dimensionless noise intensity D_tilde must be non-negative"
            )
        if self.t_correl_tilde <= 0:
            raise ValueError("Dimensionless correlation time must be positive")
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
        if self.total_time <= 0:
            raise ValueError("Total time must be positive")

    @property
    def white_noise_strength(self) -> float:
        """
        White noise strength: sqrt(2*mu*Theta)

        Returns:
            White noise diffusion coefficient
        """
        return math.sqrt(2 * self.mu * self.Theta)

    @property
    def coloured_noise_strength(self) -> float:
        """
        Coloured noise driving strength: sqrt(2*D_tilde/t_correl_tilde)

        Returns:
            Coloured noise diffusion coefficient
        """
        return math.sqrt(2 * self.D_tilde / self.t_correl_tilde)

    @property
    def equilibrium_positions(self) -> list[float]:
        """
        Calculate equilibrium positions for the double-well potential.

        For the Duffing potential V(q) = alpha*q^2/2 + beta*q^4/4:
        - Double well with equilibria at q = 0 (unstable), ±sqrt(-alpha/beta) (stable)

        Returns:
            List of equilibrium positions [0, -q_eq, q_eq]
        """
        # Double well case: equilibria at q = 0, ±sqrt(-alpha/beta)
        q_eq = math.sqrt(-self.alpha / self.beta)
        return [0.0, -q_eq, q_eq]

    @property
    def potential_type(self) -> str:
        """
        Return the potential type (always double-well).

        Returns:
            String describing potential type
        """
        return "double_well"


@dataclass(frozen=True)
class PhysicalDuffingParameters:
    """
    Physical parameters for the Duffing oscillator in SI units.

    This is the primary parameter class that defines the system in physical units.
    The dimensionless parameters are derived from these physical parameters using
    characteristic length and time scales.
    """

    # Physical system parameters
    mass: float = 1.0  # kg
    damping_coefficient: float = 0.15  # kg/s
    linear_stiffness: float = -1.0  # N/m
    nonlinear_stiffness: float = 1.0  # N/m^3

    # External forcing
    forcing_amplitude: float = 0.4  # N
    forcing_frequency: float = 0.9  # rad/s

    # Characteristic scales (if None, will be computed from physical parameters)
    characteristic_length: Optional[float] = None  # m
    characteristic_time: Optional[float] = None  # s

    # Thermal parameters
    temperature: float = 300.0  # K
    boltzmann_constant: float = 1.380649e-23  # J/K

    # Coloured noise parameters
    coloured_noise_intensity: float = 0.04  # s^3 / (m^2 * kg^2)
    coloured_noise_correlation_time: float = 0.5  # s

    # Initial conditions
    initial_position: float = 0.1  # m
    initial_velocity: float = 0.0  # m/s

    # Numerical integration parameters
    timestep: float = 0.005  # s
    total_time: float = 50.0  # s

    def __post_init__(self):
        """Validate physical parameters."""
        if self.mass <= 0:
            raise ValueError("Mass must be positive")
        if self.linear_stiffness >= 0:
            raise ValueError(
                "Linear stiffness must be negative for double-well potential"
            )
        if self.damping_coefficient < 0:
            raise ValueError("Damping coefficient must be non-negative")
        if self.nonlinear_stiffness <= 0:
            raise ValueError("Nonlinear stiffness must be positive")
        if self.temperature < 0:
            raise ValueError("Temperature must be non-negative")
        if self.coloured_noise_intensity < 0:
            raise ValueError("Coloured noise intensity must be non-negative")
        if self.coloured_noise_correlation_time <= 0:
            raise ValueError("Coloured noise correlation time must be positive")
        if self.timestep <= 0:
            raise ValueError("Timestep must be positive")
        if self.total_time <= 0:
            raise ValueError("Total time must be positive")

    @staticmethod
    def compute_characteristic_length(
        linear_stiffness: float, nonlinear_stiffness: float
    ) -> float:
        """
        Compute characteristic length for double-well: lambda = sqrt(-alpha/beta)

        Args:
            linear_stiffness: Linear stiffness coefficient (N/m) - must be negative
            nonlinear_stiffness: Nonlinear stiffness coefficient (N/m^3) - must be positive

        Returns:
            Characteristic length scale (m)
        """
        return math.sqrt(-linear_stiffness / nonlinear_stiffness)

    @staticmethod
    def compute_characteristic_time(linear_stiffness: float, mass: float) -> float:
        """
        Compute characteristic time: tau = sqrt(m / |alpha|)

        Args:
            linear_stiffness: Linear stiffness coefficient (N/m)
            mass: Mass of the oscillator (kg)

        Returns:
            Characteristic time scale (s)
        """
        if linear_stiffness != 0 and mass > 0:
            return math.sqrt(mass / abs(linear_stiffness))
        else:
            return 1.0  # fallback

    def to_dimensionless(self) -> DuffingOscillatorParameters:
        """
        Convert physical parameters to dimensionless form using characteristic scales.

        Returns:
            DuffingOscillatorParameters with dimensionless values
        """
        m = self.mass
        alpha = self.linear_stiffness
        beta = self.nonlinear_stiffness

        # Compute characteristic scales if not provided
        lambda_ = (
            self.characteristic_length
            if self.characteristic_length is not None
            else self.compute_characteristic_length(alpha, beta)
        )
        tau = (
            self.characteristic_time
            if self.characteristic_time is not None
            else self.compute_characteristic_time(alpha, m)
        )

        # Store computed characteristic scales back in the instance for reference
        # Note: Since dataclass is frozen, we use object.__setattr__
        if self.characteristic_length is None:
            object.__setattr__(self, "characteristic_length", lambda_)
        if self.characteristic_time is None:
            object.__setattr__(self, "characteristic_time", tau)

        # Dimensionless system parameters
        alpha_dim = alpha * tau**2 / m
        beta_dim = beta * lambda_**2 * tau**2 / m
        mu = self.damping_coefficient * tau / m
        gamma = self.forcing_amplitude * tau**2 / (m * lambda_)
        Omega = self.forcing_frequency * tau
        Theta = self.boltzmann_constant * self.temperature * tau**2 / (m * lambda_**2)
        D_tilde = self.coloured_noise_intensity * lambda_**2 * m**2 / tau**3
        t_correl_tilde = self.coloured_noise_correlation_time / tau

        # Dimensionless initial conditions
        initial_position_dim = self.initial_position / lambda_
        initial_velocity_dim = self.initial_velocity * tau / lambda_
        initial_coloured_noise_dim = 0.0  # Coloured noise starts at zero

        # Dimensionless time parameters
        timestep_dim = self.timestep / tau
        total_time_dim = self.total_time / tau

        return DuffingOscillatorParameters(
            alpha=alpha_dim,
            beta=beta_dim,
            mu=mu,
            gamma=gamma,
            Omega=Omega,
            Theta=Theta,
            D_tilde=D_tilde,
            t_correl_tilde=t_correl_tilde,
            timestep=timestep_dim,
            total_time=total_time_dim,
            initial_position=initial_position_dim,
            initial_velocity=initial_velocity_dim,
            initial_coloured_noise=initial_coloured_noise_dim,
        )
