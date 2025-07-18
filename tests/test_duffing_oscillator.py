"""
Unit tests for the Duffing Oscillator implementation.
"""

from __future__ import annotations

import torch
import pytest
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import sys

# Add the current directory to path to import modules
sys.path.append(".")
from parameters import DuffingOscillatorParameters
from systems.duffing_oscillator import DuffingOscillator
from utils.plotting import set_default_plotting_style, style_axis_clean, COLOURS


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create figures directory
FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


class TestDuffingOscillatorParameters:
    """Test the parameter class for the Duffing oscillator."""

    def test_default_parameters(self):
        """Test that default parameters are sensible."""
        params = DuffingOscillatorParameters()

        # Check parameter types and ranges
        assert isinstance(params.delta, float)
        assert isinstance(params.alpha, float)
        assert isinstance(params.beta, float)
        assert params.delta > 0, "Damping should be positive"
        assert params.timestep > 0, "Timestep should be positive"
        assert params.total_time > 0, "Total time should be positive"

        # Check noise parameters
        assert (
            params.white_noise_strength >= 0
        ), "White noise strength should be non-negative"
        assert (
            params.coloured_noise_strength >= 0
        ), "Coloured noise strength should be non-negative"
        assert (
            params.coloured_noise_timescale > 0
        ), "Coloured noise timescale should be positive"

        logger.info("Default parameters validation passed")


class TestDuffingOscillator:
    """Test the main Duffing oscillator implementation."""

    @pytest.fixture
    def default_oscillator(self):
        """Create a default oscillator for testing."""
        params = DuffingOscillatorParameters(
            timestep=0.01,
            total_time=10.0,
            white_noise_strength=0.0,  # Start with deterministic case
            coloured_noise_strength=0.0,
        )
        return DuffingOscillator(params)

    @pytest.fixture
    def stochastic_oscillator(self):
        """Create a stochastic oscillator for testing."""
        params = DuffingOscillatorParameters(
            timestep=0.01,
            total_time=20.0,
            white_noise_strength=0.1,
            coloured_noise_strength=0.05,
        )
        return DuffingOscillator(params)

    def test_initialisation(self, default_oscillator):
        """Test oscillator initialisation."""
        osc = default_oscillator

        assert osc.dim == 3, "State dimension should be 3 (x, v, xi)"
        assert osc.num_steps > 0, "Number of steps should be positive"
        assert len(osc.time_grid) == osc.num_steps + 1, "Time grid length mismatch"

        logger.info("Oscillator initialisation test passed")


class TestDuffingOscillatorPlotting:
    """Test plotting functionality for the Duffing oscillator."""

    @pytest.fixture
    def plotting_oscillator(self):
        """Create oscillator optimised for plotting tests."""
        params = DuffingOscillatorParameters(
            timestep=0.01,
            total_time=50.0,
            forcing_amplitude=0.3,
            forcing_frequency=1.2,
            white_noise_strength=0.05,
            coloured_noise_strength=0.03,
            initial_position=0.1,
            initial_velocity=0.0,
        )
        return DuffingOscillator(params)

    def test_time_series_plot(self, plotting_oscillator):
        """Test time series plotting."""
        set_default_plotting_style(use_tex=True)

        osc = plotting_oscillator
        time_grid, trajectory = osc.integrate_sde()

        # Convert to numpy for plotting
        t = time_grid.numpy()
        x, v = osc.get_position_velocity(trajectory)
        x_np = x.numpy()
        v_np = v.numpy()
        xi_np = osc.get_coloured_noise(trajectory).numpy()

        # Create time series plot
        fig, axes = plt.subplots(3, 1, figsize=(7, 7))

        # Position
        axes[0].plot(t, x_np, color="k", linewidth=1, label=r"Position $x(t)$")
        axes[0].set_ylabel(r"Position $x$")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        style_axis_clean(axes[0])

        # Velocity
        axes[1].plot(t, v_np, color="k", linewidth=1, label=r"Velocity $\dot{x}(t)$")
        axes[1].set_ylabel(r"Velocity $\dot{x}$")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        style_axis_clean(axes[1])

        # Coloured noise
        axes[2].plot(t, xi_np, color="k", linewidth=1, label=r"Coloured noise $\xi(t)$")
        axes[2].set_xlabel(r"Time $t$")
        axes[2].set_ylabel(r"Noise $\xi$")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        style_axis_clean(axes[2])

        plt.tight_layout()

        # Save plot
        plot_path = FIGURES_DIR / "duffing_time_series.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        assert plot_path.exists(), "Time series plot should be saved"
        logger.info(f"Time series plot saved to {plot_path}")

    def test_phase_space_plot(self, plotting_oscillator):
        """Test phase space plotting."""
        set_default_plotting_style(use_tex=True)

        osc = plotting_oscillator
        time_grid, trajectory = osc.integrate_sde()

        x, v = osc.get_position_velocity(trajectory)
        x_np = x.numpy()
        v_np = v.numpy()

        # Create phase space plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        # Plot trajectory with colour mapping to time
        scatter = ax.scatter(
            x_np,
            v_np,
            c=time_grid.numpy(),
            cmap="magma",
            s=1,
            alpha=0.6,
            label=r"Trajectory",
        )

        # Mark initial condition
        ax.plot(x_np[0], v_np[0], "ro", markersize=8, label=r"Initial condition")

        # Mark final condition
        ax.plot(x_np[-1], v_np[-1], "rs", markersize=8, label=r"Final condition")

        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Velocity $\dot{x}$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axis_clean(ax)

        # Add colourbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label(r"Time $t$")

        plt.tight_layout()

        # Save plot
        plot_path = FIGURES_DIR / "duffing_phase_space.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        assert plot_path.exists(), "Phase space plot should be saved"
        logger.info(f"Phase space plot saved to {plot_path}")

    def test_energy_plot(self, plotting_oscillator):
        """Test energy evolution plotting."""
        set_default_plotting_style(use_tex=True)

        osc = plotting_oscillator
        time_grid, trajectory = osc.integrate_sde()

        x, v = osc.get_position_velocity(trajectory)

        # Compute energies
        kinetic = osc.kinetic_energy(v)
        potential = osc.potential_energy(x)
        total = osc.total_energy(x, v)

        # Convert to numpy
        t = time_grid.numpy()
        kinetic_np = kinetic.numpy()
        potential_np = potential.numpy()
        total_np = total.numpy()

        # Create energy plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        ax.plot(
            t, kinetic_np, color=COLOURS[0], linewidth=1, label=r"Kinetic Energy $T$"
        )
        ax.plot(
            t,
            potential_np,
            color=COLOURS[1],
            linewidth=1,
            label=r"Potential Energy $V$",
        )
        ax.plot(t, total_np, color=COLOURS[2], linewidth=1, label=r"Total Energy $E$")

        ax.set_xlabel(r"Time $t$")
        ax.set_ylabel(r"Energy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axis_clean(ax)

        plt.tight_layout()

        # Save plot
        plot_path = FIGURES_DIR / "duffing_energy.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        assert plot_path.exists(), "Energy plot should be saved"
        logger.info(f"Energy plot saved to {plot_path}")

    def test_potential_landscape_plot(self, plotting_oscillator):
        """Test plotting of the Duffing potential landscape."""
        set_default_plotting_style(use_tex=True)

        osc = plotting_oscillator

        # Create position range
        x_range = torch.linspace(-2.0, 2.0, 1000)
        potential = osc.potential_energy(x_range)

        # Convert to numpy
        x_np = x_range.numpy()
        V_np = potential.numpy()

        # Create potential plot
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))

        ax.plot(x_np, V_np, color="k", linewidth=1, label=r"Potential $V(x)$")

        # Mark equilibrium points
        if osc.params.alpha < 0:  # Double well potential
            x_eq = np.sqrt(-osc.params.alpha / osc.params.beta)
            ax.axvline(
                -x_eq,
                color="k",
                linewidth=1,
                linestyle="--",
                alpha=0.7,
                label=r"Stable equilibria",
            )
            ax.axvline(x_eq, color="k", linewidth=1, linestyle="--", alpha=0.7)
            ax.axvline(
                0,
                color="k",
                linewidth=1,
                linestyle="-.",
                alpha=0.7,
                label=r"Unstable equilibrium",
            )

        ax.set_xlabel(r"Position $x$")
        ax.set_ylabel(r"Potential Energy $V(x)$")
        ax.legend()
        ax.grid(True, alpha=0.3)
        style_axis_clean(ax)

        plt.tight_layout()

        # Save plot
        plot_path = FIGURES_DIR / "duffing_potential.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        assert plot_path.exists(), "Potential landscape plot should be saved"
        logger.info(f"Potential landscape plot saved to {plot_path}")


if __name__ == "__main__":
    # Run tests if script is executed directly
    pytest.main([__file__, "-v", "--tb=short"])
