"""
Unit tests for the Stochastic Duffing Oscillator implementation.

This module tests the implementation of the stochastic Duffing oscillator as described by the SDE system:
    dq(t) = v(t) dt
    dv(t) = (-alpha q(t) - beta q(t)^3 - mu v(t) + gamma cos(Omega t) + xi(t)) dt + sqrt(2*mu*Theta) ∘ dW(t)
    dxi(t) = -1/t_correl * xi(t) dt + sqrt(2*D_tilde/t_correl_tilde) ∘ dW(t)

where q(t) is dimensionless position, v(t) is dimensionless velocity, and xi(t) is the  Ornstein-Uhlenbeck process for coloured noise.
"""

from __future__ import annotations

import torch
import pytest
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import os
import json
import math

from examples.systems.parameters.duffing_oscillator import PhysicalDuffingParameters
from examples.systems.duffing_oscillator import DuffingOscillator
from neural_dynamics.core.utils import set_default_plotting_style, style_axis_clean, COLOURS

# Mark the entire module as slow due to long integrations and plotting
pytestmark = pytest.mark.slow

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)


class TestPhysicalParameterConversion:
    """Test the conversion between physical and dimensionless parameters."""

    def test_parameter_creation_and_json_export(self):
        """Test creation of physical parameters and export to JSON."""
        physical_params = PhysicalDuffingParameters(
            mass=1.0,  # kg
            damping_coefficient=0.15,  # kg/s
            linear_stiffness=-1.0,  # N/m
            nonlinear_stiffness=1.0,  # N/m^3
            forcing_amplitude=0.4,  # N
            forcing_frequency=0.9,  # rad/s
            temperature=300.0,  # K
            boltzmann_constant=1.380649e-23,  # J/K
            coloured_noise_intensity=0.04,  # m^2/s^3
            coloured_noise_correlation_time=0.5,  # s
            initial_position=0.1,  # m
            initial_velocity=0.0,  # m/s
            timestep=0.005,  # s
            total_time=50.0,  # s
        )

        dimensionless_params = physical_params.to_dimensionless()

        physical_dict = {
            "mass": physical_params.mass,
            "damping_coefficient": physical_params.damping_coefficient,
            "linear_stiffness": physical_params.linear_stiffness,
            "nonlinear_stiffness": physical_params.nonlinear_stiffness,
            "forcing_amplitude": physical_params.forcing_amplitude,
            "forcing_frequency": physical_params.forcing_frequency,
            "characteristic_length": physical_params.characteristic_length,
            "characteristic_time": physical_params.characteristic_time,
            "temperature": physical_params.temperature,
            "boltzmann_constant": physical_params.boltzmann_constant,
            "coloured_noise_intensity": physical_params.coloured_noise_intensity,
            "coloured_noise_correlation_time": physical_params.coloured_noise_correlation_time,
            "initial_position": physical_params.initial_position,
            "initial_velocity": physical_params.initial_velocity,
            "timestep": physical_params.timestep,
            "total_time": physical_params.total_time,
        }

        dimensionless_dict = {
            "alpha": dimensionless_params.alpha,
            "beta": dimensionless_params.beta,
            "mu": dimensionless_params.mu,
            "gamma": dimensionless_params.gamma,
            "Omega": dimensionless_params.Omega,
            "Theta": dimensionless_params.Theta,
            "D_tilde": dimensionless_params.D_tilde,
            "t_correl_tilde": dimensionless_params.t_correl_tilde,
            "initial_position": dimensionless_params.initial_position,
            "initial_velocity": dimensionless_params.initial_velocity,
            "timestep": dimensionless_params.timestep,
            "total_time": dimensionless_params.total_time,
        }

        full_params = {
            "physical": physical_dict,
            "dimensionless": dimensionless_dict,
            "description": "Stochastic Duffing oscillator parameters with physical and dimensionless forms",
        }

        json_path = DATA_DIR / "duffing_parameters.json"
        with open(json_path, "w") as f:
            json.dump(full_params, f, indent=2)

        assert json_path.exists(), "Parameter JSON file should be created"

        with open(json_path, "r") as f:
            loaded_params = json.load(f)

        assert "physical" in loaded_params
        assert "dimensionless" in loaded_params
        assert loaded_params["dimensionless"]["alpha"] == dimensionless_dict["alpha"]

        logger.info(f"Physical and dimensionless parameters saved to {json_path}")


class TestDuffingOscillatorParameters:
    """Test the new parameter class structure."""

    def test_parameter_validation(self):
        """Test that parameters satisfy physical constraints."""
        params = PhysicalDuffingParameters().to_dimensionless()

        assert params.mu >= 0, "Dimensionless damping mu should be non-negative"
        assert params.timestep > 0, "Timestep should be positive"
        assert params.total_time > 0, "Total time should be positive"
        assert params.Theta >= 0, (
            "Dimensionless temperature Theta should be non-negative"
        )
        assert params.D_tilde >= 0, (
            "Dimensionless noise intensity D_tilde should be non-negative"
        )
        assert params.t_correl_tilde > 0, (
            "Dimensionless correlation time should be positive"
        )

        logger.info("Parameter validation passed")

    def test_equilibrium_points(self):
        """Test calculation of equilibrium points for double-well potential."""
        # Double well potential: alpha < 0, beta > 0
        physical_double_well = PhysicalDuffingParameters(
            linear_stiffness=-1.0, nonlinear_stiffness=1.0
        )
        params_double_well = physical_double_well.to_dimensionless()
        assert params_double_well.alpha < 0, "Should create double well potential"

        # For double well, equilibrium points at q = 0, ±sqrt(-alpha/beta)
        q_eq = math.sqrt(-params_double_well.alpha / params_double_well.beta)
        assert q_eq > 0, "Equilibrium position should be positive"

        equilibria = params_double_well.equilibrium_positions
        assert len(equilibria) == 3, "Should have 3 equilibrium points"
        assert equilibria[0] == 0.0, "Origin should be an equilibrium"
        assert equilibria[1] == -q_eq, "Left well should be at -q_eq"
        assert equilibria[2] == q_eq, "Right well should be at q_eq"

        logger.info("Equilibrium point calculations passed")


class TestStochasticDuffingOscillator:
    """Test the main stochastic Duffing oscillator implementation."""

    @pytest.fixture
    def deterministic_oscillator(self) -> DuffingOscillator:
        """Create a deterministic oscillator for testing."""
        physical_params = PhysicalDuffingParameters(
            mass=1.0,
            damping_coefficient=0.15,
            linear_stiffness=-1.0,
            nonlinear_stiffness=1.0,
            forcing_amplitude=0.4,
            forcing_frequency=0.9,
            temperature=0.0,  # No thermal noise
            coloured_noise_intensity=0.0,  # No coloured noise
            timestep=0.01,
            total_time=10.0,
        )
        params = physical_params.to_dimensionless()
        return DuffingOscillator(params)

    @pytest.fixture
    def stochastic_oscillator(self) -> DuffingOscillator:
        """Create a stochastic oscillator for testing."""
        physical_params = PhysicalDuffingParameters(
            mass=1.0,
            damping_coefficient=0.15,
            linear_stiffness=-1.0,
            nonlinear_stiffness=1.0,
            forcing_amplitude=0.4,
            forcing_frequency=0.9,
            temperature=1.0,  # Cold temperature (1 K)
            coloured_noise_intensity=0.005,  # Small coloured noise
            coloured_noise_correlation_time=0.5,
            timestep=0.005,
            total_time=20.0,
        )
        params = physical_params.to_dimensionless()
        return DuffingOscillator(params)

    def test_state_vector_structure(self, stochastic_oscillator):
        """Test that the state vector has correct structure [q, v, xi]."""
        osc = stochastic_oscillator

        assert osc.get_state_dimension() == 3, "State dimension should be 3"

        initial_state = osc.get_initial_state()
        assert initial_state.shape == torch.Size([1, 3]), (
            "Initial state should be batched with 3 components"
        )

        logger.info("State vector structure test passed")

    def test_drift_function_structure(self, stochastic_oscillator):
        """Test the drift function implements the correct SDE structure."""
        osc = stochastic_oscillator

        test_state = torch.tensor([0.5, 0.1, 0.05])
        test_time = 1.0

        drift = osc.drift_function(test_time, test_state)

        assert drift.shape == torch.Size([3]), "Drift should have 3 components"

        assert torch.isclose(drift[0], test_state[1]), "dq/dt should equal velocity v"

        q, v, xi = test_state
        expected_acceleration = (
            -osc.params.alpha * q
            - osc.params.beta * torch.pow(q, 3)
            - osc.params.mu * v
            + osc.params.gamma * torch.cos(osc.params.Omega * torch.tensor(test_time))
            + xi
        )
        assert torch.isclose(drift[1], expected_acceleration, atol=1e-6), (
            "Acceleration equation incorrect"
        )

        expected_xi_drift = -xi / osc.params.t_correl_tilde
        assert torch.isclose(drift[2], expected_xi_drift), (
            "Coloured noise drift incorrect"
        )

        logger.info("Drift function structure test passed")

    def test_diffusion_function_structure(self, stochastic_oscillator):
        """Test the diffusion function implements correct noise strengths."""
        osc = stochastic_oscillator

        test_state = torch.tensor([0.5, 0.1, 0.05])
        test_time = 1.0

        diffusion = osc.diffusion_function(test_time, test_state)

        assert diffusion.shape == torch.Size([3]), "Diffusion should have 3 components"

        assert torch.isclose(diffusion[0], torch.tensor(0.0)), "No noise on position"

        expected_white_noise = math.sqrt(2 * osc.params.mu * osc.params.Theta)
        assert torch.isclose(diffusion[1], torch.tensor(expected_white_noise)), (
            "White noise strength incorrect"
        )

        expected_coloured_noise = math.sqrt(
            2 * osc.params.D_tilde / osc.params.t_correl_tilde
        )
        assert torch.isclose(diffusion[2], torch.tensor(expected_coloured_noise)), (
            "Coloured noise strength incorrect"
        )

        logger.info("Diffusion function structure test passed")

    def test_deterministic_integration(self, deterministic_oscillator):
        """Test integration of deterministic case (no noise)."""
        osc = deterministic_oscillator

        torch.manual_seed(42)

        time_grid, trajectory = osc.integrate_sde()

        assert time_grid.shape[0] == osc.num_steps + 1, "Time grid length incorrect"
        assert trajectory.shape == (1, osc.num_steps + 1, 3), "Trajectory shape incorrect"

        assert not torch.any(torch.isnan(trajectory)), "Trajectory contains NaN"
        assert not torch.any(torch.isinf(trajectory)), (
            "Trajectory contains infinite values"
        )

        trajectory_single = trajectory.squeeze(0)

        xi_trajectory = trajectory_single[:, 2]
        # Note: xi will decay towards zero due to -xi/t_correl term
        assert torch.all(torch.abs(xi_trajectory) < 1e-10), (
            "Coloured noise should remain near zero in deterministic case"
        )

        logger.info("Deterministic integration test passed")

    def test_stochastic_integration(self, stochastic_oscillator):
        """Test integration of stochastic case with noise."""
        osc = stochastic_oscillator

        torch.manual_seed(123)

        time_grid, trajectory = osc.integrate_sde()

        assert time_grid.shape[0] == osc.num_steps + 1, "Time grid length incorrect"
        assert trajectory.shape == (1, osc.num_steps + 1, 3), "Trajectory shape incorrect"

        assert not torch.any(torch.isnan(trajectory)), "Trajectory contains NaN"
        assert not torch.any(torch.isinf(trajectory)), (
            "Trajectory contains infinite values"
        )

        trajectory_single = trajectory.squeeze(0)

        xi_trajectory = trajectory_single[:, 2]
        xi_variance = torch.var(xi_trajectory)
        assert xi_variance > 1e-6, (
            "Coloured noise should show fluctuations in stochastic case"
        )

        logger.info("Stochastic integration test passed")


class TestEnergyConservationAndPhysics:
    """Test physical properties like energy and phase space behaviour."""

    @pytest.fixture
    def physics_oscillator(self) -> DuffingOscillator:
        """Create oscillator for physics testing."""
        physical_params = PhysicalDuffingParameters(
            mass=1.0,
            damping_coefficient=0.05,  # Low damping
            linear_stiffness=-1.0,  # Double well
            nonlinear_stiffness=1.0,
            forcing_amplitude=0.0,  # No external forcing
            forcing_frequency=0.9,
            temperature=0.0,  # No noise for energy conservation test
            coloured_noise_intensity=0.0,
            timestep=0.001,
            total_time=10.0,
        )
        params = physical_params.to_dimensionless()
        return DuffingOscillator(params)

    def test_potential_energy_calculation(self, physics_oscillator):
        """Test the Duffing potential energy calculation."""
        osc = physics_oscillator

        q_test = torch.tensor([0.0, 1.0, -1.0, 2.0])
        potential = osc.potential_energy(q_test)

        expected_potential = (
            osc.params.alpha * torch.pow(q_test, 2) / 2
            + osc.params.beta * torch.pow(q_test, 4) / 4
        )

        assert torch.allclose(potential, expected_potential), (
            "Potential energy calculation incorrect"
        )

        if osc.params.alpha < 0 and osc.params.beta > 0:
            q_min = math.sqrt(-osc.params.alpha / osc.params.beta)
            V_min = osc.potential_energy(torch.tensor(q_min))
            V_zero = osc.potential_energy(torch.tensor(0.0))
            assert V_min < V_zero, (
                "Double well minima should be lower than central maximum"
            )

        logger.info("Potential energy calculation test passed")

    def test_energy_evolution_undamped(self, physics_oscillator):
        """Test energy evolution in undamped, unforced system."""
        physical_params = PhysicalDuffingParameters(
            mass=1.0,
            damping_coefficient=0.0,  # No damping
            linear_stiffness=-1.0,
            nonlinear_stiffness=1.0,
            forcing_amplitude=0.0,  # No forcing
            forcing_frequency=0.9,
            temperature=0.0,  # No noise
            coloured_noise_intensity=0.0,
            initial_position=1.2,  # Start away from equilibrium
            initial_velocity=0.0,
            timestep=0.001,
            total_time=5.0,
        )
        params = physical_params.to_dimensionless()
        osc = DuffingOscillator(params)

        time_grid, trajectory = osc.integrate_sde()
        q, v = osc.get_position_velocity(trajectory)
        q = q.squeeze(0)
        v = v.squeeze(0)

        total_energy = osc.total_energy(q, v)

        energy_drift = torch.abs(total_energy[-1] - total_energy[0]) / torch.abs(
            total_energy[0]
        )
        assert energy_drift < 0.01, (
            f"Energy drift {energy_drift:.6f} too large for undamped system"
        )

        logger.info("Energy conservation test passed")


class TestPlottingAndVisualisation:
    """Test plotting capabilities and visual output."""

    @pytest.fixture
    def plotting_oscillator(self) -> DuffingOscillator:
        """Create oscillator optimised for plotting."""
        physical_params = PhysicalDuffingParameters(
            mass=1.0,
            damping_coefficient=0.1,
            linear_stiffness=-1.0,  # Double well
            nonlinear_stiffness=1.0,
            forcing_amplitude=0.3,
            forcing_frequency=1.2,
            temperature=300.0,  # Room temperature
            coloured_noise_intensity=0.01,
            coloured_noise_correlation_time=0.8,
            initial_position=0.1,
            initial_velocity=0.0,
            timestep=0.01,
            total_time=50.0,
        )
        params = physical_params.to_dimensionless()
        return DuffingOscillator(params)

    def test_phase_space_plot(self, plotting_oscillator):
        """Test phase space plotting with coloured noise."""
        use_tex = not os.environ.get("CI", False)
        set_default_plotting_style(use_tex=use_tex)

        osc = plotting_oscillator
        torch.manual_seed(42)
        time_grid, trajectory = osc.integrate_sde()

        q, v = osc.get_position_velocity(trajectory)
        q_np = q.squeeze(0).cpu().numpy()
        v_np = v.squeeze(0).cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        scatter = ax1.scatter(
            q_np, v_np, c=time_grid.cpu().numpy(), cmap="magma", s=1, alpha=0.7
        )
        ax1.plot(q_np[0], v_np[0], "k^", markersize=8, label="Initial")
        ax1.plot(q_np[-1], v_np[-1], "ks", markersize=8, label="Final")
        ax1.set_xlabel(r"$q$")
        ax1.set_ylabel(r"$v$")
        ax1.set_title("Phase Space Trajectory")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        style_axis_clean(ax1)

        cbar = plt.colorbar(scatter, ax=ax1)
        cbar.set_label(r"$t$")

        q_range = torch.linspace(-2.5, 2.5, 1000)
        V = osc.potential_energy(q_range)
        ax2.plot(q_range.cpu().numpy(), V.cpu().numpy(), "k-", linewidth=2)
        ax2.set_xlabel(r"$q$")
        ax2.set_ylabel(r"$V(q)$")
        ax2.set_title("Duffing Potential")
        ax2.grid(True, alpha=0.3)
        style_axis_clean(ax2)

        if osc.params.alpha < 0:
            q_eq = math.sqrt(-osc.params.alpha / osc.params.beta)
            ax2.axvline(-q_eq, color="black", linestyle="--", alpha=0.7, label="Stable")
            ax2.axvline(q_eq, color="black", linestyle="--", alpha=0.7)
            ax2.axvline(0, color="black", linestyle="-.", alpha=0.7, label="Unstable")
            ax2.legend()

        plt.tight_layout()

        plot_path = FIGURES_DIR / "stochastic_duffing_phase_space.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        assert plot_path.exists(), "Phase space plot should be saved"
        logger.info(f"Enhanced phase space plot saved to {plot_path}")

    def test_noise_analysis_plot(self, plotting_oscillator):
        """Test plotting of noise components and their effects."""
        use_tex = not os.environ.get("CI", False)
        set_default_plotting_style(use_tex=use_tex)

        osc = plotting_oscillator
        torch.manual_seed(42)
        time_grid, trajectory = osc.integrate_sde()

        q, v = osc.get_position_velocity(trajectory)
        q = q.squeeze(0)
        v = v.squeeze(0)
        xi = trajectory[0, :, 2]

        t_np = time_grid.cpu().numpy()
        xi_np = xi.cpu().numpy()

        fig_noise = plt.figure(figsize=(12, 4))

        gs_noise = fig_noise.add_gridspec(1, 2, width_ratios=[1, 1])

        ax1 = fig_noise.add_subplot(gs_noise[0, 0])
        ax2 = fig_noise.add_subplot(gs_noise[0, 1])

        # The white noise affects the velocity equation: dv/dt = ... + √(2μΘ) dW/dt
        white_noise_strength = math.sqrt(2 * osc.params.mu * osc.params.Theta)
        torch.manual_seed(42)  # For reproducibility
        white_noise = (torch.randn(len(t_np)) * white_noise_strength).numpy()
        ax1.plot(t_np, white_noise, color="black", linewidth=1, label=r"$\eta_1(t)$")
        ax1.set_xlabel(r"Time $t$")
        ax1.set_ylabel(r"$\eta_1(t)$")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        style_axis_clean(ax1)

        ax2.plot(t_np, xi_np, color="black", linewidth=1, label=r"$\eta_2(t)$")
        ax2.set_xlabel(r"Time $t$")
        ax2.set_ylabel(r"$\eta_2(t)$")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        style_axis_clean(ax2)

        plt.tight_layout()

        noise_plot_path = FIGURES_DIR / "stochastic_duffing_noise_analysis.png"
        plt.savefig(noise_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        assert noise_plot_path.exists(), "Noise analysis plot should be saved"
        logger.info(f"Noise analysis plot saved to {noise_plot_path}")

        fig_energy = plt.figure(figsize=(8, 6))
        ax_energy = fig_energy.add_subplot(111)

        kinetic_energy = osc.kinetic_energy(v)
        potential_energy = osc.potential_energy(q)
        total_energy = osc.total_energy(q, v)

        ax_energy.plot(
            t_np,
            kinetic_energy.cpu().numpy(),
            color=COLOURS[0],
            linewidth=1,
            label=r"Kinetic $T$",
        )
        ax_energy.plot(
            t_np,
            potential_energy.cpu().numpy(),
            color=COLOURS[1],
            linewidth=1,
            label=r"Potential $V$",
        )
        ax_energy.plot(
            t_np,
            total_energy.cpu().numpy(),
            color="black",
            linewidth=1,
            label=r"$E(t)$",
        )
        ax_energy.set_xlabel(r"Time $t$")
        ax_energy.set_ylabel(r"$E(t)$")
        ax_energy.legend()
        ax_energy.grid(True, alpha=0.3)
        style_axis_clean(ax_energy)

        plt.tight_layout()

        energy_plot_path = FIGURES_DIR / "stochastic_duffing_energy_analysis.png"
        plt.savefig(energy_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        assert energy_plot_path.exists(), "Energy analysis plot should be saved"
        logger.info(f"Energy analysis plot saved to {energy_plot_path}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

    tester = TestPhysicalParameterConversion()
    tester.test_parameter_creation_and_json_export()

    pytest.main([__file__ + "::TestStochasticDuffingOscillator", "-v"])
