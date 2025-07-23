"""
Unit tests for the DiffusionNet implementation in the neural SDE framework.
"""

from __future__ import annotations
import torch
import pytest
import sys
import os
import matplotlib.pyplot as plt

sys.path.append(".")
from parameters.hyperparameters import NetworkArchitecture
from models.neural_sde import DiffusionNet
from config import DEVICE
from utils import plotting


class TestDiffusionNet:
    @pytest.mark.parametrize("arch,state_dim,noise_dim,activation", [
        (NetworkArchitecture(6, [32, 16], 4), 2, 2, torch.nn.Tanh),
        (NetworkArchitecture(12, [64], 6), 3, 2, torch.nn.ReLU),
        (NetworkArchitecture(15, [16, 8], 6), 2, 3, torch.nn.Sigmoid),
    ])
    def test_initialisation_and_device(self, arch, state_dim, noise_dim, activation):
        net = DiffusionNet(arch, state_dimension=state_dim, noise_dimension=noise_dim, activation=activation)
        assert net.architecture == arch
        assert isinstance(net, DiffusionNet)
        assert next(net.parameters()).device.type == DEVICE.type
        assert net.state_dimension == state_dim
        assert net.noise_dimension == noise_dim

    @pytest.mark.parametrize("batch_size,state_dim,noise_dim,input_dim,hidden_sizes", [
        (1, 2, 2, 2, [8]),
        (16, 3, 2, 4, [16, 8]),
        (32, 4, 3, 5, [32]),
    ])
    def test_forward_pass_various_batch_sizes(self, batch_size, state_dim, noise_dim, input_dim, hidden_sizes):
        # input_size = state_dim + time (1) + input_dim
        input_size = state_dim + 1 + input_dim
        output_size = state_dim * noise_dim
        arch = NetworkArchitecture(input_size, hidden_sizes, output_size)
        net = DiffusionNet(arch, state_dimension=state_dim, noise_dimension=noise_dim)
        state = torch.randn(batch_size, state_dim).to(DEVICE)
        time = torch.randn(batch_size, 1).to(DEVICE)
        external_inputs = torch.randn(batch_size, input_dim).to(DEVICE)
        output = net.compute_diffusion(state, time, external_inputs)
        assert output.shape == (batch_size, state_dim, noise_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_forward_pass_with_missing_external_inputs(self):
        # Should handle missing external_inputs by filling with zeros
        batch_size, state_dim, noise_dim, input_dim = 8, 2, 2, 3
        input_size = state_dim + 1 + input_dim
        output_size = state_dim * noise_dim
        arch = NetworkArchitecture(input_size, [8], output_size)
        net = DiffusionNet(arch, state_dimension=state_dim, noise_dimension=noise_dim)
        state = torch.randn(batch_size, state_dim, device=DEVICE)
        time = torch.randn(batch_size, 1, device=DEVICE)
        output = net.compute_diffusion(state, time, external_inputs=None)
        assert output.shape == (batch_size, state_dim, noise_dim)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_gradients_forward_and_compute_diffusion(self):
        state_dim, noise_dim, input_dim = 2, 2, 2
        input_size = state_dim + 1 + input_dim
        output_size = state_dim * noise_dim
        arch = NetworkArchitecture(input_size, [8], output_size)
        net = DiffusionNet(arch, state_dimension=state_dim, noise_dimension=noise_dim)
        batch_size = 4
        state = torch.randn(batch_size, state_dim, requires_grad=True, device=DEVICE)
        time = torch.randn(batch_size, 1, requires_grad=True, device=DEVICE)
        external_inputs = torch.randn(batch_size, input_dim, requires_grad=True, device=DEVICE)
        output = net.compute_diffusion(state, time, external_inputs)
        loss = output.sum()
        loss.backward()
        assert state.grad is not None
        assert time.grad is not None
        assert external_inputs.grad is not None
        assert not torch.any(torch.isnan(state.grad))
        assert not torch.any(torch.isnan(time.grad))
        assert not torch.any(torch.isnan(external_inputs.grad))
        for param in net.parameters():
            assert param.grad is not None
            assert not torch.any(torch.isnan(param.grad))

    def test_diffusion_output_figure(self, tmp_path):
        """
        Visualise the output of DiffusionNet for a grid of 2D states and save the figure.
        """
        state_dim, noise_dim, input_dim = 2, 2, 0
        input_size = state_dim + 1 + input_dim
        output_size = state_dim * noise_dim
        arch = NetworkArchitecture(input_size, [16], output_size)
        net = DiffusionNet(arch, state_dimension=state_dim, noise_dimension=noise_dim)
        net.eval()

        # Create a grid of states in 2D
        x = torch.linspace(-2, 2, 20)
        y = torch.linspace(-2, 2, 20)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        states = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        batch_size = states.shape[0]
        time = torch.zeros(batch_size, 1)
        # Move tensors to the correct device
        device = DEVICE
        states = states.to(device)
        time = time.to(device)
        # No external inputs
        with torch.no_grad():
            diffusion = net.compute_diffusion(states, time, None)  # [N, 2, 2]
        # For each state, plot the first column of the diffusion matrix as a vector
        U = diffusion[:, 0, 0].reshape(xx.shape).cpu().numpy()
        V = diffusion[:, 1, 0].reshape(xx.shape).cpu().numpy()
        # Set up plot
        plotting.set_default_plotting_style(use_tex=False)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.quiver(xx.cpu().numpy(), yy.cpu().numpy(), U, V, color="k")
        # Remove titles and axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        # Make axes square and set only three ticks
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        plotting.style_axis_clean(ax)
        # Save to figures directory as PDF
        outdir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(os.path.join(outdir, "diffusion_net_demonstration.pdf"), bbox_inches="tight")
        plt.close(fig)

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"]) 