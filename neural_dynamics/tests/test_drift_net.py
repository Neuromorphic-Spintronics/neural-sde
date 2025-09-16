"""
Unit tests for the DriftNet implementation in the neural SDE framework.
"""

from __future__ import annotations
import torch
import pytest
import os
from neural_dynamics.core import utils as plotting
import matplotlib.pyplot as plt

from neural_dynamics.core.hyperparameters import NetworkArchitecture
from neural_dynamics.models import DriftNet
from config import DEVICE


class TestDriftNet:
    @pytest.mark.parametrize(
        "arch,activation",
        [
            (NetworkArchitecture(4, [64, 32], 2), torch.nn.Tanh),
            (NetworkArchitecture(3, [32], 2), torch.nn.ReLU),
            (NetworkArchitecture(5, [16, 8], 3), torch.nn.Sigmoid),
        ],
    )
    def test_initialisation_and_device(self, arch, activation):
        net = DriftNet(arch, activation=activation)
        assert net.architecture == arch
        assert isinstance(net, DriftNet)
        assert next(net.parameters()).device.type == DEVICE.type

    @pytest.mark.parametrize("batch_size", [1, 16, 512])
    def test_forward_pass_various_batch_sizes(self, batch_size):
        arch = NetworkArchitecture(4, [32, 16], 2)
        net = DriftNet(arch)
        input_tensor = torch.randn(batch_size, 4).to(DEVICE)
        output = net(input_tensor)
        assert output.shape == (batch_size, 2)
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_gradients_forward_and_compute_drift(self):
        arch = NetworkArchitecture(3, [16, 8], 2)
        net = DriftNet(arch)

        # Forward pass gradients (ensure that the network is differentiable and therefore trainable)
        input_tensor = torch.randn(5, 3, requires_grad=True, device=DEVICE)
        output = net(input_tensor)
        loss = output.sum()
        loss.backward()
        assert input_tensor.grad is not None
        assert not torch.any(torch.isnan(input_tensor.grad))
        for param in net.parameters():
            assert param.grad is not None
            assert not torch.any(torch.isnan(param.grad))

        # compute_drift gradients (ensures the drift network logic is differentiable)
        state = torch.randn(3, 2, requires_grad=True, device=DEVICE)
        time = torch.randn(3, 1, requires_grad=True, device=DEVICE)
        drift = net.compute_drift(state, time, external_inputs=None)
        loss2 = drift.sum()
        loss2.backward()
        assert state.grad is not None
        assert time.grad is not None
        assert not torch.any(torch.isnan(state.grad))
        assert not torch.any(torch.isnan(time.grad))

    def test_drift_output_figure(self):
        """
        Visualise the output of DriftNet for a grid of 2D states and save the figure as a PDF.
        """
        state_dim, input_dim = 2, 0
        input_size = state_dim + 1 + input_dim
        output_size = state_dim
        arch = NetworkArchitecture(input_size, [16], output_size)
        net = DriftNet(arch)
        net.eval()

        # Create a grid of states in 2D
        x = torch.linspace(-2, 2, 20)
        y = torch.linspace(-2, 2, 20)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        states = torch.stack([xx.flatten(), yy.flatten()], dim=1)
        batch_size = states.shape[0]
        time = torch.zeros(batch_size, 1)
        # No external inputs
        device = DEVICE
        states = states.to(device)
        time = time.to(device)
        with torch.no_grad():
            drift = net.compute_drift(states, time, None)  # [N, 2]
        U = drift[:, 0].reshape(xx.shape).cpu().numpy()
        V = drift[:, 1].reshape(xx.shape).cpu().numpy()
        # Set up plot
        plotting.set_default_plotting_style(use_tex=False)
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.quiver(xx.cpu().numpy(), yy.cpu().numpy(), U, V, color="k")
        # Remove titles and axis labels
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_title("")
        # Make axes square and set only three ticks
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([-2, 0, 2])
        ax.set_yticks([-2, 0, 2])
        plotting.style_axis_clean(ax)
        # Save to figures directory as PDF
        outdir = os.path.join(os.path.dirname(__file__), "figures")
        os.makedirs(outdir, exist_ok=True)
        fig.savefig(
            os.path.join(outdir, "drift_net_demonstration.pdf"), bbox_inches="tight"
        )
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
