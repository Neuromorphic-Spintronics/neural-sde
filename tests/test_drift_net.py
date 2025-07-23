"""
Unit tests for the DriftNet implementation in the neural SDE framework.
"""

from __future__ import annotations
import torch
import pytest
import sys

sys.path.append(".")
from parameters.hyperparameters import NetworkArchitecture
from models.neural_sde import DriftNet
from config import DEVICE


class TestDriftNet:
    @pytest.mark.parametrize("arch,activation", [
        (NetworkArchitecture(4, [64, 32], 2), torch.nn.Tanh),
        (NetworkArchitecture(3, [32], 2), torch.nn.ReLU),
        (NetworkArchitecture(5, [16, 8], 3), torch.nn.Sigmoid),
    ])
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

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
