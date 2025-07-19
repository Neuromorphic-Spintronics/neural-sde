import torch
from models.integrators import rk2, rk4


def constant_derivative(t: float, y: torch.Tensor) -> torch.Tensor:
    return torch.tensor(1.0)


def test_rk2_final_time():
    y = rk2(constant_derivative, torch.tensor(0.0), 0.0, 1.0, 0.1)
    assert torch.isclose(y, torch.tensor(1.0), atol=1e-5)


def test_rk4_final_time():
    y = rk4(constant_derivative, torch.tensor(0.0), 0.0, 1.0, 0.1)
    assert torch.isclose(y, torch.tensor(1.0), atol=1e-5)
