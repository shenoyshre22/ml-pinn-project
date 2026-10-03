"""Unit tests for automatic differentiation module (pinn/derivatives.py)."""

import math
import pytest
import torch

from pinn.derivatives import grad, hessian, jacobian, laplacian, nth_derivative


def test_grad_polynomial():
    # y = x_0^3 + 2 * x_1^2
    # dy/dx_0 = 3 * x_0^2, dy/dx_1 = 4 * x_1
    x = torch.tensor([[2.0, 3.0], [1.0, -2.0]], requires_grad=True)
    y = x[:, 0:1] ** 3 + 2.0 * (x[:, 1:2] ** 2)

    g = grad(y, x)
    assert g.shape == (2, 2)

    expected_0 = 3.0 * (x[:, 0:1] ** 2)
    expected_1 = 4.0 * x[:, 1:2]

    assert torch.allclose(g[:, 0:1], expected_0, atol=1e-6)
    assert torch.allclose(g[:, 1:2], expected_1, atol=1e-6)


def test_nth_derivative():
    # y = sin(x)
    # y' = cos(x), y'' = -sin(x), y''' = -cos(x), y'''' = sin(x)
    x = torch.linspace(0.0, 2.0 * math.pi, 20).unsqueeze(-1).requires_grad_(True)
    y = torch.sin(x)

    d1 = nth_derivative(y, x, n=1)
    d2 = nth_derivative(y, x, n=2)
    d3 = nth_derivative(y, x, n=3)
    d4 = nth_derivative(y, x, n=4)

    assert torch.allclose(d1, torch.cos(x), atol=1e-5)
    assert torch.allclose(d2, -torch.sin(x), atol=1e-5)
    assert torch.allclose(d3, -torch.cos(x), atol=1e-5)
    assert torch.allclose(d4, torch.sin(x), atol=1e-5)


def test_jacobian():
    # y_0 = x_0 * x_1
    # y_1 = x_0^2 + x_1
    x = torch.tensor([[2.0, 5.0]], requires_grad=True)
    y0 = x[:, 0:1] * x[:, 1:2]
    y1 = x[:, 0:1] ** 2 + x[:, 1:2]
    y = torch.cat([y0, y1], dim=-1)

    jac = jacobian(y, x)
    assert jac.shape == (1, 2, 2)

    # dy_0/dx_0 = 5, dy_0/dx_1 = 2
    # dy_1/dx_0 = 4, dy_1/dx_1 = 1
    expected = torch.tensor([[[5.0, 2.0], [4.0, 1.0]]])
    assert torch.allclose(jac, expected, atol=1e-6)


def test_laplacian():
    # T(x, y) = x^3 + y^3
    # d^2 T / dx^2 = 6x, d^2 T / dy^2 = 6y
    # Laplacian = 6(x + y)
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
    T = x[:, 0:1] ** 3 + x[:, 1:2] ** 3

    lap = laplacian(T, x)
    assert lap.shape == (2, 1)

    expected = 6.0 * (x[:, 0:1] + x[:, 1:2])
    assert torch.allclose(lap, expected, atol=1e-6)
