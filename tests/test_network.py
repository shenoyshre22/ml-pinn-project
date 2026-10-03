"""Unit tests for neural network architectures (pinn/network.py and models)."""

import pytest
import torch

from models.baseline_pinn import BaselinePINN
from models.fourier_feature_pinn import FourierFeaturePINN
from pinn.network import MLP


def test_mlp_forward_and_shapes():
    # 2 inputs, 1 output, 4 hidden layers of 50 neurons
    model = MLP(
        input_dim=2,
        output_dim=1,
        hidden_layers=4,
        hidden_units=50,
        activation="tanh",
        initializer="glorot_normal",
    )
    x = torch.randn(32, 2)
    out = model(x)
    assert out.shape == (32, 1)
    assert model.count_parameters() > 0


def test_mlp_initializers():
    for init_name in ["glorot_normal", "glorot_uniform", "he_normal", "he_uniform"]:
        model = MLP(
            input_dim=2,
            output_dim=1,
            hidden_layers=2,
            hidden_units=20,
            activation="tanh",
            initializer=init_name,
        )
        x = torch.randn(10, 2)
        out = model(x)
        assert not torch.isnan(out).any()


def test_baseline_pinn_with_output_transform():
    # Hard boundary condition transform: u(x, t) = x * NN(x, t) ensuring u(0, t) = 0
    transform = lambda x, u: x[:, 0:1] * u
    model = BaselinePINN(
        input_dim=2,
        output_dim=1,
        hidden_layers=3,
        hidden_units=32,
        output_transform=transform,
    )
    x = torch.zeros(10, 2)
    out = model(x)
    assert torch.allclose(out, torch.zeros_like(out))


def test_fourier_feature_pinn():
    model = FourierFeaturePINN(
        input_dim=2,
        output_dim=1,
        num_fourier_features=32,
        fourier_scale=2.0,
        hidden_layers=3,
        hidden_units=32,
        activation="tanh",
    )
    x = torch.randn(16, 2, requires_grad=True)
    out = model(x)
    assert out.shape == (16, 1)

    # Verify that gradients backpropagate smoothly to inputs
    g = torch.autograd.grad(out.sum(), x, create_graph=True)[0]
    assert g.shape == (16, 2)
    assert not torch.isnan(g).any()
