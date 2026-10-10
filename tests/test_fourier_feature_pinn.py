"""Tests for the Fourier Feature PINN and heat-pipeline integration."""

import numpy as np
import torch

from experiments.run_heat import HeatExperimentConfig, run_heat_experiment
from models.fourier_feature_pinn import FourierFeatureMapping, FourierFeaturePINN
from pinn.derivatives import nth_derivative
from problems.heat_2d import Heat2DProblem


def test_fourier_mapping_dimensions() -> None:
    mapping = FourierFeatureMapping(input_dim=2, num_features=7)
    features = mapping(torch.zeros(5, 2))
    assert features.shape == (5, 14)


def test_fourier_frequency_initialization_is_seeded() -> None:
    first = FourierFeatureMapping(input_dim=2, num_features=8, seed=2026)
    second = FourierFeatureMapping(input_dim=2, num_features=8, seed=2026)
    third = FourierFeatureMapping(input_dim=2, num_features=8, seed=2027)
    assert torch.equal(first.B, second.B)
    assert not torch.equal(first.B, third.B)


def test_fourier_model_output_and_input_gradients_are_finite() -> None:
    model = FourierFeaturePINN(
        input_dim=2,
        output_dim=1,
        num_fourier_features=8,
        hidden_layers=2,
        hidden_units=8,
        fourier_seed=2026,
    )
    coordinates = torch.randn(6, 2, requires_grad=True)
    prediction = model(coordinates)
    gradient = torch.autograd.grad(
        prediction.sum(), coordinates, create_graph=True
    )[0]
    assert prediction.shape == (6, 1)
    assert gradient.shape == coordinates.shape
    assert torch.isfinite(prediction).all()
    assert torch.isfinite(gradient).all()


def test_fourier_model_integrates_with_heat_pde_pipeline() -> None:
    config = HeatExperimentConfig(
        model="fourier",
        num_fourier_features=8,
        fourier_scale=1.0,
        n_interior=4,
        n_boundary=4,
        n_test=4,
        hidden_layers=1,
        hidden_units=8,
        epochs=2,
        log_every=1,
        seed=2026,
    )
    result = run_heat_experiment(config)
    problem = Heat2DProblem(n_interior=4, n_boundary=4, n_test=4, seed=2026)
    coordinates = torch.tensor(
        problem.sample_interior(), dtype=torch.float32, requires_grad=True
    )
    prediction = result.model(coordinates)
    residual = problem.pde_residual(
        nth_derivative(prediction, coordinates, n=2, component_x=0),
        nth_derivative(prediction, coordinates, n=2, component_x=1),
    )
    assert result.predictions.shape == (4, 1)
    assert torch.isfinite(residual).all()
    assert all(np.isfinite(value) for values in result.history.values() for value in values)
