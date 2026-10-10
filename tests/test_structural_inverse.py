"""Focused tests for the inverse damped-oscillator experiment."""

import numpy as np
import pytest
import torch

from experiments.run_inverse import InverseExperimentConfig, run_inverse_experiment
from pinn.losses import PINNLoss
from problems.structural_inverse import (
    OscillatorParameters,
    StructuralInverseProblem,
    oscillator_solution,
)


def test_oscillator_solution_matches_initial_conditions() -> None:
    parameters = OscillatorParameters()
    h = 1e-6
    values = oscillator_solution(np.array([0.0, h]), parameters)
    velocity = (values[1] - values[0]) / h
    assert values[0] == 1.0
    assert velocity == pytest.approx(1.0, abs=1e-5)


def test_oscillator_equation_is_satisfied() -> None:
    parameters = OscillatorParameters()
    times = np.linspace(0.1, 2.0, 20)
    h = 1e-5
    displacement = oscillator_solution(times, parameters)
    first = (
        oscillator_solution(times + h, parameters)
        - oscillator_solution(times - h, parameters)
    ) / (2.0 * h)
    second = (
        oscillator_solution(times + h, parameters)
        - 2.0 * displacement
        + oscillator_solution(times - h, parameters)
    ) / h**2
    residual = StructuralInverseProblem.pde_residual(
        second, first, displacement, parameters.damping, parameters.stiffness
    )
    np.testing.assert_allclose(residual, 0.0, atol=1e-5)


def test_data_generation_is_reproducible_and_has_paper_counts() -> None:
    first = StructuralInverseProblem(seed=2026)
    second = StructuralInverseProblem(seed=2026)
    first_times, first_values = first.sample_observations()
    second_times, second_values = second.sample_observations()
    assert first.sample_residual_points().shape == (100, 1)
    assert first.sample_initial_points().shape == (50, 1)
    assert first.sample_test_points().shape == (10000, 1)
    np.testing.assert_array_equal(first_times, second_times)
    np.testing.assert_array_equal(first_values, second_values)
    np.testing.assert_array_equal(
        first.sample_residual_points(), second.sample_residual_points()
    )


def test_inverse_loss_weights_match_paper() -> None:
    loss = PINNLoss(w_f=0.25, w_b=0.25, w_d=0.25)
    assert (loss.w_f, loss.w_b, loss.w_d) == (0.25, 0.25, 0.25)


def test_inverse_parameters_are_trainable_and_losses_finite() -> None:
    result = run_inverse_experiment(
        InverseExperimentConfig(
            n_observations=5,
            n_residual=6,
            n_initial=4,
            n_test=10,
            epochs=2,
            log_every=1,
            seed=7,
        )
    )
    assert np.isfinite(result.learned_damping)
    assert np.isfinite(result.learned_stiffness)
    assert result.learned_damping != 0.4 or result.learned_stiffness != 4.0
    assert result.predictions.shape == (10, 1)
    assert all(
        np.isfinite(value)
        for values in result.history.values()
        for value in values
    )


def test_torch_oscillator_residual_preserves_parameter_gradients() -> None:
    damping = torch.nn.Parameter(torch.tensor(0.2))
    stiffness = torch.nn.Parameter(torch.tensor(3.0))
    displacement = torch.ones(2, 1)
    first = torch.ones(2, 1)
    second = torch.ones(2, 1)
    residual = StructuralInverseProblem.pde_residual(
        second, first, displacement, damping, stiffness
    )
    residual.mean().backward()
    assert damping.grad is not None
    assert stiffness.grad is not None
