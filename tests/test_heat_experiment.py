"""Focused integration tests for the 2D heat experiment."""

import numpy as np
from pathlib import Path

from experiments.run_heat import HeatExperimentConfig, run_heat_experiment


def test_heat_smoke_experiment_trains_without_reference_metrics() -> None:
    result = run_heat_experiment(
        HeatExperimentConfig(
            n_interior=4,
            n_boundary=4,
            n_test=4,
            hidden_layers=1,
            hidden_units=4,
            epochs=2,
            log_every=1,
            seed=7,
        )
    )

    assert result.history["epoch"] == [1, 2]
    assert result.test_coordinates.shape == (4, 2)
    assert result.predictions.shape == (4, 1)
    assert result.reference_metrics is None
    assert all(np.isfinite(result.predictions))
    for values in result.history.values():
        assert all(np.isfinite(value) for value in values)
    assert all(np.isfinite(value) for value in result.diagnostics.values())
    assert result.model.count_parameters() > 0


def test_heat_experiment_reports_reference_metrics(tmp_path: Path) -> None:
    from evaluation.reference_solutions import (
        generate_structured_fem_reference,
        save_fem_reference,
    )
    from problems.heat_2d import Heat2DProblem

    reference_path = tmp_path / "reference.npz"
    problem = Heat2DProblem(n_interior=4, n_boundary=4, n_test=9, seed=7)
    coordinates = problem.sample_test_points()
    values, _ = generate_structured_fem_reference(coordinates, subdivisions=4)
    save_fem_reference(
        reference_path,
        coordinates,
        values,
        {"reference_type": "test"},
    )

    result = run_heat_experiment(
        HeatExperimentConfig(
            n_interior=4,
            n_boundary=4,
            n_test=9,
            hidden_layers=1,
            hidden_units=4,
            epochs=1,
            log_every=1,
            seed=7,
            reference_path=str(reference_path),
        )
    )

    assert result.reference_metrics is not None
    assert all(np.isfinite(value) for value in result.reference_metrics.values())
