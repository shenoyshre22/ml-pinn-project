"""Focused integration tests for the 2D heat experiment."""

import numpy as np

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
