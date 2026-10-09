"""Focused integration tests for the 1D consolidation experiment."""

import numpy as np

from experiments.run_consolidation import (
    ConsolidationExperimentConfig,
    run_consolidation_experiment,
)


def test_consolidation_smoke_experiment_trains_and_evaluates() -> None:
    result = run_consolidation_experiment(
        ConsolidationExperimentConfig(
            n_interior=4,
            n_spatial_boundary=4,
            n_initial=4,
            n_test=5,
            hidden_layers=1,
            hidden_units=4,
            epochs=1,
            log_every=1,
            seed=7,
        )
    )

    assert result.history["epoch"] == [1]
    assert result.predictions.shape == result.reference_values.shape == (5, 1)
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert result.model.count_parameters() > 0
