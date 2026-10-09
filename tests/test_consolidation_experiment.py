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
    assert result.condition_history["epoch"] == result.history["epoch"]
    for name in ("dirichlet_loss", "neumann_loss", "initial_loss"):
        assert len(result.condition_history[name]) == len(result.history["epoch"])
        assert all(np.isfinite(value) for value in result.condition_history[name])
    assert result.predictions.shape == result.reference_values.shape == (5, 1)
    assert all(np.isfinite(value) for value in result.metrics.values())
    assert result.model.count_parameters() > 0


def test_condition_history_aligns_with_trainer_history_and_objective() -> None:
    result = run_consolidation_experiment(
        ConsolidationExperimentConfig(
            n_interior=4,
            n_spatial_boundary=4,
            n_initial=4,
            n_test=5,
            hidden_layers=1,
            hidden_units=4,
            epochs=3,
            log_every=2,
            seed=8,
        )
    )

    assert result.condition_history["epoch"] == result.history["epoch"] == [1, 2, 3]
    combined_condition = (
        np.asarray(result.condition_history["dirichlet_loss"])
        + np.asarray(result.condition_history["neumann_loss"])
        + np.asarray(result.condition_history["initial_loss"])
    ) / 3.0
    np.testing.assert_allclose(
        combined_condition,
        result.history["bc_loss"],
    )
    np.testing.assert_allclose(
        np.asarray(result.history["total_loss"]),
        np.asarray(result.history["pde_loss"]) + np.asarray(result.history["bc_loss"]),
    )
