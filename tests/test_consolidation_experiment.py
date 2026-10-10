"""Focused integration tests for the 1D consolidation experiment."""

from pathlib import Path

import numpy as np
import torch

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


def test_consolidation_checkpoint_is_atomic_and_contains_training_state(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    run_consolidation_experiment(
        ConsolidationExperimentConfig(
            n_interior=4,
            n_spatial_boundary=4,
            n_initial=4,
            n_test=5,
            hidden_layers=1,
            hidden_units=4,
            epochs=2,
            log_every=1,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every=1,
            seed=12,
        )
    )

    latest = checkpoint_dir / "consolidation_latest.pt"
    assert latest.is_file()
    assert not list(checkpoint_dir.glob("*.tmp"))
    checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
    assert checkpoint["stage"] == "complete"
    assert checkpoint["completed_epoch"] == 2
    assert checkpoint["optimizer_state_dict"] is None
    assert checkpoint["history"]["epoch"] == [1, 2]
    assert "rng_state" in checkpoint


def test_consolidation_resume_continues_adam_without_repeating_epochs(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    first = run_consolidation_experiment(
        ConsolidationExperimentConfig(
            n_interior=4,
            n_spatial_boundary=4,
            n_initial=4,
            n_test=5,
            hidden_layers=1,
            hidden_units=4,
            epochs=3,
            log_every=1,
            checkpoint_dir=str(checkpoint_dir),
            checkpoint_every=1,
            seed=13,
        )
    )
    adam_checkpoint = checkpoint_dir / (
        "consolidation_adam_epoch_00000002_lbfgs_00000000.pt"
    )
    assert adam_checkpoint.is_file()

    resumed = run_consolidation_experiment(
        ConsolidationExperimentConfig(
            n_interior=4,
            n_spatial_boundary=4,
            n_initial=4,
            n_test=5,
            hidden_layers=1,
            hidden_units=4,
            epochs=3,
            log_every=1,
            checkpoint_dir=str(tmp_path / "resumed_checkpoints"),
            checkpoint_every=1,
            resume_from=str(adam_checkpoint),
            seed=13,
        )
    )

    assert resumed.history["epoch"] == [1, 2, 3]
    assert first.history["epoch"] == [1, 2, 3]
    assert resumed.stage_histories["adam"]["epoch"] == [1, 2, 3]


def test_consolidation_lbfgs_checkpoint_restores_optimizer_state(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    config = ConsolidationExperimentConfig(
        n_interior=4,
        n_spatial_boundary=4,
        n_initial=4,
        n_test=5,
        hidden_layers=1,
        hidden_units=4,
        epochs=1,
        log_every=1,
        lbfgs_enabled=True,
        lbfgs_max_iter=2,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=1,
        seed=14,
    )
    run_consolidation_experiment(config)
    lbfgs_checkpoints = sorted(checkpoint_dir.glob("consolidation_lbfgs_*.pt"))
    assert len(lbfgs_checkpoints) >= 2
    checkpoint = next(
        torch.load(path, map_location="cpu", weights_only=False)
        for path in reversed(lbfgs_checkpoints)
        if torch.load(path, map_location="cpu", weights_only=False)[
            "optimizer_state_dict"
        ]
        is not None
    )
    assert checkpoint["stage"] == "lbfgs"
    assert checkpoint["optimizer_state_dict"] is not None
    assert checkpoint["lbfgs_iterations"] >= 1

    resumed = run_consolidation_experiment(
        ConsolidationExperimentConfig(
            **{
                **config.__dict__,
                "checkpoint_dir": str(tmp_path / "resumed"),
                "resume_from": str(
                    next(
                        path
                        for path in reversed(lbfgs_checkpoints)
                        if torch.load(
                            path, map_location="cpu", weights_only=False
                        )["optimizer_state_dict"]
                        is not None
                    )
                ),
            }
        )
    )
    assert resumed.stage_histories["lbfgs"]["epoch"]
