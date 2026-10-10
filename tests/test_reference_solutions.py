"""Tests for the independently generated heat FEM reference."""

import json

import numpy as np
import pytest

from evaluation.reference_solutions import (
    load_heat_fem_reference,
    save_fem_reference,
)
from problems.heat_2d import Heat2DProblem


def _write_reference(path, coordinates, values) -> None:
    save_fem_reference(
        path,
        coordinates,
        values,
        {"reference_type": "test"},
    )


def test_heat_fem_reference_loads_and_aligns(tmp_path) -> None:
    problem = Heat2DProblem(n_test=4)
    coordinates = problem.sample_test_points()
    values = np.arange(4, dtype=np.float64).reshape(-1, 1)
    path = tmp_path / "reference.npz"
    _write_reference(path, coordinates, values)

    loaded_coordinates, loaded_values, metadata = load_heat_fem_reference(
        path, problem=problem
    )

    np.testing.assert_array_equal(loaded_coordinates, coordinates)
    np.testing.assert_array_equal(loaded_values, values)
    assert metadata["reference_type"] == "test"


def test_heat_fem_reference_rejects_coordinate_mismatch(tmp_path) -> None:
    problem = Heat2DProblem(n_test=4)
    coordinates = problem.sample_test_points().copy()
    coordinates[0, 0] += 1e-3
    path = tmp_path / "reference.npz"
    _write_reference(path, coordinates, np.zeros((4, 1)))

    with pytest.raises(ValueError, match="do not match"):
        load_heat_fem_reference(path, problem=problem)


def test_heat_fem_reference_rejects_nonfinite_values(tmp_path) -> None:
    problem = Heat2DProblem(n_test=4)
    path = tmp_path / "reference.npz"
    with pytest.raises(ValueError, match="finite"):
        _write_reference(
            path,
            problem.sample_test_points(),
            np.array([[0.0], [1.0], [np.nan], [3.0]]),
        )


def test_heat_fem_reference_rejects_malformed_shapes(tmp_path) -> None:
    problem = Heat2DProblem(n_test=4)
    path = tmp_path / "reference.npz"
    np.savez(
        path,
        coordinates=problem.sample_test_points(),
        values=np.zeros((2, 2)),
    )
    path.with_suffix(".json").write_text(
        json.dumps({"reference_type": "test"}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="shape"):
        load_heat_fem_reference(path, problem=problem)
