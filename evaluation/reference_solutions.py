"""Reference-data generation and loading for the 2D heat problem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from problems.heat_2d import Heat2DProblem


FloatArray = NDArray[np.float64]


def _structured_nodes(subdivisions: int) -> FloatArray:
    coordinates = np.linspace(-1.0, 1.0, subdivisions + 1)
    x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
    return np.column_stack((x.ravel(), y.ravel()))


def _node_index(i: int, j: int, subdivisions: int) -> int:
    return j * (subdivisions + 1) + i


def _assemble_fem_system(
    subdivisions: int,
) -> tuple[FloatArray, Any, FloatArray, NDArray[np.bool_]]:
    nodes = _structured_nodes(subdivisions)
    size = nodes.shape[0]
    stiffness = lil_matrix((size, size), dtype=np.float64)
    load = np.zeros(size, dtype=np.float64)
    triangles: list[tuple[int, int, int]] = []

    for j in range(subdivisions):
        for i in range(subdivisions):
            lower_left = _node_index(i, j, subdivisions)
            lower_right = _node_index(i + 1, j, subdivisions)
            upper_left = _node_index(i, j + 1, subdivisions)
            upper_right = _node_index(i + 1, j + 1, subdivisions)
            triangles.extend(
                [
                    (lower_left, lower_right, upper_right),
                    (lower_left, upper_right, upper_left),
                ]
            )

    for first, second, third in triangles:
        element_nodes = nodes[[first, second, third]]
        area = 0.5 * abs(
            np.linalg.det(
                np.array(
                    [
                        [1.0, element_nodes[0, 0], element_nodes[0, 1]],
                        [1.0, element_nodes[1, 0], element_nodes[1, 1]],
                        [1.0, element_nodes[2, 0], element_nodes[2, 1]],
                    ]
                )
            )
        )
        gradients = np.array(
            [
                [
                    element_nodes[1, 1] - element_nodes[2, 1],
                    element_nodes[2, 0] - element_nodes[1, 0],
                ],
                [
                    element_nodes[2, 1] - element_nodes[0, 1],
                    element_nodes[0, 0] - element_nodes[2, 0],
                ],
                [
                    element_nodes[0, 1] - element_nodes[1, 1],
                    element_nodes[1, 0] - element_nodes[0, 0],
                ],
            ]
        ) / (2.0 * area)
        local_stiffness = area * (gradients @ gradients.T)
        local_load = np.full(3, area / 3.0)
        indices = (first, second, third)
        for local_i, global_i in enumerate(indices):
            load[global_i] += local_load[local_i]
            for local_j, global_j in enumerate(indices):
                stiffness[global_i, global_j] += local_stiffness[local_i, local_j]

    boundary = (
        np.isclose(nodes[:, 0], -1.0)
        | np.isclose(nodes[:, 0], 1.0)
        | np.isclose(nodes[:, 1], -1.0)
        | np.isclose(nodes[:, 1], 1.0)
    )
    return nodes, stiffness.tocsr(), load, boundary


def _evaluate_structured_fem(
    nodes: FloatArray,
    values: FloatArray,
    points: FloatArray,
    subdivisions: int,
) -> FloatArray:
    spacing = 2.0 / subdivisions
    result = np.empty(points.shape[0], dtype=np.float64)
    for index, (x, y) in enumerate(points):
        cell_x = min(subdivisions - 1, max(0, int(np.floor((x + 1.0) / spacing))))
        cell_y = min(subdivisions - 1, max(0, int(np.floor((y + 1.0) / spacing))))
        local_x = (x - (-1.0 + cell_x * spacing)) / spacing
        local_y = (y - (-1.0 + cell_y * spacing)) / spacing
        lower_left = _node_index(cell_x, cell_y, subdivisions)
        lower_right = _node_index(cell_x + 1, cell_y, subdivisions)
        upper_left = _node_index(cell_x, cell_y + 1, subdivisions)
        upper_right = _node_index(cell_x + 1, cell_y + 1, subdivisions)
        if local_x + local_y <= 1.0:
            weights = (1.0 - local_x - local_y, local_x, local_y)
            indices = (lower_left, lower_right, upper_left)
        else:
            weights = (local_x + local_y - 1.0, 1.0 - local_y, 1.0 - local_x)
            indices = (upper_right, lower_right, upper_left)
        result[index] = sum(
            weight * values[node_index]
            for weight, node_index in zip(weights, indices)
        )
    return result


def generate_structured_fem_reference(
    test_coordinates: FloatArray,
    *,
    subdivisions: int = 64,
) -> tuple[FloatArray, float]:
    """Solve ``-Delta T = 1`` with P1 triangles and evaluate at points."""

    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    points = np.asarray(test_coordinates, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("test_coordinates must have shape (count, 2)")
    nodes, stiffness, load, boundary = _assemble_fem_system(subdivisions)
    free = ~boundary
    values = np.zeros(nodes.shape[0], dtype=np.float64)
    values[free] = spsolve(stiffness[free][:, free], load[free])
    residual = stiffness @ values - load
    residual[boundary] = 0.0
    linear_residual = float(np.linalg.norm(residual, ord=np.inf))
    return _evaluate_structured_fem(nodes, values, points, subdivisions), linear_residual


def save_fem_reference(
    path: str | Path,
    coordinates: FloatArray,
    values: FloatArray,
    metadata: dict[str, Any],
) -> None:
    """Save coordinates/values as NPZ and provenance as a sibling JSON file."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    coordinates = np.asarray(coordinates, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    _validate_reference_arrays(coordinates, values)
    np.savez_compressed(target, coordinates=coordinates, values=values)
    target.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def _validate_reference_arrays(
    coordinates: FloatArray, values: FloatArray
) -> None:
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("reference coordinates must have shape (count, 2)")
    if values.shape != (coordinates.shape[0], 1):
        raise ValueError("reference values must have shape (count, 1)")
    if not np.isfinite(coordinates).all() or not np.isfinite(values).all():
        raise ValueError("reference coordinates and values must be finite")


def load_heat_fem_reference(
    path: str | Path,
    *,
    problem: Heat2DProblem | None = None,
    coordinate_tolerance: float = 1e-12,
) -> tuple[FloatArray, FloatArray, dict[str, Any]]:
    """Load and validate an independently generated heat FEM reference."""

    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"reference dataset not found: {target}")
    try:
        with np.load(target) as data:
            coordinates = np.asarray(data["coordinates"], dtype=np.float64)
            values = np.asarray(data["values"], dtype=np.float64)
    except (KeyError, OSError, ValueError) as error:
        raise ValueError(f"invalid reference dataset: {target}") from error
    _validate_reference_arrays(coordinates, values)
    problem = problem or Heat2DProblem(n_test=coordinates.shape[0])
    expected = problem.sample_test_points()
    if expected.shape != coordinates.shape or not np.allclose(
        expected, coordinates, rtol=0.0, atol=coordinate_tolerance
    ):
        raise ValueError("reference coordinates do not match the heat test grid")
    metadata_path = target.with_suffix(".json")
    if not metadata_path.is_file():
        raise ValueError(f"reference metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return coordinates, values, metadata