"""Problem definition and sampling for the steady 2D heat equation.

The problem is

    T_xx + T_yy + 1 = 0

on ``[-1, 1] x [-1, 1]`` with ``T = 0`` on the entire square boundary.
This module contains no neural-network, autodiff, FEM, training, or
evaluation code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
RandomSource = Union[int, np.integer, np.random.Generator, None]


@dataclass(frozen=True)
class DirichletBoundary:
    """Metadata for the zero-temperature boundary condition."""

    name: str = "dirichlet_boundary"
    target: float = 0.0


BOUNDARY_CONDITION = DirichletBoundary()


def _rng(random_state: RandomSource) -> np.random.Generator:
    """Return a generator without changing NumPy's global random state."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_count(count: int, name: str) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(count)


class Heat2DProblem:
    """Reusable definition of the nondimensional steady 2D heat problem.

    The default counts are the paper-specified 1500 interior points, 500
    boundary points, and 10000 test points.  Test points are generated as a
    deterministic ``sqrt(n_test) x sqrt(n_test)`` grid; consequently a
    configurable ``n_test`` must be a perfect square.
    """

    x_min = -1.0
    x_max = 1.0
    y_min = -1.0
    y_max = 1.0

    def __init__(
        self,
        *,
        n_interior: int = 1500,
        n_boundary: int = 500,
        n_test: int = 10000,
        seed: Optional[int] = None,
    ) -> None:
        self.n_interior = _validate_count(n_interior, "n_interior")
        self.n_boundary = _validate_count(n_boundary, "n_boundary")
        self.n_test = _validate_count(n_test, "n_test")
        grid_side = int(np.sqrt(self.n_test))
        if grid_side * grid_side != self.n_test:
            raise ValueError("n_test must be a perfect square")
        self.seed = seed

    @property
    def boundary_condition(self) -> DirichletBoundary:
        """Return metadata for ``T(x, y) = 0`` on the entire boundary."""

        return BOUNDARY_CONDITION

    def sample_interior(
        self,
        count: Optional[int] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample points strictly inside the square for PDE residuals.

        Returns an array of shape ``(count, 2)`` with columns ``(x, y)``.
        The default count is ``n_interior``.
        """

        count = self.n_interior if count is None else _validate_count(count, "count")
        generator = _rng(self.seed) if rng is None else rng
        return np.asarray(
            generator.uniform(
                (self.x_min, self.y_min),
                (self.x_max, self.y_max),
                size=(count, 2),
            ),
            dtype=np.float64,
        )

    def sample_boundary(
        self,
        count: Optional[int] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample random points on the four square boundary sides.

        Returns an array of shape ``(count, 2)`` with columns ``(x, y)``.
        Each point is assigned independently and uniformly to one of the four
        sides, then sampled uniformly along that side.  This equal side
        selection is an implementation choice; the paper specifies only the
        total boundary count.
        """

        count = self.n_boundary if count is None else _validate_count(count, "count")
        generator = _rng(self.seed) if rng is None else rng
        points = generator.uniform(-1.0, 1.0, size=(count, 2))
        sides = generator.integers(0, 4, size=count)
        points[sides == 0, 0] = self.x_min
        points[sides == 1, 0] = self.x_max
        points[sides == 2, 1] = self.y_min
        points[sides == 3, 1] = self.y_max
        return np.asarray(points, dtype=np.float64)

    def sample_test_points(self, count: Optional[int] = None) -> FloatArray:
        """Return a deterministic equispaced square grid for future evaluation.

        The default is a ``100 x 100`` grid, represented as an array of shape
        ``(10000, 2)``.  These points are evaluation points, not training
        points and do not carry generated labels.
        """

        count = self.n_test if count is None else _validate_count(count, "count")
        grid_side = int(np.sqrt(count))
        if grid_side * grid_side != count:
            raise ValueError("count must be a perfect square")
        coordinates = np.linspace(self.x_min, self.x_max, grid_side)
        x, y = np.meshgrid(coordinates, coordinates, indexing="xy")
        return np.column_stack((x.ravel(), y.ravel()))

    sample_collocation = sample_interior
    sample_test = sample_test_points

    def pde_residual(self, T_xx: ArrayLike, T_yy: ArrayLike) -> FloatArray:
        """Return the residual ``T_xx + T_yy + 1`` from supplied derivatives.

        ``T_xx`` and ``T_yy`` must have exactly matching shapes because they
        represent derivatives evaluated at the same collocation points. Their
        derivatives are supplied by the future PINN/autodiff engine; this
        method performs no differentiation itself.
        """

        return pde_residual(T_xx, T_yy)


def boundary_target(points: ArrayLike) -> FloatArray:
    """Return zero targets for boundary points.

    ``points`` is expected to have shape ``(count, 2)``; the returned array
    has shape ``(count,)``.  The coordinates are not revalidated here.
    """

    points_array = np.asarray(points)
    if points_array.ndim != 2 or points_array.shape[1] != 2:
        raise ValueError("points must have shape (count, 2)")
    return np.zeros(points_array.shape[0], dtype=np.float64)


def pde_residual(T_xx: ArrayLike, T_yy: ArrayLike) -> FloatArray:
    """Return ``T_xx + T_yy + 1`` from same-shaped computed derivatives.

    Raises ``ValueError`` when the derivative arrays do not have exactly
    matching shapes.
    """

    T_xx_array = np.asarray(T_xx, dtype=np.float64)
    T_yy_array = np.asarray(T_yy, dtype=np.float64)
    if T_xx_array.shape != T_yy_array.shape:
        raise ValueError("T_xx and T_yy must have the same shape")
    return T_xx_array + T_yy_array + 1.0


__all__ = [
    "BOUNDARY_CONDITION",
    "DirichletBoundary",
    "Heat2DProblem",
    "boundary_target",
    "pde_residual",
]