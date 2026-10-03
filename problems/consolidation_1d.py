"""Physics definition and data sampling for one-dimensional consolidation.

The problem is the nondimensional diffusion equation from Experiment 1 of
Zhang, Chen, and Yang:

    p_t - p_xx = 0,   0 < x < 1, t > 0

with ``p(x, 0) = 1``, ``p(0, t) = 0``, and ``p_x(1, t) = 0``.

This module deliberately contains no model, autodiff, or training code.
Sampling methods return NumPy arrays with columns ordered as ``(x_hat,
t_hat)``.  The finite ``time_max`` used for sampling is a project
implementation choice: the paper specifies ``t_hat > 0`` but does not
specify a finite sampling interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]
RandomSource = Union[int, np.integer, np.random.Generator, None]


@dataclass(frozen=True)
class BoundaryCondition:
    """Metadata consumed by a later loss/autodiff implementation.

    ``derivative_order`` is zero for a value (Dirichlet) condition and one
    for a first spatial derivative (Neumann) condition.  The Neumann
    condition has no target value for the solution itself; its target is the
    derivative value ``0``.
    """

    name: str
    boundary: str
    derivative_order: int
    target: float
    coordinate: float


INITIAL_CONDITION = BoundaryCondition(
    name="initial_condition",
    boundary="t=0",
    derivative_order=0,
    target=1.0,
    coordinate=0.0,
)
DIRICHLET_BOUNDARY = BoundaryCondition(
    name="left_dirichlet",
    boundary="x=0",
    derivative_order=0,
    target=0.0,
    coordinate=0.0,
)
NEUMANN_BOUNDARY = BoundaryCondition(
    name="right_neumann",
    boundary="x=1",
    derivative_order=1,
    target=0.0,
    coordinate=1.0,
)


def _rng(random_state: RandomSource) -> np.random.Generator:
    """Return a generator without changing the global NumPy random state."""

    if isinstance(random_state, np.random.Generator):
        return random_state
    return np.random.default_rng(random_state)


def _validate_count(count: int, name: str) -> int:
    if isinstance(count, bool) or not isinstance(count, (int, np.integer)):
        raise TypeError(f"{name} must be an integer")
    if count < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(count)


def _as_float_array(value: ArrayLike) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


class Consolidation1DProblem:
    """Reusable nondimensional 1D consolidation problem definition.

    Defaults match the point counts reported in the paper: 500 interior
    points, 250 spatial-boundary points, 125 initial points, and 10011 test
    points.  ``time_max=1`` is only a finite sampling interval chosen by this
    project; it is not an additional claim about the paper.
    """

    x_min = 0.0
    x_max = 1.0
    t_min = 0.0

    def __init__(
        self,
        *,
        time_max: float = 1.0,
        n_interior: int = 500,
        n_spatial_boundary: int = 250,
        n_initial: int = 125,
        n_test: int = 10011,
        seed: Optional[int] = None,
    ) -> None:
        if not np.isfinite(time_max) or time_max <= self.t_min:
            raise ValueError("time_max must be finite and greater than zero")
        self.time_max = float(time_max)
        self.n_interior = _validate_count(n_interior, "n_interior")
        self.n_spatial_boundary = _validate_count(
            n_spatial_boundary, "n_spatial_boundary"
        )
        self.n_initial = _validate_count(n_initial, "n_initial")
        self.n_test = _validate_count(n_test, "n_test")
        self.seed = seed

    @property
    def boundary_conditions(self) -> tuple[BoundaryCondition, ...]:
        """Return metadata for the initial and both spatial conditions."""

        return (INITIAL_CONDITION, DIRICHLET_BOUNDARY, NEUMANN_BOUNDARY)

    def sample_interior(
        self,
        count: Optional[int] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample residual points uniformly in ``(0, 1) x (0, time_max)``.

        Returns an array of shape ``(count, 2)`` with columns ``(x_hat,
        t_hat)``.  The default count is ``n_interior``.
        """

        count = self.n_interior if count is None else _validate_count(count, "count")
        generator = _rng(self.seed) if rng is None else rng
        points = generator.uniform(
            (self.x_min, self.t_min),
            (self.x_max, self.time_max),
            size=(count, 2),
        )
        return np.asarray(points, dtype=np.float64)

    def sample_spatial_boundary(
        self,
        count: Optional[int] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample both spatial boundaries for positive times.

        Returns shape ``(count, 2)``.  Points are split as evenly as possible
        between ``x_hat=0`` (Dirichlet) and ``x_hat=1`` (Neumann); if ``count``
        is odd, the extra point is assigned to the left boundary.
        """

        count = (
            self.n_spatial_boundary
            if count is None
            else _validate_count(count, "count")
        )
        generator = _rng(self.seed) if rng is None else rng
        left_count = (count + 1) // 2
        right_count = count - left_count
        # Avoid the initial-condition surface: spatial conditions apply only
        # for t_hat > 0, while the initial sampler owns t_hat == 0.
        positive_time_min = np.nextafter(self.t_min, self.time_max)
        times = generator.uniform(positive_time_min, self.time_max, size=count)
        points = np.empty((count, 2), dtype=np.float64)
        points[:left_count, 0] = self.x_min
        points[:left_count, 1] = times[:left_count]
        points[left_count:, 0] = self.x_max
        points[left_count:, 1] = times[left_count:]
        return points

    def sample_initial(
        self,
        count: Optional[int] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample initial-condition points on ``0 <= x_hat <= 1, t_hat=0``.

        Returns an array of shape ``(count, 2)`` with columns ``(x_hat,
        t_hat)``.  The default count is ``n_initial``.
        """

        count = self.n_initial if count is None else _validate_count(count, "count")
        generator = _rng(self.seed) if rng is None else rng
        points = np.empty((count, 2), dtype=np.float64)
        points[:, 0] = generator.uniform(self.x_min, self.x_max, size=count)
        points[:, 1] = self.t_min
        return points

    def sample_test_points(
        self,
        count: Optional[int] = None,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> FloatArray:
        """Sample reference/evaluation points in ``[0, 1] x [0, time_max]``.

        These points are for evaluation against :func:`reference_solution`,
        not training labels.  Returns shape ``(count, 2)``.
        """

        count = self.n_test if count is None else _validate_count(count, "count")
        generator = _rng(self.seed) if rng is None else rng
        points = generator.uniform(
            (self.x_min, self.t_min),
            (self.x_max, self.time_max),
            size=(count, 2),
        )
        return np.asarray(points, dtype=np.float64)

    # Names matching the generic problem-layer contract.
    sample_collocation = sample_interior
    sample_boundary = sample_spatial_boundary
    sample_observations = sample_initial
    sample_test = sample_test_points

    @staticmethod
    def pde_residual(dp_dt: ArrayLike, d2p_dx2: ArrayLike) -> FloatArray:
        """Delegate to :func:`pde_residual` for object-oriented consumers."""

        return pde_residual(dp_dt, d2p_dx2)

    @staticmethod
    def reference_solution(
        x_hat: ArrayLike, t_hat: ArrayLike, *, terms: int = 100
    ) -> FloatArray:
        """Delegate to :func:`reference_solution` for evaluation consumers."""

        return reference_solution(x_hat, t_hat, terms=terms)


def initial_condition_target(x_hat: ArrayLike) -> FloatArray:
    """Return the initial target ``p_hat(x_hat, 0) = 1``.

    The output has the broadcast shape of ``x_hat`` and is independent of
    ``x_hat``.  This represents training targets, not reference-solution
    values.
    """

    return np.ones_like(_as_float_array(x_hat))


def dirichlet_boundary_target(t_hat: ArrayLike) -> FloatArray:
    """Return the left-boundary target ``p_hat(0, t_hat) = 0``."""

    return np.zeros_like(_as_float_array(t_hat))


def neumann_boundary_condition() -> BoundaryCondition:
    """Return metadata for ``partial p_hat / partial x_hat = 0`` at ``x_hat=1``.

    No derivative is approximated here.  A later autodiff/loss module should
    evaluate the model's first spatial derivative at the returned boundary.
    """

    return NEUMANN_BOUNDARY


def pde_residual(
    dp_dt: ArrayLike,
    d2p_dx2: ArrayLike,
) -> FloatArray:
    """Compute the conceptual PDE residual ``r = p_t - p_xx``.

    ``dp_dt`` and ``d2p_dx2`` are derivatives supplied by a later model or
    autodiff layer.  They must be NumPy-broadcastable; the output has their
    broadcast shape.  No derivative calculation or finite differences occur
    in this function.
    """

    return _as_float_array(dp_dt) - _as_float_array(d2p_dx2)


def reference_solution(
    x_hat: ArrayLike,
    t_hat: ArrayLike,
    *,
    terms: int = 100,
) -> FloatArray:
    """Evaluate the analytical odd-mode series used for reference evaluation.

    Inputs are broadcast according to NumPy rules and the output has the
    broadcast shape of ``x_hat`` and ``t_hat``.  ``terms`` controls the number
    of odd modes, beginning with ``m=1``.  This function is intentionally
    separate from all training-point generation.
    """

    if isinstance(terms, bool) or not isinstance(terms, (int, np.integer)):
        raise TypeError("terms must be an integer")
    if terms < 1:
        raise ValueError("terms must be at least one")
    x, t = np.broadcast_arrays(_as_float_array(x_hat), _as_float_array(t_hat))
    odd_modes = 2 * np.arange(int(terms), dtype=np.float64) + 1.0
    mode = odd_modes.reshape((-1,) + (1,) * x.ndim)
    coefficients = 4.0 / (mode * np.pi)
    sine = np.sin(mode * np.pi * x / 2.0)
    decay = np.exp(-(mode**2) * np.pi**2 * t / 4.0)
    return np.sum(coefficients * sine * decay, axis=0)


__all__ = [
    "BoundaryCondition",
    "Consolidation1DProblem",
    "DIRICHLET_BOUNDARY",
    "INITIAL_CONDITION",
    "NEUMANN_BOUNDARY",
    "dirichlet_boundary_target",
    "initial_condition_target",
    "neumann_boundary_condition",
    "pde_residual",
    "reference_solution",
]