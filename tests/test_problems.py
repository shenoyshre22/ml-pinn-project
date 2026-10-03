import numpy as np
import pytest

from problems.consolidation_1d import (
    Consolidation1DProblem,
    dirichlet_boundary_target,
    initial_condition_target,
    neumann_boundary_condition,
    pde_residual,
    reference_solution,
)


def test_default_point_counts() -> None:
    problem = Consolidation1DProblem(seed=1)

    assert problem.sample_interior().shape == (500, 2)
    assert problem.sample_spatial_boundary().shape == (250, 2)
    assert problem.sample_initial().shape == (125, 2)
    assert problem.sample_test_points().shape == (10011, 2)


def test_interior_sampling_stays_inside_open_domain() -> None:
    problem = Consolidation1DProblem(time_max=2.5, seed=2)
    points = problem.sample_interior(count=37)

    assert points.shape == (37, 2)
    assert np.all((points[:, 0] > 0) & (points[:, 0] < 1))
    assert np.all((points[:, 1] > 0) & (points[:, 1] < problem.time_max))


def test_spatial_boundary_sampling_has_positive_times_and_both_boundaries() -> None:
    problem = Consolidation1DProblem(seed=3)
    points = problem.sample_spatial_boundary()

    assert points.shape == (250, 2)
    assert set(np.unique(points[:, 0])) == {0.0, 1.0}
    assert np.all(points[:, 1] > 0)
    assert np.all(points[:, 1] <= problem.time_max)


def test_initial_sampling_is_at_t_zero_and_has_initial_target() -> None:
    problem = Consolidation1DProblem(seed=4)
    points = problem.sample_initial(count=41)

    assert points.shape == (41, 2)
    assert np.all(points[:, 1] == 0)
    assert np.all((points[:, 0] >= 0) & (points[:, 0] <= 1))
    assert np.array_equal(initial_condition_target(points[:, 0]), np.ones(41))


def test_dirichlet_boundary_target_is_zero() -> None:
    times = np.array([0.1, 0.5, 1.0])

    assert np.array_equal(dirichlet_boundary_target(times), np.zeros(3))


def test_neumann_boundary_metadata() -> None:
    condition = neumann_boundary_condition()

    assert condition.coordinate == 1.0
    assert condition.derivative_order == 1
    assert condition.target == 0.0


def test_pde_residual_subtracts_spatial_second_derivative() -> None:
    dp_dt = np.array([[1.0, 2.0], [3.0, 4.0]])
    d2p_dx2 = np.array([[0.25, 0.5], [0.75, 1.0]])
    residual = pde_residual(dp_dt, d2p_dx2)

    assert residual.shape == dp_dt.shape
    np.testing.assert_array_equal(residual, dp_dt - d2p_dx2)


def test_reference_solution_broadcasts_and_is_finite() -> None:
    x_hat = np.array([[0.0], [0.5]])
    t_hat = np.array([[0.1, 0.5, 1.0]])

    values = reference_solution(x_hat, t_hat)

    assert values.shape == (2, 3)
    assert np.all(np.isfinite(values))
    # This is an evaluation check only; sampled training points are not
    # generated from or labeled with the reference solution.
    np.testing.assert_array_equal(values[0], np.zeros(3))


def test_reference_solution_satisfies_neumann_boundary_numerically() -> None:
    h = 1e-5
    t_hat = np.array([0.1, 0.5, 1.0])

    derivative = (
        reference_solution(1.0, t_hat)
        - reference_solution(1.0 - h, t_hat)
    ) / h

    np.testing.assert_allclose(derivative, 0.0, atol=1e-5, rtol=0.0)


def test_reproducible_sampling_with_same_seed() -> None:
    first = Consolidation1DProblem(seed=5)
    second = Consolidation1DProblem(seed=5)

    np.testing.assert_array_equal(first.sample_interior(), second.sample_interior())
    np.testing.assert_array_equal(
        first.sample_spatial_boundary(), second.sample_spatial_boundary()
    )
    np.testing.assert_array_equal(first.sample_initial(), second.sample_initial())
    np.testing.assert_array_equal(
        first.sample_test_points(), second.sample_test_points()
    )


def test_different_seeds_produce_different_samples() -> None:
    first = Consolidation1DProblem(seed=6)
    second = Consolidation1DProblem(seed=7)

    assert not np.array_equal(first.sample_interior(), second.sample_interior())


def test_custom_point_counts_are_respected() -> None:
    problem = Consolidation1DProblem(
        n_interior=3,
        n_spatial_boundary=4,
        n_initial=5,
        n_test=6,
    )

    assert problem.sample_interior().shape == (3, 2)
    assert problem.sample_spatial_boundary().shape == (4, 2)
    assert problem.sample_initial().shape == (5, 2)
    assert problem.sample_test_points().shape == (6, 2)


@pytest.mark.parametrize("count", [-1, 1.5, True])
def test_invalid_constructor_point_counts_are_rejected(count: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Consolidation1DProblem(n_interior=count)  # type: ignore[arg-type]


@pytest.mark.parametrize("time_max", [0, -1, np.inf, np.nan])
def test_invalid_time_max_is_rejected(time_max: float) -> None:
    with pytest.raises(ValueError):
        Consolidation1DProblem(time_max=time_max)


@pytest.mark.parametrize("terms", [0, -1, 1.5, True])
def test_invalid_reference_solution_terms_are_rejected(terms: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        reference_solution(0.5, 0.1, terms=terms)  # type: ignore[arg-type]