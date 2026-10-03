import numpy as np
import pytest
import torch

from problems.consolidation_1d import (
    Consolidation1DProblem,
    dirichlet_boundary_target,
    initial_condition_target,
    neumann_boundary_condition,
    pde_residual,
    reference_solution,
)
from problems.heat_2d import (
    Heat2DProblem,
    boundary_target,
    pde_residual as heat_pde_residual,
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


def test_consolidation_pde_residual_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError, match="dp_dt and d2p_dx2 must have the same shape"):
        pde_residual(np.zeros(3), np.zeros(2))


def test_consolidation_pde_residual_supports_torch_tensors() -> None:
    dp_dt = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    d2p_dx2 = torch.tensor([[0.25, 0.5], [0.75, 1.0]])

    residual = pde_residual(dp_dt, d2p_dx2)

    assert isinstance(residual, torch.Tensor)
    torch.testing.assert_close(residual, dp_dt - d2p_dx2)


def test_consolidation_pde_residual_preserves_torch_autograd() -> None:
    values = torch.tensor([[1.0, 2.0]], requires_grad=True)
    dp_dt = 2.0 * values
    d2p_dx2 = 0.5 * values

    residual = pde_residual(dp_dt, d2p_dx2)
    residual.sum().backward()

    assert residual.requires_grad
    torch.testing.assert_close(values.grad, torch.full_like(values, 1.5))


def test_consolidation_pde_residual_rejects_mismatched_torch_shapes() -> None:
    with pytest.raises(ValueError, match="dp_dt and d2p_dx2 must have the same shape"):
        pde_residual(torch.zeros(3), torch.zeros(2))


@pytest.mark.parametrize(
    ("dp_dt", "d2p_dx2"),
    [
        (torch.zeros(2), np.zeros(2)),
        (np.zeros(2), torch.zeros(2)),
    ],
)
def test_consolidation_pde_residual_rejects_mixed_backends(
    dp_dt: object, d2p_dx2: object
) -> None:
    with pytest.raises(TypeError, match="dp_dt and d2p_dx2 must use the same backend"):
        pde_residual(dp_dt, d2p_dx2)  # type: ignore[arg-type]


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

    np.testing.assert_allclose(derivative, 0.0, atol=1e-4, rtol=0.0)


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


def test_heat_default_point_counts() -> None:
    problem = Heat2DProblem()

    assert problem.sample_interior().shape == (1500, 2)
    assert problem.sample_boundary().shape == (500, 2)
    assert problem.sample_test_points().shape == (10000, 2)


def test_heat_interior_points_stay_strictly_inside_domain() -> None:
    points = Heat2DProblem().sample_interior(count=37)

    assert points.shape == (37, 2)
    assert np.all((points > -1) & (points < 1))


def test_heat_boundary_points_lie_on_square_boundary() -> None:
    points = Heat2DProblem(seed=10).sample_boundary()

    assert points.shape == (500, 2)
    on_boundary = (
        (points[:, 0] == -1)
        | (points[:, 0] == 1)
        | (points[:, 1] == -1)
        | (points[:, 1] == 1)
    )
    assert np.all(on_boundary)


def test_heat_boundary_target_is_zero() -> None:
    points = Heat2DProblem(seed=11).sample_boundary()
    targets = boundary_target(points)

    assert targets.shape == (500,)
    assert np.all(targets == 0)


def test_heat_boundary_condition_metadata() -> None:
    condition = Heat2DProblem().boundary_condition

    assert condition.name == "dirichlet_boundary"
    assert condition.target == 0.0


def test_heat_generic_sampling_aliases_are_available() -> None:
    problem = Heat2DProblem(seed=12)

    assert hasattr(problem, "sample_collocation")
    assert hasattr(problem, "sample_test")
    np.testing.assert_array_equal(
        problem.sample_collocation(), problem.sample_interior()
    )
    np.testing.assert_array_equal(
        problem.sample_test(), problem.sample_test_points()
    )


def test_heat_pde_residual_computation() -> None:
    T_xx = np.array([1.0, 2.0, 3.0])
    T_yy = np.array([4.0, 5.0, 6.0])
    expected = np.array([6.0, 8.0, 10.0])

    np.testing.assert_array_equal(heat_pde_residual(T_xx, T_yy), expected)
    np.testing.assert_array_equal(
        Heat2DProblem().pde_residual(T_xx, T_yy), expected
    )


def test_heat_pde_residual_rejects_mismatched_shapes() -> None:
    with pytest.raises(ValueError):
        heat_pde_residual(np.zeros(3), np.zeros(2))


def test_heat_test_grid_size_and_domain() -> None:
    points = Heat2DProblem().sample_test_points()

    assert points.shape == (10000, 2)
    assert np.all((points >= -1) & (points <= 1))


def test_heat_test_grid_is_100_by_100_equispaced() -> None:
    points = Heat2DProblem().sample_test_points()
    x_coordinates = np.unique(points[:, 0])
    y_coordinates = np.unique(points[:, 1])

    assert x_coordinates.size == 100
    assert y_coordinates.size == 100
    np.testing.assert_allclose(np.diff(x_coordinates), np.diff(x_coordinates)[0])
    np.testing.assert_allclose(np.diff(y_coordinates), np.diff(y_coordinates)[0])


def test_heat_random_sampling_is_reproducible_with_same_seed() -> None:
    first = Heat2DProblem(seed=42)
    second = Heat2DProblem(seed=42)

    np.testing.assert_array_equal(first.sample_interior(), second.sample_interior())
    np.testing.assert_array_equal(first.sample_boundary(), second.sample_boundary())


def test_heat_different_seeds_produce_different_interior_samples() -> None:
    first = Heat2DProblem(seed=42)
    second = Heat2DProblem(seed=43)

    assert not np.array_equal(first.sample_interior(), second.sample_interior())


def test_heat_custom_point_counts_are_respected() -> None:
    problem = Heat2DProblem(n_interior=3, n_boundary=4, n_test=100)

    assert problem.sample_interior().shape == (3, 2)
    assert problem.sample_boundary().shape == (4, 2)
    assert problem.sample_test_points().shape == (100, 2)


@pytest.mark.parametrize("parameter", ["n_interior", "n_boundary", "n_test"])
@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_heat_invalid_point_counts_are_rejected(parameter: str, value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        Heat2DProblem(**{parameter: value})  # type: ignore[arg-type]


def test_heat_non_perfect_square_test_count_is_rejected() -> None:
    with pytest.raises(ValueError):
        Heat2DProblem(n_test=101)

    assert Heat2DProblem(n_test=10000).n_test == 10000


def test_heat_boundary_target_rejects_invalid_point_shape() -> None:
    with pytest.raises(ValueError):
        boundary_target(np.zeros(5))