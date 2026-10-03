import numpy as np
import pytest

from evaluation.metrics import (
    max_absolute_error,
    mean_squared_error,
    mean_squared_residual,
    relative_l2_error,
)


def test_mean_squared_error_known_example() -> None:
    assert mean_squared_error([1, 2, 3], [2, 2, 5]) == pytest.approx(5 / 3)


def test_mean_squared_error_is_zero_for_identical_arrays() -> None:
    values = np.array([[-1.0, 2.0], [3.5, 4.0]])

    assert mean_squared_error(values, values) == 0.0


def test_relative_l2_error_known_example() -> None:
    assert relative_l2_error([3, 4], [6, 8]) == pytest.approx(1.0)


def test_relative_l2_error_is_zero_for_identical_nonzero_arrays() -> None:
    values = np.array([1.0, -2.0, 4.0])

    assert relative_l2_error(values, values) == 0.0


def test_relative_l2_error_rejects_zero_reference() -> None:
    with pytest.raises(ValueError, match="zero reference"):
        relative_l2_error([0.0, 0.0], [1.0, 2.0])


def test_max_absolute_error_known_example() -> None:
    assert max_absolute_error([1, 2, 3], [0, 4, 2]) == 2.0


def test_max_absolute_error_is_zero_for_identical_arrays() -> None:
    values = np.array([[1.0, 2.0], [3.0, 4.0]])

    assert max_absolute_error(values, values) == 0.0


def test_mean_squared_residual_known_example() -> None:
    assert mean_squared_residual([-2, 1, 3]) == pytest.approx(14 / 3)


def test_metrics_support_multidimensional_arrays() -> None:
    true = np.array([[1.0, 2.0], [3.0, 4.0]])
    pred = np.array([[2.0, 1.0], [5.0, 2.0]])

    assert mean_squared_error(true, pred) == pytest.approx(2.5)
    assert max_absolute_error(true, pred) == 2.0
    assert mean_squared_residual(pred - true) == pytest.approx(2.5)


@pytest.mark.parametrize(
    "metric",
    [mean_squared_error, relative_l2_error, max_absolute_error],
)
def test_prediction_metrics_reject_incompatible_shapes(metric) -> None:
    with pytest.raises(ValueError, match="incompatible shapes"):
        metric(np.zeros((2, 2)), np.zeros(3))
