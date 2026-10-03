"""Problem-independent numerical metrics for PINN evaluation."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def mean_squared_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the mean squared difference between reference and predictions.

    Inputs may be array-like, but must have exactly matching shapes.
    Mismatched shapes raise ``ValueError``.
    """

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.mean((pred - true) ** 2))


def relative_l2_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return ``||y_pred - y_true||_2 / ||y_true||_2``.

    Inputs may be array-like, but must have exactly matching shapes.
    Mismatched shapes raise ``ValueError``.  A zero reference norm also
    raises ``ValueError``.
    """

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denominator = np.linalg.norm(true)
    if denominator == 0:
        raise ValueError("relative L2 error is undefined for a zero reference")
    return float(np.linalg.norm(pred - true) / denominator)


def max_absolute_error(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Return the maximum absolute difference between reference and predictions.

    Inputs may be array-like, but must have exactly matching shapes.
    Mismatched shapes raise ``ValueError``.
    """

    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    if true.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    return float(np.max(np.abs(pred - true)))


def mean_squared_residual(residual: ArrayLike) -> float:
    """Return the mean squared value of already-computed PDE residuals."""

    return float(np.mean(np.asarray(residual, dtype=float) ** 2))