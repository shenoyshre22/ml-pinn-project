"""Reusable plots for heat PINN validation."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def plot_heat_validation(
    coordinates: np.ndarray,
    predictions: np.ndarray,
    reference: np.ndarray,
    history: Mapping[str, list[float]],
    output_dir: str | Path,
) -> list[Path]:
    """Save loss, field, error, and absolute-error-distribution plots."""

    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "matplotlib is required to generate validation plots"
        ) from error

    coordinates = np.asarray(coordinates)
    predictions = np.asarray(predictions).reshape(-1)
    reference = np.asarray(reference).reshape(-1)
    if coordinates.shape != (predictions.size, 2):
        raise ValueError("coordinates must have shape (count, 2)")
    if reference.shape != predictions.shape:
        raise ValueError("reference and predictions must have matching shapes")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    side = int(np.sqrt(predictions.size))
    if side * side != predictions.size:
        raise ValueError("heat plots require a square test grid")
    x = coordinates[:, 0].reshape(side, side)
    y = coordinates[:, 1].reshape(side, side)
    paths: list[Path] = []

    loss_path = output / "heat_training_losses.png"
    fig, ax = plt.subplots()
    ax.semilogy(history["epoch"], history["total_loss"], label="Total")
    ax.semilogy(history["epoch"], history["pde_loss"], label="PDE")
    ax.semilogy(history["epoch"], history["bc_loss"], label="Boundary")
    ax.set(xlabel="Epoch", ylabel="Loss", title="Heat PINN training losses")
    ax.legend()
    fig.tight_layout()
    fig.savefig(loss_path, dpi=150)
    plt.close(fig)
    paths.append(loss_path)

    fields = [
        ("heat_fem_reference.png", reference, "FEM reference temperature"),
        ("heat_pinn_prediction.png", predictions, "PINN temperature"),
        ("heat_absolute_error.png", np.abs(predictions - reference), "Absolute error"),
    ]
    for filename, values, title in fields:
        fig, ax = plt.subplots()
        image = ax.pcolormesh(x, y, values.reshape(side, side), shading="auto")
        ax.set(xlabel="x", ylabel="y", title=title, aspect="equal")
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        path = output / filename
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(path)

    path = output / "heat_absolute_error_distribution.png"
    fig, ax = plt.subplots()
    ax.hist(np.abs(predictions - reference), bins=40)
    ax.set(xlabel="Absolute error", ylabel="Count", title="Absolute-error distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    paths.append(path)
    return paths