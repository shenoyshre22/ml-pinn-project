"""End-to-end baseline experiment for the steady 2D heat equation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import torch

from models.baseline_pinn import BaselinePINN
from pinn.derivatives import nth_derivative
from pinn.losses import PINNLoss, mse_loss, residual_loss
from pinn.trainer import PINNTrainer
from problems.heat_2d import Heat2DProblem
from evaluation.metrics import (
    max_absolute_error,
    mean_squared_error,
    relative_l2_error,
)
from evaluation.reference_solutions import load_heat_fem_reference


@dataclass(frozen=True)
class HeatExperimentConfig:
    """Configurable settings for the 2D heat-equation experiment."""

    n_interior: int = 32
    n_boundary: int = 16
    n_test: int = 16
    seed: int = 0
    device: Union[str, torch.device, None] = "cpu"
    hidden_layers: int = 2
    hidden_units: int = 16
    activation: str = "tanh"
    initializer: str = "glorot_normal"
    epochs: int = 5
    learning_rate: float = 0.001
    log_every: int = 5
    pde_weight: float = 1.0
    boundary_weight: float = 1.0
    reference_path: Optional[str] = None


@dataclass
class HeatExperimentResult:
    """Trained heat model, predictions, and reference-free diagnostics."""

    model: BaselinePINN
    history: Dict[str, List[float]]
    test_coordinates: np.ndarray = field(repr=False)
    predictions: np.ndarray = field(repr=False)
    diagnostics: Dict[str, float]
    reference_metrics: Optional[Dict[str, float]] = None


def _test_residual_statistics(
    model: BaselinePINN,
    test_tensor: torch.Tensor,
    problem: Heat2DProblem,
) -> Dict[str, float]:
    """Evaluate finite residual statistics on the deterministic test grid."""

    test_tensor = test_tensor.detach().clone().requires_grad_(True)
    test_prediction = model(test_tensor)
    T_xx = nth_derivative(test_prediction, test_tensor, n=2, component_x=0)
    T_yy = nth_derivative(test_prediction, test_tensor, n=2, component_x=1)
    residual = problem.pde_residual(T_xx, T_yy).detach()
    absolute_residual = residual.abs()
    return {
        "test_residual_mean_absolute": float(absolute_residual.mean().item()),
        "test_residual_max_absolute": float(absolute_residual.max().item()),
    }


def run_heat_experiment(
    config: HeatExperimentConfig | None = None,
) -> HeatExperimentResult:
    """Train and evaluate a baseline PINN without reference-data metrics."""

    config = config or HeatExperimentConfig()
    torch.manual_seed(config.seed)

    problem = Heat2DProblem(
        n_interior=config.n_interior,
        n_boundary=config.n_boundary,
        n_test=config.n_test,
        seed=config.seed,
    )
    interior = problem.sample_interior()
    boundary = problem.sample_boundary()
    test = problem.sample_test_points()

    device = torch.device(config.device) if config.device is not None else None
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    interior_tensor = torch.tensor(
        interior, dtype=torch.float32, device=device, requires_grad=True
    )
    boundary_tensor = torch.tensor(
        boundary, dtype=torch.float32, device=device
    )
    test_tensor = torch.tensor(test, dtype=torch.float32, device=device)

    model = BaselinePINN(
        input_dim=2,
        output_dim=1,
        hidden_layers=config.hidden_layers,
        hidden_units=config.hidden_units,
        activation=config.activation,
        initializer=config.initializer,
    )
    trainer = PINNTrainer(
        model=model,
        loss_fn=PINNLoss(
            w_f=config.pde_weight,
            w_b=config.boundary_weight,
            w_d=0.0,
        ),
        device=device,
    )

    def closure() -> tuple[torch.Tensor, torch.Tensor, None]:
        interior_prediction = model(interior_tensor)
        T_xx = nth_derivative(
            interior_prediction, interior_tensor, n=2, component_x=0
        )
        T_yy = nth_derivative(
            interior_prediction, interior_tensor, n=2, component_x=1
        )
        residual = problem.pde_residual(T_xx, T_yy)
        loss_f = residual_loss(residual)

        boundary_prediction = model(boundary_tensor)
        loss_b = mse_loss(
            boundary_prediction, torch.zeros_like(boundary_prediction)
        )
        return loss_f, loss_b, None

    history = trainer.train_adam(
        closure,
        epochs=config.epochs,
        lr=config.learning_rate,
        log_every=config.log_every,
    )

    model.eval()
    with torch.no_grad():
        predictions = model(test_tensor).cpu().numpy()

    diagnostics = {
        "final_pde_loss": history["pde_loss"][-1],
        "final_boundary_loss": history["bc_loss"][-1],
    }
    diagnostics.update(_test_residual_statistics(model, test_tensor, problem))
    reference_metrics = None
    if config.reference_path is not None:
        reference_coordinates, reference_values, _ = load_heat_fem_reference(
            config.reference_path, problem=problem
        )
        if not np.array_equal(reference_coordinates, test):
            raise ValueError("reference coordinates do not match experiment test points")
        reference_metrics = {
            "mean_squared_error": mean_squared_error(reference_values, predictions),
            "relative_l2_error": relative_l2_error(reference_values, predictions),
            "max_absolute_error": max_absolute_error(reference_values, predictions),
        }

    return HeatExperimentResult(
        model=model,
        history=history,
        test_coordinates=test,
        predictions=predictions,
        diagnostics=diagnostics,
        reference_metrics=reference_metrics,
    )


if __name__ == "__main__":
    run_heat_experiment()