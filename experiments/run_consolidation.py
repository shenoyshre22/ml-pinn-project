"""End-to-end smoke-test experiment for the 1D consolidation problem."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Union

import numpy as np
import torch

from evaluation.metrics import (
    max_absolute_error,
    mean_squared_error,
    relative_l2_error,
)
from models.baseline_pinn import BaselinePINN
from pinn.derivatives import nth_derivative
from pinn.losses import PINNLoss, mse_loss, residual_loss
from pinn.trainer import PINNTrainer
from problems.consolidation_1d import Consolidation1DProblem


@dataclass(frozen=True)
class ConsolidationExperimentConfig:
    """Configurable settings for the first consolidation smoke test.

    The defaults are deliberately small wiring-test values. They are not
    validated final experimental hyperparameters or values claimed by the
    reference paper.
    """

    time_max: float = 1.0
    n_interior: int = 32
    n_spatial_boundary: int = 16
    n_initial: int = 16
    n_test: int = 32
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
    condition_weight: float = 1.0
    reference_terms: int = 100


@dataclass
class ConsolidationExperimentResult:
    """Trained model, histories, and reference-evaluation output."""

    model: BaselinePINN
    history: Dict[str, List[float]]
    condition_history: Dict[str, List[float]]
    metrics: Dict[str, float]
    predictions: np.ndarray = field(repr=False)
    reference_values: np.ndarray = field(repr=False)


def run_consolidation_experiment(
    config: ConsolidationExperimentConfig | None = None,
) -> ConsolidationExperimentResult:
    """Train and evaluate a baseline PINN on 1D consolidation.

    The three physical-condition losses are averaged before being supplied as
    the trainer's single boundary/condition loss. This equal weighting is an
    explicit smoke-test wiring choice, not a paper-mandated loss formulation.
    """

    config = config or ConsolidationExperimentConfig()
    torch.manual_seed(config.seed)

    problem = Consolidation1DProblem(
        time_max=config.time_max,
        n_interior=config.n_interior,
        n_spatial_boundary=config.n_spatial_boundary,
        n_initial=config.n_initial,
        n_test=config.n_test,
        seed=config.seed,
    )

    interior = problem.sample_interior()
    spatial_boundary = problem.sample_spatial_boundary()
    initial = problem.sample_initial()
    test = problem.sample_test_points()

    left_mask = spatial_boundary[:, 0] == problem.x_min
    right_mask = spatial_boundary[:, 0] == problem.x_max
    left_boundary = spatial_boundary[left_mask]
    right_boundary = spatial_boundary[right_mask]

    device = torch.device(config.device) if config.device is not None else None

    interior_tensor = torch.tensor(
        interior, dtype=torch.float32, device=device, requires_grad=True
    )
    left_boundary_tensor = torch.tensor(
        left_boundary, dtype=torch.float32, device=device
    )
    right_boundary_tensor = torch.tensor(
        right_boundary, dtype=torch.float32, device=device, requires_grad=True
    )
    initial_tensor = torch.tensor(initial, dtype=torch.float32, device=device)
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
            w_b=config.condition_weight,
            w_d=0.0,
        ),
        device=device,
    )
    condition_history: Dict[str, List[float]] = {
        "epoch": [],
        "dirichlet_loss": [],
        "neumann_loss": [],
        "initial_loss": [],
    }
    latest_condition_losses: Dict[str, float] = {}

    def closure() -> tuple[torch.Tensor, torch.Tensor, None]:
        interior_prediction = model(interior_tensor)
        dp_dt = nth_derivative(
            interior_prediction, interior_tensor, n=1, component_x=1
        )
        d2p_dx2 = nth_derivative(
            interior_prediction, interior_tensor, n=2, component_x=0
        )
        residual = problem.pde_residual(dp_dt, d2p_dx2)
        loss_f = residual_loss(residual)

        left_prediction = model(left_boundary_tensor)
        right_prediction = model(right_boundary_tensor)
        initial_prediction = model(initial_tensor)
        dp_dx_right = nth_derivative(
            right_prediction, right_boundary_tensor, n=1, component_x=0
        )

        loss_dirichlet = mse_loss(
            left_prediction, torch.zeros_like(left_prediction)
        )
        loss_neumann = mse_loss(dp_dx_right, torch.zeros_like(dp_dx_right))
        loss_initial = mse_loss(
            initial_prediction, torch.ones_like(initial_prediction)
        )
        loss_b = (loss_dirichlet + loss_neumann + loss_initial) / 3.0
        latest_condition_losses.update(
            {
                "dirichlet_loss": float(loss_dirichlet.detach().item()),
                "neumann_loss": float(loss_neumann.detach().item()),
                "initial_loss": float(loss_initial.detach().item()),
            }
        )

        return loss_f, loss_b, None

    def record_condition_losses(epoch: int, _: Dict[str, float]) -> None:
        condition_history["epoch"].append(epoch)
        for name in ("dirichlet_loss", "neumann_loss", "initial_loss"):
            condition_history[name].append(latest_condition_losses[name])

    history = trainer.train_adam(
        closure,
        epochs=config.epochs,
        lr=config.learning_rate,
        log_every=config.log_every,
        callbacks=[record_condition_losses],
    )

    model.eval()
    with torch.no_grad():
        predictions = model(test_tensor).cpu().numpy()

    test_numpy = test_tensor.cpu().numpy()
    reference_values = problem.reference_solution(
        test_numpy[:, 0],
        test_numpy[:, 1],
        terms=config.reference_terms,
    ).reshape(-1, 1)
    metrics = {
        "mean_squared_error": mean_squared_error(reference_values, predictions),
        "relative_l2_error": relative_l2_error(reference_values, predictions),
        "max_absolute_error": max_absolute_error(reference_values, predictions),
    }

    return ConsolidationExperimentResult(
        model=model,
        history=history,
        condition_history=condition_history,
        metrics=metrics,
        predictions=predictions,
        reference_values=reference_values,
    )


if __name__ == "__main__":
    run_consolidation_experiment()