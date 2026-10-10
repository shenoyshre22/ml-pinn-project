"""End-to-end baseline experiment for the steady 2D heat equation."""

from __future__ import annotations

from dataclasses import dataclass, field
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch
import torch.nn as nn

from models.baseline_pinn import BaselinePINN
from models.fourier_feature_pinn import FourierFeaturePINN
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
from experiments.config_utils import load_dataclass_config


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
    model: str = "baseline"
    num_fourier_features: int = 64
    fourier_scale: float = 1.0
    trainable_fourier: bool = False
    lbfgs_enabled: bool = False
    lbfgs_max_iter: int = 50000
    lbfgs_log_every: int = 100
    output_path: str | None = None


@dataclass
class HeatExperimentResult:
    """Trained heat model, predictions, and reference-free diagnostics."""

    model: nn.Module
    history: Dict[str, List[float]]
    test_coordinates: np.ndarray = field(repr=False)
    predictions: np.ndarray = field(repr=False)
    diagnostics: Dict[str, float]
    reference_metrics: Optional[Dict[str, float]] = None
    stage_histories: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


def _test_residual_statistics(
    model: nn.Module,
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

    model_name = config.model.lower().strip()
    if model_name == "baseline":
        model: nn.Module = BaselinePINN(
            input_dim=2,
            output_dim=1,
            hidden_layers=config.hidden_layers,
            hidden_units=config.hidden_units,
            activation=config.activation,
            initializer=config.initializer,
        )
    elif model_name == "fourier":
        model = FourierFeaturePINN(
            input_dim=2,
            output_dim=1,
            num_fourier_features=config.num_fourier_features,
            fourier_scale=config.fourier_scale,
            hidden_layers=config.hidden_layers,
            hidden_units=config.hidden_units,
            activation=config.activation,
            initializer=config.initializer,
            trainable_fourier=config.trainable_fourier,
            fourier_seed=config.seed,
        )
    else:
        raise ValueError(
            f"Unsupported heat model '{config.model}'. "
            "Expected 'baseline' or 'fourier'."
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
    stage_histories = {
        "adam": {key: list(values) for key, values in history.items()}
    }
    if config.lbfgs_enabled:
        adam_history_length = len(history["epoch"])
        history = trainer.train_lbfgs(
            closure,
            max_iter=config.lbfgs_max_iter,
            log_every=config.lbfgs_log_every,
        )
        stage_histories["lbfgs"] = {
            key: list(values[adam_history_length:])
            for key, values in history.items()
        }

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
        stage_histories=stage_histories,
    )


def load_heat_config(path: str | Path) -> HeatExperimentConfig:
    config = load_dataclass_config(
        path,
        HeatExperimentConfig,
        required={
            "n_interior", "n_boundary", "n_test", "seed", "device",
            "hidden_layers", "hidden_units", "activation", "initializer",
            "epochs", "learning_rate", "log_every", "pde_weight",
            "boundary_weight", "reference_path", "model",
            "num_fourier_features", "fourier_scale", "trainable_fourier",
            "lbfgs_enabled", "lbfgs_max_iter", "lbfgs_log_every", "output_path",
        },
    )
    if config.epochs < 1 or config.learning_rate <= 0:
        raise ValueError("epochs must be at least 1 and learning_rate must be positive")
    if min(config.n_interior, config.n_boundary, config.n_test) < 1:
        raise ValueError("all heat point counts must be positive")
    if min(config.pde_weight, config.boundary_weight) < 0:
        raise ValueError("heat loss weights must be non-negative")
    if config.model.lower().strip() not in {"baseline", "fourier"}:
        raise ValueError("heat model must be 'baseline' or 'fourier'")
    return config


def save_heat_result(
    result: HeatExperimentResult,
    config: HeatExperimentConfig,
    runtime_seconds: float,
) -> None:
    if config.output_path is None:
        raise ValueError("output_path is required when saving heat results")
    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix(".npz").exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    configuration = vars(config).copy()
    configuration["device"] = str(config.device)
    reference_arrays: dict[str, np.ndarray] = {}
    if config.reference_path is not None:
        reference_coordinates, reference_values, _ = load_heat_fem_reference(
            config.reference_path,
            problem=None,
        )
        reference_arrays = {
            "reference_coordinates": reference_coordinates,
            "reference_values": reference_values,
        }
    payload = {
        "experiment": config.model,
        "configuration": configuration,
        "runtime_seconds": runtime_seconds,
        "actual_epochs_completed": result.history["epoch"][-1],
        "training_stages": {
            "adam_epochs": config.epochs,
            "lbfgs_enabled": config.lbfgs_enabled,
            "lbfgs_max_iter": config.lbfgs_max_iter if config.lbfgs_enabled else 0,
        },
        "diagnostics": result.diagnostics,
        "reference_metrics": result.reference_metrics,
        "final_losses": {
            key: result.history[key][-1]
            for key in ("total_loss", "pde_loss", "bc_loss", "data_loss")
        },
        "finite_values": bool(
            np.isfinite(result.predictions).all()
            and all(np.isfinite(values).all() for values in result.history.values())
        ),
        "output_npz": str(target.with_suffix(".npz")),
    }
    np.savez_compressed(
        target.with_suffix(".npz"),
        test_coordinates=result.test_coordinates,
        predictions=result.predictions,
        **reference_arrays,
        **{key: np.asarray(values) for key, values in result.history.items()},
        **{
            f"{stage}_{key}": np.asarray(values)
            for stage, stage_history in result.stage_histories.items()
            for key, values in stage_history.items()
        },
    )
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    loaded_config = load_heat_config(args.config)
    started = time.perf_counter()
    loaded_result = run_heat_experiment(loaded_config)
    save_heat_result(loaded_result, loaded_config, time.perf_counter() - started)