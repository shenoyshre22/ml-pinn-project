"""End-to-end smoke-test experiment for the 1D consolidation problem."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import argparse
import copy
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Union

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
from experiments.config_utils import load_dataclass_config


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
    output_transform_x: bool = False
    lbfgs_enabled: bool = False
    lbfgs_max_iter: int = 50000
    lbfgs_log_every: int = 100
    output_path: str | None = None
    checkpoint_dir: str | None = None
    checkpoint_every: int = 100
    resume_from: str | None = None


@dataclass
class ConsolidationExperimentResult:
    """Trained model, histories, and reference-evaluation output."""

    model: BaselinePINN
    history: Dict[str, List[float]]
    condition_history: Dict[str, List[float]]
    metrics: Dict[str, float]
    predictions: np.ndarray = field(repr=False)
    reference_values: np.ndarray = field(repr=False)
    test_coordinates: np.ndarray = field(repr=False)
    stage_histories: Dict[str, Dict[str, List[float]]] = field(default_factory=dict)


def run_consolidation_experiment(
    config: ConsolidationExperimentConfig | None = None,
) -> ConsolidationExperimentResult:
    """Train and evaluate a baseline PINN on 1D consolidation.

    The three physical-condition losses are averaged before being supplied as
    the trainer's single boundary/condition loss. This equal weighting is an
    explicit smoke-test wiring choice, not a paper-mandated loss formulation.
    """

    config = config or ConsolidationExperimentConfig()
    if config.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
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

    output_transform = (
        (lambda coordinates, values: coordinates[:, 0:1] * values)
        if config.output_transform_x
        else None
    )
    model = BaselinePINN(
        input_dim=2,
        output_dim=1,
        hidden_layers=config.hidden_layers,
        hidden_units=config.hidden_units,
        activation=config.activation,
        initializer=config.initializer,
        output_transform=output_transform,
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
    resume_state: Dict[str, Any] | None = None
    if config.resume_from is not None:
        resume_state = _load_consolidation_checkpoint(
            config.resume_from, model, trainer, condition_history, config
        )
        _restore_rng_state(resume_state.get("rng_state"))

    def save_checkpoint(
        stage: str,
        completed_epoch: int,
        optimizer: torch.optim.Optimizer | None,
        lbfgs_iterations: int = 0,
    ) -> None:
        if config.checkpoint_dir is None:
            return
        state = {
            "checkpoint_version": 1,
            "experiment": "consolidation",
            "stage": stage,
            "completed_epoch": completed_epoch,
            "lbfgs_iterations": lbfgs_iterations,
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": (
                copy.deepcopy(optimizer.state_dict()) if optimizer is not None else None
            ),
            "history": copy.deepcopy(trainer.history),
            "condition_history": copy.deepcopy(condition_history),
            "configuration": vars(config).copy(),
            "rng_state": _capture_rng_state(),
        }
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        filename = (
            f"consolidation_{stage}_epoch_{completed_epoch:08d}"
            f"_lbfgs_{lbfgs_iterations:08d}.pt"
        )
        _atomic_torch_save(state, checkpoint_dir / filename)
        _atomic_torch_save(state, checkpoint_dir / "consolidation_latest.pt")

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

    loaded_stage = resume_state.get("stage") if resume_state else None
    completed_adam = (
        int(resume_state.get("completed_epoch", 0))
        if loaded_stage == "adam"
        else config.epochs if loaded_stage in ("lbfgs", "complete") else 0
    )
    if completed_adam < config.epochs:
        history = trainer.train_adam(
            closure,
            epochs=config.epochs,
            lr=config.learning_rate,
            log_every=config.log_every,
            callbacks=[record_condition_losses],
            start_epoch=completed_adam,
            optimizer_state=(
                resume_state.get("optimizer_state_dict")
                if loaded_stage == "adam"
                else None
            ),
            step_callback=lambda epoch, _, optimizer: (
                save_checkpoint("adam", epoch, optimizer)
                if epoch % config.checkpoint_every == 0 or epoch == config.epochs
                else None
            ),
        )
    else:
        history = trainer.history
    adam_history_length = sum(
        epoch <= config.epochs for epoch in history["epoch"]
    )
    stage_histories = {
        "adam": {
            key: list(values[:adam_history_length])
            for key, values in history.items()
        }
    }
    if config.lbfgs_enabled:
        lbfgs_iterations = (
            int(resume_state.get("lbfgs_iterations", 0))
            if loaded_stage == "lbfgs"
            else 0
        )
        lbfgs_optimizer_state = (
            resume_state.get("optimizer_state_dict")
            if loaded_stage == "lbfgs"
            else None
        )
        if loaded_stage != "lbfgs":
            save_checkpoint("lbfgs", config.epochs, None, lbfgs_iterations)
        while lbfgs_iterations < config.lbfgs_max_iter:
            chunk = min(config.checkpoint_every, config.lbfgs_max_iter - lbfgs_iterations)
            observed_step = [config.epochs + lbfgs_iterations]
            history = trainer.train_lbfgs(
                closure,
                max_iter=chunk,
                log_every=config.lbfgs_log_every,
                callbacks=[record_condition_losses],
                optimizer_state=lbfgs_optimizer_state,
                step_callback=lambda step, _, optimizer: (
                    observed_step.__setitem__(0, step),
                    save_checkpoint("lbfgs", config.epochs, optimizer, step - config.epochs),
                ),
            )
            lbfgs_iterations = max(lbfgs_iterations + chunk, observed_step[0] - config.epochs)
            lbfgs_optimizer_state = None
        stage_histories["lbfgs"] = {
            key: list(values[adam_history_length:])
            for key, values in history.items()
        }
    if config.checkpoint_dir is not None:
        save_checkpoint(
            "complete",
            history["epoch"][-1] if history["epoch"] else 0,
            None,
            config.lbfgs_max_iter if config.lbfgs_enabled else 0,
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
        test_coordinates=test,
        stage_histories=stage_histories,
    )


def load_consolidation_config(path: str | Path) -> ConsolidationExperimentConfig:
    config = load_dataclass_config(
        path,
        ConsolidationExperimentConfig,
        required={
            "time_max", "n_interior", "n_spatial_boundary", "n_initial",
            "n_test", "seed", "device", "hidden_layers", "hidden_units",
            "activation", "initializer", "epochs", "learning_rate", "log_every",
            "pde_weight", "condition_weight", "reference_terms",
            "output_transform_x", "lbfgs_enabled", "lbfgs_max_iter",
            "lbfgs_log_every", "output_path",
        },
    )
    if config.time_max <= 0 or config.epochs < 1 or config.learning_rate <= 0:
        raise ValueError("time_max must be positive, epochs at least 1, and learning_rate positive")
    if min(config.n_interior, config.n_spatial_boundary, config.n_initial, config.n_test) < 1:
        raise ValueError("all consolidation point counts must be positive")
    if min(config.pde_weight, config.condition_weight) < 0:
        raise ValueError("consolidation loss weights must be non-negative")
    if config.checkpoint_every < 1:
        raise ValueError("checkpoint_every must be at least 1")
    return config


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    return state


def _restore_rng_state(state: Dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_torch_save(state: Dict[str, Any], target: Path) -> None:
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(target)


def _load_consolidation_checkpoint(
    path: str,
    model: BaselinePINN,
    trainer: PINNTrainer,
    condition_history: Dict[str, List[float]],
    config: ConsolidationExperimentConfig,
) -> Dict[str, Any]:
    checkpoint = torch.load(
        path, map_location=trainer.device, weights_only=False
    )
    if checkpoint.get("experiment") != "consolidation":
        raise ValueError("checkpoint is not for the consolidation experiment")
    saved_config = checkpoint.get("configuration", {})
    ignored = {"checkpoint_dir", "resume_from", "output_path", "epochs"}
    current_config = vars(config)
    mismatches = [
        key
        for key in current_config
        if key not in ignored and saved_config.get(key) != current_config[key]
    ]
    if mismatches:
        raise ValueError(
            "checkpoint configuration does not match the current run: "
            + ", ".join(sorted(mismatches))
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    trainer.history = checkpoint.get("history", trainer.history)
    condition_history.update(checkpoint.get("condition_history", {}))
    return checkpoint


def save_consolidation_result(
    result: ConsolidationExperimentResult,
    config: ConsolidationExperimentConfig,
    runtime_seconds: float,
) -> None:
    if config.output_path is None:
        raise ValueError("output_path is required when saving consolidation results")
    target = Path(config.output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.with_suffix(".npz").exists():
        raise FileExistsError(f"refusing to overwrite existing output: {target}")
    configuration = vars(config).copy()
    configuration["device"] = str(config.device)
    payload = {
        "experiment": "consolidation",
        "configuration": configuration,
        "runtime_seconds": runtime_seconds,
        "actual_epochs_completed": result.history["epoch"][-1],
        "training_stages": {
            "adam_epochs": config.epochs,
            "lbfgs_enabled": config.lbfgs_enabled,
            "lbfgs_max_iter": config.lbfgs_max_iter if config.lbfgs_enabled else 0,
        },
        "metrics": result.metrics,
        "final_losses": {
            key: result.history[key][-1]
            for key in ("total_loss", "pde_loss", "bc_loss", "data_loss")
        },
        "finite_values": bool(
            np.isfinite(result.predictions).all()
            and all(np.isfinite(values).all() for values in result.history.values())
        ),
        "output_npz": str(target.with_suffix(".npz")),
        "checkpoint_directory": config.checkpoint_dir,
    }
    np.savez_compressed(
        target.with_suffix(".npz"),
        predictions=result.predictions,
        reference_values=result.reference_values,
        test_coordinates=result.test_coordinates,
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
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-every", type=int)
    parser.add_argument("--resume-from")
    args = parser.parse_args()
    loaded_config = load_consolidation_config(args.config)
    overrides = {}
    if args.checkpoint_dir is not None:
        overrides["checkpoint_dir"] = args.checkpoint_dir
    if args.checkpoint_every is not None:
        overrides["checkpoint_every"] = args.checkpoint_every
    if args.resume_from is not None:
        overrides["resume_from"] = args.resume_from
    if overrides:
        loaded_config = replace(loaded_config, **overrides)
    started = time.perf_counter()
    loaded_result = run_consolidation_experiment(loaded_config)
    save_consolidation_result(
        loaded_result, loaded_config, time.perf_counter() - started
    )