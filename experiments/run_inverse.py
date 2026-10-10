"""Train and evaluate the inverse damped-oscillator PINN."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
import torch

from models.baseline_pinn import BaselinePINN
from pinn.derivatives import nth_derivative
from pinn.losses import PINNLoss, mse_loss, residual_loss
from pinn.trainer import PINNTrainer
from problems.structural_inverse import (
    OscillatorParameters,
    StructuralInverseProblem,
)


@dataclass(frozen=True)
class InverseExperimentConfig:
    time_max: float = 10.0
    n_observations: int = 50
    n_residual: int = 100
    n_initial: int = 50
    n_test: int = 10000
    seed: int = 2026
    device: Union[str, torch.device, None] = "cpu"
    hidden_layers: int = 3
    hidden_units: int = 32
    activation: str = "tanh"
    initializer: str = "glorot_uniform"
    epochs: int = 100000
    learning_rate: float = 0.0005
    log_every: int = 1000
    pde_weight: float = 0.25
    condition_weight: float = 0.25
    data_weight: float = 0.25
    initial_damping: float = 0.2
    initial_stiffness: float = 3.0
    true_mass: float = 1.0
    true_damping: float = 0.4
    true_stiffness: float = 4.0
    initial_displacement: float = 1.0
    initial_velocity: float = 1.0


@dataclass
class InverseExperimentResult:
    model: BaselinePINN
    history: Dict[str, List[float]]
    learned_damping: float
    learned_stiffness: float
    observation_times: np.ndarray
    observations: np.ndarray
    test_times: np.ndarray
    predictions: np.ndarray


def run_inverse_experiment(
    config: InverseExperimentConfig | None = None,
) -> InverseExperimentResult:
    config = config or InverseExperimentConfig()
    torch.manual_seed(config.seed)
    problem = StructuralInverseProblem(
        time_max=config.time_max,
        n_observations=config.n_observations,
        n_residual=config.n_residual,
        n_initial=config.n_initial,
        n_test=config.n_test,
        seed=config.seed,
        true_parameters=OscillatorParameters(
            mass=config.true_mass,
            damping=config.true_damping,
            stiffness=config.true_stiffness,
            initial_displacement=config.initial_displacement,
            initial_velocity=config.initial_velocity,
        ),
    )
    residual_points = torch.tensor(
        problem.sample_residual_points(), dtype=torch.float32, requires_grad=True
    )
    initial_points = torch.tensor(
        problem.sample_initial_points(), dtype=torch.float32, requires_grad=True
    )
    observation_times, observations = problem.sample_observations()
    observation_tensor = torch.tensor(
        observation_times, dtype=torch.float32, requires_grad=True
    )
    test_times = problem.sample_test_points()
    test_tensor = torch.tensor(test_times, dtype=torch.float32)

    model = BaselinePINN(
        input_dim=1,
        output_dim=1,
        hidden_layers=config.hidden_layers,
        hidden_units=config.hidden_units,
        activation=config.activation,
        initializer=config.initializer,
    )
    damping = torch.nn.Parameter(torch.tensor(config.initial_damping))
    stiffness = torch.nn.Parameter(torch.tensor(config.initial_stiffness))
    trainer = PINNTrainer(
        model=model,
        loss_fn=PINNLoss(
            w_f=config.pde_weight,
            w_b=config.condition_weight,
            w_d=config.data_weight,
        ),
        device=config.device,
        extra_parameters=[damping, stiffness],
    )

    def closure() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual_prediction = model(residual_points)
        first_derivative = nth_derivative(residual_prediction, residual_points)
        second_derivative = nth_derivative(residual_prediction, residual_points, n=2)
        residual = (
            config.true_mass * second_derivative
            + damping * first_derivative
            + stiffness * residual_prediction
        )
        loss_f = residual_loss(residual)

        initial_prediction = model(initial_points)
        initial_velocity_prediction = nth_derivative(model(initial_points), initial_points)
        loss_b = 0.5 * (
            mse_loss(
                initial_prediction,
                torch.full_like(initial_prediction, config.initial_displacement),
            )
            + mse_loss(
                initial_velocity_prediction,
                torch.full_like(
                    initial_velocity_prediction, config.initial_velocity
                ),
            )
        )
        data_prediction = model(observation_tensor)
        loss_d = mse_loss(
            data_prediction,
            torch.tensor(observations, dtype=torch.float32),
        )
        return loss_f, loss_b, loss_d

    history = trainer.train_adam(
        closure,
        epochs=config.epochs,
        lr=config.learning_rate,
        log_every=config.log_every,
    )
    model.eval()
    with torch.no_grad():
        predictions = model(test_tensor).cpu().numpy()
    return InverseExperimentResult(
        model=model,
        history=history,
        learned_damping=float(damping.detach().cpu().item()),
        learned_stiffness=float(stiffness.detach().cpu().item()),
        observation_times=observation_times,
        observations=observations,
        test_times=test_times,
        predictions=predictions,
    )


if __name__ == "__main__":
    run_inverse_experiment()