"""Inverse identification problem for a damped single-degree oscillator."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from numpy.typing import ArrayLike, NDArray


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class OscillatorParameters:
    """Physical oscillator parameters used by the analytical solution."""

    mass: float = 1.0
    damping: float = 0.4
    stiffness: float = 4.0
    initial_displacement: float = 1.0
    initial_velocity: float = 1.0


def oscillator_solution(
    time: ArrayLike,
    parameters: OscillatorParameters = OscillatorParameters(),
) -> FloatArray:
    """Evaluate the underdamped analytical displacement solution."""

    if parameters.mass <= 0 or parameters.stiffness <= 0:
        raise ValueError("mass and stiffness must be positive")
    time_array = np.asarray(time, dtype=np.float64)
    discriminant = (
        parameters.stiffness / parameters.mass
        - (parameters.damping / (2.0 * parameters.mass)) ** 2
    )
    if discriminant <= 0:
        raise ValueError("oscillator_solution requires an underdamped oscillator")
    decay = parameters.damping / (2.0 * parameters.mass)
    frequency = np.sqrt(discriminant)
    sine_coefficient = (
        parameters.initial_velocity + decay * parameters.initial_displacement
    ) / frequency
    return np.exp(-decay * time_array) * (
        parameters.initial_displacement * np.cos(frequency * time_array)
        + sine_coefficient * np.sin(frequency * time_array)
    )


class StructuralInverseProblem:
    """Sampling and residual definitions for inverse oscillator PINNs.

    The paper specifies point counts but not an observation horizon. This
    implementation uses ``time_max=10`` as an explicit project choice.
    """

    def __init__(
        self,
        *,
        time_max: float = 10.0,
        n_observations: int = 50,
        n_residual: int = 100,
        n_initial: int = 50,
        n_test: int = 10000,
        seed: Optional[int] = None,
        true_parameters: OscillatorParameters = OscillatorParameters(),
    ) -> None:
        if not np.isfinite(time_max) or time_max <= 0:
            raise ValueError("time_max must be finite and positive")
        self.time_max = float(time_max)
        self.n_observations = n_observations
        self.n_residual = n_residual
        self.n_initial = n_initial
        self.n_test = n_test
        self.seed = seed
        self.true_parameters = true_parameters

    def _generator(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def sample_residual_points(self) -> FloatArray:
        return self._generator().uniform(0.0, self.time_max, self.n_residual).reshape(-1, 1)

    def sample_initial_points(self) -> FloatArray:
        # The initial conditions are imposed at t=0; repeated points provide
        # the requested count without changing the physical constraint.
        return np.zeros((self.n_initial, 1), dtype=np.float64)

    def sample_observations(self) -> tuple[FloatArray, FloatArray]:
        times = np.linspace(0.0, self.time_max, self.n_observations)
        values = oscillator_solution(times, self.true_parameters)
        return times.reshape(-1, 1), values.reshape(-1, 1)

    def sample_test_points(self) -> FloatArray:
        return np.linspace(0.0, self.time_max, self.n_test).reshape(-1, 1)

    @staticmethod
    def pde_residual(
        displacement_second_derivative: ArrayLike,
        displacement_first_derivative: ArrayLike,
        displacement: ArrayLike,
        damping: ArrayLike,
        stiffness: ArrayLike,
        mass: float = 1.0,
    ) -> FloatArray | torch.Tensor:
        """Return ``m*u'' + c*u' + k*u``."""

        if any(
            isinstance(value, torch.Tensor)
            for value in (
                displacement_second_derivative,
                displacement_first_derivative,
                displacement,
                damping,
                stiffness,
            )
        ):
            return (
                mass * displacement_second_derivative
                + damping * displacement_first_derivative
                + stiffness * displacement
            )
        return (
            mass * np.asarray(displacement_second_derivative)
            + np.asarray(damping) * np.asarray(displacement_first_derivative)
            + np.asarray(stiffness) * np.asarray(displacement)
        )
