"""Baseline PINN model wrapper.

Wraps the core neural network and allows optional input normalization
and output transformations (e.g., hard boundary condition enforcement).
"""

from typing import Callable, List, Optional, Union
import torch
import torch.nn as nn

from pinn.network import MLP


class BaselinePINN(nn.Module):
    """Baseline Physics-Informed Neural Network.

    Matches the exact multi-layer perceptron architecture referenced in the paper
    with configurable output transformation (such as p_hat = transform(x, NN(x, t))).

    Args:
        input_dim: Dimensionality of inputs (e.g., 2 for 1D consolidation (x, t)).
        output_dim: Dimensionality of outputs (e.g., 1 for pressure/displacement/temp).
        hidden_layers: Number of hidden layers.
        hidden_units: Neurons per hidden layer.
        activation: Non-linear activation function (default: 'tanh').
        initializer: Weight initialization scheme ('glorot_normal', 'glorot_uniform', etc.).
        output_transform: Optional callable (x, u_raw) -> u_transformed enforcing boundary/initial constraints.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        hidden_units: Union[int, List[int]],
        activation: str = "tanh",
        initializer: str = "glorot_normal",
        output_transform: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        super().__init__()
        self.net = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
            initializer=initializer,
        )
        self.output_transform = output_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward evaluation of the PINN.

        Args:
            x: Input coordinate tensor of shape (N, input_dim).

        Returns:
            Physical field prediction of shape (N, output_dim).
        """
        u_raw = self.net(x)
        if self.output_transform is not None:
            return self.output_transform(x, u_raw)
        return u_raw

    def count_parameters(self) -> int:
        """Total trainable parameters in the baseline network."""
        return self.net.count_parameters()
