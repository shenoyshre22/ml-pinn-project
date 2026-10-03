"""Fourier Feature PINN Extension Model.

Implements random Fourier feature mappings (Tancik et al., 2020) projecting
low-dimensional physical coordinates into a higher-dimensional Fourier space:
    gamma(x) = [cos(2 * pi * B @ x), sin(2 * pi * B @ x)]^T
alleviating spectral bias in standard coordinate-based MLPs while maintaining
exact identical training loss, boundary conditions, and evaluation metrics.
"""

from typing import Callable, List, Optional, Union
import math
import torch
import torch.nn as nn

from pinn.network import MLP


class FourierFeatureMapping(nn.Module):
    """Random Fourier Feature Projection Layer.

    Maps input coordinates x in R^{d_in} to R^{2 * num_features} via:
        [cos(2 * pi * x @ B^T), sin(2 * pi * x @ B^T)]

    Args:
        input_dim: Dimension of coordinate space (e.g. 1, 2, or 3).
        num_features: Number of random frequencies (output dimension will be 2 * num_features).
        scale: Standard deviation (sigma) of Gaussian frequency matrix B.
        trainable: If False, frequencies B remain fixed random features.
    """

    def __init__(
        self,
        input_dim: int,
        num_features: int,
        scale: float = 1.0,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_features = num_features
        self.scale = scale

        # Sample Gaussian random matrix B of shape (num_features, input_dim)
        B = torch.randn(num_features, input_dim) * scale
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Transforms input coordinates to Fourier space.

        Args:
            x: Input coordinates of shape (N, input_dim).

        Returns:
            Projected features of shape (N, 2 * num_features).
        """
        # x is (N, input_dim), B is (num_features, input_dim)
        # x @ B.T is (N, num_features)
        proj = 2.0 * math.pi * torch.matmul(x, self.B.t())
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class FourierFeaturePINN(nn.Module):
    """Physics-Informed Neural Network with Fourier Feature Embeddings.

    Maintains exact parity with BaselinePINN interface, ensuring fair
    comparison without contaminating baseline physics or trainer interfaces.

    Args:
        input_dim: Dimensionality of inputs (e.g., 2 for (x, t)).
        output_dim: Dimensionality of outputs (e.g., 1 for scalar field u).
        num_fourier_features: Number of Fourier frequencies M (embeds to 2 * M).
        fourier_scale: Standard deviation (sigma) of the Gaussian distribution for frequencies.
        hidden_layers: Number of hidden layers in downstream MLP.
        hidden_units: Neurons per hidden layer.
        activation: Activation function ('tanh', 'gelu', etc.).
        initializer: Weight initializer for MLP linear layers.
        output_transform: Optional callable (x, u_raw) -> u_transformed.
        trainable_fourier: Whether Fourier projection matrix B is optimized during training.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_fourier_features: int = 64,
        fourier_scale: float = 1.0,
        hidden_layers: int = 4,
        hidden_units: Union[int, List[int]] = 50,
        activation: str = "tanh",
        initializer: str = "glorot_normal",
        output_transform: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        trainable_fourier: bool = False,
    ) -> None:
        super().__init__()
        self.fourier_map = FourierFeatureMapping(
            input_dim=input_dim,
            num_features=num_fourier_features,
            scale=fourier_scale,
            trainable=trainable_fourier,
        )
        mlp_input_dim = 2 * num_fourier_features
        self.net = MLP(
            input_dim=mlp_input_dim,
            output_dim=output_dim,
            hidden_layers=hidden_layers,
            hidden_units=hidden_units,
            activation=activation,
            initializer=initializer,
        )
        self.output_transform = output_transform

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward evaluation with Fourier Feature mapping.

        Args:
            x: Input coordinate tensor of shape (N, input_dim).

        Returns:
            Physical field prediction of shape (N, output_dim).
        """
        features = self.fourier_map(x)
        u_raw = self.net(features)
        if self.output_transform is not None:
            return self.output_transform(x, u_raw)
        return u_raw

    def count_parameters(self) -> int:
        """Total trainable parameters in the Fourier Feature model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
