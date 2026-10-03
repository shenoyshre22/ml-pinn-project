"""Neural Network architectures for Physics-Informed Neural Networks (PINNs).

Supports arbitrary depth, layer-width, custom activations, and standard
initializations (Glorot Normal, Glorot Uniform, He, etc.) specified in the reference paper.
"""

from typing import Callable, List, Optional, Union
import torch
import torch.nn as nn


ACTIVATION_REGISTRY = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "sigmoid": nn.Sigmoid,
    "sin": torch.sin,
    "silu": nn.SiLU,
    "identity": nn.Identity,
}


def get_activation(act_name: str) -> Union[nn.Module, Callable[[torch.Tensor], torch.Tensor]]:
    """Retrieves activation function by string key."""
    key = act_name.lower().strip()
    if key not in ACTIVATION_REGISTRY:
        raise ValueError(
            f"Unsupported activation '{act_name}'. Supported: {list(ACTIVATION_REGISTRY.keys())}"
        )
    act = ACTIVATION_REGISTRY[key]
    if isinstance(act, type) and issubclass(act, nn.Module):
        return act()
    return act


def initialize_weights(
    module: nn.Module,
    init_type: str = "glorot_normal",
) -> None:
    """Initializes linear layer weights according to the specified scheme.

    Args:
        module: PyTorch module (usually an nn.Linear layer or sequential).
        init_type: Weight initialization scheme.
            Options: 'glorot_normal' (or 'xavier_normal'),
                     'glorot_uniform' (or 'xavier_uniform'),
                     'he_normal' (or 'kaiming_normal'),
                     'he_uniform' (or 'kaiming_uniform'),
                     'zeros', 'orthogonal'.
    """
    init_key = init_type.lower().replace("-", "_").strip()

    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_key in ["glorot_normal", "xavier_normal"]:
                nn.init.xavier_normal_(m.weight)
            elif init_key in ["glorot_uniform", "xavier_uniform"]:
                nn.init.xavier_uniform_(m.weight)
            elif init_key in ["he_normal", "kaiming_normal"]:
                nn.init.kaiming_normal_(m.weight, nonlinearity="tanh")
            elif init_key in ["he_uniform", "kaiming_uniform"]:
                nn.init.kaiming_uniform_(m.weight, nonlinearity="tanh")
            elif init_key == "orthogonal":
                nn.init.orthogonal_(m.weight)
            elif init_key == "zeros":
                nn.init.zeros_(m.weight)
            else:
                raise ValueError(f"Unknown weight initialization scheme: '{init_type}'")

            if m.bias is not None:
                nn.init.zeros_(m.bias)


class SinActivation(nn.Module):
    """Sine activation function module."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


class MLP(nn.Module):
    """Configurable Multi-Layer Perceptron for PINN approximation.

    Args:
        input_dim: Dimensionality of inputs (e.g., 2 for (x, t) or (x, y)).
        output_dim: Dimensionality of outputs (e.g., 1 for scalar field u).
        hidden_layers: Number of hidden layers (excluding input and output).
        hidden_units: Neurons per hidden layer (integer or list of integers).
        activation: Name of activation function ('tanh', 'relu', etc.).
        initializer: Weight initialization scheme ('glorot_normal', 'glorot_uniform', etc.).
        use_bias: Whether linear layers include bias terms.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: int,
        hidden_units: Union[int, List[int]],
        activation: str = "tanh",
        initializer: str = "glorot_normal",
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers_count = hidden_layers
        self.initializer_name = initializer
        self.activation_name = activation

        if isinstance(hidden_units, int):
            layer_sizes = [hidden_units] * hidden_layers
        elif isinstance(hidden_units, list):
            if len(hidden_units) != hidden_layers:
                raise ValueError(
                    f"Length of hidden_units ({len(hidden_units)}) must match hidden_layers ({hidden_layers})"
                )
            layer_sizes = hidden_units
        else:
            raise TypeError("hidden_units must be an int or a list of ints.")

        self.layer_sizes = [input_dim] + layer_sizes + [output_dim]

        # Build layers sequentially
        layers: List[nn.Module] = []
        for i in range(len(self.layer_sizes) - 1):
            layers.append(
                nn.Linear(
                    self.layer_sizes[i],
                    self.layer_sizes[i + 1],
                    bias=use_bias,
                )
            )
            # Add activation for all but final layer
            if i < len(self.layer_sizes) - 2:
                if activation.lower().strip() == "sin":
                    layers.append(SinActivation())
                else:
                    layers.append(get_activation(activation))

        self.net = nn.Sequential(*layers)
        self.apply_initialization(initializer)

    def apply_initialization(self, initializer: str) -> None:
        """Applies designated weight initialization to network layers."""
        initialize_weights(self.net, initializer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (N, input_dim).

        Returns:
            Output tensor of shape (N, output_dim).
        """
        return self.net(x)

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
