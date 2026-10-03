"""Loss calculation module for Physics-Informed Neural Networks.

Computes:
- PDE residual loss: L_f
- Boundary / Initial condition loss: L_b
- Observational data loss: L_d
- Weighted composite loss: L = w_f * L_f + w_b * L_b + w_d * L_d
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn


def mse_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean Squared Error (L2) loss between predictions and targets."""
    return torch.mean((prediction - target) ** 2)


def residual_loss(residual: torch.Tensor) -> torch.Tensor:
    """Computes MSE residual loss against zero (i.e. mean(residual^2)).

    Args:
        residual: Tensor representing PDE/ODE residual evaluated at collocation points.

    Returns:
        Scalar MSE tensor.
    """
    return torch.mean(residual ** 2)


class PINNLoss(nn.Module):
    """Composite weighted loss function for Physics-Informed Neural Networks.

    Formulation:
        L = w_f * L_f + w_b * L_b + w_d * L_d

    Args:
        w_f: Weight for PDE residual loss (default 0.25).
        w_b: Weight for boundary/initial condition loss (default 0.25).
        w_d: Weight for observation data loss (default 0.0 or 0.25).
    """

    def __init__(
        self,
        w_f: float = 0.25,
        w_b: float = 0.25,
        w_d: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_f = float(w_f)
        self.w_b = float(w_b)
        self.w_d = float(w_d)

    def forward(
        self,
        loss_f: torch.Tensor,
        loss_b: torch.Tensor,
        loss_d: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculates total weighted loss and returns diagnostic component scalars.

        Args:
            loss_f: Scalar tensor for PDE residual loss.
            loss_b: Scalar tensor for boundary/initial condition loss.
            loss_d: Optional scalar tensor for observation data loss.

        Returns:
            Tuple of:
                total_loss: Differentiable scalar torch.Tensor.
                loss_dict: Dictionary with detached float values for logging:
                           {'total': float, 'pde': float, 'bc': float, 'data': float}
        """
        total = self.w_f * loss_f + self.w_b * loss_b

        loss_d_val = 0.0
        if loss_d is not None and self.w_d > 0.0:
            total = total + self.w_d * loss_d
            loss_d_val = float(loss_d.item())

        loss_dict: Dict[str, float] = {
            "total": float(total.item()),
            "pde": float(loss_f.item()),
            "bc": float(loss_b.item()),
            "data": loss_d_val,
        }

        return total, loss_dict
