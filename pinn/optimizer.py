"""Optimizer factory and configuration module for PINN training.

Configures Adam and L-BFGS optimizers according to reference paper specifications.
"""

from typing import Any, Dict, Iterable, Optional
import torch
import torch.optim as optim


def create_adam_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float = 0.001,
    betas: tuple = (0.9, 0.999),
    eps: float = 1e-8,
    weight_decay: float = 0.0,
) -> optim.Adam:
    """Creates configured Adam optimizer.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (e.g., 0.001 for 1D consolidation, 0.0005 for Heat/Inverse).
        betas: Coefficients for computing running averages of gradient and its square.
        eps: Term added to denominator to improve numerical stability.
        weight_decay: Weight decay (L2 penalty).

    Returns:
        Configured Adam optimizer instance.
    """
    return optim.Adam(
        params,
        lr=lr,
        betas=betas,
        eps=eps,
        weight_decay=weight_decay,
    )


def create_lbfgs_optimizer(
    params: Iterable[torch.nn.Parameter],
    lr: float = 1.0,
    max_iter: int = 50000,
    max_eval: Optional[int] = None,
    tolerance_grad: float = 1e-7,
    tolerance_change: float = 1e-9,
    history_size: int = 50,
    line_search_fn: str = "strong_wolfe",
) -> optim.LBFGS:
    """Creates configured L-BFGS optimizer for second-stage fine convergence.

    Args:
        params: Iterable of parameters to optimize.
        lr: Learning rate (default 1.0 for quasi-Newton step scaling).
        max_iter: Maximal number of iterations per optimization step.
        max_eval: Maximal number of function evaluations per optimization step.
        tolerance_grad: Termination tolerance on first-order optimality.
        tolerance_change: Termination tolerance on function value/parameter changes.
        history_size: Update history size.
        line_search_fn: Line search algorithm ('strong_wolfe' or None).

    Returns:
        Configured L-BFGS optimizer instance.
    """
    if max_eval is None:
        max_eval = int(max_iter * 1.25)

    return optim.LBFGS(
        params,
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn=line_search_fn,
    )
