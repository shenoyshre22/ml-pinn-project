"""Automatic Differentiation module for Physics-Informed Neural Networks.

Provides vectorized first-order and second-order derivative operators
required for PDE/ODE residual computation in PyTorch.
"""

from typing import List, Optional, Tuple, Union
import torch


def grad(
    y: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
    retain_graph: bool = True,
    allow_unused: bool = False,
) -> torch.Tensor:
    """Computes the gradient dy/dx where y is a scalar or single-output tensor.

    Args:
        y: Dependent variable tensor of shape (N, 1) or scalar.
        x: Independent variable tensor of shape (N, D) with requires_grad=True.
        create_graph: Whether to create graph for higher-order derivatives.
        retain_graph: Whether to retain graph for subsequent backward passes.
        allow_unused: Whether to return zeros if x was not used in computing y.

    Returns:
        Gradient tensor dy/dx of matching shape to x (N, D).
    """
    if not x.requires_grad:
        raise ValueError("Independent variable 'x' must have requires_grad=True.")

    grad_outputs: torch.Tensor = torch.ones_like(y)
    computed_grad = torch.autograd.grad(
        outputs=y,
        inputs=x,
        grad_outputs=grad_outputs,
        create_graph=create_graph,
        retain_graph=retain_graph,
        allow_unused=allow_unused,
    )[0]

    if computed_grad is None:
        if allow_unused:
            return torch.zeros_like(x)
        raise RuntimeError("Gradient computation returned None. Check graph connectivity.")

    return computed_grad


def nth_derivative(
    y: torch.Tensor,
    x: torch.Tensor,
    n: int = 1,
    component_x: Optional[int] = None,
    create_graph: bool = True,
) -> torch.Tensor:
    """Computes the n-th derivative of y with respect to x or x[:, component_x].

    Args:
        y: Dependent variable tensor of shape (N, 1).
        x: Independent variable tensor of shape (N, D).
        n: Order of differentiation (>= 1).
        component_x: If specified, index of the coordinate in x to differentiate with respect to.
        create_graph: If True, preserves computational graph for higher order computation.

    Returns:
        Tensor of shape (N, 1) containing d^n y / dx_i^n.
    """
    if n < 1:
        raise ValueError("Order of differentiation 'n' must be >= 1.")

    current = y
    for order in range(n):
        # Keep graph on all steps except possibly the last if create_graph is False
        keep_graph = True if order < (n - 1) else create_graph
        dy_dx = grad(current, x, create_graph=keep_graph, retain_graph=True)
        if component_x is not None:
            current = dy_dx[:, component_x : component_x + 1]
        else:
            current = dy_dx

    return current


def jacobian(
    y: torch.Tensor,
    x: torch.Tensor,
    create_graph: bool = True,
) -> torch.Tensor:
    """Computes the Jacobian matrix dy/dx for vector-valued outputs.

    Args:
        y: Tensor of shape (N, M) where M is output dimension.
        x: Tensor of shape (N, D) where D is input dimension.
        create_graph: Preserve graph for higher derivatives.

    Returns:
        Jacobian tensor of shape (N, M, D) where J[n, i, j] = dy_i / dx_j.
    """
    N, M = y.shape
    D = x.shape[1]
    jac = torch.zeros(N, M, D, dtype=y.dtype, device=y.device)

    for i in range(M):
        grad_i = grad(y[:, i : i + 1], x, create_graph=create_graph, retain_graph=True)
        jac[:, i, :] = grad_i

    return jac


def hessian(
    y: torch.Tensor,
    x: torch.Tensor,
    component_y: int = 0,
    create_graph: bool = True,
) -> torch.Tensor:
    """Computes the Hessian matrix (second derivatives) of y[:, component_y] with respect to x.

    Args:
        y: Tensor of shape (N, M).
        x: Tensor of shape (N, D).
        component_y: Output component to compute Hessian for.
        create_graph: Preserve graph.

    Returns:
        Hessian tensor of shape (N, D, D) where H[n, i, j] = d^2 y / (dx_i dx_j).
    """
    first_grad = grad(y[:, component_y : component_y + 1], x, create_graph=True, retain_graph=True)
    N, D = x.shape
    hess = torch.zeros(N, D, D, dtype=x.dtype, device=x.device)

    for j in range(D):
        grad_j = grad(first_grad[:, j : j + 1], x, create_graph=create_graph, retain_graph=True)
        hess[:, j, :] = grad_j

    return hess


def laplacian(
    y: torch.Tensor,
    x: torch.Tensor,
    component_y: int = 0,
    create_graph: bool = True,
) -> torch.Tensor:
    """Computes the Laplacian operator: nabla^2 y = sum_i (d^2 y / dx_i^2).

    Required for heat equation: d^2 T/dx^2 + d^2 T/dy^2.

    Args:
        y: Dependent variable tensor of shape (N, M).
        x: Spatial coordinate tensor of shape (N, D).
        component_y: Index of the output variable.
        create_graph: Preserve graph.

    Returns:
        Tensor of shape (N, 1) representing the Laplacian sum.
    """
    first_grad = grad(y[:, component_y : component_y + 1], x, create_graph=True, retain_graph=True)
    D = x.shape[1]
    lap = torch.zeros_like(y[:, component_y : component_y + 1])

    for i in range(D):
        second_grad = grad(
            first_grad[:, i : i + 1],
            x,
            create_graph=create_graph,
            retain_graph=True,
        )
        lap = lap + second_grad[:, i : i + 1]

    return lap
