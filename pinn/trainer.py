"""Two-stage Trainer for Physics-Informed Neural Networks.

Handles:
- Stage 1: Adam optimization with mini-batch or full-batch evaluation.
- Stage 2: L-BFGS optimization using full-batch closures.
- Loss component tracking (total, pde, bc, data).
- Logging, progress reporting, and state checkpointing.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time
import torch
import torch.nn as nn

from pinn.losses import PINNLoss
from pinn.optimizer import create_adam_optimizer, create_lbfgs_optimizer


class PINNTrainer:
    """Two-Stage Training Manager for PINNs.

    Args:
        model: PyTorch PINN model (nn.Module).
        loss_fn: PINNLoss instance with specified weights.
        device: Device to train on ('cpu' or 'cuda').
        extra_parameters: Optional list of additional parameters (e.g., learnable ODE parameters c, k).
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn: Optional[PINNLoss] = None,
        device: Optional[Union[str, torch.device]] = None,
        extra_parameters: Optional[List[nn.Parameter]] = None,
    ) -> None:
        if device is None:
            # Check if CUDA is available and functional (handle unsupported architecture gracefully)
            if torch.cuda.is_available():
                try:
                    # Test a small tensor operation to ensure kernel availability
                    _ = torch.zeros(1, device="cuda") + 1
                    self.device = torch.device("cuda")
                except Exception:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.loss_fn = loss_fn if loss_fn is not None else PINNLoss()
        self.extra_parameters = extra_parameters or []
        for parameter in self.extra_parameters:
            parameter.data = parameter.data.to(self.device)
            if parameter.grad is not None:
                parameter.grad.data = parameter.grad.data.to(self.device)

        # All parameters to be optimized
        self.all_params = list(self.model.parameters()) + self.extra_parameters

        # History tracking
        self.history: Dict[str, List[float]] = {
            "epoch": [],
            "total_loss": [],
            "pde_loss": [],
            "bc_loss": [],
            "data_loss": [],
            "time_elapsed": [],
        }

    def train_adam(
        self,
        closure_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        epochs: int = 50000,
        lr: float = 0.001,
        log_every: int = 1000,
        callbacks: Optional[List[Callable[[int, Dict[str, float]], None]]] = None,
    ) -> Dict[str, List[float]]:
        """Executes Adam optimization phase.

        Args:
            closure_fn: Callable returning (loss_f, loss_b, loss_d).
            epochs: Number of Adam iterations.
            lr: Learning rate.
            log_every: Frequency of console reporting.
            callbacks: Optional list of callback functions (epoch, loss_dict).

        Returns:
            Dictionary containing loss histories.
        """
        optimizer = create_adam_optimizer(self.all_params, lr=lr)
        start_time = time.time()

        for epoch in range(1, epochs + 1):
            optimizer.zero_grad()
            loss_f, loss_b, loss_d = closure_fn()
            total_loss, loss_dict = self.loss_fn(loss_f, loss_b, loss_d)
            total_loss.backward()
            optimizer.step()

            # Record history
            if epoch % log_every == 0 or epoch == 1 or epoch == epochs:
                elapsed = time.time() - start_time
                self.history["epoch"].append(epoch)
                self.history["total_loss"].append(loss_dict["total"])
                self.history["pde_loss"].append(loss_dict["pde"])
                self.history["bc_loss"].append(loss_dict["bc"])
                self.history["data_loss"].append(loss_dict["data"])
                self.history["time_elapsed"].append(elapsed)

                if callbacks:
                    for cb in callbacks:
                        cb(epoch, loss_dict)

                print(
                    f"[Adam] Epoch {epoch:6d}/{epochs} | "
                    f"Total: {loss_dict['total']:.5e} | "
                    f"PDE: {loss_dict['pde']:.5e} | "
                    f"BC: {loss_dict['bc']:.5e} | "
                    f"Data: {loss_dict['data']:.5e} | "
                    f"Time: {elapsed:.2f}s"
                )

        return self.history

    def train_lbfgs(
        self,
        closure_fn: Callable[[], Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]],
        max_iter: int = 50000,
        log_every: int = 100,
        callbacks: Optional[List[Callable[[int, Dict[str, float]], None]]] = None,
    ) -> Dict[str, List[float]]:
        """Executes L-BFGS optimization phase.

        Args:
            closure_fn: Callable returning (loss_f, loss_b, loss_d).
            max_iter: Maximum quasi-Newton iterations.
            log_every: Frequency of console reporting.
            callbacks: Optional list of callback functions (step, loss_dict).

        Returns:
            Updated history dictionary.
        """
        optimizer = create_lbfgs_optimizer(self.all_params, max_iter=max_iter)
        start_time = time.time()
        start_step = self.history["epoch"][-1] if self.history["epoch"] else 0
        step_counter = [0]

        def step_closure() -> torch.Tensor:
            optimizer.zero_grad()
            loss_f, loss_b, loss_d = closure_fn()
            total_loss, loss_dict = self.loss_fn(loss_f, loss_b, loss_d)
            total_loss.backward()

            step_counter[0] += 1
            curr_eval = step_counter[0]
            curr_global_step = start_step + curr_eval

            if curr_eval % log_every == 0 or curr_eval == 1:
                elapsed = time.time() - start_time
                self.history["epoch"].append(curr_global_step)
                self.history["total_loss"].append(loss_dict["total"])
                self.history["pde_loss"].append(loss_dict["pde"])
                self.history["bc_loss"].append(loss_dict["bc"])
                self.history["data_loss"].append(loss_dict["data"])
                self.history["time_elapsed"].append(elapsed)

                if callbacks:
                    for cb in callbacks:
                        cb(curr_global_step, loss_dict)

                print(
                    f"[L-BFGS] Eval {curr_eval:6d} (Step {curr_global_step:6d}) | "
                    f"Total: {loss_dict['total']:.5e} | "
                    f"PDE: {loss_dict['pde']:.5e} | "
                    f"BC: {loss_dict['bc']:.5e} | "
                    f"Data: {loss_dict['data']:.5e} | "
                    f"Time: {elapsed:.2f}s"
                )

            return total_loss

        optimizer.step(step_closure)
        return self.history

    def save_checkpoint(self, filepath: str, extra_state: Optional[Dict[str, Any]] = None) -> None:
        """Saves model weights, extra parameters, and history to a checkpoint file."""
        state = {
            "model_state_dict": self.model.state_dict(),
            "history": self.history,
        }
        if self.extra_parameters:
            state["extra_params"] = [p.detach().cpu() for p in self.extra_parameters]
        if extra_state:
            state.update(extra_state)
        torch.save(state, filepath)

    def load_checkpoint(self, filepath: str) -> Dict[str, Any]:
        """Loads state checkpoint."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "history" in checkpoint:
            self.history = checkpoint["history"]
        if "extra_params" in checkpoint and self.extra_parameters:
            for p, saved_p in zip(self.extra_parameters, checkpoint["extra_params"]):
                p.data.copy_(saved_p.to(self.device))
        return checkpoint
