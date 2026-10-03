"""Unit tests for optimizer and training loops (pinn/trainer.py)."""

import pytest
import torch

from models.baseline_pinn import BaselinePINN
from pinn.losses import PINNLoss
from pinn.trainer import PINNTrainer


def test_trainer_adam_convergence():
    # Fit y = x^2 with small baseline PINN
    model = BaselinePINN(
        input_dim=1,
        output_dim=1,
        hidden_layers=2,
        hidden_units=16,
        activation="tanh",
    )
    trainer = PINNTrainer(model=model, loss_fn=PINNLoss(w_f=1.0, w_b=0.0, w_d=0.0))

    x_data = torch.linspace(-1.0, 1.0, 30, device=trainer.device).unsqueeze(-1)
    y_target = x_data ** 2

    def closure():
        pred = model(x_data)
        loss = torch.mean((pred - y_target) ** 2)
        zero_bc = torch.tensor(0.0, device=trainer.device)
        return loss, zero_bc, None

    history = trainer.train_adam(closure, epochs=50, lr=0.01, log_every=25)
    assert len(history["total_loss"]) >= 2
    # Verify loss decreases
    assert history["total_loss"][-1] < history["total_loss"][0]


def test_trainer_checkpoint(tmp_path):
    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    trainer = PINNTrainer(model=model)

    ckpt_path = str(tmp_path / "model_ckpt.pt")
    trainer.save_checkpoint(ckpt_path)

    new_model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    new_trainer = PINNTrainer(model=new_model)
    new_trainer.load_checkpoint(ckpt_path)

    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)


def test_trainer_two_stage_adam_and_lbfgs():
    # Fit y = sin(pi * x)
    model = BaselinePINN(
        input_dim=1,
        output_dim=1,
        hidden_layers=2,
        hidden_units=16,
        activation="tanh",
    )
    trainer = PINNTrainer(model=model, loss_fn=PINNLoss(w_f=1.0, w_b=0.0, w_d=0.0))

    x_data = torch.linspace(-1.0, 1.0, 30, device=trainer.device).unsqueeze(-1)
    y_target = torch.sin(3.14159 * x_data)

    def closure():
        pred = model(x_data)
        loss = torch.mean((pred - y_target) ** 2)
        zero_bc = torch.tensor(0.0, device=trainer.device)
        return loss, zero_bc, None

    # Stage 1: Adam
    trainer.train_adam(closure, epochs=20, lr=0.01, log_every=10)
    loss_after_adam = trainer.history["total_loss"][-1]

    # Stage 2: L-BFGS
    trainer.train_lbfgs(closure, max_iter=20, log_every=5)
    loss_after_lbfgs = trainer.history["total_loss"][-1]

    assert loss_after_lbfgs <= loss_after_adam
    assert trainer.history["epoch"][-1] > 20
