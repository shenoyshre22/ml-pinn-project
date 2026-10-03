"""Unit tests for PINN loss functions (pinn/losses.py)."""

import pytest
import torch

from pinn.losses import PINNLoss, mse_loss, residual_loss


def test_mse_and_residual_loss():
    pred = torch.tensor([1.0, 2.0, 3.0])
    target = torch.tensor([1.0, 1.0, 1.0])
    # diff = [0, 1, 2], squared = [0, 1, 4], mean = 5/3
    loss = mse_loss(pred, target)
    assert pytest.approx(loss.item(), 1e-5) == 5.0 / 3.0

    res = torch.tensor([2.0, -2.0])
    res_l = residual_loss(res)
    assert pytest.approx(res_l.item(), 1e-5) == 4.0


def test_pinn_loss_weighting():
    criterion = PINNLoss(w_f=0.25, w_b=0.25, w_d=0.5)

    loss_f = torch.tensor(4.0)
    loss_b = torch.tensor(8.0)
    loss_d = torch.tensor(2.0)

    # 0.25 * 4 + 0.25 * 8 + 0.5 * 2 = 1 + 2 + 1 = 4.0
    total, log_dict = criterion(loss_f, loss_b, loss_d)

    assert pytest.approx(total.item(), 1e-5) == 4.0
    assert log_dict["pde"] == 4.0
    assert log_dict["bc"] == 8.0
    assert log_dict["data"] == 2.0
    assert log_dict["total"] == 4.0


def test_pinn_loss_non_negative():
    criterion = PINNLoss()
    loss_f = torch.abs(torch.randn(1))
    loss_b = torch.abs(torch.randn(1))

    total, _ = criterion(loss_f, loss_b)
    assert total.item() >= 0.0
