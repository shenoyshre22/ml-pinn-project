"""Unit tests for optimizer and training loops (pinn/trainer.py)."""

import pytest
import torch

from models.baseline_pinn import BaselinePINN
from pinn.losses import PINNLoss
from pinn.trainer import PINNTrainer


def test_trainer_moves_extra_parameters_to_model_device() -> None:
    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    extra_parameter = torch.nn.Parameter(torch.tensor(1.0))
    trainer = PINNTrainer(
        model=model,
        device="cpu",
        extra_parameters=[extra_parameter],
    )

    model_device = next(trainer.model.parameters()).device
    assert extra_parameter.device == model_device
    assert any(parameter is extra_parameter for parameter in trainer.all_params)
    assert extra_parameter.requires_grad


def test_trainer_moves_extra_parameters_to_usable_cuda_device() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this PyTorch installation")

    try:
        torch.zeros(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    except Exception as error:
        pytest.skip(f"CUDA is reported available but unusable: {error}")

    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    extra_parameter = torch.nn.Parameter(torch.tensor(1.0))
    trainer = PINNTrainer(
        model=model,
        device="cuda",
        extra_parameters=[extra_parameter],
    )

    model_device = next(trainer.model.parameters()).device
    assert model_device.type == "cuda"
    assert extra_parameter.device == model_device
    assert any(parameter is extra_parameter for parameter in trainer.all_params)
    assert extra_parameter.requires_grad

    x_data = torch.zeros(2, 1, device=trainer.device)

    def closure():
        prediction = trainer.model(x_data) + extra_parameter
        loss = torch.mean(prediction**2)
        zero_boundary_loss = torch.zeros((), device=trainer.device)
        return loss, zero_boundary_loss, None

    trainer.train_adam(closure, epochs=1, lr=0.001, log_every=1)


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


def test_trainer_checkpoint_restores_extra_parameters(tmp_path):
    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    saved_parameter = torch.nn.Parameter(torch.tensor(3.5))
    trainer = PINNTrainer(model=model, extra_parameters=[saved_parameter])
    checkpoint_path = str(tmp_path / "extra_params.pt")

    trainer.save_checkpoint(checkpoint_path)

    new_model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    restored_parameter = torch.nn.Parameter(torch.tensor(-2.0))
    new_trainer = PINNTrainer(
        model=new_model,
        extra_parameters=[restored_parameter],
    )
    new_trainer.load_checkpoint(checkpoint_path)

    assert restored_parameter.item() == pytest.approx(3.5)


def test_trainer_checkpoint_rejects_unexpected_extra_parameters(tmp_path):
    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    saved_parameter = torch.nn.Parameter(torch.tensor(1.0))
    checkpoint_path = str(tmp_path / "extra_params.pt")
    PINNTrainer(model=model, extra_parameters=[saved_parameter]).save_checkpoint(
        checkpoint_path
    )

    new_model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    new_trainer = PINNTrainer(model=new_model)
    with pytest.raises(ValueError, match="Checkpoint contains extra parameters"):
        new_trainer.load_checkpoint(checkpoint_path)


def test_trainer_checkpoint_rejects_missing_extra_parameters(tmp_path):
    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    checkpoint_path = str(tmp_path / "no_extra_params.pt")
    PINNTrainer(model=model).save_checkpoint(checkpoint_path)

    new_model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    extra_parameter = torch.nn.Parameter(torch.tensor(1.0))
    new_trainer = PINNTrainer(
        model=new_model,
        extra_parameters=[extra_parameter],
    )
    with pytest.raises(ValueError, match="Current trainer has extra parameters"):
        new_trainer.load_checkpoint(checkpoint_path)


def test_trainer_checkpoint_rejects_different_extra_parameter_counts(tmp_path):
    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    checkpoint_path = str(tmp_path / "one_extra_param.pt")
    PINNTrainer(
        model=model,
        extra_parameters=[torch.nn.Parameter(torch.tensor(1.0))],
    ).save_checkpoint(checkpoint_path)

    new_model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    new_trainer = PINNTrainer(
        model=new_model,
        extra_parameters=[
            torch.nn.Parameter(torch.tensor(2.0)),
            torch.nn.Parameter(torch.tensor(3.0)),
        ],
    )
    with pytest.raises(ValueError, match="different numbers of extra parameters"):
        new_trainer.load_checkpoint(checkpoint_path)


def test_trainer_checkpoint_rejects_extra_parameter_shape_mismatch(tmp_path):
    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    checkpoint_path = str(tmp_path / "shaped_extra_param.pt")
    PINNTrainer(
        model=model,
        extra_parameters=[torch.nn.Parameter(torch.ones(2))],
    ).save_checkpoint(checkpoint_path)

    new_model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    new_trainer = PINNTrainer(
        model=new_model,
        extra_parameters=[torch.nn.Parameter(torch.ones(3))],
    )
    with pytest.raises(ValueError, match=r"Extra parameter 0.*shape"):
        new_trainer.load_checkpoint(checkpoint_path)


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


@pytest.mark.parametrize(
    ("train_method", "log_every"),
    [
        ("train_adam", 0),
        ("train_adam", -1),
        ("train_lbfgs", 0),
        ("train_lbfgs", -1),
    ],
)
def test_trainer_rejects_non_positive_log_every(train_method, log_every):
    model = BaselinePINN(input_dim=1, output_dim=1, hidden_layers=1, hidden_units=8)
    trainer = PINNTrainer(model=model)

    def closure():
        prediction = model(torch.zeros(1, 1, device=trainer.device))
        zero_loss = torch.zeros((), device=trainer.device)
        return prediction.mean(), zero_loss, None

    with pytest.raises(ValueError, match="^log_every must be at least 1$"):
        if train_method == "train_adam":
            trainer.train_adam(closure, epochs=1, log_every=log_every)
        else:
            trainer.train_lbfgs(closure, max_iter=1, log_every=log_every)
