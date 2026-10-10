"""Focused validation for paper-specific configurations and runners."""

from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.run_consolidation import (
    load_consolidation_config,
    run_consolidation_experiment,
    save_consolidation_result,
)
from experiments.run_heat import (
    load_heat_config,
    run_heat_experiment,
    save_heat_result,
)
from experiments.run_inverse import (
    load_inverse_config,
    run_inverse_experiment,
    save_inverse_result,
)


ROOT = Path(__file__).resolve().parents[1]


def test_paper_yaml_values_and_optimizer_protocol() -> None:
    consolidation = load_consolidation_config(ROOT / "configs/consolidation_paper.yaml")
    assert (consolidation.hidden_layers, consolidation.hidden_units) == (5, 50)
    assert consolidation.time_max == 0.5
    assert (consolidation.epochs, consolidation.learning_rate) == (50000, 0.001)
    assert (consolidation.pde_weight, consolidation.condition_weight) == (0.25, 0.25)
    assert consolidation.output_transform_x is True
    assert consolidation.lbfgs_enabled is True

    heat = load_heat_config(ROOT / "configs/heat_paper.yaml")
    assert (heat.hidden_layers, heat.hidden_units) == (4, 50)
    assert (heat.epochs, heat.learning_rate) == (50000, 0.0005)
    assert (heat.pde_weight, heat.boundary_weight) == (0.5, 0.5)
    assert heat.initializer == "glorot_uniform"
    assert heat.lbfgs_enabled is True

    inverse = load_inverse_config(ROOT / "configs/inverse_paper.yaml")
    assert (inverse.hidden_layers, inverse.hidden_units) == (3, 32)
    assert (inverse.epochs, inverse.learning_rate) == (100000, 0.0005)
    assert (inverse.pde_weight, inverse.condition_weight, inverse.data_weight) == (
        0.25,
        0.25,
        0.25,
    )


def test_paper_smoke_runs_save_unique_outputs(tmp_path: Path) -> None:
    consolidation_config = load_consolidation_config(
        ROOT / "configs/consolidation_paper.yaml"
    ).__class__(
        **{
            **load_consolidation_config(
                ROOT / "configs/consolidation_paper.yaml"
            ).__dict__,
            "n_interior": 4,
            "n_spatial_boundary": 4,
            "n_initial": 4,
            "n_test": 9,
            "epochs": 1,
            "log_every": 1,
            "lbfgs_max_iter": 1,
            "output_path": str(tmp_path / "consolidation.json"),
        }
    )
    consolidation_result = run_consolidation_experiment(consolidation_config)
    assert torch.allclose(
        consolidation_result.model(torch.zeros(3, 2)),
        torch.zeros(3, 1),
    )
    save_consolidation_result(consolidation_result, consolidation_config, 0.1)
    assert consolidation_result.history["epoch"][-1] > 1
    assert (tmp_path / "consolidation.json").is_file()
    assert (tmp_path / "consolidation.npz").is_file()

    heat_config = load_heat_config(ROOT / "configs/heat_paper.yaml").__class__(
        **{
            **load_heat_config(ROOT / "configs/heat_paper.yaml").__dict__,
            "n_interior": 4,
            "n_boundary": 4,
            "n_test": 16,
            "epochs": 1,
            "log_every": 1,
            "lbfgs_max_iter": 1,
            "reference_path": None,
            "output_path": str(tmp_path / "heat.json"),
        }
    )
    heat_result = run_heat_experiment(heat_config)
    assert not hasattr(heat_result.model, "output_transform") or (
        heat_result.model.output_transform is None
    )
    save_heat_result(heat_result, heat_config, 0.1)
    assert heat_result.history["epoch"][-1] > 1
    assert (tmp_path / "heat.json").is_file()

    inverse_config = load_inverse_config(ROOT / "configs/inverse_paper.yaml").__class__(
        **{
            **load_inverse_config(ROOT / "configs/inverse_paper.yaml").__dict__,
            "n_observations": 5,
            "n_residual": 6,
            "n_initial": 4,
            "n_test": 10,
            "epochs": 1,
            "log_every": 1,
            "output_path": str(tmp_path / "inverse.json"),
        }
    )
    inverse_result = run_inverse_experiment(inverse_config)
    assert np.isfinite(inverse_result.learned_damping)
    assert np.isfinite(inverse_result.learned_stiffness)
    save_inverse_result(inverse_result, inverse_config, 0.1)
    assert (tmp_path / "inverse.json").is_file()


def test_invalid_paper_configuration_fails_clearly(tmp_path: Path) -> None:
    invalid = tmp_path / "invalid.yaml"
    invalid.write_text(
        "epochs: 0\nlearning_rate: -1\nunknown_key: true\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown configuration keys"):
        load_heat_config(invalid)
