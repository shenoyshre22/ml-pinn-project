"""Run one matched baseline heat experiment and compare it with Fourier Run C."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from evaluation.metrics import (
    max_absolute_error,
    mean_squared_error,
    relative_l2_error,
)
from evaluation.reference_solutions import load_heat_fem_reference
from experiments.run_heat import HeatExperimentConfig, run_heat_experiment
from problems.heat_2d import Heat2DProblem


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "heat.yaml"
REFERENCE_PATH = ROOT / "results" / "heat_fem_reference.npz"
FOURIER_JSON_PATH = (
    ROOT / "results" / "heat_fourier_sensitivity_run_c_10000ep_lr_0001.json"
)
FOURIER_NPZ_PATH = (
    ROOT / "results" / "heat_fourier_sensitivity_run_c_10000ep_lr_0001.npz"
)
BASELINE_STEM = "heat_baseline_matched_10000ep_lr_0001"
BASELINE_JSON_PATH = ROOT / "results" / f"{BASELINE_STEM}.json"
BASELINE_NPZ_PATH = ROOT / "results" / f"{BASELINE_STEM}.npz"
COMPARISON_JSON_PATH = (
    ROOT / "results" / "heat_baseline_vs_fourier_matched_comparison.json"
)
COMPARISON_CSV_PATH = (
    ROOT / "results" / "heat_baseline_vs_fourier_matched_comparison.csv"
)


def _load_yaml_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {CONFIG_PATH}")
    return loaded


def _resolved_config() -> HeatExperimentConfig:
    values = _load_yaml_config()
    config_fields = {field.name for field in fields(HeatExperimentConfig)}
    values = {key: value for key, value in values.items() if key in config_fields}
    values.update(
        {
            "model": "baseline",
            "epochs": 10000,
            "learning_rate": 0.001,
            "reference_path": str(REFERENCE_PATH),
        }
    )
    config = HeatExperimentConfig(**values)
    assert config.model == "baseline"
    assert config.epochs == 10000
    assert config.learning_rate == 0.001
    assert config.reference_path == str(REFERENCE_PATH)
    return config


def _config_for_json(config: HeatExperimentConfig) -> dict[str, Any]:
    values = asdict(config)
    values["device"] = str(config.device)
    return values


def _assert_preflight(config: HeatExperimentConfig) -> None:
    output_paths = (
        BASELINE_JSON_PATH,
        BASELINE_NPZ_PATH,
        COMPARISON_JSON_PATH,
        COMPARISON_CSV_PATH,
    )
    if len(output_paths) != len(set(output_paths)):
        raise AssertionError("Matched-comparison output filenames are not unique")
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite existing outputs: {existing}")
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"FEM reference file not found: {REFERENCE_PATH}")
    if not FOURIER_JSON_PATH.is_file() or not FOURIER_NPZ_PATH.is_file():
        raise FileNotFoundError("Fourier Run C JSON/NPZ artifact is missing")
    print("Verified matched baseline configuration:")
    print(json.dumps(_config_for_json(config), indent=2))
    print(f"FEM reference exists: {REFERENCE_PATH}")
    print(f"Fourier Run C JSON exists: {FOURIER_JSON_PATH}")
    print(f"Fourier Run C NPZ exists: {FOURIER_NPZ_PATH}")
    print(f"Unique baseline output stem: {BASELINE_STEM}")


def _loss_record(
    result: Any, config: HeatExperimentConfig, runtime_seconds: float
) -> dict[str, Any]:
    actual_epochs = int(result.history["epoch"][-1])
    if actual_epochs != config.epochs:
        raise AssertionError(
            f"Baseline completed {actual_epochs} epochs, expected {config.epochs}"
        )
    return {
        "model": "conventional_baseline_pinn",
        "configuration": _config_for_json(config),
        "actual_epochs_completed": actual_epochs,
        "actual_learning_rate": config.learning_rate,
        "runtime_seconds": runtime_seconds,
        "initial_losses": {
            key: result.history[key][0]
            for key in ("total_loss", "pde_loss", "bc_loss")
        },
        "final_losses": {
            key: result.history[key][-1]
            for key in ("total_loss", "pde_loss", "bc_loss")
        },
        "diagnostics": result.diagnostics,
        "reference_metrics": result.reference_metrics,
        "finite_values": bool(
            np.isfinite(result.predictions).all()
            and all(np.isfinite(values).all() for values in result.history.values())
        ),
        "array_output": str(BASELINE_NPZ_PATH.relative_to(ROOT)),
        "metrics_output": str(BASELINE_JSON_PATH.relative_to(ROOT)),
    }


def main() -> None:
    config = _resolved_config()
    _assert_preflight(config)
    problem = Heat2DProblem(
        n_interior=config.n_interior,
        n_boundary=config.n_boundary,
        n_test=config.n_test,
        seed=config.seed,
    )

    started = time.perf_counter()
    result = run_heat_experiment(config)
    runtime_seconds = time.perf_counter() - started
    baseline_record = _loss_record(result, config, runtime_seconds)
    reference_coordinates, reference_values, _ = load_heat_fem_reference(
        REFERENCE_PATH, problem=problem
    )
    np.savez_compressed(
        BASELINE_NPZ_PATH,
        test_coordinates=result.test_coordinates,
        predictions=result.predictions,
        reference_coordinates=reference_coordinates,
        reference_values=reference_values,
        **{key: np.asarray(values) for key, values in result.history.items()},
    )
    BASELINE_JSON_PATH.write_text(
        json.dumps(baseline_record, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Completed baseline: epochs={baseline_record['actual_epochs_completed']}, "
        f"lr={baseline_record['actual_learning_rate']}, "
        f"runtime={runtime_seconds:.2f}s"
    )

    fourier_record = json.loads(FOURIER_JSON_PATH.read_text(encoding="utf-8"))
    with np.load(FOURIER_NPZ_PATH) as fourier_arrays:
        fourier_test = np.asarray(fourier_arrays["test_coordinates"])
        fourier_reference_coordinates = np.asarray(
            fourier_arrays["reference_coordinates"]
        )
        fourier_reference_values = np.asarray(fourier_arrays["reference_values"])
    same_test_coordinates = np.array_equal(
        result.test_coordinates, fourier_test
    )
    same_reference_coordinates = np.array_equal(
        reference_coordinates, fourier_reference_coordinates
    )
    same_reference_values = np.array_equal(
        reference_values, fourier_reference_values
    )
    if not (same_test_coordinates and same_reference_coordinates and same_reference_values):
        raise AssertionError("Matched comparison arrays do not match")

    baseline_metrics = baseline_record["reference_metrics"]
    fourier_metrics = fourier_record["reference_metrics"]
    baseline_relative = relative_l2_error(reference_values, result.predictions)
    baseline_mse = mean_squared_error(reference_values, result.predictions)
    baseline_max_error = max_absolute_error(reference_values, result.predictions)
    if not np.isclose(baseline_relative, baseline_metrics["relative_l2_error"]):
        raise AssertionError("Baseline relative L2 metric mismatch")
    if not np.isclose(baseline_mse, baseline_metrics["mean_squared_error"]):
        raise AssertionError("Baseline MSE metric mismatch")
    comparison_rows = [
        {
            "model": "conventional_baseline_pinn",
            "run": "matched_baseline",
            "epochs": baseline_record["actual_epochs_completed"],
            "learning_rate": baseline_record["actual_learning_rate"],
            "relative_l2_error": baseline_relative,
            "mean_squared_error": baseline_mse,
            "max_absolute_error": baseline_max_error,
            "final_pde_loss": baseline_record["final_losses"]["pde_loss"],
            "final_boundary_loss": baseline_record["final_losses"]["bc_loss"],
            "final_total_loss": baseline_record["final_losses"]["total_loss"],
            "runtime_seconds": baseline_record["runtime_seconds"],
        },
        {
            "model": "fourier_feature_pinn",
            "run": "fourier_run_c",
            "epochs": fourier_record["actual_epochs_completed"],
            "learning_rate": fourier_record["actual_learning_rate"],
            "relative_l2_error": fourier_metrics["relative_l2_error"],
            "mean_squared_error": fourier_metrics["mean_squared_error"],
            "max_absolute_error": fourier_metrics["max_absolute_error"],
            "final_pde_loss": fourier_record["final_losses"]["pde_loss"],
            "final_boundary_loss": fourier_record["final_losses"]["bc_loss"],
            "final_total_loss": fourier_record["final_losses"]["total_loss"],
            "runtime_seconds": fourier_record["runtime_seconds"],
        },
    ]
    fourier_row = comparison_rows[1]
    comparison = {
        "experiment": "heat_baseline_vs_fourier_matched",
        "metric_formula": (
            "relative_l2_error = ||prediction - reference||_2 / "
            "||reference||_2; MSE = mean((prediction-reference)^2)"
        ),
        "evaluation_match": {
            "test_coordinates_match": same_test_coordinates,
            "reference_coordinates_match": same_reference_coordinates,
            "reference_values_match": same_reference_values,
            "reference_path": str(REFERENCE_PATH.relative_to(ROOT)),
            "metrics_module": "evaluation/metrics.py",
        },
        "runs": comparison_rows,
        "relative_l2_error_change_percent_baseline_vs_fourier": (
            (baseline_relative - fourier_row["relative_l2_error"])
            / fourier_row["relative_l2_error"]
            * 100.0
        ),
        "mse_change_percent_baseline_vs_fourier": (
            (baseline_mse - fourier_row["mean_squared_error"])
            / fourier_row["mean_squared_error"]
            * 100.0
        ),
        "interpretation": (
            "Positive change percentages mean the baseline metric is higher "
            "(deterioration) than Fourier Run C; negative values mean improvement."
        ),
    }
    COMPARISON_JSON_PATH.write_text(
        json.dumps(comparison, indent=2) + "\n", encoding="utf-8"
    )
    fields_for_csv = list(comparison_rows[0])
    with COMPARISON_CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields_for_csv)
        writer.writeheader()
        writer.writerows(comparison_rows)
    print(f"Comparison JSON: {COMPARISON_JSON_PATH}")
    print(f"Comparison CSV: {COMPARISON_CSV_PATH}")


if __name__ == "__main__":
    main()
