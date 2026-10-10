"""Run the controlled Fourier-feature heat-equation sensitivity study."""

from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from experiments.run_heat import HeatExperimentConfig, run_heat_experiment
from evaluation.reference_solutions import load_heat_fem_reference
from problems.heat_2d import Heat2DProblem


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "heat_fourier.yaml"
RESULTS_DIR = ROOT / "results"
REFERENCE_PATH = RESULTS_DIR / "heat_fem_reference.npz"
SUMMARY_PATH = RESULTS_DIR / "heat_fourier_sensitivity_summary.json"
COMPARISON_PATH = RESULTS_DIR / "heat_fourier_sensitivity_comparison.csv"

RUNS: tuple[tuple[str, int, float, str], ...] = (
    ("A", 2000, 0.001, "heat_fourier_sensitivity_run_a_2000ep_lr_0001"),
    ("B", 5000, 0.001, "heat_fourier_sensitivity_run_b_5000ep_lr_0001"),
    ("C", 10000, 0.001, "heat_fourier_sensitivity_run_c_10000ep_lr_0001"),
    ("D", 5000, 0.0005, "heat_fourier_sensitivity_run_d_5000ep_lr_00005"),
    ("E", 5000, 0.002, "heat_fourier_sensitivity_run_e_5000ep_lr_0002"),
)


def _load_base_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping in {CONFIG_PATH}")
    return loaded


def _resolved_configs() -> list[tuple[str, HeatExperimentConfig, str]]:
    base = _load_base_config()
    config_fields = {field.name for field in fields(HeatExperimentConfig)}
    resolved: list[tuple[str, HeatExperimentConfig, str]] = []
    for run, epochs, learning_rate, stem in RUNS:
        values = {key: value for key, value in base.items() if key in config_fields}
        values["epochs"] = epochs
        values["learning_rate"] = learning_rate
        values["reference_path"] = str(REFERENCE_PATH)
        config = HeatExperimentConfig(**values)
        if config.epochs != epochs or config.learning_rate != learning_rate:
            raise AssertionError(f"Run {run} does not match the authoritative table")
        resolved.append((run, config, stem))
    return resolved


def _configuration_for_json(config: HeatExperimentConfig) -> dict[str, Any]:
    values = asdict(config)
    values["device"] = str(config.device)
    values["reference_path"] = str(config.reference_path)
    return values


def main() -> None:
    resolved = _resolved_configs()
    stems = [stem for _, _, stem in resolved]
    output_paths = [
        RESULTS_DIR / f"{stem}.{suffix}"
        for stem in stems
        for suffix in ("json", "npz")
    ]
    if len(stems) != len(set(stems)) or len(output_paths) != len(set(output_paths)):
        raise AssertionError("Sensitivity output filenames are not unique")
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(f"FEM reference file not found: {REFERENCE_PATH}")
    if SUMMARY_PATH.exists() or COMPARISON_PATH.exists():
        raise FileExistsError(
            "Aggregate output already exists; refusing to overwrite sensitivity results"
        )
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Run output already exists: {existing}")

    print("Verified Fourier sensitivity configurations:")
    for run, config, stem in resolved:
        print(
            f"Run {run}: epochs={config.epochs}, learning_rate={config.learning_rate}, "
            f"output_stem={stem}"
        )
    print(f"FEM reference: {REFERENCE_PATH} (exists)")
    print("All output filenames are unique. Training will run sequentially.")

    records: list[dict[str, Any]] = []
    for run, config, stem in resolved:
        started = time.perf_counter()
        result = run_heat_experiment(config)
        runtime_seconds = time.perf_counter() - started
        actual_epochs = int(result.history["epoch"][-1])
        if actual_epochs != config.epochs:
            raise AssertionError(
                f"Run {run} completed {actual_epochs} epochs, expected {config.epochs}"
            )
        reference_coordinates, reference_values, _ = load_heat_fem_reference(
            REFERENCE_PATH,
            problem=Heat2DProblem(
                n_interior=config.n_interior,
                n_boundary=config.n_boundary,
                n_test=config.n_test,
                seed=config.seed,
            ),
        )
        npz_path = RESULTS_DIR / f"{stem}.npz"
        json_path = RESULTS_DIR / f"{stem}.json"
        np.savez_compressed(
            npz_path,
            test_coordinates=result.test_coordinates,
            predictions=result.predictions,
            reference_coordinates=reference_coordinates,
            reference_values=reference_values,
            **{key: np.asarray(values) for key, values in result.history.items()},
        )
        initial_losses = {
            key: result.history[key][0]
            for key in ("total_loss", "pde_loss", "bc_loss")
        }
        final_losses = {
            key: result.history[key][-1]
            for key in ("total_loss", "pde_loss", "bc_loss")
        }
        record = {
            "run": run,
            "configuration": _configuration_for_json(config),
            "actual_epochs_completed": actual_epochs,
            "actual_learning_rate": config.learning_rate,
            "runtime_seconds": runtime_seconds,
            "initial_losses": initial_losses,
            "final_losses": final_losses,
            "diagnostics": result.diagnostics,
            "reference_metrics": result.reference_metrics,
            "finite_values": bool(
                np.isfinite(result.predictions).all()
                and all(np.isfinite(values).all() for values in result.history.values())
            ),
            "array_output": str(npz_path.relative_to(ROOT)),
            "metrics_output": str(json_path.relative_to(ROOT)),
        }
        json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        records.append(record)
        print(
            f"Completed Run {run}: epochs={actual_epochs}, "
            f"lr={config.learning_rate}, runtime={runtime_seconds:.2f}s"
        )

    sorted_records = sorted(
        records, key=lambda record: record["reference_metrics"]["relative_l2_error"]
    )
    summary = {
        "experiment": "heat_fourier_sensitivity",
        "base_config": str(CONFIG_PATH.relative_to(ROOT)),
        "reference_path": str(REFERENCE_PATH.relative_to(ROOT)),
        "runs": records,
        "comparison_sorted_by": "relative_l2_error",
        "comparison": [
            {
                "run": record["run"],
                "epochs": record["actual_epochs_completed"],
                "learning_rate": record["actual_learning_rate"],
                "relative_l2_error": record["reference_metrics"]["relative_l2_error"],
                "mean_squared_error": record["reference_metrics"]["mean_squared_error"],
                "max_absolute_error": record["reference_metrics"]["max_absolute_error"],
                "runtime_seconds": record["runtime_seconds"],
                "final_pde_loss": record["final_losses"]["pde_loss"],
                "final_boundary_loss": record["final_losses"]["bc_loss"],
                "final_total_loss": record["final_losses"]["total_loss"],
            }
            for record in sorted_records
        ],
        "note": (
            "These runs use shorter training budgets than the best observed "
            "conventional PINN run, which used 20000 epochs."
        ),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    fields = list(summary["comparison"][0])
    with COMPARISON_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary["comparison"])
    print(f"Summary: {SUMMARY_PATH}")
    print(f"Comparison table: {COMPARISON_PATH}")


if __name__ == "__main__":
    main()
