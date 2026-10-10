# ML PINN Project

Reproduction and extension of:

**Data Driven Solutions and Discoveries in Mechanics Using Physics Informed Neural Network**

by Qi Zhang, Yilin Chen, and Ziyi Yang.

## Project Goal

REPRODUCE → VALIDATE → ADD ONE MODEL → COMPARE

## Team

| Name | SRN |
|------|-----|
| Shashank A | PESU2G24CS461 |
| Shreya Shenoy | PES2UG24CS487 |

## Reference Experiments

1. 1D Consolidation
2. 2D Steady-State Heat Conduction
3. Inverse Structural Dynamics

## Extension

Fourier Feature PINN

## Repository Structure

- `problems/` — Physics/problem definitions
- `pinn/` — Generic PINN engine
- `models/` — Baseline and extension models
- `evaluation/` — Metrics, reference solutions and comparisons
- `configs/` — Experiment configurations
- `experiments/` — Experiment runners
- `tests/` — Automated tests
- `results/` — Experiment results
- `figures/` — Generated figures
- `docs/` — Project documentation

The heat runner supports both `model: baseline` and `model: fourier`.
The Fourier configuration uses a fixed Gaussian frequency matrix and maps each
coordinate to sine and cosine features before the same downstream MLP. See
`configs/heat_fourier.yaml` for the reproducible 20,000-epoch comparison
configuration. The full training run is intentionally separate from the
baseline results.

## Paper-specific reproduction configurations

The paper-specific YAML files preserve the existing smoke-test, baseline, and
Fourier configurations. Each runner now requires an explicit configuration
path when invoked as a module and records JSON/NPZ output at the configured
`output_path`. Install dependencies with:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Run the three paper configurations on CPU (including CPU-only Google Colab)
with:

```powershell
.venv\Scripts\python.exe -m experiments.run_consolidation --config configs/consolidation_paper.yaml
.venv\Scripts\python.exe -m experiments.run_heat --config configs/heat_paper.yaml
.venv\Scripts\python.exe -m experiments.run_inverse --config configs/inverse_paper.yaml
```

The consolidation runner supports resumable checkpoints for its Adam and
L-BFGS stages. The paper configuration writes checkpoints under
`checkpoints/consolidation_paper`. Checkpoints contain model weights,
optimizer state, completed stage progress, histories, and Python/NumPy/PyTorch
random-number-generator state. Checkpoint writes use a temporary file followed
by an atomic replacement. To override the directory or frequency:

```powershell
.venv\Scripts\python.exe -m experiments.run_consolidation `
  --config configs/consolidation_paper.yaml `
  --checkpoint-dir checkpoints/consolidation_retry `
  --checkpoint-every 100
```

Resume from a periodic checkpoint with:

```powershell
.venv\Scripts\python.exe -m experiments.run_consolidation `
  --config configs/consolidation_paper.yaml `
  --resume-from checkpoints/consolidation_paper/consolidation_latest.pt
```

For a checkpoint taken during Adam, the runner continues at the next epoch
without repeating completed epochs. For L-BFGS, checkpoints are taken only at
safe outer `optimizer.step` boundaries and restore PyTorch's L-BFGS history;
the closure is recreated from the current experiment data. The final
`complete` checkpoint intentionally omits an optimizer state because no
further training remains. Do not resume from it unless the goal is only to
re-evaluate the saved model.

Default paper outputs are:

- `results/consolidation_paper_50000ep_lr_0001.json` and `.npz`
- `results/heat_paper_50000ep_lr_00005.json` and `.npz`
- `results/inverse_paper_100000ep_lr_0005.json` and `.npz`

The first two configurations run Adam followed by the existing trainer's
L-BFGS stage. The inverse paper configuration intentionally runs Adam only:
the available paper description specifies the 100,000-epoch Adam procedure but
does not specify an inverse L-BFGS stage.

These are configuration reproductions, not claims of numerical reproduction.
The consolidation paper does not specify a finite time horizon in the
available description; this project uses `time_max: 0.5` because it is the
paper's normalized maximum time supplied for this audit. The inverse
observation horizon is not specified; `time_max: 10.0` is an explicit project
assumption. Its 50 initial-condition points are repeated at `t=0` so both
displacement and velocity constraints can be imposed.

The heat reference currently available at
`results/heat_fem_reference.npz` is an independently generated structured
piecewise-linear triangular FEM approximation. It matches the heat PDE,
domain, and boundary conditions, but it is not the paper's deal.II tutorial 3
output. The paper-reported inverse parameter values are stored for comparison
only; they are not asserted until a full run is evaluated.