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