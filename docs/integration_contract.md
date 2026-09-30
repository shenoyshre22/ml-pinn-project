# Integration Contract

This document defines the interfaces between the problem layer,
PINN engine, model layer and evaluation layer.

## Problem Interface

Each problem should provide the following conceptual interface:

- `sample_collocation()`
- `sample_boundary()`
- `sample_observations()`
- `pde_residual(model, points)`
- `boundary_loss(model)`
- `data_loss(model)`
- `reference_solution(points)`

Not every problem uses every method.

## Trainer Interface

The trainer should conceptually support:

```python
trainer.train(
    model=model,
    problem=problem,
    config=config
)