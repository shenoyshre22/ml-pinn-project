# Integration Contract

This document defines the interfaces between the problem layer,
PINN engine, model layer and evaluation layer.

## Problem Interface

Each problem should provide the following conceptual interface:

- `sample_collocation()`
- `sample_boundary()`
- `sample_observations()`
- `pde_residual(...)` — computes the governing-equation residual from the
  derivatives supplied by the PINN engine.
- `boundary_loss(model)`
- `data_loss(model)`
- `reference_solution(points)`

Not every problem uses every method.

The problem layer defines the governing equation and combines the supplied
derivatives into the residual. The generic PINN engine evaluates the model
and performs automatic differentiation before supplying those derivatives.
Different problems may require different derivative arguments, so this
contract intentionally does not prescribe one universal concrete
`pde_residual` signature.

## Trainer Interface

The trainer should conceptually support:

```python
trainer.train(
    model=model,
    problem=problem,
    config=config
)