# Examples

All runnable examples live under this single `examples/` tree.

- Top-level scripts demonstrate common workflows (start with
  `fixed_boundary_run.py`).
- `optimization/`: precise QA/QH/QP/QI from a circular torus — one file each,
  simsopt-style (`(function, target, weight)` terms + one least-squares call
  per `max_mode` continuation stage, implicit adjoint gradients).  All read
  `VMEC_JAX_EXAMPLES_CI=1` to shrink budgets for the CI smoke tests
  (`tests/core_new/test_examples.py`).
- `data/`: bundled input decks and small checked-in fixtures.
- `data/single_grid/`: fixed-boundary single-grid benchmark inputs and optional
  fetched reference assets.

Generated outputs should go to ignored `results/`, `outputs/`, or a user-chosen
directory.  Do not commit generated WOUT, mgrid, Boozer, PDF, or plot files
unless they are compact reviewed documentation artifacts.
