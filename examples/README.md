# Examples

All runnable examples live under this single `examples/` tree.

- Top-level scripts demonstrate common workflows (start with
  `fixed_boundary_run.py`):
  - `fixed_boundary_run.py` — read `&INDATA`, converge, write/plot the wout.
  - `plot_and_boozer.py` — every built-in `plot_wout` figure plus the Boozer
    transform (`run_booz_xform` + `plot_boozmn`) on one converged equilibrium.
  - `profiles_power_and_spline.py` — the same equilibrium from power-series and
    cubic-spline pressure/iota profiles (they agree); `NCURR=0` vs `NCURR=1`.
  - `run_from_json.py` — read/convert VMEC++-style JSON (`to_json` /
    `from_file`); the JSON and `&INDATA` forms describe one equilibrium.
  - `hot_restart_scan.py` — seed each scan point from the previous converged
    state; warm restarts converge in ~1 iteration and recompile nothing.
  - `finite_beta_scan.py` — ramp the pressure (hot-restarted) and read beta,
    the Shafranov shift (magnetic-axis motion), and Mercier `DMerc` stability.
  - `take_gradients.py` — exact fixed-boundary gradients of wout scalars
    (aspect, magnetic energy, ...) by implicit differentiation, checked against
    finite differences; O(1) memory, no step size to tune.
  - `free_boundary_mgrid.py` — free-boundary equilibrium from coil currents and
    an mgrid vacuum field (NESTOR); the LCFS is solved for, not prescribed.
  - `free_boundary_beta_scan.py` — ramp the pressure of the free-boundary case
    (coil currents fixed); the LCFS is re-solved by NESTOR at each beta.
  - `take_free_boundary_gradients.py` — differentiate a free-boundary field
    diagnostic through the virtual-casing vacuum field.
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

`mirror_free_boundary_beta_scan.py` solves a two-coil straight-axis mirror at
central beta from 0 to 50%. It writes solved LCFS, field, paraxial comparison,
residual-history, coil, 3D surface, and cap-to-cap field-line plots under
`results/mirror_free_boundary_beta_scan/`, including comparison to the
resolution-qualified Pleiades reference in `examples/data/`.
