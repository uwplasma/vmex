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
  - `free_boundary_essos_coils.py` — free-boundary beta scan directly from
    ESSOS coils (direct JAX Biot-Savart, no mgrid file); `PRES_SCALE` is
    calibrated per point so the *actual* wout `betatotal` targets 0--5 % with
    bounded adaptive continuation.
  - `free_boundary_tokamak_coils.py` — construct circular tokamak TF/PF coils,
    generate their mgrid, and solve the same finite-beta LCFS through direct
    Biot-Savart and mgrid backends with a quantitative boundary comparison.
  - `take_free_boundary_gradients.py` — differentiate a free-boundary field
    diagnostic through the virtual-casing vacuum field.
  - `mirror_free_boundary_beta_scan.py` — solve the straight-axis two-coil
    free-boundary mirror through 50% beta and render horizontal 3D coils,
    cap-to-cap field lines, solved LCFS, `|B|`, pressure, and residual plots.
    Each accepted point is also written as mirror-native `mout_*.nc`; rerun
    its four endpoint figures with `vmec --plot mout_*.nc`.
  - `mirror_fixed_boundary_gradients.py` — solve a finite-pressure,
    finite-current flared mirror, differentiate an interior radius with one
    preconditioned implicit adjoint, validate boundary/flux/pressure/current
    gradients against reconverged central differences, and write MOUT plus
    3D, `|B|`, cross-section, residual, and sensitivity figures.
  - `toroidal_stellarator_mirror_hybrid.py` — trace the 16-coil vacuum axis,
    build a flux-conserving square-torus seed, continue rotating corner
    ellipses, solve the finite-current equilibrium, and write WOUT plus 3D
    coils/LCFS/pitched-field-line, `|B|`, cross-section, profile, and residual
    plots.
  - `toroidal_stellarator_mirror_hybrid_free_boundary.py` — construct the same
    16 coils, solve the coil-matched LCFS with NESTOR, and pressure-continue
    only accepted equilibria. The requested schedule extends through 50%, but
    the present Fourier corrector stops honestly at its documented sub-1%
    conditioning barrier; no prescribed high-beta surfaces are plotted.
  - `single_stage_free_boundary_opt.py` — optimize coil currents to confine a
    target plasma by minimizing <(B.n)^2> with the exact virtual-casing gradient.
- `optimization/`: precise QA/QH/QP/QI from a circular torus — one file each,
  plus `QA_bootstrap_selfconsistent.py` / `QH_bootstrap_selfconsistent.py`,
  which reproduce the self-consistent Redl bootstrap-current workflow of
  arXiv:2205.02914. The QI ESS example reports each omnigenity component and
  writes the standard
  cross-section, profile, `|B|`, and field-line-overlaid 3D figures.
  `QA_optimization_ess.py` is the single-call variant. The QI ESS script
  releases all large-`max_mode` harmonics at once, then runs three measured
  constraint-restoration calls without changing `max_mode`. Both use
  Exponential Spectral Scaling (`use_ess`) instead of a mode ladder and remain
  simsopt-style (`(function, target, weight)` terms + one least-squares call
  per `max_mode` continuation stage, implicit adjoint gradients).  All read
  `VMEC_JAX_EXAMPLES_CI=1` to shrink budgets for the CI smoke tests
  (`tests/test_examples.py`).
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
