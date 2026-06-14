Code structure
==============

Top-level package layout (selected):

- ``vmec_jax/api.py``: stable public import surface for solve, I/O, plotting,
  diagnostics, and documented optimization workflow objects
- ``vmec_jax/namelist.py``: minimal Fortran namelist reader (``&INDATA``)
- ``vmec_jax/config.py``: run discretization config extracted from inputs
- ``vmec_jax/static.py``: static grids + Fourier basis tensors (PyTrees)
- ``vmec_jax/state.py``: coefficient containers (PyTrees)
- ``vmec_jax/boundary.py``: boundary coefficient parsing + VMEC m=1 constraint
- ``vmec_jax/init_guess.py``: profil3d-style initial guess (axis + boundary)
- ``vmec_jax/coords.py`` / ``vmec_jax/geom.py``: geometry and metric kernels
- ``vmec_jax/field.py`` / ``vmec_jax/energy.py``: B-field and energy diagnostics
- ``vmec_jax/vmec_tomnsp.py``: VMEC ``fixaray`` tables + DFT tomnsps kernels
- ``vmec_jax/vmec_bcovar.py``: bcovar/metric assembly and half-mesh pipeline
- ``vmec_jax/preconditioner_1d.py``: VMEC-style preconditioner operators
- ``vmec_jax/solve_force_norm_helpers.py``: force-block weighting, lambda
  residual norms, and stability-guard timestep helpers extracted from the
  residual iteration hot path
- ``vmec_jax/solve_tolerance_helpers.py``: dtype-aware gradient, conjugate
  gradient, and Levenberg-Marquardt tolerance policies
- ``vmec_jax/solve_constraint_helpers.py``: fixed-boundary edge constraints,
  magnetic-axis regularity, lambda-gauge projection, and related NumPy/JAX
  coefficient-slice helpers
- ``vmec_jax/solve_gradient_helpers.py``: state gradient-descent updates and
  feasible-gradient projections for fixed-boundary/axis/lambda constraints
- ``vmec_jax/solve_preconditioner_helpers.py``: fixed-boundary mode-diagonal
  and radial Dirichlet smoothing preconditioner kernels, tridiagonal policy
  resolution, metric preconditioner scales, and radial mesh scale factors
- ``vmec_jax/solve_jit_cache_helpers.py``: environment-controlled JIT-cache
  limits, structural cache keys, LRU helpers, and scan-cache miss diagnostics
- ``vmec_jax/solve_axis_reset_helpers.py``: initial magnetic-axis reset
  control decisions, axis-state merging, and optional axis coefficient dumps
- ``vmec_jax/solve_free_boundary_control_helpers.py``: free-boundary cadence,
  turn-on, constraint-baseline, and velocity-block control helpers
- ``vmec_jax/solve_scan_resume_helpers.py``: VMEC2000-style scan resume-state
  initialization and carry-field restoration
- ``vmec_jax/solve.py``: fixed-boundary solvers + VMEC2000 iteration loop
- ``vmec_jax/driver.py``: CLI-facing fixed/free-boundary drivers, output
  policies, staged solve dispatch, and wout writing
- ``vmec_jax/free_boundary.py``: mgrid loading, NESTOR-like vacuum coupling,
  and free-boundary runtime state helpers
- ``vmec_jax/optimization.py``: exact fixed-boundary optimizer, boundary DOF
  maps, accepted-point replay, and discrete-adjoint Jacobian plumbing
- ``vmec_jax/optimization_workflow.py``: user-facing optimization problem
  objects, objective tuples, continuation stages, and example workflow helpers
- ``vmec_jax/qi_diagnostics.py``: smooth and legacy QI diagnostics, seed
  ranking metadata, mirror/elongation gates, and acceptance annotations
- ``vmec_jax/plotting.py``: VMEC-style geometry, ``|B|`` contour, Boozer-grid,
  objective-history, and publication-panel plotting helpers
- ``vmec_jax/wout.py``: minimal ``wout_*.nc`` reader for regression

The ``examples/`` folder contains user-facing scripts and curated parity demos.
Developer-only diagnostics and research utilities live under ``tools/``:

- ``tools/diagnostics/source_health.py``: report largest Python source files
  and optionally fail above a line-count threshold for staged refactor ratchets.
- ``tools/diagnostics/vmec2000_exec_stage_trace_compare.py``: per-iteration
  VMEC2000 vs vmec_jax parity comparator.
- ``tools/diagnostics/qh_vmec_vs_vmecjax.py``: QH comparison figures.
- ``tools/diagnostics/readme_fsq_trace.py``: README fsq_total traces.

For most scripts, prefer ``import vmec_jax as vj`` or ``import vmec_jax.api as
vj``.  Import lower-level modules directly only when developing kernels,
validation tools, or tests that need implementation-specific behavior.

Refactoring direction
---------------------

The current code intentionally preserves VMEC2000 semantics, but several
translation-era modules are now too large for long-term research development.
Use ``plan_differentiability.md`` as the source of truth for the staged
refactor.  Before starting a large extraction, run:

.. code-block:: bash

   python tools/diagnostics/source_health.py --top 30

The diagnostic is report-only by default.  A future refactor PR can ratchet it
with ``--fail-lines`` once a target module has been split and the compatibility
tests are green.
