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
- ``vmec_jax/solvers/fixed_boundary/residual/force_norms.py``: force-block weighting, lambda
  residual norms, VMEC residual ``FSQ`` scalar assembly, and stability-guard
  timestep helpers extracted from the residual iteration hot path
- ``vmec_jax/solvers/fixed_boundary/residual/payload_blocks.py``: residual-force payload
  containers, edge masking, staged VMEC ``m=1``/zero/scalxc transforms, and
  preconditioner-output block assembly
- ``vmec_jax/solvers/fixed_boundary/optimization/tolerances.py``: dtype-aware gradient, conjugate
  gradient, and Levenberg-Marquardt tolerance policies
- ``vmec_jax/solvers/fixed_boundary/optimization/constraints.py``: fixed-boundary edge constraints,
  magnetic-axis regularity, lambda-gauge projection, and related NumPy/JAX
  coefficient-slice helpers
- ``vmec_jax/solvers/fixed_boundary/optimization/gradient.py``: state gradient-descent updates and
  feasible-gradient projections for fixed-boundary/axis/lambda constraints
- ``vmec_jax/solvers/fixed_boundary/preconditioning/operators.py``:
  fixed-boundary mode-diagonal
  and radial Dirichlet smoothing preconditioner kernels, tridiagonal policy
  resolution, metric preconditioner scales and bcovar wrapper helpers, radial
  mesh scale factors, and VMEC ``m=1`` preconditioner scaling helpers
- ``vmec_jax/solve_jit_cache_helpers.py``: environment-controlled JIT-cache
  limits, structural cache keys, LRU helpers, and scan-cache miss diagnostics
- ``vmec_jax/solvers/fixed_boundary/preconditioning/payload.py``:
  cached strict-update,
  preconditioner-output, fused preconditioner-apply, accepted-control, and
  ``ptau`` JIT payload helpers used by the residual-iteration hot path
- ``vmec_jax/solvers/fixed_boundary/diagnostics/first_step.py``: first-step VMEC residual,
  preconditioner, force-channel, and update diagnostic assembly used by the
  public ``solve.first_step_diagnostics`` wrapper
- ``vmec_jax/solvers/fixed_boundary/optimization/lambda_gd.py``: lambda-only fixed-geometry magnetic
  energy optimizer used by the public ``solve.solve_lambda_gd`` wrapper while
  preserving historical solve-module monkeypatch hooks
- ``vmec_jax/solvers/fixed_boundary/optimization/energy.py``: shared fixed-boundary
  magnetic-energy context/evaluator setup for GD and L-BFGS optimizers, with
  solve-module dependency injection for historical monkeypatch compatibility
- ``vmec_jax/solvers/fixed_boundary/optimization/gd.py``: fixed-boundary
  gradient-descent optimizer loop used by the public
  ``solve.solve_fixed_boundary_gd`` wrapper
- ``vmec_jax/solvers/fixed_boundary/optimization/lbfgs.py``: fixed-boundary
  L-BFGS optimizer loop used by the public
  ``solve.solve_fixed_boundary_lbfgs`` wrapper
- ``vmec_jax/solvers/fixed_boundary/optimization/residual_context.py``: shared VMEC flux/profile,
  force-kernel ``wout``-like context, trig-table, and fixed-edge setup used by
  residual-objective L-BFGS and Gauss-Newton optimizers
- ``vmec_jax/solvers/fixed_boundary/optimization/residual_lbfgs.py``:
  fixed-boundary VMEC-style force-residual L-BFGS optimizer loop used by the
  public ``solve.solve_fixed_boundary_lbfgs_vmec_residual`` wrapper
- ``vmec_jax/solvers/fixed_boundary/optimization/residual_gn.py``:
  fixed-boundary VMEC-style force-residual Gauss-Newton/CG optimizer loop used
  by the public ``solve.solve_fixed_boundary_gn_vmec_residual`` wrapper
- ``vmec_jax/solvers/fixed_boundary/diagnostics/hlo.py``: optional JAX HLO lowering dump
  helpers for solver-kernel diagnostics
- ``vmec_jax/solvers/fixed_boundary/diagnostics/axis_reset.py``: initial magnetic-axis reset
  control decisions, axis-state merging, and optional axis coefficient dumps
- ``vmec_jax/solvers/fixed_boundary/residual/update.py``:
  residual-iteration
  velocity-block containers, host momentum update, and generic velocity
  zeroing/scaling helpers
- ``vmec_jax/solvers/free_boundary/control.py``: free-boundary cadence,
  turn-on, and constraint-baseline control helpers
- ``vmec_jax/solvers/free_boundary/diagnostics.py``: solve-facing
  free-boundary external-field diagnostic adapters
- ``vmec_jax/solvers/fixed_boundary/diagnostics/force.py``: optional force-channel,
  TOMNSP, scalar residual, and force-kernel debug dump helpers
- ``vmec_jax/solvers/fixed_boundary/diagnostics/bsub.py``: optional covariant-field debug
  dumps for scaled full-mesh, half-mesh, and radial ``B_s`` diagnostics
- ``vmec_jax/solvers/fixed_boundary/diagnostics/lambda_debug.py``: optional lambda residual,
  lambda-preconditioner, lambda-derivative, and radial-preconditioner debug
  dump helpers
- ``vmec_jax/solvers/fixed_boundary/diagnostics/metric.py``: optional metric,
  preconditioner-input, and VMEC internal state-vector debug dump helpers
- ``vmec_jax/solve_result_types.py``: solver result dataclasses, scan carry
  containers, and ``wout``-like force-kernel PyTree containers shared by solve,
  driver, implicit differentiation, and tests
- ``vmec_jax/solvers/fixed_boundary/scan/resume.py``: VMEC2000-style scan
  resume-state initialization and carry-field restoration
- ``vmec_jax/solvers/fixed_boundary/residual/runtime.py``:
  residual-iteration
  runtime seams for scan readiness, optional debug printing, timing reports,
  resume-state summaries, and free-boundary external-field diagnostic
  attachment
- ``vmec_jax/solvers/fixed_boundary/residual/mode_transform.py``: host DGEMM
  projection matrices, NumPy ``scalxc`` setup, and mode-diagonal weights used
  by the residual-iteration host update path
- ``vmec_jax/solvers/fixed_boundary/residual/setup.py``:
  VMEC-grid reuse checks,
  free-boundary provider policy, scan-disablement, and CPU/GPU strict-update
  setup decisions for residual iteration
- ``vmec_jax/solvers/fixed_boundary/residual/finalize.py``: final timing
  diagnostics, resume-state payload packing, and residual-iteration result
  assembly
- ``vmec_jax/solvers/fixed_boundary/residual/force_cache.py``: structural force
  JIT cache keys and callable selection while preserving solver-owned cache
  objects
- ``vmec_jax/solvers/fixed_boundary/residual/force_payload.py``:
  residual-iteration force-payload mask-pack selection, edge masking, Z-force
  square-sum diagnostics, NaN preservation, and VMEC scalar force-norm
  assembly seams
- ``vmec_jax/solve.py``: fixed-boundary solvers + VMEC2000 iteration loop
- ``vmec_jax/driver.py``: CLI-facing fixed/free-boundary drivers, output
  policies, staged solve dispatch, and wout writing
- ``vmec_jax/driver_policy_helpers.py``: backend-aware driver policy,
  residual-convergence, staged-budget, and resume-state helper functions kept
  outside the CLI-facing workflow while preserving driver compatibility aliases
- ``vmec_jax/driver_result_helpers.py``: staged/chunked solver-result merging,
  timing aggregation, final-force payload propagation, and VMEC history
  comparison helpers shared by driver tests and runtime finish policy
- ``vmec_jax/driver_solve_helpers.py``: lightweight fixed-boundary solve entry
  helpers used by optimization scripts, with ``driver.py`` injecting the
  historical initial-guess and solver callables
- ``vmec_jax/driver_flux_helpers.py``: post-solve current-driven
  flux/profile reconciliation helpers, with ``driver.py`` retaining a small
  wrapper for historical monkeypatch hooks
- ``vmec_jax/driver_io_helpers.py``: bundled example path resolution,
  lightweight input/wout loaders, and NumPy archive writing helpers, with
  ``driver.py`` injecting historical monkeypatch hooks
- ``vmec_jax/driver_output_helpers.py``: VMEC-style residual scalar
  reconstruction and fixed-boundary ``wout`` construction helpers, with
  ``driver.py`` retaining wrappers for downstream monkeypatch compatibility
- ``vmec_jax/free_boundary.py``: mgrid loading, NESTOR-like vacuum coupling,
  and free-boundary runtime state helpers
- ``vmec_jax/free_boundary_adjoint_trace_controls.py``: accepted-trace
  reset/status/controller-mask helpers used by fingerprint-gated
  free-boundary adjoint replay reports
- ``vmec_jax/free_boundary_adjoint_trace_metadata.py``: accepted-trace
  replay-graph shape compaction, segment-summary trimming, and strict
  JSON-safe fingerprint conversion helpers used by branch-local adjoint
  diagnostics
- ``vmec_jax/free_boundary_adjoint_trace_fingerprint.py``: accepted-trace
  branch fingerprints and JSON-safe fingerprint-delta summaries used to reject
  changed adaptive branches before comparing branch-local AD/JVP results with
  complete-solve finite differences
- ``vmec_jax/optimization.py``: exact fixed-boundary optimizer, boundary DOF
  maps, accepted-point replay, and discrete-adjoint Jacobian plumbing
- ``vmec_jax/optimization_workflow.py``: user-facing optimization problem
  objects, objective tuples, continuation stages, and example workflow helpers
- ``vmec_jax/qi_diagnostics.py``: smooth and legacy QI diagnostics, seed
  ranking metadata, mirror/elongation gates, and acceptance annotations
- ``vmec_jax/plotting.py``: VMEC-style geometry, ``|B|`` contour, Boozer-grid,
  objective-history, and publication-panel plotting helpers
- ``vmec_jax/wout.py``: minimal ``wout_*.nc`` reader/writer and VMEC-style
  output synthesis compatibility surface
- ``vmec_jax/wout_diagnostics.py``: persisted-WOUT diagnostic reconstruction
  helpers, including the fallback ``DMerc``/Glasser ``D_R`` algebra used when
  older ``wout`` files do not contain the newer stability fields

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
