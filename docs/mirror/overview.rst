Mirror Geometry Overview
========================

The mirror lane targets fixed-boundary, open-ended ideal-MHD equilibria in
coordinates ``(s, theta, xi)``.  The axial coordinate ``xi`` is nonperiodic and
uses Chebyshev-Gauss-Lobatto nodes in increasing physical order; ``theta`` is
periodic and uses a real Fourier representation.  This is intentionally not a
large-aspect-ratio torus and does not store results as classic ``wout`` files.

The current mirror package provides fixed-boundary axisymmetric and first
theta-dependent paths, plus validation surfaces needed to grow the backend
without coupling it to toroidal VMEC assumptions:

- ``vmec_jax.mirror`` as a domain package;
- static mirror resolution/configuration objects;
- Chebyshev-Gauss-Lobatto nodes, differentiation matrices, interpolation, modal
  filtering, and Clenshaw-Curtis quadrature;
- uniform-theta Fourier grids, derivatives, and quadrature;
- fixed side boundaries, state projection, metric, Jacobian, field, energy, and
  residual kernels for straight-axis cylinder/flared tubes and first
  theta-dependent cylindrical-radius surfaces;
- scalar radial profiles, contravariant/covariant/cartesian magnetic-field
  kernels, and magnetic/pressure energy integrals;
- differentiable axisymmetric energy wrappers, projected residuals,
  reduced-coordinate residual/Jacobian utilities for implicit-differentiation
  work, dense reduced linear solves for tiny forward/adjoint validation grids,
  manufactured-solution source helpers, and gradient/Hessian validation gates;
- fixed-boundary projected-gradient, scaled L-BFGS-B, dense/block-dense
  residual-Newton reference solves, and matrix-free residual-Newton diagnostics
  with pressure-continuation trace diagnostics;
- a VMEC-like reduced-coordinate residual preconditioner, adaptive inner
  linear-solve budgets, and split radius/lambda block-LSMR controls for
  residual Newton;
- mirror-native ``mout_*.nc`` read/write helpers, plot-data extraction, PNG
  writing, 3D boundary field-vector plots, cap-to-cap field-line overlays,
  radial beta/open-field-pitch/well-proxy diagnostics, mirror-Boozer-like
  Jacobian-weighted surface-average and pitch-proxy diagnostics, ``.npz``/CSV
  export helpers, and ``vmec --plot mout_*.nc`` dispatch;
- WHAM-inspired circular-loop fixture metadata, deterministic vacuum-field
  reference checks, optional ``magpylib`` comparison hooks, and low-resolution
  runnable axisymmetric/nonaxisymmetric examples;
- a free-boundary mirror diagnostic bridge with ESSOS-compatible circular-loop coil
  parameters, direct-coil external-field sampling on the mirror axis/boundary,
  reusable JSON setup export, initial fixed-boundary flux-tube construction
  from sampled on-axis fields, and optional low-resolution fixed-boundary
  baseline outputs plus side-boundary normal-field and total-pressure
  imbalance diagnostics and a damped, cap-tapered axisymmetric radius-update
  proposal for the planned 1%, 3%, and 10% circular coil studies, including
  optional low-resolution pilot steps that apply the proposal and report
  actual before/after diagnostics with a combined pressure/normal-field merit,
  the corresponding normalized LCFS residual vector for future coupled solves,
  a combined equilibrium-plus-LCFS residual assembly helper for least-squares
  prototypes, and a finite-difference, line-searched least-squares boundary
  coefficient step, a reduced residual-vector least-squares step that can use
  either finite-difference or JAX Jacobians for differentiable prototypes, a
  reduced residual-vector nonlinear least-squares solve loop with explicit
  target/rejection/stagnation/max-step stop reasons and per-step Jacobian
  rank, nullity, condition, singular-value, selected-JAX-mode, and
  predicted/actual reduction diagnostics, optional adaptive ridge-candidate
  selection, plus a reusable
  state/callback guarded least-squares loop and guarded realized fixed-boundary
  trial loop for the first true coupled-solve iterations,
  and normal-field-aware candidate selection between local, shape-preserving
  scale, normal-field-slope, mixed scale/normal-field, no-op, and realized
  coupled trial-scoring updates, top-level free-boundary status values that can
  distinguish target-merit convergence from non-converged pilot or coupled-loop
  runs, a low-resolution target-merit smoke that converges the default 1%, 3%,
  and 10% beta rows to ``target_merit=0.1`` with ``baseline_maxiter=5``, a
  ``0.05`` relative boundary step cap, an explicit ``fsq`` growth guard, and a
  two-step reduced residual-vector inner-solve fallback, ordered
  polynomial-degree candidates with selected-degree and attempt-summary
  diagnostics, recorded LS polynomial degree and ridge-candidate diagnostics
  with safe rejection of nonpositive high-order trial boundaries, plus an optional strict
  normal-field guard
  that records allowed strategies,
  no-op rejection reasons, workflow status, requested beta-scan points, and
  aggregate pilot counts and stop reasons in JSON output, with optional
  target-merit/stagnation stop criteria, an optional fixed-boundary ``fsq``
  growth guard, a compact ``mirror_free_boundary_circular_coil_beta_scan`` JSON
  schema version that names top-level, beta-row, and pilot-row contract fields
  for ESSOS handoff scripts, explicit pilot ``fsq`` growth-ratio and
  last-accepted-state diagnostics, a compact baseline/last-accepted/final-trial
  CSV report, and a cross-beta LCFS/final-``fsq`` metrics summary plot;
- a repo-root ``examples/mirror_two_coil_axisym.py`` analytic benchmark that
  builds a fixed boundary from the closed-form on-axis field of two circular
  coils, overlays mirror ``B_z`` against that reference, draws the coils, and
  runs on-axis/off-axis convergence checks;
- repo-root residual-Newton, solver-comparison, manufactured fixed-boundary,
  finite-current pitch, implicit-sensitivity, and fixed-boundary
  solve-diagnostic examples with standard mirror plot bundles or targeted
  validation figures;
- a first repo-root straight-axis hybrid fixed-boundary support fixture that
  uses a central rotating elliptical cross-section, smooth tapering into
  circular mirror end sections, standard geometry/field plots, and explicit
  metrics labels pointing to the toroidal hybrid lane as the final target;
- a repo-root toroidal stellarator-mirror hybrid input fixture that writes
  ordinary VMEC ``RBC``/``ZBS`` boundary coefficients, with mirror-like side
  arcs, localized rotating-ellipse stellarator corners, side/corner
  orientation diagnostics, convergence-grid reports, and low-resolution
  VMEC2000 parity diagnostics;
- focused tests for node ordering, polynomial exactness, interpolation, filtering,
  theta orthogonality, analytic axisymmetric geometry, field identities, and
  analytic energy, gradient checks, Hessian symmetry, MMS stationarity, I/O
  roundtrip, plotting numerical content, ``vmec --plot`` dispatch, WHAM fixture
  parity, and example smoke coverage.

The fast CLI/example path may use NumPy, SciPy, and Matplotlib when that keeps
runtime and memory use low.  Research-grade differentiable APIs should stay in
JAX kernels and use implicit/root or custom linear-solve differentiation rather
than differentiating through long host-side solver loops.
For the current free-boundary bridge, that means finite-difference Jacobians are
the documented default for host-side CLI loops that call fixed-boundary trial
solves or reporting callbacks, while reduced pure-JAX residual-vector
prototypes should use ``jacobian_backend="jax"`` with ``jax_mode="auto"``.
Automatic mode uses forward differentiation when the number of boundary
parameters is no larger than the residual-vector length, and reverse
differentiation for smaller residual or scalar-like targets.  The
``examples/mirror_free_boundary_vector_ls_benchmark.py`` example records the
backend comparison, selected mode, rank/nullity, conditioning, singular values,
selected ridge, and predicted-versus-actual residual reduction used to keep
that guidance tested.

Current solver status:

- ``dense_lstsq`` and ``block_dense_lstsq`` are small-to-moderate-grid
  correctness references.  They reach tight residuals on the current
  finite-current two-coil benchmark rows, including ``ns=9``, ``nxi=17``.
- Matrix-free ``lsmr``/``lsqr``/``block_lsmr`` paths are diagnostic scalable
  paths.  They expose useful residual, condition, and dense-step comparison
  metrics, but the moderate finite-current row remains lambda dominated and is
  not yet a tight-convergence production claim.
- Reduced-coordinate implicit sensitivity tests currently use tiny dense
  validation grids with a manufactured reduced source and small state ridge.
  These tests validate the residual/Jacobian/linear-solve differentiation
  machinery; they are not yet a production differentiable equilibrium API.
- Open-field pitch diagnostics measure cap-to-cap field-line advance and turns.
  They should not be interpreted as toroidal rotational transform.
- Toroidal hybrid VMEC/JAX versus VMEC2000 parity rows currently compare
  solved outcomes from the same generated ``input.*`` file.  The convergence
  CSV/JSON rows label the VMEC/JAX and VMEC2000 initialization policies, the
  VMEC/JAX axis-initialization branch, and two residual-start diagnostics:
  ``direct_initial_*`` values are VMEC/JAX residual scalars evaluated on the
  pre-iteration initial state, while ``initial_*`` values are the first stored
  VMEC/JAX solve-history row.  VMEC2000 comparisons use the first parsed
  ``threed1`` row.  A low-resolution audit showed the VMEC/JAX
  boundary-inferred direct initial residual agrees with the VMEC2000 first row
  to within plotting/diagnostic precision, whereas the raw-axis parity branch
  is a different, deliberately stricter initialization.  Use ``--nstep 1`` and
  ``--full-solver-diagnostics`` and ``--no-cli-finish`` in the toroidal-hybrid
  convergence example when comparing full VMEC-style iteration trajectories and
  solver step controls; accelerated scan rows report scan time-step histories
  when terminal step-status histories are not produced.  The same convergence
  rows also record CLI finish budgets, finish residuals, finish modes, and
  fallback flags so fast-path final residuals can be separated from raw
  fixed-iteration trajectories.  They also record target and fitted side/corner
  orientation spans, covariance anisotropy ranges, and valid-axis fractions so
  low-mode boundary fits can be audited before interpreting residual trends.
  ``--resolution-preset target`` writes the current target
  ladder, ``ns = 7,9,15`` and ``mpol:ntor = 5:20,6:24``.  Office GPU runs of
  that ladder first reached total-``fsq`` convergence at ``ftol=1e-8`` for all
  six rows with VMEC2000 outputs present.  Rows and aggregate reports record
  largest residual-component names, component values divided by requested
  ``ftol``, strict-component pass counts, and strict bottleneck counts.  A
  targeted 160-iteration office closure run then strict-converged all six
  target rows in 124-134 iterations, with the largest VMEC/JAX residual
  component below ``0.98`` times requested ``ftol``.  ``--case-filter`` accepts
  comma-separated shell patterns for splitting that target campaign into
  smaller row subsets.
  ``--aggregate-json`` reads one or more existing convergence JSON files from
  split campaigns and writes a de-duplicated aggregate CSV/JSON plus optional
  plots, which keeps remote target-ladder evidence compact and avoids copying
  generated WOUT or ``threed1`` trees into the repository.

Later phases finish production differentiable optimization APIs, production
free-boundary LCFS solves, and broader toroidal stellarator-mirror hybrid
convergence studies.
The current ESSOS-compatible circular-coil beta scan remains diagnostic/pilot
evidence. The toroidal square-coil stellarator-mirror hybrid lane now runs real
direct-coil VMEC free-boundary solves and plots solved states, but remains a
convergence-diagnostic path until fresh final force residuals meet the requested
``FTOL``. The current target is ``FTOL=1e-12`` with explicit staged
``NS_ARRAY``/``NITER_ARRAY``/``FTOL_ARRAY`` controls and VMEC-compatible
negative ``PHIEDGE`` for the default positive-current square-coil orientation.
Sign-corrected generated-``mgrid`` profiling now shows ``vmec_jax`` and VMEC2000
agree on the widened ``DELT=0.02`` deck through 10000 iterations: ``vmec_jax``
reaches about ``1.30e-7`` total residual and VMEC2000 reaches about ``1.11e-7``.
Both remain well above ``1e-12`` on the present square-coil setup.
Initial-boundary provider parity is good
on the widened deck: generated mgrid and exact direct Biot-Savart sampling
agree to about ``3.2e-4`` RMS relative field-vector error and ``1.5e-3`` RMS
relative coil-only ``B.n`` error, so the direct-coil blocker is nonlinear solve
closure rather than a simple field-convention mismatch. A matching direct-coil
run improves from about ``4.1e-4`` residual at 1000 iterations to ``4.7e-6`` at
3000 iterations, ``1.35e-6`` at 5000 iterations, and ``1.88e-7`` at 10000
iterations, essentially matching the generated-mgrid/VMEC2000 floor at the same
budget. A 25000-iteration direct-coil extension did not reach ``1e-12``: its
best fresh residual was about ``1.07e-7`` near iteration 11140, while the final
fresh recompute was about ``4.18e-7``. This points to low-resolution nonlinear
cycling rather than a plain iteration-budget miss. The same profiling shows that
underresolved ``NZETA`` can fail before useful force iterations; the
square-coil path now records ``recommended_nzeta`` and guards production-style
example runs against known-underresolved toroidal grids. Finite-beta promotion
should be based on VMEC force residuals, total-pressure balance, and
plasma-field/virtual-casing diagnostics rather than coil-only ``B.n``. Older
coarse square-coil scans have strict active free-boundary convergence evidence
through beta ``5%`` at ``FTOL=1e-8``; beta ``7%`` is the first high-beta stall
for that coarse configuration. The square-coil lane now also has a native
direct-coil-to-``mgrid`` writer, a direct/mgrid/VMEC2000 backend profiler, and a
low-bandwidth rounded ``axis_kind="spline"`` square-axis option to reduce
``NTOR`` sensitivity before VMEC Fourier projection. The source helper and
profiler now record ``boundary_projection`` truncation errors for the selected
``MPOL``/``NTOR`` grid, so mode changes can be separated from nonlinear
free-boundary convergence. On the current square-coil shape, the spline envelope
cuts max component projection error from about ``3.2e-4`` to ``1.3e-4`` at
``MPOL=5, NTOR=12``. Direct-coil convergence
candidates are gated by a fresh residual recompute using the current
plasma-current normalization, and the square-coil example records near-axis
``|B|`` and mirror-ratio response plots for comparison with the expected
finite-beta diamagnetic field-depression and effective mirror-ratio increase in
linear mirror traps.
