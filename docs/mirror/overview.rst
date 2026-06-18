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
  manufactured-solution source helpers, and gradient/Hessian validation gates;
- fixed-boundary projected-gradient, scaled L-BFGS-B, dense/block-dense
  residual-Newton reference solves, and matrix-free residual-Newton diagnostics
  with pressure-continuation trace diagnostics;
- a VMEC-like reduced-coordinate residual preconditioner, adaptive inner
  linear-solve budgets, and split radius/lambda block-LSMR controls for
  residual Newton;
- mirror-native ``mout_*.nc`` read/write helpers, plot-data extraction, PNG
  writing, 3D boundary field-vector plots, cap-to-cap field-line overlays,
  radial beta/open-field-pitch/well-proxy diagnostics, ``.npz``/CSV export
  helpers, and ``vmec --plot mout_*.nc`` dispatch;
- WHAM-inspired circular-loop fixture metadata, deterministic vacuum-field
  reference checks, optional ``magpylib`` comparison hooks, and low-resolution
  runnable axisymmetric/nonaxisymmetric examples;
- a first free-boundary mirror bridge with ESSOS-compatible circular-loop coil
  parameters, direct-coil external-field sampling on the mirror axis/boundary,
  reusable JSON setup export, initial fixed-boundary flux-tube construction
  from sampled on-axis fields, and optional low-resolution fixed-boundary
  baseline outputs plus side-boundary normal-field and total-pressure
  imbalance diagnostics and a damped, cap-tapered axisymmetric radius-update
  proposal for the planned 1%, 3%, and 10% circular coil studies, including
  optional low-resolution pilot steps that apply the proposal and report
  actual before/after diagnostics with a combined pressure/normal-field merit
  and normal-field-aware candidate selection between local and shape-preserving
  scale updates, plus an optional strict normal-field guard with no-op
  fallback;
- a repo-root ``examples/mirror_two_coil_axisym.py`` analytic benchmark that
  builds a fixed boundary from the closed-form on-axis field of two circular
  coils, overlays mirror ``B_z`` against that reference, draws the coils, and
  runs on-axis/off-axis convergence checks;
- repo-root residual-Newton, solver-comparison, manufactured fixed-boundary,
  finite-current pitch, and fixed-boundary solve-diagnostic examples with
  standard mirror plot bundles;
- focused tests for node ordering, polynomial exactness, interpolation, filtering,
  theta orthogonality, analytic axisymmetric geometry, field identities, and
  analytic energy, gradient checks, Hessian symmetry, MMS stationarity, I/O
  roundtrip, plotting numerical content, ``vmec --plot`` dispatch, WHAM fixture
  parity, and example smoke coverage.

The fast CLI/example path may use NumPy, SciPy, and Matplotlib when that keeps
runtime and memory use low.  Research-grade differentiable APIs should stay in
JAX kernels and use implicit/root or custom linear-solve differentiation rather
than differentiating through long host-side solver loops.

Current solver status:

- ``dense_lstsq`` and ``block_dense_lstsq`` are small-to-moderate-grid
  correctness references.  They reach tight residuals on the current
  finite-current two-coil benchmark rows, including ``ns=9``, ``nxi=17``.
- Matrix-free ``lsmr``/``lsqr``/``block_lsmr`` paths are diagnostic scalable
  paths.  They expose useful residual, condition, and dense-step comparison
  metrics, but the moderate finite-current row remains lambda dominated and is
  not yet a tight-convergence production claim.
- Open-field pitch diagnostics measure cap-to-cap field-line advance and turns.
  They should not be interpreted as toroidal rotational transform.

Later phases finish differentiable optimization APIs, free-boundary LCFS
updates, stellarator-mirror hybrid boundaries, and ESSOS circular-coil beta
scan examples.
