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
- fixed-boundary projected-gradient, scaled L-BFGS-B, and matrix-free
  residual-Newton solve prototypes with pressure-continuation trace
  diagnostics;
- a VMEC-like reduced-coordinate residual preconditioner and an adaptive inner
  linear-solve budget policy for residual Newton;
- mirror-native ``mout_*.nc`` read/write helpers, plot-data extraction, PNG
  writing, 3D boundary field-vector and radial beta/twist/well-proxy diagnostics,
  ``.npz``/CSV export helpers, and ``vmec --plot mout_*.nc`` dispatch;
- WHAM-inspired circular-loop fixture metadata, deterministic vacuum-field
  reference checks, optional ``magpylib`` comparison hooks, and low-resolution
  runnable axisymmetric/nonaxisymmetric examples;
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

Later phases finish finite-current validation, mirror straight-field-line
diagnostics, differentiable optimization APIs, and free-boundary design.
