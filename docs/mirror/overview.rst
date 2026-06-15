Mirror Geometry Overview
========================

The mirror lane targets fixed-boundary, open-ended ideal-MHD equilibria in
coordinates ``(s, theta, xi)``.  The axial coordinate ``xi`` is nonperiodic and
uses Chebyshev-Gauss-Lobatto nodes in increasing physical order; ``theta`` is
periodic and uses a real Fourier representation.  This is intentionally not a
large-aspect-ratio torus and does not store results as classic ``wout`` files.

The first committed phase provides only the numerical scaffolding needed by the
future solver:

- ``vmec_jax.mirror`` as a domain package;
- static mirror resolution/configuration objects;
- Chebyshev-Gauss-Lobatto nodes, differentiation matrices, interpolation, modal
  filtering, and Clenshaw-Curtis quadrature;
- uniform-theta Fourier grids, derivatives, and quadrature;
- axisymmetric fixed side boundaries, state projection, metric and Jacobian
  kernels for straight-axis cylinder/flared tubes;
- scalar radial profiles, contravariant/covariant/cartesian magnetic-field
  kernels, and magnetic/pressure energy integrals;
- differentiable axisymmetric energy wrappers, projected residuals, and
  manufactured-solution source helpers;
- an experimental fixed-boundary axisymmetric projected-gradient solve path
  with pressure-continuation trace diagnostics;
- mirror-native ``mout_*.nc`` read/write helpers, plot-data extraction, PNG
  writing, ``.npz``/CSV export helpers, and ``vmec --plot mout_*.nc`` dispatch;
- focused tests for node ordering, polynomial exactness, interpolation, filtering,
  theta orthogonality, analytic axisymmetric geometry, field identities, and
  analytic energy, gradient checks, Hessian symmetry, MMS stationarity, I/O
  roundtrip, and plotting numerical content.

Later phases add WHAM-inspired validation, nonaxisymmetric boundaries, mirror
straight-field-line diagnostics, and optimization workflows.
