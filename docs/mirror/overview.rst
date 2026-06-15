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
- focused tests for node ordering, polynomial exactness, interpolation, filtering,
  theta orthogonality, and analytic axisymmetric geometry.

Later phases add the divergence-free contravariant field representation,
variational energy and residuals, fixed-boundary solves, mirror native ``mout``
output, plotting, WHAM-inspired validation, nonaxisymmetric boundaries, mirror
straight-field-line diagnostics, and optimization workflows.
