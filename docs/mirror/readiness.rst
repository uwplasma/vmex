Mirror Readiness Matrix
=======================

This page states what the mirror lane currently supports, what is a validated
prototype, and what remains a diagnostic or deferred path.  It is meant to keep
review scope clear while the package grows toward production mirror equilibria.

Status Labels
-------------

``supported``
   Covered by public APIs, examples, and focused tests.  These paths are
   suitable for research workflows at the documented resolutions and with the
   documented assumptions.

``validated prototype``
   Numerically tested building block for the next production path.  The API and
   diagnostics are useful, but the scope is deliberately narrower than a full
   equilibrium workflow.

``diagnostic``
   A plotting, benchmarking, or solver-study fixture.  These paths are meant to
   expose residuals, conditioning, and geometry behavior without claiming final
   production convergence.

``deferred``
   Planned work that is intentionally outside the current PR claim.

Current Scope
-------------

.. list-table::
   :header-rows: 1
   :widths: 24 20 56

   * - Lane
     - Status
     - Current claim
   * - Straight-axis mirror grids and bases
     - supported
     - Chebyshev-Gauss-Lobatto ``xi`` nodes, Fourier ``theta`` nodes, radial
       grids, quadrature, interpolation, filtering, and derivative checks.
   * - Fixed-boundary axisymmetric mirrors
     - supported
     - Scalar-pressure fixed-boundary solves, residual diagnostics, ``fsq`` and
       normalized-force reporting, mirror-native output, fixed-boundary
       diagnostic examples, solver-comparison reports, and standard plots.
   * - Theta-dependent fixed-boundary surfaces
     - validated prototype
     - Geometry, field, I/O, plotting, and solver stress tests are present.
       Full 3-D production convergence studies remain narrower than the
       axisymmetric path.
   * - Residual-Newton and preconditioning
     - validated prototype
     - Dense and block-dense paths are correctness references on small grids.
       Matrix-free Krylov paths expose scalable diagnostics and VMEC-like
       preconditioning, but are not yet the broad production default.
   * - Two-coil analytic validation
     - supported
     - The root example compares on-axis and low-radius off-axis fields against
       circular-loop Biot-Savart formulas, records convergence evidence, and
       writes checked coil/field/``|B|`` plots.
   * - Finite-current pitch examples
     - supported diagnostic
     - Field-line pitch is visible and quantified as open-field cap-to-cap
       advance.  It is not a toroidal rotational transform.
   * - Mirror-Boozer-like diagnostics
     - supported diagnostic
     - Surface-average ``|B|``, ripple, mirror ratio, pitch proxies, and
       well-proxy profiles are exported.  No toroidal Boozer coordinate claim is
       made.
   * - Mirror ``mout`` files and ``vmec --plot``
     - supported
     - Mirror-native NetCDF output and CLI plotting write geometry, cross
       sections, fields, residual histories, radial diagnostics, and
       mirror-Boozer-like profiles.
   * - Differentiable fixed-boundary APIs
     - validated prototype
     - Reduced axisymmetric residual, Jacobian, linear-solve, forward
       sensitivity, adjoint, and custom-VJP wrappers are tested on tiny grids.
       Dense-vs-matrix-free benchmark plots are covered.  These are method
       gates, not a broad differentiable solved-equilibrium API.
   * - Free-boundary circular-coil bridge
     - diagnostic
     - ESSOS-compatible circular coils, external-field sampling, LCFS residual
       vectors, candidate updates, guarded pilot loops, and reduced
       residual-vector least-squares solves are available, with explicit
       Jacobian rank, nullity, conditioning, selected JAX mode, predicted
       reduction, actual reduction, and optional adaptive ridge-candidate
       diagnostics.  Plotted beta-scan and reduced-vector benchmark evidence
       are present.  This is not a converged production free-boundary
       equilibrium solver.
   * - ESSOS beta-scan fixture
     - diagnostic
     - The 1%, 3%, and 10% beta cases share a compact JSON/CSV schema with
       baseline, last-accepted, and final-trial pilot fields plus checked
       summary and per-beta diagnostic plots for handoff studies.
   * - Straight-axis stellarator-mirror support fixture
     - diagnostic
     - The rotating-ellipse straight-axis example is a support fixture for
       geometry and plotting.  It is not the final hybrid target.
   * - Toroidal stellarator-mirror hybrid
     - validated prototype
     - The repo-root example writes ordinary VMEC boundary coefficients with
       mirror-like side arcs and stellarator-like corners, plus convergence and
       VMEC2000 parity diagnostics.  The convergence runner has named smoke,
       promotion, and target ladders, split-campaign aggregation, scan
       diagnostics, and CLI finish reporting.  Office GPU/VMEC2000 evidence
       covers all six named target rows with total-``fsq`` convergence at
       ``ftol=1e-8``; strict component convergence remains a documented caveat.
   * - Anisotropic pressure, kinetic closures, sheath/end physics
     - deferred
     - These closures are outside the fixed-boundary scalar-pressure mirror
       scope.

Derivative Policy
-----------------

Fast CLI examples may use NumPy, SciPy, and finite differences when that keeps
runtime and memory small.  Differentiable research APIs should keep residuals
and derivative rules in JAX and use implicit, adjoint, forward, reverse, or
custom linear-solve differentiation according to the residual shape.  The
current rule is:

- use finite-difference Jacobians for host-side workflows that run
  fixed-boundary trial solves, write output files, or call plotting callbacks;
- use JAX forward mode when the number of parameters is no larger than the
  residual-vector length;
- use JAX reverse mode for smaller residual vectors or scalar-like targets;
- keep dense solves as the tiny-grid correctness reference before promoting a
  matrix-free or external linear-operator backend.

Reduced free-boundary least-squares rows record the selected derivative mode,
rank/nullity, condition number, singular values, selected ridge candidate, and
predicted versus realized residual reduction so rank-deficient or
over-aggressive boundary parameterizations are visible in the JSON before they
are coupled to expensive fixed-boundary trial solves.

Review Gate Before Undrafting
-----------------------------

Before the draft PR is ready for review, the current branch should satisfy:

1. the full local mirror test suite passes;
2. the Sphinx docs build with warnings as errors;
3. the draft PR body points to the latest implementation-log section;
4. generated example outputs remain under ignored ``results/`` paths or are
   deliberately compressed before being tracked;
5. GitHub checks have no failing jobs at the latest pushed head.
