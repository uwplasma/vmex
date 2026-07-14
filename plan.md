# Final mirror-equilibrium implementation plan

Status: active implementation plan for PR #22.

Baseline: `codex/mirror-geometry` at `6a295b55`, based on `main` at
`ed4ac7ac`, reviewed 2026-07-14. This document replaces every older mirror
roadmap. Git history and compact benchmark JSON files are the execution log.

## 1. Mission and finish line

Deliver a small, fast, research-grade extension of `vmec_jax` for open magnetic
mirrors and a toroidal stellarator-mirror hybrid. The supported result must be
an actual nested-surface ideal-MHD equilibrium, not a prescribed tube drawn in
an external magnetic field.

The final supported models are:

1. straight-axis axisymmetric fixed-boundary mirror;
2. straight-axis nonaxisymmetric fixed-boundary mirror;
3. straight-axis axisymmetric free-boundary mirror;
4. straight-axis nonaxisymmetric free-boundary mirror;
5. a closed toroidal hybrid with two long straight mirror legs and two curved
   stellarator returns, represented and solved natively with B-splines;
6. isotropic and ANIMEC-consistent anisotropic pressure where the model remains
   elliptic;
7. host-controlled fast CLI solves and implicit derivatives of converged
   equilibria.

The branch is finished only when supported lanes pass analytic or independent
physics checks, resolution studies, convergence gates, derivative checks,
documented examples, and compact reproducible plots. A lane that cannot meet
those gates within its milestone is explicitly deferred and its experimental
API is deleted. It must not remain as a scaffold.

## 2. Non-negotiable architecture

### 2.1 Repository ownership

- **vmec_jax** owns equilibrium coordinates, MHD energies and residuals,
  boundary coupling, equilibrium continuation, equilibrium I/O, and plots of
  solved states.
- **ESSOS** owns coils, coil geometry, Biot--Savart, field-line integration in
  coil fields, and mgrid generation. A forward free-boundary solve accepts
  `MgridField`; a differentiable residual accepts an `xyz -> B` callable.
- **SOLVAX** owns reusable Krylov methods, structured direct solves, implicit
  root/linear solves, generic preconditioner primitives, and chunked AD.
- **SciPy** may remain the host driver for nondifferentiable CLI
  `minimize`/`least_squares`. SOLVAX does not currently replace robust nonlinear
  globalization, so adding a local nonlinear-optimizer framework would increase
  code without improving the model.

No new public coil, Biot--Savart, field-line tracer, generic GMRES/PCG, or
generic B-spline package is added to vmec_jax.

### 2.2 One equilibrium formulation

There is one mirror state, one geometry evaluation path, one isotropic energy,
one anisotropic energy, and one set of force diagnostics. Axisymmetry is the
`mpol=0` specialization of the same equations. Free boundary adds the vacuum
and interface residual; it does not copy the plasma equations. B-spline and
Chebyshev discretizations implement the same small axial-basis contract.

Do not copy DESC's branch-level `MirrorEquilibrium` and mirror-objective tree.
Do not create a second hybrid equilibrium under `vmec_jax/core`.

### 2.3 Native B-spline meaning

"Native B-spline" means that the axial/arc-length equilibrium unknowns and the
boundary/centerline controls are B-spline coefficients. Sampling B-spline
controls and projecting them to a global Fourier torus is not native support.

- Open mirrors use clamped cubic B-splines in the nonperiodic axial coordinate.
- Closed hybrids use periodic cubic B-splines in normalized arc length.
- The poloidal direction remains Fourier because cross-sections are periodic.
- The radial direction retains the VMEC-like staggered flux-surface mesh. A
  radial B-spline rewrite has no demonstrated physics or maintenance benefit
  and is outside this plan.
- Chebyshev--Gauss--Lobatto remains a reference discretization until B-spline
  parity and refinement are established, then becomes an internal validation
  option rather than the default public model.

This combination provides local control through straight-to-curved transitions
without discarding the established VMEC radial representation.

## 3. Current branch audit

The main branch is merged and PR #22 is mergeable. At this baseline all unit,
mirror, parity, implicit-gradient, console, and documentation jobs pass. The
example-smoke job fails because
`examples/mirror_fixed_boundary_gradients.py` imports `plot_mout` from the
package root after main removed that export; the function is correctly owned by
`vmec_jax.mirror`. This is Milestone 0, not a physics failure.

The branch currently adds about 23,170 lines and changes 149 files relative to
main. The mirror package alone is 8,088 lines in 20 modules. In addition:

- `vmec_jax/core/coils.py`: 885 lines, contrary to ESSOS ownership;
- `vmec_jax/core/hybrid.py`: 299 lines, builds a square/superellipse target and
  Fourier-projects it into ordinary VMEC boundary coefficients;
- `vmec_jax/core/hybrid_free_boundary.py`: 335 lines, repeats coil and mgrid
  orchestration;
- `tests/test_coils.py`: 403 lines for code that should live in ESSOS;
- two hybrid examples describe the experimental Fourier-projection lane;
- `vmec_jax/mirror/__init__.py` exports too many research internals.

### 3.1 Capability assessment

| Lane | Present evidence | Completion | Main blocker |
|---|---|---:|---|
| Axisymmetric fixed mirror | Real variational solve, `ftol=1e-12` examples, force diagnostics, implicit gradients | 90% | B-spline axial parity and final API cleanup |
| Axisymmetric free mirror | Coupled plasma/vacuum solve, beta continuation to 50%, interface and `B.n` diagnostics | 80% | independent refinement evidence and ESSOS-only input cleanup |
| Nonaxisymmetric fixed mirror | Theta-dependent state, forces, one refinement benchmark | 60% | no paraxial rotating-ellipse coefficient validation |
| Nonaxisymmetric free mirror | Theta-dependent BIE and coupled residual exist | 45% | no converged refinement/panel study tied to an analytic geometry |
| ANIMEC closure | Variational functional, consistent moments, isotropic limit, validity indicators | 75% | source-equation audit and independent finite-beta benchmark |
| Implicit derivatives | Fixed and axisymmetric free adjoints with FD checks | 75% | duplicate SciPy GMRES and incomplete nonaxisymmetric scaling evidence |
| Native B-spline straight mirror | none; current axial basis is Chebyshev | 0% | basis/state implementation |
| Native B-spline toroidal hybrid | none; current lane is Fourier projection | 0% | centerline/frame/state/residual formulation |
| ESSOS ownership migration | main established the contract, mirror branch still carries legacy coils | 25% | remaining consumers and tests |
| Source simplification | main refactors merged; mirror remains broad | 30% | delete legacy coil/hybrid paths and consolidate public API |

Percentages indicate evidence completed, not lines written.

### 3.2 Physics already worth keeping

The following implementation has a coherent research basis and should be
consolidated rather than replaced:

- coordinates `(s, theta, xi)`, with open `xi` endpoints;
- regular embedding `r = sqrt(s) a(s,theta,xi)`;
- VMEC-like contravariant representation
  `sqrt(g) B^theta = I'(s) - d(lambda)/dxi` and
  `sqrt(g) B^xi = Psi'(s) + d(lambda)/dtheta`, which makes discrete
  `div(B)=0` follow from commuting derivatives;
- mass-conserving isotropic VMEC energy;
- the ANIMEC energy and thermodynamic identity
  `p_perp = p_parallel - B (d p_parallel/dB)_s`;
- independent variational-force and continuum tensor-force diagnostics;
- the closed side-plus-end-cap surface and free-space boundary-integral vacuum;
- mirror-native `mout` I/O, horizontal 3-D plots, cross-sections, `|B|`, field
  lines, and convergence histories;
- host solve plus implicit differentiation of the converged residual.

## 4. External source and literature conclusions

### 4.1 VMEC, VMEC2000, and VMEC++

VMEC minimizes ideal-MHD energy on nested flux surfaces using a divergence-free
field representation and a preconditioned descent/Newton strategy. Those are
the correct inherited ideas. VMEC2000 cannot directly represent open axial
topology or a long straight/short-return B-spline torus; a very-large-aspect
ratio Fourier torus is useful only as a loose compatibility experiment. It is
not the truth model for the mirror lanes.

Use VMEC2000/VMEC++ for:

- toroidal regression and sign/normalization checks;
- finite-aspect-ratio limiting studies where both codes represent the same
  smooth closed surface;
- solver-history and resource comparisons.

Do not use VMEC2000 convergence to certify an open mirror or use an mgrid file
to hide a geometry mismatch.

### 4.2 DESC branch review

The local clone was fetched and the following remote branches were inspected:
`mirror`, `mirror_anisotropy`, `finite_element_basis`, and
`finite_element_basis_alan`.

The useful ideas are:

- a nonperiodic coordinate needs explicit basis, grid, transform, endpoint,
  and resolution-change semantics;
- `ChebyshevFourierSeries` and `ChebyshevRZToroidalSurface` cleanly separate a
  nonperiodic Chebyshev coordinate from periodic poloidal Fourier modes;
- analytic straight-coordinate fixtures in `tests/ana_straight*.py` are useful
  references for manufactured tests;
- objective normalization and boundary-condition ownership must be explicit;
- continuation should be able to apply shaping and pressure in either order.

The branches are not suitable implementation templates. Relative to DESC
master, `mirror` adds roughly 11,000 source/test lines and
`mirror_anisotropy` roughly 14,700. They add a 2,255-line copied
`mirror_equilibrium.py`, a 1,393-line mirror objective module, large analytic
scripts/notebooks, and rename or disable substantial parts of the normal test
tree. Their straight solve tests use tolerances around `1e-6`. The
`finite_element_basis` branches implement broad finite-element mesh machinery,
not a small production B-spline mirror state. The lesson is to borrow equations,
basis identities, and fixtures, while retaining vmec_jax's smaller mirror
package and normal CI.

### 4.3 Paraxial mirrors and the rotating ellipse

Appendix C of Rodriguez, Helander, and Goodman (JPP 90, 2024) makes the
near-axis/straight-mirror equivalence explicit. For a straight axis it supplies
the decisive low-radius gates:

- flux conservation
  `X1c Y1s - X1s Y1c = Bbar/B0(z)`;
- no first-order poloidal `|B|` variation, `B1c = B1s = 0`;
- leading transverse variation is quadrupolar,
  `B = B0 + r^2 (B20 + B2c cos(2 alpha) + B2s sin(2 alpha)) + O(r^3)`;
- the sigma/Riccati relation determines the evolution and orientation of the
  leading ellipse.

This is the analytic target for an ellipse whose principal axis rotates by
90 degrees. Poloidal `|B|` variation in a finite-radius numerical equilibrium is
not automatically a bug: an `m=1` term at first order is a bug, while an
`m=2`, order-`r^2` quadrupole is expected. The test must fit radial order and
poloidal mode rather than inspect one LCFS plot.

The Straight Field Line Mirror (SFLM) literature is a second, independent
fixture. Its paraxial Clebsch construction gives nonparallel but straight field
lines and analytic elliptical flux-tube cross-sections. It should not be
conflated with the rotating-ellipse fixture: SFLM's principal axes need not
rotate. Implement both fixtures and use them for different assertions.

Pearlstein's quadrupole-symmetric tandem-mirror report and Goodman, Freidberg,
and Lane's long-thin finite-beta expansion add the next-order validation: finite
beta changes quadrupolar surfaces and can drive diamond-like distortion with a
predictable beta/long-thin scaling. This is the physics benchmark for the
nonaxisymmetric finite-beta scan, not a visual expectation that every radius
must change dramatically by beta=10%.

### 4.4 ANIMEC

Cooper et al. define the anisotropic free-boundary model through

`W = integral [B^2/(2 mu0) + p_parallel/(Gamma-1)] d^3x`,

with total-pressure continuity
`[p_perp + B^2/(2 mu0)] = 0` at the plasma-vacuum interface and a Neumann vacuum
condition. The current mirror formulation follows these equations and derives
`p_perp`; it does not prescribe both moments independently. Suzuki/Asahi's
bi-Maxwellian model is an appropriate first closure.

Promotion still requires:

- equation-by-equation checks of the hot-particle form factor on both sides of
  its critical field;
- isotropic and zero-hot-fraction limits;
- AD versus finite-difference verification of
  `p_perp = p_parallel - B dp_parallel/dB`;
- positivity of the firehose coefficient and mirror ellipticity throughout
  every solved state;
- anisotropic interface pressure balance;
- a resolution-stable finite-beta observable, not only solver termination.

An anisotropic run that violates ellipticity is rejected, not plotted as an
equilibrium.

### 4.5 Differentiation and SOLVAX

The correct derivative of a converged equilibrium is implicit. If
`F(u,p)=0`, tangent and adjoint solves use `F_u` and `F_u^T`; they do not
differentiate through thousands of host iterations. JAX's custom derivative and
`custom_linear_solve` documentation supports this design. The 2025 fast
automated spectral-adjoint paper reinforces the same resource principle:
differentiate the sparse/spectral operator graph and solve an adjoint system,
instead of storing the nonlinear trajectory.

Use:

- forward-mode JVP for few input directions or state-linearization tests;
- reverse-mode implicit adjoints for scalar objectives with many controls;
- batched/chunked Jacobians only for small dense interface blocks;
- finite differences only as a frozen-path validation oracle;
- direct differentiation through the solver only in tiny tests.

SOLVAX 0.7 supplies `gmres`, `gcrot`, `pcg`, block Thomas, banded and periodic
banded LU, `root_solve`, `linear_solve`, preconditioner primitives, and
`chunked_jacfwd/jacrev/jacobian`. The toroidal core already uses these. Mirror
physics-specific state packing and separable preconditioning remain here;
generic linear iteration and chunking move to SOLVAX after parity tests.

## 5. Numerical contracts

### 5.1 Residual and convergence

`ftol` always means the normalized variational force norm, not SciPy's step or
cost termination flag. Every result records:

- normalized component forces and combined `fsq`;
- independent continuum force norm;
- `div(B)`;
- minimum Jacobian and nested-surface clearance;
- free-boundary `B.n` and total-pressure jump when applicable;
- nonlinear iteration, accepted/rejected steps, linear iterations, and reason
  for termination;
- peak memory, compile time, and steady-state solve time for production runs.

The default research target is `fsq <= 1e-12`. A lane may use `1e-11` only when
a refinement study shows the discretization floor and all physical observables
are stable; that exception must be documented per benchmark. `max_iterations`
is at least 1000 for promotion runs and does not substitute for convergence.

### 5.2 Preconditioning

Retain the VMEC strategy: exploit radial near-block-tridiagonal structure and
the separability of radial, poloidal, and axial operators. For B-splines the
axial operator is banded; for the periodic hybrid it is periodic banded. The
sequence is:

1. physics scaling of radius and lambda residual blocks;
2. radial block solve using SOLVAX block Thomas;
3. axial/poloidal line or Kronecker preconditioner using SOLVAX primitives;
4. matrix-free SOLVAX GMRES/GCROT on exact JVPs;
5. compare with unpreconditioned and current Chebyshev solves.

Promotion requires iteration counts that remain bounded or grow slowly under
`ns`, `mpol`, and axial-knot refinement. A faster solve that changes the
converged state beyond discretization error is rejected.

### 5.3 Free boundary

The plasma boundary is an unknown. The coupled residual includes plasma force,
normal-field matching, total-pressure balance, vacuum Neumann compatibility,
and geometry validity. External fields come from ESSOS/MGRID. End caps close
the boundary-integral surface only for the exterior Laplace problem; they do
not make plasma field lines periodic or impose a toroidal equilibrium.

For symmetric end coils, boundary/end data and the solved state must preserve
the intended poloidal symmetry to tolerance. For nonaxisymmetric fixtures, the
allowed modes are specified analytically and unintended modes are reported.

## 6. Ordered implementation milestones

Each milestone ends in a small commit and a benchmark JSON update. Do not begin
the next physics milestone until its predecessor's gate passes or is explicitly
deferred.

### Milestone 0: restore a green baseline

1. Import `plot_mout` from `vmec_jax.mirror` in the fixed-gradient example.
2. Run the focused example smoke test, ruff, and strict docs.
3. Record PR checks; do not poll CI between commits.

Gate: all required checks green.

### Milestone 1: freeze the reference physics

Files: consolidate within `mirror/model.py`, `geometry.py`, `forces.py`,
`diagnostics.py`, and existing mirror tests.

1. Audit every normalization/sign against VMEC and ANIMEC equations.
2. Add explicit invariants for discrete `div(B)`, energy-gradient/force
   consistency, isotropic limit, thermodynamic moment identity, and anisotropy
   validity.
3. Re-run axisymmetric fixed/free reference cases at three resolutions, beta
   `0, 0.01, 0.03, 0.10, 0.25, 0.50`, and `max_iterations >= 1000`.
4. Diagnose stalls using projected gradient, step acceptance, preconditioned
   linear residual, and discretization floor.

Gate: axisymmetric fixed and free observables converge, `fsq <= 1e-12` or the
documented `1e-11` floor, and all interface/geometry gates pass.

### Milestone 2: analytic nonaxisymmetric fixtures

Files: add at most one small `mirror/analytic.py`; extend
`tests/mirror/test_fixed_boundary_3d.py` and existing diagnostics.

1. Implement data-only paraxial rotating-ellipse and SFLM evaluators.
2. For the rotating ellipse, return `B0`, leading cross-section coefficients,
   quadrupole coefficients, flux determinant, and expected ellipse angle.
3. For SFLM, return Clebsch labels, field, and analytic flux-tube sections.
4. Verify the evaluators against symbolic identities and low-radius finite
   differences before using them to test the solver.

Gate: analytic identities pass independently of equilibrium code.

### Milestone 3: nonaxisymmetric fixed boundary

1. Solve the 90-degree rotating ellipse through the existing mirror residual.
2. Fit `|B|` by radial order and theta mode. Require vanishing first-order
   `m=1`, correct second-order `m=2` amplitude/phase, and flux conservation.
3. Run `(ns, mpol, nxi)` refinement and both pressure-first and shaping-first
   continuation.
4. Compare low-beta and finite-beta distortion with the long-thin expansion,
   including the predicted quadrupole/diamond trend.
5. Add one top-level, parser-free example with inputs at the top. It produces
   residual history, horizontal 3-D solved surfaces and field lines,
   cross-sections, `|B|`, mode fits, pressure/current/iota-like pitch
   diagnostics, and refinement plots.

Gate: all paraxial coefficients converge with the expected radial order,
physical observables stabilize, and `fsq` meets the convergence contract.

### Milestone 4: nonaxisymmetric free boundary

1. Feed an ESSOS external-field fixture to the theta-dependent BIE path.
2. Continue vacuum to finite beta through the same solved rotating-ellipse
   family; do not prescribe a different boundary at each beta.
3. Refine plasma resolution, side panels, cap panels, and exterior quadrature
   independently.
4. Require resolved LCFS displacement, `B.n`, total-pressure balance,
   anisotropy validity, and consistent field-line pitch at every beta.
5. Compare fixed and free solutions in the limit where the external field pins
   the reference boundary.

Gate: all interface and refinement measures converge. If not achieved in two
documented formulations, mark nonaxisymmetric free boundary deferred, keep one
compact negative benchmark, and remove its public promotion claim.

### Milestone 5: native B-spline axial basis

Files: add only `mirror/splines.py`; adapt `basis.py`, state packing, and the
existing tests. Generic knot/banded algebra should be proposed to SOLVAX rather
than duplicated if it has another user.

1. Implement clamped and periodic cubic basis evaluation, first derivatives,
   quadrature/collocation matrices, knot insertion, and coefficient transfer.
2. Expose the existing axial-basis operations through a minimal internal
   protocol; avoid a public inheritance hierarchy.
3. Store axial equilibrium variation in coefficients and evaluate lazily at
   quadrature nodes.
4. Test partition of unity, polynomial reproduction, endpoint behavior,
   derivative convergence, exact knot insertion, and JVP/VJP versus finite
   differences.
5. Re-run Milestones 1 and 3 with Chebyshev and B-spline bases. B-spline becomes
   default only after state, force, and observable parity.

Gate: same converged physics within discretization error, fewer coefficients
for long straight sections, no worse memory scaling, and stable knot refinement.

### Milestone 6: toroidal stellarator-mirror hybrid

Files: the implementation belongs in the mirror geometry/state family, not
`core/hybrid*.py`.

1. Define a periodic B-spline Cartesian centerline with two long straight legs
   and two curved returns. Enforce closure and at least C2 continuity.
2. Construct a rotation-minimizing local frame; represent semi-major axis,
   semi-minor axis, and ellipse angle with periodic B-splines. Avoid Frenet
   singularities on zero-curvature straight legs.
3. Validate centerline closure, join continuity, intended straightness,
   curvature, frame holonomy, positive Jacobian, tube self-clearance, and knot
   insertion.
4. Extend the mirror variational state from straight `z` to centerline arc
   length and its local frame. Curvature terms must enter the metric and force
   kernels; no Fourier projection is used in the solve.
5. Solve fixed boundary first, then couple the same state to an ESSOS/MGRID
   vacuum for free boundary and beta `0..0.50` continuation.
6. Validate the straight-leg limit against the open mirror, the smooth-torus
   limit against ordinary vmec_jax/VMEC2000, and derivatives against FD.

Gate: a converged native B-spline fixed equilibrium with both limiting checks.
Free boundary is promoted only after the same interface/refinement gates as
Milestone 4. Otherwise document it as deferred and keep only the fixed model.

### Milestone 7: solver and derivative consolidation

1. Replace SciPy GMRES in `mirror/implicit.py` and
   `mirror/free_boundary_implicit.py` with SOLVAX GMRES after residual and
   gradient parity.
2. Replace the manual free-boundary Jacobian chunk loop with SOLVAX chunked AD
   if its memory and output contract are equivalent.
3. Benchmark SOLVAX PCG against SciPy CG in `mirror/vacuum.py`; migrate only if
   runtime, memory, and convergence are no worse.
4. Keep SciPy nonlinear host drivers and mirror-specific preconditioner
   assembly. Do not rewrite them solely for aesthetic uniformity.
5. Validate JVP and VJP for fixed/free, isotropic/anisotropic, and B-spline
   controls. Report adjoint residuals as well as FD agreement.

Gate: derivative error, primal state, iteration count, runtime, and peak memory
are all recorded; no duplicate generic Krylov implementation remains.

### Milestone 8: delete experimental ownership and shrink the branch

1. Migrate every remaining coil consumer to ESSOS/MGRID/callable inputs.
2. Delete `core/coils.py` and `tests/test_coils.py` after ESSOS contract tests
   cover the needed field normalization and mgrid interchange.
3. Delete `core/hybrid.py` and `core/hybrid_free_boundary.py` after the native
   hybrid replaces them. Delete the Fourier-only hybrid tests/examples and
   stale benchmark data.
4. Reduce `mirror/__init__.py` to user-level state/config/solve/I/O/plot names;
   tests import internals from their owning modules.
5. Consolidate the exterior BIE helper modules where ownership is artificial,
   while keeping geometry, quadrature, and solve responsibilities readable.
6. Remove superseded figures and scripts. Generated MOUT/WOUT/mgrid/trace data
   remain ignored.

Hard budget: the final branch must contain fewer source lines and fewer public
modules than it did at this baseline. Added B-spline/hybrid production code must
be offset by deleting at least the 1,519 lines in legacy coil/Fourier-hybrid
source, plus obsolete tests and examples. No production source file exceeds
800 lines without a documented reason.

### Milestone 9: release evidence and documentation

1. README: one compact capability table and four reproducible results:
   toroidal VMEC, axisymmetric free mirror, rotating-ellipse mirror, and native
   B-spline hybrid.
2. `docs/mirror_geometry.rst`: coordinates, field representation, isotropic and
   ANIMEC energies, boundary conditions, residual normalization, paraxial/SFLM
   validations, B-spline formulation, and known limits.
3. `docs/architecture.rst`: vmec_jax/ESSOS/SOLVAX ownership and derivative path.
4. Examples use editable constants at the top and no parser. Every scientific
   example plots actual solved boundaries, residual history, cross-sections,
   horizontal 3-D surfaces and visible field lines, `|B|`, and relevant
   pressure/current/pitch quantities.
5. Store only compressed documentation figures, normally below 300 KiB each.
   Regeneration scripts and compact JSON/CSV reference data are committed;
   simulation output directories are not.
6. Run ruff, strict Sphinx, all focused mirror tests, example smoke, full CI,
   and one GPU production benchmark over SSH office. Compare CPU/GPU cold
   compile, steady solve, peak memory, nonlinear iterations, and gradients.

Gate: documentation reproduces the supported claims, required CI is green, and
the draft PR is ready for one final scientific review and merge.

## 7. Promotion matrix

Every supported lane must have all applicable columns checked.

| Model | Analytic/independent | `fsq` | refinement | boundary physics | derivatives | example/docs |
|---|---|---|---|---|---|---|
| Axisymmetric fixed | two-loop axis/off-axis field, MMS | required | `ns`, axial knots | fixed geometry | JVP/VJP | required |
| Nonaxisymmetric fixed | rotating ellipse, SFLM, long-thin beta | required | `ns`, `mpol`, knots | fixed geometry | JVP/VJP | required |
| Axisymmetric free | two-loop/ESSOS and fixed-limit | required | plasma + exterior | `B.n`, pressure jump | JVP/VJP | required |
| Nonaxisymmetric free | paraxial fixture and fixed-limit | required | plasma + all panels | `B.n`, pressure jump | JVP/VJP | required |
| B-spline hybrid fixed | straight and smooth-torus limits | required | knots, radial, poloidal | fixed geometry | JVP/VJP | required |
| B-spline hybrid free | fixed-limit plus ESSOS/MGRID | required | plasma + exterior | `B.n`, pressure jump | JVP/VJP | required |

## 8. Explicit deferrals

The following are not required to finish this plan:

- differentiating through CLI iterations;
- arbitrary open-axis curvature before the periodic hybrid metric is validated;
- kinetic end losses, sheaths, open-field transport, or stability;
- independent coil optimization inside vmec_jax;
- a general-purpose finite-element framework;
- a radial B-spline rewrite;
- forcing open-mirror output into classic toroidal WOUT;
- claiming VMEC2000 parity for topology VMEC2000 cannot represent;
- `ftol < 1e-12` when double-precision discretization/error diagnostics show no
  physical benefit.

## 9. Primary references

- Hirshman and Whitson, *Steepest-descent moment method for three-dimensional
  magnetohydrodynamic equilibria* (1983), and VMEC/STELLOPT documentation:
  <https://princetonuniversity.github.io/STELLOPT/VMEC.html>
- DESC source and documentation: <https://github.com/PlasmaControl/DESC> and
  <https://desc-docs.readthedocs.io/>
- Rodriguez, Helander, and Goodman, *The maximum-J property in
  quasi-isodynamic stellarators*, Appendix C, JPP 90 (2024):
  <https://doi.org/10.1017/S0022377824000345>
- Rodriguez et al., *Constructing precisely quasi-isodynamic magnetic fields*
  (rotating elongation and mirror term):
  <https://doi.org/10.1017/S0022377823001125>
- Pearlstein, *Three-dimensional equilibrium in quadrupole symmetric tandem
  mirrors in paraxial limit*, UCRL-89767 (1983):
  <https://digital.library.unt.edu/ark:/67531/metadc1102940/>
- Goodman, Freidberg, and Lane, *Analytic mirror equilibria with new long-thin
  terms* (1986): <https://doi.org/10.1063/1.865851>
- Kesner et al., *Theory of the straight field line mirror* (2005), and Agren
  and Moiseenko's analytic-field/coil follow-up (2022):
  <https://doi.org/10.46813/2022-142-013>
- Cooper et al., *Three-dimensional anisotropic pressure free boundary
  equilibria* (2009):
  <https://pure.mpg.de/pubman/item/item_2141784_1/component/file_2141783/Cooper_Three.pdf>
- Asahi et al., *MHD Equilibrium Analysis with Anisotropic Pressure in LHD*
  (2011): <https://www.jspf.or.jp/PFR/PDF/pfr2011_06-2403123.pdf>
- Skene and Burns, *Fast automated adjoints for spectral PDE solvers* (2025):
  <https://arxiv.org/abs/2506.14792>
- JAX custom derivative and implicit linear-solve documentation:
  <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>
  and <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html>
- SOLVAX source and solver contracts: <https://github.com/uwplasma/SOLVAX>

## 10. Immediate next action

Execute Milestone 0, then Milestone 1. Do not add another planning document.
