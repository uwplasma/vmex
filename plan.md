# Mirror equilibrium final execution plan

Status: active and authoritative plan for draft PR #22. This file replaces all
older mirror roadmaps. Do not add another plan file. Git commits and compact
benchmark JSON files are the execution log.

Review baseline: `codex/mirror-geometry` at `92c5c307`, based on `main` at
`ed4ac7ac`, reviewed 2026-07-14. The branch is 240 commits ahead and zero behind
`origin/main`; PR #22 is open, draft, and mergeable. The last CI run passed every
job except example smoke, whose stale `plot_mout` import is fixed in the current
working tree and passes locally.

## 1. Mission and finish line

Deliver a small, fast, research-grade extension of `vmec_jax` for nested-surface
ideal-MHD equilibria in:

1. straight-axis axisymmetric fixed-boundary mirrors;
2. straight-axis nonaxisymmetric fixed-boundary mirrors;
3. straight-axis axisymmetric free-boundary mirrors;
4. straight-axis nonaxisymmetric free-boundary mirrors;
5. a closed toroidal stellarator-mirror hybrid with two long straight mirror
   legs and two curved stellarator returns.

Open mirrors use a nonperiodic axial coordinate. The closed hybrid uses periodic
arc length around a Cartesian centerline. Full B-spline support means that all
axial or centerline dependence of the equilibrium state, boundary, and frame is
represented by B-spline coefficients. The radial flux mesh remains staggered and
the periodic poloidal direction remains Fourier; replacing either with splines
would add code without addressing the straight-section problem.

A supported result is an equilibrium solved from an MHD residual. A prescribed
tube sampled in an external field, a Fourier projection of a square target, or a
surface with small `B.n` but no plasma-force solve is not a supported result.

The branch is complete when each promoted lane has:

- a discrete equilibrium residual at or below `1e-12`, or a documented
  double-precision floor no larger than `1e-11`;
- a matching staggered weak-force check and stable physical observables under
  independent resolution refinement;
- valid nested geometry and `div(B)` near roundoff;
- free-boundary `B.n` and total-pressure balance where applicable;
- analytic, manufactured, or independent-code validation;
- implicit JVP/VJP checks for the supported differentiable inputs;
- one reproducible example and current documentation with compact plots.

Any lane that fails its gate after the bounded attempts specified below is
explicitly deferred and its public scaffold is removed. The plan must end in a
finite number of decisions, not an indefinitely growing research branch.

## 2. Decisions that will not be revisited

### 2.1 Repository ownership

- **vmec_jax** owns equilibrium coordinates, MHD functionals and residuals,
  boundary coupling, continuation, equilibrium I/O, and plots of solved states.
- **ESSOS** owns coils, Biot-Savart, coil-field line tracing, and mgrid creation.
  A vmec_jax free-boundary solve accepts `MgridField` or an `xyz -> B` callable.
- **SOLVAX** owns generic Krylov and structured direct solvers, generic
  preconditioner tools, chunked AD, and implicit root/linear-solve machinery.
- **SciPy** remains acceptable as the fast, nondifferentiable host driver for
  CLI nonlinear solves. There is no requirement to trace or differentiate its
  iterations.

No public coil model, Biot-Savart implementation, field-line integrator, generic
GMRES/PCG, or general finite-element framework will be added to vmec_jax.

### 2.2 One plasma model

There is one state contract, geometry evaluator, magnetic-field representation,
isotropic energy, anisotropic energy, and diagnostic family. Axisymmetry is
`mpol=0`; free boundary adds interface and vacuum equations; B-spline and
Chebyshev bases satisfy the same internal axial-basis protocol. There will be no
copied `MirrorEquilibrium` hierarchy and no second hybrid equilibrium under
`vmec_jax/core`.

### 2.3 Solver and derivative policy

The forward CLI path is optimized for runtime and memory and may use SciPy host
control. Derivatives are taken implicitly from the converged residual

`F(u, p) = 0`, `F_u du = -F_p dp`, and `F_u^T lambda = objective_u`.

- use forward JVPs for a few parameter directions;
- use reverse implicit adjoints for scalar objectives with many controls;
- never retain thousands of nonlinear iterates for reverse AD;
- use finite differences only as a validation oracle;
- use SOLVAX linear solves once primal and derivative parity is demonstrated.

JAXopt and Lineax confirm the same implicit-differentiation pattern, but adding
either as another runtime dependency provides no current advantage over the
SOLVAX path already used by vmec_jax.

### 2.4 Scope deferrals

The following do not block completion:

- kinetic end losses, sheaths, transport, and MHD stability;
- arbitrary open-axis curvature beyond the closed hybrid geometry;
- radial B-splines or poloidal finite elements;
- differentiating through CLI iterations or with respect to an initial guess;
- classic toroidal WOUT output for an open mirror;
- VMEC2000 parity for open topology that VMEC2000 cannot represent;
- independent coil optimization in vmec_jax;
- free-boundary hybrid promotion if the fixed hybrid passes but the vacuum
  coupling does not pass its bounded validation attempts.

## 3. Current source and evidence audit

### 3.1 Branch size and ownership debt

Relative to main after the Milestone 2 deletions, the branch changes 130 files,
adds 20,953 lines, deletes 4,255 lines, and has a net addition of 16,698 lines.
The mirror package contains about 8,430 Python lines in 20 modules. The largest
files are `forces.py` (1,085), `solver.py` (935), `exterior_bie.py` (809), and
`exterior_mesh.py` (737).

Milestone 2 removed the duplicated coil implementation, prescribed Fourier
hybrid, their tests/examples, and obsolete figures. It also reduced the eager
top-level mirror namespace to 47 lazy user contracts. Remaining size debt is
concentrated in force assembly, nonlinear solves, and exterior BIE assembly;
those files are simplified only alongside their Milestone 8 solver work so
validated numerical paths are not mixed speculatively.

### 3.2 Physics worth retaining

- open coordinates `(s, theta, xi)` and regular embedding
  `r = sqrt(s) a(s, theta, xi)`;
- the divergence-free contravariant representation
  `sqrt(g) B^theta = I'(s) - d(lambda)/dxi` and
  `sqrt(g) B^xi = Psi'(s) + d(lambda)/dtheta`;
- the VMEC-like staggered radial mesh and mass-conserving isotropic energy;
- the ANIMEC thermodynamic identity
  `p_perp = p_parallel - B (d p_parallel/dB)_s`;
- the side-plus-cap exterior BIE for open mirrors;
- host-controlled fixed/free solves, continuation, restarts, native MOUT I/O,
  horizontal plots, cross-sections, `|B|`, field lines, and residual histories;
- fixed and axisymmetric-free implicit derivative paths.

### 3.3 Measured results and the active blocker

The production axisymmetric exterior-vacuum scan used `ns=7`, `nxi=13`,
`ftol=1e-12`, `max_iterations=2000`, and beta
`[0, 0.01, 0.03, 0.10, 0.25, 0.50]`. It converged in 6--10 iterations with:

- variational maxima between `3.6e-15` and `7.0e-15` and independently
  assembled staggered-weak maxima between `7.1e-16` and `1.4e-15`;
- normalized `div(B)` between `1.25e-15` and `1.28e-15`;
- normal-stress residuals below `3.4e-15` and normalized `B.n` below `2e-16`;
- a 7.6% center-radius increase and 24.9% center-field decrease at beta 0.50;
- a beta-0.50 field ratio of 0.75095 versus the paraxial
  `sqrt(1-beta)=0.70711`, a 6.2% relative difference.

Thus beta is coupled to the solved state and the high-beta trend is physically
visible. A bounded `(3,5)`, `(5,9)`, `(7,13)` scan completed in 137 seconds and
peaked at 12.1 GiB RSS. Between the two finest levels, the beta-0.50 radius and
field ratio change by 0.24% and 1.1%. A `(9,17)` attempt was stopped after more
than five minutes in dense Jacobian assembly; this is a performance blocker,
not equilibrium evidence.

The unresolved diagnostic is the pointwise isotropic force reconstruction. At
beta 0.10 its normalized RMS is `0.134`, `0.0362`, and `0.0645` on the three
bounded levels, while the constrained discrete and weak residuals remain near
machine precision. The pointwise value is therefore recorded but non-gating.

VMEC2000 computes magnetic pressure, kinetic pressure, covariant fields, and
force kernels on a radial half mesh and forms full-mesh forces with explicit
staggered averaging. The current mirror diagnostic instead differentiates
full-mesh reconstructed fields and pressure independently. It is therefore not
a like-for-like independent residual and cannot be a promotion gate in its
current form.

### 3.4 Residual contract

Every solve reports three distinct levels:

1. **Discrete variational residual**: normalized energy gradient on active
   degrees of freedom. This defines `ftol` and nonlinear convergence.
2. **Staggered weak-force residual**: projection of the tensor force onto the
   same admissible variations and quadrature used by the energy. It must agree
   with the variational residual under refinement without being computed by
   simply calling the same gradient routine.
3. **Pointwise reconstructed force**: `J x B - grad(p)` evaluated from a
   documented half-to-full-mesh reconstruction. This is a spatial accuracy
   diagnostic, not a solve gate, until manufactured and refinement tests show
   convergence.

`div(B)`, Jacobian sign, clearance, `B.n`, pressure jump, and physical
observables are separate gates. SciPy success or a small step is never an
equilibrium criterion.

### 3.5 Capability status

| Lane | Completion | Evidence retained | Blocking evidence |
|---|---:|---|---|
| Axisymmetric fixed mirror | 90% | real solve, `ftol=1e-12`, MMS, weak force, gradients, spline parity | spline implicit derivatives and final release evidence |
| Axisymmetric free mirror | 80% | coupled solve, beta 0--50%, `B.n`, stress, weak force, three-grid paraxial trend | dense-Jacobian scaling and spline coupling |
| Nonaxisymmetric fixed mirror | 72% | native-spline rotating ellipse and SFLM below `ftol`, symmetry/direction oracles, plotted example | amplitude/refinement, finite beta, derivatives |
| Nonaxisymmetric free mirror | 30% | theta-dependent BIE and residual exist | no converged analytic fixture or panel study |
| ANIMEC model | 55% | functional, moments, isotropic limit, indicators | source-equation audit and independent finite-beta case |
| Implicit derivatives | 65% | fixed and axisymmetric-free FD checks | duplicated Krylov path and missing spline/nonaxisymmetric scaling |
| Preconditioning | 45% | separable prototype and Newton-GMRES | no bounded-iteration basis/resolution study |
| Native B-spline open mirror | 65% | tested basis/state/transfer, fixed solve, knot convergence, coefficient preconditioner | free-boundary state and derivative parity |
| Native B-spline closed hybrid | 0% | Fourier target is not reusable physics | centerline/frame/metric/residual implementation |
| ESSOS ownership cleanup | 100% | MGRID/callable contract and live ESSOS smoke | none |
| Source simplification | 40% | Fourier-hybrid and coil ownership removed | public API and large mirror modules remain |

Percentages measure promotion evidence, not implementation effort or line count.

## 4. External source conclusions

### 4.1 VMEC2000, VMEC++, and free boundary

Retain VMEC's variational principle, divergence-free magnetic representation,
radial half-mesh regularization, force normalization, continuation, and
block-radial preconditioning. Mirror force kernels should follow the staggering
in VMEC2000 `bcovar.f`, `forces.f`, `residue.f90`, `fbal.f`, and `precon2d.f`,
adapted to an open axial basis rather than copied line by line.

VMEC2000 cannot represent an open plasma or fixed-flux end cuts. Use it only for
sign and normalization checks and for the smooth closed-torus limit of the
hybrid. The Hirshman-van Rij-Merkel free-boundary formulation confirms that the
plasma boundary is an unknown set by total-pressure continuity and a vacuum
Neumann problem; mgrid input does not replace either equation.

### 4.2 DESC branches

The local clone and remote branches `mirror`, `mirror_anisotropy`,
`finite_element_basis`, `finite_element_basis_alan`, and `dd/cylindrical` were
inspected.

Useful contracts to reproduce in a smaller form:

- a nonperiodic coordinate has explicit nodes, quadrature, differentiation,
  interpolation, endpoint semantics, and coefficient transfer;
- `ChebyshevFourierSeries`/`ChebyshevRZToroidalSurface` separate open axial and
  periodic poloidal representations cleanly;
- straight-coordinate analytic fixtures are independent test data;
- continuation must support pressure-first and shape-first paths;
- state scaling and objective normalization must be explicit.

Do not port the branches. `mirror` adds roughly 11,000 source/test lines and a
2,255-line copied equilibrium class; `mirror_anisotropy` adds roughly 14,700
lines and contains commits that explicitly describe suspicious force results;
the finite-element branches add 4,000--4,750 lines of broad triangular,
tetrahedral, and interval mesh machinery. Their straight solve tolerances near
`1e-6` are insufficient for this plan.

### 4.3 Paraxial and straight-field-line mirrors

Appendix C of Rodriguez, Helander, and Goodman gives the low-radius straight
mirror gates:

- flux determinant `X1c Y1s - X1s Y1c = Bbar/B0(z)`;
- no first-order poloidal field-strength variation: `B1c = B1s = 0`;
- leading transverse variation
  `B = B0 + r^2 [B20 + B2c cos(2 alpha) + B2s sin(2 alpha)] + O(r^3)`;
- a Riccati/sigma equation for ellipse magnitude and orientation.

This supplies a 90-degree rotating-ellipse fixture. An `m=1`, order-`r` signal
is an error; an `m=2`, order-`r^2` signal is expected and is not evidence that a
mirror incorrectly lost poloidal symmetry.

The Straight Field Line Mirror (SFLM) is a separate fixture: its Clebsch field
has straight but nonparallel field lines and analytic elliptical flux tubes.
Use it to validate field, flux surfaces, and ellipticity, not to require ellipse
rotation. Goodman-Freidberg-Lane and Pearlstein provide finite-beta/long-thin
quadrupole and diamond-distortion trends. Pastukhov's toroidally linked mirror
report provides the closest analytic context for the eventual two-leg/two-return
hybrid.

### 4.4 ANIMEC

ANIMEC uses an anisotropic energy with a distribution-derived `p_parallel(s,B)`
and consistent `p_perp`, not two independently prescribed pressure fields. Its
VMEC2000 source modifies the half-mesh magnetic kernels by `sigma`, adds
`p_perp` to interface pressure, and reports firehose/mirror criteria. Promotion
requires:

- equation-by-equation parity of the bi-Maxwellian form factor above and below
  the critical field;
- AD/FD checks of `p_perp = p_parallel - B dp_parallel/dB`;
- isotropic and zero-hot-fraction limits;
- positive firehose and mirror-ellipticity indicators at every node;
- anisotropic total-pressure balance and a resolution-stable finite-beta
  observable.

ANIMEC is not allowed to delay scalar-pressure mirror promotion. Until these
gates pass it remains an explicitly experimental API; if it cannot pass after
one source-parity and one independent benchmark attempt, it is deferred and
removed from top-level exports.

### 4.5 B-splines, solvers, and adjoints

Use clamped cubic B-splines for open axial dependence and periodic cubic
B-splines for the closed centerline. Local support gives banded derivative and
mass matrices, stable local refinement, and far fewer controls than a global
Fourier fit for long straight sections. Use a rotation-minimizing/Bishop frame
for the hybrid because the Frenet frame is singular where curvature vanishes.

The 2025 fast spectral-adjoint work supports differentiating the residual
operator and solving a transpose system instead of differentiating through the
nonlinear trajectory. JAX `custom_linear_solve`, JAXopt implicit roots, Lineax,
and SOLVAX all follow this principle. SOLVAX is the chosen dependency because it
already supplies the needed GMRES/GCROT/PCG, block Thomas, banded and periodic
banded solves, chunked Jacobians, and implicit root/linear-solve contracts.

## 5. Ordered milestones

Each milestone ends with focused tests, a compact benchmark update, a small
commit, and a push. CI is checked after grouped work, not polled after every
commit. No later physics milestone starts until the current gate passes or the
specified deferral decision is recorded.

### Milestone 0: restore and record the baseline

Status: implementation complete locally.

1. Fix both stale `plot_mout` imports.
2. Persist normalized `div(B)` and clearly named pointwise-force diagnostics in
   fixed/free results, MOUT, and the beta-scan CSV.
3. Preserve backward reading of MOUT files without the new attributes.
4. Run example smoke, focused output/geometry/free-boundary tests, ruff on
   Python files, and strict Sphinx.

Gate: local tests and docs pass; the next pushed CI run is green.

### Milestone 1: freeze scalar-pressure reference physics

Status: discrete/weak implementation and bounded axisymmetric evidence
complete. The pointwise reconstruction failed its refinement criterion and is
explicitly non-gating. Dense free-boundary Jacobian scaling is carried into the
solver simplification work.

Files: only existing `mirror/forces.py`, `geometry.py`, `diagnostics.py`, and
focused tests unless a small helper clearly reduces code.

1. Derive the isotropic first variation on the radial half mesh and document
   every interpolation and sign against VMEC2000.
2. Implement a staggered weak-force projection independent of `jax.grad`; test
   its correlation and convergence to the energy gradient on manufactured
   cylinder, flared, and finite-beta states.
3. Reconstruct pointwise `J x B - grad(p)` from compatible half-mesh quantities.
   Require convergence on manufactured fields before using its norm as evidence.
4. Repeat fixed and free axisymmetric runs at three `(ns,nxi)` resolutions and
   beta `[0,.01,.03,.10,.25,.50]`, `max_iterations >= 1000`.
5. Record discrete force, weak force, pointwise force, `div(B)`, geometry,
   interface residuals, central radius/field, mirror ratio, compile time, steady
   runtime, and peak memory.
6. Diagnose any stall with projected gradient, accepted/rejected steps, linear
   residual, condition estimate, and the measured discretization floor.

Gate: axisymmetric fixed and free observables converge; discrete and weak force
meet the residual contract; the beta trend and paraxial field depression are
resolution stable. Require less than 0.5% center-radius and 2% center-field
change between the two finest bounded levels. The current evidence passes.
Pointwise force may become gating only after a compatible reconstruction passes
manufactured refinement tests.

### Milestone 2: remove known false and misowned lanes

Status: implementation complete locally. Fourier hybrid and duplicated coil
ownership are removed. Mirror source accepts MGRID or vectorized ``xyz -> B``
fields. The top-level namespace is lazy and reduced from 113 flattened symbols
to 47 user contracts; numerical kernels remain in their owning submodules.
The replacement field contract passes focused tests and a live ESSOS
``BiotSavart`` evaluation on a VMEC-JAX vacuum grid.

1. Change current hybrid docs/examples to explicit historical/experimental
   status, transfer any reusable plotting requirements, then delete the Fourier
   target implementation, tests, examples, and showcase figures.
2. Migrate remaining mirror consumers to ESSOS/MGRID/callable fields and delete
   vmec_jax coil source/tests after interchange normalization tests pass.
3. Remove duplicate helpers and reduce top-level mirror exports to config,
   state, solve, continuation, MOUT, and plotting APIs.
4. Recount changed files, source lines, modules, and public symbols.

Gate: no supported example claims an unsolved hybrid and no vmec_jax module owns
coil physics. This deletion occurs before adding B-spline/hybrid code so the
branch gets smaller during implementation, not only at the end.

### Milestone 3: independent nonaxisymmetric fixtures

Status: implementation complete locally. One compact ``mirror/analytic.py``
contains both fixtures and their explicit validity domains; five independent
tests cover flux, vacuum consistency, Riccati, quadrupole, curl/divergence,
straight-line, ellipticity, finite-difference radial order, and JAX derivatives.

Add at most one compact `mirror/analytic.py`.

1. Implement data-only rotating-ellipse paraxial coefficients: `B0`, first-order
   section coefficients, flux determinant, quadrupole coefficients, and angle.
2. Implement the SFLM Clebsch labels, Cartesian field, and analytic flux-tube
   sections.
3. Verify both with symbolic identities and low-radius finite differences that
   do not call equilibrium code.
4. Add long-thin finite-beta expected scaling for a bounded parameter range;
   label it asymptotic rather than exact.

Gate: all analytic identities pass independently and each fixture has an
explicit domain of validity.

### Milestone 4: native B-spline axial state

Status: basis, coefficient state, and bounded fixed-boundary solve layers are
complete locally. Clamped and periodic cubic values, derivatives, quadrature,
fitting, exact open-knot insertion, and JVP/VJP tests pass. Coefficient-native
state/boundary projection uses endpoint-augmented Gauss evaluation and matches
a 41-node Chebyshev energy with 9 controls to ``5.0e-13`` relative. The common
host nonlinear policy converges axisymmetric and finite-current ``mpol=1``
coefficient solves below ``ftol=1e-12``; the independent staggered pullback is
also below ``7e-16``. An ``ns=5`` finite-beta parity case uses 31 active spline
variables versus 45 Chebyshev variables with relative differences of
``5.1e-7`` in energy, ``5.9e-6`` in volume, and ``3.2e-4`` in center radius.
Coefficient-space preconditioning, larger knot refinement, and all Milestone 1
case parity remain open.

The first coefficient-space preconditioner reuses the nodal radial/Fourier
tensor inverse with a spline Galerkin axial stiffness. A 585-variable
``mpol=1`` cylinder crosses the dense threshold and converges both residuals to
``1.23e-15`` while recovering the analytic radius to ``5.6e-15``. Its 1,000
inner GMRES iterations make this a correctness gate only; bounded Krylov
scaling remains Milestone 8 work.

A five-level knot study on the finite-pressure/current flared case is monotone:
from 5 to 11 coefficients, relative energy error against ``nxi=17`` Chebyshev
drops from ``1.09e-6`` to ``5.14e-8`` and volume error from ``1.19e-5`` to
``2.18e-6``. All residuals remain below ``9e-15``. Fixed-boundary spline parity
therefore passes at bounded resolution; coefficient-native free-boundary
coupling remains open before any default switch.

Add only `mirror/splines.py`; keep the basis protocol in `basis.py` small.

1. Implement clamped and periodic cubic basis values, first derivatives,
   quadrature/collocation, knot insertion, and coefficient transfer.
2. Test partition of unity, cubic reproduction, endpoint/periodic continuity,
   derivative convergence, exact knot insertion, and JVP/VJP versus FD.
3. Store axial state and boundary variation in coefficients and evaluate at
   quadrature nodes; do not merely interpolate a nodal Chebyshev state after the
   solve.
4. Run Milestone 1 cases with Chebyshev as the reference and B-splines as the
   candidate. Compare state, energy, all residuals, observables, runtime, memory,
   and coefficient count.
5. Make B-spline the public default only after parity; retain Chebyshev as an
   internal spectral oracle for open mirrors.

Gate: identical converged physics within discretization error, stable knot
refinement, and fewer axial controls for long straight regions without worse
memory scaling.

### Milestone 5: nonaxisymmetric fixed-boundary mirrors

Status: active. A parser-free root example now runs five-stage native-spline
continuations for both independent fixtures and writes solved MOUT files plus
horizontal 3-D field-line, section, ``|B|``, convergence, and validation plots.
The rotating-ellipse continuation reaches the full
90-degree, elongation-1.5 thin tube at ``ns=5`` with discrete/weak residuals
below ``1.2e-16`` and ``div(B)<7.4e-15``. Even theta quadrature is mandatory:
12 nodes preserve half-turn symmetry and reduce the forbidden ``m=1`` signal
below ``9e-15``; 5 or 9 nodes alias even nonlinear products into odd modes.
The full-mesh axis ``|B|`` reconstruction does not converge radially and is not
a valid paraxial oracle. The compatible half-mesh field recovers the expected
``m=2`` phase but not yet its amplitude. The independent SFLM field-direction
comparison passes: mean/minimum direction cosines improve from
``0.999956/0.999316`` to ``0.999988/0.999810`` when tube radius is halved. Its
full-mesh magnitude is nonconvergent and remains non-gating. At ``ns=7`` the
solve stays nested only
after invalid-Jacobian states receive infinite optimizer merit; it reaches
``ftol`` but requires 2,500--17,017 inner GMRES iterations. Thus equilibrium
existence and symmetry pass, while paraxial amplitude and preconditioner gates
remain open. MOUT ``|B|`` now uses the compatible radial-Gauss reconstruction,
while Cartesian full-mesh samples are retained for field-line direction.

1. Solve the 90-degree rotating ellipse with the common scalar-pressure
   residual and B-spline axial state.
2. Fit `|B|` by radial order and poloidal mode. Require vanishing first-order
   `m=1`, correct second-order `m=2` amplitude/phase, flux determinant, and
   ellipse angle.
3. Solve the independent SFLM boundary and compare field and section geometry.
4. Refine `ns`, `mpol`, theta quadrature, and axial knots; run pressure-first
   and shape-first continuation.
5. Compare finite-beta quadrupole/diamond trends within the long-thin validity
   range.
6. Add one parser-free root example with constants at the top. It plots
   horizontal 3-D solved surfaces and visible field lines, sections, `|B|`,
   residual history, mode fits, pressure/current/pitch, and refinement.

Gate: paraxial coefficients have the expected radial order, observables and
weak force converge, and the residual contract is met.

### Milestone 6: nonaxisymmetric free-boundary mirrors

1. Supply an ESSOS/MGRID or analytic external-field fixture to the existing
   theta-dependent vacuum path.
2. Continue one solved rotating-ellipse family from vacuum to beta 0.50; never
   prescribe a separate beta-dependent boundary.
3. Refine plasma variables, side panels, cap panels, singular quadrature, and
   exterior field interpolation independently.
4. Require stable LCFS displacement, field depression, `B.n`, total-pressure
   balance, weak force, geometry, and field-line pitch.
5. Compare with the fixed solution as the external field increasingly pins the
   reference boundary.

Gate: all plasma and vacuum refinements converge. After at most two documented
vacuum/discretization formulations, failure means this lane is deferred, one
compact negative benchmark is retained, and its public solve claim is removed.

### Milestone 7: closed toroidal stellarator-mirror hybrid

The hybrid is implemented in the mirror geometry/state family, not under
`core/hybrid*.py`.

1. Define a periodic cubic B-spline Cartesian centerline with two long straight
   legs and two smooth curved returns; enforce closure and at least C2 continuity.
2. Build a rotation-minimizing frame and periodic B-spline semi-major axis,
   semi-minor axis, and ellipse angle. Validate frame holonomy explicitly.
3. Extend the embedding and metric from straight `z` to centerline arc length.
   Include curvature and frame-rotation terms in geometry, magnetic field,
   energy, weak force, and pointwise diagnostics.
4. Validate closure, join continuity, straightness, curvature, self-clearance,
   positive Jacobian, `div(B)`, and knot insertion before solving MHD.
5. Solve scalar-pressure fixed boundary first. Validate the long-straight-leg
   limit against the open mirror and a smooth near-circular limit against normal
   vmec_jax and VMEC2000.
6. Add toroidal current or a nonaxisymmetric transform-producing state and
   verify nonzero rotational transform/field-line pitch.
7. Only after fixed promotion, couple the periodic closed LCFS (no end caps) to
   ESSOS/MGRID and attempt beta 0--0.50 free-boundary continuation.
8. Add one parser-free root example with solved beta surfaces, horizontal
   straight legs, curved returns, field lines, sections, `|B|`, iota/pitch,
   pressure, and residual/refinement plots.

Required gate: converged fixed hybrid with both limiting checks and implicit
derivatives. Conditional gate: free hybrid passes the Milestone 6 interface and
refinement requirements; otherwise it is documented as deferred.

### Milestone 8: preconditioning, implicit derivatives, and ANIMEC decision

1. Preserve VMEC-like radial block structure. Apply physics scaling, SOLVAX
   block Thomas radially, banded/periodic-banded axial solves, and matrix-free
   SOLVAX GMRES or GCROT on exact JVPs.
2. Compare no preconditioner, current separable preconditioner, and the structured
   preconditioner across `ns`, `mpol`, and knot refinement. Require bounded or
   slowly growing Krylov iterations and unchanged converged states.
3. Replace mirror-local SciPy GMRES/chunk loops only where SOLVAX matches primal,
   derivative, runtime, and memory contracts. Keep SciPy nonlinear host drivers.
4. Validate fixed/free JVP and VJP for geometry, profiles, external-field inputs,
   and spline controls. Report tangent/adjoint linear residuals and FD error.
5. Complete the ANIMEC source and benchmark gates in Section 4.4. Promote or
   explicitly defer it; do not leave ambiguous top-level claims.

Gate: no duplicate generic Krylov implementation remains; primal/adjoint
residuals, runtime, peak memory, and gradient error are recorded.

### Milestone 9: final simplification and release evidence

1. Consolidate modules only where ownership is artificial; keep basis, geometry,
   force, vacuum, solve, I/O, and plotting responsibilities readable.
2. Require every public function/class to have a short purpose-first docstring,
   inputs/units, output contract, and important validity condition. Comments
   explain staggering, gauges, and singular treatment, not obvious syntax.
3. Final source must have fewer lines, public symbols, and modules than the
   Section 3 baseline. New spline/hybrid code must be offset by deleting at
   least the 1,519 legacy coil/Fourier-hybrid source lines plus obsolete tests
   and examples. No source file exceeds 800 lines without a written reason.
4. README: one capability table and four reproducible results: normal toroidal
   VMEC, axisymmetric free mirror, rotating-ellipse mirror, and native B-spline
   hybrid.
5. Docs: equations, discretization/staggering, boundary conditions, residual
   meanings, paraxial/SFLM validation, spline/hybrid geometry, derivative path,
   ownership, examples, and known limits.
6. Store only compressed showcase figures, normally below 300 KiB each. Commit
   regeneration scripts and compact JSON/CSV, not MOUT/WOUT/mgrid/trace output.
7. Run ruff, strict Sphinx, all mirror tests, example smoke, full CI, and one SSH
   office GPU benchmark. Compare CPU/GPU compile, steady runtime, peak memory,
   iterations, and gradients.

Gate: documentation reproduces every supported claim, CI is green, the branch
meets its size budget, and draft PR #22 is ready for scientific review.

## 6. Promotion matrix

| Model | Independent physics | residual/refinement | boundary physics | derivatives | example/docs |
|---|---|---|---|---|---|
| Axisymmetric fixed | cylinder, flared MMS, paraxial pressure balance | `ns`, knots | fixed geometry | JVP/VJP | required |
| Axisymmetric free | two-loop/ESSOS, fixed-limit, `sqrt(1-beta)` | plasma + exterior | `B.n`, pressure jump | JVP/VJP | required |
| Nonaxisymmetric fixed | rotating ellipse, SFLM, long-thin beta | `ns`, `mpol`, knots | fixed geometry | JVP/VJP | required |
| Nonaxisymmetric free | paraxial fixture and fixed-limit | plasma + all panels | `B.n`, pressure jump | JVP/VJP | required |
| Hybrid fixed | open-leg and smooth-torus limits | radial, poloidal, knots | fixed geometry | JVP/VJP | required |
| Hybrid free | fixed-limit plus ESSOS/MGRID | plasma + closed BIE | `B.n`, pressure jump | JVP/VJP | conditional |
| ANIMEC | source parity, isotropic limit, finite-beta observable | all active grids | `p_perp` jump | JVP/VJP | conditional |

## 7. Primary sources reviewed

- Hirshman and Whitson, *Steepest-descent moment method for three-dimensional
  magnetohydrodynamic equilibria* (1983), and STELLOPT/VMEC source:
  <https://princetonuniversity.github.io/STELLOPT/VMEC.html>
- Hirshman, van Rij, and Merkel, *Three-dimensional free boundary calculations
  using a spectral Green's function method* (1986):
  <https://www.osti.gov/servlets/purl/5272232>
- VMEC2000 source, especially `bcovar`, `forces`, `residue`, `fbal`, and
  `precon2d`: <https://github.com/PrincetonUniversity/STELLOPT>
- DESC master and branches `mirror`, `mirror_anisotropy`,
  `finite_element_basis`, `finite_element_basis_alan`, and `dd/cylindrical`:
  <https://github.com/PlasmaControl/DESC>
- Rodriguez, Helander, and Goodman, *The maximum-J property in
  quasi-isodynamic stellarators*, Appendix C (2024):
  <https://doi.org/10.1017/S0022377824000345>
- Rodriguez et al., *Constructing precisely quasi-isodynamic magnetic fields*
  (2023): <https://doi.org/10.1017/S0022377823001125>
- Agren and Savenko, *Theory of the straight field line mirror* (2005):
  <https://info.fusion.ciemat.es/OCS/EPS2005/pdf/P4_069.pdf>
- Goodman, Freidberg, and Lane, *Analytic mirror equilibria with new long-thin
  terms* (1986): <https://doi.org/10.1063/1.865851>
- Pearlstein, *Three-dimensional equilibrium in quadrupole symmetric tandem
  mirrors in paraxial limit*, UCRL-89767 (1983):
  <https://digital.library.unt.edu/ark:/67531/metadc1102940/>
- Pastukhov, *Finite beta plasma equilibrium in toroidally linked mirrors*
  (1993): <https://digital.library.unt.edu/ark:/67531/metadc1384995/>
- Cooper et al., *Three-dimensional anisotropic pressure free boundary
  equilibria* (2009):
  <https://pure.mpg.de/pubman/item/item_2141784_1/component/file_2141783/Cooper_Three.pdf>
- Asahi et al., *MHD Equilibrium Analysis with Anisotropic Pressure in LHD*
  (2011): <https://doi.org/10.1585/pfr.6.2403123>
- Skene and Burns, *Fast automated adjoints for spectral PDE solvers* (2025):
  <https://arxiv.org/abs/2506.14792>
- JAX implicit linear solve:
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html>
- JAXopt implicit differentiation: <https://jaxopt.github.io/stable/implicit_diff.html>
- Lineax: <https://arxiv.org/abs/2311.17283>
- SOLVAX source and contracts: <https://github.com/uwplasma/SOLVAX>

## 8. Immediate execution order

1. Finish Milestone 5 with a bounded radial/poloidal/knot study, finite-beta
   continuation, and spline implicit JVP/VJP validation. Promote the converged
   coefficients and observables; retain full-mesh magnitude only as non-gating.
2. Attempt Milestone 6 on the existing theta-dependent exterior solve with a
   coefficient-native plasma state. Run the two bounded vacuum formulations and
   either promote converged refinement evidence or defer the public lane.
3. Implement and gate the fixed closed hybrid in Milestone 7. Attempt its free
   boundary only after the fixed limits pass.
4. Complete the structured preconditioner, derivative, and ANIMEC decisions in
   Milestone 8, then execute the deletion, documentation, GPU, and CI gates in
   Milestone 9. No new physics lane is added before these finite decisions.
