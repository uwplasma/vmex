# Mirror equilibrium release plan

Status: final authoritative implementation plan for draft PR #22, revised
after the 2026-07-15 source, literature, and worktree audit. This file
supersedes the original `/Users/rogeriojorge/Downloads/plan_mirror.md` and
every earlier version of this plan. Do not create parallel roadmaps. Commits,
tests, and the four compact benchmark JSON files are the execution log.

Audit baseline (2026-07-15 CDT, final source/literature/worktree review through
T8 and the T7 promotion record):

- pushed head is `94addb68` on `codex/mirror-geometry`; base `origin/main` is
  `ed4ac7ac`, and the branch is zero commits behind and 334 ahead. It includes
  main's coil-ownership cleanup, simultaneous boundary/coil derivative work,
  and solver-sensitive finite-difference policy. No main merge is pending;
- draft PR #22 is open and mergeable. At the audit, 12 jobs on `94addb68` had
  completed successfully, the manual full-physics job was skipped as intended,
  and one parity shard was still running. CI is inspected in batches, not
  polled between implementation steps;
- the pushed diff has 53 files, 17,505 insertions, and 1,616 deletions. The
  mirror package has 13 modules and, after T9a, 8,052 lines; mirror tests have
  10 files and 4,492 lines; `splines.py` has 1,087 lines; the lazy public API
  has 20 names.
  All exceed at least one final reduction target, so T11 is mandatory;
- the reviewed T7 promotion tranche contains only the schema-5 benchmark,
  example labels,
  documentation, support-ceiling test, final plan reconciliation, and revised
  compressed figure. No raw GPU run tree is in git;
- public `ntheta` is gone. `mpol` is the highest retained poloidal Fourier mode
  and `ntheta = 2*mpol+1` is derived. The fixed and free production states,
  restarts, tangent, and adjoint are coefficient-native cubic B-splines;
- the strong-force diagnostic reconstructs covariant field, current, and
  pressure from the radial Gauss cells independently of the variational
  residual. Fixed axisymmetric and nonaxisymmetric open mirrors pass exact,
  paraxial, three-grid, and derivative gates at `ftol <= 1e-12`;
- T6's square free-boundary residual contains Galerkin boundary work,
  coefficient energy gradients, and optional pressure calibration. Above the
  tiny oracle threshold it uses exact JVP/VJP actions rather than a dense
  coupled Jacobian. Repeated actions measured 129 s and 1.96 GiB versus 188 s
  and 3.28 GiB for cached `jax.linearize` on the representative four-beta case;
- T7 found and repaired self-similar overwrite of the internal end-cut
  surfaces. The combined `(ns,ntheta,nxi)=(5,8,7)->(7,12,13)->(9,16,17)`
  sequence gives fine all-volume force `2.08e-3` at beta 0, `1.44e-2` at 10%,
  `3.37e-2` at 25%, and `6.69e-2` at 50%. Independent radial/angular families
  fail the declared `5e-2` gate at 25%, so the supported ceiling is 10%; 25%
  and 50% remain clearly labeled scalar-pressure research states. Fine
  `B_center/B_vac` is `0.99569`, `0.98702`, and `0.95620` at 1%, 3%, and 10%,
  within about `0.03%`, `0.10%`, and `0.36%` of the pinned Pleiades trend;
- T8 maps mirror flux to matched VMEC `APHI/PHIEDGE`, repairs nonuniform WOUT
  flux integration, and enforces smooth polar-axis regularity. The circular
  `ns=5,9,17` all-volume force sequence is
  `4.72e-6, 7.35e-7, 1.25e-7`; `ns=17` bulk force is `3.44e-9`; all solves
  reach `ftol <= 1e-12`. The greater-than-1,024-variable cyclic factor gate
  passes in about 7.4 s. The complete bounded mirror suite passes 105 tests
  with 9 expected full/nightly skips in 245.34 s; strict Sphinx, pre-commit,
  and `git diff --check` pass;
- direct sampling of analytic polar sections on an odd angular grid can alias
  unresolved even harmonics into retained odd modes. The SFLM regression uses
  an oversampled projection through the declared modes and retains an explicit
  aliasing test;
- T9a replaces the global weighted-pivot solve coordinate with one fixed
  physical stream coefficient per closed radial surface and restores the
  weighted mean only on output. Disjoint-support coloring reduces a refined
  16-control graph from 712 columns to 145 probes (`4.91x`). The finite-current
  graph uses 433 rather than 1,004 probes; cold linearization plus sparse
  factor setup takes 1.99 s at 1.21 GiB. The complete finite-current racetrack
  smoke reaches `ftol=1e-12` in 56.1 s below 3 GiB with nonzero iota, but its
  all-volume strong force is `0.548`. T9's blocker is now staged physical
  continuation and refinement, not factor construction.

Execution update (2026-07-15): T1--T9a are complete. They established one
matrix-free coefficient path, physical supplied-field initialization, the
local spline factor, promoted fixed-open analytic/paraxial records, the square
free residual and implicit derivative, the 10% axisymmetric free-boundary
ceiling, matched VMEC circular parity, closed-axis regularity, and the scalable
colored closed factor. The remaining sequence is T9b--T9d closed-hybrid
promotion, T10 bounded 3D
free-boundary disposition, T11 deletion, and T12 release audit.
After T9a, the complete normal mirror suite passes 109 tests with 9 expected
full/nightly skips in 242.60 s; the finite-current racetrack and large circular
full gates pass separately in 55.92 s and 13.20 s.

The branch is not release-ready. Its remaining physics blocker is strong-force
and refinement evidence for the finite-current closed hybrid; its conditional
research blocker is the 3D free
boundary; and its repository blocker is excess files, tests, public names, and
figures. Artificial end disks remain only exterior integration closures and
are tested by cap compatibility and cut-location independence; they are not
plasma boundaries. No downstream derivative, plot, or output can promote a
lane whose primal force/refinement gate fails.

### Final technical decisions

This audit converts the literature and source review into seven implementation
decisions. They are constraints on the milestones, not optional alternatives.

| Topic | Decision | Evidence and consequence |
| --- | --- | --- |
| Open longitudinal representation | Keep coefficient-native clamped cubic B-splines | Local support, exact endpoint constraints, and knot refinement fit a long straight mirror better than axial Fourier modes. Chebyshev remains independent quadrature/validation, not an optimized state. |
| Closed hybrid representation | Keep periodic cubic B-splines along a smooth racetrack axis and Fourier only around each section | A transported frame represents long straight legs and rotating elliptical returns with few section modes. VMEC-like longitudinal Fourier projection is used only for circular-limit parity, never as the production hybrid state. |
| Primal CLI solver | Keep SciPy host L-BFGS/least-squares, GMRES, and sparse LU where measured faster | The CLI is explicitly nondifferentiable. T4 reduces the SFLM true linear residual from `7.83e-2` to `9.18e-11`, cuts Krylov work from 2,000 to 660, and reduces wall time from 4.53 s to 4.14 s. |
| Differentiation | Differentiate only the converged coefficient residual with exact JVP/VJP actions | Forward implicit tangents serve few controls; reverse adjoints serve scalar outputs/many controls. T6b selected repeated JVP/VJP actions after cached `jax.linearize` retained substantially more memory and ran slower on the representative free scan. Unrolled solver AD and derivatives of unconverged states are rejected. |
| Generic solver ownership | Use released SOLVAX APIs only when an A/B change is a net deletion and preserves safeguards and diagnostics | SOLVAX 0.8.3 supplies matrix-free Newton--Krylov, pytree GMRES, and cyclic/banded operators, but its root solver has no bounds or line search. The bounded free-boundary CLI may therefore retain SciPy trust-region least squares; primal/transpose Krylov wrappers should move to SOLVAX when parity and true-residual gates pass. Mirror geometry, gauges, residuals, continuation, and sparsity remain in `vmec_jax`. Do not add JAXopt, Lineax, Optimistix, Optax, or a direct Equinox dependency in this PR. |
| Physics scope | Finish scalar-pressure fixed open, axisymmetric free open, and fixed closed hybrid; bound the 3D free-open attempt; defer ANIMEC | ANIMEC changes the energy, pressure closure, effective current, interface stress, and stability constraints. It is not a second pressure array and cannot be added safely inside this PR. |
| Analytic fixtures | Keep SFLM and rotating ellipse as separate models | The Agren--Savenko SFLM changes ellipticity without a prescribed rigid 90-degree section rotation. The Rodriguez--Helander--Goodman paraxial fixture rotates an ellipse. Combining their names or acceptance quantities would create a false literature comparison. |

The go/no-go outcomes are therefore explicit:

- **go**: axisymmetric and nonaxisymmetric fixed open mirrors, axisymmetric
  free open mirrors through beta 10%, and the fixed closed spline hybrid;
- **bounded attempt**: nonaxisymmetric free open mirrors, with promotion or
  deletion of public scaffolding at M6;
- **defer**: free-boundary hybrid, arbitrary curved open axes, anisotropic
  ANIMEC physics, kinetic closures, stability, and mirror Boozer output;
- **remove**: coil construction, Biot--Savart formulas, ESSOS-owned runners,
  stale shaped records, and any solver path that misses its measured gate.

## 1. Release contract

PR #22 will support these scalar-pressure equilibrium models:

1. straight-axis, fixed-boundary, axisymmetric mirrors;
2. straight-axis, fixed-boundary, nonaxisymmetric mirrors, including both the
   Agren--Savenko straight-field-line mirror and the independent 90-degree
   rotating-ellipse paraxial case;
3. straight-axis, axisymmetric free-boundary mirrors in a supplied external
   field, supported through beta 10% with labeled research continuation at
   25% and 50%;
4. fixed-boundary toroidal stellarator--mirror hybrids represented by periodic
   cubic B-splines, with two long straight mirror legs connected by two smooth
   stellarator returns.

All four models use coefficient-native cubic B-splines for every optimized
longitudinal geometry and stream-function degree of freedom. The open models
use clamped splines and the closed hybrid uses periodic splines. This is the
single production representation; nodal CGL states remain only as quadrature,
exterior-collocation, manufactured-test, and migration references.

Differentiability is part of a supported model, but it follows the primal
physics gates. The fast CLI may use SciPy host optimization and need not be
differentiable. A separate implicit layer differentiates the converged
coefficient residual. Unrolled differentiation through nonlinear iterations,
host callbacks presented as end-to-end JAX solves, and derivatives of an
unconverged state are outside the release contract.

One lane is conditional:

- nonaxisymmetric open free boundary receives one bounded promotion attempt
  after the strong-force and structured-preconditioner milestones. If it
  cannot pass a third-grid refinement gate within the stated resource budget,
  retain a compact negative benchmark, keep it out of the public API, and
  defer it without blocking the four required models.

The following are explicitly deferred to later PRs:

- free-boundary closed hybrids and their coil coupling;
- anisotropic pressure and ANIMEC physics;
- arbitrary curved open axes;
- kinetic end losses, sheaths, transport, stability, islands, and stochastic
  fields;
- mirror Boozer output and use of open mirrors in toroidal WOUT consumers;
- coil construction and Biot--Savart calculations, which belong in ESSOS.

A supported result is a converged nested-surface ideal-MHD equilibrium. A
sampled coil field, prescribed beta-dependent tube, small `B.n` without a
plasma-force solve, optimizer success flag, or Fourier fit of a square is not
an equilibrium result.

### 1.1 Promotion gates

Every supported lane must pass all applicable gates:

- component-wise normalized variational residual at or below `1e-12`;
- a documented double-precision exception no larger than `1e-11` only after a
  resolution study demonstrates the numerical floor;
- an independently assembled staggered weak first variation converging to the
  same floor without differentiating the production energy;
- an independently reconstructed `J x B - grad(p)` residual that passes exact
  manufactured tests and decreases under physical refinement. For the open
  lanes, the finest all-volume value must be below `5e-2`, the central core
  below `2e-2`, and Richardson extrapolation must be consistent with zero;
- positive Jacobian, nested surfaces, adequate self-clearance, and normalized
  `div(B)` near roundoff;
- physical observables stable on three independently refined grids, assessed
  by observed order or Richardson extrapolation and a predeclared tolerance;
- for open free boundary, separately reported area-weighted `B.n`, total
  pressure jump, and artificial-cap compatibility residuals;
- for open free boundary, a quadrature-weighted boundary-work residual pulled
  back to the free spline coefficients. Pointwise pressure jump remains an
  independent diagnostic and is never substituted for discrete stationarity;
- an analytic, asymptotic, or independent-code comparison;
- forward and reverse implicit derivatives for every advertised control,
  checked against reconverged centered finite differences after the primal
  lane passes all preceding gates. The true primal or transpose linear
  residual must be at most `1e-8`;
- one parser-free root example, one compact benchmark record, current docs,
  and compressed publication-quality figures.

The nonlinear variational residual, weak residual, strong force, and boundary
constraints are distinct diagnostics. None may be relabeled as another.

Observable refinement tolerances are fixed before each benchmark run. The
default is an extrapolated relative error below 0.5% for on-axis field and
central radius; a tighter 0.1% target is used when the observed order supports
it. A single pairwise difference, especially a historical 0.05% field target,
is not a valid convergence claim.

## 2. Physical models

### 2.1 Open straight mirrors

The coordinates are

`(s, theta, xi) in [0,1] x [0,2*pi) x [-1,1]`.

The axis is straight, `theta` is periodic, and `xi` is nonperiodic. The
lateral surface `s=1` is the plasma--vacuum interface. The planes at
`xi = +/-1` are fixed computational cuts crossed by magnetic flux; they are
not material interfaces and must not impose `B.n = 0`.

For a symmetric mirror, both cuts receive the same axisymmetric prescribed
section and compatible through-flux, with opposite outward normals. More
general fixed-boundary fixtures may prescribe different end sections, but the
values and stream data are explicit Dirichlet continuation data, not equations
invented by the optimizer. Moving the cuts outward while retaining the same
central physical field must leave central observables unchanged to the
declared refinement tolerance. The artificial disks used to close the exterior
BIE carry only Neumann compatibility data; total-pressure and tangency
conditions are enforced on the lateral plasma interface, not on those disks.

The production representation is Fourier in `theta`, VMEC-like and staggered
in `s`, and coefficient-native clamped cubic B-spline in `xi` for boundary,
interior geometry, and stream function. Chebyshev--Gauss--Lobatto nodes remain
an independent collocation, quadrature, exterior-panel, and validation
representation. “Full B-spline mirror” means that every optimized longitudinal
geometry and stream coefficient uses the same spline space; it does not mean
replacing the periodic poloidal angle or radial nested-surface mesh with
splines.

Regularity is a physical constraint, not post-processing:

- all `m > 0` geometry and stream coefficients vanish with the correct power
  of radius at the magnetic axis;
- scalar axis values are single-valued in `theta`;
- stream-function gauges are removed before optimization and linear solves;
- end values and end derivatives follow the declared fixed-cut conditions.

The free-boundary solve evaluates spline coefficients on the existing CGL and
panel nodes before calling the one cap-aware exterior backend. Shape
derivatives pass through this linear evaluation map. Promotion requires tests
of endpoint value and derivative constraints, cap-rim continuity, knot
refinement, cut-location independence, and shape derivatives. Do not create a
second spline-specific BIE. Regional force masks diagnose the central 80% and
outer 20% collars separately, but the all-volume norm remains a release gate.

### 2.2 Closed stellarator--mirror hybrid

The hybrid remains toroidal. Its magnetic axis is one smooth periodic curve
with two long nearly straight mirror legs and two curved stellarator returns.
Periodic cubic B-splines describe the centerline, Bishop-frame section shape,
and longitudinal stream coordinates. Fourier modes describe the periodic
cross-section angle; the radial coordinate remains VMEC-like.

The construction must provide:

- periodic position, tangent, frame, and spline derivatives through the joins;
- up--down and leg-exchange symmetries as coefficient maps, not duplicated
  geometry;
- positive section Jacobian and clearance along the entire circuit;
- a circular-axis limit matching ordinary `vmec_jax` and VMEC2000;
- a long-leg local limit matching the fixed open B-spline mirror;
- a rotating noncircular section in the returns capable of generating
  rotational transform.

Fixed-boundary beta scans change the interior equilibrium surfaces while the
LCFS stays fixed by definition. Claims of beta-driven LCFS motion belong only
to a future free-boundary hybrid.

### 2.3 Pressure model

This PR retains scalar pressure `p(s)`. ANIMEC is not equivalent to adding two
pressure arrays: its energy depends on `p_parallel(s, B)`, includes
`p_perpendicular`, anisotropy `sigma`, effective-current terms, fixed-`B`
derivatives, and firehose/mirror constraints. Current STELLOPT source also
threads these quantities through initialization, timestep, strong-force,
free-boundary, and output paths and reports separate `sigma`/`tau` stability
limits. A partial implementation would be misleading and is removed from this
scope.

Requested beta is defined against the vacuum reference field stated in each
benchmark. Achieved on-axis and volume-averaged beta are reported separately.
The scalar solver can converge research states at beta 50%, but the current
strong-force/refinement evidence supports only beta 10%. Neither status may be
described as a quantitative model of high-beta mirror experiments whose
pressure is strongly anisotropic.

## 3. Current evidence ledger

### 3.1 Accepted evidence

- Axisymmetric fixed boundary reaches `ftol = 1e-12` and passes cylinder and
  flared-tube analytic checks, three-grid spline/CGL parity, an independent
  weak residual, physical observable refinement, and implicit derivative
  tests.
- The coefficient-native open spline representation reproduces CGL geometry,
  energy, and fields with decreasing error while using fewer active axial
  unknowns.
- Spline reverse derivatives agree with reconverged finite differences at
  about `3e-10`, and the forward implicit tangent test passes on small fixed
  systems. This validates the method, not any primal lane that still fails
  force balance.
- A finite-beta radial pressure-balance manufactured case and the VMEC-style
  half-to-full reconstruction show second-order radial convergence for a
  cylindrical polynomial state with nonzero pressure, current, and lambda.
  The current-free circular first-row force decreases by about four on each
  radial refinement from `ns=5` through 17.
- A nonaxisymmetric shaped coordinate map with uniform Cartesian field has
  force below `1e-12`.
- The coefficient-native SFLM field initializer accepts callable or sampled
  Cartesian fields, infers the analytic flux within `6.3e-5` relative, passes
  a field-amplitude JVP, and reaches field/force errors below
  `5e-4`/`6e-3`. Its analytic polar section is oversampled before truncation
  so unresolved even harmonics do not alias into axis-regular odd modes.
- The equal-end polynomial vacuum mirror is exactly curl-free and
  divergence-free. Moving its cuts through half-lengths `0.6, 0.8, 1.0` changes
  central radius by only `1.6e-9` relative and central field by `1.0e-5`, while
  all solves retain variational/weak residuals below `2e-15` and strong force
  below `5.7e-3`.
- The rotating-ellipse combined sequence `(ns,mpol,elements)=(5,4,4),
  (7,6,6),(9,8,8)` reduces all-volume strong force
  `6.39e-2 -> 2.41e-2 -> 9.76e-3`, with a zero-limit fit of order `2.69`.
  The SFLM sequence `(7,6,6),(9,8,8),(11,10,10)` reduces it
  `4.18e-2 -> 2.11e-2 -> 1.09e-2`, with order `2.63`. Bulk force is below
  `2e-2` on accepted grids and both extrapolations are consistent with zero.
- Fine rotating/SFLM volume adjoints agree with reconverged centered finite
  differences to `1.7e-10`/`2.0e-10`; tangent and transpose true residuals are
  below `7.4e-10`. The finest primal true residuals are `3.6e-11` and
  `2.4e-9`, respectively.
- Axis regularity now enforces a single-valued axis `|B|` and a consistent
  derivative pullback.
- The schema-5 axisymmetric free-boundary matrix reaches small variational,
  weak, interface, cap-compatibility, and divergence residuals. Independent
  radial, axial, exterior, angular, and combined refinements promote beta
  through 10%; 25% and 50% fail at least one strong-force or observable gate.
  At 10%, the fine all-volume/core force is `1.44e-2`/`1.62e-3` and
  `B_center/B_vac` is `0.95620`.
- Periodic cubic racetrack geometry, Bishop transport, circular geometry, and
  field tracing have focused tests.
- Matched circular periodic inputs agree among the mirror solver, ordinary
  vmec_jax, VMEC2000, and WOUT flux conventions. The `ns=5,9,17` all-volume
  force sequence is `4.72e-6, 7.35e-7, 1.25e-7`, with `3.44e-9` bulk force on
  the fine grid and nonlinear residuals at or below `1e-12`.
- Pleiades data are independently generated from commit
  `0161abb3e9a1d85143c650f068ec524d672fc9ab` and provide low-beta external-code
  evidence at 1%, 3%, and 10%.

### 3.2 Current canonical records and remaining regeneration

Any shaped JSON from before the axis-regularity correction, exact `mpol`
semantics, or accepted M1 force reconstruction remains invalid. The canonical
fixed-open schema-4 record and axisymmetric free schema-5 record have now been
regenerated under the current representation. Circular parity is current in
tests and the T8 execution record, but the closed-hybrid JSON must be replaced
only after racetrack/current continuation, the open-leg limit, derivatives,
and MOUT pass T9. The nonaxisymmetric free record is retained as research
evidence until T10 either promotes it or replaces it with a compact negative
record. Never carry a positive status forward from an incompatible discrete
space.

The detailed mirror documentation and beta-scan tutorial now distinguish the
10% supported ceiling from the 25%/50% research continuation and narrow the
Pleiades comparison to on-axis field depression. The repository README still
needs the final capability table and three-workflow showcase. T12 publishes
only the lanes that pass their final gates.

### 3.3 Rejected or incomplete evidence

- In the regenerated nested-cut axisymmetric free beta scan, combined
  refinement reduces beta-50% pointwise force
  `0.130 -> 0.111 -> 0.0669`, but this is not yet a demonstrated zero-limit;
  the high-beta point remains unpromoted despite small variational and
  boundary residuals.
- At 50%, the medium/fine central radius differs by about 0.137%, while the
  central field differs by about 1.02%. The old 0.05% field target was
  unjustified; the replacement is the observed-order/extrapolation gate in
  section 1.1.
- Nonaxisymmetric free-boundary local `m=1` coefficients change by 73--81%; a
  medium pair took about 801 seconds and 4.25 GiB. It is a research result, not
  a supported equilibrium.
- The rotating-ellipse fine-grid `m=2` response differs by about 4.2% from the
  first-order paraxial estimate and by 8.7% in the half-radius run. The analytic
  section supplies only first-order transverse data, so this coefficient is a
  diagnostic, not a promotion gate, until a second-order compatible end field
  is derived. Flux conservation, section angle, field tangency, forbidden
  `m=1`, and strong-force refinement remain gates.
- The old zero-stream continuation basin was demonstrably wrong for both
  nonaxisymmetric vacuum fixtures. Those large-force values remain historical
  negative evidence only; the supplied-field initializer and T5 refinement
  sequences supersede them.
- The old periodic preconditioner stalled at 3,000 GMRES iterations with
  linear residual about 0.136; CG/MINRES reached only about 0.016. T9a
  supersedes it with the local-gauge colored factor. The finite-current
  racetrack now reaches true linear residual `5.29e-11` and iota `0.0422`, but
  its all-volume strong force is `0.548`; this is computational convergence,
  not T9b physics promotion.
- Circular hybrid VMEC parity is accepted. There is still no accepted
  racetrack/current continuation, open-leg limit, release MOUT, or hybrid
  implicit-derivative benchmark.

Failed diagnostics must remain visible. Do not tune tolerances, omit boundary
collars, or change normalizations after seeing results without documenting the
reason and rerunning all comparison grids.

## 4. Conclusions from external sources

### 4.0 Reproducible source-review ledger

The final review used source, tests, and branch history rather than project
descriptions alone:

Each source has one evidence role. A **design reference** can justify a
discretization or algorithm but cannot validate mirror physics. A
**comparator** must solve the same physical model on reproducible matched
inputs. A **physics reference** supplies an exact or asymptotic observable only
inside its stated ordering. No result changes role merely because its plot
looks similar.

| Source | Revision reviewed | Adopt | Reject or defer |
| --- | --- | --- | --- |
| `vmec_jax` main | `ed4ac7ac`; mirror branch is 0 behind | existing VMEC residual separation, continuation diagnostics, exact JVP actions, SOLVAX-backed toroidal Krylov patterns, coil-agnostic external-field ownership, and solver-sensitive FD policy | forcing open topology through toroidal Fourier/WOUT conventions or restoring deleted coil ownership |
| VMEC2000/ANIMEC in STELLOPT | current `develop` `e03e72e9`; `animec` `91bfd08e`; `animec_adjoint` `561f430b`; `forces.f`, `residue.f90`, `bcovar.f`, `fbal.f`, `jxbforce.f`, `precon2d.f` | radial staggering, separate raw/preconditioned diagnostics, neighboring-radial block physics, explicit one-sided axis/edge handling, full anisotropic closure inventory | VMEC closed-LCFS boundary conditions on open cuts; treating old ANIMEC branches as a maintained comparator |
| DESC master/release | `24aa7b9dc`; latest release `v0.17.2` at `d454fb47` | released continuation, matrix-free JVP memory lessons, and closed-interface boundary-condition separation | importing another optimizer/objective stack into this PR |
| DESC `mirror` | `0dba071da` | Chebyshev--Zernike formulas and the need for explicit cap constraints as review material | direct code transfer: 2,255-line equilibrium and 1,393-line objective modules, placeholder end-cap helpers, assertion-free cap test, disabled upstream tests |
| DESC `mirror_anisotropy` | `805b77fc0` | confirms that mirror anisotropy is a distinct closure problem | treating branch results as validated ANIMEC parity |
| DESC `dd/cylindrical` | `6f85f50ae` | double-nonperiodic-Chebyshev/Fourier basis formulas | changing basis: branch has initial basis tests but no mirror solve, cap, axis, force, or free-boundary evidence |
| DESC finite-element branches | `829e2db0f`, `fb90e65ab` | none required | incomplete JAX integration, debugging artifacts, and no validated mirror equilibrium |
| SOLVAX | installed and published `0.8.3`; main `255d280`; untagged release branch `4808695` | generic JAX Krylov/direct/implicit wrappers when they delete local code and preserve safeguards | replacing bounded/trust-region physics drivers with an unsafeguarded root step; upgrade for feature count alone |
| GVEC/G-frame work | arXiv:2410.17595 and current docs | transported-frame geometry, independent quadrature, compact section coordinates | claiming its radial B-splines validate longitudinal splines directly |
| VEPEC report | UCRL-53099 | divergence-preserving potential variables and spline interpolation precedent | numerical parity without reproducible inputs/code |
| Pleiades | `develop` `0161abb3`; `compute_equilibrium` and the pinned three-grid script | independent axisymmetric low-beta flux/current and on-axis diamagnetic-field trend at 1%, 3%, and 10% | LCFS-radius parity: the fixture fixes the 0.25 m midplane pressure support and does not solve the same moving-boundary problem |

The branch heads and tags above were rechecked directly with `git ls-remote`
on 2026-07-15. The review also inspected the current mirror modules, tests, examples, four
benchmark records, documentation, original Chebyshev plan, complete branch
history, draft PR metadata, and latest CI. External formulas become tests only
after their assumptions, normalization, and reproducible input are recorded.

### 4.1 VMEC2000, VMEC++, and ANIMEC

VMEC2000 source at current STELLOPT `develop` commit `e03e72e9` was
reviewed through
`forces.f`, `residue.f90`, `bcovar.f`, `fbal.f`, `jxbforce.f`, `precon2d.f`,
and the block-tridiagonal solvers. It separates raw force assembly from
preconditioned residuals and uses radial staggering consistently.
`precon2d.f` preserves neighboring radial, Fourier-mode, and field-component
coupling in block-tridiagonal radial factors, and can spill those factors to
disk when memory is insufficient, instead of reducing the system to scalar
diagonal scaling. These are the primary
references for the mirror strong-force reconstruction and coupled
preconditioner.

The concrete `jxbforce.f` pattern is the acceptance reference: covariant field
is stored on half surfaces, radial differences place current on full interior
surfaces, `sqrt(g) B^u` and `sqrt(g) B^v` are averaged before division by an
averaged Jacobian, and pressure uses a half-to-full difference. Axis and edge
values receive explicit one-sided treatment. VMEC's `fsqr/fsqz/fsql` remain
preconditioned variational residuals and are not interchangeable with this
pointwise diagnostic.

VMEC's fixed-boundary condition `B.n=0` applies to a closed material LCFS and
must not be copied onto open mirror cuts. VMEC2000 and VMEC++ cannot validate
open topology. They are independent references only for the circular
closed-hybrid limit and matched toroidal inputs. VMEC++ adds operational
lessons: restartable continuation, explicit residual history, bounded retries,
and broad VMEC2000 parity.

Cooper's 1992 variational formulation, the 2009 free-boundary ANIMEC paper,
later LHD reports, `animec` branch `91bfd08e`, `animec_adjoint` branch
`561f430b`, and the current `_ANIMEC` source paths confirm the scope decision
in section 2.3. These old branches are source-history references, not
maintained release comparators. The historical `animec` branch changes at
least eleven VMEC source files (827 insertions and 351 deletions in the
reviewed diff), rather than adding an isolated profile class. The implementation touches
the pressure profile construction, covariant field, force balance, strong
force diagnostic, interface condition, timestep path, and output schema. It
computes `p_parallel(s,B)` and `p_perpendicular`, uses
`sigma = 1 + (p_perpendicular-p_parallel)/(B^2/mu0)` in the effective current
`curl(sigma B)`, and adds profile derivatives evaluated at fixed `B` to radial
force balance. Its free-boundary condition is continuity of
`p_perpendicular+B^2/(2 mu0)`, not scalar pressure. ANIMEC remains a future
physics model, not a pressure-array option inside this scalar solver.

### 4.2 DESC branches

The public DESC `mirror` (`0dba071d`, 2025-09-12) and
`mirror_anisotropy` (`805b77fc`, 2025-11-12) branches provide useful
prototypes for Chebyshev--Zernike coordinates, end-cap constraints, and
continuation. They are not validated production references:

- mirror equilibrium and objective logic is largely duplicated in large
  branch-only modules;
- `FixEndCapR/Z` contains placeholder behavior and explicit mode selection is
  not implemented;
- the nominal mirror boundary-condition test is six lines long and contains
  imports rather than assertions;
- substantial upstream tests are renamed or disabled on the branch;
- notebook and binary artifacts account for branch diffs above one million
  inserted lines and make direct code transfer undesirable.

The branch heads were fetched and rechecked against current DESC master
`24aa7b9dc` on 2026-07-15 and have not advanced. The `mirror` diff spans 123
files and more than one million inserted lines because it includes notebooks,
binary arrays, and renamed upstream tests; the production additions alone
include a 2,255-line equilibrium class and a 1,393-line objective module.
Current released DESC retains
`ChebyshevDoubleFourierBasis` and `ChebyshevPolynomial`, but no released open
mirror equilibrium. This makes Chebyshev a useful independent collocation
oracle, not a second production state or a reason to duplicate DESC's branch
architecture here.

DESC's newer `dd/cylindrical` branch (`6f85f50a`, 2026-06-26) implements
`DoubleChebyshevFourierBasis`: two nonperiodic Cartesian Chebyshev coordinates
(`R` and `Z`) and one periodic Fourier angle. It is not an axial-Chebyshev
mirror equilibrium implementation. Its two commits test values and first and
second derivatives, explicitly leaving third derivatives unimplemented. It
has no mirror equilibrium, end-cap, axis-regularity, free-boundary, or
force-refinement evidence. The `finite_element_basis` branches last changed in
2024 and add
experimental scikit-fem paths, debugging output, and incomplete JAX
integration. Use formulas from these branches as review material only. None is
a reason to replace the compact Fourier/radial/longitudinal-spline tensor
product in this PR.

DESC's released free-boundary work is more relevant conceptually: on a closed
plasma--vacuum interface, tangency, total-pressure jump, and any sheet-current
condition are distinct equations. DESC minimizes those conditions subject to
force balance and evaluates virtual-casing/sheet-current fields with high-order
singular quadrature; this branch instead solves one cap-aware exterior Neumann
problem. The equations and diagnostic separation transfer, but the
smooth-toroidal quadrature and optimizer do not remove the open cap--rim
singularity or justify a second backend. The conditions apply only to the
mirror's lateral LCFS.

### 4.3 Mirror and numerical literature

The literature does not provide a general closed-form three-dimensional,
finite-beta, free-boundary open-mirror equilibrium that can serve as a single
gold file. Validation must therefore be triangular: exact vacuum/manufactured
solutions, paraxial asymptotics in their ordering, and an independent
axisymmetric code. For scalar-pressure open field lines, `B dot grad(p)=0`
requires pressure to be a flux function and the truncated-domain problem must
prescribe compatible end data. This supports the fixed-cut formulation but
makes cut-location independence a mandatory physics test, not a plotting
choice.

Ågren and Savenko's straight-field-line-mirror construction supplies exact
paraxial fixtures already represented in `analytic.py`:

- `x = x0 (1 + z/c)` and `y = y0 (1 - z/c)`;
- `B_axis = B0 / (1 - z^2/c^2)`;
- section ellipticity `(1 + |z|/c) / (1 - |z|/c)`;
- straight but nonparallel vacuum field lines, a marginal minimum-`B` field,
  and zero vacuum cross-field drift.

This SFLM section changes ellipticity but does not execute a prescribed
90-degree rigid rotation. The rotating ellipse in `RotatingEllipseParaxial`
comes from the independent near-axis construction in Appendix C of Rodriguez,
Helander, and Goodman. The two cases must have separate names, fixtures, and
error tables.

Related first-order finite-beta SFLM work gives the long-thin trend
`B approximately B_vacuum * sqrt(1 - beta)`. It supports an asymptotic trend
test and a first-order ellipticity correction, not an exact finite-beta
equilibrium benchmark. Goodman--Freidberg--Lane expands simultaneously in beta
and inverse aspect ratio and predicts additional quadrupole/diamond distortion;
that supplies section-shape observables at low beta and thin radius. The
rotating ellipse must first match its own vacuum paraxial limit as minor radius
and beta approach zero, then be tested away from that limit.

The Goodman--Freidberg--Lane comparison to VEPEC is especially useful because
it predicts both beta scaling and non-elliptical section distortion. The gate
must compare fitted quadrupole/diamond moments, not only a mean radius. No
paraxial formula is used as an accuracy target at beta 50%; that endpoint tests
the numerical scalar-pressure model and continuation only.

The linked-mirror configuration of Feng et al. supports the closed topology:
two straight mirror sections joined by two half-tori, with nonparallel
sections producing transform. It is a geometry/orbit reference, not an MHD
equilibrium comparator. The 1993 Ilgisonis--Berk--Pastukhov report solves a
free-boundary linked-quadrupole asymptotic model and predicts nonlinear outward
boundary displacement through order `(beta/(epsilon E))^2`. That displacement
cannot validate a fixed-LCFS hybrid. It is reserved for a future free-boundary
hybrid; only its low-beta internal multipole trends may be reported
qualitatively here.

Rodríguez, Helander, and Goodman's maximum-`J` paper contains useful paraxial
near-axis mirror equations in its appendix; it is not itself a
straight-field-line-mirror construction.

GVEC validates several architectural choices--coefficient-native B-splines,
independent quadrature, Fourier periodic angles, and a general transported
G-frame--but its published splines are radial rather than longitudinal. Its
G-frame study reduced a strongly shaped case from Fourier `(10,15)` to `(2,10)`
and from 16,000 to 800 iterations. It supports the Bishop-like hybrid frame and
the requirement for valid initial maps, not direct reuse of a GVEC
representation. VEPEC provides historical precedent for vector-potential
variables and divergence-preserving tricubic spline interpolation in high-beta
minimum-`B` mirror studies.

Goodman, Freidberg, and Lane expand simultaneously in beta and the long-thin
parameter and compare section distortions with VEPEC. Their formulas are an
asymptotic low-beta/thin-tube gate, not a beta-50 parity result. Near-unity-beta
diamagnetic-bubble and RealTwin studies generally require anisotropic or
kinetic closures, so they cannot validate the scalar-pressure model
quantitatively.

Pleiades remains the only independent open-mirror numerical comparison in the
current branch, but its role is narrower than previously stated. The pinned
`compute_equilibrium` fixture iterates an axisymmetric Grad--Shafranov
flux/current response while defining the pressure support from a fixed 0.25 m
midplane radius. It therefore validates low-beta on-axis field depression and
grid/iteration trends at 1%, 3%, and 10%; it does not provide independent
free-LCFS radius or shape parity. The 25% and 50% states must be judged by
vmec_jax refinement, force/interface residuals, and asymptotic trends only.
RealTwin's high-field tandem-mirror study and diamagnetic-bubble literature
provide qualitative high-beta context, including pressure anisotropy and
difficult near-unity-beta behavior; neither is scalar-pressure numerical
parity. Beta 50% is a demanding validation point, not a claim of a
diamagnetic-bubble model.

### 4.4 SOLVAX and differentiation

The branch environment now runs the published SOLVAX `v0.8.3`. SOLVAX current
`main` (`255d280`, 2026-07-14) and the untagged `release/0.8.4` branch
(`4808695`) add pytree GMRES, matrix-free Newton--Krylov, cyclic tridiagonal
solves, symmetric additive and Galerkin preconditioners, elliptic helpers,
host SuperLU wrappers, independent transpose solvers, and extended
diagnostics. `vmec_jax` already uses SOLVAX block Thomas in the toroidal path.
Generic linear algebra belongs in SOLVAX, but mirror-specific geometry,
residuals, gauges, coefficient maps, continuation, and physics sparsity stay
here.

The T4 isolated trial applied current-main SOLVAX right-preconditioned FGMRES
to the same 591-variable SFLM Hessian and tensor inverse. Its true-residual
iteration curve matches SciPy's host GMRES. SOLVAX 0.8.3
`newton_krylov` is fully matrix-free but takes unconstrained full Newton steps;
it does not replace the bound-preserving, trust-region free-boundary CLI
without a measured basin-of-attraction study. The nondifferentiable CLI may
therefore keep faster host sparse LU, GMRES, and bounded least squares. Exact
JAX JVP/VJP actions retain differentiability in the implicit layer.

T6/T11 may replace local generic host-factor and implicit-linear wrappers with
released SOLVAX APIs only if the change is a net deletion, preserves auxiliary
convergence diagnostics and independent transpose control, and passes the
same primal/tangent/adjoint A/B records. SOLVAX 0.8.4 is not used until it is a
tagged package. The mirror sparse pattern builder is physics-specific and
remains here.

The production derivative strategy remains:

- solve the nonlinear equilibrium to the declared primal tolerance;
- differentiate the converged residual with exact JAX JVP/VJP actions;
- use forward implicit tangents for few controls;
- use reverse adjoints for scalar outputs with many controls;
- solve primal and transpose systems with the same physical preconditioner;
- never reverse-differentiate through the nonlinear iteration history;
- validate against fully reconverged centered finite differences and report
  both nonlinear and linear residuals.

Skene and Burns' sparse-spectral adjoint construction strengthens this choice:
the efficient object is the converged sparse residual graph and its transpose,
not the nonlinear iteration trace. JAX supplies exact JVP/VJP actions here;
the mirror coefficient map and structured factor supply the sparsity that a
generic AD package cannot infer. JAX documents that `jax.linearize` avoids
relinearization across repeated JVPs but stores partial-evaluation data with a
memory cost resembling reverse mode. T6b's measured A/B selected repeated
`jax.jvp`/`jax.vjp`: 129 seconds and 1.96 GiB versus 188 seconds and 3.28 GiB
for cached actions on the representative four-beta case. This is a measured
choice, not a claim that repeated actions are universally faster. JAXopt,
Lineax, and Optimistix remain possible future
wrapper reductions only when an A/B change deletes local generic code,
preserves independent transpose control, and passes the same derivative and
memory gates.

JAX `custom_vjp`, `lax.custom_root`, and `linearize`, JAXopt `custom_root`,
Optimistix, and Lineax all encode variants of the same implicit-function solve.
Lineax offers useful operator abstractions and transpose-aware solver state,
while Optax supplies first-order optimizers and Equinox supplies PyTree
modules; none provides the missing mirror block physics. The Skene--Burns
spectral-adjoint method confirms that sparse operator graphs and transposed
solves are the scalable reverse-mode design, but its symbolic Dedalus graph is
not portable here. The research-grade choice is therefore exact JAX residual
JVP/VJP actions plus an implicit primal/transpose solve, not unrolled AD. Add no
new optimization or linear-solver dependency in this PR.

## 5. Architecture and simplification rules

The current physical ownership boundaries are sound:

| Module | Owner |
| --- | --- |
| `model.py`, `basis.py`, `splines.py` | input contract, bases, coefficient maps, regularity |
| `geometry.py`, `analytic.py` | coordinate metrics and validation solutions |
| `forces.py` | energy, weak residual, strong-force diagnostics |
| `solver.py` | fixed-boundary nonlinear solve and continuation |
| `exterior.py`, `exterior_bie.py` | open-vacuum fields and cap-aware shape calculus |
| `free_boundary.py` | coupled open free-boundary solve |
| `implicit.py` | converged-state tangent and adjoint solves |
| `output.py` | MOUT serialization, diagnostics, and plotting data |

Do not collapse unrelated physics into one file. Simplification means removing
duplicated packing, gauges, linear actions, and configuration fields; using
one shared implementation per operation; and deleting failed public
scaffolding. It does not mean hiding distinct equations behind generic names.

T2 and T6c already removed the private nodal fixed solve, nodal custom VJP,
nodal vectorizers/preconditioners, interpolation compatibility path, and
duplicate live-solve tests. Do not restore them. Nodal arrays remain evaluation
fixtures, never solver unknowns or restart state.

The final source audit identifies these remaining reductions:

- keep the private free-equilibrium problem in `free_boundary.py` as the
  residual/operator/result seam; do not split it into new modules merely to
  reduce per-file size;
- retain its one coefficient residual in primal, tangent, and adjoint paths;
  delete any diagnostic wrapper that reconstructs the same block vector;
- retain T6b's JVP/VJP coupled operator and tiny-only dense oracle. The
  exterior Laplace BIE remains a dense globally coupled operator;
  it is measured separately and is not mislabeled matrix-free;
- retain coefficient-native free results and schema-3 restarts. Evaluated CGL
  arrays have explicit `evaluated_*` names and exist for diagnostics/MOUT, not
  as a second restart state;
- retain one exterior BIE. `exterior.py` owns closed panels and quadrature;
  `exterior_bie.py` owns the Laplace solve and field coupling;
- retain externally supplied `coil_xyz` only as optional output metadata;
  delete `benchmarks/run_mirror_exterior_endpoints.py` after its canonical
  record is reproducible from the root free-boundary example or an ESSOS-side
  runner. No coil constructor or Biot--Savart formula returns to vmec_jax;
- remove private imports from root examples. Promote a helper only when two
  retained examples need the same operation and its ownership is scientific,
  not merely plotting convenience;
- combine duplicate output summaries and restart validation inside
  `output.py`; retain one schema writer, one reader, and one schema-2 migration;
- consolidate repeated spline fixtures in `test_splines.py`, repeated panel
  quadrature fixtures in `test_exterior.py`, and algorithm-choice assertions
  that do not test public behavior. Keep one exact, one refinement, and one
  derivative test per behavior;
- remove `mirror_performance.png`; fold any retained performance evidence into
  one of the three scientific showcase figures.

The current hot spots are `splines.py` (1,011 lines), `free_boundary.py` (945),
`forces.py` (935), `output.py` (868), and `exterior.py` (773). T11 reaches the
7,200/4,000 budgets by deletion and fixture consolidation, not by moving the
same logic to new files.

Concrete branch budgets at merge:

- no more than 46 changed files and 13 mirror source modules;
- no increase above 8,000 mirror source lines at any accepted milestone; the
  current 7,956-line implementation must return below 7,200 before merge;
- fewer than 4,000 mirror test lines after duplicate nodal tests are removed;
- no mirror source file above 1,000 lines;
- no more than 18 public mirror names, with removals preferred;
- exactly four canonical compact benchmark JSON files;
- at most three root examples and three compressed showcase figures;
- no generated run directories, dense arrays, notebooks, or uncompressed
  raster sequences in git.

No new module is added unless the same commit deletes at least as much obsolete
code and the ownership boundary is clearer. Refactors that do not reduce a
duplicate implementation, public API, or measured complexity are deferred.

Public `ntheta` has been removed from `MirrorResolution`; exterior quadrature
resolution stays independent. Do not add a public dealiasing knob unless a
refinement test demonstrates aliasing; if needed, overintegration is initially
internal and derived from `mpol`.

Docstrings state inputs, units, coordinate location, normalization, and failure
conditions in plain language. Comments explain non-obvious discretization or
physics decisions, not individual assignments.

## 6. Ordered implementation milestones

These milestones are sequential. A failed gate is fixed before downstream
examples or derivatives are promoted. Commit and push after each coherent
substep; inspect CI in batches rather than waiting after every push.

### M0. Evidence reset and API cleanup -- complete

1. Remove redundant public `ntheta`; derive exact collocation from `mpol`.
2. Mark all shaped benchmark records stale in their metadata until regenerated.
3. Add benchmark provenance fields for code SHA, schema, basis, represented
   modes, grid, hardware class, and promotion status.
4. Remove remaining example-only helpers and obsolete compatibility paths when
   a source owner already exists.
5. Run unit, API, import, Ruff, strict Sphinx, and example smoke tests.

Exit: one unambiguous resolution API and no benchmark that silently describes
a different discrete space.

### M1. Repair the independent strong-force diagnostic -- complete

1. Trace VMEC2000's half/full radial mesh placement through `forces.f`,
   `bcovar.f`, `residue.f90`, and `jxbforce.f`.
2. Write a short discretization note mapping every mirror field, pressure,
   metric, current, and derivative to its radial and axial location.
3. Replace mixed unstaggered reconstruction with conservative half-to-full
   interpolation and metric-consistent curl and pressure gradient.
4. Add exact polynomial manufactured cases with nonzero pressure, current,
   lambda, and nonaxisymmetric geometry. Add a separate closed-axis regular
   manufactured case; a regular cylinder alone cannot expose the first-row
   toroidal coordinate limit.
5. Report axis, physical interior, first radial row, end collar, and volume
   integral separately. Coordinate-singular endpoint samples must not dominate
   the physical norm, but no region may be silently dropped.
6. Demonstrate expected refinement order in `ns`, `mpol`, and axial knots on
   both open and circular periodic coordinates. The first active row must
   decrease independently; a bulk-only decrease is insufficient.

Exit: manufactured order is established and the physical strong force
decreases on three grids for accepted fixed equilibria.

### M2. Finish the coefficient path and physical initialization -- complete

1. Commit the compact current fixed-boundary evidence only after JSON, focused
   tests, strict docs, and the complete mirror suite pass on the active solver
   policy.
2. Delete the private nodal fixed solve, nodal custom-VJP, nodal fixed adjoint
   configuration, and duplicate live-solve tests. Retain CGL operators and
   evaluated-state parity fixtures only.
3. Add one spline-owned initializer that projects a supplied Cartesian vacuum
   field onto the mirror Clebsch variables. It accepts sampled field values or
   a callable; it does not construct coils or perform Biot--Savart integration.
4. Pin the initializer with the analytic SFLM: reconstructed field relative
   RMS below `5e-4`, all-volume strong force below `6e-3`, correct flux, finite
   lambda, and positive geometry. Project analytic polar sections from an
   oversampled theta grid through exactly the retained Fourier modes; direct
   collocation samples are an aliasing test, not the reference fixture.
5. Use the projected initializer in the root SFLM example and continuation.
   Retain the failed homothetic continuation as compact negative evidence, not
   as a default path.
6. Add an equal-axisymmetric-end fixture and a cut-location study so the
   central solution is demonstrably independent of the artificial collars.

Exit: one public coefficient-native fixed solver remains; the analytic SFLM
initializer is a binary regression test; source and test line counts decrease.

The exact polynomial mirror now supplies equal symmetric ends and the required
cut-location study. M2 is complete.

### M3. Build the structured solver and promote fixed open mirrors -- complete

1. Define the shared spline state coefficient map used by fixed primal,
   tangent, and adjoint solves and designed for free-boundary composition in
   M4. Radius, stream function, gauge, fixed ends, and scaling are explicit
   blocks rather than unrelated flat slices.
2. Freeze a local approximate Hessian retaining neighboring radial coupling,
   axial B-spline support, and the poloidal-mode coupling required by
   nonaxisymmetric geometry. T4 showed that radius--stream cross blocks were
   unnecessary before the axis correction. T8 supersedes that conclusion:
   the regular odd-mode extrapolation spans two radial rows and its local
   radius--stream blocks are required by the cyclic true-residual gate. Keep
   those local blocks, but no dense global channel coupling.
3. A/B test the then-current SciPy/SOLVAX `0.7.3` path and current-main SOLVAX
   pytree GMRES/Newton--Krylov in an isolated environment. Keep SciPy's host
   optimizer for the nondifferentiable CLI when faster. Upgrade the dependency
   only if the release gates improve.
4. Use the same residual action, scaling, preconditioner, and transpose action
   for Newton, forward tangent, and reverse adjoint solves. Apply gauges and
   fixed-end constraints before factorization.
5. A/B test no preconditioner, the current separable preconditioner, and the
   frozen local preconditioner on axisymmetric open, analytic-seeded SFLM, rotating
   ellipse, and circular periodic cases. Record cold/warm runtime, peak memory,
   nonlinear iterations, Krylov iterations, and true linear residual.
6. Rerun SFLM and rotating-ellipse studies with independent refinement of
   `ns`, `mpol`, axial knots, and quadrature, followed by a three-grid combined
   refinement and half-radius paraxial study.
   Begin with the equal-axisymmetric-end/cut-location fixture from M2. At each
   shaped grid record the projected analytic seed and the converged state
   separately. Compare production energy gradient, independent staggered weak
   variation, and reconstructed pointwise force. If the first two disagree,
   repair the discretization; if they agree while strong force extrapolates to
   a nonzero limit, stop and amend the open variational model rather than tune
   the optimizer.
7. Require finite current and lambda, positive Jacobian, nestedness, axis
   regularity, weak residual, repaired strong force, section matrix,
   ellipticity/orientation, axis field, and field-line slope gates.
8. Only after the primal gates pass, rerun forward and reverse implicit
   derivatives and regenerate `mirror_fixed_boundary.json` and the root
   fixed-mirror figures.

Exit gates: both axisymmetric and nonaxisymmetric fixed open mirrors satisfy
section 1.1; true linear residual is at most `1e-8`; Krylov growth is bounded;
the structured path is at least 2x faster or enables a previously blocked
case; and M3 ends within the repository budgets in section 5. Remove any new
solver path that cannot meet these gates.

T5 meets these gates. The sparse factor changes the medium SFLM from a stalled
`0.842` true residual at 19,000 Krylov iterations to `2.69e-11` at 3,120
iterations and reduces wall time from about 55 seconds to 13 seconds. The
three-grid rotating/SFLM strong-force sequences converge to zero with observed
orders `2.69`/`2.63`; equal-end, half-radius, geometry, and derivative gates
pass. The rotating `m=2` paraxial amplitude remains diagnostic for the
first-order reason recorded in section 3.3.

### M4. Promote the axisymmetric open free-boundary beta ceiling

Implementation status: steps 1--13 are complete. The independent and combined
T7 matrix supports beta through 10%. The 25% point fails the independent
all-volume force gate; 50% also fails the fine all-volume force and center-field
refinement gates. Both remain labeled research evidence in the same compact
record.

1. Represent boundary, interior geometry, and lambda with the same clamped
   axial spline coefficients used by fixed boundary. Evaluate them on the
   existing CGL/panel nodes and differentiate through that linear map.
2. Form the lateral interface equation as virtual boundary work. Multiply the
   total-pressure jump by lateral quadrature, area, and the normal component of
   each radial boundary basis variation, then pull it back through the spline
   evaluation map. This produces exactly one residual per free boundary
   coefficient. Retain the pointwise pressure jump as an independent
   refinement diagnostic. A raw list of CGL stress values is not the
   coefficient equilibrium equation.
3. In `free_boundary.py`, compose one private free-boundary coefficient map
   from the existing spline state map. It owns free boundary coefficients,
   interior radius/lambda coefficients, gauges, fixed ends, physical scales,
   bounds, and the optional
   mass scale. The same map is used by primal solve, weak pullback, restart,
   tangent, and adjoint code; no copied index masks are allowed.
4. Build one private free-equilibrium problem object in `free_boundary.py`
   containing pack/unpack, residual blocks, exact JVP/VJP actions, scaling, and
   result assembly. Replace materialized full coupled-Jacobian columns with a
   SciPy `LinearOperator`. A/B cached `jax.linearize` plus VJP closures against
   repeated exact JVP/VJP actions; use the faster path only if its retained
   residual graph also passes the peak-memory gate. Use bounded trust-region
   iterative least squares for production and keep a dense Jacobian only below
   the tiny-test threshold. Gate host and device memory separately; chunking a
   full identity is not matrix-free and does not pass this step.
5. Prove the discrete formulation before production runs: compare the
   coefficient boundary-work residual with directional finite differences,
   compare dense `jacfwd` with operator JVP/VJP on a tiny square problem, verify
   the transpose identity, and show nodal/coefficient parity for a basis that
   spans the nodal fixture. Above the threshold, a test must fail if `np.eye`,
   full `jacfwd`, or a dense coupled Jacobian is reached.
6. Preserve the one cap-aware exterior BIE and separately monitor lateral
   tangency, pressure jump, raw/corrected cap compatibility, cap-rim continuity,
   equal symmetric end data, endpoint shape constraints, cut-location
   independence, and shape gauge. Report the dense BIE matrix memory separately
   from the coupled-equilibrium operator memory.
7. Make restart schema 3 coefficient-native and migrate schema 2 exactly once
   through a tested nodal-to-spline fit. Results own coefficient boundary/state;
   evaluated arrays are clearly named diagnostic/output data. Resume and direct
   continuation must converge to the same coefficient solution.
8. Move the free adjoint onto the same square coefficient residual and map.
   Delete its reconstructed nodal residual and copied masks. Validate the
   pressure-amplitude and external-field-scale gradients against fully
   reconverged centered finite differences, not frozen-boundary differences.
9. Continue one physical equilibrium through beta
   `0, 0.01, 0.03, 0.10, 0.25, 0.50`, warm-starting only between adjacent
   values and recording retries. The external field comes from an ESSOS
   callable or MGRID; no coil representation enters mirror source code.
   Prescribe the full nested endpoint profiles from the supplied field's
   enclosed flux. Record their represented-flux error after the spline fit;
   correcting only the LCFS while retaining self-similar internal cuts is not
   an admissible initialization.
10. Run three independently refined radial, axial, exterior, and angular grids;
   refine one family at a time before the combined study.
11. Compare only on-axis field depression at 1%, 3%, and 10% with the pinned
   Pleiades three-grid fixture; do not claim Pleiades LCFS-radius parity.
   Compare vmec_jax radius expansion and field response through 25% only to
   declared paraxial/diamagnetic trends, including
   `B/B_vacuum approximately sqrt(1-beta)`. The helper and gate stop at beta
   0.3. The 50% point is assessed only by numerical refinement, force balance,
   interface conditions, and continuation; no low-beta formula is extrapolated
   to make it pass.
12. Diagnose any beta-insensitive result by checking pressure normalization,
   enclosed volume, boundary work, field-energy balance, and profile
   interpolation before changing geometry.
13. Regenerate the free-boundary benchmark and example figures only after the
   coefficient, operator, physics, and derivative gates pass.

Exit: beta 10% is the documented supported limit from the schema-5 nested-cut
matrix. The 25% and 50% states are retained as scalar-pressure research
equilibria, not ANIMEC or kinetic predictions. Raising the ceiling requires a
new independent force/refinement pass; weak convergence alone is insufficient.

### M5. Promote the fixed-boundary closed hybrid

Status: circular parity, axis regularity, and the scalable colored factor are
complete. The racetrack continuation, open-leg limit, beta/current physics,
derivatives, and MOUT are not yet promoted.

1. Maintain the completed circular-axis gate. For major radius `R0`, minor
   radius `a`, and area coordinate `s`, the concentric vacuum fixture uses
   `dPsi/ds = psi0/sqrt(1-(a/R0)^2 s)` and matched VMEC
   `APHI[k] = binom(2k,k)*((a/R0)^2)^k/(4^k*(k+1))`. `PHIEDGE` follows VMEC's
   represented 101-point integral. VMEC2000 and ordinary vmec_jax agree to
   roundoff; the periodic spline solve reaches `ftol <= 1e-12` and its
   `ns=5,9,17` strong force decreases to `1.25e-7`. Any change to the closed
   map, flux normalization, or axis treatment must rerun this binary gate.
2. Keep the entire longitudinal representation in periodic cubic B-spline
   coefficients: centerline, transported frame controls, section amplitudes,
   and stream function. Fourier is used only in the periodic cross-section
   angle; no longitudinal Fourier projection is introduced.
3. Validate periodic spline position and first/second derivatives, Bishop
   frame and holonomy correction, section Jacobian, clearance, up--down and
   leg-exchange coefficient symmetries, and join smoothness.
   Frame closure and derivatives must converge when only the axis quadrature is
   refined; otherwise the discrete transport is part of the geometry error and
   the equilibrium study does not start.
4. Define one coefficient-generated geometry family with two symmetry-related
   straight legs and two symmetry-related curved returns. Continue from the
   circular torus by changing one control group at a time: racetrack aspect,
   straight-leg length, return curvature, section ellipticity, then 90-degree
   section rotation. At least half of each straight leg must remain below the
   declared curvature tolerance. Symmetry is enforced by coefficient maps,
   never by copying sampled points.
   Before this continuation, replace the current one-HVP-per-column factor
   assembly. For closed states, fix one physical stream-function gauge
   coefficient per radial surface instead of reconstructing a weighted pivot;
   this removes the artificial rank-one axial coupling while preserving the
   same gauge-invariant field. Apply the weighted-mean convention only when
   exporting or comparing states, not inside the factorized coordinates.
   Build the exact structural sparsity graph from
   radial support, periodic cubic-spline overlap, poloidal coupling, and local
   radius--lambda blocks. Color columns whose row supports do not intersect,
   evaluate one Hessian-vector product per color, and scatter the responses
   into the frozen sparse factor. Verify on tiny cases against every-column
   assembly and the exact Hessian, including transpose and gauge invariance.
   Reject the implementation unless factor setup uses at least 4x fewer HVPs,
   peak memory is below 4 GiB, current-free setup is below 60 s on the
   reference CPU, and the circular true linear residual remains at most
   `1e-8`. A/B SOLVAX periodic/banded primitives only where the block structure
   matches; keep the lower-runtime, lower-memory, shorter path.
   T9a meets these gates. The refined graph uses 145 probes for 712 columns,
   cold closed factor setup is 1.99 s at 1.21 GiB, and the greater-than-1,024
   circular true-residual test passes. The finite-current smoke finishes in
   56.1 s below 3 GiB, so continuation may proceed; its `0.548` strong force
   remains an explicit T9b failure to resolve by staged continuation and
   refinement.
5. Compare a fixed central fraction of each leg with the promoted fixed open
   B-spline mirror along a three-member sequence of increasing
   leg-length/return-radius separation. Match local flux, pressure, current,
   section shape, and axis field before comparing geometry, `|B|`, and force.
   Use the central 50% of each leg and three declared leg-to-return length
   ratios. Geometry, axial field, section moments, and local strong force must
   converge toward the open result; global closed-circuit iota is not an
   open-limit observable.
6. Demonstrate nonzero rotational transform and circuit-spanning field lines
   from the solved equilibrium. First continue the geometric racetrack at zero
   current, then ramp the declared `current_derivative` through
   `0, 0.25, 0.5, 0.75, 1.0` of its target. Record iota from at least four
   field-line labels over at least 20 turns and separate geometric from
   current-driven transform. Do not attribute vacuum-coil physics to the
   equilibrium solver.
7. Run fixed-LCFS beta continuation at `0, 0.01, 0.03, 0.10`; attempt 0.25 only
   as labeled research evidence. Show interior surfaces, `|B|`, iota,
   magnetic well, weak residual, strong force, and convergence histories.
   Report low-beta internal multipole trends beside the linked-mirror report
   only as qualitative context. Do not compare its free-boundary displacement
   with this fixed-LCFS lane. The LCFS must not move; any beta-dependent
   boundary shown in an artifact is a bug.
8. Revalidate forward/reverse derivatives with respect to pressure/current,
   section coefficients, and centerline B-spline controls. Compare the latter
   to reconverged finite differences so the geometry is demonstrably usable in
   later design optimization. Produce a periodic MOUT that is explicitly
   distinct from WOUT.
9. Regenerate `mirror_hybrid_fixed_boundary.json` and the root hybrid example.

Exit: the circular parity case, open-leg limit, racetrack continuation,
derivatives, and output round trip all pass.

### M6. Bounded nonaxisymmetric free-boundary attempt

This milestone starts only after M5 passes so it cannot consume the resources
needed by the required closed hybrid.

1. Seed the free solve from the promoted weakly rotating fixed equilibrium.
2. Continue one parameter at a time: pressure, ellipticity, then rotation.
3. Run three grids with explicit limits of 1,000 nonlinear iterations, 30
   minutes wall time, and 8 GiB peak memory per state on the reference CPU;
   GPU results are supplementary until CPU parity is shown.
4. Require local nonaxisymmetric coefficients, observables, boundary
   diagnostics, and strong force to satisfy the same promotion contract.

Exit A: promote and regenerate `mirror_free_boundary_nonaxisymmetric.json`.
Exit B: store only a compact negative record with failure mode and resource
measurements, remove public/example scaffolding, and defer the lane.

### M7. Release reduction, documentation, and artifacts

1. Delete superseded benchmark runners, duplicate plotting helpers, temporary
   configurations, and generated outputs. At minimum remove the one-shot
   exterior endpoint runner after canonical data are regenerated and fold the
   standalone performance raster into a retained three-panel showcase.
2. Meet the line, file, API, benchmark, example, and image budgets in section 5.
3. Update the README capability table with `supported`, `research`, and
   `deferred` labels and showcase the three canonical workflows.
4. Document equations, coordinates, boundary conditions, spline spaces,
   normalization, residual definitions, continuation, implicit derivatives,
   MOUT schema, failure modes, and external validation. Remove current
   `supported` wording for free-boundary derivatives and examples until their
   primal spline lane passes the promotion gates.
5. Root examples remain parser-free scripts with parameters at the top and
   produce polished horizontal-mirror 3D geometry, visible equilibrium field
   lines, `|B|`, cross-sections, convergence histories, and relevant profiles.
6. Store no raw run trees. Commit only compact JSON/CSV evidence and at most
   three compressed final figures; CI regenerates smoke-resolution plots.
7. Run all tests, strict Sphinx, examples, packaging/import checks, CPU/GPU
   parity where available, and review the complete PR diff before marking the
   PR ready.

Exit: the draft PR contains only supported code and clearly labeled research
evidence, with no experimental public surface left behind.

### Finite commit sequence

This is the execution order. A tranche is committed and pushed only after its
listed gate passes; failed experiments are removed in the same tranche or
recorded as compact negative evidence.

| Tranche | Change | Required gate |
| --- | --- | --- |
| T1 (complete) | Validate and commit the active force-region, compact benchmark, docs, and matrix-free policy edits | full mirror suite, strict Sphinx, Ruff, JSON schema |
| T2 (complete) | Delete nodal fixed solve/custom-VJP and redundant tests | public API/import tests; axisymmetric spline parity unchanged; net line reduction |
| T3 (complete) | Add supplied-field-to-Clebsch spline initializer and use it for SFLM | analytic field error `<5e-4`; force `<6e-3`; example smoke |
| T4 (complete) | Implement and A/B the frozen local spline preconditioner, including current-main SOLVAX trial and stale-CI-path repair | 100 passed/6 skipped; strict docs; true linear residual `9.18e-11`; Krylov work 2,000 to 660 |
| T5 (complete) | Regenerate rotating-ellipse and SFLM fixed-boundary evidence | 102 passed/7 skipped; full physics regressions, strict docs, pre-commit, and promoted record pass |
| T6a (complete) | Add the composed free coefficient map and Galerkin boundary-work residual | 3 focused and 28 non-full spline tests; directional JAX/FD work parity; square coefficient map; pre-commit pass |
| T6b (complete) | Replace the coupled Jacobian with measured-memory JVP/VJP `LinearOperator` actions and migrate result/restart schema | dense tiny-case and transpose parity; repeated actions selected by memory/runtime A/B; no full coupled Jacobian above threshold; schema-3 restart and continuation pass |
| T6c (complete) | Reuse the primal coefficient residual in the free tangent/adjoint and delete all nodal production packing | external-field and finite-pressure reconverged-FD gradients; true transpose residual; 103 normal and 3 representative full tests pass; source `7,880`, tests `4,227`, `splines.py 999`; strict docs/static checks pass |
| T7 (complete) | Set the supported ceiling and regenerate the nested-cut axisymmetric beta record | schema-5 record supports through 10%; 25%/50% retained as research; independent/combined force, observable, interface, and Pleiades gates recorded |
| T8 (complete) | Land matched-flux circular parity and axis-regular strong-force repair | `94addb68`; APHI/VMEC/WOUT parity; `ns=5,9,17` force sequence; 105 passed/9 skipped; strict docs/static gates |
| T9a (complete) | Restore local closed-stream gauge sparsity and color the frozen Hessian graph | exact tiny Hessian/scatter/transpose/gauge parity; `4.91x` refined probe reduction; 1.99 s/1.21 GiB setup; circular true residual `<=1e-8` |
| T9b | Continue the periodic B-spline axis from circle to symmetric racetrack, rotating returns, and finite current | positive map and clearance; C2 joins; target-current `ftol<=1e-12`; circuit-spanning lines and nonzero iota |
| T9c | Establish the three-member open-leg limit and fixed-LCFS beta continuation | central-leg observables converge to fixed open mirror; 0--10% beta force/refinement gates; 25% labeled research at most |
| T9d | Validate hybrid tangents/adjoints, MOUT, plots, benchmark, and root example | reconverged FD agreement; true transpose residual `<=1e-8`; round trip; polished horizontal-leg geometry/field/profile/convergence figures |
| T10 | Run the bounded nonaxisymmetric free-boundary attempt | promotion or explicit negative record within stated budget |
| T11 | Remove the ESSOS-owned runner, compatibility paths, redundant figures/tests, and stale records | at most 46 changed files, 7,200 mirror source lines, 4,000 mirror-test lines, 18 public names, 4 JSON records, and 3 root examples/figures |
| T12 | Regenerate README/docs/examples and perform final release audit | all local/CI gates green; PR diff reviewed; draft removed only then |

No additional physics lane enters this PR. ANIMEC, free-boundary hybrid,
arbitrary curved open axes, stability, and mirror Boozer work begin only in a
new plan after T12.

### Remaining reduction contract

T11 changes existing owners only and is driven by deletion. `splines.py` must
return below 1,000 lines without splitting it into another module. The spline
fixed/free maps and one physics-specific preconditioner remain; duplicate
packing, migration, plotting, and algorithm-choice helpers do not. `output.py`
retains one MOUT schema and one legacy migration path. Test files retain one
analytic, one operator, one refinement, and one derivative gate per behavior;
duplicated setup and tests of private implementation choices are removed.

The root keeps only the fixed-open and free-axisymmetric executable examples
plus the fixed closed hybrid after promotion. The temporary ESSOS benchmark
runner and coil construction leave this repository; vmec_jax retains only the
documented external-field callable/MGRID interface. Canonical benchmark data
are regenerated in place, never added as timestamped variants. T11 is complete
only when all repository budgets in section 5 pass together.

## 7. Canonical artifacts and reporting

Canonical benchmark files:

1. `mirror_fixed_boundary.json`;
2. `mirror_free_boundary_axisymmetric.json`;
3. `mirror_free_boundary_nonaxisymmetric.json` only as promoted or explicitly
   negative evidence;
4. `mirror_hybrid_fixed_boundary.json`.

Every record includes commit, clean/dirty state, platform, precision, grid,
basis, represented modes, tolerances, iterations, wall time, peak memory,
variational/weak/strong residuals, geometry checks, observables, comparison
errors, derivative errors when applicable, and promotion status.

Every work report states:

- steps taken;
- results obtained, including failed gates;
- tests and hardware used;
- files changed and why ownership remains clear;
- best next steps;
- completion percentages for all open lanes;
- any concrete input needed from the user.

The plan changes only when evidence invalidates a gate or scope decision. A
plan amendment must cite the benchmark or source that caused the change and
replace, not append to, the affected decision.

## 8. Completion estimate at this audit

Percentages measure promotion evidence, not lines written:

| Lane | Complete | Main remaining evidence |
| --- | ---: | --- |
| Fixed open axisymmetric | 100% | maintain gates while shared solver code changes |
| Fixed open nonaxisymmetric | 100% | maintain gates during shared-core changes |
| Open fixed B-spline representation | 100% | maintain coefficient and cut-location gates |
| Free open axisymmetric | 100% | maintain the 10% support gates; 25%/50% remain research until a later refinement pass |
| Free open nonaxisymmetric | 35% | conditional three-grid local-mode promotion attempt after M5 |
| Fixed closed B-spline hybrid | 74% | staged current/shape continuation; strong-force refinement; open-leg limit, beta, derivatives, MOUT |
| Strong-force diagnostic | 100% | maintain gates in promoted equilibrium lanes |
| Structured preconditioning | 100% | maintain open and closed true-residual/resource gates during continuation |
| Implicit differentiation | 90% | rerun hybrid derivatives only after primal promotion |
| Code/API simplification | 64% | current 53-file/8,052-source/4,492-test/20-name tree must meet all T11 budgets |
| Docs/examples/artifacts | 73% | free support-ceiling figure and T9a record are current; hybrid showcase and final README/API reduction remain |
| ESSOS ownership separation | 90% | remove the remaining ESSOS-owned benchmark runner; retain field-callable integration only |

Weighted completion of the required release models is approximately 88%.
Free closed hybrid and ANIMEC are deferred and excluded from that percentage.

## 9. Primary references

- VMEC2000/ANIMEC source: <https://github.com/PrincetonUniversity/STELLOPT>
- ANIMEC code description and solver outline:
  <https://www.epfl.ch/research/domains/swiss-plasma-center/research/theory/codes/animec/>
- VMEC++ paper: <https://arxiv.org/abs/2502.04374>
- DESC source and experimental branches: <https://github.com/PlasmaControl/DESC>
- DESC mirror branch: <https://github.com/PlasmaControl/DESC/tree/mirror>
- DESC mirror-anisotropy branch:
  <https://github.com/PlasmaControl/DESC/tree/mirror_anisotropy>
- DESC cylindrical/Chebyshev branch:
  <https://github.com/PlasmaControl/DESC/tree/dd/cylindrical>
- DESC experimental finite-element branch:
  <https://github.com/PlasmaControl/DESC/tree/finite_element_basis>
- DESC free-boundary formulation: <https://arxiv.org/abs/2412.05680>
- SOLVAX: <https://github.com/uwplasma/SOLVAX>
- SOLVAX untagged 0.8.4 release branch:
  <https://github.com/uwplasma/SOLVAX/tree/release/0.8.4>
- GVEC G-frame paper: <https://arxiv.org/abs/2410.17595>
- GVEC G-frame documentation: <https://gvec.readthedocs.io/develop/tutorials/notebooks/052_gframe.html>
- Pleiades: <https://github.com/eepeterson/pleiades>
- Pleiades pinned comparator revision:
  <https://github.com/eepeterson/pleiades/tree/0161abb3e9a1d85143c650f068ec524d672fc9ab>
- Ågren and Savenko, *Theory of the straight field line mirror*, 32nd EPS
  Conference on Plasma Physics, ECA 29C, P-4.069 (2005):
  <https://info.fusion.ciemat.es/OCS/EPS2005/pdf/P4_069.pdf>
- Cooper, *Three-dimensional magnetohydrodynamic equilibria with anisotropic
  pressure*, Comput. Phys. Commun. 72 (1992):
  <https://doi.org/10.1016/0010-4655(92)90002-G>
- Cooper et al., *Three-dimensional anisotropic pressure free boundary
  equilibria*, Comput. Phys. Commun. 180 (2009):
  <https://doi.org/10.1016/j.cpc.2009.04.006>
- EPFL record for the free-boundary ANIMEC formulation:
  <https://infoscience.epfl.ch/entities/publication/32f71d02-8e35-49da-b916-f5af429ba739>
- Asahi et al., *MHD equilibrium analysis with anisotropic pressure in LHD*:
  <https://www.jstage.jst.go.jp/article/pfr/6/0/6_0_2403123/_article>
- EPFL ANIMEC code description:
  <https://www.epfl.ch/research/domains/swiss-plasma-center/research/theory/codes/animec/>
- Rodríguez, Helander, and Goodman, *The maximum-J property in
  quasi-isodynamic stellarators*: <https://doi.org/10.1017/S0022377824000345>
- Feng et al., linked mirror concept: <https://arxiv.org/abs/2103.09457>
- Ilgisonis, Berk, and Pastukhov, finite-beta toroidally linked mirrors:
  <https://doi.org/10.2172/10179323>
- Skene and Burns, automated spectral adjoints:
  <https://arxiv.org/abs/2506.14792>
- VEPEC technical report: <https://www.osti.gov/biblio/6351313>
- Goodman, Freidberg, and Lane, analytic long-thin mirror equilibria:
  <https://doi.org/10.1063/1.865851>
- Savenko and Ågren, finite-beta SFLM ellipticity:
  <https://doi.org/10.1063/1.2401153>
- Savenko, *MHD Stability and Confinement of Plasmas in a Single Mirror
  Cell* (thesis and collected SFLM derivations):
  <https://www.diva-portal.org/smash/get/diva2:167915/FULLTEXT01.pdf>
- Patch et al., 2025 DESC/open-mirror development status:
  <https://archive.aps.org/dpp/2025/jo05/4/>
- Beklemishev, diamagnetic bubble equilibria:
  <https://arxiv.org/abs/1606.05454>
- Beklemishev et al., paraxial high-beta mirror pressure balance:
  <https://nucleus.iaea.org/sites/fusionportal/Shared%20Documents/FEC%202016/fec2016-preprints/preprint0415.pdf>
- JAX custom derivative and implicit-iteration guide:
  <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>
- JAX cached forward linearization and its memory tradeoff:
  <https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html>
- JAX `custom_root` and `custom_linear_solve`:
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html>,
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html>
- JAXopt implicit root differentiation:
  <https://jaxopt.github.io/stable/_autosummary/jaxopt.implicit_diff.custom_root.html>
- Lineax operators and transpose-aware solvers: <https://docs.kidger.site/lineax/>
- Optimistix implicit adjoints: <https://docs.kidger.site/optimistix/api/adjoints/>

The reference list supports decisions; only reproduced tests and compact
benchmark records count as release evidence.
