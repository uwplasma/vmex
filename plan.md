# Mirror equilibrium release plan

Status: final authoritative plan for draft PR #22. Revised 2026-07-15 after
auditing the pushed branch and clean research worktree, current `origin/main`,
VMEC2000/ANIMEC, DESC and its experimental mirror/Chebyshev branches, GVEC,
SOLVAX 0.8.3/current main, Pleiades, primary mirror literature, and differentiable-solver
literature. This file replaces all earlier plans, including `plan_mirror.md`.
Do not create another roadmap.

The objective is a small, research-grade mirror-equilibrium extension of
`vmec_jax`, not a collection of geometry demonstrations. A promoted lane must
solve the stated ideal-MHD model, reach its physical convergence gates,
refine toward an independent result, support converged-state derivatives where
claimed, and expose one simple documented workflow.

## 1. Audit state

Repository state at this revision:

- audited scientific source checkpoint: `codex/mirror-geometry` at
  `9e2c171d`; this plan-only revision follows it;
- base: `origin/main` at `ed4ac7ac`; the source checkpoint is 0 behind and
  350 ahead;
- PR: <https://github.com/uwplasma/vmec_jax/pull/22>, open, mergeable, draft;
- CI on `b2eb72ef`: all 14 required checks pass; the manual/nightly full job is
  skipped as intended;
- source-checkpoint diff against fetched `origin/main`: 53 files, 18,687
  insertions, 1,602 deletions. Never use a stale local `main` for branch-size
  accounting;
- mirror source: 13 modules, 8,696 lines at pushed HEAD;
- mirror tests: 10 files, 5,007 lines;
- public mirror API: 20 lazy names;
- normal mirror suite after the refined center-map solver tests: 119 passed
  and 11 expected full/nightly skips in 311.46 s. Strict Sphinx, pre-commit,
  and `git diff --check` pass;
- no coil constructor or Biot-Savart implementation is part of the mirror
  package. Those belong to ESSOS.

The pushed worktree is clean. Earlier C2-return/control-allocation experiments
remain rejected diagnostic evidence outside the repository; no experimental
geometry helper or generated output was retained.

### 1.1 What is already achieved

| Lane | Evidence | Status |
| --- | --- | --- |
| Fixed open, axisymmetric | spline coefficient solve, analytic/paraxial gates, `ftol <= 1e-12`, strong-force refinement | supported |
| Fixed open, nonaxisymmetric | rotating ellipse and Agren-Savenko SFLM, supplied-field projection, three grids, tangents/adjoints | supported |
| Free open, axisymmetric | square coefficient residual, exterior BIE, beta continuation, low-beta Pleiades trend, implicit adjoint | supported through 10% beta |
| Free open, 25%/50% beta | converged scalar-pressure states but failed independent force/refinement ceiling | research only |
| Free open, nonaxisymmetric | solver exists, local-mode refinement has not passed | conditional research |
| Fixed closed circular limit | VMEC2000/ordinary-vmec_jax flux parity, axis regularity, `ns=5,9,17` force refinement | supported validation limit |
| Fixed closed hybrid | periodic B-spline geometry, Bishop frame, rotating section, finite-current solve, field-line tracer | not promoted |
| Preconditioning | coupled sparse local factor passes dense-oracle tests; the 2,664-variable center map converges in 4,300 GMRES iterations | supported through the half-straight radial restart |
| Differentiation | converged-residual open tangents/adjoints and free-axisymmetric adjoint | supported for promoted open lanes |

Important measured results:

- Open free-boundary fine-grid normalized strong force is `2.08e-3` at beta
  0, `1.44e-2` at 10%, `3.37e-2` at 25%, and `6.69e-2` at 50%. Independent
  radial/angular sequences fail the declared gate at 25%, so 10% is the
  support ceiling. The 25% and 50% states remain useful nonlinear research
  evidence, not promoted physics.
- The 10% free-boundary center field ratio is `0.95620`, 0.36% from the pinned
  Pleiades low-beta result. This comparison is limited to the on-axis
  diamagnetic trend because Pleiades fixes a pressure-support radius and does
  not solve the same moving-boundary problem.
- The matched circular closed limit has normalized strong force
  `4.72e-6, 7.35e-7, 1.25e-7` for `ns=5,9,17`; the finest bulk value is
  `3.44e-9` and all solves reach `ftol <= 1e-12`.
- Closed Hessian coloring reduces 712 columns to 145 probes (`4.91x`) on the
  refined graph. Cold linearization plus sparse factor setup takes 1.99 s and
  1.21 GiB. A greater-than-1,024-variable circular solve passes its true
  linear-residual gate.
- The direct 16-control finite-current racetrack reaches variational
  `ftol=1e-12` in 55.9 s below 3 GiB and has iota `0.04215`, but normalized
  strong force is `0.54757`. At 32 controls the same endpoint takes 284.5 s,
  force falls to `0.27929`, and iota remains `0.04227`. Axial representation
  error is therefore dominant, but the endpoint is not converged.
- Replacing scalar `dPsi/ds` with a geometry-matched radial profile changed
  the racetrack force only from about `0.54757` to `0.54756`. That experiment
  was removed rather than retained as an ineffective helper.
- Setting the racetrack current to zero leaves force near `0.54733` and reduces
  iota to `5.3e-9`. The force defect is geometric/discretization error, while
  the observed endpoint transform is currently current-driven.
- T9b's return-localized fixture exposed the missing closed-equilibrium degree
  of freedom. Before the center map, the closed vectorizer fixed the supplied
  centerline and activated only radial rows `1:-1`; unlike VMEC, it could not
  move the magnetic axis while holding the physical LCFS fixed. On the same
  `ns=5` grid a
  circular axis has strong force `2.55e-3`, but half-straightening raises it to
  `0.602`. Ellipticity changes the endpoint by only about `0.014`, return twist
  adds about `0.279`, and current continuation from zero to target changes
  force by only `3e-4` while driving iota from zero to `0.04013`.
- A fixed C2 Hermite return and return-weighted controls improve the
  current-free circular-section sequence from `0.34424` at 16 controls to
  `0.14032` at 32 and `0.12603` at 64. Refining `ns=5 -> 9` at 64 controls
  changes force only to `0.11033`. The `<5e-2` gate therefore fails within the
  64-control limit. Direct 24-control endpoint solving also exceeds the
  30-minute state budget; production refinement must use staged restarts.
- On GPU, the dirty 32-control C2/return-weighted finite-current endpoint
  reaches variational `1.11e-14` in 56 nonlinear and 1,095 linear iterations
  in 64.7 s, with divergence `9.13e-15` and iota `0.04346`, but normalized
  strong force remains `0.19547` (`0.25158` on axis). Tight nonlinear
  tolerance and acceleration therefore do not repair the missing physical
  variation.
- The two-component center map now fixes that missing degree of freedom. The
  displaced circular reference reaches variational and weak residuals below
  `5e-13` at `ftol=1e-12`, keeps the Cartesian LCFS fixed, and has positive
  Jacobian. Poloidal refinement `mpol=3,5,7` reduces normalized strong force
  from `2.82e-3` to `5.71e-6` to `1.24e-7`. At `ns=17, mpol=3`, bulk strong
  force is `1.32e-4`; the all-volume `3.05e-3` remains axis-dominated because
  `mpol=3` is under-resolved.
- Periodic control counts 8, 12, and 16 all reach `ftol=1e-12`. Dense residual
  Newton now uses rank-revealing QR, rejects invalid geometry, bounds numerical
  stagnation, honors `MirrorConfig.max_iterations`, and is retained through
  2,048 packed variables. Differentiable radial/axial restart transfer is part
  of the existing spline method rather than a continuation subsystem.
- Warm radial refinement resolves the circular map within the resource gate.
  At `mpol=7`, `ns=9 -> 17` reduces strong force `7.53e-8 -> 4.85e-8` in
  43 s and 212 s. At `mpol=5`, the same warm transfer reaches `1.8e-14` in
  120 s instead of the rejected greater-than-30-minute cold matrix-free path.
  Peak memory remains below 2.5 GiB.
- The zero-current half-straight state now genuinely solves. With 24 controls,
  `mpol=5`, all stages through fraction 0.5 reach `ftol=1e-12`; the endpoint
  takes 30.0 s, has positive Jacobian, and strong force `0.06565`. With 16
  controls it has strong force `0.05277`. Transferring that state from
  `ns=5 -> 9` changes force only to `0.05262`. The original matrix-free
  restart stalled at variational `2.39e-10` after 200,000 GMRES iterations
  with true linear residual `0.391`. Enabling the already-tested colored
  sparse local factor for center-map states converges the same 2,664-variable
  problem to variational `1.56e-13` in 55 nonlinear and 4,300 linear
  iterations. The solve takes 54.3 s for the refined state, stays below
  2.4 GiB for the continuation, and reports true linear residual `4.98e-10`.
  Scalable preconditioning therefore passes; the strong-force gate remains
  open by about 5.2%.
- Directly regenerating the control-polygon axis at 16, 24, and 32 controls is
  not a valid same-geometry refinement: arc length and curvature change while
  strong force rises. A resolution-independent C2 Hermite return converges but
  raises half-straight force to `0.44854`; increasing return radius to 4 raises
  it to `0.14858`. Both experiments were removed. Future longitudinal evidence
  must refine one 16-control physical axis into nested 32/64 spline spaces.

The release is blocked by a same-geometry half-straight strong-force
refinement below `5e-2`, then code reduction. The
conditional nonaxisymmetric free-boundary attempt must not delay those required
steps.

## 2. Release scope

PR #22 will retain exactly these promoted scalar-pressure, nested-surface
models:

1. fixed-boundary straight-axis axisymmetric mirrors;
2. fixed-boundary straight-axis nonaxisymmetric mirrors;
3. free-boundary straight-axis axisymmetric mirrors in a supplied external
   field through central beta 10%;
4. fixed-boundary toroidal stellarator-mirror hybrids with two straight mirror
   legs and two smooth stellarator returns, only if the decisive T9b
   same-geometry refinement passes.

One bounded attempt is allowed:

- free-boundary straight-axis nonaxisymmetric mirrors. It is promoted only if
  all three-grid local-mode, force, interface, and resource gates pass. If not,
  its public entry points are removed and one compact negative record remains.

Explicitly deferred to later PRs:

- free-boundary closed hybrids and coil-plasma optimization;
- anisotropic pressure, ANIMEC, kinetic pressure closures, end losses, sheaths,
  transport, and stability;
- arbitrary curved open axes;
- islands, stochastic fields, and open-mirror Boozer output;
- coil construction and Biot-Savart calculations.

This is a go/no-go scope, not an indefinite research queue. A candidate that
misses its stated grid, time, memory, or physics gate is deleted from the
public API and documentation in this PR. The negative benchmark may remain as
one compact record, but no public scaffold remains.

The 25% and 50% axisymmetric free-boundary cases remain labeled
scalar-pressure research continuations. They do not validate a high-beta
mirror model; modern WHAM/Pleiades high-beta work uses anisotropic pressure.

## 3. Physical model

### 3.1 Open mirror

The open computational domain is a finite flux tube between two fixed,
flux-carrying cuts. The cuts are not periodic, not plasma-vacuum interfaces,
and not physical end plates. Field lines may cross them. Geometry and normal
flux are prescribed independently and poloidally symmetrically at both cuts
for the axisymmetric lane. The lateral `s=1` surface is the fixed or solved
plasma-vacuum interface.

For straight axis `z`, the coordinate map is

```
x(s, theta, z) = (r cos(theta), r sin(theta), z),
r = sqrt(s) a(s, theta, z).
```

The divergence-free field is represented by radial profiles `Psi'(s)` and
`I'(s)` plus a zero-mean stream function `lambda`:

```
sqrt(g) B^theta = I'(s) - partial_z lambda,
sqrt(g) B^z     = Psi'(s) + partial_theta lambda,
B^s             = 0.
```

The scalar-pressure energy is

```
W = integral [B^2/(2 mu0) + p/(gamma - 1)] dV,
p(s) = M(s) / V'(s)^gamma.
```

`M(s)` is VMEC-style conserved mass. Nonlinear convergence is the normalized
coefficient first variation of this energy. Promotion additionally requires
an independently assembled staggered first variation and an independently
reconstructed `J x B - grad(p)` residual.

Free-boundary lateral conditions are

```
B_plasma . n = 0,
B_vacuum . n = 0,
p + B_plasma^2/(2 mu0) = B_vacuum^2/(2 mu0).
```

The exterior Neumann domain is closed numerically with disks at the two cuts.
Those disks are integration closures only. Cap compatibility, cut-location
independence, panel refinement, and field reconstruction are required tests.

### 3.2 Closed stellarator-mirror hybrid

The hybrid is toroidal even though most of its reference curve is straight.
Its fixed-boundary map is

```
x(s, theta, u) = c_ref(u) + q1(s,u) n(u) + q2(s,u) b(u)
                 + r(s,theta,u)
                   [cos(theta) n(u) + sin(theta) b(u)],
q1(1,u) = q2(1,u) = 0.
```

Here `c_ref(u)` is a periodic cubic B-spline reference curve and `(n,b)` is a
rotation-minimizing frame with a distributed periodic holonomy correction.
The solved magnetic axis is
`c_mag(u)=c_ref(u)+q1(0,u)n(u)+q2(0,u)b(u)`. The two transverse fields move
interior surface centers while preserving the Cartesian LCFS. A tangent shift
is excluded because it only reparameterizes `u`; this follows VMEC's
fixed-toroidal-label geometry and GVEC's two-coordinate flexible mapping. The
same Clebsch field and energy are used with periodic `u`.

The target family is precise:

- a smooth planar racetrack axis with two symmetry-related straight legs and
  two smooth curved returns;
- at least the central 50% of each leg has curvature below a declared
  tolerance;
- an ellipse has constant orientation on each straight leg;
- the two leg orientations differ by 90 degrees;
- section rotation occurs only in the curved returns, with zero twist rate on
  the declared straight interiors;
- leg-exchange and up-down symmetry are coefficient maps, not copied sampled
  points;
- position, tangent, curvature, frame, and section map are periodic and C2;
- minor radius stays below local axis clearance and the Jacobian keeps one
  sign.

This is a smooth equilibrium-oriented variant of a linked mirror, not a claim
to reproduce Feng et al.'s coil configuration. Their vacuum concept obtains
transform from nonparallel straight legs and explicitly allows non-tangent
coil-center connections. This plan uses a smooth axis and rotating elliptical
returns because differentiable nested equilibrium coordinates require them.

Geometric and current-driven transform are reported separately. A nonzero
current-free transform is not assumed; it must be measured. The requested
on-axis toroidal-current profile is then continued from zero and may provide
the operational nonzero iota.

### 3.3 Pressure-model boundary

ANIMEC is not “VMEC plus `p_parallel` and `p_perp` arrays.” Its variational
energy uses `p_parallel(s,B)`, parallel force balance gives

```
p_perp = p_parallel - B (partial p_parallel / partial B)|s,
```

and the force calculation uses an effective current
`K = curl(sigma B)` with firehose and mirror parameters. The free interface
condition contains `p_perp`, not a scalar `p`. The reviewed STELLOPT history
shows both the 24,684-line legacy standalone ANIMEC tree and the later
13-file integrated `_ANIMEC` change; the latter still modifies pressure,
covariant field/current, radial force, free-interface stress, input, and
output kernels. Anisotropy is therefore a new physical model, not a profile
option, and remains outside PR #22.

## 4. Representation decisions

### 4.1 Production bases

| Direction | Production representation | Reason |
| --- | --- | --- |
| radial `s` | VMEC-like full/half mesh | preserves staggered energy/force structure and circular VMEC parity |
| poloidal `theta` | Fourier modes with `ntheta = 2*mpol+1` | periodic sections and exact represented-mode contract |
| open axial | clamped cubic B-spline coefficients | local support, exact independent endpoints, knot insertion, sparse factors |
| closed longitudinal | periodic cubic B-spline coefficients | straight interiors and curved returns without global Fourier ringing |
| quadrature/reference | CGL/Chebyshev nodes where already used | endpoint-inclusive independent evaluation, not optimized state |

All optimized geometry and stream-function unknowns are coefficient-native.
Nodal CGL states are never a second solver, restart, or derivative path.
Longitudinal Fourier projection is retained only for the circular VMEC limit.

“Full B-spline hybrid” has a precise meaning here: the periodic longitudinal
dependence of the reference axis, LCFS section coefficients, interior radius,
two center-map components, and stream function is represented in the same
periodic cubic B-spline space. Poloidal section dependence remains Fourier and
the radial direction remains the tested VMEC-like full/half mesh. No toroidal
Fourier fit is used to create, solve, restart, or optimize the racetrack.

### 4.2 What DESC changes and does not change

The reviewed DESC branches are experiments, not validation authorities:

| Branch | Finding | Decision here |
| --- | --- | --- |
| `mirror` (`0dba071d`, 2025-09-12) | adds `Equilibrium_mirror`, Chebyshev-Zernike axial bases, and independent end-cap constraints; boundary-condition test file has no tests; notebooks use vacuum/zero pressure and `ftol=1e-8`; commit history says initial constraints fail | retain the concept of independent fixed cuts; do not port code or use as parity |
| `mirror_anisotropy` (`805b77fc`, 2025-11-12) | combines unfinished mirror and anisotropy work; source comments call force results suspicious | no validation use; ANIMEC remains deferred |
| `rg/racetrack` (`2014ed0e`, 2025-09-15) | two-file Chebyshev surface experiment, no promoted solve | confirms global nonperiodic Chebyshev was exploratory, not a closed-racetrack solution |
| `dd/cylindrical` (`6f85f50a`, 2026-06-26) | adds/test-drives a double-Chebyshev-Fourier basis in cylindrical `R,Z,phi`; no mirror equilibrium | useful basis implementation reference only |
| `tq/straight-stellarator` (`8cf50b58`, 2022-08-19) | hard-coded straight-coordinate edits and an interactive debugger | no reusable solver or validation evidence |
| `finite_element_basis` / `_alan` (`829e2db0` / `fb90e65`, 2024) | experimental tetrahedral/spline reconstructions; branch logs retain unresolved periodicity, derivative, optimization, accuracy, and performance problems | do not import a second volume discretization |

DESC master `24aa7b9d` remains a strong reference for continuation,
oversampled local force residuals, exact JVP/VJP objectives, and
differentiation discipline, but its toroidal Fourier-Zernike state does not
solve the representation problem addressed here. Its experimental branches
confirm that a global nonperiodic Chebyshev interval is appropriate for two
independent open cuts but is not a coherent periodic racetrack basis.

### 4.3 VMEC2000, GVEC, and Pleiades roles

VMEC2000 validates only the smooth circular closed limit: flux conventions,
half/full-mesh force structure, axis regularity, and output normalization.
Standard VMEC's global toroidal Fourier representation is not the production
hybrid basis. Its fixed-boundary residual sets only the edge geometry force to
zero; interior Cartesian geometry remains active and the magnetic axis is
read from the solved innermost geometry. This is the structural behavior the
current fixed-reference hybrid is missing.

GVEC is the closest production design analogue for the closed hybrid: its
fixed map can be a generalized moving frame, while two cross-section
coordinates and `lambda` are solved with a fixed boundary. Its radial
B-splines and linear polar-axis constraints confirm that the magnetic axis is
part of the volume state, not the supplied frame. This PR retains its tested
VMEC-like radial full/half mesh rather than starting a second radial rewrite;
only the two transverse center fields are adopted. GVEC is a design and
circular-limit comparison source, not a racetrack parity oracle.

The pinned 2021 Pleiades Green-function solve is an independent low-beta
axisymmetric comparator for on-axis diamagnetic field through 10%. It is not a
moving-boundary or nonaxisymmetric oracle. Modern Pleiades/WHAM high-beta work
uses anisotropic distribution-derived pressure and reinforces the 10% support
ceiling for this scalar-pressure PR.

### 4.4 Analytic fixtures

Keep these fixtures separate:

- the circular-loop on-axis field and low-radius off-axis expansion validate
  supplied external fields, not equilibrium by themselves;
- `AxisymmetricPolynomialMirror` validates paraxial axisymmetric geometry;
- `StraightFieldLineMirror` represents the Agren-Savenko marginal-minimum-B
  construction and its changing ellipticity;
- `RotatingEllipseParaxial` validates an independent 90-degree rotating
  ellipse in the thin-tube limit;
- circular torus/VMEC validates the closed topology and flux normalization;
- Feng et al. provide qualitative topology and transform context: two
  nonparallel straight mirrors joined by half-tori. Their coil-center joins
  are explicitly non-tangent, so their vacuum field is not a boundary or
  spline-coefficient oracle for this smooth equilibrium map;
- Ilgisonis, Berk, and Pastukhov provide a second-order finite-beta trend for
  quadrupole mirrors linked by elliptical toroidal cells. Use its outward,
  nonlinear pressure displacement as a sign/scaling check only; the expansion
  parameter depends on beta, inverse aspect ratio, and ellipticity, and the
  model is not the same scalar-pressure 3-D volume problem;
- the paraxial relation `B_vac^2 = B^2 + 2 mu0 p_perp` supplies a local
  diamagnetic sign check. It is not a high-beta scalar-pressure validation.

## 5. Solver and differentiation decisions

### 5.1 Fast primal path

The CLI may be host-based and nondifferentiable. Keep the measured policy:

- L-BFGS globalizes fixed-boundary states;
- exact residual Newton-GMRES polishes them;
- sparse local/cyclic factors are rebuilt only under the documented policy;
- bounded SciPy trust-region least squares solves free-boundary states;
- optimizer success never substitutes for the physical residual gate;
- every reported linear solve records a true, explicitly recomputed residual.

For closed geometry, use continuation and warm starts rather than raising
`max_iterations` as a substitute for conditioning. Adaptive bisection inserts
a midpoint when a continuation stage fails. A failed stage is retained only
as compact diagnostic evidence.

### 5.2 SOLVAX boundary

SOLVAX 0.8.3 provides pytree GMRES/GCROT, periodic banded operators,
preconditioners, matrix-free Newton-Krylov, and implicit root/linear wrappers.
It should own generic algorithms only when an A/B replacement:

1. deletes net code from `vmec_jax`;
2. preserves bounds, line search/globalization, history, and true residuals;
3. matches or improves runtime and peak memory on CPU;
4. passes primal and transpose parity.

`solvax.newton_krylov` currently has no bounds or line search, so it does not
replace the free-boundary trust-region solve or the fixed-boundary
globalization. The earlier open GMRES A/B followed the same iteration curve
without a runtime benefit, so SciPy host GMRES remains acceptable. Revisit one
consolidated GMRES/implicit-wrapper A/B during the deletion phase, not while
hybrid physics is changing. If a SOLVAX 0.8.3 API is adopted, raise the package
minimum from the stale `>=0.2.0` declaration.

Do not add JAXopt, Lineax, Optimistix, Optax, or Equinox in this PR. JAXopt and
Optimistix both recommend implicit differentiation for converged roots, and
Lineax supplies reusable transpose-aware linear solves, but their generic
wrappers do not supply mirror residuals, end conditions, shape calculus, or
the local factors already required here. Replacing a proven local wrapper is
allowed only in T11 when it removes net code and preserves diagnostics.

### 5.3 Differentiation

Differentiate the converged coefficient residual `F(u,p)=0`, never unrolled
nonlinear iterations:

```
F_u du = -F_p dp                         (forward tangent),
F_u^T lambda = Q_u^T                    (reverse adjoint),
dQ/dp = Q_p - lambda^T F_p.
```

Use forward mode for a few control directions and reverse mode for scalar
outputs with many controls. JAX supplies exact residual JVP/VJP actions;
linear systems reuse the primal preconditioner and report true residuals.
Cached `jax.linearize` is used only where measured faster within memory limits;
the free-boundary benchmark favored repeated actions.

The Skene-Burns spectral-adjoint work supports this sparse-operator plus
transpose-solve architecture, but its symbolic Dedalus graph is not portable
code for this solver. JAX `custom_root`, JAXopt, and SOLVAX express the same
implicit-function theorem; adopting another wrapper is valuable only if it
deletes local code without losing diagnostics.

No tangent or adjoint is promoted until the corresponding primal state passes
strong-force and refinement gates.

## 6. Promotion gates

Every promoted equilibrium lane must satisfy all applicable gates.

### 6.1 Numerical and physical gates

- normalized component-wise variational maximum `<= 1e-12`;
- a documented double-precision exception no larger than `1e-11` only if a
  refinement study demonstrates the floor;
- independent staggered weak residual at the same floor;
- true primal/transpose linear relative residual `<= 1e-8`;
- normalized `div(B) <= 2e-12` for the closed Clebsch discretization and the
  existing stricter open gates where applicable;
- one-sign Jacobian, nested radial surfaces, positive clearance, finite fields;
- independently reconstructed strong force decreases monotonically in the
  declared physical refinement family;
- promoted open/hybrid finest all-volume strong force `< 5e-2`, with bulk and
  axis regions reported separately;
- observables (`B`, geometry, pressure, iota, magnetic well) converge, not
  merely residuals;
- `ftol`, iteration count, wall time, peak memory, and residual history are
  recorded.

### 6.2 Free-boundary gates

- plasma and vacuum tangency each refine toward zero;
- normalized normal-stress residual refines toward zero;
- exterior Neumann compatibility and cap balance pass;
- panel order, angular panels, and plasma grid are refined independently;
- external field callable and MGRID paths agree where both apply;
- beta calibration reaches the requested central pressure without silently
  changing the reference definition;
- every plotted beta surface comes from that beta's solved equilibrium.

### 6.3 Derivative gates

- tangent and adjoint actions agree with dense tiny oracles;
- transpose identity passes;
- promoted gradients agree with centered finite differences of fully
  reconverged equilibria;
- finite differences vary step size and retain a stable error plateau;
- centerline, section, pressure, current, and supplied-field controls are
  tested only for lanes that expose them.

## 7. Finite execution plan

The remaining work is six tranches. Finish them in order. Commit and push
after each accepted tranche. Do not expand scope between tranches.

### T9b. Converge the closed hybrid primal

The center map, its independent weak work, the radius-translation gauge, dense
oracle, radial transfer, and coupled sparse factor are complete. Do not reopen
their design unless a regression is demonstrated. The remaining work is:

1. Freeze the accepted 16-control half-straight physical axis. Refine its
   periodic cubic spline space to 32 and 64 controls without regenerating a
   control polygon. Because the knot sets are nested under doubling, transfer
   by exact periodic knot insertion. A collocation fit is acceptable only if
   dense value, first-derivative, and second-derivative parity all pass at
   `2e-12`; otherwise implement periodic Boehm insertion in `basis.py`.
2. Transfer the LCFS, interior radius, center map, and stream function through
   the same nested hierarchy. Add one coefficient-transfer behavior test and
   reuse it for every state block; do not add a continuation module or public
   helper merely for this experiment.
3. Solve the unchanged half-straight geometry at controls `16,32,64`, radial
   grids `ns=5,9`, poloidal orders `mpol=5,7`, and two longitudinal quadrature
   orders. Use warm starts and midpoint continuation. Record represented
   geometry error, variational/weak/strong force, axis/bulk force, true linear
   residual, Jacobian, time, and peak memory.
4. Require `ftol<=1e-12`, weak parity, linear residual `<=1e-8`, positive
   geometry, monotonically decreasing same-geometry strong force, finest
   all-volume force `<5e-2`, and stable observables. Compare circular-limit
   center work and magnetic-axis motion with ordinary vmec_jax/VMEC2000 and
   the GVEC two-coordinate map; there is no racetrack parity oracle.
5. Stop after 64 controls, 30 minutes, or 8 GiB per state. If the gate fails,
   remove the closed hybrid from release scope and skip T9c/T9d. Retain the
   circular closed validation and one compact negative benchmark, but delete
   hybrid public API, example, output, and plotting scaffolds.
6. Only after the half-straight gate passes, continue ellipse aspect ratio,
   return-localized 90-degree section rotation, and current fractions
   `0,0.25,0.5,0.75,1`. The straight central 50% must keep zero section-twist
   rate to tolerance. Trace at least four radial/phase labels for 20 turns at
   zero and target current; each line must traverse both straight legs and both
   returns. Report geometric and current-driven iota separately.

Exit: the solved magnetic axis and target geometry reach `ftol <= 1e-12`, weak
parity, true linear residual `<=1e-8`, a fixed physical LCFS, positive
geometry, and the strong-force refinement gate.

### T9c. Establish physics limits

1. Build three hybrids with increasing straight-leg-to-return scale, using
   the same central-leg flux, pressure, current, and section data.
2. Compare the central 50% of each leg against the promoted fixed-open spline
   mirror. Require convergence of geometry, axis field, section moments,
   `|B|`, local force, and pressure. Global closed-circuit iota is not an
   open-limit observable.
3. Run fixed-LCFS beta continuation at `0, 0.01, 0.03, 0.10`. The LCFS must
   not move. Attempt 25% only as labeled research evidence after the supported
   sequence passes; do not add 50% to the required closed-hybrid matrix.
4. In the long-leg sequence, compare the sign and normalized pressure scaling
   of the interior-axis/section displacement with the linked-quadrupole
   expansion of Ilgisonis, Berk, and Pastukhov. Require the diamagnetic sign
   implied by `B_vac^2 = B^2 + 2 mu0 p_perp`, but do not require coefficient
   parity outside that model's small-beta, small-inverse-aspect-ratio regime.
5. Report interior surfaces, `|B|`, pressure, iota, magnetic well, strong/weak
   force, and residual histories. Do not compare fixed-LCFS displacement with
   free-boundary linked-mirror literature.

Exit: the three-member open-leg limit and 0-10% fixed-LCFS sequence pass
independent refinement and observable gates.

### T9d. Promote derivatives, output, and the example

1. Validate closed-hybrid tangents and adjoints for pressure, current, section
   coefficients, and centerline B-spline controls against reconverged finite
   differences.
2. Write/read one periodic MOUT schema explicitly distinct from toroidal WOUT;
   verify round trip and CLI `--plot mout_*.nc`.
3. Add one parser-free root hybrid example with editable inputs at the top.
   It runs the promoted continuation and produces horizontal straight-leg 3-D
   geometry, field lines, `|B|` on the LCFS, cross-sections, iota, magnetic
   well, pressure/current profiles, and convergence histories.
4. Regenerate `benchmarks/mirror_hybrid_fixed_boundary.json` and one compressed
   hybrid showcase figure. Generated run directories stay ignored.

Exit: primal, derivatives, MOUT, CLI plots, benchmark, and root example all
use the same promoted state and pass release tests.

### T10. Bound the nonaxisymmetric free-open lane

1. Seed from the promoted weakly rotating fixed-open state and supplied-field
   projection.
2. Continue pressure, ellipticity, and rotation separately on three grids.
3. Limit each state to 1,000 nonlinear iterations, 30 minutes, and 8 GiB on
   the reference CPU. GPU is supplementary until CPU parity is shown.
4. Require local nonaxisymmetric coefficients, interface residuals, strong
   force, geometry, and observables to refine consistently.
5. If any third-grid gate fails, remove the lane from the public API and docs,
   retain one compact negative JSON entry, and defer it. Do not tune only the
   optimizer residual or preserve public scaffolding.

Exit: promotion with all gates or an explicit, code-reducing deferral.

### T11. Simplify and consolidate

1. A/B one SOLVAX 0.8.3 GMRES/implicit-wrapper consolidation. Keep it only if
   it deletes net code and passes runtime, memory, primal, transpose, and
   diagnostics parity. Candidate deletions are duplicated SciPy GMRES and
   implicit linear-solve plumbing in `solver.py`, `free_boundary.py`, and
   `implicit.py`; `solvax.newton_krylov` does not replace bounded or
   trust-region globalization.
2. Delete `benchmarks/run_mirror_exterior_endpoints.py` after its retained
   evidence is reproducible from the root example or an ESSOS-side runner.
3. Remove `mirror_performance.png`; fold retained performance data into a
   scientific figure.
4. Consolidate repeated spline, panel, beta, and derivative fixtures. Keep one
   exact, one refinement, and one derivative test per behavior. In particular,
   split the 1,946-line spline test only if the result is easier to navigate
   and the total test line count decreases.
5. Remove private imports from examples, stale schema paths beyond one
   documented migration, duplicate packers/residual wrappers, and
   algorithm-choice tests that do not verify behavior.
6. Keep the 13-module ownership layout unless a deletion naturally removes a
   module. Do not move code merely to satisfy a per-file count.
7. Audit docstrings for equations, shapes, units, boundary meaning, and
   differentiability. Comments explain non-obvious numerical choices, not
   line-by-line mechanics.

Merge budgets, measured against fetched `origin/main`:

- at most 46 changed files;
- at most 7,500 mirror source lines, with a stretch target of 7,200;
- at most 4,000 mirror-test lines;
- at most 18 public mirror names;
- exactly four compact mirror benchmark JSON files;
- at most three root mirror examples and three compressed showcase figures;
- no generated MOUT, CSV run tree, restart, cache, `.DS_Store`, or raw GPU
  output in git.
- every retained runtime/memory benchmark is measured at a committed clean
  tree; exploratory dirty-tree measurements are labeled and then replaced.

Exit: every budget passes together and the full scientific gates remain
unchanged.

### T12. Release documentation and audit

1. Make README show the four supported models, one compact equation block,
   one API snippet, and the three retained scientific figures. Do not present
   research states as supported.
2. Make `docs/mirror_geometry.rst` the full user/theory page: model and cut
   semantics, bases, solver, force definitions, free boundary, derivatives,
   outputs, validation matrix, limitations, and reproduction commands.
3. Keep examples self-contained, parser-free, and configured at the top.
   Examples call public scientific helpers rather than reimplementing geometry
   or output internals.
4. Run normal and full mirror suites, core regression tests affected by the
   diff, strict Sphinx, pre-commit, package build/install smoke, example smoke,
   MOUT/CLI plot smoke, and coverage.
5. Inspect CI once after the full local push, fix failures in one batch, review
   the final PR diff against `origin/main`, and keep the PR draft until all
   gates and budgets pass.

Exit: all checks green, artifacts reviewed, branch mergeable, no unresolved
promotion caveats. Only then mark PR #22 ready for review.

## 8. Ownership and file contract

| File | Scientific owner |
| --- | --- |
| `model.py` | input, resolution, end-condition, and state contracts |
| `basis.py` | Fourier/Chebyshev/B-spline operators and quadrature |
| `splines.py` | coefficient maps, projection, initialization, and fixed solve adapter |
| `geometry.py` | open/closed coordinates, frame, metrics, and field conversion |
| `analytic.py` | independent analytic/paraxial fixtures only |
| `forces.py` | energy, variational/weak residuals, and strong-force diagnostics |
| `solver.py` | shared fixed-boundary host globalization/Newton policy |
| `exterior.py` | closed panels, quadrature, and geometric layer kernels |
| `exterior_bie.py` | exterior Laplace solve and field coupling |
| `free_boundary.py` | square coupled free-boundary residual and continuation |
| `implicit.py` | converged-state tangent and adjoint solves |
| `output.py` | MOUT, restart migration, scientific summaries, and plots |
| `__init__.py` | small lazy public API |

Do not add `continuation.py`, `plotting.py`, `restart.py`, `vacuum.py`, coil
modules, or topology-specific API facades. Those prior scaffolds were removed.
Simplification means deleting duplication while preserving equation ownership,
not hiding unrelated physics in a generic file.

## 9. Canonical evidence

Only these benchmark files are canonical:

1. `benchmarks/mirror_fixed_boundary.json`;
2. `benchmarks/mirror_free_boundary_axisymmetric.json`;
3. `benchmarks/mirror_free_boundary_nonaxisymmetric.json`, promoted or
   explicitly negative;
4. `benchmarks/mirror_hybrid_fixed_boundary.json`.

Each record includes commit and dirty state, hardware, precision, basis,
resolution, represented modes, tolerances, iterations, wall time, peak memory,
variational/weak/strong residuals, geometry checks, observables, independent
comparison errors, derivative errors where applicable, and promotion status.
The current fixed-boundary record is explicitly marked
`"measurement_dirty": true`, and the hybrid record predates the accepted
center map. Both are noncanonical until regenerated from the final clean
implementation; neither may support a README performance claim.

Figures are derived summaries, never the numerical authority. Every work
report states steps, results including failed gates, tests/hardware, files and
ownership, next steps, lane percentages, and concrete user input needed.

### 9.1 Center-map execution checkpoint (2026-07-15)

Commits `b8895aa0` through `9e2c171d` add, condition, and converge the optional
two-component closed-volume map through the existing state, spline, geometry,
force, solver, free-boundary-diagnostic, and implicit block contracts. They
introduce no module or public facade. Accepted evidence is:

- zero-map metric and Cartesian parity is bitwise exact;
- the Cartesian LCFS is fixed while the represented magnetic axis moves;
- periodic coefficient transfer and pack/unpack preserve both components;
- autodiff and the independent radial-Gauss first variation agree for radius,
  stream function, and center fields;
- all three separable preconditioner blocks are finite;
- commit `f3da1f67` removes the interior radius field's real `m=1` translation
  mode whenever center variables are active, defining a unique section center;
- the dense residual Newton lane covers center-map systems through 2,048
  variables. Larger states retain matrix-free Newton;
- invalid infinite-energy trial states are rejected before line-search
  acceptance, and the configured nonlinear iteration limit is honored;
- rank-revealing QR and radial restart transfer recover refined circular solves
  below five minutes without adding a module or public name;
- pre-commit passes, the normal mirror suite is `119 passed, 11 skipped` in
  311.46 s on the local JAX 0.10.2 CPU environment.

The early `mpol=0` nonlinear diagnostics were invalid for a closed torus:
without poloidal stream-function modes, center virtual work was `0.953` and
the solve ran to bounds. Those results are retained only as a resolution guard,
not evidence against the map. Empirical center bounds, a restricted
`q=(1-s)Q(u)` trial, energy-globalized post-polish, and exact block alternation
were rejected and are not in source.

Dense Hessians at zero and small displacement are symmetric to `2e-16`; manual
and autodiff center work agree from `2e-16` (`mpol=0`) to `1e-13` (`mpol=1`).
They expose strong radius-center coupling and lambda-dominated small modes.
Removing the radius `m=1` gauge reduces the accepted `ns=5`, `mpol=3`,
four-control system to 200 variables. Direct dense residual Newton reaches
variational `4.24e-13`, true linear residual `1.78e-13`, strong force
`2.82e-3`, center shift `8.62e-3`, and positive Jacobian in four steps. The
12-control full test passes in 33.75 s on one RTX A4000 with CUDA JAX 0.6.2;
peak host memory is 2.68 GiB.

The circular center-map formulation passes independent poloidal, radial, and
periodic-control trends. The coupled colored sparse factor resolves the
2,664-variable matrix-free blocker: the half-straight radial restart now
converges in 4,300 rather than 200,000 Krylov iterations. Its strong force
still plateaus just above `5e-2`. Continue in this order:

1. Preserve the accepted sparse support oracle and frozen center-map local
   factor. The measured `ns=9, mpol=5`, 16-control result is 4,300 Krylov
   iterations, true linear residual `4.98e-10`, 54.3 s, and less than 2.4 GiB;
   regressions above 5,000 iterations, 30 minutes, or 8 GiB fail this gate.
2. Freeze the 16-control physical axis and transfer it into nested 32- and
   64-control periodic spline spaces. Do not regenerate a different stadium at
   each count. Transfer boundary and state with the same knot hierarchy.
3. Repeat half-straight `ns=5,9`, `mpol=5,7` refinement. Require positive
   geometry, weak parity, `ftol`, and monotone same-geometry strong force below
   `5e-2`. Compare the center map's Cartesian work with VMEC2000 `R/Z` and
   GVEC's two-coordinate map in this tranche.
4. If the gate still fails at 64 controls after the scalable solve, remove the
   fixed closed hybrid from release scope, retain one compact negative record,
   and skip T9c/T9d as required by T9b. Otherwise continue ellipse, twist, and
   current continuation.

## 10. Completion estimate

Percentages represent promotion evidence, not implementation volume.

| Lane | Complete | Remaining evidence |
| --- | ---: | --- |
| Fixed open axisymmetric | 100% | maintain shared-core gates |
| Fixed open nonaxisymmetric | 100% | maintain shared-core gates |
| Free open axisymmetric through 10% | 100% | maintain support ceiling |
| Free open nonaxisymmetric | 35% | bounded three-grid disposition |
| Fixed closed B-spline hybrid | 75% | same-geometry half-straight gate, then section/current or explicit deferral |
| Structured preconditioning | 98% | retain the accepted center-map regression gates during refinement |
| Implicit differentiation | 90% | closed-hybrid derivatives after primal promotion |
| Code/API simplification | 55% | 53 files, 8,696 source lines, 5,007 test lines, and 20 names must meet T11 budgets |
| Docs/examples/artifacts | 73% | hybrid showcase and final README/docs reduction |
| ESSOS ownership separation | 90% | remove the remaining ESSOS-owned runner |

Weighted completion of required release models is approximately 82%.
Free closed hybrid and ANIMEC are excluded because they are explicitly
deferred.

## 11. Reviewed sources

Source revisions reviewed locally on 2026-07-15:

- `vmec_jax` main `ed4ac7ac` and audited mirror source `9e2c171d`;
- DESC master `24aa7b9d`, mirror `0dba071d`, mirror-anisotropy `805b77fc`,
  racetrack `2014ed0e`, cylindrical/Chebyshev `6f85f50a`, and
  straight-stellarator `8cf50b58`, plus finite-element branches `829e2db0`
  and `fb90e65`;
- STELLOPT develop `e03e72e9`, legacy `ANIMEC` `2337455d`, integrated
  `animec` `91bfd08e`, and local VMEC2000 `728af8ba`, including force,
  pressure, free-interface, preconditioner, and output source;
- SOLVAX 0.8.3 `a904ac20` and current main `255d280b`, including the latter's
  independent transpose solver and auxiliary linear diagnostics;
- Pleiades `0161abb3` pinned reference script and data.

Primary references:

- VMEC2000/ANIMEC source: <https://github.com/PrincetonUniversity/STELLOPT>
- Hirshman and Whitson, VMEC inverse-coordinate energy principle:
  <https://www.osti.gov/biblio/5497291>
- Hirshman, van Rij, and Merkel, free-boundary VMEC:
  <https://doi.org/10.1016/0010-4655(86)90058-5>
- Hirshman and Betancourt, VMEC radial preconditioning:
  <https://doi.org/10.1017/S0022377800015509>
- ANIMEC overview: <https://www.epfl.ch/research/domains/swiss-plasma-center/research/theory/codes/animec/>
- Cooper et al., anisotropic free-boundary equilibria:
  <https://doi.org/10.1016/j.cpc.2009.04.006>
- Asahi et al., ANIMEC pressure model:
  <https://doi.org/10.1585/pfr.6.2403123>
- DESC source and branches: <https://github.com/PlasmaControl/DESC>
- DESC free-boundary formulation: <https://arxiv.org/abs/2412.05680>
- GVEC flexible-coordinate formulation: <https://arxiv.org/abs/2410.17595>
- GVEC theory and implementation: <https://doi.org/10.21105/joss.09670>
- SOLVAX: <https://github.com/uwplasma/SOLVAX>
- Pleiades: <https://github.com/eepeterson/pleiades>
- WHAM axisymmetric-mirror physics: <https://doi.org/10.1017/S0022377823000806>
- Frank et al., high-field tandem-mirror equilibrium:
  <https://doi.org/10.1017/S002237782510055X>
- Agren et al., straight-field-line mirror:
  <https://info.fusion.ciemat.es/OCS/EPS2005/pdf/P4_069.pdf>
- Savenko and Agren, finite-beta SFLM ellipticity:
  <https://doi.org/10.1063/1.2401153>
- Ilgisonis, Berk, and Pastukhov, finite-beta linked quadrupole mirrors:
  <https://doi.org/10.2172/10179323>
- Rodriguez, Helander, and Goodman, paraxial mirror/maximum-J analysis:
  <https://doi.org/10.1017/S0022377824000345>
- Feng et al., linked mirror concept: <https://arxiv.org/abs/2103.09457>
- Skene and Burns, automated spectral adjoints:
  <https://arxiv.org/abs/2506.14792>
- Blondel et al., modular implicit differentiation:
  <https://arxiv.org/abs/2105.15183>
- JAXopt implicit differentiation:
  <https://jaxopt.github.io/stable/implicit_diff.html>
- Optimistix implicit adjoints:
  <https://docs.kidger.site/optimistix/api/adjoints/>
- Lineax structured and transpose-aware solves:
  <https://docs.kidger.site/lineax/>
- JAX custom derivatives:
  <https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html>
- JAX `linearize`, `custom_root`, and `custom_linear_solve`:
  <https://docs.jax.dev/en/latest/_autosummary/jax.linearize.html>,
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_root.html>,
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html>

References justify design decisions. Only reproduced tests and compact
benchmark records count as release evidence.
