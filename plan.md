# Mirror equilibrium release plan

Status: authoritative plan for draft PR #22. This file supersedes
`/Users/rogeriojorge/Downloads/plan_mirror.md` and every earlier version of
`plan.md`. Do not create another roadmap. Commits and compact benchmark JSON
files are the execution log.

Audit baseline (2026-07-14):

- branch: `codex/mirror-geometry` at `ec3e2c5a`;
- base: `origin/main` at `ed4ac7ac`, zero commits behind and 291 ahead;
- PR #22: open, draft, and mergeable;
- diff: 67 files, 17,957 insertions, and 1,632 deletions;
- mirror package: 15 modules, 8,977 source lines, and 24 public names;
- mirror tests: 5,199 lines in 15 files including `__init__.py`;
- representative current-head run: five required-topology tests passed and one
  optional test skipped in 52.21 seconds;
- grouped CI: build/docs, console, fast Python 3.10/3.12, every mirror shard,
  examples, and parity a/d passed at the audit cutoff. Three inherited core
  shards were still running and are not being polled repeatedly.

The branch contains real equilibrium work, but it is not release-ready. In
particular, one nominally axisymmetric ESSOS benchmark offsets the two coils in
opposite x directions, the closed-hybrid pointwise force reconstruction is not
converged, the periodic preconditioner omits important geometry-stream
coupling, and research ANIMEC code lacks source parity. These defects determine
the order below.

## 1. Final product and scope

PR #22 will deliver three supported scalar-pressure equilibrium models:

1. straight-axis fixed-boundary mirrors, both axisymmetric and
   nonaxisymmetric;
2. straight-axis axisymmetric free-boundary mirrors in a supplied external
   field, including beta continuation through 50%;
3. a fixed-boundary toroidal stellarator-mirror hybrid represented natively by
   periodic cubic B-splines, with two long straight mirror legs joined by two
   smooth stellarator returns.

The following are conditional experiments, not release blockers:

- nonaxisymmetric open free boundary;
- free-boundary closed hybrid;
- an anisotropic-pressure/ANIMEC model.

For this PR the first and third are deferred. The free closed hybrid receives
one bounded attempt only after the fixed hybrid is promoted and only if the
needed smooth-toroidal virtual-casing API is released and reproducible. A
failed conditional experiment leaves one compact negative benchmark and no
public API or example.

A supported result must be a converged nested-surface ideal-MHD equilibrium. A
prescribed tube sampled in a coil field, a Fourier fit of a square, a small
`B.n` surface without a plasma-force solve, or an optimizer success flag is not
an equilibrium result.

### 1.1 Promotion contract

Every supported lane must satisfy all applicable gates:

- component-wise normalized variational residual no larger than `1e-12`;
- if a resolution-dependent double-precision floor is demonstrated, the only
  allowed exception is a documented floor no larger than `1e-11`;
- an independently assembled staggered weak first variation that converges to
  the same floor without calling `jax.grad` on the production energy;
- a pointwise `J x B - grad(p)` reconstruction that passes manufactured
  refinement and decreases under physical grid refinement;
- positive Jacobian, nested surfaces, adequate self-clearance, and normalized
  `div(B)` near double-precision roundoff;
- stable physical observables on three independently refined grids;
- for free boundary, separate area-weighted `B.n`, total-pressure jump, and
  artificial-cap compatibility diagnostics;
- an analytic, asymptotic, or independent-code comparison;
- forward and reverse implicit derivatives for every advertised differentiable
  control, checked against reconverged centered finite differences;
- one parser-free root example, one compact benchmark record, current docs,
  and polished compressed figures.

The nonlinear residual, weak residual, pointwise force, and boundary
constraints are separate quantities. None may be relabeled as another.

### 1.2 Explicit non-goals

This PR does not implement kinetic end losses, sheaths, transport, stability,
islands, stochastic regions, a mirror Boozer file format, radial B-splines,
poloidal finite elements, coil construction, Biot-Savart, lab-frame coil field
line tracing, or a second optimization framework.

Open mirrors are not written as toroidal WOUT files and are not sent through
`booz_xform_jax`. VMEC2000 is not an oracle for an open topology.

## 2. Physical and numerical model

### 2.1 Open straight mirrors

The coordinates are

`(s, theta, xi) in [0,1] x [0,2*pi) x [-1,1]`.

The axis is straight, `theta` is periodic, and `xi` is nonperiodic. The lateral
surface `s=1` is the plasma-vacuum interface. The planes `xi=-1` and `xi=1`
are fixed computational cuts through which magnetic flux passes. They are not
material interfaces and do not impose `B.n=0`.

The divergence-free field is

`sqrt(g) B^theta = I'(s) - d(lambda)/dxi`,

`sqrt(g) B^xi = Psi'(s) + d(lambda)/dtheta`,

with `B^s=0`, fixed axial flux, and a weighted zero-surface-mean gauge for
`lambda`. Axisymmetry is the `mpol=0` case of the same model.

Fixed boundary prescribes the lateral radius and both cut sections. The
supported free-boundary path varies the lateral LCFS while keeping both cuts
fixed. Disks close the two cuts only for the unbounded exterior Green problem.
Their Neumann data continue the nonzero through-flux; pressure balance is
enforced only on the lateral LCFS.

### 2.2 Closed stellarator-mirror hybrid

The hybrid remains toroidal. Its periodic Cartesian centerline has two long,
parallel straight legs and two smooth curved returns. Up-down and leg-exchange
symmetries are imposed through control-point maps, not by defining another
basis. The general basis still supports asymmetric verification cases.

A Bishop/rotation-minimizing frame is mandatory because a Frenet frame is
undefined on zero-curvature straight spans. Periodic frame holonomy must be
removed explicitly. The complete surface, section size and orientation, and
stream function are periodic in the spline coordinate and have no end caps.

The production representation is:

- clamped cubic B-splines in the open axial coordinate for fixed mirrors;
- periodic cubic B-splines for the hybrid centerline, section coefficients,
  and stream function;
- the existing VMEC-like staggered radial mesh;
- Fourier modes in the periodic poloidal angle.

This is the intended meaning of full B-spline mirror support. It solves the
long-straight-span problem without replacing the radial and poloidal
discretizations. A global Fourier centerline is rejected because resolving a
long exactly straight span and localized returns requires many nonlocal modes.

The supported open free-boundary state remains the Chebyshev-Lobatto nodal
state. Two coefficient-native spline attempts failed an independent exterior
shape-gradient gate at the artificial cap rim. Reopening that cap-corner
problem would add a third exterior formulation and violate the finite scope.

### 2.3 Scalar pressure only in PR #22

The release energy is the scalar ideal-MHD functional already used by
vmec_jax. Pressure is a radial profile and the free interface condition uses

`p + B_plasma^2/(2*mu0) = B_vacuum^2/(2*mu0)`.

ANIMEC is deferred to a separate proposal. Its correct model is substantially
more than accepting two pressure arrays: `p_parallel(s,B)` determines

`p_perp = p_parallel - B * (d p_parallel/dB)_s`,

the force uses `K = curl(sigma B)`, free-boundary stress uses `p_perp`, and the
piecewise bi-Maxwellian radial derivatives and firehose/mirror ellipticity
conditions must match VMEC2000. The current branch does not meet this contract.

## 3. Audited implementation state

### 3.1 Credible evidence

- Axisymmetric open fixed boundary reaches `ftol=1e-12`, has analytic cylinder
  and flared-tube checks, an independent weak residual, and implicit
  derivatives.
- Nonaxisymmetric open fixed boundary jointly solves geometry and gauge-free
  stream function at finite current. Native open splines and Chebyshev states
  converge toward each other over three grids. At the finest retained grid,
  relative errors in energy, volume, sampled radius, Cartesian B, and `|B|`
  are about `5.6e-8`, `1.0e-6`, `2.9e-5`, `7.0e-4`, and `3.1e-4`.
- Fixed-spline reverse derivatives agree with reconverged finite differences
  near `3e-10` in tested directions. The new true forward tangent solves
  `F_u du = -F_p dp` and passes its current open-state checks.
- The analytic module contains independent two-loop, paraxial rotating-ellipse,
  and Straight Field Line Mirror fixtures.
- The axisymmetric free solver produces finite-beta equilibria with roundoff
  variational/weak residuals, increasing radius, decreasing center field, and
  separate tangency/stress diagnostics in analytic-field tests.
- Periodic cubic centerlines, Bishop-frame closure, circular and racetrack
  embeddings, a complete radius/lambda closed solve, and periodic
  equilibrium-coordinate field-line tracing exist and pass focused tests.
- A finite-current rotating-ellipse racetrack reaches the discrete and weak
  residual floors and has nonzero pitch/iota.

### 3.2 Evidence that is not accepted

1. `benchmarks/run_mirror_exterior_endpoints.py` currently sets opposite
   x-offsets on both ESSOS loops for axisymmetric and nonaxisymmetric runs. The
   nominal axisymmetric direct-coil scan is therefore not axisymmetric.
2. The existing axisymmetric JSON uses a different initial radius from the
   current runner and cannot yet be reproduced by that runner. It remains
   historical evidence, not the release benchmark.
3. Earlier physical `B.n` values were computed with panel normals while the
   field was reconstructed with analytic LCFS normals. The implementation now
   separates BIE panel data from the physical trace and gives near-roundoff
   tangency in focused tests, but the corrected concentric-coil refinement has
   not been completed.
4. Nonaxisymmetric free-boundary global residuals converge, but local Fourier
   observables change by 73-81% between available grids and the medium case
   costs roughly 801 seconds and 4.25 GiB. This lane is deferred.
5. The finite-beta rotating-ellipse `m=2` field-strength amplitude is about 48%
   above the direct paraxial estimate at the finest bounded knot level.
6. The periodic preconditioner preserves the state but leaves a relative
   linear residual of `0.136` after 3,000 GMRES iterations on an 892-variable
   racetrack. CG and MINRES are also unacceptably stalled. Closed matrix-free
   primal solves are disabled.
7. The complete circular-torus pointwise-force norm improves only from `0.709`
   at `ns=5` to `0.570` at `ns=7`. A roundoff energy gradient does not validate
   this reconstruction.
8. ANIMEC-style closures and tests are exploratory. They have no
   equation-by-equation parity with current STELLOPT `_ANIMEC` and no matched
   independent finite-beta equilibrium.
9. There is no release hybrid example, closed MOUT contract, circular
   VMEC/vmec_jax parity result, open-leg limit, or hybrid derivative result.

### 3.3 Current code structure

The retained ownership is mostly sound:

| Concern | Current source |
|---|---|
| contracts and profiles | `model.py` |
| Chebyshev/Fourier grids | `basis.py` |
| open/closed geometry | `geometry.py` |
| scalar energy and force diagnostics | `forces.py` |
| nodal fixed solve and open preconditioner | `solver.py` |
| spline basis, state, and closed solve | `splines.py` |
| free solve and beta continuation | `free_boundary.py` |
| exterior surface and Green solve | `exterior*.py` |
| converged-state derivatives | `implicit.py`, `free_boundary_implicit.py` |
| MOUT, restart, diagnostics, plots | `output.py` |
| independent analytic fixtures | `analytic.py` |

The remaining bloat is concentrated rather than diffuse: `forces.py` has
1,098 lines, `splines.py` 1,063, `solver.py` 1,001, and `output.py` 912.
ANIMEC branches, duplicated nonlinear-solve plumbing, three exterior modules,
and separate fixed/free derivative modules are the primary simplification
targets.

## 4. External source and literature decisions

### 4.1 VMEC2000 and VMEC++

STELLOPT `origin/develop` at `2f79b9ba` and the local VMEC2000 source are the
reference for the variational principle, radial half mesh, full/half parity,
component residual normalization, continuation, and preconditioning.

The most relevant files are:

- `General/bcovar.f`: metrics, contravariant fields, magnetic pressure, and
  pressure on the half mesh;
- `General/forces.f`: half-to-full averaging and free-boundary edge force;
- `General/residue.f90`: raw `fsqr/fsqz/fsql`, preconditioned residuals, and
  `m=1` constraints;
- `General/precon2d` and `blocktridiagonalsolver`: coupled structured
  preconditioning;
- `NESTOR_vacuum/*`: smooth periodic free-boundary potential solve;
- `_ANIMEC` branches in `fbal.f`, `bcovar.f`, and `jxbforce.f`.

VMEC2000 cannot represent open axial topology. It validates signs,
normalizations, solver structure, and the smooth circular closed-hybrid limit.
VMEC++ is a modern robustness, restart, and software-architecture reference;
it is not an open-mirror oracle.

### 4.2 DESC and its research branches

The audit refreshed and inspected DESC `master` at `24aa7b9dc`, `mirror` at
`0dba071da`, `mirror_anisotropy` at `805b77fc0`, `dd/cylindrical` at
`6f85f50ae`, and both finite-element branches.

Transferable ideas are:

- a nonperiodic basis owns nodes, quadrature, derivatives, interpolation,
  endpoint semantics, and resolution transfer;
- continuation changes pressure, shaping, and resolution separately and can
  backtrack;
- free-boundary `B.n` and magnetic-pressure jump are distinct area-weighted
  residuals;
- automatic perturbations are initial guesses for a reconverged state, not a
  substitute for convergence.

The branches are not production dependencies. The `mirror` branch adds a
2,255-line equilibrium class and a 1,393-line objective file plus large
notebook/binary payloads. Its nominal mirror tests are nearly empty, explicit
end-cap mode selection raises `NotImplementedError`, and many upstream tests
are renamed out of collection. The anisotropy branch records unchecked or
suspicious force quantities. The double-Chebyshev branch adds and tests a basis
inside a repository-wide grid refactor but contains no mirror equilibrium or
boundary-condition validation. None supplies native B-spline straight spans,
cap physics, or a validated open free boundary.

DESC's published high-order free-boundary method is directly relevant to the
future smooth closed hybrid: it treats tangency, total-pressure jump, and an
optional sheet-current condition separately. Its smooth toroidal singular
quadrature does not remove the open cap-rim singularity.

### 4.3 GVEC, VEPEC, and Pleiades

GVEC confirms two design choices: a variational fixed-boundary solver can use
B-splines without changing the physics, and a generalized axis-aligned frame
is preferable for complicated closed axes. GVEC's production splines are
radial and its surfaces remain Fourier-periodic, so its code is not ported.
Its Frenet documentation also explicitly warns that zero curvature makes the
normal and binormal undefined, supporting the Bishop-frame choice here.

VEPEC is the closest historical 3-D open-mirror code. It evolves a vector
potential and uses tricubic splines so interpolation preserves `div(B)`. It is
validation precedent for spline mirror fields, not an available dependency.

Pleiades is the independent axisymmetric finite-beta reference. Published
high-beta mirror calculations show outward flux-surface expansion and the
paraxial trend `B_center/B_vac ~= sqrt(1-beta)`, and check firehose/mirror
ellipticity for anisotropic cases. Its Green-function formulation differs from
vmec_jax, making agreement in sign, magnitude, and refinement useful.

### 4.4 Mirror analytic references

Required open nonaxisymmetric checks are:

- flux determinant `X1c*Y1s - X1s*Y1c = Bbar/B0(z)`;
- no order-`r`, `m=1` variation in `|B|`;
- order-`r^2`, `m=2` quadrupole magnitude and phase;
- ellipse magnitude/orientation from the paraxial Riccati equation;
- Straight Field Line Mirror Clebsch labels, field direction, ellipticity, and
  low-radius truncation order;
- exact two-circular-loop on-axis field and the low-radius off-axis `B_r` and
  `B_z` expansion.

Goodman-Freidberg-Lane and Pearlstein provide finite-beta long-thin trends, not
full-radius equality. The 2025 Pleiades/RealTwin results provide an independent
finite-beta axisymmetric comparison.

### 4.5 Linked mirrors and the hybrid

Feng et al. analyze two straight mirrors joined by two half tori and obtain
rotational transform from nonparallel sections. Ilgisonis, Berk, and Pastukhov
analyze finite-beta toroidally linked quadrupole mirrors. Ranjan uses helical
returns. These works validate the topology and expected pitch, magnetic-well,
and beta trends, but do not define a unique boundary.

The hybrid therefore has three independent limits:

1. circular-axis parity with normal vmec_jax and VMEC2000;
2. increasing straight-leg length approaching the open spline mirror locally;
3. nonzero pitch/iota from current and return geometry with nested field lines.

### 4.6 Derivatives and solver libraries

For a converged residual `F(u,p)=0`, production derivatives are

`F_u du = -F_p dp`

and

`F_u^T lambda = Q_u`, `dQ/dp = Q_p - lambda^T F_p`.

The policy is:

- use forward implicit tangents for a small number of control directions;
- use reverse implicit adjoints for scalar objectives with many controls;
- use exact JAX JVP/VJP actions for `F_u` and its transpose;
- use reconverged centered finite differences only as validation;
- never reverse-differentiate the nonlinear iteration history;
- report nonlinear and linearized residuals with every derivative.

JAX `custom_linear_solve`/`custom_root`, JAXopt, Optimistix, Lineax, and the
automated spectral-adjoint work all support this implicit-function approach.
They do not remove the need for a physics-aware preconditioner. Adding another
solver/object model would increase this branch without changing the dominant
operator, so JAXopt, Optax, Lineax, Equinox, and Optimistix are not added.

SOLVAX `main` at `255d280` provides reusable Krylov, block-Thomas, banded,
operator, and implicit-solve tools. The tested SOLVAX GMRES replacement failed
the current residual/runtime gates. PR #22 keeps SciPy's host nonlinear and
Krylov control plus the already-used SOLVAX block-Thomas factorization. Generic
new solver machinery belongs in SOLVAX only after a measured mirror case proves
it.

## 5. Ownership and target repository shape

- vmec_jax owns equilibrium coordinates, energies/residuals, continuation,
  boundary coupling, converged-state sensitivities, MOUT, and plots of solved
  states.
- ESSOS owns coils, coil geometry, Biot-Savart, mgrid generation, and lab-frame
  coil field-line tracing. vmec_jax accepts a vectorized `xyz -> B` callable or
  `MgridField`.
- SOLVAX owns reusable linear solvers and generic preconditioners.
- virtual-casing-jax owns generic singular virtual-casing kernels. Mirror code
  owns only its open-surface geometry and cap data.
- SciPy may drive the fast nondifferentiable CLI solve. Differentiability is
  attached to the converged residual.

Final branch targets relative to `origin/main`:

- no more than 58 changed files;
- no more than 13 mirror source modules;
- no more than 8,000 mirror source lines;
- no more than 22 public lazy names;
- no source file above 1,000 lines; prefer below 900;
- no mandatory dependency needed only by an unreleased optional lane;
- three root examples and four compact benchmark JSON/CSV files at most;
- only compressed showcase figures, normally below 300 KiB each;
- no generated MOUT/WOUT/mgrid/raw trace output in git.

Target module layout:

1. `model.py`, `basis.py`, `geometry.py`, `forces.py`;
2. `solver.py`, `splines.py`, `free_boundary.py`, `implicit.py`;
3. `exterior.py`, `exterior_bie.py`;
4. `analytic.py`, `output.py`, `__init__.py`.

`exterior_mesh.py` merges into `exterior.py` and
`free_boundary_implicit.py` merges into `implicit.py`. This reduces ownership
boundaries instead of merely moving lines. ANIMEC code is deleted rather than
relocated.

Target test layout is at most ten substantive files: analytic, model/basis,
geometry/fields, forces, fixed solver, splines/hybrid, exterior, free boundary,
implicit, and output/examples. Small continuation and promotion-guard files
merge into their owners.

The public API will contain only model/configuration, fixed/free solves, beta
continuation, the promoted spline-hybrid solve, implicit tangent/adjoint entry
points, and MOUT/plot functions. Analytic fixtures, vectorizers, panels,
preconditioners, and experimental lanes remain internal.

## 6. Ordered work packages

Each package ends with focused tests, a compact evidence update, a small commit,
and a push. CI is checked after grouped work, not polled after every commit. No
new physics lane is introduced before these packages finish.

### WP0 - Correct the baseline and evidence

1. Make ESSOS coil offsets conditional on `not axisymmetric` in
   `run_mirror_exterior_endpoints.py`.
2. For the axisymmetric case, verify concentric-loop symmetry at multiple
   azimuths and exact on-axis agreement before starting a beta solve.
3. Regenerate the axisymmetric beta scan on three grids for
   `[0,.01,.03,.10,.25,.50]`, using one documented initial radius and
   independent plasma, side-panel, cap-panel, and singular-order refinements.
4. Record raw and corrected Neumann compatibility, physical-normal `B.n`,
   stress, variational/weak/pointwise residuals, center radius and field,
   mirror ratio, volume beta, runtime, and peak memory.
5. Replace, do not append to, stale benchmark JSON and figures.
6. Inspect the remaining three CI jobs once they finish and repair only real
   branch regressions.

Gate: the direct-coil benchmark is actually axisymmetric, exactly reproducible,
and all stored claims identify their field model and grid.

### WP1 - Remove unsupported breadth and simplify

1. Delete public ANIMEC, bi-Maxwellian, tabulated `(s,B)` closure, anisotropic
   solve, anisotropic benchmarks, and research-only tests. Retain a short docs
   section explaining the deferred physics and source requirements.
2. Restore unrelated `core/device.py`, `core/freeboundary_diff.py`, and
   virtual-casing dependency edits to `origin/main` unless a retained mirror
   call site proves they are required.
3. Merge the two artificial module boundaries listed in Section 5.
4. Share nodal/spline host solve history, failure reporting, packed linear
   diagnostics, and tangent/adjoint linear-solve plumbing inside mirror code.
5. Simplify MOUT to scalar pressure and one residual vocabulary.
6. Consolidate examples and benchmark files to the target set. Fold the fixed
   gradient demonstration into the fixed nonaxisymmetric example rather than
   retaining a fourth root script.

Gate: changed-file/module/API/line targets are met before new hybrid features
are added; ruff, strict Sphinx, focused tests, and `git diff --check` pass.

### WP2 - Promote open fixed-boundary mirrors

1. Repair the VMEC-like half-to-full pointwise-force reconstruction. Add
   manufactured cylinder, flared tube, and nonaxisymmetric fields with known
   force, then require monotone radial and axial convergence. Report axis,
   first-row, bulk, and end-collar norms separately.
2. Complete three-grid axisymmetric fixed B-spline/Chebyshev parity.
3. For the rotating ellipse, independently refine tube radius, `ns`, `mpol`,
   theta quadrature, and spline knots. Compare the extrapolated low-radius
   `m=2/r^2` coefficient and phase with the paraxial solution; do not hide a
   normalization mismatch by loosening tolerance.
4. For SFLM, verify Clebsch labels, flux determinant, field direction,
   ellipticity, straight nonparallel field lines, and expected low-radius
   truncation order.
5. Show pressure-first and shape-first continuation reach the same state and
   reject crossed surfaces at every trial.
6. Validate open spline tangents and scalar-objective adjoints for boundary,
   pressure, flux, and current over finite-difference step sweeps.

Gate: both axisymmetric and nonaxisymmetric fixed lanes satisfy Section 1.1;
the rotating-ellipse amplitude discrepancy is explained and reduced, not just
plotted.

### WP3 - Promote axisymmetric open free boundary

1. Complete WP0's corrected concentric two-loop scan and the independent
   refinement matrix.
2. Compare beta 1%, 3%, and 10% observables with the retained Pleiades reference
   and beta through 50% with the paraxial pressure-balance trend. Treat
   `sqrt(1-beta)` as an asymptotic trend, not an exact high-beta target.
3. Add exact on-axis and low-radius off-axis circular-loop comparisons before
   plasma response is enabled.
4. Require raw cap compatibility to decrease with panel refinement; the
   cap-only compatibility correction must remain small relative to physical
   through-flux and must never alter lateral data.
5. Measure CPU and office-GPU compile time, warm time, RSS/device memory, and
   nonlinear/linear iterations. Replace dense Jacobian assembly above the
   bounded small-system threshold.
6. Revalidate the existing free-boundary adjoint after simplification for the
   supported scalar controls and one external-field parameter.

Gate: all six beta points converge to `ftol=1e-12`; medium-to-fine center-radius
change is below 0.1% and center-field change below 0.05% at every nonzero beta;
boundary and derivative gates pass.

### WP4 - Build one coupled structured preconditioner

The existing open separable model and the failed periodic axial-only model miss
radius-stream coupling. Make one final physics-aware implementation:

1. Form a frozen approximate Jacobian/Hessian from the same radial-Gauss terms
   as the energy, retaining the 2x2 radius/lambda coupling.
2. Fourier-transform in theta so modes decouple approximately. Preserve radial
   nearest-neighbor and cubic-spline four-span locality.
3. Apply endpoint constraints and the lambda gauge before factorization.
4. Factor the resulting open banded or periodic cyclic sparse blocks with an
   existing SciPy/SOLVAX direct primitive. Do not write another Krylov solver.
5. Use the factor and its transpose as right preconditioners for primal and
   implicit linear solves.
6. Compare no preconditioner, the old separable model, and the coupled model on
   the required open 3-D and closed-racetrack cases.

Gate: medium cases avoid dense Jacobians; true linear residual is below
`1e-8`; iteration growth is bounded when resolution doubles; solved states are
unchanged within discretization error; warm runtime and peak memory improve.
If this single coupled attempt fails, retain the robust bounded dense path,
state its maximum supported size, and defer scalable periodic solves without
adding more preconditioner variants.

### WP5 - Promote the fixed toroidal B-spline hybrid

1. Geometry: enforce periodic C2 closure, positive Jacobian, self-clearance,
   up-down symmetry, leg exchange, two measured zero-curvature straight spans,
   and smooth return curvature.
2. Circular limit: refine radial, theta, and spline resolution independently;
   compare matched boundary/flux/pressure/current cases with normal vmec_jax
   and local VMEC2000 in volume, energy, iota, `|B|`, and force residuals.
3. Racetrack: perform three-grid convergence of iota, mirror ratio, section
   orientation, energy, volume, weak/pointwise force, nestedness, and traced
   field lines.
4. Open-leg limit: increase straight-leg length at fixed minor radius and show
   leg-center sections, field direction, `|B|` modes, and paraxial coefficients
   approach the open B-spline solution over at least three lengths.
5. Fixed-boundary beta scan: run `[0,.01,.03,.10,.25,.50]` with a fixed LCFS
   and hot starts. Report internal-surface motion, field depression, iota,
   mirror ratio, magnetic well, and residuals. Do not claim LCFS displacement
   from a fixed-boundary scan.
6. Derivatives: extend the spline residual to periodic axis/control points and
   validate tangents plus scalar-objective adjoints for centerline, section,
   pressure, flux, and current controls.
7. Output: add one parser-free root example and closed MOUT/plot support for
   horizontal 3-D surfaces, visible field lines, cross sections, `|B|`,
   pressure, iota/pitch, magnetic well, residual history, and refinement.

Gate: complete fixed hybrid with both limits, Section 1.1 residuals, stable
pitch/iota, verified derivatives, and reproducible compressed plots.

### WP6 - One conditional free-hybrid attempt

Start only after WP5 passes. Use the smooth periodic surface, an ESSOS/MGRID
external field, and a released generic virtual-casing interface. Coils remain
in ESSOS. Solve tangency and total-pressure jump separately, continue beta to
50%, and validate LCFS displacement and lab-frame field lines.

Stop after one bounded implementation. Promote only if three-grid boundary
observables, nestedness, residuals, runtime, and memory pass. Otherwise remove
the API/example and record one compact negative result. This package cannot
delay the fixed-hybrid release.

### WP7 - Release evidence and review

1. Recount the repository against Section 5 and remove stale artifacts.
2. Give every public class/function a short purpose-first docstring with units,
   output contract, and validity range. Comments explain gauges, staggering,
   cap data, frame holonomy, and singular quadrature rather than syntax.
3. Add a README capability table and three mirror showcases: axisymmetric free,
   rotating-ellipse fixed, and fixed B-spline hybrid.
4. Keep one focused mirror docs page covering equations, topology, boundary
   conditions, bases, residuals, validation, derivatives, ownership, examples,
   and known limits.
5. Run ruff, `git diff --check`, strict Sphinx, all mirror tests, example smoke,
   complete CI, and final CPU/GPU runtime-memory benchmarks.
6. Keep PR #22 draft until every required lane is promoted and each conditional
   lane is explicitly deferred or supported.

## 7. Promotion matrix

Percentages estimate accepted promotion evidence, not code volume.

| Lane | Completion | Final status / remaining gate |
|---|---:|---|
| Axisymmetric fixed mirror | 90% | pointwise reconstruction, final spline derivatives, release evidence |
| Nonaxisymmetric fixed mirror | 82% | paraxial amplitude, SFLM refinement, pointwise force, release evidence |
| Axisymmetric free mirror | 74% | corrected concentric-coil three-grid scan, scaling, adjoint rerun |
| Nonaxisymmetric free mirror | 55% | explicitly deferred; remove public claim and retain negative evidence |
| Open native B-splines | 84% | fixed-lane promotion; spline free boundary explicitly deferred |
| Fixed closed B-spline hybrid | 58% | limits, pointwise force, beta scan, derivatives, output |
| Free closed hybrid | 10% | one conditional attempt after fixed promotion |
| Structured preconditioning | 48% | one coupled radius/lambda block attempt |
| Implicit derivatives | 78% | closed-axis controls and simplified free adjoint |
| Source/API simplification | 70% | ANIMEC deletion, module/test consolidation, restore unrelated core |
| Documentation/examples | 68% | concise docs, README showcases, hybrid example |
| ESSOS ownership | 100% | retain callable/MGRID interchange only |

Weighted completion of the required release path is approximately 72%. The
percentage will only increase when a gate above passes; deleting an unsupported
lane counts toward simplification, not toward its physics promotion.

## 8. Deferred work after PR #22

- ANIMEC/kinetic pressure, after a separate design reproduces current
  VMEC2000 source branches and an independent matched equilibrium;
- coefficient-native open-spline free boundary, after a cap-corner shape
  calculus or a physically different open exterior formulation is established;
- nonaxisymmetric open free boundary, after that spline/free prerequisite and
  a scalable structured solve exist;
- arbitrary curved open axes;
- a free hybrid if WP6 fails or its external dependency is unreleased;
- scalable periodic preconditioning if WP4's one coupled attempt fails;
- unrolled differentiation, mirror-specific optimizer classes, mirror Boozer,
  VTK, TOML/YAML inputs, and a second plotting/output stack.

## 9. Primary references

- Original attached roadmap: `/Users/rogeriojorge/Downloads/plan_mirror.md`
- VMEC method and documentation:
  https://princetonuniversity.github.io/STELLOPT/VMEC.html
- VMEC2000/ANIMEC source:
  https://github.com/PrincetonUniversity/STELLOPT
- VMEC++ numerics: https://arxiv.org/abs/2502.04374
- DESC source and inspected research branches:
  https://github.com/PlasmaControl/DESC
- DESC high-order free boundary: https://arxiv.org/abs/2412.05680
- GVEC and generalized frame: https://gvec.readthedocs.io/develop/
- Rodriguez, Helander, and Goodman mirror near-axis analysis:
  https://doi.org/10.1017/S0022377824000345
- Agren and Savenko Straight Field Line Mirror:
  https://info.fusion.ciemat.es/OCS/EPS2005/pdf/P4_069.pdf
- Goodman, Freidberg, and Lane long-thin equilibrium:
  https://doi.org/10.1063/1.865851
- Pearlstein quadrupole paraxial equilibrium:
  https://digital.library.unt.edu/ark:/67531/metadc1102940/
- VEPEC: https://www.osti.gov/biblio/6351313
- Pleiades/RealTwin high-beta mirror equilibrium:
  https://doi.org/10.1017/S002237782510055X
- Feng et al. linked mirror: https://arxiv.org/abs/2103.09457
- Ilgisonis, Berk, and Pastukhov linked mirrors:
  https://www.osti.gov/servlets/purl/10179323
- Ranjan helically linked mirror:
  https://digital.library.unt.edu/ark:/67531/metadc1194643/
- ANIMEC variational model: https://doi.org/10.1016/0010-4655(92)90002-G
- ANIMEC bi-Maxwellian model: https://doi.org/10.1088/0029-5515/46/7/001
- ANIMEC free boundary:
  https://www.ornl.gov/publication/three-dimensional-anisotropic-pressure-free-boundary-equilibria
- Fast automated spectral adjoints: https://arxiv.org/abs/2506.14792
- JAX implicit linear solve:
  https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html
- JAXopt implicit differentiation: https://jaxopt.github.io/stable/implicit_diff.html
- Lineax: https://arxiv.org/abs/2311.17283
- SOLVAX: https://github.com/uwplasma/SOLVAX

## 10. Work report contract

Every implementation update reports:

1. steps taken;
2. numerical and software results;
3. tests and benchmark environment;
4. files changed and how the structure follows Section 5;
5. best next steps;
6. completion percentage for every open lane;
7. anything needed from the user.

No user input is currently required. The next executable action is WP0: fix
the benchmark symmetry definition, regenerate the concentric-coil evidence,
and then perform the WP1 reduction before adding more hybrid code.
