# Mirror equilibrium final implementation plan

Status: active and authoritative plan for draft PR #22. This file supersedes
`/Users/rogeriojorge/Downloads/plan_mirror.md` and every earlier roadmap in the
branch. Do not add another plan file. Commits and compact benchmark JSON/CSV
files are the execution log.

Review baseline: `codex/mirror-geometry` at `15529dc9`, based on
`origin/main` at `ed4ac7acae11`, reviewed 2026-07-14 after refreshing all
remotes and the primary literature. The branch is 281 commits ahead and zero
behind `origin/main`; PR #22 is open, draft, and mergeable. It contains 67
changed files with 17,730 additions and 1,633 deletions. The grouped CI run at
`a4620a41` passed both fast shards, build, console smoke, and parity-a/d. Its
field shard failed before collection because the workflow still named the
deleted `test_vacuum.py`; commit `15529dc9` points it at
`test_free_boundary.py`. The next grouped run, rather than repeated polling,
is the confirmation gate. The unreleased virtual-casing extender remains
capability-gated and is not part of the core install contract.

## 1. Mission and finish line

Deliver a small, fast, research-grade extension of `vmec_jax` for nested-surface
ideal-MHD equilibria in:

1. straight-axis axisymmetric fixed-boundary mirrors;
2. straight-axis nonaxisymmetric fixed-boundary mirrors;
3. straight-axis axisymmetric free-boundary mirrors;
4. a bounded research decision on straight-axis nonaxisymmetric free boundary;
5. a closed toroidal stellarator-mirror hybrid with two long straight mirror
   legs and two curved stellarator returns.

The scalar-pressure fixed and axisymmetric-free lanes, the nonaxisymmetric fixed
lane, and the fixed closed hybrid are required deliverables. Nonaxisymmetric
free boundary, closed-hybrid free boundary, and ANIMEC are promoted only if they
pass the bounded attempts below. A failed conditional lane is explicitly
deferred, removed from the top-level API, and documented with one compact
negative benchmark. It does not keep the PR open indefinitely.

A supported result is solved from an MHD residual. A prescribed tube sampled in
an external field, a Fourier projection of a square target, or a surface with
small `B.n` but no plasma-force solve is not an equilibrium result.

### 1.1 Required numerical contract

Every promoted lane must have:

- a component-wise discrete equilibrium residual no larger than `1e-12`, or a
  documented double-precision floor no larger than `1e-11` after a resolution
  study;
- an independently assembled staggered weak-force check consistent with the
  discrete first variation;
- stable physical observables on at least three resolutions, with the two
  finest levels satisfying lane-specific tolerances;
- positive, nested geometry and normalized `div(B)` near roundoff;
- `B.n` and total-pressure balance for free boundaries;
- analytic, manufactured, or independent-code validation;
- implicit JVP/VJP checks for every advertised differentiable input;
- one parser-free example, compact benchmark data, and current documentation.

SciPy success, a small step, a small energy change, or a visually smooth surface
is never a convergence criterion.

### 1.2 Decisions relative to the original plan

The attached 2,000-line CGL roadmap was useful for establishing equations and
tests, but its proposed package tree and feature list conflict with the current
simplicity goal. This plan preserves the physics while making these explicit
scope decisions:

- Chebyshev-Lobatto remains the nodal oracle and supported axisymmetric free
  path; clamped/periodic cubic B-splines are the coefficient-native fixed and
  hybrid paths needed for exact straight spans.
- A separate mirror-Boozer package and `mbmn` file are deferred. The release
  computes iota from equilibrium-coordinate field lines and Fourier diagnostics
  of `|B|` directly from MOUT. It never sends an open mirror through toroidal
  `booz_xform_jax`.
- Mirror-specific optimizer classes are deferred. Promoted implicit JVP/VJP
  functions expose the gradients needed by external optimizers without adding
  another objective framework; the release includes an FD-verified gradient
  example, not a second optimization stack.
- TOML/YAML mirror input, VTK export, and a large nested mirror package tree are
  deferred. Parser-free Python examples, dataclass inputs, mirror-native MOUT,
  and one plotting entry point are the supported interface.
- Axisymmetric free boundary moved into required scope because it now has a
  real coupled finite-beta solve. Native spline free boundary and
  nonaxisymmetric free boundary moved back out after their bounded numerical
  gates failed.

## 2. Physical model and topology

### 2.1 Open straight mirrors

Open mirrors use coordinates

`(s, theta, xi) in [0,1] x [0,2*pi) x [-1,1]`,

with a straight Cartesian axis, periodic poloidal angle, and nonperiodic axial
coordinate. The nested surfaces are open flux tubes. The lateral surface
`s=1` is the LCFS. The planes `xi=-1` and `xi=+1` are fixed computational cuts
through which magnetic flux passes; they are not material plasma-vacuum
interfaces and do not impose `B.n=0`.

For the unbounded exterior Green problem, disks close those cuts geometrically.
Their Neumann data must match the plasma and applied fields crossing the cuts.
The disks do not turn the mirror into a closed plasma. This distinction must be
present in equations, docstrings, diagnostics, and free-boundary tests.

The divergence-free field is retained in contravariant form,

`sqrt(g) B^theta = I'(s) - d(lambda)/dxi`,

`sqrt(g) B^xi = Psi'(s) + d(lambda)/dtheta`,

with `B^s=0`, a weighted zero-mean gauge for `lambda`, and fixed axial flux.
Axisymmetry is `mpol=0`; it is not a separate equilibrium class.

Fixed boundary prescribes the lateral radius and both cut sections. Free
boundary varies the lateral LCFS interior in `xi`, while the two cut sections
remain fixed. The exterior solve enforces tangency on the lateral LCFS and
consistent through-flux on the artificial caps, then the plasma solve enforces
total-pressure continuity on the lateral LCFS.

### 2.2 Closed stellarator-mirror hybrid

The hybrid remains toroidal. A periodic Cartesian centerline contains two long
straight legs joined by two smooth curved returns. The coordinate along the
centerline is periodic and has no end caps. A rotation-minimizing/Bishop frame
is used because the Frenet frame is undefined on the zero-curvature legs.
Periodic frame holonomy is corrected explicitly and tested.

The production reference family enforces up-down symmetry about its midplane
and exchange symmetry between the two mirror legs through constraints on spline
control points and section coefficients. The periodic spline basis itself is
not symmetry-specific, so asymmetric verification cases remain possible without
adding another geometry representation.

"Full B-spline support" has a precise scope:

- open axial dependence of boundary, radius, and stream function uses clamped
  cubic B-spline coefficients;
- closed centerline, section size, section orientation, radius, and stream
  function use periodic cubic B-spline coefficients;
- the radial mesh remains VMEC-like and staggered;
- the periodic poloidal direction remains Fourier.

Replacing the radial or poloidal discretization with splines is out of scope:
it adds machinery without solving the long-straight-section representation
problem.

### 2.3 Scalar and anisotropic pressure

Scalar pressure is the release-blocking model. The energy and field kernels are
shared by all scalar lanes.

ANIMEC is a conditional extension. Its pressure is distribution-derived
`p_parallel(s,B)` with

`p_perp = p_parallel - B * (d p_parallel / dB)_s`.

It is not two independently prescribed pressure fields. The plasma-vacuum
stress uses `p_perp + B^2/(2*mu0)`. Firehose and mirror-ellipticity indicators
must remain positive for a promoted result.

## 3. Repository ownership and API decisions

### 3.1 Ownership

- **vmec_jax** owns equilibrium coordinates, MHD energies and residuals,
  boundary coupling, continuation, implicit equilibrium sensitivities, MOUT,
  and plots of solved states. It may trace its own contravariant equilibrium
  field in flux coordinates to measure iota and make equilibrium plots.
- **ESSOS** owns coils, Biot-Savart, lab-frame coil-field-line tracing, and
  mgrid creation. vmec_jax accepts `MgridField` or a vectorized `xyz -> B`
  callable; it does not construct coils.
- **SOLVAX** owns reusable Krylov/direct solvers, generic preconditioners,
  chunked AD, and implicit linear/root-solve machinery.
- **virtual-casing-jax** owns generic singular Laplace/virtual-casing kernels.
  Mirror code owns only the open-surface geometry and boundary data needed to
  call those kernels.
- **SciPy** may control fast nondifferentiable CLI nonlinear solves. There is no
  requirement to trace or differentiate the host iteration history.

No coil model, Biot-Savart implementation, lab-frame coil-field integrator,
general BIE library, general finite-element package, or duplicate GMRES
implementation is added to vmec_jax.

### 3.2 Solver and derivative policy

The converged residual is `F(u,p)=0`. Derivatives use

`F_u du = -F_p dp` and `F_u^T lambda = objective_u`.

- use forward implicit JVPs for a few parameter directions;
- use reverse implicit adjoints for scalar objectives with many controls;
- never reverse-differentiate thousands of nonlinear iterates;
- use centered finite differences only as a validation oracle;
- report primal and linearized residuals with every derivative result.

JAX `custom_linear_solve`, JAXopt, Lineax, and the spectral-adjoint literature
all support this high-level implicit approach. SOLVAX remains the only added
solver dependency because vmec_jax already uses it. Only released SOLVAX APIs
may be required. Tag `v0.8.3` provides GMRES, GCROT, PCG, block Thomas,
ordinary and periodic banded LU, operators, chunked Jacobians, implicit solves,
and Newton-Krylov; SOLVAX `main` at `255d280b` is preparing 0.8.4. The clean
0.8.3 parity attempt described in Phase 3 failed the production residual and
runtime gates, so this PR keeps `solvax>=0.2.0`, SciPy host GMRES, and only the
already-tested SOLVAX block-Thomas API. JAXopt, Optax, Lineax, Equinox, and
Optimistix are not added: none removes the dominant need for a physics-aware
periodic preconditioner, and another solver abstraction would increase the
branch before numerical parity exists.

### 3.3 Public API target

The public mirror API contains only:

- configuration, boundary/state/profile contracts;
- fixed/free solve and beta-continuation workflows;
- implicit JVP/VJP workflows for promoted lanes;
- MOUT read/write and one plot entry point.

Analytic fixtures, quadrature, BIE panels, vectorizers, preconditioners, and
reconstruction kernels remain internal. Experimental lanes are imported from
their owning module and are not flattened into `vmec_jax.mirror`.

## 4. Current evidence and defects

### 4.1 Branch footprint

The initial audit relative to `origin/main` found 137 changed files, 24,370
added lines, and 4,255 deleted lines. Phase 1 and the current cleanup reduce the
working diff to 67 files, 17,941 added lines, and 1,632 deleted lines: 70
unrelated files and about 6,600 added lines are gone. `vmec_jax/mirror` now
contains 8,977 lines in 15 modules and exposes 24 lazy names. Continuation lives
with the free-boundary workflow, restart/plot/diagnostic output lives in one
output module, and exterior interpolation lives with the BIE solve. The largest
files remain `forces.py` (1,098), `splines.py` (1,063), `solver.py` (1,001), and
`output.py` (912), so the module-count gate is met but the line and oversized-
file gates are not. There are 134 collected mirror tests; the removed tests
exercised only deleted finite-cylinder, generic interior-Laplace, or full-node
virtual-casing paths.

The earlier QI, direct-coil, optimization, and core-refactor work is restored to
`origin/main`. Only the mirror package, mirror evidence, and small CLI, device,
packaging, and documentation integration hooks remain. Local ruff, strict
Sphinx, and both branch-wide whitespace checks pass.

### 4.2 Results that are credible

- Axisymmetric fixed boundary has real scalar solves at `ftol=1e-12`,
  manufactured/cylindrical checks, weak residuals, and implicit derivatives.
- Axisymmetric free boundary has a six-point beta scan through 50%, with
  variational residuals `3.6e-15` to `7.0e-15`, weak residuals `7.1e-16` to
  `1.4e-15`, normalized `div(B)` about `1.3e-15`, normal stress below
  `3.4e-15`, and normalized `B.n` below `2e-16`.
- At beta 50%, center radius rises 7.6% and center field falls 24.9%. The solved
  field ratio 0.75095 is within 6.2% of the paraxial `sqrt(1-beta)` value.
- Nonaxisymmetric fixed boundary solves the rotating-ellipse and SFLM fixtures
  in native open B-spline coefficients. The forbidden `m=1` signal is near
  roundoff with even theta quadrature, and field direction approaches the SFLM
  solution as tube radius decreases.
- Reverse implicit derivatives of fixed spline equilibria agree with
  reconverged finite differences near `3e-10` relative in tested directions.
- Complete nonaxisymmetric free-boundary solves pack radius and gauge-free
  `lambda` and reach roundoff global residuals through beta 50% on two grids.
  Global observables change by at most 0.96%.
- Periodic B-spline centerline, Bishop frame, closed embedding, and geometry
  derivatives pass circle/racetrack closure, volume, metric, and `div(B)` tests.
- A finite-current rotating-ellipse racetrack solve reaches variational
  `3.11e-15` and normalized `div(B)=3.50e-13` with solved `lambda`.
- Its periodic coordinate-space field-line trace is finite over three circuits
  and has nonzero iota. The circular fixture recovers `iota=I'/Psi'` and
  `d(iota)/dI'=1/Psi'` to `2e-13` relative.
- The complete circular-torus solve now advances radius and `lambda` from a
  tested `1/R` initializer. At `ns=5` it reaches variational/independent weak
  residuals `1.88e-15/1.83e-15` in 27 residual-Newton evaluations; at `ns=7`
  they remain below `3.0e-15`.

### 4.3 Results that are not promotion evidence

- The complete circular torus has a pointwise reconstructed force of `0.709`
  at `ns=5`, improving to `0.570` at `ns=7`. The variational and independently
  assembled weak residuals are closed. Consistent with Section 4.4, the current
  pointwise reconstruction is not a promotion gate; it must be labeled as a
  nonconverged diagnostic and excluded from scientific claims until a
  manufactured half-to-full reconstruction refines.
- Closed periodic preconditioning is disabled.
- The open pointwise `J x B - grad(p)` reconstruction does not converge
  monotonically even where the discrete and weak residuals are at roundoff. It
  remains a non-gating diagnostic until a VMEC-like half-to-full reconstruction
  passes manufactured refinement.
- Nonaxisymmetric free-boundary local `m=1` observables change 73--81% between
  the two available grids, and the medium pair costs 800.7 seconds and 4.25 GiB
  RSS. It is not supported before structured linear algebra changes this bound.
- The finite-beta rotating-ellipse `m=2` amplitude is still about 48% above the
  direct paraxial estimate at the finest bounded knot level.
- ANIMEC has useful closures and tests but lacks equation-by-equation source
  parity and an independent finite-beta benchmark.
- No hybrid example, MOUT contract, iota/pitch refinement, VMEC2000 limit, or
  free-boundary hybrid result is promotion-ready.

### 4.4 Residual contract

Every solve reports separate quantities:

1. **Discrete variational residual**: the normalized energy gradient on active
   degrees of freedom. This defines `ftol`.
2. **Staggered weak-force residual**: an independently assembled first
   variation projected onto the same admissible basis and quadrature.
3. **Pointwise reconstructed force**: a documented spatial reconstruction of
   `J x B - grad(p)`. It is diagnostic until manufactured refinement passes.
4. **Constraint diagnostics**: `div(B)`, Jacobian sign, nestedness, self
   clearance, `B.n`, cap compatibility, and pressure jump.

The weak residual may share formulas and quadrature with the energy, but it may
not call `jax.grad` on the same scalar objective.

### 4.5 Current source and test map

| Ownership | Source | Primary verification |
|---|---|---|
| inputs, states, pressure closures | `model.py` | `test_model_basis.py`, `test_pressure_closures.py` |
| Chebyshev/Fourier grids | `basis.py` | `test_model_basis.py` |
| clamped/periodic B-splines and closed solve | `splines.py` | `test_splines.py` |
| open/closed geometry and fields | `geometry.py` | `test_geometry_fields.py`, `test_splines.py` |
| energy, weak and pointwise residuals | `forces.py` | `test_isotropic_forces.py`, `test_pressure_closures.py` |
| fixed host solve and open preconditioner | `solver.py` | `test_fixed_boundary_3d.py`, `test_promotion_guards.py` |
| free solve and beta continuation | `free_boundary.py` | `test_free_boundary.py`, `test_continuation.py` |
| cap/side exterior geometry and Green solve | `exterior*.py` | `test_exterior.py`, `test_free_boundary.py` |
| fixed/free implicit adjoints | `*implicit.py` | `test_*implicit.py` |
| MOUT, restart, diagnostics, plots | `output.py` | `test_output.py`, example smoke |
| independent analytic fixtures | `analytic.py` | `test_analytic.py` |

The audit identifies four code defects that determine the remaining order:

1. closed spline solves deliberately pass no matrix-free context and therefore
   use dense residual Newton;
2. `spline_fixed_boundary_adjoint` reconstructs `mirror_energy` without the
   closed periodic axis and cannot advertise hybrid derivatives;
3. axisymmetric free-boundary optimizer convergence is tighter than its current
   independent exterior-observable convergence;
4. generic BIE helpers, duplicated host callbacks, and research ANIMEC paths
   keep four modules above 900 lines and the package above 8,500 lines.

## 5. External code and literature conclusions

### 5.1 VMEC2000 and VMEC++

STELLOPT `origin/develop` at `205b7fd7` and the local VMEC2000 sources were
inspected directly. Retain VMEC's variational principle, divergence-free
representation, radial
half mesh, full/half-mesh parity handling, component residual normalization,
continuation, and radial block preconditioning. The relevant VMEC2000 source is
`bcovar.f`, `forces.f`, `residue.f90`, `fbal.f`, and `precon2d`/
`blocktridiagonalsolver`.

VMEC2000's `bcovar` forms metrics, contravariant fields, magnetic pressure, and
kinetic pressure on the radial half mesh. `forces` explicitly averages half
mesh terms before forming full-mesh force kernels. `residue` separates raw
`fsqr/fsqz/fsql` from preconditioned residuals and applies `m=1` constraints.
These are the direct references for repairing the pointwise mirror diagnostic
and designing the preconditioner.

VMEC2000 cannot represent an open axial topology. It is used only for sign,
normalization, and the smooth closed-torus limit. VMEC++ is a modern parity and
software-architecture reference, not a mirror oracle.

### 5.2 DESC and its research branches

DESC `master` at `2471d550`, `mirror` at `0dba071d`, `mirror_anisotropy` at
`805b77fc`, `finite_element_basis`, `finite_element_basis_alan`, and
`dd/cylindrical` at `6f85f50a` were inspected locally after fetching the
2026-07-14 remote refs.

Useful ideas:

- a nonperiodic basis owns nodes, quadrature, differentiation, interpolation,
  endpoint semantics, and coefficient transfer;
- Chebyshev/Fourier products cleanly separate an open coordinate from a
  periodic coordinate;
- continuation and objective scaling are explicit;
- DESC's current free-boundary work treats `B.n` and magnetic-pressure jump as
  separate area-weighted residuals.

Do not port these branches. The `mirror` diff changes 123 files and exceeds one
million added lines because it includes generated binary/notebook data. Its
two nominal mirror test files contain only six import lines and zero lines,
respectively. Its end-cap objective module contains repeated no-op helpers and
raises `NotImplementedError` for explicit mode selection. The anisotropy branch
itself labels some force quantities suspicious or unchecked. These branches
are valuable for basis formulas and failure modes, not reusable production
code or independent validation.

The current `dd/cylindrical` head adds `DoubleChebyshevFourierBasis`, a tensor
product of two Chebyshev coordinates and one Fourier coordinate. The mirror
content is two basis/test commits: construction, resolution change, validation,
and hashability. It has no equilibrium or boundary-condition test, while the
branch also performs a repository-wide grid refactor. It does not provide
B-spline straight spans, hybrid centerlines, end-cap physics, or a validated
free-boundary model. DESC master does contain mature radial Chebyshev-Fourier
machinery and high-order toroidal free-boundary residuals. The 2024
free-boundary formulation separately minimizes `B.n` and total-pressure jump,
uses virtual casing plus a surface-current potential, and validates with direct
field-line tracing and VMEC. That is the correct contract for a future closed
hybrid free boundary, but its smooth toroidal singular quadrature does not cure
the open mirror's cap-edge trace error.

### 5.3 Mirror analytic validation

The paraxial/near-axis mirror expansion gives required low-radius checks:

- `X1c*Y1s - X1s*Y1c = Bbar/B0(z)` for flux conservation;
- no order-`r`, `m=1` variation of `|B|`;
- expected order-`r^2`, `m=2` quadrupole variation;
- ellipse magnitude/orientation governed by the sigma/Riccati equation.

The Straight Field Line Mirror is an independent Clebsch-field fixture with
analytic elliptical flux tubes and straight but nonparallel field lines. It
validates flux labels, field direction, ellipticity, and finite-beta trends; it
does not by itself require a 90-degree ellipse rotation.

The two-equal-circular-loop fixture independently checks the exact on-axis
Biot-Savart field and low-radius `B_r/B_z` expansion. Pleiades commit
`0161abb3` supplies a separate axisymmetric finite-beta calculation: its
three-grid scan at beta 1%, 3%, and 10% converges the on-axis field ratios to
`0.99537`, `0.98605`, and `0.95275`. Pleiades uses an axisymmetric
Green-function/diamagnetic-current formulation, so it validates the sign,
magnitude, and refinement trend but not vmec_jax's variational algorithm or 3-D
lanes. The 2025 RealTwin/Pleiades report independently observes outward flux-
surface expansion and `B0 ~= Bvac*sqrt(1-beta)` and checks the same firehose and
mirror ellipticity conditions used here.

VEPEC is the closest historical 3-D open-mirror equilibrium code. It uses a
vector potential and tricubic splines so `div(B)` is controlled. Its published
minimum-B and long-thin comparisons support spline and paraxial validation, but
its implementation is not a dependency.

Goodman-Freidberg-Lane and Pearlstein provide finite-beta quadrupole and diamond
distortion trends. These are asymptotic gates, not exact full-radius targets.

### 5.4 Linked mirrors and the closed hybrid

Pastukhov/Ilgisonis/Berk analyze finite-beta quadrupole mirrors linked by
elliptical toroidal cells and predict nonlinear outward displacement with beta.
Feng et al. propose two straight mirrors joined by two half tori and obtain
rotational transform from nonparallel straight sections. Ranjan's helically
linked mirror uses curved helical returns to combine straight mirror sections,
transform, and magnetic well.

These sources validate the topology and expected observables, not a unique
boundary. The implementation therefore uses three independent gates:

1. circular-axis limit against normal vmec_jax and VMEC2000;
2. long-straight-leg limit against the open spline mirror;
3. nonzero pitch/iota from current or geometric return twist, with nested field
   lines and stable beta response.

Periodic cubic B-splines are appropriate because of local support and exact
straight spans. A Bishop frame is required at zero curvature. Fourier fitting
of the entire centerline is explicitly rejected as the production
representation.

### 5.5 ANIMEC

The 1992 variational paper, 2006 bi-Maxwellian model, 2009 free-boundary paper,
and VMEC2000 `_ANIMEC` source agree on the following contracts:

- minimize `B^2/(2*mu0) + p_parallel/(Gamma-1)`;
- derive `p_perp` consistently from `p_parallel(s,B)`;
- modify magnetic force kernels through `sigma`;
- use `p_perp` in the interface total pressure;
- report firehose (`sigma`) and mirror (`tau`) validity measures;
- recover scalar VMEC exactly in the isotropic/hot-fraction-zero limit.

The source is more specific than the papers alone. `fbal.f` defines the
piecewise bi-Maxwellian `H(s,B)` above and below `B_crit`; `bcovar.f` adds
`p_perp` to magnetic pressure and replaces the effective current by
`K=curl(sigma B)`; `jxbforce.f` uses
`sigma = 1 + (p_perp-p_parallel)/(2*bsq)` in VMEC's pressure normalization.
The current mirror ANIMEC lane may be retained only if those exact branches,
normalizations, sigma/tau limits, and scalar limit pass one bounded source
parity implementation and one independent finite-beta case.

### 5.6 Differentiation and linear-solver conclusion

The numerical-method review produces one implementation policy rather than a
menu of interchangeable frameworks:

- the CLI solve remains a nondifferentiable SciPy host iteration because it is
  faster to develop, reports failure clearly, and need not retain an iteration
  tape;
- differentiable APIs apply the implicit-function theorem to the converged,
  gauge-free discrete residual. Forward JVPs solve `F_u du=-F_p dp`; reverse
  VJPs solve `F_u^T lambda=Q_u` and form `Q_p-lambda^T F_p`;
- exact JAX JVP/VJP actions define both operators. Centered, reconverged finite
  differences validate them but are never the production derivative;
- the automated spectral-adjoint result supports transposing a sparse
  discretized operator graph and reusing its structure. It does not justify
  differentiating nonlinear iterations or adding a symbolic PDE layer here;
- `jax.lax.custom_linear_solve` is appropriate only when both primal and
  transpose solves are traceable and certify their residuals. The current
  SciPy host GMRES wrappers therefore remain explicit custom VJPs for this PR;
- forward mode is the default for one to roughly four controls; reverse mode is
  the default for one scalar objective and many boundary/profile controls;
- dense Jacobians are verification or small-problem fallbacks only. Production
  medium cases use matrix-free actions and a structure-aware preconditioner.

This policy rules out unrolled reverse AD, finite-difference production
gradients, and adding JAXopt/Optax/Lineax/Equinox/Optimistix without a measured
residual, runtime, or memory improvement on an existing promotion case.

## 6. Ordered implementation phases

Each phase ends with focused tests, a compact benchmark update, one or more
small commits, and a push. CI is checked after grouped work rather than polled
after every commit. No new physics lane is added.

### Phase 0: correct the baseline and CI

1. Confirm the `test_vacuum.py -> test_free_boundary.py` field-shard repair in
   the next grouped CI run and audit any completed failure once, after other
   work has accumulated.
2. Replace the frozen-lambda torus test with a complete `solve_lambda=True`
   solve. Add a vacuum `1/R` stream-function initializer and verify its sign,
   gauge, and radial convergence.
3. Remove every 23-iteration solved-torus claim from docs and benchmarks.
4. Add the explicit open-end topology and cap-Neumann contract to equations and
   docstrings.
5. Fix `git diff --check`; rerun ruff, strict Sphinx, example smoke, and focused
   closed/open tests.

Gate: the baseline is scientifically honest and locally green. No later hybrid
claim uses an incomplete state.

Execution status (2026-07-14): items 2--4 are implemented. The former
radius-only torus test is now a complete 27-evaluation solve with an independent
closed weak residual, the stale documentation claim is removed, and the cap
through-flux contract is explicit. The grouped run confirms both repaired fast
shards and every build, example, mirror, implicit, and parity job. The aggregate
core-only coverage gate is `94%` versus `95%` because inherited branch-only core
modules add untested statements; do not lower the threshold. Phase 1 must
restore that unrelated core diff before Phase 0 can be declared globally green.
The Phase 1 restoration subsequently put that core implementation back at
`origin/main`. The next grouped run passed every build, fast, mirror, implicit,
and parity job but exposed the virtual-casing example capability check described
in the review baseline. Its focused fix passes in both dependency states: four
extender tests skip in the core environment, while the editable extender
environment reports 6 passed and 3 skipped in 426.68 seconds. The branch-wide
whitespace check and all focused local tests pass. A later field shard named
the deleted finite-annulus test and exited before collection; `15529dc9`
repairs the path. Only grouped confirmation of that integration fix remains
for closing Phase 0 globally.

### Phase 1: reduce the PR before adding physics

1. Classify all 137 changed files as mirror-required, current-main integration,
   or inherited unrelated work.
2. Restore unrelated QI, direct-coil, optimization, and core files to
   `origin/main` in ordinary commits. Keep only integration changes required by
   mirror CLI, plotting, packaging, and shared solver contracts.
3. Reduce the public mirror namespace from 47 names to at most 24.
4. Choose one production exterior formulation: unbounded panel/Green solve.
   Delete the annulus after parity; do not expose two vacuum models.
   Remove unused spectral/curved-panel variants that do not improve the bounded
   convergence gate.
5. Consolidate modules only where ownership is artificial. Target at most 15
   mirror modules, at most 8,500 mirror source lines, and no file above 900
   lines without a written reason.
6. Delete stale benchmarks, generated outputs, duplicate examples, and docs for
   removed paths.

Execution status (2026-07-14): items 1 and 2 are complete in the first
restoration tranche. The remaining 67-file diff contains only mirror-owned
source, tests, examples, evidence, documentation, and narrow shared integration
hooks. The public-API target is complete. The failed curved-side/high-order-cap
exterior option and its module are deleted after its bounded endpoint run did
not complete in 690 seconds; the retained spectral-side, linear-panel path
passes its exterior and shape-derivative tests. Continuation, plotting, exterior
interpolation, and scalar diagnostics are now colocated with their owning
workflows. The finite-cylinder annulus, its grid/potential/restart contracts,
and its model-specific tests are deleted after the free-space isotropic,
anisotropic, tabulated-pressure, and restart tests passed. This reduces the
package to 15 modules. The first bounded reduction removed generic full-node
virtual-casing adapters, the unused interior Neumann solve, and duplicate
axisymmetric/nonaxisymmetric result records. The retained reduced exterior
operator passes its analytic dipole, singular identity, spectral-density,
axisymmetric/nonaxisymmetric coupling, and shape-JVP tests. This removes 252
production lines and left 8,781 lines before the next feature. Explicit
raw/corrected Neumann diagnostics and the shared forward/reverse implicit
linear solve subsequently bring the current count to 8,977. The source-line
and oversized-file gates remain active. The bounded
reduction order is: share open/closed
staggered magnetic assembly, share primal/adjoint packing and linear-solve
diagnostics, then simplify output serialization. Do not split a large file
merely to satisfy the file-size target. If ANIMEC is deferred in Phase 6,
delete its public solve and research-only diagnostics rather than carrying an
unsupported second force model.

A 2026-07-14 full three-resolution beta gate reached the requested nonlinear
`ftol <= 1e-12` at every point but did not meet its independent discretization
thresholds. For the unbounded exterior solve, the medium-to-fine beta-10% center
field changed by `8.38e-4` against a `5e-4` threshold. The now-deleted annulus
oracle changed by `6.12e-4` against `1e-4`. These are unresolved spatial-
convergence failures, not optimizer failures. Do not loosen the retained test:
refine the exterior panel, radial, and axial studies until physical observables
converge.

Gate: the diff is materially smaller, all retained benchmark claims reproduce,
and no physics result depends on an unrelated branch-only core refactor.

### Phase 2: finish promoted open scalar mirrors

1. Keep native open B-splines for fixed boundary and the retained nodal
   Chebyshev path for free boundary. Compare fixed B-spline and Chebyshev states
   across three radial/axial grids; do not reopen spline free boundary.
2. Re-run the axisymmetric nodal free solve over beta
   `[0,.01,.03,.10,.25,.50]`, refining plasma radial/axial nodes, side/cap
   panels, and singular order independently. Record achieved beta, radius,
   on-axis field, mirror ratio, `B.n`, stress, weak residual, runtime, and RSS.
3. Finish nonaxisymmetric fixed validation with the 90-degree rotating ellipse
   and SFLM. Refine `ns`, theta quadrature, `mpol`, and knots independently.
4. Fit `|B|` by radial order and poloidal mode. Require vanishing order-`r`,
   `m=1`, correct order-`r^2`, `m=2` phase, and a documented amplitude error
   within the paraxial validity range.
5. Verify pressure-first and shape-first continuation converge to the same
   state, and reject crossed surfaces during every trial.
6. Keep the pointwise force non-gating unless the VMEC-like staggered
   reconstruction passes manufactured refinement.

Gates:

- axisymmetric free: less than 0.1% center-radius and 0.05% center-field change
  between the two finest bounded grids at every nonzero beta;
- nonaxisymmetric fixed: field direction, ellipse angle, flux determinant,
  quadrupole phase, energy, volume, pitch, and weak residual converge;
- all promoted residual and geometry contracts in Section 1 pass.

Execution status (2026-07-14): the required open fixed-boundary representation
parity gate is complete. Independent finite-pressure/current solves at
`(ns,nxi)=(5,9),(7,13),(9,17)` use 7, 9, and 11 spline coefficients. Energy,
volume, sampled radius, Cartesian-field, and `|B|` errors all decrease; finest
errors are `5.60e-8`, `1.04e-6`, `2.94e-5`, `7.02e-4`, and `3.05e-4`,
respectively. Both representations remain below `1.3e-15` in variational and
independent weak residuals. The full test evaluates spline coefficients on the
nodal grid and gates physical fields rather than a relative error in the
near-zero gauge stream function.

The first six-beta exterior rerun exposed a hidden acceptance failure despite
`6e-15` force residual: raw cap-flux quadrature defects were `3.03e-5` and
`2.18e-6` on the first two grids. The exterior solve now applies the standard
Neumann compatibility projection only on artificial cap data, preserving all
lateral LCFS values. It reports corrected compatibility separately from
`raw_compatibility_error`; the former gates solvability and the latter must
decrease and reach `1e-6` on the finest promotion grid. Manufactured dipole,
axisymmetric/nonaxisymmetric adapter, and cap-only projection tests pass. The
exact six-beta GPU rerun is the next active gate.

Execution note (2026-07-14): the first native free-boundary spline attempt
reused the fixed coefficient vectorizer and the existing exterior solve. Both
a weighted Galerkin projection of total-pressure jump and Greville-point
collocation drove their coefficient residuals and the independent plasma weak
residual below `2.3e-15`. They failed the independent interface gate: with
spectral side density, pointwise stress RMS changed `1.38e-7 -> 2.79e-5 ->
1.25e-4` for 2, 4, and 8 axial elements. That adapter was deleted rather than
retained as a scaffold. Do not retry projected stress. The second and final
bounded attempt has four ordered gates before any coupled solve is added:

1. derive the discrete exterior magnetic energy for the solved Neumann Green
   problem, including correction, applied-field cross term, moving-domain
   background term, and cap through-flux signs;
2. on manufactured sphere/cylinder-like closed surfaces, compare that boundary
   energy with independent finite-volume energy and verify its shape derivative
   against centered finite differences;
3. compare the same discrete shape derivative with integrated Maxwell virtual
   work under normal boundary perturbations and require refinement in panel and
   singular-quadrature order independently;
4. only after gates 1--3 pass, pull the verified shape gradient back through
   open B-spline boundary coefficients and couple it to the plasma first
   variation.

If the energy, finite-volume, and Maxwell routes do not agree and refine, the
coefficient-native free boundary is deferred and the promoted nodal free solve
remains the public path. No third discretization strategy is permitted.

Final decision (2026-07-14): defer coefficient-native open-spline free
boundary. The second attempt derived and tested the exterior correction,
fixed-source, and excluded-volume terms. A decaying dipole agreed with an
independent unbounded cylindrical volume integral within 5%, and AD agreed
with centered shape differences to `2e-6` relative. The independent Maxwell
virtual-work comparison did not reach a promotion-quality limit: its mismatch
decreased only from 4.30% to 2.83% over 8--32 angular panels, from 3.71% to
3.55% over singular orders 4--12, and from 3.00% to 2.53% over axial grids
9--17. The cap-edge trace is therefore still the accuracy limiter. The unused
energy adapter and tests were removed, leaving the converged nodal free solve
as the sole supported path. This closes the two-attempt stop rule; do not add a
third projected-stress or energy scaffold in PR #22.

### Phase 3: structured solver and preconditioner

1. Share one small host optimization/history helper between nodal and spline
   fixed-boundary solves while retaining topology-specific vectorizers.
2. Preserve radial blocks, Fourier-poloidal modes, and local spline
   mass/stiffness matrices. Do not materialize a dense axial derivative in the
   preconditioner.
3. Implement the missing gauge-free periodic spline preconditioner. Its axial
   block must use every periodic coefficient and a cyclic stiffness operator;
   it must not reuse the open `1:-1` endpoint slice.
4. Keep SciPy host GMRES and the released SOLVAX block-Thomas calls already in
   use. Do not raise the SOLVAX floor or retry SOLVAX GMRES in PR #22.
5. Compare no preconditioner, current open separable model, and periodic model
   across `ns`, `mpol`, and coefficient count. Record compile time, warm time,
   peak RSS/device memory, nonlinear evaluations, Krylov iterations, primal
   residual, and linear residual on CPU and one office GPU.
6. Accept the periodic model only if the solved state is unchanged within the
   discretization tolerance and Krylov growth is bounded or slowly growing.

Gate: medium open and closed cases avoid dense Jacobian materialization,
Krylov growth is bounded or slowly growing, converged states are unchanged,
and the closed racetrack no longer depends on the dense residual-Newton path.

SOLVAX decision (2026-07-14): do not replace the host Krylov path in PR #22.
A clean editable v0.8.3 checkout passed exact tensor-preconditioner parity, and
the existing preconditioner was made traceable for the trial. On the established
large spline gate, however, right-preconditioned SOLVAX GMRES returned relative
linear residual `3.48e-5` against the existing `<1e-5` contract, reported a
breakdown, and forced the residual-Newton fallback; the second rotating-ellipse
case exceeded five minutes and was terminated. The trial code and dependency
floor bump were reverted. Retain SciPy's tested host GMRES and SOLVAX's existing
block-Thomas use. A future SOLVAX integration requires explicit left/right
preconditioner parity and a faster full-solve result; it does not block this PR.

Periodic decision (2026-07-14): the all-coefficient cyclic Galerkin block is
implemented and passes gauge-free packing/linearity plus the existing circular
torus and finite-current racetrack tests. It does not pass the primal scaling
gate. On the identical 892-variable racetrack, forced GMRES is faster than the
dense reference (`5.81 s` versus `8.71 s`) and preserves energy to `3.3e-16`
and radius to `9.5e-11`, but takes 3,000 Krylov iterations and leaves linear
residual `0.136`. Bounded CG and MINRES variants still leave `1.67e-2` and
`1.58e-2` after 2,000 and 1,852 iterations. The likely missing term is the
geometry--stream coupling, not the periodic axial stiffness. Do not enable the
matrix-free closed primal path in this PR. Retain the periodic block for the
bounded closed-adjoint gate and keep promoted closed studies below the 1,024
variable dense limit.

### Phase 4: deferred nonaxisymmetric free-boundary record

The attempted route was:

1. Use the coefficient-native spline plasma state and the single retained
   unbounded exterior formulation.
2. Continue one rotating-ellipse family from beta 0 through
   `[.10,.25,.50]`; never prescribe separate beta-dependent boundaries.
3. Refine plasma variables, side panels, cap panels, singular quadrature, and
   field interpolation independently.
4. Require stable LCFS displacement, local Fourier modes, field depression,
   `B.n`, cap compatibility, total-pressure balance, weak force, nestedness,
   and pitch.
5. Compare with the fixed solution as the external field increasingly pins the
   reference boundary.

Stop rule: run at most two bounded exterior discretization strategies after the
structured solver exists. If local observables still fail or runtime/memory
remain impractical, keep one compact negative benchmark, remove the public
claim, and defer the lane.

Decision: Phase 4 is deferred because its required coefficient-native open
spline state failed the Phase-2 shape-gradient gate. The existing nodal
nonaxisymmetric result remains research evidence, not a supported free-boundary
lane. Do not substitute a Fourier or projected-stress retry.

### Phase 5: promote the fixed B-spline hybrid

1. **Geometry gate.** Finish periodic refinement, minimum self-clearance,
   positive-Jacobian, up-down symmetry, and mirror-leg exchange constraints.
   Retain the general periodic basis; impose symmetry with control maps.
2. **Circular limit.** The `1/R` initializer, complete radius/lambda solve, and
   independent closed weak residual are done. Refine radial, theta, and spline
   resolution independently, then compare a near-circular case with normal
   vmec_jax and local VMEC2000 at matched boundary, flux, pressure, and current.
   Compare volume, energy, iota, `|B|`, and normalized residuals, not iteration
   counts.
3. **Racetrack equilibrium.** The finite-current rotating-ellipse solve and
   three-circuit field-line trace are done. Add three-resolution convergence of
   iota, mirror ratio, section orientation, energy, volume, weak residual, and
   nestedness. The known pointwise-force reconstruction remains non-gating and
   must be plotted only as an explicitly unconverged diagnostic.
4. **Open-limit validation.** Increase straight-leg length at fixed minor
   radius and compare leg-center sections, field direction, `|B|` modes, and
   paraxial coefficients with the open B-spline solver. Require monotone
   approach in at least three leg lengths.
5. **Fixed-boundary beta scan.** Run beta `[0,.01,.03,.10,.25,.50]` by hot
   starting the solved state at one fixed LCFS. Measure internal-surface shift,
   field depression, iota, mirror ratio, magnetic well, and residuals. Do not
   claim LCFS displacement from a fixed-boundary solve. The nonlinear outward
   LCFS displacement predicted for linked mirrors is reserved for the later
   conditional free-boundary experiment.
6. **Hybrid derivatives.** Extend the spline residual/adjoint contract to carry
   the periodic axis. The current `spline_fixed_boundary_adjoint` calls the open
   geometry energy and is not a closed-hybrid derivative. Validate one forward
   tangent and scalar-objective adjoints for centerline controls, section
   coefficients, profiles, flux, and current against reconverged centered FD
   sweeps; require both nonlinear and transpose residuals.
7. **Release example and output.** Add one parser-free root example and closed
   MOUT/plot support for horizontal 3-D surfaces, visible equilibrium field
   lines, cross sections, `|B|`, pressure, iota/pitch, magnetic well, and
   residual/refinement histories. Keep generated MOUT and raw traces out of git;
   commit only compact evidence and compressed figures.

Gate: complete fixed hybrid, both limiting checks, `ftol<=1e-12` (or a studied
floor below `1e-11`), converged independent weak residual, stable pitch/iota,
verified implicit derivatives, and reproducible plots.

Execution status (2026-07-14): periodic cubic geometry, Bishop-frame holonomy,
the complete circular solve, the closed weak residual, and the finite-current
90-degree racetrack solve are implemented. Commit `15529dc9` adds the periodic
equilibrium-coordinate tracer. The circular iota/derivative, circular solve,
and solved-racetrack trace tests pass together (`3 passed` in 72.54 seconds);
the racetrack's mixed-derivative divergence floor is `1.70e-12` and is bounded
by a documented `2e-12` x64 roundoff threshold. Geometry constraints, periodic
preconditioning, both physical limits, beta refinement, closed-axis adjoints,
and release output remain open.

Only after this gate may a closed-hybrid free-boundary attempt use the normal
smooth-toroidal virtual-casing contract with an ESSOS field callable or MGRID.
It must solve `B.n` and total-pressure jump separately and validate against
ESSOS lab-frame field-line tracing. Apply the same two-attempt stop rule as
Phase 4. Free hybrid is conditional and cannot delay fixed-hybrid promotion.

### Phase 6: derivatives and ANIMEC decision

1. Expose implicit JVP and VJP only for lanes promoted in Phases 2 and 5; the
   deferred nonaxisymmetric free lane gets no differentiable public API.
2. Check state tangents and scalar-objective adjoints against reconverged
   centered finite differences over step-size sweeps. Report the tangent or
   adjoint linear residual and condition estimate.
3. Use `jax.lax.custom_linear_solve` or SOLVAX implicit wrappers at the
   converged residual; do not add JAXopt, Optax, Lineax, or Equinox solely to
   wrap the existing root.
4. Complete one source-parity attempt against STELLOPT `origin/develop` at
   `205b7fd7`: the piecewise `H` and radial derivative terms in `fbal.f`,
   `p_perp` and `K=curl(sigma B)` terms in `bcovar.f`, and force normalization
   in `jxbforce.f`. Test above/below `Bcrit`, `p_perp` AD/FD, isotropic and
   zero-hot-fraction limits, sigma/tau, and interface stress.
5. Complete one independent attempt: build local VMEC2000 with `_ANIMEC`, run a
   matched smooth closed-torus case and its isotropic limit, then run one open
   finite-beta resolution study against the long-thin trend. If the executable
   cannot be built reproducibly, the normalizations do not match, or either
   observable study fails, defer ANIMEC and remove its public solver path in
   this PR. Do not invent a second anisotropic closure to rescue the lane.

Gate: supported derivatives have verified primal/linear residuals and FD error;
ANIMEC has an unambiguous supported or deferred status.

### Phase 7: final simplification and release evidence

1. Recount files, source lines, modules, public names, tests, and artifacts
   against Section 4.1. Final targets are at most 60 changed files, 15 mirror
   modules, 8,500 mirror source lines, and 24 public lazy names. Every file over
   900 lines needs a specific ownership reason; no new module is added merely
   to move lines between files.
2. Give every public function/class a short purpose-first docstring with units,
   output contract, and important validity condition. Comments explain gauges,
   staggering, cap data, and singular quadrature, not syntax.
3. README: one capability table and four reproducible showcases: normal
   toroidal VMEC, axisymmetric free mirror, rotating-ellipse fixed mirror, and
   native B-spline fixed hybrid.
4. Docs: equations, topology/boundary conditions, basis and staggering,
   residual meanings, paraxial/SFLM validation, spline hybrid geometry,
   derivatives, ownership, examples, and known limits.
5. Commit only compact JSON/CSV and compressed showcase figures, normally below
   300 KiB each. Do not commit MOUT/WOUT/mgrid/field-line output.
6. Run ruff, `git diff --check`, strict Sphinx, all mirror tests, example smoke,
   full CI, and one SSH-office GPU benchmark. Compare CPU/GPU compile, warm
   runtime, peak memory, iterations, and gradients.
7. Keep PR #22 draft until every required gate passes and every conditional
   lane is either promoted or explicitly deferred.

## 7. Promotion matrix and current completion

Percentages measure accepted promotion evidence, not code written.

| Lane | Current | Required remaining evidence |
|---|---:|---|
| Axisymmetric fixed mirror | 90% | spline derivative/release evidence |
| Axisymmetric free mirror | 84% | exterior observable refinement and scaling |
| Nonaxisymmetric fixed mirror | 86% | amplitude and preconditioned refinement |
| Nonaxisymmetric free mirror | 55% | deferred after failed local-mode and spline-shape gates |
| Open native B-splines | 82% | fixed-boundary release evidence; free boundary deferred |
| Fixed closed B-spline hybrid | 58% | limiting cases, beta refinement, derivatives, release evidence |
| Free closed hybrid | 10% | conditional after fixed promotion |
| Preconditioning | 55% | geometry--stream coupled model; periodic primal attempt bounded/deferred |
| Implicit derivatives | 80% | closed-axis and centerline-control tangent/adjoint contract |
| ANIMEC | 50% | source parity and independent finite-beta benchmark |
| Source/API simplification | 82% | reduce duplicated assembly and oversized files |
| ESSOS ownership cleanup | 100% | retain interchange tests only |

## 8. Explicit deferrals

The following do not block completion:

- kinetic end losses, sheaths, transport, and MHD stability;
- arbitrary curved open axes;
- radial B-splines or poloidal finite elements;
- differentiating CLI iteration histories or initial guesses;
- coefficient-native open-spline free boundary with the current cap-edge BIE;
- nonaxisymmetric free boundary pending a promoted native open-spline state;
- SOLVAX GMRES replacement of the tested SciPy host path;
- classic toroidal WOUT for open mirrors;
- VMEC2000 parity for open topology;
- coil optimization or lab-frame coil-field tracing inside vmec_jax;
- porting DESC mirror, double-Chebyshev, or finite-element branches;
- making the current pointwise-force reconstruction a promotion gate;
- four-dimensional phase-space or kinetic closures;
- a free hybrid if the fixed hybrid passes and the bounded vacuum attempts do
  not.

## 9. Primary references

- Hirshman and Whitson, VMEC variational method:
  https://princetonuniversity.github.io/STELLOPT/VMEC.html
- Hirshman, van Rij, and Merkel, free-boundary Green-function method:
  https://www.osti.gov/servlets/purl/5272232
- VMEC2000/ANIMEC source: https://github.com/PrincetonUniversity/STELLOPT
- VMEC++ numerics: https://arxiv.org/abs/2502.04374
- DESC source and inspected branches: https://github.com/PlasmaControl/DESC
- DESC high-order free boundary: https://arxiv.org/abs/2412.05680
- Rodriguez, Helander, and Goodman, straight-mirror near-axis analysis,
  Appendix C: https://doi.org/10.1017/S0022377824000345
- Agren and Savenko, Straight Field Line Mirror:
  https://info.fusion.ciemat.es/OCS/EPS2005/pdf/P4_069.pdf
- Goodman, Freidberg, and Lane, long-thin finite-beta mirror equilibrium:
  https://doi.org/10.1063/1.865851
- Pearlstein, quadrupole tandem-mirror paraxial equilibrium:
  https://digital.library.unt.edu/ark:/67531/metadc1102940/
- Anderson, Breazeal, and Sharp, VEPEC vector-potential mirror code:
  https://www.osti.gov/biblio/6351313
- Frank et al., high-field tandem-mirror Pleiades/RealTwin equilibria:
  https://doi.org/10.1017/S002237782510055X
- Ilgisonis, Berk, and Pastukhov, finite-beta toroidally linked mirrors:
  https://www.osti.gov/servlets/purl/10179323
- Feng et al., linked mirror with rotational transform:
  https://arxiv.org/abs/2103.09457
- Ranjan, helically linked mirrors:
  https://digital.library.unt.edu/ark:/67531/metadc1194643/
- Cooper et al., anisotropic variational equilibrium (1992):
  https://doi.org/10.1016/0010-4655(92)90002-G
- Cooper et al., bi-Maxwellian ANIMEC model (2006):
  https://doi.org/10.1088/0029-5515/46/7/001
- Cooper et al., anisotropic free boundary (2009):
  https://www.ornl.gov/publication/three-dimensional-anisotropic-pressure-free-boundary-equilibria
- Skene and Burns, automated spectral adjoints:
  https://arxiv.org/abs/2506.14792
- JAX implicit linear solve:
  https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html
- JAXopt implicit differentiation:
  https://jaxopt.github.io/stable/implicit_diff.html
- Lineax: https://arxiv.org/abs/2311.17283
- SOLVAX: https://github.com/uwplasma/SOLVAX

## 10. Immediate execution order

Execute these work packages in order. Each package ends in focused tests, a
small commit, and a push; CI is inspected after several packages or when a
completed run reports failure.

1. **Baseline closeout.** Confirm the repaired CI field shard, run the three
   closed tracer/solve tests, ruff, and `git diff --check`, and update the plan
   baseline. No physics changes.
2. **Low-risk reduction.** Use call-site searches and coverage to prune unused
   generic BIE functions and share duplicated host history/linear diagnostics.
   Preserve all benchmark outputs. Target at least 250 net deleted source lines
   before adding another feature.
3. **Required open promotion.** Run the axisymmetric free six-beta independent
   refinement gate and the open fixed B-spline/Chebyshev parity gate. Then close
   rotating-ellipse/SFLM amplitude and forward-tangent tests. Update only the
   compact benchmark JSON files and one compressed figure per showcase.
4. **Periodic preconditioning.** Implement and benchmark the cyclic spline
   block model with current SciPy/SOLVAX dependencies. Retain it only if it
   removes dense closed solves without changing the state; otherwise document
   the bounded failure and keep the largest supported closed resolution below
   the dense-memory limit.
5. **Fixed hybrid promotion.** Complete Phase-5 geometry constraints, circular
   VMEC/vmec_jax parity, racetrack refinement, open-leg limit, fixed beta scan,
   closed-axis implicit derivatives, and the parser-free example in that order.
   Do not start free hybrid before all seven fixed gates pass.
6. **Conditional free hybrid.** Make at most two smooth-toroidal virtual-casing
   attempts with ESSOS/MGRID input. Promote only with converged `B.n`, pressure
   jump, nested field lines, and beta-dependent LCFS displacement; otherwise
   remove the API/example and retain one compact negative benchmark.
7. **ANIMEC decision.** Perform the one source-parity and one independent-code
   attempt in Phase 6. Promote or delete/defer immediately; do not leave a
   research scaffold in the public API.
8. **Release reduction and evidence.** Reach the file/API/line targets, refresh
   README and docs from reproducible examples, run strict Sphinx, all mirror and
   example tests, full CI, and one CPU/GPU runtime-memory comparison. Keep PR
   #22 draft until every required lane is promoted and each conditional lane is
   either promoted or explicitly deferred.

This sequence is finite. A conditional lane gets two bounded attempts after its
prerequisites; it is then promoted or deferred. No new lane is introduced before
PR #22 reaches scientific review.
