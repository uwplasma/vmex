# Mirror equilibrium final implementation plan

Status: active and authoritative plan for draft PR #22. This file supersedes
`/Users/rogeriojorge/Downloads/plan_mirror.md` and every earlier roadmap in the
branch. Do not add another plan file. Commits and compact benchmark JSON/CSV
files are the execution log.

Review baseline: `codex/mirror-geometry` at `eb4955cd696f`, based on
`origin/main` at `ed4ac7acae11`, reviewed 2026-07-14. The branch is 261 commits
ahead and zero behind `origin/main`; PR #22 is open, draft, and mergeable. The
latest CI run exposed one concrete workflow defect: both fast jobs referenced
the deleted `tests/test_coils.py`. Removing that stale path makes the exact local
shard pass with 200 passed and 6 skipped. The grouped CI run still needs to
confirm this fix and the parity-c job was pending at the audit cutoff.

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
  and plots of solved states.
- **ESSOS** owns coils, Biot-Savart, coil-field-line tracing, and mgrid creation.
  vmec_jax accepts `MgridField` or a vectorized `xyz -> B` callable.
- **SOLVAX** owns reusable Krylov/direct solvers, generic preconditioners,
  chunked AD, and implicit linear/root-solve machinery.
- **virtual-casing-jax** owns generic singular Laplace/virtual-casing kernels.
  Mirror code owns only the open-surface geometry and boundary data needed to
  call those kernels.
- **SciPy** may control fast nondifferentiable CLI nonlinear solves. There is no
  requirement to trace or differentiate the host iteration history.

No coil model, Biot-Savart implementation, field-line integrator, general BIE
library, general finite-element package, or duplicate GMRES implementation is
added to vmec_jax.

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
solver dependency because vmec_jax already uses it. Only APIs released on
SOLVAX `main` may be required. The CI-resolved and current PyPI release, SOLVAX
0.8.3, provides GMRES, GCROT, PCG, block Thomas, ordinary and periodic banded
LU, operators, chunked Jacobians, implicit solves, and Newton-Krylov. The local
0.7.3 environment already has periodic banded LU but not the released
Newton-Krylov interface. Raise vmec_jax's current broad dependency floor only
after API and numerical parity at 0.8.3; no mirror-local cyclic solver is needed.

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

Relative to `origin/main`, the branch changes 137 files, adds 24,370 lines, and
deletes 4,255 lines. `vmec_jax/mirror` contains 10,146 lines in 22 modules and
exposes 47 lazy names. Its largest files are `forces.py` (1,049), `solver.py`
(987), `splines.py` (907), `exterior_bie.py` (809), and `exterior_mesh.py`
(737). There are 148 collected mirror tests.

The branch also carries earlier QI, direct-coil, optimization, and core-refactor
work that is unrelated to the final mirror diff. This is the first
simplification blocker. Local ruff and strict Sphinx pass. `git diff --check`
currently reports four blank-line-at-EOF errors, three in unrelated core files.

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

### 4.3 Results that are not promotion evidence

- The current circular-torus test freezes `lambda`. Its 23-iteration,
  `1.05e-16` variational result is only a radius subproblem and is not a
  complete vacuum equilibrium. With `solve_lambda=True`, a zero-lambda start
  requires about 812 evaluations at `ns=5` but lowers the pointwise force from
  about 4.90 to 0.71; at `ns=7` it falls to 0.57. The plan and docs must not
  cite the frozen-lambda result as a solved torus.
- Closed spline solves have no independent staggered weak residual yet.
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

## 5. External code and literature conclusions

### 5.1 VMEC2000 and VMEC++

Retain VMEC's variational principle, divergence-free representation, radial
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

DESC `master`, `mirror`, `mirror_anisotropy`, `finite_element_basis`,
`finite_element_basis_alan`, and `dd/cylindrical` were inspected locally after
fetching current remote refs.

Useful ideas:

- a nonperiodic basis owns nodes, quadrature, differentiation, interpolation,
  endpoint semantics, and coefficient transfer;
- Chebyshev/Fourier products cleanly separate an open coordinate from a
  periodic coordinate;
- continuation and objective scaling are explicit;
- DESC's current free-boundary work treats `B.n` and magnetic-pressure jump as
  separate area-weighted residuals.

Do not port these branches. The mirror branches copy a large equilibrium class,
use solve tolerances near `1e-6`, and contain unfinished boundary-objective
methods. The finite-element branches add broad 1D/2D/3D mesh machinery. The
June 2026 `dd/cylindrical` branch has useful double-Chebyshev basis tests, but it
is an early basis experiment rather than a demonstrated mirror solver.

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

The current mirror ANIMEC lane may be retained only if these source equations,
normalizations, and limits pass one bounded parity implementation and one
independent finite-beta case.

## 6. Ordered implementation phases

Each phase ends with focused tests, a compact benchmark update, one or more
small commits, and a push. CI is checked after grouped work rather than polled
after every commit. No new physics lane is added.

### Phase 0: correct the baseline and CI

1. Push the stale `tests/test_coils.py` workflow-path fix, confirm both fast
   shards in the next grouped CI run, and audit the pending parity-c result.
   Address any additional failure without polling unrelated long jobs.
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

### Phase 1: reduce the PR before adding physics

1. Classify all 137 changed files as mirror-required, current-main integration,
   or inherited unrelated work.
2. Restore unrelated QI, direct-coil, optimization, and core files to
   `origin/main` in ordinary commits. Keep only integration changes required by
   mirror CLI, plotting, packaging, and shared solver contracts.
3. Reduce the public mirror namespace from 47 names to at most 24.
4. Choose one production exterior formulation: unbounded panel/Green solve.
   Keep the annulus only as a small internal oracle or delete it after parity.
   Remove unused spectral/curved-panel variants that do not improve the bounded
   convergence gate.
5. Consolidate modules only where ownership is artificial. Target at most 16
   mirror modules, at most 8,500 mirror source lines, and no file above 800
   lines without a written reason.
6. Delete stale benchmarks, generated outputs, duplicate examples, and docs for
   removed paths.

Gate: the diff is materially smaller, all retained benchmark claims reproduce,
and no physics result depends on an unrelated branch-only core refactor.

### Phase 2: finish promoted open scalar mirrors

1. Re-run axisymmetric fixed and free cases with native open B-spline state.
   Compare Chebyshev as an internal oracle across three radial/axial grids and
   beta `[0,.01,.03,.10,.25,.50]`.
2. Make coefficient-native B-spline free-boundary boundary/state packing share
   the fixed solver's gauge, masks, scaling, and residual assembly.
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

- axisymmetric free: less than 0.5% center-radius and 2% center-field change
  between the two finest bounded grids;
- nonaxisymmetric fixed: field direction, ellipse angle, flux determinant,
  quadrupole phase, energy, volume, pitch, and weak residual converge;
- all promoted residual and geometry contracts in Section 1 pass.

### Phase 3: structured solver and preconditioner

1. Define one packed residual/vectorizer contract for nodal and spline open or
   periodic states. Remove duplicated optimizer callbacks and history code.
2. Preserve radial block structure and Fourier mode blocks. Assemble spline
   mass/stiffness matrices from local support rather than dense global axial
   differentiation matrices.
3. Use released SOLVAX block Thomas/banded kernels for separable blocks and
   SOLVAX GMRES/GCROT on exact JVPs. Keep SciPy as the host trust-region or
   minimization driver where it is faster.
4. Establish numerical/API parity on SOLVAX 0.8.3, then set that as the minimum
   version if Newton-Krylov is retained. Do not depend accidentally on whichever
   SOLVAX version the environment happens to resolve.
5. Implement gauge-free periodic spline preconditioning with released periodic
   banded LU; do not slice periodic coefficients as though they had open
   endpoints.
6. Compare no preconditioner, current separable model, and structured model
   across `ns`, `mpol`, and knot count. Record compile time, steady time, peak
   RSS/device memory, nonlinear evaluations, Krylov iterations, and residual.
7. Remove mirror-local generic Krylov code after SOLVAX parity.

Gate: medium cases avoid dense Jacobian materialization, Krylov growth is
bounded or slowly growing, converged states are unchanged, and the
nonaxisymmetric medium solve no longer requires an 800-second dense path.

### Phase 4: one bounded nonaxisymmetric free-boundary retry

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

### Phase 5: promote the fixed B-spline hybrid

1. Complete circle geometry derivative, periodic knot-refinement, holonomy,
   self-clearance, and positive-Jacobian tests.
2. Add control-point constraints and tests for midplane up-down symmetry and
   mirror-leg exchange symmetry without specializing the spline basis.
3. Complete the vacuum circular-torus solve with the `1/R` initializer. Refine
   radial, theta, and periodic spline resolution independently.
4. Compare the near-circular result with normal vmec_jax and local VMEC2000 at
   matched flux, pressure, current, and boundary. Compare energy, volume,
   iota, `|B|`, and force residual definitions rather than raw iteration count.
5. Implement the closed staggered weak residual and require agreement with the
   coefficient variational residual.
6. Solve the symmetric rotating-ellipse racetrack with finite current. Measure field-line
   pitch/iota from `dtheta/du = B^theta/B^u`, follow field lines for multiple
   turns, and verify nested closure.
7. Increase straight-leg length and compare leg-center sections, fields, and
   paraxial coefficients with the open spline solver.
8. Run beta `[0,.01,.03,.10,.25,.50]` at fixed boundary. Compare the nonlinear
   outward displacement/field response qualitatively with linked-mirror
   theory, without treating an asymptotic paper as an exact target.
9. Validate forward and reverse implicit derivatives with respect to centerline
   controls, section coefficients, profiles, flux, and current.
10. Add one parser-free root example and MOUT/plot support for horizontal 3-D
   surfaces, visible field lines, cross sections, `|B|`, pressure, iota/pitch,
   magnetic well, and residual/refinement histories.

Gate: complete fixed hybrid, both limiting checks, converged weak residual,
stable pitch/iota, implicit derivatives, and reproducible plots.

Only after this gate may one closed-hybrid free-boundary attempt use the normal
toroidal vacuum contract with ESSOS/MGRID. Apply the same two-attempt stop rule
as Phase 4. Free hybrid is conditional and cannot delay fixed-hybrid promotion.

### Phase 6: derivatives and ANIMEC decision

1. Expose implicit JVP and VJP only for lanes promoted in Phases 2, 4, and 5.
2. Check state tangents and scalar-objective adjoints against reconverged
   centered finite differences over step-size sweeps. Report the tangent or
   adjoint linear residual and condition estimate.
3. Use `jax.lax.custom_linear_solve` or SOLVAX implicit wrappers at the
   converged residual; do not add JAXopt, Optax, Lineax, or Equinox solely to
   wrap the existing root.
4. Audit ANIMEC source equations against `fbal.f`, `bcovar.f`, `forces.f`, and
   `jxbforce.f`. Test the bi-Maxwellian form factor above/below the critical
   field, `p_perp` AD/FD, isotropic limit, hot-fraction-zero limit, sigma/tau,
   and interface stress.
5. Run one independent finite-beta anisotropic mirror benchmark and one
   resolution study. Promote or defer ANIMEC immediately after these attempts.

Gate: supported derivatives have verified primal/linear residuals and FD error;
ANIMEC has an unambiguous supported or deferred status.

### Phase 7: final simplification and release evidence

1. Recount files, source lines, modules, public names, tests, and artifacts
   against Section 4.1. The final numbers must be lower after Phase 1.
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
| Axisymmetric free mirror | 80% | native spline coupling and scaling |
| Nonaxisymmetric fixed mirror | 82% | amplitude, forward tangent, preconditioned refinement |
| Nonaxisymmetric free mirror | 55% | structured-solver retry and local-mode convergence |
| Open native B-splines | 70% | free-boundary coefficients and public-default decision |
| Fixed closed B-spline hybrid | 32% | complete torus, weak force, limits, iota, derivatives |
| Free closed hybrid | 10% | conditional after fixed promotion |
| Preconditioning | 45% | periodic blocks and bounded Krylov scaling |
| Implicit derivatives | 74% | spline forward tangent, hybrid and retained free lanes |
| ANIMEC | 50% | source parity and independent finite-beta benchmark |
| Source/API simplification | 25% | remove inherited diff, vacuum variants, modules, and exports |
| ESSOS ownership cleanup | 100% | retain interchange tests only |

## 8. Explicit deferrals

The following do not block completion:

- kinetic end losses, sheaths, transport, and MHD stability;
- arbitrary curved open axes;
- radial B-splines or poloidal finite elements;
- differentiating CLI iteration histories or initial guesses;
- classic toroidal WOUT for open mirrors;
- VMEC2000 parity for open topology;
- coil optimization or field-line tracing inside vmec_jax;
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

1. Execute Phase 0: fix CI, the incomplete torus claim, and the open-cap
   boundary contract.
2. Execute Phase 1 before adding another solver or geometry path.
3. Finish the promoted open scalar/B-spline lanes in Phase 2.
4. Complete structured linear algebra in Phase 3, then make the one bounded
   nonaxisymmetric free-boundary decision in Phase 4.
5. Promote the fixed closed hybrid in Phase 5; attempt free hybrid only after
   that gate.
6. Close derivative and ANIMEC decisions in Phase 6.
7. Execute the deletion, documentation, GPU, and release gates in Phase 7.

This sequence is finite. A conditional lane gets two bounded attempts after its
prerequisites; it is then promoted or deferred. No new lane is introduced before
PR #22 reaches scientific review.
