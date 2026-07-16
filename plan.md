# Mirror equilibrium final plan

Status: final authoritative implementation and release plan for draft PR #22.
Revised 2026-07-15 after a source-level audit of the branch, current
`origin/main`, PR and CI, every mirror source/test/example/result, the original
15,583-line `plan_mirror.md`, DESC and its mirror/Chebyshev/racetrack branches,
VMEC2000 and ANIMEC, GVEC, SOLVAX 0.8.4, Pleiades, and the primary literature
listed in section 13.

This file replaces every earlier mirror roadmap. Do not create another plan.
Record execution results in section 12 and in the four compact benchmark JSON
files. A failed lane is removed from the supported code instead of remaining
as a public scaffold.

## 1. Goal

Deliver a small, research-grade `vmec_jax.mirror` backend for scalar-pressure,
nested-flux-surface equilibria in a finite straight-axis flux tube. The code
must:

1. solve the stated fixed- or free-boundary ideal-MHD model rather than only
   draw a prescribed surface;
2. use a nonperiodic longitudinal representation with independent end cuts;
3. reach component-wise equilibrium `ftol <= 1e-12` and independently
   verified weak-force, strong-force, and refinement gates;
4. provide implicit derivatives of converged supported equilibria;
5. keep the fast forward CLI independent of differentiability constraints;
6. consume external magnetic fields supplied by ESSOS or MGRID without
   owning coils or Biot-Savart calculations;
7. expose one compact API, two root examples, mirror-native output, and
   reviewed plots; and
8. remain substantially simpler than the current research branch.

The release is not required to solve anisotropic kinetic mirrors, end losses,
stability, or a closed stellarator-mirror hybrid. Those are different models
or failed promotion candidates and are separated in section 10.

## 2. Audited repository state

Pushed source snapshot used for this revision:

- branch source checkpoint: `codex/mirror-geometry` at `3b5d2715`;
- fetched base: `origin/main` at `ed4ac7ac`;
- ancestry: 0 commits behind and 361 commits ahead of `origin/main`;
- PR: <https://github.com/uwplasma/vmec_jax/pull/22>, open, mergeable, draft;
- pushed CI at the completed audit: all 13 executed jobs pass, including the
  implicit-gradient shard and 95% coverage gate; the manual/nightly full
  physics job is expectedly skipped and remains an R5 local/manual gate;
- diff at pushed HEAD: 44 files, 15,017 insertions, 1,590 deletions;
- mirror package: 13 files and 8,040 lines;
- mirror tests: 10 files and 4,388 lines;
- public mirror namespace: 18 lazy names;
- retained examples: two root scripts;
- retained figures: two compressed PNGs;
- retained evidence: four compact schema-1 JSON files.

The R1 removal implemented in the current worktree changes that release
candidate to 13 source files / 6,939 lines, 10 test files / 3,373 lines, and
17 public names. It removes 2,290 lines while adding 154 focused replacement
lines. These numbers become the new baseline when R1 is committed.

Fresh local verification during this revision:

- post-R1 normal mirror suite: 83 passed, 6 expected `full` deselections in
  186.08 s;
- strict Sphinx HTML build: passed with warnings treated as errors;
- Ruff lint and `git diff --check`: passed. The repository's existing format
  baseline is not globally Ruff-format clean, so R1 does not reformat
  unrelated code;
- changing the fixed-nonaxisymmetric example to the benchmark's medium
  `(ns,mpol,elements)=(7,6,6)` setting reached optimizer `ftol`, but selected
  a bad SFLM continuation basin: rotating-ellipse strong force was `0.0418`,
  while SFLM strong force was `1.09`, dominated by the fixed-end collars;
- the attempted fine example default (`9,8,8`) remained in Newton-GMRES after
  five minutes on the local CPU. It is a valid standalone refinement setting,
  not a practical continuation-example default. Keep `(5,8,4)` only as a
  runtime smoke until R2 establishes a resolution-robust continuation.

The pushed CI result validates the pre-R1 tree; R1's full tests and final
packaging are recorded separately in section 12 and repeated from a clean tree
in R5. This release tranche also includes the README showcase and removal of
one stale figure. Preserve those changes while executing this plan.

## 3. What is scientifically achieved

| Lane | Reproduced evidence | Decision |
| --- | --- | --- |
| Fixed open axisymmetric | exact polynomial-vacuum fixture; three spline grids; strong force `0.04349 -> 0.03474 -> 0.02873`; field error refines | release candidate |
| Fixed open nonaxisymmetric | rotating ellipse and Agren-Savenko SFLM; supplied-field initialization; three grids; strong force reaches `0.00976` and `0.01086`; tangent/adjoint checks | release candidate; example basin remains |
| Free open axisymmetric, beta 0-10% | coefficient LCFS/plasma solve, unbounded exterior BIE, three grids, pressure calibration, Pleiades trend, free adjoint | release candidate through central beta 10% |
| Free open axisymmetric, beta 25/50% | nonlinear residual reaches `1e-12`, but independent force/refinement gates fail; beta 50% gives radius `+7.73%` and center field `-23.73%` | research continuation only |
| Free open nonaxisymmetric | global observables look stable, but local `m=1` changes 73-81%; 3-grid beta pair costs 293/944/2995 s and 2.74/4.57/7.35 GiB | deferred; implementation removed |
| Fixed closed B-spline hybrid | exact 16/32/64 transfer and absolute gates pass, but same-geometry strong force is nonmonotone for two quadrature orders | deferred; closed runtime removed in R1 |
| Differentiation | fixed-open JVP/VJP and free-axisymmetric VJP agree with fully reconverged finite differences near `2e-10` | supported only for promoted lanes |
| Preconditioning | open separable/local sparse factor is effective; closed colored factor was effective but serves a deferred model | retain open path; archive closed result |

"Release candidate" means that committed numerical evidence passes the
scientific gate; no model is advertised as released until R5 passes. The
optimizer residual is not the equilibrium acceptance criterion. The coarse
example failure demonstrates why the release requires a staggered weak
variation, reconstructed `J x B - grad(p)`, and grid convergence in addition
to `ftol`.

## 4. Findings from other codes and literature

### 4.1 VMEC2000

VMEC2000 is a toroidal, doubly periodic inverse-coordinate solver. Its useful
lessons are the variational energy principle, exact divergence-free field
representation, full/half radial staggering, magnetic-axis regularity,
continuation, VMEC-style `fsq`, and Hessian-based preconditioning. Its NESTOR
vacuum solve is also a boundary-integral reference.

VMEC2000 is not an open-mirror oracle. A large-aspect-ratio torus still has
periodic field lines and no independent flux-carrying cuts. VMEC2000 parity is
therefore reserved for a future circular closed limit, not a release gate for
the open backend.

### 4.2 ANIMEC

ANIMEC is not scalar VMEC with two pressure arrays. The standalone
`STELLOPT/ANIMEC` tree and the integrated `origin/animec` branch change 13
VMEC files: pressure evaluation in `bcovar.f`, effective current and radial
force in `fbal.f`, flux addition, residuals, interface/output diagnostics,
input, and serial/parallel solve paths. In source, `bcovar.f` adds
perpendicular pressure to magnetic pressure and replaces `curl(B)` by the
effective current `curl(sigma B)`; `fbal.f` adds the fixed-`B` parallel
pressure derivative to radial balance. This is a coupled model change, not a
profile substitution.

Its variational model uses `p_parallel(s,B)` and parallel force balance gives

```text
p_perp = p_parallel - B * partial_B(p_parallel)|s
K = curl(sigma * B)
```

The free-interface pressure is `p_perp + B^2/(2 mu0)`, not the scalar-pressure
stress used by this PR. ANIMEC source also evolves its mass normalization with
the `B`-dependent pressure, computes parallel/perpendicular energies, and
reports firehose/mirror admissibility. Consequently, beta 25-50% scalar runs
cannot validate a modern high-beta mirror model. Anisotropy is a later
physical package, not an option added to `MirrorConfig`.

### 4.3 DESC branches

The DESC branches are implementation references, not validation authorities:

| Branch and audited SHA | Finding | Use here |
| --- | --- | --- |
| `master` `24aa7b9d` | mature toroidal continuation, constraints, objectives, and JAX patterns | algorithm/API reference |
| `mirror` `0dba071d` | separate 2,000-line `Equilibrium_mirror` and Chebyshev-Zernike work, but 123 changed files, notebook/output artifacts, a six-line end-boundary test with no test body, `1e-6` solve tests, and recorded boundary-condition failures | retain independent-cut concept only |
| `mirror_anisotropy` `805b77fc` | combines unfinished mirror and anisotropy experiments | no parity claim |
| `rg/racetrack` `2014ed0e` | eight-file, +694-line nonperiodic Chebyshev surface experiment; no periodic racetrack closure, volume solve, or added equilibrium test | basis experiment only |
| `dd/cylindrical` `6f85f50a` | tested double-Chebyshev-Fourier basis inside a 103-file development fork, no open mirror solve | basis identities/tests only |
| `tq/straight-stellarator` `8cf50b58` | old periodic straight-stellarator modifications | no open-cut semantics |
| finite-element branches `829e2db0`, `fb90e65` | exploratory spline/finite-element reconstructions with unresolved complexity and performance | do not add a second volume discretization |

DESC confirms that a global Chebyshev interval can represent an open axis and
independent ends. It does not provide a converged end-condition benchmark or
show that global Chebyshev coefficients are the best production unknowns. The
present clamped cubic B-spline state has local support, exact end constraints,
exact knot insertion, and a sparse physical Hessian. Keep
Chebyshev-Gauss-Lobatto nodes for independent evaluation and quadrature, not
as a second solved state.

### 4.4 GVEC, Pleiades, and linked mirrors

GVEC's flexible frame supports the future closed-hybrid design decision: use a
reference curve and a rotation-minimizing transverse frame instead of forcing
long straight sections through cylindrical toroidal Fourier modes. Its radial
B-spline/Fourier discretization is also a preconditioning reference.

Pleiades is an axisymmetric Green-function Grad-Shafranov solver and a useful
independent low-beta diamagnetic trend. It does not solve the same 3D
nested-coordinate variational problem. The current 10% center-field ratio
differs from the pinned Pleiades result by 0.36%, which is supporting evidence
rather than full parity. The 2026 WHAM reconstruction paper describes a
free-boundary anisotropic Pleiades solve with kinetic pressure bases; it
confirms that scalar beta alone is not a high-beta device validation.

PlasmaControl/FreeMHD was also screened at its current public revision. Its
tree is presently distribution and third-party infrastructure rather than a
usable mirror-equilibrium implementation, so it supplies neither production
code nor a release gate for this branch.

The Agren-Savenko SFLM gives an analytic paraxial vacuum field and straight,
nonparallel field lines. Its potential becomes singular near its formal axial
ends, so validation is restricted to a thin central interval. Savenko's thesis
reports only modest finite-beta ellipticity change for beta below about 0.2;
this is a future anisotropic validation target, not a scalar-pressure gate.

Feng et al.'s linked mirror has two nonparallel straight mirrors joined by
half-tori and obtains transform from the three-dimensional connection. The
Ilgisonis-Berk-Pastukhov report treats finite-beta linked quadrupole mirrors in
a small `beta/(epsilon*ellipticity)` expansion and predicts nonlinear outward
displacement. These works motivate a closed hybrid, but do not validate the
current failed periodic candidate.

### 4.5 Differentiable solver libraries

The correct derivative is the implicit derivative of the converged discrete
residual `F(u,p)=0`:

```text
F_u du = -F_p dp                         # tangent
F_u^T lambda = objective_u               # adjoint
d objective/dp = objective_p - lambda^T F_p
```

This matches JAX `custom_root`/`custom_linear_solve`, JAXopt implicit
differentiation, Lineax transpose-aware operators, and the sparse-spectral
adjoint strategy of Skene and Burns. It avoids retaining or differentiating
through nonlinear iterations.

Use forward mode for a few parameter directions or many outputs. Use an
adjoint for many controls and one/few scalar objectives. Reuse the primal
preconditioner for both and always report the true linear residual.

SOLVAX current main (`255d280`, version 0.8.4, audited 2026-07-15) now provides
pure-JAX FGMRES/GCROT, matrix-free Newton-Krylov, implicit root/linear solves,
transpose-aware operators, block-tridiagonal factors, bordered Schur
preconditioners, multigrid helpers, and true-residual diagnostics. It is
already a vmec_jax dependency through the toroidal core.

SOLVAX still does not provide the bound-constrained trust-region
globalization or host sparse factorization used by the mirror CLI, and its
Newton-Krylov path currently takes full steps without a merit line search.
Therefore R2 uses a measured replacement rule: adopt SOLVAX for a complete
primal/transpose path only if the spike deletes at least 250 local lines,
preserves bounds and all physics gates, reports true residuals, avoids new
host/device crossings, and stays within 10% of the faster wall time and peak
memory. Otherwise retain SciPy for the fast host CLI and record the negative
A/B result. Never maintain both production implementations.

### 4.6 Rejected shortcuts

| Shortcut | Decision |
| --- | --- |
| Large-aspect-ratio VMEC2000 torus as an open mirror | reject: topology and end cuts differ |
| Global axial Fourier modes for straight mirrors | reject: periodicity and inefficient straight-section fitting |
| Global Chebyshev coefficients as a second production state | reject: no benefit over local clamped splines has been demonstrated |
| Fourier projection of the toroidal racetrack hybrid | reject: the longitudinal geometry must remain periodic B-spline-native |
| Scalar beta 25-50% as a WHAM/high-beta validation | reject: anisotropic kinetic pressure is essential |
| Unrolled reverse AD through nonlinear iterations | reject: iteration memory/cost and derivative dependence on solver path |
| Public free-nonaxisymmetric or closed-hybrid scaffolds before promotion | reject: failed refinement/resource gates |
| Wholesale SOLVAX replacement without deletion/performance evidence | reject: globalization and host-factor gaps remain |

## 5. Supported physical model

This release solves static, scalar-pressure ideal MHD with nested open flux
surfaces. "Supported through beta 10%" is a numerical statement about this
model and its validation matrix, not a claim that isotropic MHD predicts a
specific high-beta mirror experiment. The code does not model loss cones,
sloshing ions, anisotropic pressure, flow, end loss, sheath physics, sources,
transport, or stability.

### 5.1 Domain and cuts

The open domain is a finite flux tube with coordinates
`(s, theta, xi)`, where `s in [0,1]`, `theta` is periodic, and
`xi in [-1,1]` maps to physical `z in [z_min,z_max]`.

For the supported straight axis,

```text
x = r cos(theta)
y = r sin(theta)
z = z(xi)
r = sqrt(s) a(s,theta,xi)
```

The two `xi` cuts carry prescribed geometry and normal magnetic flux. They are
not periodically identified, material end plates, plasma-vacuum interfaces,
or zero-normal-field boundaries. Field lines cross them. End losses, sources,
sheaths, and transport are outside this equilibrium model.

### 5.2 Field and energy

The nested-surface field is

```text
sqrt(g) B^theta = I'(s) - partial_z lambda
sqrt(g) B^z     = Psi'(s) + partial_theta lambda
B^s             = 0
```

and the scalar-pressure energy is

```text
W = integral [B^2/(2 mu0) + p/(gamma-1)] dV
p(s) = M(s) / V'(s)^gamma
```

`M(s)` is conserved VMEC-style mass. Geometry and `lambda` are varied while
the fixed cuts and lateral fixed boundary are projected exactly.

### 5.3 Free boundary

The supported free-boundary lane is axisymmetric. It jointly solves spline
LCFS coefficients, plasma coefficients, and a reduced unbounded exterior
Laplace problem. On the lateral LCFS:

```text
B_plasma . n = 0
B_vacuum . n = 0
p + B_plasma^2/(2 mu0) = B_vacuum^2/(2 mu0)
```

Graded disks close the two cuts only for the Green identity. They are not
physical plasma-vacuum interfaces. Discrete Neumann compatibility may be
corrected on these artificial caps only; lateral data must remain unchanged.

The external field contract is a differentiable vectorized `xyz -> B`
callable or an MGRID provider. ESSOS owns coil curves, currents, Biot-Savart,
and coil optimization.

## 6. Representation and solver decisions

| Direction/object | Production representation |
| --- | --- |
| radial plasma coordinate | VMEC-like full/half finite-difference mesh |
| poloidal cross-section | real Fourier collocation through `mpol`, with `ntheta=2*mpol+1` |
| open longitudinal state | clamped cubic B-spline coefficients |
| independent longitudinal evaluation | CGL nodes, differentiation, and Clenshaw-Curtis quadrature |
| free exterior | axisymmetry-reduced triangular-panel Green BIE with local Duffy quadrature |
| output | mirror-native MOUT, never a fake toroidal WOUT |

The forward solver remains a hybrid host/JAX implementation:

1. exact JAX energy/residual/JVP/VJP kernels in float64;
2. bounded SciPy globalization for robust CLI solves;
3. dense exact-Jacobian trust-region solves only as a tiny-system oracle;
4. matrix-free Newton-GMRES with bounds and merit backtracking for production
   systems;
5. local sparse/separable physical preconditioning;
6. hot continuation in shape, beta, and resolution;
7. true residual checks after every linear and nonlinear solve.

GPU is supplementary for this release. A GPU result must match CPU physics,
but it is not considered a performance success while host SciPy callbacks
cause device crossings.

The differentiable lane uses the same converged coefficient residual as the
primal lane and applies the implicit function theorem. It must never
differentiate the CLI iteration history. A tangent is used for a few control
directions or many outputs; one adjoint is used for many controls and a scalar
objective. The primal preconditioner is reused for both transpose directions.

## 7. Promotion gates

Every promoted state must pass all applicable gates. Optimizer success alone
never promotes a state.

### 7.1 Equilibrium gates

- float64 component-wise variational maximum `<= 1e-12`;
- independently assembled staggered weak maximum `<= 1.1e-12`;
- true primal and transpose linear relative residual `<= 1e-8`;
- normalized `div(B)` at the existing analytic/discrete floor;
- finite fields, one-sign Jacobian, nested surfaces, and positive radius;
- independently reconstructed all-volume strong force `< 5e-2` on the finest
  promoted grid, with axis, first-row, bulk, and end-collar values reported;
- strong force decreases over a declared three-grid physical refinement;
- geometry, `|B|`, pressure, flux, and current observables converge;
- iteration history, wall time, peak memory, and hardware are recorded.

### 7.2 Free-boundary gates

- plasma and vacuum tangency refine toward zero;
- normalized interface stress refines toward zero;
- raw and corrected Neumann compatibility are reported; corrected
  compatibility closes near roundoff;
- plasma, panel-angle, cap, and quadrature orders are refined independently;
- MGRID and callable external fields agree on a common fixture;
- requested and achieved central beta agree to the solve tolerance;
- every plotted beta boundary and field line comes from that beta's solved
  state;
- beta 25/50 remain research-only until the complete three-grid force and
  observable matrix passes.

### 7.3 Derivative gates

- JVP/VJP transpose identity passes;
- tangent and adjoint linear residuals pass `1e-8`;
- gradients agree with centered differences of fully reconverged equilibria;
- at least three finite-difference steps show a stable error plateau;
- no derivative is advertised for a deferred primal model.

### 7.4 Resource gates

- normal CPU test suite completes in under 8 minutes;
- each default root example completes in under 10 minutes and 8 GiB on the
  reference CPU;
- no required single equilibrium exceeds 30 minutes or 8 GiB;
- generated run directories, MOUT, CSV, restart, raw logs, and caches remain
  ignored;
- every committed PNG is compressed and under 400 KiB.

## 8. Finite release plan for PR #22

Execute R1-R5 in order. Commit and push after each accepted tranche. Do not
add another scientific lane before R5 passes.

### R1. Remove closed and dead contracts

The closed hybrid failed its declared same-geometry refinement gate and is not
in the release scope. Git history and
`benchmarks/mirror_hybrid_fixed_boundary.json` preserve the work.

Implementation status: code removal complete in the 2026-07-15 R1 worktree;
acceptance waits only on the full-test shard and the commit recorded in section
12. The interrupted local shard is rerun from the pushed checkpoint on the
office host; an interrupted run is not evidence.

1. Remove periodic/closed branches from `basis.py`, `geometry.py`,
   `forces.py`, `splines.py`, `solver.py`, and tests.
2. Remove center-map state fields, closed axis/frame evaluators, periodic
   knot transfer, closed initializers, cyclic/colored closed factors, and
   circular-torus-only tests.
3. Remove the `axis=` argument and closed wording from the public fixed solve.
4. Delete `EndCondition`; it has one valid value and adds no information.
   `MirrorConfig` documents fixed flux cuts directly.
5. Retain only the compact negative hybrid JSON and a short deferred-design
   section in documentation.
6. Prove fixed-open, free-axisymmetric, and their derivatives are unchanged.

Exit: no runtime path suggests closed equilibria are supported; public names
are at most 17; the staged R1 mirror source is at most 7,000 lines and tests
at most 3,500 lines; all promoted numerical values remain within recorded
tolerance. R2 must reduce source to the final 6,700-line ceiling.

### R2. Consolidate the open solver

1. Freeze two model residuals only: the fixed coefficient energy gradient and
   the coupled free-axisymmetric coefficient/interface residual. Their
   pack/unpack maps are the sole primal, tangent, and adjoint definitions.
2. Move fixed and free Newton-GMRES polishing onto one bounded host routine in
   `solver.py`; delete the duplicate loops currently named
   `_matrix_free_newton_polish` and `_polish_free_equilibrium`.
3. Move primal and transpose Krylov setup onto one true-residual helper. The
   fixed/free vectorizers and physics preconditioners remain model-owned; the
   generic iteration loop does not.
4. Keep one dense exact-Jacobian tiny oracle and one production matrix-free
   path. Remove algorithm-choice tests that verify implementation rather than
   behavior.
5. Run the SOLVAX 0.8.4 replacement A/B defined in section 4.5. Adopt it only
   if one complete local path disappears; otherwise keep the negative result
   out of the runtime and do not add an option flag.
6. Re-run open preconditioner A/B tests on the medium and fine fixed
   nonaxisymmetric grids. Record setup time, Krylov count, true residual, wall
   time, and peak memory.
7. Add a continuation-basin regression across the compact example and the
   three refinement grids. A resolution change must either reach the promoted
   strong-force branch or fail explicitly; optimizer `ftol` alone is not a
   successful equilibrium.
8. Keep SciPy/JAX host boundaries in one solver module. Do not add JAXopt,
   Optax, Equinox, Optimistix, or Lineax to this backend. SOLVAX is considered
   only under the complete-replacement rule above.
9. Reduce functions over 150 lines when extraction gives a named scientific
   operation; do not split files only to satisfy a line count.

Exit: one residual per model, one fixed solver path, one free solver path, and
one implicit linear-solve helper; at least 250 further source lines removed;
normal tests and benchmark values pass; mirror source is at most 6,700 lines.

### R3. Correct examples, plots, and output

1. Treat the current compact `(5,8,4)` fixed-nonaxisymmetric example only as
   a runtime smoke: its strong-force gates fail. After the R2 basin audit,
   choose the cheapest robust configuration that passes both cases. Keep the
   standalone `(5,4,4) -> (7,6,6) -> (9,8,8)` rotating-ellipse and
   `(7,6,6) -> (9,8,8) -> (11,10,10)` SFLM refinement studies separate.
2. Require both fixed example cases to assert `ftol`, weak force, strong force,
   divergence, geometry, and derivative gates before writing success output.
3. Replace free-example curves at `1.05*LCFS radius` with solved interior flux
   surface curves. In the zero-current axisymmetric lane these are physical
   constant-theta field lines from cap to cap. Do not draw decorative lines
   and label them as magnetic field lines.
4. Keep `z` horizontal in 3D and cross-section plots. Show coils supplied by
   ESSOS in the free-boundary example,
   solved LCFS/interior surfaces, actual field lines, `|B|`, pressure,
   requested/achieved beta, and residual history.
5. Make `vmec --plot mout_*.nc` reproduce 3D geometry, field lines, LCFS
   `|B|`, cross-sections, profiles, and convergence without example-private
   data.
6. Keep both root examples parser-free, with editable constants at the top and
   no example-private geometry, field-line, CSV, or plotting algorithms. Move
   reusable work behind public scientific/output APIs and target at most 220
   lines per example.
7. The free example may display beta `0,1,3,10,25,50%`, but it must mark 25%
   and 50% as research continuations and report their failed force gates. It
   must never interpolate one solved state to depict another beta.
8. Visually inspect every retained plot and perform nonblank pixel/dimension
   checks. Keep only the two current compressed showcase figures unless a new
   figure replaces one.

Exit: both root examples finish within the resource gate, all plots are from
solved states, MOUT round trips, and CLI plots are polished and nonblank.

### R4. Make documentation exact and executable

1. Finish the README mirror showcase with exactly the three supported models,
   one complete runnable API snippet, and the two retained figures.
2. Update or remove the stale repository code-size table after R1/R2. Never
   compare current source against old counts.
3. Correct `docs/mirror_geometry.rst`:
   - remove references to nonexistent `exterior_mesh.py`;
   - remove duplicated headings and obsolete restart-schema text;
   - describe the free adjoint as supported through beta 10% if its final
     coefficient-residual tests pass, otherwise remove it from public API;
   - remove claims that it reconstructs a former nodal residual;
   - make file ownership match the actual 12-or-fewer module tree;
   - distinguish supported, research-only, and deferred results everywhere.
4. Keep equations, shapes, units, cut semantics, force definitions,
   convergence gates, derivative scope, limitations, and exact reproduction
   commands. Move historical experiment narratives to compact benchmark JSON
   or delete them.
5. Keep examples parser-free with editable constants at the top and only
   public scientific API calls where practical.
6. Reach the 42-file diff budget without hiding code: remove the unnecessary
   one-line `tests/mirror/__init__.py`, move the eight mirror-listing lines out
   of the branch-only `examples/README.md` into the existing tutorials page,
   and delete that branch-only README. Do not merge scientific modules merely
   to reduce a counter.

Exit: strict Sphinx passes, every documented symbol/link exists, README code
runs, and no text overstates beta, free-nonaxisymmetric, or hybrid support.

### R5. Final release audit

Run from a clean tree at the final source commit:

1. `pytest tests/mirror -m "not full" -q`;
2. all `full` mirror tests, including three-grid and reconverged derivative
   gates;
3. affected core CLI, MGRID, device, output, and package tests;
4. strict Sphinx, pre-commit on changed files, and `git diff --check`;
5. wheel and sdist build, clean-venv install/import, console-script smoke;
6. both root examples in clean temporary directories;
7. MOUT read/write and `vmec --plot` smoke;
8. coverage of the retained mirror package with a 95% target;
9. final CPU timing/memory and one supplementary office-GPU parity run;
10. one batched CI review after the final push.

Final merge budgets, measured against fetched `origin/main`:

- at most 42 changed files;
- at most 6,700 mirror source lines;
- at most 3,500 mirror-test lines;
- at most 17 public mirror names;
- exactly two root mirror examples and two showcase figures;
- exactly four compact benchmark JSON records;
- no generated scientific output in Git;
- PR remains draft until every R1-R5 exit gate passes.

Exit: CI is green, branch is mergeable, final diff is reviewed, no supported
claim lacks independent evidence, and PR #22 can be marked ready for review.

## 9. File and API contract

The target is at most 13 files including `__init__.py` (12 owner modules).
Keep ownership clear; merge a file only when one coherent owner remains.

| File | Owner |
| --- | --- |
| `model.py` | open inputs, state, configuration, schemas |
| `basis.py` | theta Fourier, CGL evaluation, clamped cubic B-splines |
| `geometry.py` | straight-axis embedding, metric, Jacobian, field conversion |
| `forces.py` | energy, mass-pressure relation, weak/strong/interface residuals |
| `analytic.py` | exact polynomial, rotating ellipse, SFLM fixtures |
| `splines.py` | coefficient maps, initialization, fixed solve adapter |
| `solver.py` | bounded globalization, Newton-GMRES, open preconditioner |
| `exterior.py` | closed integration surface, panels, singular quadrature |
| `exterior_bie.py` | axisymmetric exterior Neumann solve and LCFS trace |
| `free_boundary.py` | coupled axisymmetric free solve and beta continuation |
| `implicit.py` | supported converged-residual tangents and adjoints |
| `output.py` | MOUT, restart, diagnostics, and plots |

The public namespace contains only user inputs, spline states/discretization,
the three solve workflows, supported derivatives, and MOUT/plot helpers.
Result dataclasses and numerical kernels remain in their owner modules.

No coils, Biot-Savart, coil geometry, coil optimization, generic virtual
casing, toroidal WOUT compatibility shim, or second solved basis belongs in
this package.

## 10. Deferred plans after PR #22

These are separate go/no-go PRs. They do not block PR #22 and must not leave
public scaffolds if their bounded gates fail.

### N1. Free open nonaxisymmetric mirror

Prerequisite: merged open release and a structured 3D exterior operator. The
current dense panel Jacobian scales too poorly and failed local-mode
refinement.

1. Design a matrix-free/high-order surface BIE with explicit corner and cap
   treatment, fast operator actions, and transpose/shape actions.
2. Validate harmonic manufactured solutions and near-surface fields before
   coupling plasma.
3. Couple beta zero first to the promoted fixed rotating ellipse; require
   local Fourier coefficients, not only global radius and field.
4. Continue ellipticity, section rotation, current, and beta independently.
5. Require three grids, local `m=1/m=2` convergence, all release physics
   gates, under 30 minutes and 8 GiB per state.
6. Add derivatives and a public entry point only after the primal matrix
   passes.

If any third-grid or resource gate fails, retain one negative JSON and remove
the implementation.

### H1. Toroidal stellarator-mirror hybrid with full longitudinal B-splines

Prerequisite: a fresh branch after PR #22. Recover useful algorithms from Git
history, not the old public scaffold.

The target is a closed toroidal racetrack: two long straight mirror legs and
two smooth curved stellarator returns, with exact leg-exchange/up-down
symmetry. It is not the old square Fourier projection. Let `ell` be a periodic
longitudinal parameter and write

```text
x(s,theta,ell) = c(ell)
               + X(s,theta,ell) e1(ell)
               + Y(s,theta,ell) e2(ell).
```

Periodic cubic B-splines represent the reference axis `c`, every longitudinal
coefficient of `X`, `Y`, and `lambda`, and any interior section-center map.
Poloidal dependence of `X` and `Y` remains real Fourier; radial dependence
remains VMEC-like staggered. A Bishop/rotation-minimizing frame supplies
`e1,e2`; a smooth periodic holonomy correction plus an independent section
angle makes the frame/section close exactly. Symmetry is imposed by tying
spline coefficients, not by fitting or projecting a sampled Fourier surface.
This is the required meaning of a full B-spline hybrid.

1. Basis gate: implement periodic cubic evaluation, two derivatives, cyclic
   band structure, and exact dyadic knot insertion in an isolated module.
   Test partition of unity, local support, periodic closure, JVP/VJP transpose,
   and exact geometry preservation before adding equilibrium code.
2. Geometry-only gate: exact closure of `c`, tangent, frame, and section
   through two derivatives; at least 50% low-curvature length on each leg;
   smooth curvature ramps in the returns; 90-degree difference between the
   two straight-leg ellipses; frame holonomy cancelled explicitly; positive
   clearance/Jacobian; exact leg symmetry; and exact nested knot refinement.
3. Circular-limit gate: compare a periodic spline circle with ordinary
   `vmec_jax` and VMEC2000 at identical physical boundary, flux, pressure,
   current, and radial/poloidal resolution.
4. Open-leg gate: increase straight-to-return scale on three geometries and
   compare the central halves with the promoted fixed-open rotating ellipse
   and SFLM geometry, `|B|`, field-line, and force observables. The local
   straight section is validated before the returns are trusted.
5. Residual gate: diagnose the previous nonmonotone 16/32/64 strong-force
   sequence before increasing resolution. Check mapping gauge, axis
   regularity, independent staggered force, and quadrature aliasing on one
   exactly preserved geometry.
6. Solver gate: start from the promoted open residual/preconditioner, replace
   only longitudinal boundary conditions and metric/frame terms, and retain
   one coefficient residual. Require monotone 16/32/64 longitudinal,
   three-grid radial/poloidal, and independent quadrature refinement.
7. Transform gate: report current-free geometric transform separately from
   transform driven by continued on-axis current. Do not assume either sign or
   magnitude.
8. Finite-beta fixed-LCFS gate: beta `0,1,3,10%`, with Ilgisonis et al. used
   only for sign/scaling in its asymptotic regime.
9. Only after monotone three-grid strong-force convergence: add implicit
   derivatives, periodic MOUT, CLI plots, and one root example.
10. Free boundary and ESSOS coil coupling are later than the fixed hybrid.
    Coils and Biot-Savart remain entirely in ESSOS.

The old result `0.0528, 0.107, 0.0235` is a failed refinement sequence even
though the finest value is below `0.05`. Do not promote by changing the gate
or running only 128 controls.

### A1. Anisotropic mirror equilibrium

Prerequisite: stable scalar open release and a written physical closure.

1. Define `p_parallel(s,B)` and derive `p_perp`, effective current, energy,
   weak force, and free-interface stress from one model.
2. Reproduce an ANIMEC isotropic limit and at least one published anisotropic
   toroidal benchmark before applying the closure to open mirrors. Port the
   pressure normalization, effective current, radial force, edge stress, and
   admissibility diagnostics as one coherent model.
3. Validate paraxial perpendicular pressure balance
   `B_vac^2 = B^2 + 2 mu0 p_perp` and firehose/mirror admissibility.
4. Compare axisymmetric WHAM/Pleiades field depression and surface expansion
   over a physically specified distribution, not scalar beta alone. Include a
   fixture from Frank et al. (2026) or a released Pleiades equivalent when its
   inputs are reproducible.
5. Revisit SFLM finite-beta ellipticity below beta 0.2.
6. Treat 25/50% beta as supported only after anisotropic force, refinement,
   and stability-domain gates pass.

## 11. Canonical evidence

Only these tracked numerical records remain:

1. `benchmarks/mirror_fixed_boundary.json`;
2. `benchmarks/mirror_free_boundary_axisymmetric.json`;
3. `benchmarks/mirror_free_boundary_nonaxisymmetric.json` (negative);
4. `benchmarks/mirror_hybrid_fixed_boundary.json` (negative).

Each record uses schema `vmec_jax.benchmark.mirror/1`, identifies a committed
measurement source and hardware, and stores only data needed to decide a gate.
Figures are presentation artifacts, not numerical truth. Raw outputs and
exploratory logs remain outside Git.

## 12. Execution log and completion

At this revision:

| Lane | Completion | Remaining work |
| --- | ---: | --- |
| Fixed open axisymmetric physics | 100% | regression only |
| Fixed open nonaxisymmetric physics | 90% | robust continuation/default and final smoke |
| Free open axisymmetric through 10% | 95% | final example/plot and full-suite audit |
| Implicit derivatives for promoted lanes | 95% | final reconverged audit after simplification |
| Open preconditioning | 90% | post-removal A/B record |
| Closed hybrid disposition | 100% | negative record retained; future H1 is separate |
| Nonaxisymmetric free disposition | 100% | compact negative evidence retained |
| API/code simplification | 78% | commit R1; R2 shared solver and final line budget |
| README/docs/examples/plots | 78% | R3/R4 corrections and visual review |
| Packaging/CI/release audit | 75% | post-R1 full suite and R5 clean audit |

Weighted completion of PR #22 is approximately 83%. Deferred N1/H1/A1 work
is not included in that percentage.

### 2026-07-15 R1 removal and final plan audit

- Steps: fetched all audited repositories; reviewed current main/PR/CI, the
  complete mirror package and history, original plan, DESC experimental
  branches, VMEC2000/ANIMEC source, GVEC, SOLVAX 0.8.4, Pleiades, and mirror
  literature; removed every periodic/closed runtime branch and the one-value
  end-condition enum.
- Results: mirror source `8,040 -> 6,939` lines, tests `4,388 -> 3,373`, public
  names `18 -> 17`; no file or dependency was added. Open fixed/free numerical
  behavior is unchanged in the normal suite.
- Tests: 83 normal mirror tests pass with 6 full tests deselected in 186.08 s;
  strict Sphinx `-W`, Ruff lint, compileall, and `git diff --check` pass. The
  six full tests must pass from the R1 checkpoint before R2 is accepted.
- Files/API: 13 owner modules remain; no new file or dependency was added.
  Closed axis/frame, center-map, periodic transfer, closed solver, and closed
  preconditioner APIs are gone. The compact negative hybrid JSON remains.
- Best next step: commit/push the reviewed R1 checkpoint, run its full shard
  on the office host, then execute R2's one shared bounded Newton/Krylov path
  and continuation-basin matrix.
- Open lanes: fixed axisymmetric 100%, fixed nonaxisymmetric 90%, free
  axisymmetric 95%, derivatives 95%, preconditioning 90%, simplification 78%,
  docs/examples 78%, release audit 75%.
- User input: none required for R1-R5. N1/H1/A1 begin only after a separate
  post-release go/no-go decision.

After every implementation tranche, append one short dated entry here with:

- steps taken;
- numerical results and failed gates;
- tests and hardware;
- files/API affected and current budgets;
- best next step;
- percentages for every open lane;
- any user input genuinely required.

## 13. Reviewed sources

Local source revisions audited on 2026-07-15:

- `vmec_jax` main `ed4ac7ac`, mirror baseline `3b5d2715`, the R1 worktree,
  and all 361 PR #22 commits relative to main;
- DESC master `24aa7b9d`, mirror `0dba071d`, mirror-anisotropy
  `805b77fc`, racetrack `2014ed0e`, cylindrical/Chebyshev `6f85f50a`,
  straight-stellarator `8cf50b58`, and finite-element `829e2db0`/
  `fb90e65`;
- STELLOPT develop `e03e72e9`, standalone ANIMEC branch, and integrated
  ANIMEC commits `81379809`, `8ec7875a`, `91bfd08e`;
- local VMEC2000 `728af8ba`;
- SOLVAX current main `255d280` (0.8.4), including changes since `005ca387`;
- Pleiades pinned reference `0161abb3`.

Primary implementation and physics references:

- Hirshman and Whitson, VMEC energy principle:
  <https://www.osti.gov/biblio/5497291>
- Hirshman, van Rij, and Merkel, free-boundary VMEC:
  <https://doi.org/10.1016/0010-4655(86)90058-5>
- Hirshman and Betancourt, VMEC radial preconditioning:
  <https://doi.org/10.1016/0021-9991(91)90267-O>
- VMEC2000 and ANIMEC source:
  <https://github.com/PrincetonUniversity/STELLOPT>
- ANIMEC overview:
  <https://www.epfl.ch/research/domains/swiss-plasma-center/research/theory/codes/animec/>
- Cooper et al., anisotropic free-boundary equilibria:
  <https://doi.org/10.1016/j.cpc.2009.04.006>
- Asahi et al., ANIMEC pressure model:
  <https://doi.org/10.1585/pfr.6.2403123>
- DESC source and experimental branches:
  <https://github.com/PlasmaControl/DESC>
- DESC free-boundary formulation:
  <https://arxiv.org/abs/2412.05680>
- GVEC flexible coordinate frame:
  <https://arxiv.org/abs/2410.17595>
- GVEC implementation paper:
  <https://doi.org/10.21105/joss.09670>
- de Boor, local B-spline basis, knot insertion, and band structure:
  <https://doi.org/10.1007/978-1-4612-6333-3>
- Bishop, rotation-minimizing frames for curves:
  <https://doi.org/10.1080/00029890.1975.11993807>
- Pleiades:
  <https://github.com/eepeterson/pleiades>
- WHAM physics basis:
  <https://doi.org/10.1017/S0022377823000806>
- Frank et al., nonlinear anisotropic WHAM/Pleiades reconstruction:
  <https://doi.org/10.1063/5.0306291>
- Frank et al., anisotropic high-field tandem-mirror equilibrium:
  <https://doi.org/10.1017/S002237782510055X>
- Agren et al., straight-field-line mirror:
  <https://info.fusion.ciemat.es/OCS/EPS2005/pdf/P4_069.pdf>
- Savenko, SFLM finite-beta and confinement thesis record:
  <https://urn.kb.se/resolve?urn=urn:nbn:se:uu:diva-6637>
- Rodriguez, Helander, and Goodman, paraxial rotating-section formulas:
  <https://arxiv.org/abs/2311.14439>
- Ilgisonis, Berk, and Pastukhov, finite-beta linked mirrors:
  <https://doi.org/10.2172/10179323>
- Feng et al., linked mirror:
  <https://arxiv.org/abs/2103.09457>
- Skene and Burns, automated sparse-spectral adjoints:
  <https://arxiv.org/abs/2506.14792>
- Knoll and Keyes, Jacobian-free Newton-Krylov methods:
  <https://doi.org/10.1016/j.jcp.2003.08.010>
- Blondel et al., modular implicit differentiation:
  <https://arxiv.org/abs/2105.15183>
- JAX implicit linear solves:
  <https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html>
- JAXopt implicit roots:
  <https://jaxopt.github.io/stable/implicit_diff.html>
- Lineax structured/transpose-aware solvers:
  <https://docs.kidger.site/lineax/>
- SOLVAX:
  <https://github.com/uwplasma/SOLVAX>

References justify model and algorithm choices. Only reproduced tests and
committed compact benchmark records count as promotion evidence.
