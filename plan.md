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

Deliver a small, validated `vmec_jax.mirror` backend for scalar-pressure,
nested-flux-surface equilibria in open straight-axis flux tubes and a closed
two-leg stellarator-mirror hybrid. The code
must:

1. solve the stated fixed- or free-boundary ideal-MHD model rather than only
   draw a prescribed surface;
2. use a nonperiodic longitudinal representation with independent end cuts
   for open mirrors and a full periodic B-spline representation for hybrids;
3. reach component-wise equilibrium `ftol <= 1e-12` and independently
   verified weak-force, strong-force, and refinement gates;
4. provide implicit derivatives of converged supported equilibria;
5. keep the fast forward CLI independent of differentiability constraints;
6. consume external magnetic fields supplied by ESSOS or MGRID without
   owning coils or Biot-Savart calculations;
7. expose one compact API, three root examples, mirror-native open output, and
   reviewed plots; and
8. remain substantially simpler than the archived experimental scaffold.

The release is not required to solve anisotropic kinetic mirrors, end losses,
stability, or a free-boundary hybrid. Those are different models or later
promotion candidates and are separated in section 10.

## 2. Audited repository state

The completed open-mirror R1-R5 snapshot was:

- branch source checkpoint: `codex/mirror-geometry` at `5ff15d99`;
- fetched base: `origin/main` at `ed4ac7ac`, with no commits behind;
- PR: <https://github.com/uwplasma/vmec_jax/pull/22>, open and mergeable;
- pushed CI: all executed jobs pass, including mirror, implicit-gradient,
  packaging, examples, and the 95% coverage gate;
- diff at the source checkpoint: 42 files, 13,034 insertions, 1,622 deletions;
- mirror package: 13 files and 6,999 lines;
- mirror tests: 10 files and 3,443 lines;
- public mirror namespace: 17 lazy names;
- retained examples: two root scripts;
- retained figures: two compressed PNGs;
- retained evidence: four compact schema-1 JSON files.

H1 now supersedes only the closed-hybrid disposition. At pushed checkpoint
``7df5056f`` the branch has 44 changed files, 7,915 mirror-source lines, 3,665
mirror-test lines, 20 public mirror names, three root mirror examples, and
three nonredundant compressed mirror showcase figures after the current
documentation trim. No coil/Biot-Savart source or new scientific module was
added.

Final verification for this revision:

- normal mirror suite: 87 passed, 6 expected `full` deselections in 185.62 s;
- all six full mirror tests pass; the office RTX A4000 shard took 23:59, with
  the corrected three-grid test separately confirmed in 14:36;
- both root examples pass in clean temporary directories in 9:02 total;
- strict Sphinx HTML build: passed with warnings treated as errors;
- pre-commit, compileall, package build/install, console smoke, CLI plot
  round trips, and `git diff --check`: passed;
- the corrected-cut medium `(ns,mpol,elements)=(7,6,6)` example reaches
  variational `ftol`; rotating-ellipse strong force is `0.0418`, while the
  SFLM strong force is `1.09`, dominated by the fixed-end collars. Starting
  directly with Newton reproduces the same state, so this is not merely an
  L-BFGS basin failure;
- the two showcase figures and every CLI plot were visually reviewed; all
  retained figures come from solved MOUT states and remain compressed.

## 3. What is scientifically achieved

| Lane | Reproduced evidence | Decision |
| --- | --- | --- |
| Fixed open axisymmetric | exact polynomial-vacuum fixture; three spline grids; strong force `0.04349 -> 0.03474 -> 0.02873`; field error refines | supported |
| Fixed open nonaxisymmetric rotating ellipse | larger `0.12 m` LCFS, strong force `0.0267`, divergence `6.7e-15`, reconverged adjoint error `5.9e-10` | supported |
| Agren-Savenko straight-field-line target | larger `0.10 m` LCFS reaches variational `1.7e-16`, but reconstructed strong force is `0.335` and collar force is `0.701` | unsupported validation case |
| Free open axisymmetric, beta 0-10% | coefficient LCFS/plasma solve, unbounded exterior BIE, three grids, pressure calibration, Pleiades trend, free adjoint | supported through central beta 10% |
| Free open axisymmetric, beta 25/50% | nonlinear residual reaches `1e-12`, but independent force/refinement gates fail | validation continuation only |
| Free open nonaxisymmetric | global observables look stable, but local `m=1` changes 73-81%; 3-grid beta pair costs 293/944/2995 s and 2.74/4.57/7.35 GiB | deferred; implementation removed |
| Fixed closed B-spline hybrid | periodic representation, exact straight spans, Bishop-frame closure, converged primal, finite-current field lines, and root example; coarse strong force `0.573` | active H1 validation; not promoted |
| Differentiation | fixed-open JVP/VJP and free-axisymmetric VJP agree with fully reconverged finite differences near `2e-10` | supported only for promoted lanes |
| Preconditioning | open separable/local sparse factor is effective; closed colored factor was effective but serves a deferred model | retain open path; archive closed result |

The optimizer residual is not the equilibrium acceptance criterion. The
failed validation cases demonstrate why support requires a staggered weak
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
- beta 25/50 remain validation-only until the complete three-grid force and
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

## 8. Completed open-mirror baseline for PR #22

R1-R5 are the completed historical open-mirror baseline. H1 in section 10 is
the only active scientific lane; its execution status is in section 12.

### R1. Remove closed and dead contracts

The former closed hybrid failed its declared same-geometry refinement gate and
was removed from the open baseline. Git history and
`benchmarks/mirror_hybrid_fixed_boundary.json` preserve that work.

Implementation status: complete and accepted by the R5 audit in section 12.

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
tolerance. R2 must preserve the final 7,000-line ceiling.

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
   corrected-cut refinement grids. A resolution change must either reach the
   promoted strong-force branch or fail explicitly; optimizer `ftol` alone is
   not a successful equilibrium. The SFLM now fails explicitly and remains a
   research fixture until a corrected-cut refinement passes.
8. Keep SciPy/JAX host boundaries in one solver module. Do not add JAXopt,
   Optax, Equinox, Optimistix, or Lineax to this backend. SOLVAX is considered
   only under the complete-replacement rule above.
9. Reduce functions over 150 lines when extraction gives a named scientific
   operation; do not split files only to satisfy a line count.

Exit: one residual per model, one shared bounded Newton driver, and one shared
primal/transpose Krylov helper; no duplicate iteration loop or solver-choice
flag; normal tests and benchmark values pass; mirror source remains below
7,000 lines. R1 and R2 together must remove at least 1,000 package lines.

### R3. Correct examples, plots, and output

1. Use `(ns,mpol,elements)=(7,6,6)` as the compact corrected-cut
   nonaxisymmetric example. Require the rotating ellipse to pass every gate;
   execute the SFLM beside it as explicitly labelled negative research
   evidence. Rebuild either refinement sequence only from current cut
   semantics; the removed pre-cut numbers are not promotion evidence.
2. Require the rotating ellipse to assert `ftol`, weak force, strong force,
   divergence, geometry, and derivative gates before writing success output.
   Require the SFLM to report its failed independent-force gates without
   writing a supported status.
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

Exit: both root examples finish within the resource gate, every supported
case asserts its scientific gates, all plots are from solved states, MOUT
round trips, and CLI plots are polished and nonblank.

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
6. all three root mirror examples in clean temporary directories;
7. MOUT read/write and `vmec --plot` smoke;
8. coverage of the retained mirror package with a 95% target;
9. final CPU timing/memory and one supplementary office-GPU parity run;
10. one batched CI review after the final push.

H1 merge budgets, measured against fetched `origin/main`:

- at most 44 changed files;
- at most 8,000 mirror source lines;
- at most 3,700 mirror-test lines;
- at most 20 public mirror names;
- exactly three root mirror examples and three nonredundant showcase figures;
- exactly four compact benchmark JSON records;
- no generated scientific output in Git except the three compressed reviewed
  showcase figures;
- PR remains draft until every required H1 gate has a documented disposition.

Exit: CI is green, branch is mergeable, final diff is reviewed, no supported
claim lacks independent evidence, and PR #22 can be marked ready for review.
The PR remains draft while any required H1 gate is active.

## 9. File and API contract

The target is at most 13 files including `__init__.py` (12 owner modules).
Keep ownership clear; merge a file only when one coherent owner remains.

| File | Owner |
| --- | --- |
| `model.py` | open inputs, state, configuration, schemas |
| `basis.py` | theta Fourier, CGL evaluation, clamped and periodic cubic B-splines |
| `geometry.py` | straight/closed-axis embedding, Bishop frame, metric, field conversion |
| `forces.py` | energy, mass-pressure relation, weak/strong/interface residuals |
| `analytic.py` | exact polynomial, rotating ellipse, SFLM fixtures |
| `splines.py` | open/periodic coefficient maps, initialization, fixed solve, tracing |
| `solver.py` | bounded globalization, Newton-GMRES, shared preconditioner |
| `exterior.py` | closed integration surface, panels, singular quadrature |
| `exterior_bie.py` | axisymmetric exterior Neumann solve and LCFS trace |
| `free_boundary.py` | coupled axisymmetric free solve and beta continuation |
| `implicit.py` | supported converged-residual tangents and adjoints |
| `output.py` | open MOUT, restart, diagnostics, and open/closed plots |

The public namespace contains only user inputs, spline states/discretization,
the three solve workflows, supported derivatives, and MOUT/plot helpers.
Result dataclasses and numerical kernels remain in their owner modules.

No coils, Biot-Savart, coil geometry, coil optimization, generic virtual
casing, toroidal WOUT compatibility shim, or second solved basis belongs in
this package.

## 10. Active and deferred scientific lanes

H1 is active on the draft PR. N1 and A1 are deferred go/no-go lanes and must
not leave public scaffolds if their bounded gates fail.

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

Status: active on the draft PR after the open-mirror release audit. Useful
algorithms were recovered selectively from Git history; the removed center-map,
free-boundary, and colored-Hessian scaffolds were not restored.

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

1. **Complete.** Basis gate: implement periodic cubic evaluation, two derivatives, cyclic
   band structure, and exact dyadic knot insertion in an isolated module.
   Test partition of unity, local support, periodic closure, JVP/VJP transpose,
   and exact geometry preservation before adding equilibrium code.
2. **Complete at 32 controls.** Geometry-only gate: exact closure of `c`, tangent, frame, and section
   through two derivatives; at least 50% low-curvature length on each leg;
   smooth curvature ramps in the returns; 90-degree difference between the
   two straight-leg ellipses; frame holonomy cancelled explicitly; positive
   clearance/Jacobian; exact leg symmetry; and exact nested knot refinement.
3. **Partial.** Circular-limit gate: compare a periodic spline circle with ordinary
   `vmec_jax` and VMEC2000 at identical physical boundary, flux, pressure,
   current, and radial/poloidal resolution.
4. **Pending.** Open-leg gate: increase straight-to-return scale on three geometries and
   compare the central halves with the promoted fixed-open rotating ellipse
   and SFLM geometry, `|B|`, field-line, and force observables. The local
   straight section is validated before the returns are trusted.
5. **Failed promotion gate.** Residual gate: diagnose the previous nonmonotone 16/32/64 strong-force
   sequence before increasing resolution. Check mapping gauge, axis
   regularity, independent staggered force, and quadrature aliasing on one
   exactly preserved geometry.
6. **Failed at the fine resource gate.** Solver gate: start from the promoted open residual/preconditioner, replace
   only longitudinal boundary conditions and metric/frame terms, and retain
   one coefficient residual. Require monotone 16/32/64 longitudinal,
   three-grid radial/poloidal, and independent quadrature refinement.
7. **Partial.** Transform gate: report current-free geometric transform separately from
   transform driven by continued on-axis current. Do not assume either sign or
   magnitude.
8. **Deferred until beta zero passes.** Finite-beta fixed-LCFS gate: beta `0,1,3,10%`, with Ilgisonis et al. used
   only for sign/scaling in its asymptotic regime.
9. **Example complete; derivatives and periodic MOUT are deferred.** The root
   example and solved-state plot were added for review before promotion because
   they expose the failed strong-force gate. Add implicit derivatives and a
   periodic MOUT only after monotone three-grid convergence.
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
| Fixed open nonaxisymmetric rotating ellipse | 100% | regression only |
| SFLM validation disposition | 100% | independent-force failure retained; not supported |
| Free open axisymmetric through 10% | 100% | regression only |
| Implicit derivatives for supported lanes | 100% | regression only |
| Open preconditioning | 100% | regression only |
| Closed hybrid fixed boundary | 70% | diagnose beta-zero strong-force floor; promotion, finite beta, and derivatives remain blocked |
| Nonaxisymmetric free disposition | 100% | compact negative evidence retained |
| API/code simplification | 100% | preserve final line and public-API budgets |
| README/docs/examples/plots | 100% | regression only |
| Packaging/CI/release audit | 85% | rerun after H1 validation disposition |

The open-mirror R1-R5 release work is complete. H1 is now active on the draft
PR and is tracked separately so its failed gates cannot alter open-mirror
promotion status. N1 and A1 remain deferred.

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

### 2026-07-16 R2 shared nonlinear and linear solve path

- Steps: moved fixed/free bound projection, Newton convergence, GMRES,
  backtracking, iteration counts, and true-residual verification into one host
  driver in `solver.py`; moved implicit primal/transpose GMRES through the same
  linear helper; removed the duplicate free-equilibrium Newton loop.
- Results: source is 6,928 lines, down 1,112 from the audited branch baseline;
  the public namespace remains 17 names. Fixed uses energy Armijo acceptance,
  free uses residual-merit acceptance, and both retain model-owned
  preconditioners and coefficient residuals.
- SOLVAX A/B: 0.8.4 and SciPy both required 9 iterations on the 315-variable
  mirror stiffness fixture, with true residuals `2.71e-9` and `2.86e-9`.
  Warm CPU times were 0.41/0.45 ms, but SOLVAX compilation cost 0.266 s and its
  nonlinear driver has neither bounds nor merit backtracking. It fails the
  complete-replacement gate, so no option or second production path is added.
- Tests: 83 normal integration tests plus both shared-driver modes pass; Ruff,
  compileall, and `git diff --check` pass. The pushed R1 full shard continues
  on one office RTX A4000; R2 is not accepted until its own full shard passes.
- Files/API: only `solver.py`, `free_boundary.py`, `implicit.py`, this plan, and
  one existing test file change; no file, dependency, or public name is added.
- Best next step: commit/push R2 after the focused continuation regression,
  then generate the two solved-state README showcases in R3.
- Open lanes: fixed axisymmetric 100%, fixed nonaxisymmetric 90%, free
  axisymmetric 95%, derivatives 97%, preconditioning 95%, simplification 90%,
  docs/examples 78%, release audit 76%.
- User input: none required.

### 2026-07-16 R2 basin audit and R3 solved-state plots

- Steps: imposed explicit self-similar end cuts from the LCFS sections;
  started supplied-field finite-current states in the shared Newton/Krylov
  path; removed decorative free-example field curves; made `plot_mout` trace
  lines from saved Cartesian fields; reran both root examples.
- Results: the medium rotating ellipse passes with variational `1.72e-16`,
  weak `1.69e-16`, strong force `0.04178`, divergence `6.61e-15`, and
  adjoint/fully reconverged finite-difference error `5.01e-10`. The corrected
  SFLM reaches variational `3.10e-16` but fails strong force at `1.089`; it is
  now research-only. The beta scan independently solves 0, 1, 3, 10, 25, and
  50%, with center radius `+7.52%` and axis field `-22.94%` at 50% relative
  to vacuum; 25/50% remain research because pointwise-force gates fail.
- Tests: 86 normal mirror tests pass with six full tests deselected in 163.69
  seconds on Apple CPU; both examples complete at `ftol=1e-12`; Ruff,
  compileall, and diff checks pass. A clean full shard for `e02dcea7` is
  running unattended on office GPU 0. The earlier R1 attempt ended in an XLA
  allocation failure while both GPUs were occupied and is not physics
  evidence.
- Files/API: no module, dependency, or public name was added. Source remains
  6,994 lines. Reusable cut enforcement and field-line rendering live in
  `splines.py` and `output.py`; examples only orchestrate public workflows.
- Best next step: finish R3/R4 benchmark and documentation corrections,
  simplify both examples toward the line target, then perform the clean R5
  audit after the office full shard reports.
- Open lanes: fixed axisymmetric 100%, fixed rotating-ellipse 92%, SFLM
  research 55%, free axisymmetric 97%, derivatives 97%, preconditioning 95%,
  simplification 92%, docs/examples 90%, release audit 78%.
- User input: none required for R1-R5.

### 2026-07-16 R3/R4 example and documentation simplification

- Steps: deleted both example-private plotting implementations and routed all
  retained figures through `plot_mout`; replaced the private CSV table with a
  compact JSON summary; updated nightly example contracts; removed stale
  SFLM, free-adjoint, exterior-module, and code-size documentation claims.
- Results: the fixed example is `322 -> 211` lines and the free example is
  `298 -> 180`. Both rerun from the repository root: fixed completes in about
  90 seconds and the six-state beta scan in about 9.5 minutes. The two README
  figures use only standard MOUT outputs, are visually reviewed, and are 307
  and 182 kB.
- Tests: 87 normal mirror tests pass with six full deselections in 155.46
  seconds on Apple CPU. Strict Sphinx, Ruff, compileall, output pixel checks,
  and `git diff --check` pass. The exact `e02dcea7` full shard remains active
  on office GPU 0 with `RUN_FULL=1`.
- Files/API: the branch is exactly 42 files against main, has exactly two root
  mirror examples and two tracked mirror figures, retains 17 public names,
  and has 6,996 mirror source lines. No source file or dependency was added.
- Best next step: accept or diagnose the office full shard, rerun the final
  source commit in a clean tree, then execute R5 packaging, install, CLI,
  coverage, CI, and final-diff gates.
- Open lanes: fixed axisymmetric 100%, fixed rotating-ellipse 92%, SFLM
  research 55%, free axisymmetric 98%, derivatives 97%, preconditioning 95%,
  simplification 97%, docs/examples 98%, release audit 82%.
- User input: none required.

### 2026-07-16 R5 release audit

- Steps: restored JIT for all six full-physics tests, limited direct Newton to
  its measured local basin, aligned free-boundary observable tolerances with
  the canonical benchmark, rebuilt both examples, and audited CI, packaging,
  documentation, plots, API, file counts, runtime, and memory.
- Results: the rotating ellipse remains unchanged at variational
  `1.72e-16`, strong force `0.04178`, divergence `6.61e-15`, and reconverged
  derivative error `5.01e-10`. The free medium-to-fine maximum observable
  changes are `1.28e-3` through 10% beta and `1.11e-2` at research-only 50%.
  A larger exploratory grid approached the 8 GiB gate and was stopped rather
  than becoming a release requirement.
- Tests: 87 normal tests pass in 185.62 s on Apple CPU. Five full tests pass in
  one 23:59 office RTX A4000 shard and the corrected sixth passes in 14:36;
  three full spline tests take 40.73 s locally. Both clean temporary examples
  pass in 9:02. Strict Sphinx, pre-commit, compileall, wheel/sdist build,
  clean-venv import/console smoke, fixed/free CLI plots, and pushed CI pass.
- Files/API: final budgets are 42 changed files, 6,999 mirror source lines,
  3,443 mirror-test lines, 17 public names, 13 modules, two root examples, two
  compressed figures, and four benchmark records. Generated run products are
  ignored and the worktree is clean.
- Best next step: mark PR #22 ready for review and review only the supported
  fixed-open and free-axisymmetric lanes. Start N1, H1, or A1 only in a
  separate PR after an explicit go/no-go decision.
- Open lanes: every R1-R5 lane is 100%. SFLM, nonaxisymmetric free boundary,
  and the closed stellarator-mirror hybrid are closed dispositions, not
  incomplete release lanes.
- User input: none required for PR #22; post-release model selection is a new
  decision.

### 2026-07-16 H1 periodic spline representation and primal solve

- Steps: restored only periodic cubic evaluation/refinement; implemented the
  racetrack axis, Bishop frame, rotating elliptical section, periodic metric,
  closed force/divergence diagnostics, shared fixed-boundary solve dispatch,
  vacuum stream-function initialization, and RK4 field-line tracing; added one
  parser-free root example and one reusable reviewed plot.
- Results: the circular limit reaches variational ``3.66e-14``, strong force
  ``8.32e-3``, and divergence ``4.78e-15``. The default 16-control hybrid
  reaches variational ``6.74e-14``, divergence ``1.69e-14``, axis closure
  ``1.75e-15``, and ``iota=0.0856``. Its strong force is ``0.573`` and blocks
  promotion. Exact same-geometry 16/32/64 transfer on an RTX A4000 gives
  monotone strong force ``0.5733 -> 0.3556 -> 0.3325`` with all variational
  residuals below ``6.7e-14``. Longitudinal aliasing improves the result but
  is not the remaining dominant error. The tracked figure reports that
  failure directly.
- Tests: periodic partition/C2/refinement/JVP/VJP, exact straight spans,
  frame closure, 90-degree section exchange, positive nested volume,
  divergence identity, circular solve, hybrid solve, and field-line pitch all
  pass. The two jitted full solves pass in 15.96 seconds on Apple CPU.
- Files/API: no historical center map, free-boundary hybrid, colored Hessian,
  or coil code was restored. Three focused public operations were added:
  ``build_stellarator_mirror_hybrid``, ``trace_closed_field_line``, and
  ``plot_stellarator_mirror_hybrid``. ESSOS remains the sole coil/Biot-Savart
  owner.
- Best next step: run exact-transfer 16/32/64 longitudinal and three-grid
  radial/poloidal studies on office GPU, diagnose the strong-force sequence,
  and either pass H1 gates or retain the example as an explicitly failed
  validation case. Then complete circular/open-leg and finite-beta comparisons.
- Open lanes: fixed open axisymmetric 100%, fixed open rotating ellipse 100%,
  free open axisymmetric 100%, open derivatives 100%, open preconditioning
  100%, H1 basis 100%, H1 geometry 90%, H1 primal 75%, H1 validation 30%, H1
  derivatives 0%, docs/examples 90%, final audit 80%.
- User input: none required for the fixed-boundary H1 validation sequence.

### 2026-07-16 H1 refinement disposition and showcase refresh

- Steps: reran the parser-free hybrid example with 32 spline controls;
  performed exact same-geometry 16/32/64 longitudinal refinement and
  64-control radial/poloidal refinement on office; inspected and compressed
  all three README figures; refreshed terminology and metrics; and removed
  duplicated figure-save code.
- Results: the default 32-control example converges in 479 residual evaluations
  to variational ``2.36e-14`` and divergence ``3.14e-14``, with ``iota=0.0851``
  and strong force ``0.430``. Exact longitudinal refinement gives
  ``0.5733 -> 0.3556 -> 0.3325``. Increasing to ``ns=7, mpol=4`` gives
  ``0.2271`` with variational ``3.90e-16``; ``ns=9, mpol=5`` exceeded the
  30-minute state limit at 12,672 variables and 2.11 GiB RSS. H1 therefore
  fails the beta-zero absolute-force and fine-resource gates. Finite-beta and
  derivative work is deferred rather than run on an unvalidated primal model.
  A beta-zero ablation gives strong force ``0.424`` without current, ``0.158``
  with a circular racetrack section, and ``0.164`` with a fixed ellipse,
  compared with ``0.430`` for the rotating ellipse. The defect is localized
  to racetrack curvature plus longitudinal section rotation, not current. A
  nearest skew-connection projection of the sampled frame derivative changed
  the result only from ``0.42973`` to ``0.42874`` and was reverted rather than
  retain extra code without material diagnostic value.
- Tests: the 32-control example completed locally; the focused circular and
  hybrid solves pass; 20 model/output tests and strict Sphinx pass; every PNG
  passed visual, size, and nonblank checks. The API-budget CI failure was a
  stale 17-name assertion and now explicitly covers the three intended hybrid
  operations within the 20-name ceiling.
- Files/API: the branch remains at 44 changed files, 7,915 mirror-source lines,
  3,665 mirror-test lines, 20 public names, three parser-free root examples,
  three compressed figures, and four compact benchmark records. No coil or
  Biot-Savart implementation was added.
- Best next step: compare the closed metric/strong-force axial derivatives for
  the circular, fixed-ellipse, and rotating-ellipse racetracks, especially the
  radial covariant force through the return transitions. Continue H1 only if a
  bounded correction makes the beta-zero three-grid gate pass;
  otherwise retain the current hybrid as an explicit validation example and
  close the lane without derivatives or finite-beta claims.
- Open lanes: open fixed/free physics, derivatives, and preconditioning 100%;
  H1 basis 100%, geometry 100%, primal 80%, validation 55%, derivatives 0%;
  docs/examples 100%; final audit 85%.
- User input: none required for the bounded beta-zero diagnostic.

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
