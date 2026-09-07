# VMEX research plan

Authoritative plan, consolidated **2026-09-06** by an independent review of the
three proposals that preceded it: [#274](https://github.com/uwplasma/vmex/pull/274)
(closed), [#282](https://github.com/uwplasma/vmex/pull/282) (merged: its
README, validation-page and evidence edits are on main) and
[#283](https://github.com/uwplasma/vmex/pull/283) (merged as this document).
Base: main after those merges, 0.8.1. It is a planning change: no equilibrium
algorithm, derivative guarantee or performance claim is promoted here.

**Release hold.** No tag, version bump or publication date is scheduled. One
release follows the gates in §8 and §9.

## 0. How to use this plan

An agent resuming this work reads §1 (decision), §3 (technical decisions),
§9 (phases and the PR list), §10 (dispositions), §11 (environment) and the
last entries of §12 (logbook), in that order. It then checks the remote PR
state and dirty worktrees, and continues at the first unmet gate. After each
implementation PR it appends one logbook entry in the format of §12 and
updates the affected gate in place. It does not rerun the historical
inventories, launch a multi-day polish, or add a work package because another
code has a feature. The evidence behind every number below is in
[`benchmarks/review_20260905.json`](benchmarks/review_20260905.json)
(`second_review_20260905` and `focused_review_20260906` blocks), the
committed benchmark records, and the sources linked in the text.

## 1. Decision: fewer promises, stronger results

VMEX is the fast, differentiable, VMEC-compatible equilibrium solver with an
independent continuous force certificate and certified implicit derivatives.
Keep the VMEC-compatible solve as the reliable starting point for toroidal
research. Concentrate implementation on **correct acceptance, a demonstrably
better force-balance formulation, and time to a validated optimized design**.
The high-order method deserves one bounded recovery experiment with a
decision, not an expanding family of solvers, coordinate systems or long runs.

The strongest near-term product is a small, documented equilibrium and
optimization API with trustworthy status and reproducible CPU/GPU evidence.
The ambitious result remains accurate native 3-D force balance with useful
implicit derivatives. The two are separated: ordinary boundary optimization
does not wait for polishing, anisotropic mirrors or a coordinate rewrite.

| Priority | Deliverable | Owner | Exit decision |
|---|---|---|---|
| P0 | Honest solve/derivative status, reconciled PRs, and a repository that is fast to test | VMEX | Failed nonlinear or linear solves cannot silently supply certified gradients; every PR lane under 25 minutes; no stale claim on main |
| P1 | One physical force contract and a small benchmark matrix | VMEX | Native accuracy, discretization, reconstruction and solver error are distinguishable; one residual-versus-resolution figure exists |
| P2 | Formulation and solver decision for the high-order lane | VMEX physics; SOLVAX iteration primitives | A written verdict from E1–E3 and the resolution ladder; promotion only on demonstrated accuracy |
| P3 | Faster valid gradients and one validated optimized design | VMEX + SOLVAX; ESSOS coils | Three-way gradient agreement; a feasible design at stated accuracy with measured total time and memory |
| P4 | Reproducible comparisons, documentation, code size and the publication package | VMEX | Claims survive independent reruns; documentation mass down; release scope and remaining PR dispositions explicit |

P0 → P1 → P2 is the force-balance decision path. P3 proceeds after P0 on the
ordinary solver and depends on P2 only for claims about polished designs.
P4 runs throughout. There is no dependency on a distributed nonlinear solver.
A negative P2 result does not fulfil the 3-D accuracy goal by relabelling it,
and does not authorize a release with a silently reduced scope.

### What is set aside, with the criterion that reopens it

- **Distributed single-solve sharding and multi-host scaling.** Reopen for a
  named workload that does not fit one device's memory or time. Placement
  tests and independent-case ensembles stay. No equilibrium code ships
  multi-device single solves; DESC's sharding PR has been open over a year.
- **Normal-equation preconditioners, tensor kernels, promoted-state
  differentiation (old C3–C5).** Reopen after E1–E3 decide the functional.
- **Anisotropic closure, high-beta mirrors, periodic toroid–mirror hybrids,
  generalized toroidal angles, new gyrokinetic coupling.** Reopen for a
  mirror design question that needs them; DESC's released anisotropic force
  balance is the oracle then. The documented isotropic fixed/free-boundary
  mirror cases and their limits are maintained.
- **New diagnostics and downstream consumers.** Reopen for a consumer with a
  parity test.
- **Learned corrections, deflation, further Krylov variants, alternative
  equilibrium models, cross-version bit reproducibility.** Not before the
  above, and never solely because another code or preprint has one.
- **Broad transport, bootstrap, turbulence and multi-objective campaigns.**
  One validated objective and one constrained application come first.

The previous research questions remain readable in the archived plans at
[`ae0e410f`](https://github.com/uwplasma/vmex/blob/ae0e410f6ecc9bc15b66472039f755fdd6dd3ef6/plan.md)
and in the three closed proposals. Old A/B map to P0/P1, C to P2, D/F to P3,
I/J to P4; old E/G/H are the conditional lanes above.

## 2. Evidence at the baseline

Measured on 2026-09-05/06 on clean main (`ae0e410f`, then the merges of
#266, #276 and #282), on an Apple M4 and an Apple M3 Max with JAX 0.9.2 and
SOLVAX 0.20.0, and by three read-only audits (code, documentation, literature
and release pages). Shared-host timings are diagnostic samples, not rankings.

| Fact | Value | Source |
|---|---|---|
| Code | `vmex/` 63,342 lines (core 52,563 in 58 files, mirror 9,376 in 14); 1,275 top-level definitions, 93 exported; 120 public definitions (4,883 lines) consumed only by tests; ruff, mypy and the docstring guard clean | AST census, 2026-09-06 |
| Tests | 1,898 collected; 115 `full` decorators; local PR lanes with CI's own selectors: fast 544 passed in 78 s, core 106 in 6 m 15 s, implicit-response-a/b 9 and 15 in 3 m 20 s, free-boundary-adjoint 1 in 1 m 48 s, field-api 86 in 4 m, mirror-equilibrium 49, mirror-field 31; radial-basis and strong-force suites 67 passed in 620 s; Solov'ev and selected implicit tests 9 passed in 148 s | lane runs |
| CI | e-polish 42–44 min, c3d 31–44 min, parity budget 55 min; a merge touching the plan logbook or a rewritten docstring dirties every queued PR | run logs, 2026-09-05 |
| Where the force error lives | shaped tokamak, VMEC state lifted to the continuous basis: near axis 977 N m⁻³, bulk 163, edge 409; after polish 67 / 147 / 284 | [`benchmarks/polish_force_error_2026-09-03.json`](benchmarks/polish_force_error_2026-09-03.json) |
| What the polish improves | bounded `eps_F` L2 7.1×, dimensional volume L2 1.61×, `<\|F\|>/<\|∇p\|>` 1.32×, near-axis dimensional L2 14.5× | same record |
| What the polish costs | 80 Gauss-Newton steps, 47,308 CG iterations, 463 s, 148 unknowns; solver flag false, certificate true | [`benchmarks/strong_force_cases_m4.json`](benchmarks/strong_force_cases_m4.json) |
| W7-X polish setup | certificate 3.24 GiB; chart 16.6 GiB and 1,751 s before any solve | [`benchmarks/polish_memory_w7x.json`](benchmarks/polish_memory_w7x.json) |
| Frozen operator probe | 156×17 real-MHD operator after scaling: cond(J) ≈ 7.15e3; CG/diagonal-CG/LSMR 31/30/35 iterations at damping 1e-3; exact-Hessian versus Gauss-Newton relative curvature 2.08e-4 | `focused_review_20260906.frozen_operator` |
| Same-deck VMEC++ 0.7.3 parity | tokamak: R/Z relative L2 below 1.2e-15, lambda below 3e-14, iterations 158/159; finite-beta QA: R 3.73e-6, Z 1.79e-5, lambda 3.87e-4, iota 4.06e-7, iterations 730/775 | `focused_review_20260906.same_deck_comparison` |
| Native DESC | tokamak WOUT, L=12/M=6/N=0, converged in 28 iterations; `<\|F\|>/<\|∇p\|>` 8.06e-5; the earlier lifted-WOUT number 7.13e-2 is a different norm on a reconstructed object | `focused_review_20260906.native_desc` |
| Optimization cost | #266's QA profile: 7.63 s per value/gradient call, 0.35 s VMEC loop, 3.86 s Newton refinement, 3.06 s adjoint; a max-mode-2 run 265 s | #266 |
| Stale claims still on main | withdrawn 26-fold number in `CHANGELOG.md:57`; `docs/reference/performance.rst:899–953` cites `gpu_office.json`, `gpu_office_cmdbuf.json` and `benchmarks/traces/`, none of which exist; two benchmark records and three test docstrings cite the retired "31.2-R" numbering | grep |
| Test hygiene | four modules flip `jax_disable_jit` and never restore it: `test_scaling.py:131`, `test_cli_freeboundary.py:50,65`, `test_tracing.py:42`, `test_optimize.py:63,329` | grep |
| Vocabulary | "sharding" in code and docs means single-device placement; no `Mesh` or `NamedSharding` exists | grep |
| #277 | its e-polish lane fails only on CI (JAX 0.11.1, Python 3.12: 1 failed, 226 passed in 2,519 s); the same test passes on JAX 0.9.2 on two machines | run 34005068178 |

**The field, checked 2026-09-06.** VMEC++ 0.6.0–0.7.3 shipped LASYM (fixed
and free boundary), multigrid in mpol/ntor, Python-driven iteration, an opt-in
Enzyme AD build, an exact force Jacobian and an O(1) boundary adjoint with a
SIMSOPT wrapper; its `ftol` is a solver metric, not a physical residual
([releases](https://github.com/proximafusion/vmecpp/releases),
[PR 573](https://github.com/proximafusion/vmecpp/pull/573),
[PR 581](https://github.com/proximafusion/vmecpp/pull/581)). DESC 0.17.3 has
QR reuse across the Levenberg–Marquardt sweep and exponential spectral
scaling ([Jang, Conlin, Landreman](https://arxiv.org/abs/2509.16320)); its
least-squares default is QR with SVD as the accuracy reference; anisotropic
pressure has been released code since v0.8.0; master pins `jax<0.10`; its
AD-versus-FD tolerance is `rtol=1e-2`. Thun, Merlo, Conlin, Panici and
Böckenhoff (Nucl. Fusion 66, 2026, [arXiv:2507.03119](https://arxiv.org/abs/2507.03119))
define `F_norm = |J×B−∇p| / ⟨|∇|B|²/2μ0|⟩`, show VMEC's normalized residual
rising to about 1e-1 below ρ = 0.15 even at 2,048 surfaces, and show that a
truncated spectral basis regularizes the problem, so residual floors tighten
only with spectral resolution. GVEC ([JOSS 11(120), 2026](https://joss.theoj.org/papers/10.21105/joss.09670))
minimizes the energy by gradient descent on radial B-splines and assembles
mode-separated banded radial preconditioners (`mhd3d_evalfunc.F90`). SPECTRE
([arXiv:2607.27135](https://arxiv.org/abs/2607.27135)) succeeds SPEC for
topology comparisons. A DESC-based QA free-boundary paper was withdrawn in
2026 after its equilibria proved inconsistent with their bootstrap currents
([arXiv:2605.02139](https://arxiv.org/abs/2605.02139)).

## 3. The three technical decisions

### 3.1 Functional and solver for the high-order lane

**First question: is the lane minimizing the right functional in enough
directions?** In `core/polish.py::_strong_collocation_residual` the residual
packs signed radial and helical force densities with `2*abs(sqrt_g)/normalization`.
Its squared norm is not the Cartesian volume L2: the Jacobian is squared,
Gauss weights are absent, and a two-component nonorthogonal formulation needs
the full Gram metric. Derive and implement the reference residual whose
squared norm is `sum_q w_q |sqrt_g(q)| |F_cart(q)|² / F_ref²`, with
square-root quadrature weights and a declared physical scaling, and use that
one definition for optimization, stationarity and Hessian products.
`make_strong_structured_chart` eliminates Z directions as a gauge by a
poloidal reparameterization that divides by `Z_theta`, which can vanish, and
its layout passes through the low-order prolongation image, so local
B-splines do not automatically give independent high-order freedom.
Construct a tiny reference with independent native R/Z/lambda coefficients,
fixed-boundary constraints, exact axis regularity and an explicit gauge
quotient; check rank, nullspaces and allowed perturbations as h and p grow.

**Second question: which operator.** Three operators are in play:

| Problem | Operator | Implication |
|---|---|---|
| Ordinary discrete equilibrium `F(u,p)=0` | Newton uses `F_u v`, not necessarily a symmetric energy Hessian after scaling and constraints | the existing implicit refinement already uses this route; audit its eligibility (P0) |
| High-order least squares `min ½‖r‖²` | Gauss-Newton uses `JᵀJ`; exact Newton uses `H = JᵀJ + Σ r_i ∇²r_i`; the polished implicit derivative needs `H` | the force is second order in the geometry (B carries first derivatives, J second), so `JᵀJ` is a fourth-order operator whose conditioning grows as h⁻⁴; the tokamak record's 590 CG iterations per step is that scaling, and another preconditioner for `JᵀJ` is the wrong target |
| Discrete energy minimization `min W`, `W = ∫(B²/2μ₀ + p/(γ−1))√g` | `∂W/∂c` is the weak-form force from one reverse pass; `∂²W` is second order and symmetric, positive on stable equilibria; its action is one forward-over-reverse pass | Hirshman and Betancourt's block-tridiagonal radial preconditioner per (m, n) (J. Comput. Phys. 96, 1991) was built for this operator and is banded on splines; valid only after E1 shows the differentiated constrained energy yields the intended force and closure |

`polish_driver._gauss_newton_polish_lane` passes variable scaling to SOLVAX
but no physics preconditioner or admissibility callback. That gap is not
filled by another Krylov variant; it is decided by the experiments below.
`cond(JᵀJ) = cond(J)²` does not by itself predict CG iteration counts, and
aggregate nonlinear iteration counts do not estimate a condition number; the
probe numbers in §2 are for a 17-unknown Solov'ev lift, not the 3-D problem.

**Experiments, in order, each with a kill rule.** Budgets: reference
matrices ≤ 2 GiB, diagnostic jobs ≤ 10 minutes, nonlinear demonstrations
≤ 30 minutes including setup, enforced by the experiment runner because AUTO
does not interrupt. Fixed boundary, stellarator symmetry, prescribed iota,
GAMMA = 0 until the closure audit in P1 admits more. Failed and capped
attempts are saved as such.

- **E1, functional consistency (one day).** Compare AD `dW[c]v` with complete
  virtual work, including lambda (or displacement at fixed field-line labels),
  closure and boundary terms. Fixed-angle `∂x/∂c v` alone misses lambda work;
  see the [VMEC variation](https://princetonuniversity.github.io/STELLOPT/VMEC.html#theory).
  On Solov'ev and the shaped tokamak, separate quadrature-refinement error
  from the 1e-10 AD/JVP/VJP consistency bar; audit rank and the gauge quotient.
  An unexplained, quadrature-converged work mismatch stops the energy route;
  diagnose missing terms or quadrature first, before interpreting solver failure.
- **E2, dense reference step (one week).** Assemble `J` and the constrained
  reference `H`, retaining independent R/Z, on the frozen tokamak and one affordable finite-beta QA
  linearization; compare spectrum and rank, augmented QR/SVD reference steps
  with the actual scaling, and the physical step, cost reduction and linear
  residual, not only coefficient norms. Budget residual rows × coordinates ×
  dtype size plus factorization workspace; 3,000 unknowns alone do not bound
  Jacobian memory. Measure assembly/factorization costs rather than extrapolating. Pass: the independent
  certificate improves by an order of magnitude on the QA target. Fail: the
  obstruction is representation, reachability or closure; no Krylov work
  follows. **Result 2026-09-07: the shaped tokamak passes, QA does not.**
  Converge the toroidal sampling before reading any QA number: six planes
  under-resolves this nfp=2 deck and inverts the chart comparison.
- **E3, variational Newton (one to two weeks, only if E1 passes).** Newton on
  `∂W/∂c = 0` with matrix-free Hessian products, MINRES or CG inner solves at
  Eisenstat–Walker forcing, the banded per-(m, n) preconditioner assembled
  from the second variation with axis, metric and closure terms, trust-region
  acceptance with geometry rejection, and the strong certificate as judge.
  Pass: the certified tokamak value at fewer than 50 Hessian products per
  step and fewer than 20 steps, and h/p convergence on Solov'ev at the
  degree's order. Diagnose negative curvature at a certified stable equilibrium;
  an indefinite off-root Hessian alone is not a kill rule (see the
  [trust-region reference](https://docs.scipy.org/doc/scipy/tutorial/optimize.html)).
  Kill on failed force/geometry convergence or no 5× operator reduction against E2.

**Decision tree.** If even the trusted dense step cannot improve the
independent force, investigate representation, closure or admissibility and
stop tuning Krylov iterations. **This branch is now taken for 3-D**: E2's QA
run reaches 2.31x against a 10x gate, so representation, closure and
admissibility come before any 3-D solver or chart replacement work. If the dense step works but no iterative solve
does, the 3-D solver is dense LM with an explicit memory-bounded resolution
cap. If E3 passes, it is the scalable mode and E2 the reference. If both work
but cost is excessive, profile the demonstrated bottleneck. If the bounded QA
recovery fails, the lane keeps experimental status and the logbook records
the obstruction. No W7-X run and no global coordinate change without that
diagnosis.

### 3.2 Coordinates and the axis

**E1 finding (2026-09-07): fix reachability before Newton tuning.** On the
shaped tokamak, the actual structured chart reduces sampled normal-motion
rank from 36 to 20 and cannot move the vertical extrema normally, although
those motions exist in the full constrained layout. This is a restriction of
physical directions, not solely a coordinate gauge. E2 must retain independent
R and Z in its reference; compare any replacement chart against that space.
The prescribed-profile energy identity passes; this does not promote E3.

Keep `ρ^|m| q(s)` with B-splines in s and quadrature in ρ, with the analytic
first and second axis limits already in `radial_basis.evaluate_regularized_mode`.
It is the DESC regularity written locally, and it removes the axis spike
(977 → 67 N m⁻³ near the axis on the tokamak). "Regularize the axis" alone is
not a new task. A ρ-uniform mesh for the finite-difference lane is not
pursued: Thun et al. show the spike at 2,048 uniform surfaces, so the cause
is the axis row (`X(js=1) = X(js=2)`, `geometry.py:22`) and the half-mesh
closure, not the spacing.

| Option | Assessment and trigger |
|---|---|
| Existing `ρ^|m| q(s)` | Retain first |
| Normalize or localize the regular basis and its constraint elimination | First candidate if E1's rank and conditioning audit identifies a basis problem; compare equal function spaces and DOFs |
| Independent geometry components or a regular normal-displacement chart | Test against the full constrained reference; an off-axis normal parameterization still needs independent axis motion |
| Smooth polar center-splines ([Jiang et al. 2026](https://arxiv.org/html/2601.17841v1)) | Conditional alternative if powers of ρ or global constraints cause conditioning or locality failure; a small basis replacement, not a second framework |
| Better nested-coordinate initialization ([Tecchiolli et al.](https://arxiv.org/html/2405.08173v1)) | Conditional on poor Jacobian or admission at strongly shaped boundaries |
| Generalized toroidal angle ([DESC #2282](https://github.com/PlasmaControl/DESC/pull/2282)) | Deferred; reconsider only when laboratory-angle geometry is the demonstrated limitation |

What changes for users now: `solve()`'s summary and the validation page
report the near-axis, bulk and edge force error separately, so the
finite-difference lane's axis error is visible without polishing. If E3
passes, the regular representation becomes a mode of the same solve, seeded
by the finite-difference lane with the VMEC mesh kept for parity and export,
and the polish driver's two routes, its preconditioner variants, AUTO pricing
and heartbeat collapse into one Newton loop and one preconditioner.

### 3.3 3-D: a resolution ladder, not a budget

The angular truncation is a floor only resolution crosses. The 3-D experiment
is the same finite-beta QA deck at (M, N) = 5, 7, 9, 11 in whichever mode
E1–E3 select, under at least three radial refinements and two angular
resolutions, with the certificate's spectral tail as the refinement signal.
Plot `F_norm` against (M, N) and against wall time with DESC and VMEC++ 0.7.3
at matching resolution on the same axes. The W7-X production polish is
replaced by this curve. Kill the 3-D lane as a product if the floor at
(M, N) = 11 sits more than 3× above DESC's at the same resolution; keep it as
research with the figure as its honest status.

## 4. P0. Acceptance, integration and the cost of working

Owner: VMEX (`implicit.py`, `polish_driver.py`, `polish_implicit.py`, the
public optimization wrappers, CI); SOLVAX owns generic true-residual reporting.

1. **Stale claims — merged in #284.** Remove the withdrawn 26-fold number from
   `CHANGELOG.md`; remove or generate the three artifacts `performance.rst`
   cites; point the two benchmark records and three test docstrings at the
   P letters; extend `tools/check_docs_prose.py` to scan `.rst` pages and
   numbers in `CHANGELOG.md`, and add a cited-path existence test for
   `benchmarks/` references. Add README ≤ 300 and CHANGELOG ≤ 200 line caps to
   the same gate.
2. **JAX policy and #277 — repaired assertion and version matrix tested; integration CI pending.**
   Keep floor 0.9.2 and head 0.11.1 on Python 3.12; compare accuracy, not bits.
   The isolated office reproduction did not reproduce the old CI assertion;
   the two XLA flags are not a diagnosed cause. The test now checks the public
   derivative stationarity bar, retaining the tight MHD fixture and both
   physical acceptance and derivative duality/Taylor checks. See §12 for
   measured versions, scaling and attained stationarity. Do not repeat the
   invalid `env-0.8.0` setup or claim that a missing-pytest log is numerical evidence.
3. **CI tiering — hosted 25-minute ceiling passed; integration/nightly validation pending.** Every test that runs a polish or
   a free-boundary implicit adjoint moves to `full`; `test_polish_preconditioner.py`
   splits into Gauss-Newton (PR), homotopy (nightly) and linear (PR);
   `test_run_options.py` exercises directive parsing against a mocked driver
   and keeps one real solve; the campaign-class modules in `pr-parity-a1`
   move to nightly; the four `jax_disable_jit` leaks use the restoring
   pattern of `test_freeboundary.py:801–808`.
4. **Ordinary derivative eligibility.** `implicit._refined_state` applies a
   matrix-free Newton/GCROT refinement but may return a best-effort nonroot.
   Gate derivative eligibility on the actual nonlinear equation, admissible
   geometry and true linear residual; a small adjoint residual at an
   unconverged primal certifies nothing. Preserve a value-only fallback with
   explicit status.
5. **Three statuses for a polished gradient.** Physical force acceptance,
   stationarity `g = Jᵀr`, and the true residual of the stationarity
   derivative solve; host and `jit`/`vmap` callers get the same status;
   transformed NaN rejection never becomes a finite best-effort gradient in
   an optimizer.
6. **AUTO wording.** The CLI describes a hard ceiling; the code prices
   anticipated Gauss-Newton work after setup and does not interrupt at
   `--polish-budget`. Record setup, admitted work and elapsed time separately;
   an enforced timeout needs its own tested cancellation contract.

**Gate:** real converged and failed-root cases under eager and JIT, both
derivative directions, parameter updates; true residuals recomputed outside
iterative recurrences; no missing-fixture blanket skips; every PR lane under
25 minutes on the shared runners; #277 green on both JAX versions; no stale
claim grep-able on main. Coverage and physics tolerances preserved.

## 5. P1. One physical contract and a small matrix

Owner: VMEX. Reuse the certificate, the analytic oracle and the benchmark
runner; consolidate them instead of adding a scoring implementation.

**Contract.** Record boundary, axis convention, NFP/LASYM, units, total
toroidal flux, pressure and prescribed iota or current functions, GAMMA and
mass closure, radial coordinate, reference field and length; hash the deck
and the native state; compare physical functions, not similarly named
coefficient arrays. A projected state, a nonlinear solve and a WOUT
reconstruction are different objects. Report Cartesian `F = J × B − ∇p` with
converged quadrature: dimensional volume L1/L2/L∞, `mean(|F|)/mean(|∇p|)`
where meaningful, the magnetic-pressure normalization, and a fixed physical
reference `B_ref²/(μ0 L_ref)` when a normalization vanishes. Always report the
axis neighbourhood, bulk and edge, the whole volume, and one named literature
window, stating whether it is in `s` or `ρ` (`[0.1, 0.99]` differs between
them). The *published* residual is Thun's `F_norm` with DESC's volume average
over s ∈ [0.1, 0.99]; the bounded `eps_F ≤ 2` is acceptance only. Certify on an
off-grid mesh with independently evaluated derivatives and radial and angular
refinement; two paths that share the derivative or normalization under test
are not independent. Sampled Jacobian orientation does not prove global
injectivity; also check boundary/interior geometry and crossing diagnostics.

**Matrix, expanded only when a gate needs it.**

| Case | Purpose and acceptance evidence |
|---|---|
| Closed-form Solov'ev (`tests/test_strong_force_solovev.py`) | Separate projection error from recovery of the same solution from perturbed coefficients; radial h/p convergence at the degree's order |
| `input.shaped_tokamak_pressure_polished` | Same-input VMEC++/VMEC2000 parity; native high-order force and gradient test; E1–E3's first case |
| `input.nfp2_QA_smooth_beta` | First genuine finite-beta 3-D recovery; native DESC with matched boundary and profiles in the same norm; the ladder deck |
| Existing vacuum and LASYM small decks | Vanishing-denominator behaviour, orientation, Fourier families, ordinary implicit derivatives |
| One prescribed-current deck and one small NESTOR/ESSOS case | Ordinary solver closure and free-boundary interface regression; admission to polishing requires its own closure proof |

Audit `phip`, `chip` and pressure updates before admitting NCURR = 1 or
finite-GAMMA/mass constraints to the high-order lane; freezing flux or profile
arrays while geometry changes must not silently solve a different problem.
W7-X is a later stress case, not the first recovery test.

**Deliverable.** One committed script, one hashed artifact, one figure:
residual versus `ns` for the VMEC lane and versus spline count and degree for
the regular representation, on the D-shape, the Mb = Nb = 12 W7-X and precise
QA, with DESC and VMEC++ points on the same axes. Where a native DESC or GVEC
adapter is missing, the comparison is recorded as unavailable, never filled
with a WOUT-relift score. Publication comparisons archive native states as
well as interchange files.

**Gate:** reproducible reference values with units and norms; the VMEC lane
shows its first-order slope and axis behaviour; the regular representation
shows h/p convergence on the axisymmetric cases; every number in the figure
comes from a hashed artifact.

## 6. P2. Formulation and solver

E1 in week 1; E2 in weeks 2–3; E3 in weeks 3–5 if E1 passes; the ladder of
§3.3 in weeks 4–6 on the office box. No new knobs, budgets, heartbeats or
routes are added to the polish lane while this runs. SOLVAX's existing
Krylov and globalization APIs are reused; a missing generic capability is
added there only for a reproducible VMEX need.

**Gate:** the decision of §3.1 written in §12 with numbers, and the ladder
figure with its kill criterion evaluated.

## 7. P3. Derivatives and one validated design

1. **External oracle first (one week).** On `input.shaped_tokamak_pressure_polished`
   and the QA deck, compare VMEX's implicit boundary gradient with VMEC++
   0.7.2's adjoint gradient and simsopt's analytic gradient as vectors
   (relative L2 and angle). Accept at 1e-6 relative on the tokamak and 1e-4
   on QA; a larger disagreement is a finding to report, never a tolerance to
   widen. Pin the VMEC++ wheel and source; VMEC++'s Hessian product is a
   centered finite difference of analytic forces, and its adjoint test
   reconverges perturbed equilibria: adopt that discipline.
2. **Taylor test with re-solves.** Independent nonlinear re-solves over a
   perturbation sequence for boundary and profile directions, both states
   solved more tightly than the derivative error measured, objective Taylor
   remainders with rates over at least four halvings (target 1.9–2.1), the
   simsopt `err < 1e-9 or err < 0.3·err_old` rule, duality at 1e-6,
   forward/reverse agreement, and the adjoint residual as a hard gate, under
   JIT and on CPU and GPU once the CPU contract passes. This replaces the
   fixed-step FD checks; it does not duplicate them. Differentiate the
   accepted equilibrium equations, not an iteration history.
3. **One ordinary QA optimization, profiled.** Reproduce one accepted step
   with changed parameters and one complete run with fixed seed, objective
   scaling and feasibility criteria; measure compile, primal, refinement,
   linearization, adjoint or Jacobian, rejected trials and plotting
   separately. Refinement and response solves are the targets, not the
   already cheap primal. Test linearization reuse, preconditioner update
   frequency and adaptive refinement first; recycled forward Krylov vectors
   do not solve the transposed, state-dependent adjoint. Use XProf, Perfetto
   and HLO inspection when stage timing points at compilation, transfers or
   kernels. A kernel speedup that worsens accepted optimization time fails.
4. **Flagship.** Landreman and Paul precise QA from simsopt's
   `input.LandremanPaul2021_QA` and DESC's `precise_QA.py`: a reduced student
   example with a stated resolution gap, and a research run reproducing the
   target, validated by simsopt's QS ratio residual, a Boozer spectrum,
   ε_eff, ESSOS orbits and bootstrap consistency of the final equilibrium,
   with total cost including compilation. Then a tokamak shape/profile
   example and an ESSOS coil example against
   [single-stage optimization](https://arxiv.org/abs/2302.10622), reporting
   coil length, curvature, separation, normal-field error and equilibrium
   accuracy beside QS/QI; a lower penalty with infeasible coils or an
   uncertified equilibrium is not success. ESSOS owns coil fields and their
   derivatives; VMEX owns equilibrium response, interface physics and
   acceptance; MHD closure does not move into SOLVAX.

**Gate:** three-way gradient agreement published; cold re-evaluation of the
final design with force, gradient and feasibility checks, preserved restart,
readable history and measured total CPU/GPU cost against the same baseline;
the student example finishes in minutes on a documented CPU.

## 8. P4. Evidence, documentation, code size, publication

**Performance and parallelism.** A small case manifest in the existing
benchmark framework using P1's inputs and P3's optimization; VMEX, released
VMEC++ and VMEC2000, and native DESC at common physical tolerances and
separately at common input resolution; GVEC only after a matched, converged
native adapter. Pin source, wheel hash, dependencies, threads, GPU model,
precision, input hash, status, DOFs, quadrature and stopping criteria. Fresh
cold, cache-reload, warm changed-parameter, resolution-change and full
optimized-design runs; at least five warm samples and three complete runs
where affordable; spread, failures and censored timeouts reported; profiling
separate from timing; peak host RSS and device memory with definitions; an
idle host and matched thread budgets for any release comparison. One-device
CPU and office GPU first, then independent-case throughput with bounded
workers; the vocabulary is "placement", and distributed single solves stay
set aside.

**Documentation.** #282's 220-line README with a 300-line ceiling; this
plan under 650 lines; CHANGELOG at release-note grade (Keep-a-Changelog
groups, ≤ 25 unreleased lines, ≤ 200 total, older releases rolled to GitHub
releases); the validation page under the reference contracts; plan-like
sections out of `vmec2000-compatibility.rst` (690–786) and status sections
out of `mirror-geometry.rst`; the benchmark records as the single narrative
owner of every polish number; a generated `benchmarks/INDEX.md` listing file,
generator, commit, date and citing pages, with a test that fails on an
artifact absent from the index or a cited path that does not exist; the
review baseline, measurements and literature map of the archived plan in
`benchmarks/review_20260905.md` and `docs/explanation`. One owner per
document: tutorials to learn, how-to pages for tasks, reference for
contracts, explanations for derivations, validation for evidence, this plan
for unfinished work. Root prose goes from 2,228 to about 1,050 lines.

**Code size.** Move the homotopy and pseudo-arclength route
(`polish_driver.py` 62–85 and 804–1578, about 767 lines) and its
homotopy-only family in `polish.py` (`StrongModeBlockPreconditioner`, the
physical chart helpers, `strong_projection_diagnostics`, the refresh-policy
quartet, about 600 lines) to `vmex/core/polish_homotopy.py` with nightly
tests; `solver.py` reaches only `polish_legacy_solution`, and the two routes
share one helper. Delete the `vmec_jax` shim, `freeboundary_diff.py`, the
`boozer_bmnc_*` aliases, `wout_field_names` and `value_and_grad_bnormal`;
retire or fold `freeboundary_linear.py`; privatise the 120 test-only public
definitions (none are in `vmex.__all__`); split `optimize.py` into objectives
and drivers (its 1,132-line `_least_squares_implicit` is where
`test_optimize.py`'s 31 minutes live) and `plotting.py` by artifact type;
merge the duplicated `_tree_norm`, lazy-export block and matplotlib guard.
About 1,600 lines leave `vmex/` with no product path touched; four of these
need a changelog note. Track added and deleted lines, file count, installed
size and artifact bytes per implementation PR.

**Publication and release.** Paper 1 is the product: VMEC compatibility,
the certificate, three-way gradient agreement, the residual figure, the
cross-code table, timing on named hardware, archived inputs and environments,
`CITATION.cff`, and an AI-assistance disclosure; JOSS, if chosen, requires six
months of public history, a research-impact statement and that disclosure.
Paper 2 exists only with P2's solved finite-beta 3-D case, its convergence
and a time-to-accuracy comparison. Before any release: every open PR merged
or explicitly deferred; P0, P1 and P3 gates passed; any remaining 3-D
limitation carries an explicit scope decision; the supported Python/JAX
matrix, source and wheel installs, required CI, docs and examples pass;
inputs, environments, traces and native states archived; licences and
citation metadata recorded; claims regenerated from records.

## 9. Phases and the pull requests

Each phase lists the PRs an agent opens, in order, with the verification it
runs before pushing. Local verification is the merge bar; admin-merge is
permitted only when every real lane is green and only the unsigned-commit
gate or an external status is red. Commits are authored by the maintainer
with no tool attribution. Work happens in worktrees, never in the shared
checkout; one heavy local job at a time; the office box takes one heavy job.

**Phase 1, week 1: truth and cost.**

| PR | Files | Verification |
|---|---|---|
| Fix the stale claims and extend the prose gate (P0.1) | `CHANGELOG.md`, `docs/reference/performance.rst`, two benchmark records, three test docstrings, `tools/check_docs_prose.py`, a new `tests/test_cited_paths.py` | prose gate, `tools/preflight.py --static`, strict Sphinx |
| CI tiering and jit-leak fixes (P0.3) | `tests/manifest.json`, `.github/workflows/ci.yml`, the four leaking test modules, the split of `test_polish_preconditioner.py` | `python tools/test_manifest.py check`; every PR lane run locally with its selector; wall times recorded in §12 |
| JAX policy and the #277 repair (P0.2) | `.github/workflows/ci.yml` matrix, `docs/reference/performance.rst` tolerance policy, `tests/test_polish_preconditioner.py:1853` | the bisect on the office box; #277 green on 0.9.2 and 0.11.x |
| E1 functional consistency (P2) | `benchmarks/e1_functional_consistency.py`, `benchmarks/e1_*.json` with provenance | the E1 pass criterion; result in §12 |

**Phase 2, weeks 1–2: smaller repository.**

| PR | Files | Verification |
|---|---|---|
| Plan and record split (P4) | `plan.md`, `benchmarks/review_20260905.md`, `docs/explanation/*` | prose gate; nav test |
| CHANGELOG to release-note grade (P4) | `CHANGELOG.md` | prose gate with the new caps |
| Homotopy route to its own module (P4) | `vmex/core/polish_homotopy.py`, `polish_driver.py`, `polish.py`, `tests/test_polish_homotopy.py`, `benchmarks/strong_root.py`, `benchmarks/strong_polish.py` | API guard; the Gauss-Newton PR lane; nightly homotopy lane |
| Dead code, shims and test-only API (P4) | the deletions of §8; `vmex/__init__.py`; `CHANGELOG.md` notes | API guard; fast lane |
| Artifact index and cited-path test (P4) | `benchmarks/INDEX.md`, `tools/render_performance_docs.py`, `tests/test_performance_docs.py` | the new test; figure provenance test |

**Phase 3, weeks 2–3: the contract and the figure.**

| PR | Files | Verification |
|---|---|---|
| Physical force contract and reporting (P1) | `vmex/core/strong_force.py` (F_norm and window in `s` or `ρ`), `solver.py` summary, validation page | strong-force and Solov'ev suites; docstring guard |
| Residual-versus-resolution figure (P1) | `benchmarks/residual_vs_resolution.py`, artifacts, `docs/_static/figures`, validation page | figure provenance; the generator run twice with identical bytes |
| E2 dense reference step (P2) | `benchmarks/e2_dense_step.py`, artifacts | the E2 pass criterion; result in §12 |

**Phase 4, weeks 3–5: solver decision and derivative oracles.**

| PR | Files | Verification |
|---|---|---|
| E3 variational Newton prototype, if E1 passed (P2) | `vmex/core/polish_energy.py` behind a flag, `benchmarks/e3_*.py`, artifacts | the E3 pass criterion; certificate suite |
| Three-way gradient comparison (P3.1) | `benchmarks/gradient_oracles.py`, artifacts, validation page | the acceptance numbers of §7.1 |
| Taylor test with re-solves (P3.2) | `tests/test_derivative_certificate.py`, artifact, docs table | replaces the fixed-step checks it covers |
| Ordinary derivative eligibility and the three statuses (P0.4–0.6) | `implicit.py`, `polish_implicit.py`, `cli.py`, docs | implicit-response lanes; run-options lane |

**Phase 5, weeks 4–6: the ladder and the profiled optimization.**

| PR | Files | Verification |
|---|---|---|
| Resolution ladder on the QA deck (P2) | `benchmarks/ladder_qa.py`, artifacts, figure, validation page | kill criterion evaluated; office wall time recorded |
| Profiled QA optimization (P3.3) | `benchmarks/profile_optimization.py`, artifacts, `docs/reference/performance.rst` | stage timings with changed parameters |

**Phase 6, weeks 5–7: the flagship.**

| PR | Files | Verification |
|---|---|---|
| Precise QA reproduction and validation (P3.4) | `examples/optimization/precise_qa_*.py`, artifacts, validation page | the gate of §7 |
| Tokamak and ESSOS coil applications (P3.4) | examples, artifacts | the gate of §7 |

**After:** the paper-1 package (P4), then the release when §8's conditions
hold.

## 10. Open pull requests

| PR | Disposition |
|---|---|
| #277 | Merge after P0.2; the nonlinear stationarity gate is necessary and the assertion is on the wrong quantity. Its local 26-test pass on JAX 0.9.2 is not sufficient evidence. |
| #274, #282, #283 | Consolidated here; #274 closed, #282 and #283 merged. |
| #266, #276 | Merged 2026-09-06 (`52214e11`, `e6c962de`). |

## 11. Environment and runbook

- **Machines.** Apple M4 and M3 Max laptops (JAX 0.9.2, SOLVAX 0.20.0); the
  office workstation (`ssh office`, 36 cores, two RTX A4000). Use
  `~/venvs/vmex-gpu` (SOLVAX 0.20.0, JAX 0.9.2, ESSOS present) with
  `JAX_PLATFORMS=cpu` for CPU numerics, and `~/vmex_sweep/env-0.8.0` (JAX
  0.11.1) to reproduce CI; `~/stellarator_venv` has SOLVAX 0.7.3 and fails on
  current vmex. The single-stage examples need ESSOS main (which contains
  #58): clone it to `~/essos58` and prepend it to `PYTHONPATH`; the venv lacks
  `pyevtk`, so the examples finish the optimization and then fail in the VTK
  export. Filter the harmless `cpu_aot_loader` lines from office logs.
- **External references.** VMEC++ 0.7.3 wheel plus source
  `07ef6710078e78e29ccabab0443cb3ec0ad7e375`; DESC `ad105c5e525fbf26824d6cf9dde48775db0f8a2c`
  in an isolated environment (`jax<0.10`); GVEC `c0dc66fe2b9faa147c76a728e5cd053ce693d472`;
  SOLVAX `5a49926992fe1a3aebac4b8b8cb098798e977c14`; simsopt for the QS
  metrics and analytic gradient.
- **Reproduction commands** (from the checkout; review probes use float64):

```bash
VMEX_COMPILATION_CACHE=disabled JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_radial_basis.py tests/test_strong_force.py
VMEX_COMPILATION_CACHE=disabled JAX_PLATFORMS=cpu python -m pytest -q \
  tests/test_strong_force_solovev.py tests/test_implicit_grad.py \
  -k 'not test_implicit_grad or test_solovev_gradients_vs_fd or test_adjoint_gmres_preconditioner_value'
FILES="$(python tools/test_manifest.py select pr-fast)"; JAX_ENABLE_X64=1 \
  pytest -q -n 4 -m "not full and not weekly" $FILES --durations=20
python tools/preflight.py --static
```

- **Discipline.** Never benchmark a shared checkout: pin the SHA, verify it is
  clean, alternate A/B runs. Every artifact carries the `_provenance` block
  and is regenerated twice with identical bytes before commit. A run that
  ends on an exit code without an observation is not a result. Push and base
  retargets within the same minute start two CI runs and cancel one, so wait
  on the surviving run id; deleting a merged PR's branch auto-closes every PR
  stacked on it, so retarget dependents first; under `set -o pipefail`,
  `grep -c` exits 1 on zero matches.

## 12. Execution logbook

Format for each entry: date, PR (base and head), owning gate, command and environment, result with units, time and memory,
limitation or failure, next action. Abandoned experiments stay as evidence with their stop reason. Update the gate's status in
place; do not append work packages.

**2026-09-05–06, integrated history.** Earlier plan-ordered PRs are merged; #271 was reopened as #281 after base deletion. #254
bounded cache rescans (0.028 ms/entry); W7-X certificate memory fell from 34.23 to 3.24 GiB (16.6 GiB chart). LASYM tcon audit
passed; #258 redefined spectral-tail/nestedness diagnostics; #280 pins baseline norms at 1e-10 relative/1e-12 absolute floors.
#266/#276 and review #282 merged; #283 became this plan's authority; #274 closed. #282's M3/JAX 0.9.2 probes cover VMEC++ 0.7.3,
DESC, frozen 156×17 operator and capped GVEC (§2); README 573→220 lines. Raw evidence: `vmex-review-evidence-20260906`, compact
records: `focused_review_20260906`. Clean-main fast/seven physics lanes passed. Lost parity logs were rerun in #286; office
recovery and the unreproduced #277 failure are recorded below.

**2026-09-06, P0.1/P0.3.** #284 (`417fb4fc`, base `2a0d4356`) implements the documentation gates: 73 guards and
strict Sphinx passed. #285 (`40c1792b`, base #284) restores JIT state in four modules: 73 tests passed in 396.88
s; the regression fails on the parent. Both passed CI and merged on 2026-09-07 with user-authorized admin merge;
their branches are retained.

**2026-09-06, #277 / P0.2.** Main and this plan merged at `068b2cbc`; assertion repair pushed at `6371a36b`, without changing
tolerances. The old office log only lacked pytest. On clean `db6092b7`, Python 3.12.13 / JAX 0.11.1 / SOLVAX 0.20, the original
tight MHD test passed in 333.88 s; with both XLA flags disabled it passed in 313.34 s (17 iterations, stationarity 2.598e-9,
reference 138.346, residual scale 6.83505). CI's failure remains unreproduced. The repaired selection passed 27 cases locally
(JAX 0.9.2, 228.87 s) and on office (0.11.1, 365.96 s); static checks passed. Raw logs:
`vmex-review-evidence-20260906/{277-repair-*,office-277-*}.log`; RSS unmeasured.

**2026-09-06, #286 / CI tiering.** `fix/phase1-ci-tiers` in `vmex-ci-tiers`, initially base #277; integration `2ae2a716`, split
`0bfab61d`, scoped-JIT GN `32f01c0c`. All 1,962 original cases/64 polish bodies retained; 1,974 collected. GN/linear/homotopy
share fixtures, nonlinear cases explicitly scheduled; option routing mocked with one real PR solve/export. Final static/docs: 77
guards/17.94 s. Local full homotopy 12/385.24 s, options 41/102.23 s, adjoints 15/932.33 s without skips. NCSX asset assertion
found the wrong bundle; corrected to ncsx-mgrid with a manifest-linked guard. Office linear/MHD: 96 passes on both JAX versions
(406.44/577.27 s); one test returned to GN for fixture reuse. Final PR linear: 93 passes/7.21 s locally, 16.24/18.98 s office.
Interpreted GN stopped at 115/1413.93 s; compiled GN 124/579.02 s office, 349.79 s local. Raw results/stopped attempts:
`vmex-phase1-ci-evidence`. CI 34080867732 on `d58223b4` passed: polish 5.37 min versus 43.4, misc 24.58 versus 27.7, c2 22.60;
every PR job <25 min, both JAX gates green. No solver speedup claimed. Retargeted to main and merged #277 `98049a1a`, resolving
only logbook conflicts; integration CI/Nightly 34132378608 remain pending.

**2026-09-07, #287 / E1 and cleanup.** Branch `research/e1-functional-consistency`, base #286 `6a4d560f`; measured script
`b17352c1`. Complete prescribed-profile/GAMMA=0 energy work passes off-root on Solovev (30 coordinates, 7.931e-14 discrepancy)
and shaped tokamak (56, 8.210e-13); omitting lambda leaves 3.085%/13.295%. Office JAX 0.11.1 with 128 angles gives 8.354e-13.
Structured/layout normal ranks 20/36; Z-extremum response ratios <9e-15. Raw results, commands and deletion SHAs: sibling
`vmex-e1-evidence`. Clean shaped chart: local 43.14 s/1117.7 MiB, office 43.55 s/1043.7 MiB; repeated numerical output identical
excluding time/RSS. Deleted 32 remote/73 local merged branches using ancestry or exact merged-PR heads; open
dependencies/worktrees preserved. Nightly 34132378608 exposed #280's stale example regex after a successful polish; #287 repairs
all three metric checks (real example passed/149.72 s). Its strict preflight passed 77 guards/19.00 s. E1 identifies a physical
reachability restriction; it does not promote E3.

**2026-09-07, E2 shaped reference.** `research/e2-full-rz-reference`, base #287 `4ba23d30`; clean measurement `c725c407`.
CPU/float64/cache off: `python benchmarks/e2_dense_reference.py --order 10 --angles 48 --output /tmp/e2.json` (3 steps). Full
R/Z/lambda versus structured coordinates, same ns=9, degree=3/two-span seed, frozen iota/pressure; not the fixed-current or
production-resolution solve. QR/GELSY and SVD/GELSD agree; full J rank 56, FD/JVP error <2e-9, Hessian asymmetry <4e-16.
Independent force RMS: 2194.742 → 188.549/335.259 N/m³ (full/structured), peak 863.77/905.48; positive sampled Jacobians and
certificate radial discrepancy <6e-13. Order 6/24-angle RMS: 188.558/336.526. Local JAX 0.9.2: 52.27 s/1105.1 MiB; office
0.11.1: 115.58 s/2127.8 MiB. Physical corrections agree to 2.3e-11 relative. Reference J/H: 1.25 MiB, separate from process RSS.
The earlier `7048b71b` metric depended on arbitrary SVD coordinates; physical-coefficient damping removed its platform-dependent
trajectories. Raw results and both attempts: sibling `vmex-e2-evidence`. The off-root Hessian's small negative curvature (~-276
versus maximum 3.7e8) does not alone reject Newton. SciPy supplies the independent diagnostic; SOLVAX's production GN remains
unchanged. Next: affordable finite-beta QA reference and bounded convergence, then chart/solver integration. E2 is not passed
for QA; E3, a1 routing, P0.4, fixed-current closure and release remain gated. Integration CI is pending.

**2026-09-07, E2 on finite-beta 3-D QA.** Branch `research/e2-qa-reference`,
base #288; clean measurement commit `e9412c93` (`measurement_dirty` false).
`benchmarks/e2_dense_reference.py` gains a `qa_state` seed from
`input.nfp2_QA_smooth_beta` and field-period sampling; one plane reproduces the
axisymmetric grid and weights exactly, and the shaped case re-ran bit-identical
(`188.549`/`335.256`, FD `1.575e-9`), so #288's numbers stand. Seed resolution
is a flag because the representation is what E2 varies. CPU/JAX 0.9.2/float64,
`--seed-ns 13 --order 8 --angles 32 --steps 5`; every run passed the script's
own FD, Hessian-symmetry, QR/SVD and geometry gates.
Toroidal convergence at 2 spans, full R/Z certificate gain: 1.373x (6 planes),
2.291x (8), 2.302x (10), 2.306x (12), 2.306x (16). **Six planes under-resolves
this nfp=2 deck**; readings below use 12. Radial refinement at 12 planes:
2 spans (152 coordinates, initial 167752 N/m^3) 2.306x full versus 1.918x
structured; 3 spans (187, 135583) 1.820x versus 1.588x; 4 spans (222, 128655)
1.694x versus 1.515x. Structured columns are 112/137/162, so the chart drops
26-28% of the constrained space, matching E1's rank finding in 3-D.
Readings: retaining independent R/Z helps at every refinement, by 12-20%, so
E1's reachability result holds in 3-D; refining radially lowers the initial
force error but also lowers what a bounded step recovers; the minimum signed
Jacobian stays 0.0094 at every seed, against 5.84 on the tokamak. **E2 is not
passed on QA**: the best gain anywhere is 2.306x against the order-of-magnitude
gate, so no chart replacement, E3 promotion or Krylov work is justified by this
evidence. Next: representation, closure and admissibility, starting with the
0.0094 seed geometry, not a solver. An earlier 6-plane sweep suggested the
chart advantage vanished under radial refinement; that was an under-resolved
artifact and is superseded by the 12-plane numbers above. Process RSS reached
13.0 GiB at 4 spans / 8 planes while reference matrices stayed under 18 MiB, so
the 2 GiB matrix budget does not bound the job. The verdict is version
independent: Python 3.12 with JAX/jaxlib 0.11.1, numpy 2.5.3 and scipy 1.18.1
reproduces both 12-plane readings exactly (2.306x/1.918x and 1.694x/1.515x),
and E1's Solov'ev work discrepancy comes back 8.014e-14 against the floor's
7.931e-14, which is round-off at that magnitude. JAX 0.11 requires Python 3.12
or newer, so the floor environment cannot install it; the persistent head
environment is `~/local/venvs/vmex-head`. Raw JSON and logs, both versions:
sibling `vmex-e2qa-evidence` (`floor/` and `head/`).
