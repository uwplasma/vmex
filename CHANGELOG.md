# Changelog

Release notes are mirrored from the GitHub releases, which carry each release
in full. A number appears here only where a committed artifact backs it, and
`benchmarks/INDEX.md` lists every benchmark artifact with its generator, the
revision it was measured at, and the pages that cite it.

## Unreleased fork changes

Both scalar examples share setup and public accepted-state optimizers; separate qualification fingerprints shared code.
Inputs/coils live in `examples/single_stage_support/data/`, geometry tests in `tests/`,
and historical records in `benchmarks/single_stage_provenance/`. Constraint settings are per-case;
physics defaults and input bytes are unchanged. Upstream root anchoring receives field parameters and failure policy explicitly.

## 0.11.2 - 2026-09-24

A compilation-cache directory set through `JAX_COMPILATION_CACHE_DIR` or
`VMEX_COMPILATION_CACHE_DIR` is now split by machine, like the default one
(#449). On clusters, a shared cache path was read by nodes with different CPU
features; XLA rejected the other node's executables ("Target machine feature
... is not supported on the host machine") and recompiled on every run.

## 0.11.1 - 2026-09-23

Free-boundary trials are anchored on the coupled root their gradients
differentiate, the exterior field is accurate next to the plasma surface,
optimization results are the states the objective scored, and the
optimization examples were re-tuned against a five-minute budget. See the
GitHub release for the full notes.

### Fixed

- **Free-boundary values and gradients referred to different points** (#432).
  VMEC's `ftol` does not place a free-boundary solve on the coupled
  plasma--vacuum root the adjoint differentiates. Every certifiable solve is
  now Newton-anchored on that root (`refine_tol`); one that cannot be
  anchored is the new status 3, never a silent pass. A stalled restart from
  the reference gets the iterations the cold reference needed before the
  cold retry of #416.
- **`equilibrium_from_x` returned the host solve, not the refined state the
  objective read** (#427). On `li383_low_res` the mean iota differed by
  1.6e-3 relative; it now matches the objective, and so does the WOUT
  written from it.
- **Un-jitted gradients of state objectives recompiled on every call** (#438).
  A concrete solve status now takes its branch in Python instead of a
  `lax.cond`: an eager repeat of `value_and_grad` on a Solov'ev deck takes
  0.9-1.7 s with nothing compiled, against 12-15 s and three programs.
- **The exterior field from a free-boundary wout without coil currents had no
  coils** (#430): `VmecExtender.from_wout` now raises. The eager derivative
  check no longer warns "up to inf" far from the surface
  (`virtual-casing-jax>=0.0.8`, the new `freeb` floor).
- **The QI mirror hybrid example crashed** on a private import; the mirror
  and gradient examples follow the example template (#434).

### Added

- **The exterior field is accurate next to the plasma surface** (#441).
  Eager `VmecExtender` calls switch each point the direct quadrature cannot
  resolve to a target-graded rule (`near_surface="auto"`, default): 1e-12 in
  B down to 0.01 minor radii on the 2.5 % beta QA deck, where the default
  grid was off by order one. `with_graded_quadrature()` uses it everywhere,
  under `jit` too. `from_wout`/`from_state` accept `project_current` (off by
  default, #381).
- **Six single-stage examples** (#411): penalty, augmented Lagrangian, least
  squares, 0.5 % beta, and free boundary at zero and 0.5 % beta. Each meets
  every target it states in 129-275 s on the reference laptop.

### Changed

- **Cheaper free-boundary anchor and boundary-Schur adjoint** (#439). The
  anchor's preconditioner includes NESTOR's edge coupling through a Woodbury
  update, cutting its Krylov iterations 4-6x (1483 -> 310 at the optimized
  coils of the finite-beta single stage) with the anchored values unchanged
  to 10 digits.
- **Optimization example defaults** (#426): QA uses helicity [1, 2]; QI and
  QP use [1, 3], with QI at aspect 8 and an aspect weight of at least 0.01;
  finite beta is set through PHIEDGE. 18 scripts were timed under five
  minutes with a converged NS = 71 check; scripts over budget keep their
  previous defaults, and timing the remaining scripts is a follow-up.
- The implicit `max_fsq_ratio` default is `1e2`, matching every optimize
  entry point (#434); `take_gradients.py` is merged into
  `take_fixed_boundary_gradients.py`.
- Optimization movies colour each frame from a plain forward solve (#436):
  the single-stage movie takes 12.6-21.9 s cold against 57.0 s.
- Direct-path exterior-field derivatives use closed-form layer kernels
  (#441, same values to 1e-12): first `B`..`gradgradgradB` calls 6.0 s ->
  2.9 s, warm `gradgradgradB` 4-6x faster
  (`benchmarks/extender_ab_20260923.json`).
- The summary figure plots bootstrap `<J.B>` on the force-balance panel,
  computes the effective ripple on 7 surfaces, and no longer writes
  `_profiles.png`; the README shows two single-stage movies (#442).
- The mean-iota implicit derivative's 1-3 % gap to re-solve finite
  differences is the m = 1 angle-gauge drift of the solver's path, not a
  missing term (#428).
- CI: the GPU workflow is retired, the weekly and publish lanes are fixed,
  every manifest lane must run in a workflow or be local-only (#437), and
  parity lane c3d is split into c3 and d (#438). Uncited benchmark records
  are removed (#431); the docs, plan and README are consolidated (#413, #429,
  #433, #435, #440).

### Removed

- `VmecExtender.with_near_surface_continuation` and `near_surface_plan`
  (1.6-2.4 % error floor, 18.5 GB to prepare); use `with_graded_quadrature()`.

### Added

- Matched fixed/free-boundary coil-constraint benchmarks: order-16 coils,
  length, peak/mean-squared curvature, clearance and geometry verification.
  Baseline examples moved to `examples/single-stage-benchmarks/`.
- `FreeBoundaryProblem.from_loss(coil_quantities=...)` includes direct coil
  and implicit equilibrium derivatives for moving-boundary constraints.
- Qualification fingerprints record unavailable optional analysis metadata.
- Opt-in coupled-root polishing and measured LU refresh for free-boundary
  scalar optimization, with bounded dense recovery and derivative agreement
  checks before replacing retained factors.
- Public fixed-boundary accepted-state views and composed-objective acceptance
  hooks, so both scalar single-stage examples use the same optimizer and
  physical-constraint interface without accessing private solver caches.

### Changed

- Fixed- and free-boundary scalar examples share input/WOUT loading, coil
  initialization, and run options. Both default to SLSQP, fit generated or
  supplied initial coils, and accept saved fitted coils without refitting.
  Derivative verification remains a separate workflow.

### Fixed

- Fixed scalar restarts use the public `restart_from` constructor argument;
  stage-two fitting varies coil coordinates alone and preserves currents.

## 0.11.0 - 2026-09-21

See the GitHub release for this version in full.

### Changed

- **The interior field is built from the native VMEC form** (#403).
  `B^theta = (chi' - lambda_zeta)/sqrt(g)` and `B^zeta = (phi' + lambda_theta)/sqrt(g)`
  with `sqrt(g)` from the same `R`, `Z` series as the position, and a C2
  radial interpolant. On the breathing-circle oracle at ns = 41 the second
  derivative of B improves from 2.9e-2 to 2.2e-4 relative error and
  `|div B|/|grad B|` is at round-off (8.5e-18). Spectra without `lmns`,
  `phipf` and `chipf` keep the previous fitted path.
- **Minimum versions** (#410): `booz_xform_jax>=0.4.0` (its `Booz_xform`
  class rebuilds cached grids when the resolution changes and reports a
  correct `__version__`; the JAX kernel is unchanged from 0.3.0),
  `solvax>=0.21.0`, and `gkx>=1.8.0` for the turbulence extra.
- **Force-balance polishing is compared on one mesh** (#412). The example and
  README now state that the gain is concentrated near the axis (RMS force
  2.9e3 -> 61 N/m^3 for rho < 0.2) with a small volume-average change
  (2.27e-3 -> 1.91e-3), and that the summary plot's force panel cannot
  resolve it.

### Fixed

- **`surface_field_data_from_state` raised `TypeError`** for every caller
  after #403 added three keys to the live-state spectra (#409).
- **Free-boundary recovery depended on trial history** (#416). A shared
  cold-rebuild budget meant that, after unrelated rejected trials, a
  recoverable point returned the stalled restart instead of its converged
  root. Each stalled restart now gets exactly one deterministic cold retry.
- **Radial lifts accepted under-determined spline fits** (#414), inventing
  curvature in unsampled spans; they now raise.
- **Warm workflow profiles timed only the first stage** (#420); schema 2
  repeats every stage.

## 0.10.0 - 2026-09-20

The interior field's geometry table carried a spurious `sqrt(s)` on every
odd-`m` coefficient, which put `li383_low_res` surfaces 1.2 cm (R) and 5.9 cm
(Z) off at mid-radius; it now reproduces the wout table to 6.7e-16 m and all
126 adjacent surface pairs nest. Points well outside the plasma return NaN
instead of raising, the inversion starts from the best of three seeds, and a
query on the magnetic axis no longer returns NaN. Full notes in the GitHub
release.

## 0.9.1 - 2026-09-16

Optimization gradients compile on a machine that has a GPU (the host-callback
pin now applies only within one platform; jax-ml/jax#40722), the coil examples
install from PyPI (`essos>=0.17`, new `all` extra), `max_fsq_ratio` defaults to
`1e2`, and the exterior field sizes its source grid from the boundary (4.31e-07
one minor radius out on the QA wout, against 2.46e-02 before). Full notes in
the GitHub release.

## 0.9.0 - 2026-09-15

The optimization release. The two workflows users reported as slow and
inconclusive — quasi-isodynamic boundary design and single-stage plasma-and-coil
design — now converge and meet the targets they print: the refinement every
trial pays is a Newton finish through one block factorization, the implicit
adjoint is solved exactly through that same factorization, and the compiled
lanes stopped recompiling. The exterior field reports the accuracy it achieved,
every evaluation carries counters, and the CLI reads DESC inputs. It does not
close cold compile (median wall per case 23.1 s against VMEC++'s 3.35 s in the
independent `itpplasma/benchmark_vmec` corpus) or block-Jacobian assembly.

The PRs behind it and the full notes are in the GitHub release: #300, #310,
#311, #312, #333, #344 and the rest of the 36 merged since 0.8.1.

## 0.8.1 - 2026-09-02

The cold-start performance release: cold CLI, Python and optimization runs
faster than every previous release on every deck measured, from 773 XLA
programs at QA resolution to 343 (#227-#234). Full notes in the GitHub
release.
