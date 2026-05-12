# VMEC-JAX Research-Grade Roadmap

Last updated: 2026-05-12
Primary branch: `main`
Baseline release: `v0.0.7`

This is the living execution plan for making `vmec_jax` accurate, fast,
differentiable, documented, and usable by external researchers. Update it when
acceptance criteria or evidence changes.

## Working Rules

- Keep VMEC2000/VMEC++ parity as a correctness gate, not a demo.
- Keep differentiable objectives smooth enough for JAX AD; keep legacy
  non-smooth metrics as diagnostics and ranking gates.
- Keep examples SIMSOPT-like: users assemble objectives, weights, targets, and
  optimizer choices in plain scripts.
- Keep CI under 10 minutes for required jobs; move expensive validation to
  nightly/manual jobs with committed summaries.
- Commit only source, tests, docs, and lightweight curated fixtures; keep
  generated sweeps and large artifacts out of git unless explicitly curated.

## Current State

- Fixed-boundary and free-boundary CLI/API paths exist and have parity fixtures.
- Optimization examples cover QA, QH, QP, and QI with continuation/ESS policies.
- Smooth differentiable QI, mirror ratio, elongation, LgradB, aspect, iota, beta,
  volavgB, magnetic-well, DMerc, and JXBFORCE profile objectives exist in the
  workflow layer.
- GPU execution works. Small/medium optimization cases are now much closer to
  CPU after the backend-adaptive replay-bucket fix, but accepted-point tape
  replay and tape build still dominate on GPU.
- Continuation correctness is now protected by both synthetic control-flow tests
  and a real projected-boundary stage test. Repeated stage schedules such as
  ``[1, 1, 2]`` carry the optimized VMEC input forward and keep projected
  high-mode coefficients zero.
- Exact-history correctness is now protected against relaxed trial-solve drift:
  final ``input.final`` and ``wout_final.nc`` use the best exact accepted point
  when the last trial-accepted point replays worse.
- Required CI coverage is 66.60% locally on the Python 3.11 CI-equivalent
  required suite, above the current 63% gate but still far below the long-term
  95% goal.
- `solve.py`, `wout.py`, `free_boundary.py`, `driver.py`, and optimization
  modules are too large and need staged refactoring after parity gates are locked.

## Milestone 1: QI Truth And Robustness

- [x] Port the smooth differentiable QI residual into `vmec_jax`.
- [x] Add QI mirror-ratio, elongation, and LgradB objective components.
- [x] Move the legacy Goodman-style branch/shuffle QI diagnostic into source as
      an explicit non-differentiable validation utility.
- [ ] Add regression data comparing smooth QI, legacy QI, and
      `omnigenity_optimization` on the same Boozer spectra.
- [x] Make QI sweeps report both optimizer objective and legacy diagnostic so
      visually bad QI fields cannot pass unnoticed.
- [ ] Prove seed robustness with a curated multi-seed matrix: start QI from QI,
      QP, QH, QA, and a simple non-omnigenous seed; document which policies
      reliably converge. This is deferred validation, not a required PR gate,
      until the rows and Boozer contour audits are generated.
- [x] Diagnose remaining QI noisiness by one-DOF scans of Boozer/QI metrics and
      choose default resolutions/weights that preserve ranking while remaining
      differentiable.
- [x] Add and exercise bounded prefine-manifest summaries that rank candidate
      seeds by final objective/improvement, flag objective-history regressions,
      and recommend the next reviewed probe. A 2026-05-12 top-seed smoke
      probe from the optional NFP=2 QP/QI seed reduced the mode-1 QI objective
      by 5.0% with no summary regression flags.
- [x] Fix prefine acceptance so a completed, monotone, finite probe with an
      already-low final QI objective can be promoted even if a tiny
      two-evaluation smoke does not move. Re-summarizing the 2026-05-12
      seven-seed smoke now promotes `qi_omnigenity_nfp1` as the accepted
      stable-low-objective candidate (`objective_final=2.70e-2`).
- [x] Expose bounded QI-prefine ESS controls in the audit CLI and manifest
      commands (`--prefine-use-ess/--no-prefine-use-ess`,
      `--prefine-ess-alpha`). A 2026-05-12 mode-2 continuation ablation on
      `qi_omnigenity_nfp1` failed with the same NaN error with ESS enabled and
      disabled, so the next QI bug is in the mode-2 continuation/objective
      derivative path rather than ESS scaling alone.
- [x] Fix mode-continuation stage seeding so each next stage starts from the
      previous stage's optimized VMEC input instead of rebuilding from the
      original deck plus lifted parameter increments. This prevents projected
      high-order seed modes from reappearing at later stages. The failing
      `qi_omnigenity_nfp1` smoke (`1,1,2`, mode 2) now completes with ESS and
      without ESS; the mode-2 initial exact Jacobian is finite with max entry
      about `3.9e8` instead of the prior `1e120` derivative blow-up.
- [x] Guard final optimization outputs against relaxed trial-solve acceptance
      drift: the optimizer now tracks the best exact accepted-point residual,
      selects it for final outputs when SciPy's last trial-accepted point
      replays worse, and filters worse exact replay rows out of the accepted
      objective history. The bounded `qi_omnigenity_nfp1` mode-3 smoke
      (`1,1,2,2,3`) reached `3.05e-4` final QI objective in about 63 s with
      monotone exact objective history and one rejected trial-exact replay row.
- [x] Add regression coverage for repeated and higher continuation stages using
      both synthetic workflow tests and a real boundary-projection stage test.
      These tests verify that later stages rebuild from the previous optimized
      input, restart with zero increments, and do not resurrect high-order seed
      modes that were intentionally projected out.
- [x] Extend the bounded QI-prefine manifest/probe lane to include optional
      mirror-ratio and elongation penalties, so seed audits can reject or repair
      QI-looking seeds that have poor engineering geometry. A 2026-05-12 capped
      QP/NFP2 mode-1 probe with QI, mirror, and elongation terms reduced the
      field objective from `3.31` to `0.815` in two function evaluations and
      completed with monotone accepted history.
- [x] Run a tier-3 constrained seed probe comparing promoted QP/NFP2 and
      rejected QA starts. Both histories were monotone, but neither is accepted
      as robust QI evidence: the QP seed reduced the constrained scalar
      objective while still failing mirror/elongation audits, and the QA seed
      passed engineering constraints while remaining poor by smooth/legacy QI.
      The next QI step is objective/diagnostic decomposition, not larger caps.

Acceptance:

- Best QI has poloidally closed Boozer `|B|` contours, low legacy QI diagnostic,
  mirror ratio at target, acceptable elongation, `abs(iota) >= 0.41`, and aspect
  near the chosen target.
- Seed-robust QI is not accepted from scalar objectives alone; every accepted
  seed lane needs the numerical gates above plus visual Boozer `|B|` contour
  review or an equivalent committed diagnostic artifact.

## Milestone 2: Physics Objectives And Diagnostics

- [x] Promote differentiable `DMerc` from diagnostic/parity into an objective
      with objective-routing and JAX AD tests.
- [x] Add differentiable `J dot B`/JXBFORCE profile objective accessors with
      documented normalization and VMEC/wout parity tests.
- [x] Add differentiable state-derived toroidal current-profile objective
      accessors with documented normalization.
- [x] Add differentiable vector `J` and vector `B` objective accessors with
      documented normalization.
- [x] Add differentiable finite-beta stage-one metric/objective coverage for
      pressure/current-profile-adjacent terms, beta, volavgB, iota, aspect,
      Mercier, Redl bootstrap-current mismatch, and quasisymmetry/omnigenity
      residual wiring.  The finite-beta examples intentionally keep direct
      `FixedBoundaryExactOptimizer` calls visible because their stage-local
      pressure/current closures do not fit the public objective-tuple wrapper
      cleanly yet.
- [ ] Validate the finite-beta stage-one examples against converged
      reference/SIMSOPT finite-beta equilibria, including publication-quality
      Redl Boozer-geometry comparisons.
- [ ] Add physics gates for force residuals, profiles, fluxes, magnetic-field
      reconstruction, Mercier, and Boozer transforms.

Acceptance:

- Each physics objective has a source-level function, an example term, an AD test,
  a finite-difference test, and a documented normalization.

## Milestone 3: Differentiation Architecture

- [ ] Lock AD-vs-finite-difference derivative gates for QA/QH/QP/QI max_mode=1.
- [ ] Revisit the residual-root implicit layer: reduced state packing, boundary
      control embedding, lambda gauge/branch conditions, and custom VJP/JVP.
- [ ] Reduce accepted-point replay/Jacobian count in optimization without changing
      accepted equilibria.
- [ ] Add matrix-free/scalar-adjoint optimizer pathways for larger mode numbers.

Acceptance:

- Exact derivatives agree with finite differences on small cases, and optimization
  trajectories are stable without finite-difference fallback.

## Milestone 4: CPU/GPU Performance

- [ ] Profile cold and warm fixed-boundary solves by stage: input, build_static,
      VMEC iterations, residual assembly, Boozer, objective, AD replay, and output.
- [ ] Remove unnecessary accepted-point replays and cache invalidations.
- [ ] Keep CPU fast for small/medium runs; avoid regressions from GPU-specific work.
- [ ] Build a GPU-native tape/replay path that avoids excessive host transfer and
      recompilation.
- [ ] Benchmark LASYM true/false, max_mode 1/2/3, QA/QH/QP/QI, CPU/GPU.
- [x] Add real JSON comparison tooling for exact-optimizer CPU/GPU reports.
      A 2026-05-12 QH max_mode=2 Jacobian profile on commit `f0225ff` measured
      local CPU at 11.29 s for two new Jacobian points and RTX A4000 GPU at
      54.83 s. GPU was 4.86x slower overall, with the ratio tracking
      `jacobian_tape_replay` almost exactly; the next GPU optimization target
      is replay/tangent batching, not residual convergence.
- [x] Make dynamic replay bucketing backend-adaptive. CPU keeps the smaller
      bucket (`32`) because profiling showed it remains faster there; GPU uses
      bucket `128` by default. On the same QH max_mode=2 accepted-point
      Jacobian profile, RTX A4000 runtime dropped to about 15.8 s for two
      perturbed Jacobian points, only 1.40x the 11.29 s CPU baseline, with
      replay time only 1.11x CPU.
- [x] Re-profile the remaining exact-Jacobian cost after backend-adaptive
      bucketing. A 2026-05-12 local CPU smoke spent about 96% of callback time
      in `jacobian_tape_replay`, `jacobian_initial_tangents`, and
      `jacobian_residual_tangents`; an `office` cold fixed-boundary smoke still
      favored CPU (`13.80 s` versus `68.86 s` GPU). The next concrete
      performance target is tangent fusion/reuse, not fixed-boundary force
      kernels.
- [x] Reuse cached affine initial-state tangents in the scalar-gradient path.
      `objective_and_gradient_fun` now projects the reverse tape cotangent
      through the same cached tangent map as dense exact Jacobians instead of
      rebuilding an initial-state VJP each callback. A local QA `max_mode=1`
      CPU smoke improved the same-branch warm gradient callback from about
      `0.279 s` to `0.188 s`; dense least-squares Jacobian cost is unchanged.
- [x] Add detailed accepted-point preconditioner timing and re-run CPU/GPU
      exact-optimizer profiles. On `office`, QH mode-1 cold GPU tape callback
      was `9.29 s` versus `16.97 s` on the same host CPU stack, but QH mode-2
      two-point GPU tape callbacks were still `16.48 s` versus `10.34 s` on the
      local CPU. The remaining GPU bottleneck is tape-build preconditioner
      dispatch plus dense residual-tangent projection, not VMEC convergence.

Acceptance:

- Published performance tables identify when GPU is expected to win, when CPU is
  preferred, and why.

## Milestone 5: VMEC Parity And Numerical Gates

- [ ] Expand VMEC2000/VMEC++ parity gates to fixed/free, axisymmetric/non-axis,
      stellarator-symmetric/LASYM, single-grid/multigrid, and finite-beta cases.
- [ ] Compare converged equilibria rather than fragile short finite-step states
      unless the test is explicitly a solver-trace regression.
- [ ] Add efficient wout-field parity gates for geometry, profiles, B, J, iota,
      aspect, Mercier, and force residuals.
- [x] Add a required-tier bundled VMEC2000 `chipf` parity gate for QA/QH/QI,
      circular, and finite-beta shaped-tokamak wout fixtures. This locks the
      half-mesh `chipf` convention and the `chips_from_wout_chipf` round-trip
      without running VMEC2000 in required CI.
- [x] Add a required-tier bundled VMEC2000 stored-field parity gate for QA/QH/QI
      wout fixtures. The test reconstructs Cartesian `|B|` from stored
      contravariant `bsup*` fields and checks it against the same wout's
      `bmnc/bmns` Fourier representation, covering NFP angle handling, Nyquist
      evaluation, and vector-basis conversion without external executables.

## Progress Snapshot

Updated 2026-05-12 after the continuation, exact-history, and scalar-gradient
reuse push:

- Continuation correctness: 98%. Source fix is implemented and covered by
  synthetic repeated-stage tests plus a real boundary-projection stage test.
- Exact accepted-point history/output correctness: 93%. Best-exact selection is
  implemented and tested; remaining risk is rare exact-state unavailability on
  failed replay paths.
- Seed-robust QI: 78%. The tier-2 bounded QP/NFP2 seed probe improved the QI
  prefine objective by 42% with monotone history, and the constrained-prefine
  path now runs QI+mirror+elongation terms end-to-end. A tier-3 constrained
  audit showed scalar improvement is still not enough to promote a seed; robust
  multi-seed evidence and polished Boozer contour review remain open.
- CPU/GPU performance: 73%. Backend-adaptive replay bucketing removed the
  largest GPU replay regression for small/medium exact optimizations, the
  scalar-gradient path now reuses cached initial tangents, and detailed
  preconditioner subphase timing is available for the next replay/tape-build
  pass. GPU dense-Jacobian residual tangent projection remains open.
- VMEC parity and physics gates: 86%. Required-tier wout `chipf`, stored
  `B`-field, geometry/aspect, Mercier decomposition, and JXBFORCE endpoint
  parity gates now use bundled fixtures; full fixed/free/LASYM/finite-beta
  converged-equilibrium parity is still open.
- Refactor/API/examples: 83%. Examples are SIMSOPT-like and clearer, and the
  finite-beta examples now explicitly document why they use direct
  `FixedBoundaryExactOptimizer`; large solver/wout/free-boundary module splits
  remain deferred behind parity gates.
- Docs/release hygiene: 82%. Performance/discrete-adjoint docs reflect the
  current replay policy, finite-beta examples document their lower-level
  workflow, and diagnostics docs cover detailed preconditioner timing; final
  seed-robust QI and GPU-production artifacts remain open.

Overall average across these active lanes: about 85%. The continuation and
exact-history lanes are near done, but reaching a defensible 90% overall still
requires real QI robustness and performance progress, not just more tests.

Acceptance:

- CI and nightly tests catch real physics/numerics regressions without relying on
  large committed output files.

## Milestone 6: Refactoring And API Hygiene

- [ ] Continue narrow seam extractions that already have parity evidence:
      force/residual helpers, Mercier/Redl algebra, wout schema helpers, and
      optimization tuple/routing policy.
- [ ] Defer broad `solve.py` decomposition into orchestration, residual kernels,
      convergence, fallback policy, tracing, and output modules until solver
      parity and coverage have enough margin to catch regressions.
- [ ] Defer broad `wout.py` and `free_boundary.py` splits until each proposed
      seam has a targeted test and, for free-boundary coupling, a parity or
      executable-backed validation command.
- [ ] Keep compatibility imports in `vmec_jax.__init__`, but move implementation
      to smaller documented modules.
- [ ] Add docstrings and comments where they clarify physics, numerical method, or
      AD behavior.

Acceptance:

- Core modules are small enough to review, tests still pass, and public imports are
  backward compatible.

## Milestone 7: Tests And Coverage

- [ ] Raise required CI coverage from the current fast gate toward 95% in staged
      steps: 65%, 75%, 85%, 95%.
- [ ] Add targeted tests before coverage-only tests: physics gates first, branch
      logic second, I/O/schema third.
- [ ] Keep required CI under 10 minutes by using small fixtures and nightly heavy
      matrices.
- [ ] Document the testing strategy and explain which tests are smoke, unit,
      parity, physics, and performance gates.

Acceptance:

- Coverage is high because important code paths are tested, not because smoke
  tests call APIs superficially.

## Milestone 8: Docs, Examples, And Release

- [ ] Keep README short: install, CPU/GPU selection, one free-boundary example,
      best QA/QH/QP/QI plots, and reproduction commands.
- [ ] Move detailed sweeps, all-policy panels, validation plots, and equations into
      docs.
- [ ] Make every optimization example self-contained and SIMSOPT-like: define
      parameters, construct the VMEC object, build objective tuples, solve, inspect
      `result`, save/plot selected outputs.
- [ ] Keep Read the Docs full build working and fast enough to diagnose failures.
- [ ] Release only after tests/docs pass or after explicitly documenting any
      temporary release risk.

## Immediate Implementation Queue

1. [x] Finish source-level legacy QI diagnostic tests and wire comparison
   scripts to the shared utility.
2. [x] Make QI sweep summaries distinguish the smooth differentiable objective
   (`qi_raw_total`) from the true legacy branch/shuffle diagnostic
   (`qi_legacy_total`).
3. [x] Run a small QI one-DOF noise/ranking audit with the source-level
   diagnostic.
4. [x] Add the LASYM=True derivative branch to the AD-safe state-level `DMerc`
   diagnostic and keep the user-facing `DMerc` objective on that shared path.
5. [x] Add the first differentiable Redl/bootstrap-current mismatch objective
   on the state-level finite-beta path.
6. [x] Compare the new Redl residual against SIMSOPT's RedlGeomVmec path on a
   committed finite-pressure wout fixture.
7. [ ] Add subprocess-isolated RedlGeomBoozer comparison where booz_xform is
   available and stable.
8. [x] Start the first refactor with a low-risk extraction from the largest
   modules after the new tests are green.
9. [x] Add a bounded no-optimization QI seed audit using
   `qi_diagnostics_from_state` on available solved `input`/`wout` pairs. It
   compares smooth QI, legacy QI, mirror ratio, elongation, iota, and aspect
   ratio before launching a full optimization sweep.
10. [x] Prepare the first deferred seed-robust QI sweep design artifact:
    explicit ranked seeds, hard-capped QI-prefine probe settings, run commands,
    and output locations via the prefine manifest. Full QI/QP/QH/QA/simple
    optimization rows and Boozer contour curation remain open.

## Reassessment After 2026-05-12 Push

Realistic next targets for this development cycle:

1. Keep the required CI coverage gate at 63% until GitHub has clear margin,
   then raise toward 65% using fast, physics-relevant
   branch tests in `vmec_bcovar`, `vmec_realspace`, `finite_beta`, `wout`, and
   `optimization_workflow`.  Do not raise the CI floor again until GitHub's
   measured coverage has clear margin above the target.
2. Add real solved-state QI diagnostics using `qi_diagnostics_from_state` on a
   small bundled or generated fixture, then compare smooth QI, legacy QI,
   mirror ratio, elongation, and Boozer `|B|` contour quality at higher
   resolution.
3. Continue GPU optimization only where profiling points to a concrete source
   change.  Current evidence says exact tape build is subsecond in bounded CPU
   cases; the bottleneck is replay/residual linearization/VJP in
   `checkpoint_tape_state_jvp_columns` and related dynamic-basepoint replay.
4. Keep refactors small and test-backed.  Good next seams are pure helper
   extractions or schema/routing cleanup already covered by fast tests.  Broad
   rewrites of `solve.py`, `wout.py`, `free_boundary.py`, and replay
   architecture stay deferred until they have a concrete validation command and
   rollback plan.
5. Keep optional VMEC2000/SIMSOPT validation expanding, but keep required CI
   under 10 minutes by default.

Defer beyond the current cycle:

1. Full seed-robust QI from QA/QH/simple non-omnigenous starting points.  This
   needs multiple long sweeps plus visual Boozer audits, not just unit tests.
2. Large module refactors of `solve.py`, `wout.py`, and `free_boundary.py`.
   These should follow higher parity/coverage gates to avoid destabilizing the
   solver.
3. A fully GPU-native replay architecture.  The next step is narrower
   profiling-driven replay reduction; replacing the differentiation architecture
   is a larger design project.
4. Turning multi-seed QI sweeps into required CI.  Keep them manual/nightly
   until the matrix is reliable, bounded, and summarized as lightweight
   artifacts.

## Activity Log

- 2026-05-10: Replaced stale QH-only plan with current research-grade roadmap.
  Started QI validation lane by promoting the legacy branch/shuffle diagnostic
  from an example script into source and tests.
- 2026-05-10: Updated QI sweep diagnostics so `qi_legacy_total` is the actual
  non-differentiable branch/shuffle score, while `qi_raw_total` remains the
  smooth differentiable objective used by the optimizer.
- 2026-05-10: Added `examples/optimization/scan_qi_boozer_mode.py` to scan a
  selected Boozer coefficient and compare smooth-vs-legacy QI metric roughness.
  First bundled QI seed audit selected Boozer mode `(m=0, n=2)` and found both
  metrics minimized at scale `1.25`, with roughness about `1e-2`.
- 2026-05-10: Added a JAX AD gate for existing finite-beta workflow objectives
  (`VolavgB`, `BetaTotal`, magnetic well). `DMerc` remains a larger source port
  because the current parity implementation lives in the NumPy/wout Mercier
  kernel.
- 2026-05-10: Started the `DMerc` port by adding
  `mercier_terms_from_profile_integrals`, a JAX-differentiable implementation
  of the VMEC Mercier algebra once the surface integrals are known. The
  remaining port is the geometric surface-average assembly.
- 2026-05-10: Added `mercier_surface_integrals_from_realspace`, a JAX
  implementation of the VMEC Mercier `tpp/tbb/tjb/tjj` reductions from
  real-space channels. The remaining `DMerc` objective work is state wiring for
  `gpp` and `bdotk`.
- 2026-05-10: Added `mercier_bdotk_from_covariant_derivatives`, a JAX
  implementation of the VMEC jxbforce `itheta/izeta/bdotk` block from filtered
  covariant-field derivatives. The remaining hard part is deriving those
  filtered derivative channels from the state path without falling back to wout.
- 2026-05-10: Added `mercier_gpp_from_realspace_geometry`, a JAX
  implementation of the VMEC Mercier contravariant `gpp` geometry channel from
  even/odd real-space geometry. The remaining `DMerc` wiring is now focused on
  filtered covariant-field derivatives.
- 2026-05-10: Added `mercier_bsubs_derivatives_lasym_false`, a JAX
  implementation of the stellarator-symmetric jxbforce spectral derivative
  reconstruction for `bsubsu`/`bsubsv`. The next DMerc step is state-based
  `bsubs` wiring plus the LASYM=True branch.
- 2026-05-10: Added `mercier_bsubs_half_mesh_from_geometry` and
  `mercier_bsubs_full_mesh_from_half_mesh`, covering the VMEC `bss.f` radial
  covariant field assembly and jxbforce full-mesh averaging. Remaining DMerc
  wiring needs state-synthesized `rv12/zv12` geometry and the LASYM=True branch.
- 2026-05-10: Added `mercier_zeta_half_mesh_from_realspace_geometry`, covering
  the VMEC half-mesh `rv12/zv12` toroidal derivative geometry from even/odd
  channels. Remaining DMerc wiring needs state synthesis of those channels and
  the LASYM=True derivative branch.
- 2026-05-10: Added `mercier_realspace_geometry_channels_from_state`, a
  source-level JAX synthesis helper for the VMEC even/odd R/Z geometry channels
  used by Mercier. Remaining work is composing the state-level `DMerc` residual
  and adding LASYM=True derivative reconstruction.
- 2026-05-10: Added `mercier_terms_from_state`, a differentiable state-level
  Mercier diagnostic for stellarator-symmetric equilibria. It composes the
  JAX geometry, `gpp`, `bsubs`, derivative, `bdotk`, surface-integral, and
  algebra kernels. This was the base path later wrapped by `vj.DMerc`.
- 2026-05-10: Added `vj.DMerc`, a smooth lower-bound objective wrapper around
  `mercier_terms_from_state` for stellarator-symmetric finite-beta
  optimization examples. The same wrapper now also covers LASYM=True after the
  derivative branch below.
- 2026-05-10: Added the JAX LASYM=True `bsubs` derivative reconstruction and
  LASYM cos/sin phase geometry branch for state-level `DMerc`, with transform
  parity and AD tests. Remaining finite-beta source objective work is Redl
  bootstrap-current mismatch plus higher-level parity gates.
- 2026-05-10: Added VMEC/wout parity coverage for state-level `DMerc` on the
  bundled finite-beta QI input. This also fixed the bss half-mesh geometry path
  so `rs12/zs12` include VMEC's odd-channel correction before forming `B_s`.
- 2026-05-10: Added differentiable VMEC/JXBFORCE profile accessors
  (`jdotb`, `bdotb`, `bdotgradv`) to the state-level Mercier path, exposed
  them as `vj.JDotB`, `vj.BDotB`, and `vj.BDotGradV` objectives, and added
  unit plus full-gated VMEC/wout parity tests.
- 2026-05-10: Added `vj.ToroidalCurrent` and `vj.ToroidalCurrentGradient`
  objective objects for the state-derived VMEC/Mercier `torcur` and `ip`
  profiles, with example comments, documentation, AD checks, and finite-
  difference derivative coverage.
- 2026-05-10: Added `vj.BVector` for Cartesian magnetic-field targeting and
  `vj.JVector` for VMEC-coordinate current-density targeting
  `(itheta/sqrtg, izeta/sqrtg)`, with docs, example comments, and derivative
  tests.  This closes the first finite-beta vector-diagnostics API lane.
- 2026-05-10: Added a differentiable `vj.RedlBootstrapMismatch` objective.
  The Redl algebra follows SIMSOPT/Redl et al.; vmec_jax evaluates the needed
  geometry from state-level VMEC channels using fixed trapped-fraction
  quadrature. Added tests, docs, and example wiring. The next lane is parity
  against SIMSOPT RedlGeomVmec/RedlGeomBoozer on finite-beta equilibria.
- 2026-05-10: Added an optional SIMSOPT parity test for the Redl algebra
  itself, comparing `jdotB`, `L31`, and `L32` against
  `simsopt.mhd.bootstrap.j_dot_B_Redl` when SIMSOPT is installed.
- 2026-05-10: Added an optional SIMSOPT `RedlGeomVmec` parity test on the
  committed `wout_shaped_tokamak_pressure.nc` fixture. With shared SIMSOPT
  Redl geometry the residuals agree tightly; the public differentiable
  vmec_jax state-geometry path is held to a 2% envelope. The Boozer geometry
  lane remains subprocess-isolated because the local SIMSOPT Boozer path exits
  the Python process in this environment.
- 2026-05-10: Started the source-organization refactor by splitting pure Redl
  bootstrap profile, trapped-fraction, current-fit, and mismatch algebra into
  `vmec_jax.redl_bootstrap`. `finite_beta` keeps the state-dependent geometry
  and Mercier/JXBFORCE plumbing while preserving existing public imports.
- 2026-05-10: Continued the source split by extracting pure Mercier algebra,
  Mercier surface-integral reductions, and JXBFORCE profile reductions into
  `vmec_jax.mercier`. State-level VMEC composition remains in `finite_beta`,
  and the existing public `finite_beta`/top-level imports are preserved.
- 2026-05-10: Ran the CI-equivalent fast coverage gate locally:
  `389 passed, 21 skipped, 85 deselected`, total coverage `60.77%`, runtime
  about `6:07`. Raised the CI floor from 58% to 60%; the next coverage step
  should be targeted tests for `driver`, `optimization_workflow`, `plotting`,
  and selected solver branch logic before moving to 65%.
- 2026-05-10: Added focused optimization-workflow API tests for QI tuple
  semantics: nonzero QI targets are rejected, mixed `QuasiIsodynamicOptions`
  objects are rejected, and QI field objectives fail clearly outside the shared
  QI solve path.
- 2026-05-10: Added edge-case tests for the split finite-beta kernels: Mercier,
  Mercier surface-integral, and JXBFORCE helpers return zero profiles on
  too-short radial meshes, and trapped-fraction geometry rejects invalid array
  shapes explicitly.
- 2026-05-10: Added cheap driver-policy unit tests for solver-mode aliases,
  scalar/list parsing, final-FTOL selection, integer budget allocation, residual
  convergence checks, and history-based final residual extraction. This targets
  CI-relevant control-flow branches without adding additional VMEC solves.
- 2026-05-10: Added `least_squares_solve` dispatch tests for ordinary QS and QI
  problems. These verify SIMSOPT-like objective tuple metadata and shared
  `QuasiIsodynamicOptions` are routed to the correct low-level optimization
  path without executing expensive optimization runs.
- 2026-05-10: Added synthetic rendering coverage for the public optimization
  plotting helpers: 3-D LCFS comparison, LCFS `|B|` contour panels, and
  objective-history plots. This protects README/docs plot wrappers without
  adding real VMEC output fixtures.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the driver,
  workflow-dispatch, and plotting tests: `400 passed, 21 skipped, 85 deselected`,
  total coverage `61.44%`, runtime about `6:57`. Raised the CI floor from 60%
  to 61%; the next material coverage step is 65% and should still come from
  targeted tests/refactors, not synthetic coverage padding.
- 2026-05-10: Added focused optimization-workflow tests for mode-continuation
  policy helpers and QI objective factories. These cover QI residual weighting,
  Boozer surface slicing for mirror-ratio penalties, elongation penalties, and
  LgradB state-objective plumbing without executing VMEC or Boozer solves.
- 2026-05-10: Added plotting API compatibility tests for the legacy
  `plot_qh_optimization` wrapper and angle-label formatting. The examples use
  the generic plotting helpers directly, while the wrapper remains protected for
  downstream scripts that still call it.
- 2026-05-10: Added driver-policy tests for JAX backend fallback, dynamic-scan
  environment parsing, and diagnostic-history convergence fallbacks. These
  target CPU/GPU runtime-policy control flow and convergence reporting without
  adding new heavy solve cases.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the QI,
  plotting-wrapper, and driver-policy test slices: `405 passed, 21 skipped,
  85 deselected`, total coverage `61.65%`, runtime about `7:01`. Kept the CI
  floor at 61%; there is not enough platform margin yet to raise it to 62%.
- 2026-05-10: Added workflow tests for `FixedBoundaryVMEC.from_input`, the
  `mpol/ntor >= max(min_vmec_mode, max_mode+2)` resolution policy, QS objective
  wrappers, and lower-bound/QI helper edge paths. These protect the
  SIMSOPT-like example API and the high-mode optimization resolution policy.
- 2026-05-10: Added synthetic rendering tests for bundled overview and
  `|B|`/contravariant/covariant parity plot writers. The tests monkeypatch
  VMEC/JAX kernels and use toy `wout` data, so docs/validation plotting entry
  points are covered without adding solver runtime.
- 2026-05-10: Added synthetic coverage for the `plot_wout()` CLI diagnostic
  renderer. This protects the four-output `vmec_jax --plot wout_*.nc` workflow
  used in docs and user-facing examples.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after workflow and
  plotting coverage slices: `411 passed, 21 skipped, 85 deselected`, total
  coverage `62.74%`, runtime about `6:31`. Raised the CI floor from 61% to 62%;
  63% is still too close to the local value for platform-safe CI gating.
- 2026-05-10: Added CLI entrypoint coverage for `python -m vmec_jax`, including
  the XLA/PjRt warning-suppression environment defaults and the module-level
  dispatch to `vmec_jax.cli.main`.
- 2026-05-10: Expanded `_compat` coverage around NumPy-mode dispatch, no-op JIT
  fallback behavior, compilation-cache environment policy, and best-effort JAX
  cache configuration updates. This protects the CPU/GPU cache and warning
  suppression defaults that affect first-run user experience.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the module
  entrypoint and `_compat` tests: `409 passed, 21 skipped, 85 deselected`, total
  coverage `62.91%`, runtime about `6:28`. Kept the CI floor at 62%; 63% still
  needs more margin before it is safe to enforce in CI.
- 2026-05-10: Added synthetic optimization-helper coverage for max-mode
  boundary extension, LASYM boundary coefficient families, fixed-boundary
  context precomputation, no-op boundary truncation, and Gauss-Newton edge
  exits. Verified with `python -m pytest tests/test_optimization_helpers.py -q`
  (`48 passed, 1 skipped`) and `ruff check tests/test_optimization_helpers.py`.
- 2026-05-10: Added wout helper coverage for JAX VMEC weights, beta/aspect edge
  cases, bsubv equilibrium correction guards, and ctor branch selection. Verified
  with `python -m pytest tests/test_wout_helpers.py -q` (`6 passed`) and
  `ruff check tests/test_wout_helpers.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the optimization
  and wout helper slices: `420 passed, 21 skipped, 85 deselected`, total coverage
  `63.15%`, runtime about `6:32`. Kept the CI floor at 62%; the current 63%
  margin is still too narrow for a platform-safe threshold bump.
- 2026-05-10: Added CLI dispatch coverage for CPU default policy selection,
  explicit parity/fast modes, VMEC++ restart flags, profiling hooks, explicit
  output paths, missing-input errors, invalid `--jit-forces`, and read failures.
  Verified with `python -m pytest tests/test_cli_helpers.py -q` (`11 passed`)
  and `ruff check tests/test_cli_helpers.py`.
- 2026-05-10: Added residual diagnostic API guard coverage for missing-JAX,
  singular `gamma=1`, and pressure-shape validation paths. Verified with
  `python -m pytest tests/test_fast_physics_kernels.py -q` (`11 passed`) and
  `ruff check tests/test_fast_physics_kernels.py`.
- 2026-05-10: Added visualization helper coverage for VTK scalar/vector array
  serialization, structured-grid and polyline validation errors, 3-D point data,
  automatic wout path resolution, and the public surface/fieldline/volume export
  workflow using synthetic kernels. Verified with
  `python -m pytest tests/test_visualization_vtk.py -q` (`6 passed`) and
  `ruff check tests/test_visualization_vtk.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after CLI, residuals,
  and visualization slices: `429 passed, 21 skipped, 85 deselected`, total
  coverage `63.53%`, runtime about `6:25`. Kept the CI floor at 62%; the next
  safe bump remains 65% after more meaningful coverage/refactor work.
- 2026-05-10: Added VMEC2000 helper coverage for executable discovery, missing
  executable errors, fake subprocess execution, namelist patching inside
  `run_xvmec2000`, threed1 discovery, trace parsing, and runtime capture.
  Verified with `python -m pytest tests/test_fast_physics_kernels.py -q`
  (`12 passed`) and `ruff check tests/test_fast_physics_kernels.py`.
- 2026-05-10: Added profile edge-case coverage for LRFP iota inversion, empty
  cubic-spline current auxiliary arrays, non-monotone AUX trimming, signed
  `SPRES_PED` normalization, and unsupported profile-type errors. Verified with
  `python -m pytest tests/test_fast_physics_kernels.py -q` (`13 passed`) and
  `ruff check tests/test_fast_physics_kernels.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after VMEC2000 helper
  and profile edge-case slices: `431 passed, 21 skipped, 85 deselected`, total
  coverage `63.67%`, runtime about `6:21`. Kept the CI floor at 62%; 64% still
  has too little cross-platform margin, and the next safe threshold bump remains
  65% after more meaningful coverage/refactor work.
- 2026-05-10: Added initial-guess helper coverage for axis alias parsing,
  VMEC theta-flip logic, frozen zero-axis initialization, axis override
  extraction, and NumPy/JAX axis-recompute parity. This exposed and fixed a
  real JAX-path mismatch: when the axis scan finds no positive min-Jacobian
  improvement, `_recompute_axis_from_state_vmec_jax` now keeps VMEC's midpoint
  fallback instead of moving to the least-bad grid point. Verified with
  `python -m pytest tests/test_init_guess.py -q` (`8 passed`) and
  `ruff check vmec_jax/init_guess.py tests/test_init_guess.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the
  initial-guess parity fix: `438 passed, 21 skipped, 85 deselected`, total
  coverage `64.08%`, runtime about `6:27`. Kept the CI floor at 62%; the next
  threshold bump should wait for a wider margin from additional physics-kernel
  or solver-control coverage.
- 2026-05-10: Added runtime-neutral helper coverage for state pack/unpack,
  `VMECState` and `Coords` PyTree roundtrips, VMEC mode ordering, Nyquist mode
  padding, and VMEC default grid sizing including axisymmetric `NZETA=1`
  collapse. Verified with
  `python -m pytest tests/test_state_coords_helpers.py tests/test_modes_helpers.py -q`
  (`6 passed`) and `ruff check tests/test_state_coords_helpers.py tests/test_modes_helpers.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after state/coords
  and mode helper tests: `444 passed, 21 skipped, 85 deselected`, total coverage
  `64.13%`, runtime about `6:26`. Kept the CI floor at 62%; 64% remains too
  tight for a threshold bump.
- 2026-05-10: Added cheap helper coverage for radial derivative edge cases
  (`ns=1`, `ns=2`), periodic angle-step validation, and Nyquist mode/basis cache
  reuse. Verified with
  `python -m pytest tests/test_grid_radial_nyquist_helpers.py -q` (`3 passed`)
  and `ruff check tests/test_grid_radial_nyquist_helpers.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the
  radial/grid/Nyquist helper slice: `447 passed, 21 skipped, 85 deselected`,
  total coverage `64.15%`, runtime about `6:28`. Kept the CI floor at 62%;
  next meaningful target is enough coverage margin to raise the gate directly to
  65% without CI flakiness.
- 2026-05-10: Added VMEC transform-helper coverage for wrout Nyquist cosine/sine
  coefficients, jxbforce cosine/sine coefficients, Nyquist synthesis, JAX/NumPy
  `chipf` edge cases, and pure driver budget/residual helper edge cases. These
  tests compare vectorized paths against explicit loop formulas using synthetic
  VMEC trig tables, so they are physics-relevant without running a solve.
  Verified with
  `python -m pytest tests/test_wout_helpers.py tests/test_driver_api.py -q`
  (`57 passed, 1 skipped`) and
  `ruff check tests/test_driver_api.py tests/test_wout_helpers.py`.
- 2026-05-10: Re-ran the CI-equivalent fast coverage gate after the transform
  and driver-helper slice: `450 passed, 21 skipped, 85 deselected`, total
  coverage `64.66%`, runtime about `6:29`. The GitHub py3.11 job on commit
  `7ec28db` ran a slightly different optional-test mix (`440 passed, 31 skipped,
  85 deselected`) and reported `62.99%`, so the enforced CI floor stays at 62%
  until solver/physics-kernel coverage creates a wider CI-side margin. Keep the
  next threshold target at 65% only after that margin exists.
- 2026-05-11: Added fast numerical-helper coverage for Fourier projection
  roundtrips, non-stacked helical-basis fallback evaluation, VMEC-style
  angle/radial integral conventions, two-power and cubic-spline profile
  variants, LRFP iota inversion, pedestal clipping, and unsupported-profile
  guards. Verified with
  `python -m pytest tests/test_numerics_helper_edges.py -q` (`6 passed`),
  `ruff check tests/test_numerics_helper_edges.py`, and a combined helper slice
  (`63 passed, 1 skipped`). Re-ran the CI-equivalent fast coverage gate:
  `456 passed, 21 skipped, 85 deselected`, total coverage `64.84%`, runtime
  about `6:29`. This is progress but still not enough margin to raise the
  enforced CI floor above 62%.
- 2026-05-11: Added fast VMEC half-mesh and residual-diagnostic coverage for
  pshalf staggering, even/odd metric decompositions, one-surface half-mesh
  behavior, and the normalized `force_residuals_from_state` path using a
  controlled differentiable synthetic geometry. Verified the targeted slice with
  `python -m pytest tests/test_fast_physics_kernels.py tests/test_bcovar_lambda_axis_closure.py -q`
  (`19 passed`) and `ruff check tests/test_fast_physics_kernels.py tests/test_bcovar_lambda_axis_closure.py`.
  Re-ran the CI-equivalent fast coverage gate:
  `459 passed, 21 skipped, 85 deselected`, total coverage `65.00%`, runtime
  about `7:11`. Keep the enforced CI floor at 62% for now: local coverage only
  barely rounds to 65%, and previous GitHub jobs have reported lower coverage
  for the same tree due to optional-test/platform differences.
- 2026-05-11: Added fast VMEC force-balance and force-helper coverage for
  `lforbal` radial weights, VMEC current/equif/plascur formulas, lforbal
  m=1,n=0 correction application, synthetic preconditioner-derived factors,
  VMEC force radial stencils, parity coefficient selection, and R/Z residual
  scalar normalization. Verified with
  `python -m pytest tests/test_vmec_forces_fast_helpers.py tests/test_lforbal_fast_helpers.py -q`
  (`9 passed`) and `ruff check tests/test_vmec_forces_fast_helpers.py tests/test_lforbal_fast_helpers.py`.
  Re-ran the CI-equivalent fast coverage gate:
  `468 passed, 21 skipped, 85 deselected`, total coverage `65.61%`, runtime
  about `7:06`. Raised the enforced py3.11 CI coverage floor conservatively
  from 62% to 63%; do not raise to 65% until GitHub's measured coverage has
  enough cross-platform margin.
- 2026-05-11: Added fast VMEC constraint-helper coverage for `faccon` indexing,
  heuristic `tcon` scaling, short-mesh preconditioner guards, fallback
  preconditioner integration weights, and `alias_gcon` shape guards. Also
  enabled the lasym `alias_gcon` loop-parity test; it now matches the explicit
  VMEC-style reference loops to floating-point noise. Verified with
  `python -m pytest tests/test_vmec_constraints_fast_helpers.py tests/test_vmec_alias_gcon.py -q`
  (`7 passed`) and `ruff check tests/test_vmec_constraints_fast_helpers.py tests/test_vmec_alias_gcon.py`.
  Re-ran the CI-equivalent fast coverage gate:
  `474 passed, 20 skipped, 85 deselected`, total coverage `65.65%`, runtime
  about `6:19`. Keep the enforced CI floor at 63% until GitHub py3.11 has
  enough margin for a safe 64-65% bump.
- 2026-05-11: Added optional local validation gates against VMEC2000 and
  SIMSOPT without making required CI depend on either package. The VMEC2000
  gate runs two short executable stage-trace comparisons (`input.circular_tokamak`
  and `input.basic_non_stellsym_pressure`) behind `VMEC2000_INTEGRATION=1`;
  the SIMSOPT gate compares the VMEC-only QH quasisymmetry residual formula on
  `wout_nfp4_QH_warm_start.nc` behind `RUN_SIMSOPT_VALIDATION=1`. Added
  deterministic `vmec_residue` helper coverage for VMEC m=1 constraints,
  constrained Z-force zeroing, `scalxc`, getfsq radial edge policy, and dynamic
  scalar residual normalization. Verified locally with the optional validation
  commands, full docs build, and the CI-equivalent fast gate:
  `476 passed, 21 skipped, 86 deselected`, total coverage `65.80%`, runtime
  about `7:08`; GitHub Actions run `25660215594` passed.
- 2026-05-11: Hardened the `RUN_FULL=1` scalar-residual parity test so it
  prefers downloaded `_reference.nc` assets but falls back to the bundled
  VMEC2000 `wout_circular_tokamak.nc` when the larger asset bundle has not been
  fetched. Verified with
  `RUN_FULL=1 python -m pytest tests/test_residue_getfsq_parity.py -q`
  (`1 passed`).
- 2026-05-11: Added a synthetic-only `vmec_realspace` invariant slice covering
  LASYM full-grid synth/analyze round-trip, helical mixed-partial commutation
  with physical zeta derivatives, and circular axisymmetric geometry identities.
  This raises targeted `vmec_realspace.py` statement coverage from about 47% to
  about 80% without adding large fixtures or slow VMEC solves. Verified with
  `python -m pytest tests/test_vmec_realspace.py tests/test_vmec_realspace_invariants.py -q`
  and `python -m ruff check tests/test_vmec_realspace_invariants.py`; the
  CI-equivalent fast gate now reports `479 passed, 21 skipped, 86 deselected`,
  total coverage `65.83%`, runtime about `6:24`.
- 2026-05-11: Performance lane next targets after profiling review: avoid
  replay-only `update_rms` work unless update limiting or diagnostics need it,
  reuse residual-derived accepted-point objective totals for generic/QI
  workflows, and make scalar-adjoint trial acceptance cheaper while exact-checking
  accepted states. These are intentionally deferred from the docs/validation
  patch because they touch discrete-adjoint replay semantics.
- 2026-05-11: Implemented the first narrow replay-cost reduction: replay-only
  VJP/JVP maps pass `need_update_rms=False`, while the default accepted-solve
  path still reports update RMS diagnostics and any solve with
  `limit_update_rms=True` still computes RMS for clipping. Verified with
  `python -m pytest tests/test_discrete_adjoint_qh.py tests/test_discrete_adjoint_chunking.py tests/test_optimization_helpers.py -q`
  (`76 passed, 19 skipped`). A small CPU exact-Jacobian diagnostic on QH
  `max_mode=1`, 8 DOFs, 20 inner iterations completed in `7.707 s`
  (`jacobian_tape_replay=2.924 s`, `exact_tape_build=0.319 s`).
- 2026-05-11: Reassessed the QI validation and refactor roadmap from the
  current docs/tests. Fast QI tests now cover metric semantics and synthetic
  ranking, but not multi-seed optimizer robustness. The next validation step is
  a small solved-state `qi_diagnostics_from_state` gate; the full QI/QP/QH/QA
  and simple non-omnigenous seed matrix remains deferred manual/nightly work.
  Refactoring stays limited to narrow tested seams until solver/free-boundary
  parity gates have more margin.
- 2026-05-11: Added a bounded no-optimization QI seed audit CLI at
  `examples/optimization/audit_qi_seed_suitability.py`. It ranks solved
  `input`/`wout` seed pairs by QI diagnostics and engineering constraints,
  uses `OMNIGENITY_OPTIMIZATION_ROOT` for optional external reference seeds,
  and writes JSON/CSV summaries for selecting QI/QP/QH/QA/simple starts before
  expensive seed-robust optimization sweeps.
- 2026-05-11: Started the large-module refactor with narrow tested seams:
  `vmec_jax._solve_runtime` owns scan runtime helpers previously embedded in
  `solve.py`, and `vmec_jax.wout_schema` owns `WoutData` plus low-level wout
  schema helpers while `vmec_jax.wout` keeps compatibility re-exports.
- 2026-05-11: Extended the QI seed audit with `--prefine-probes plan|run`.
  The default remains audit-only. The plan mode writes a hard-capped manifest
  for tiny QI-only prefine probes, including selected seeds, commands, expected
  artifacts, and objective settings before any expensive seed-robust run.
- 2026-05-11: Continued the refactor with additional narrow seams:
  `vmec_jax.wout_io` owns low-level netCDF mode/string/variable helpers, and
  `_solve_runtime` now also owns dump iteration parsing plus simple dump gate
  policy. Public compatibility through `vmec_jax.wout` and `vmec_jax.solve`
  was preserved.
- 2026-05-12: Added continuation regression tests for repeated and higher mode
  schedules, including synthetic workflow control-flow and a real
  boundary-projection stage gate. These protect the fix that carries optimized
  VMEC inputs forward between continuation stages and keeps projected high
  modes zero.
- 2026-05-12: Added bundled wout `chipf` parity tests across circular,
  finite-beta shaped tokamak, QH, QA, and QI fixtures. The tests verify the
  VMEC2000 half-mesh convention and the `chips_from_wout_chipf` round-trip in
  required-tier CI without running an external executable.
- 2026-05-12: Re-profiled performance lanes after backend-adaptive replay
  bucketing. `office` cold fixed-boundary smoke still favors CPU (`13.80 s`
  CPU versus `68.86 s` GPU), while local exact-Jacobian callback time is
  dominated by replay and tangent construction/projection. The next performance
  patch should target accepted-point tangent fusion/reuse.
- 2026-05-12: Added the first tangent-reuse performance patch:
  scalar-gradient callbacks now reuse the cached affine initial-state tangent
  map and project reverse tape cotangents through it. This reduced one local
  same-branch warm QA gradient callback from about `0.279 s` to `0.188 s`
  while preserving dense-Jacobian gradient parity.
- 2026-05-12: Ran a tier-2 bounded QI seed-robust probe. The promoted QP/NFP2
  seed improved the prefine objective from `2.4465e-2` to `1.4150e-2`
  with monotone history, but final-state audit still failed mirror and
  elongation. The QA low-resolution comparator did not improve and remains a
  rejected QI seed candidate for this policy.
- 2026-05-12: Added a required-tier bundled stored-field parity gate:
  reconstructed Cartesian `|B|` from VMEC2000 `bsup*` fields agrees with
  bundled `bmnc/bmns` fields for QA, QH, and QI fixtures within a small
  envelope.
