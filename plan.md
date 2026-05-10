# VMEC-JAX Research-Grade Roadmap

Last updated: 2026-05-10
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
- GPU execution works, but small/medium optimization cases are still often CPU
  faster because replay/compile/host overhead dominates.
- Required CI coverage is still far below the long-term 95% goal.
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
- [ ] Prove seed robustness: start QI from QI, QP, QH, QA, and a simple
      non-omnigenous seed; document which policies reliably converge.
- [x] Diagnose remaining QI noisiness by one-DOF scans of Boozer/QI metrics and
      choose default resolutions/weights that preserve ranking while remaining
      differentiable.

Acceptance:

- Best QI has poloidally closed Boozer `|B|` contours, low legacy QI diagnostic,
  mirror ratio at target, acceptable elongation, `abs(iota) >= 0.41`, and aspect
  near the chosen target.

## Milestone 2: Physics Objectives And Diagnostics

- [x] Promote differentiable `DMerc` from diagnostic/parity into an objective
      with objective-routing and JAX AD tests.
- [x] Add differentiable `J dot B`/JXBFORCE profile objective accessors with
      documented normalization and VMEC/wout parity tests.
- [x] Add differentiable state-derived toroidal current-profile objective
      accessors with documented normalization.
- [x] Add differentiable vector `J` and vector `B` objective accessors with
      documented normalization.
- [ ] Add finite-beta stage-one objectives matching the finite-beta optimization
      references: pressure/current profiles, beta, volavgB, iota, aspect,
      Mercier, and quasisymmetry/omnigenity terms.
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

Acceptance:

- CI and nightly tests catch real physics/numerics regressions without relying on
  large committed output files.

## Milestone 6: Refactoring And API Hygiene

- [ ] Split `solve.py` into solver orchestration, residual kernels, convergence,
      fallback policy, tracing, and output modules.
- [ ] Split `wout.py` into schema, writer, reader, and derived diagnostics.
- [ ] Split `free_boundary.py` into mgrid I/O, Nestor kernels, runtime state, and
      diagnostics.
- [ ] Keep compatibility imports in `vmec_jax.__init__`, but move implementation
      to smaller documented modules.
- [ ] Add docstrings and comments where they clarify physics, numerical method, or
      AD behavior.

Acceptance:

- Core modules are small enough to review, tests still pass, and public imports are
  backward compatible.

## Milestone 7: Tests And Coverage

- [ ] Raise required CI coverage from the current fast gate toward 95% in staged
      steps: 62%, 65%, 75%, 85%, 95%.
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
