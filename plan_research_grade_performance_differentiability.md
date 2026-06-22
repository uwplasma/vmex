# Research-Grade Performance, Differentiability, and Refactor Plan

Status: active authoritative plan for PR #20 and follow-up work.
Created: 2026-06-21.
Branch: `codex/differentiability-refactor-plan`.

This file supersedes the scattered planning notes in `plan_differentiability.md`
for the remaining performance, differentiability, and maintainability work.
Older plan files are historical logs unless this file explicitly points back to
them.

## Single Goal

Make `vmec_jax` a research-grade VMEC implementation that is:

- fast enough to be competitive for fixed-boundary and free-boundary production
  workflows, with measured cold, warm, optimization, CPU, and GPU performance;
- memory efficient enough for long optimization loops without retained
  JAX/XLA/tape growth;
- end-to-end differentiable from Python for fixed-boundary, free-boundary,
  diagnostics, Boozer/QS/QI objectives, direct-coil fields, and optimization
  loops whenever the mathematical map is smooth;
- honest about adaptive branch changes: branch-local differentiability must be
  exact and validated, while arbitrary discontinuous branch switches are treated
  as piecewise-smooth events with fingerprint gates or explicit smooth
  controller replacements;
- organized into domain modules with short, testable functions and pedagogical
  docstrings, while retaining compatibility with the current public API and
  VMEC2000 inputs/outputs.

## Reference Sources

### Literature and Documentation

- VMEC/STELLOPT documentation
  (`https://princetonuniversity.github.io/STELLOPT/VMEC.html`): VMEC uses a
  variational energy formulation, Fourier expansions in poloidal/toroidal
  coordinates, and a Richardson iteration scheme for the pseudo-time evolution
  of the parabolic force-balance equations.
- VMEC2000 source: local reference at
  `/Users/rogeriojorge/local/STELLOPT/VMEC2000/Sources`.
- VMEC++ paper and source: local source at
  `/Users/rogeriojorge/local/vmecpp`; paper
  `https://arxiv.org/abs/2502.04374`.
- Fast automated adjoints for spectral PDE solvers: Skene and Burns,
  `https://arxiv.org/abs/2506.14792`, motivates symbolic/spectral
  operator-level adjoints that retain sparse spectral solver memory and speed.
- JAX custom derivative rules: `jax.custom_jvp`, `jax.custom_vjp`, remat,
  buffer donation, profiling, AOT lowering, and memory profiling are first-line
  tools
  (`https://docs.jax.dev/en/latest/notebooks/Custom_derivative_rules_for_Python_code.html`).
- JAXopt implicit differentiation: `custom_root`, `custom_fixed_point`,
  `root_jvp`, and `root_vjp` define the production path for converged implicit
  VMEC states (`https://jaxopt.github.io/stable/implicit_diff.html`).
- Candidate numerical libraries for implementation experiments: `jax`, `jaxopt`,
  `lineax`, `optimistix`, `equinox`, `optax`, and `orthax`.

### Local Algorithm Anchors

- VMEC2000 fixed-boundary loop:
  `eqsolve -> evolve -> funct3d -> bcovar/tomnsp/totzsp/forces/residue`.
- VMEC2000 speed-critical files:
  `funct3d.f`, `bcovar.f`, `forces.f`, `residue.f90`, `tomnsp_mod.f`,
  `totzsp_mod.f`, `precondn.f`, `precon2d.f`, and block-tridiagonal solvers.
- VMEC2000 performance lessons:
  compact Fortran arrays, in-place updates, predictable loop nests, explicit
  timing buckets, radial preconditioning, and no per-iteration Python/XLA setup.
- VMEC++ performance/code-structure lessons:
  domain-specific C++ classes, precomputed Fourier bases, checkpoint/restart
  flow-control state, bounded free-boundary geometry objects, and explicit
  physical failure handling.
- Current `vmec_jax` pain points from source-health:
  `vmec_jax/solvers/fixed_boundary/residual/iteration.py` has a 2645-line
  function; several core modules are 1800-2000 lines; tests contain large
  monolithic fixtures. This is not maintainable enough for the target project.

## Open Lanes and Current Completion

- Performance benchmark and profiling harness: 98%.
  The PR #20 single-grid matrix, current-vs-main comparator, VMEC2000 rows, and
  VMEC++ optional rows are regenerated with CSV/JSON provenance after the
  full-JIT R/Z preconditioner speedup. Remaining work is deeper kernel-level
  decomposition and long-term dashboard automation.
- Fixed-boundary production differentiability: 90%.
  AD-vs-central-FD evidence now passes `1e-9` for fixed-boundary geometry,
  profiles, QS/QI diagnostics, `DMerc`, and `D_R`. Remaining work is
  operator-level implicit/JVP/VJP productionization.
- Free-boundary production differentiability: 87%.
  Direct coil fields, JAX mgrid interpolation, accepted-branch replay, and
  fingerprint-gated branch-local gates pass `1e-9` evidence for selected
  physical scalars. Dynamic replay now tolerates minimal/full preconditioner
  cache payload structure changes in both JVP and VJP paths. Arbitrary adaptive
  branch differentiation remains unclaimed.
- Single-stage coil optimization: 86%.
  Examples and branch-local derivative proposal paths exist; complete solves
  still need to remain the acceptance authority until the full adaptive seam is
  validated.
- CPU/GPU runtime and memory footprint: 87%.
  The single-grid matrix shows no PR regression against `origin/main`, and warm
  CPU `vmec_jax` beats VMEC2000 on 7 of 16 rows. The 3D preconditioner R/Z
  matrix hotspot has been reduced by the default full-JIT builder, and ordinary
  host iota smoothing no longer pays JAX scatter/update startup cost. Finite
  beta setup profiling now splits profile-data construction from trig-table
  setup. Concrete finite-beta/current-profile CPU runs automatically use the
  host profile setup path, reducing profile-data setup from about `0.39 s` to
  about `0.001 s` on the promoted QH finite-beta probe. Fixed-boundary
  iteration can now reuse the setup-time axis force probe on strict no-reset
  iteration-1 branches, reducing the promoted bounded finite-beta wall time
  from `6.36 s` to `5.78 s` without changing residual scalars. Preconditioner
  apply timing now exposes m=1 RHS scaling, R/Z apply, fused payload, output
  block, and sync sub-buckets; the m=1 RHS scaling path now uses a compiled
  channel scaler, reducing the promoted finite-beta preconditioner apply from
  about `0.36 s` to `0.27 s`. Memory remains materially higher than VMEC2000,
  with LASYM finite-beta layouts and preconditioner apply/seed costs still the
  main targets.
- Refactor/API/examples: 45%.
  Public examples are better, but core source files and tests are still too
  large and too entangled. The fixed-boundary residual timing/setup seam is now
  slightly cleaner, but the main residual loop still needs a larger split.
- VMEC2000/VMEC++ parity and physics gates: 96%.
  The PR #20 four-row executable WOUT parity gate passed, and the single-grid
  runtime matrix records VMEC++ availability per row. More bounded
  free-boundary external parity remains future work.
- Docs/release hygiene: 96%.
  README is concise, runtime/memory detail lives in docs, and benchmark plus
  AD-FD provenance are refreshed. Remaining work is Sphinx gating and pruning
  historical performance prose after review.
- Overall completion: 89%.
  PR #20 readiness gates for benchmark, current-vs-main regression,
  differentiation evidence, and selected WOUT parity are now substantially
  complete; the long-term research-grade performance/refactor work remains
  active.

## Definition of Done

The plan is complete only when all of the following are true:

- The public README is concise and does not expose bulky performance/memory
  plots; detailed benchmarks live in docs with CSV/JSON provenance.
- A full benchmark matrix measures current branch, `origin/main`, VMEC2000, and
  VMEC++ where supported, with per-row convergence/parity status.
- No accepted current-branch regression above `10%` runtime or `15%` peak memory
  remains unclassified.
- Representative fixed-boundary runs are within a documented target factor of
  VMEC2000 for cold and warm CPU runs, and GPU wins on at least the cases whose
  resolution is large enough to amortize compilation and transfer costs.
- AD-vs-FD evidence uses `1e-9` relative/absolute tolerance for smooth
  deterministic scalars when numerically stable; documented exceptions must
  name the noise source and use the tightest reliable tolerance.
- Production fixed-boundary derivatives use operator-level or implicit
  differentiation rather than retaining full long unrolled tapes by default.
- Free-boundary derivatives are validated for direct-coil and mgrid pathways
  through at least one complete-loop fingerprint-gated branch-local physical
  scalar.
- Adaptive branch switching claims are conservative and mathematically precise:
  piecewise-smooth derivatives within a branch are supported; branch switches
  require fingerprints, smoothing, or explicit nonsmooth optimization treatment.
- No production source file exceeds 1500 lines without an explicit exception;
  no production function exceeds 250 lines without an explicit exception.
- Tests remain physics/numerics/parity tests, not only smoke tests, and CI keeps
  runtime bounded.
- Docs explain performance, differentiability, file organization, examples,
  limitations, and reproduction commands in one coherent flow.

## Milestones

### M0: Plan and README Scope

Tasks:

- Remove the top-level README runtime/memory figure.
- Keep detailed performance artifacts in docs, not the README front page.
- Add this authoritative plan and point older plans here.

Gates:

- `git diff --check`.
- README no longer references `readme_runtime_compare.png`.
- Plan includes milestones, tests, parity gates, and source-file simplification
  targets.

Status: 100%.

### M1: Reference Algorithm Audit

Tasks:

- Map VMEC2000 timing buckets to source routines and corresponding `vmec_jax`
  kernels.
- Profile VMEC2000 with `gprof`, compiler timing flags, or source timers in a
  throwaway local build, not committed unless generally useful.
- Profile VMEC++ on the same fixed-boundary input rows, including unsupported
  input reasons.
- Produce a small markdown table mapping each major VMEC2000 routine to the
  current `vmec_jax` implementation and its timing bucket.

Gates:

- At least one representative row has VMEC2000, VMEC++, cold `vmec_jax`, and
  warm `vmec_jax` timing decomposition.
- WOUT parity remains within existing tolerances for rows used in performance
  claims.

Status: 26%.

### M2: `vmec_jax` Profiling Decomposition

Tasks:

- Split runtime into import/startup, input parsing, grid/mode setup, JAX trace,
  XLA compile, steady solve, diagnostics, and WOUT write.
- Add opt-in debug/profiling flags that are silent by default and safe for CI.
- Use JAX profiling, Perfetto traces, device memory profiling, and host
  `cProfile`/sampling profiles.
- Compare CPU and GPU traces on `ssh office` for cold and warm solves.

Gates:

- A profile report identifies the top five CPU and GPU costs by row.
- Each material regression is assigned to startup, compile, steady solve,
  output, branch controller, or optimizer callback.

Status: 38%.

### M3: Fast Fixed-Boundary Solve Path

Tasks:

- Separate CLI-performance path from Python-differentiable path where needed.
  CLI can be non-differentiable if it is faster; Python API must remain
  differentiable.
- Precompute and cache Fourier bases, mode maps, radial stencils, and profile
  arrays by static shape.
- Replace Python loops/large pytrees in the hot solve with shape-stable JAX
  scans and compact arrays.
- Audit memory donation and avoid retaining unneeded histories in production
  solve calls.
- Reduce WOUT write overhead and avoid postprocessing fields not requested.

Gates:

- Cold and warm fixed-boundary benchmark rows improve or have a documented
  reason they cannot.
- No WOUT parity regression.
- No AD-vs-FD regression.

Status: 18%.

### M4: Operator-Level Differentiability

Tasks:

- Promote linearized residual operators for fixed-boundary force balance.
- Use implicit differentiation for converged equilibrium states:
  solve `(dF/dx)^T lambda = dObjective/dx` using matrix-free VJP/JVP when
  possible.
- Keep unrolled AD only for short validation gates and debugging.
- Follow the spectral-adjoint idea: differentiate operator graphs, not every
  low-level iteration state, whenever the solver has converged.
- Add explicit derivative method metadata to evidence artifacts:
  unrolled AD, implicit/custom-VJP, matrix-free JVP/VJP, branch-local replay,
  or discrete adjoint.

Gates:

- AD-vs-FD panel passes `1e-9` where stable.
- DMerc and `D_R` remain covered.
- Same scalar derivatives agree across unrolled and implicit lanes on tiny
  problems.

Status: 35%.

### M5: Free-Boundary Production Differentiability

Tasks:

- Keep current branch-local/fingerprint-gated gates conservative.
- Add one complete-loop, direct-coil, physical-scalar AD-vs-central-FD gate
  through a tiny free-boundary solve with a fixed branch fingerprint.
- Decide explicitly whether to implement a JAX-visible adaptive controller or
  keep host adaptive selection nondifferentiable with branch-local custom VJPs.
- For discontinuous branch changes, expose branch fingerprints and require
  nonsmooth/black-box outer optimization or smoothing; do not overclaim.

Gates:

- Direct-coil scalar derivative passes central FD under a matching branch
  fingerprint.
- The docs table says exactly what is differentiable and what remains branch
  local.

Status: 65%.

### M6: GPU-Native Performance Path

Tasks:

- Avoid dense SciPy callback paths for high-mode exact optimization on GPU.
- Promote scalar-trust or projected-replay paths that keep tangent/replay data
  on device.
- Profile XLA kernels for force assembly, transforms, preconditioner/update,
  NESTOR/source response, and Boozer diagnostics.
- Use GPU only when problem size is large enough to amortize compile/transfer;
  document CPU as preferred for small cold solves when true.

Gates:

- GPU benchmark matrix has explicit pass/fail/unsupported rows.
- At least one medium/large solve or optimization path is faster on GPU than
  CPU after warmup.

Status: 25%.

### M7: Refactor and File Simplification

Target domain layout:

- `vmec_jax/core`: immutable problem/state/config dataclasses, grids, modes,
  profile representations, and pytrees.
- `vmec_jax/spectral`: Fourier bases, transforms, de-aliasing, interpolation,
  and mode projections.
- `vmec_jax/equilibrium`: fixed-boundary and free-boundary residual operators,
  controllers, scan policies, and preconditioners.
- `vmec_jax/fields`: geometry, magnetic field, current density, profiles, and
  Boozer-ready field diagnostics.
- `vmec_jax/stability`: Mercier, Glasser/resistive terms, magnetic well, and
  related profile derivatives.
- `vmec_jax/objectives`: QS, QP, QI, aspect, iota, mirror, elongation, coil,
  and stability objective terms.
- `vmec_jax/diff`: implicit solves, custom JVP/VJP rules, replay/fingerprint
  utilities, and derivative evidence helpers.
- `vmec_jax/io`: VMEC input, WOUT, mgrid, Boozer files, and asset downloaders.
- `vmec_jax/optimization`: small composable least-squares and continuation
  utilities; examples assemble objective tuples explicitly.

Rules:

- Public compatibility facades can remain, but new code should live in the
  domain modules above.
- Target production file length: `<1500` lines.
- Target production function length: `<250` lines.
- Docstrings should explain physical meaning, array shapes, differentiability
  status, and failure modes.
- Comments should describe non-obvious numerical choices, not restate code.

First refactor tranche:

- Split `solvers/fixed_boundary/residual/iteration.py` into state packing,
  residual-step kernels, convergence/restart policy, output history, and public
  wrapper modules.
- Add tests at the new seams before deleting compatibility imports.

Gates:

- Existing tests pass.
- Source-health no longer reports a >2000-line production file.
- Public imports continue to work.

Status: 11%.

### M8: CI, Coverage, and Physics Gates

Tasks:

- Keep fast CI under the current time budget by sharding expensive exact tests.
- Preserve literature-anchored physics tests: force residuals, WOUT parity,
  profile spline/polynomial handling, Mercier/Glasser derivatives, Boozer/QS/QI
  diagnostics, and free-boundary direct-coil/mgrid parity.
- Add performance-regression gates that compare against stored JSON baselines
  with noise-aware thresholds.

Gates:

- Coverage does not drop below the current Codecov project threshold.
- No new smoke-only tests unless paired with a physics or numerical assertion.

Status: 75%.

### M9: Documentation and Release Hygiene

Tasks:

- Move engineering benchmark detail out of README and into docs.
- Add a clear differentiability table:
  direct coil field, JAX mgrid, fixed-boundary implicit scalars, accepted
  free-boundary replay, fixed-branch custom VJP, adaptive branch limitations.
- Add performance caveats for cold JAX runs, warm optimization loops, CPU vs
  GPU, and VMEC++ optional rows.
- Keep the repo light: no large WOUT/mgrid/output artifacts in git; figures must
  be compressed and generated from provenance.

Gates:

- `python -m sphinx -W -j auto -b html docs docs/_build/html`.
- `python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50 --max-file-mib 2`.

Status: 70%.

## Immediate Next Steps

1. Continue M2/M3 with the next measured fixed-boundary performance bottlenecks:
   cold setup (`setup_axis_reset_s`, boundary/profile setup) and LASYM
   finite-beta memory layout. The R/Z preconditioner matrix hotspot is no
   longer the first target.
2. Start M7 in a larger tranche by splitting
   `solvers/fixed_boundary/residual/iteration.py` at stable seams while keeping
   compatibility wrappers and tests.
3. Continue M4/M5 by moving validated derivative paths from evidence-only
   helpers toward production APIs: fixed-boundary implicit/operator derivatives
   first, branch-local free-boundary derivative proposals second.
4. Keep README performance-light. Runtime/memory matrix evidence stays in
   `docs/performance.rst`; README should only summarize differentiability and
   user-facing capabilities.
5. Check PR #20 CI periodically, not continuously. If it fails, fix the failed
   shard locally before pushing another tranche.

## User Decisions Needed

- Whether to accept a fast non-differentiable CLI path alongside a fully
  differentiable Python API path. Recommendation: yes, because executable users
  care about solve latency while optimizer users care about gradients.
- Whether to make a JAX-visible adaptive free-boundary controller a near-term
  requirement. Recommendation: no for this PR; keep branch-local exact
  derivatives and fingerprints, then revisit if optimization failures clearly
  require differentiating through branch-selection logic.
- Whether VMEC++ parity should be a release gate or a documented optional
  comparison. Recommendation: optional per-row comparison, because VMEC++ input
  support/convergence coverage is not identical to VMEC2000.
- Whether PR #20 should be moved back to draft while the long-term refactor
  tranches continue. Current GitHub state is `isDraft=false`; the PR-readiness
  evidence gates are complete locally, but the broader research-grade
  performance/refactor plan is intentionally not complete.

## Progress Log

### 2026-06-21: M1/M2 profiling decomposition scaffold

Steps taken:

- Removed the README runtime/peak-memory figure while keeping detailed
  benchmark artifacts available in docs/provenance.
- Added `tools/diagnostics/fixed_boundary_performance_decomposition.py`.
- Extended `tools/diagnostics/profile_fixed_boundary.py` so profiler JSON
  includes process peak RSS in MiB.
- Ran a map-only report:
  `python tools/diagnostics/fixed_boundary_performance_decomposition.py --skip-runs --outdir outputs/performance_decomposition_map_only`.
- Ran a bounded 3-iteration QH warm-start probe with VMEC2000:
  `JAX_ENABLE_X64=1 python tools/diagnostics/fixed_boundary_performance_decomposition.py --input examples/data/input.nfp4_QH_warm_start --iters 3 --outdir outputs/performance_decomposition_qh_smoke_vmec2000 --vmec2000-exec ~/bin/xvmec2000`.
- Added optional VMEC++ CLI discovery to the same decomposition tool and ran:
  `JAX_ENABLE_X64=1 python tools/diagnostics/fixed_boundary_performance_decomposition.py --input examples/data/input.nfp4_QH_warm_start --iters 3 --outdir outputs/performance_decomposition_qh_smoke_all --vmec2000-exec ~/bin/xvmec2000`.

Results obtained:

- Report artifacts were written under
  `outputs/performance_decomposition_qh_smoke_vmec2000/`.
- VMEC2000 wall time for the 3-iteration fixed-budget probe: `0.312 s`.
- `vmec_jax` cold timed run wall: `5.215 s`.
- `vmec_jax` warm same-process timed run wall: `0.130 s`.
- `vmec_jax` process peak RSS in the profiler child processes:
  about `584-597 MiB`.
- With all backends enabled on the same row, VMEC++ was discovered at
  `/Users/rogeriojorge/Library/Python/3.11/bin/vmecpp`, ran successfully,
  wrote one WOUT, and took `1.215 s` for this tiny fixed-budget probe.
- The measured solver body for this tiny probe is about `0.05 s`; the cold
  gap is dominated by process/import/backend/trace/compile setup rather than
  the three residual iterations.

Best next steps:

1. Run the decomposition on the full historical single-grid matrix and compare
   current branch against `origin/main`.
2. Split cold startup into Python import, JAX import/backend init, config/input
   load, static setup, JAX trace, XLA compile, and first device-ready solve.
3. Start the first refactor tranche by separating the giant fixed-boundary
   iteration function into setup, residual step, controller policy, and output
   collection seams.

### 2026-06-21: Explicit cold/warm phase buckets

Steps taken:

- Extended `profile_fixed_boundary.py` to time warmup explicitly.
- Extended `fixed_boundary_performance_decomposition.py` to derive phase
  buckets and runtime ratios from profiler JSON:
  child process elapsed, import, JAX device discovery, warmup wall, profiled
  run wall, solver setup, solver iteration loop, force evaluation,
  preconditioner, state update, run-minus-solver overhead, and peak RSS.
- Reran the all-backend QH 3-iteration smoke:
  `JAX_ENABLE_X64=1 python tools/diagnostics/fixed_boundary_performance_decomposition.py --input examples/data/input.nfp4_QH_warm_start --iters 3 --outdir outputs/performance_decomposition_qh_phase_smoke2 --vmec2000-exec ~/bin/xvmec2000`.

Results obtained:

- `vmec_jax` cold profiled run: `5.160 s`; solver body: `0.0499 s`;
  run-minus-solver overhead: `5.110 s`; child elapsed: `6.375 s`; peak RSS:
  `586 MiB`.
- Warmup run in the warm child process: `5.178 s`; post-warmup profiled run:
  `0.128 s`; solver body: `0.0465 s`; run-minus-solver overhead: `0.0812 s`;
  peak RSS: `592 MiB`.
- VMEC2000: `0.204 s`; VMEC++: `1.233 s`.
- Runtime ratios for this tiny probe:
  `vmec_jax_cold/vmec2000 = 25.25`, `vmec_jax_warm/vmec2000 = 0.625`,
  `vmec_jax_cold/vmec_jax_warm = 40.4`.

Best next steps:

1. Decompose the `5.11 s` cold run-minus-solver bucket into input/config load,
   static setup, JAX trace, XLA compile, first device-ready execution, and
   output materialization.
2. Decompose the `0.081 s` warm run-minus-solver bucket so optimization-loop
   overhead can be reduced after compile/cache warmup.
3. Run the same phase report on at least one larger finite-beta/multigrid row
   before deciding whether the fast CLI path should bypass richer diagnostics
   or change default solver policy.

### 2026-06-21: cProfile evidence for the fast CLI policy split

Steps taken:

- Added optional `--cprofile-out` and `--cprofile-text-out` to
  `profile_fixed_boundary.py`.
- Added `--cprofile` and `--no-auto-cli-policy` to
  `fixed_boundary_performance_decomposition.py`.
- Ran the short QH probe with the public CLI finish policy and with the raw
  requested solver path:
  `JAX_ENABLE_X64=1 python tools/diagnostics/fixed_boundary_performance_decomposition.py --input examples/data/input.nfp4_QH_warm_start --iters 2 --skip-vmec2000 --skip-vmecpp --cprofile --outdir outputs/performance_decomposition_qh_cprofile_smoke`
  and
  `JAX_ENABLE_X64=1 python tools/diagnostics/fixed_boundary_performance_decomposition.py --input examples/data/input.nfp4_QH_warm_start --iters 2 --skip-vmec2000 --skip-vmecpp --no-auto-cli-policy --cprofile --outdir outputs/performance_decomposition_qh_raw_cprofile_smoke`.

Results obtained:

- With the CLI finisher enabled, the cold profile spends most cumulative time in
  `_maybe_finish_cli_fixed_boundary_run`, which performs additional finish
  attempts and triggers many JAX compilations. In the 2-iteration diagnostic
  run, the profiled cold wall was about `6.10 s`, peak RSS about `596 MiB`, and
  the profile showed about `219` compile/cache events.
- With the raw requested path (`--no-auto-cli-policy`), cold profiled wall fell
  to about `1.57 s`, peak RSS to about `338 MiB`, and compile/cache events fell
  to about `81`.
- The raw path is intentionally less converged because it respects the small
  requested iteration budget exactly. This is not a replacement for normal
  converged CLI runs; it is evidence that the fast non-differentiable CLI path
  must make finish policy explicit and budget-aware rather than adding
  compile-heavy finish attempts unconditionally.

Best next steps:

1. Add a public CLI/API fast-path option that clearly separates "run exactly
   the requested budget" from "finish to convergence", without weakening the
   default converged CLI behavior.
2. Teach the profiler to compare all three policies on the same case:
   raw requested path, current CLI finisher, and a future cheap-budget finisher.
3. Inspect the preconditioner cold compile path, since the raw cold run is now
   dominated by `seed_preconditioner_cache_from_bcovar_update` and
   `rz_preconditioner_matrices`.

### 2026-06-21: Public fixed-boundary finish policy

Steps taken:

- Added a normalized public `finish_policy` option to `run_fixed_boundary`.
- Added CLI flags `--finish-policy {auto,none,bounded,converge}` and
  `--no-finish`.
- Routed profiler/decomposition tools through the public option while keeping
  the legacy `--no-auto-cli-policy` diagnostic alias.
- Added focused unit coverage for policy aliases, CLI propagation, and the
  bounded path suppressing additional finish attempts.

Results obtained:

- Default behavior remains unchanged: `finish_policy="auto"` preserves the
  current converged CLI behavior.
- `finish_policy="none"` / `"bounded"` gives an explicit finite-budget path
  for profiling, benchmark rows, and exact finite-step parity checks.
- Result diagnostics now record `fixed_boundary_finish_policy` and
  `cli_fixed_boundary_finish_enabled`, so benchmark provenance can distinguish
  converged production runs from bounded diagnostic runs.
- Short QH warm-start smoke profiles with `--iters 2 --solver-mode auto
  --no-use-scan --solver-device cpu`:
  - `--finish-policy none`: `1.303 s`, finish disabled, no finish budgets.
  - `--finish-policy auto`: `5.044 s`, finish enabled, budgets `[2, 2]`.
  - `--finish-policy converge`: `5.044 s`, finish enabled, budgets `[2, 2]`.

Best next steps:

1. Continue cold-path decomposition inside the bounded path: input/config load,
   static setup, preconditioner trace/compile, and first device-ready solve.
2. Decide whether the README quick-start should mention `--no-finish` only in
   performance/developer notes or also as a general “run exactly N iterations”
   option.
3. Prototype a cheaper bounded finish policy that can use a small existing
   compiled stage instead of recompiling full finish attempts.

Updated lane percentages:

- Performance benchmark/profiling harness: 78%.
- Fixed-boundary production differentiability: 82%.
- Free-boundary production differentiability: 78%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 70%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 90%.
- Docs/release hygiene: 89%.
- Overall: 72%.

### 2026-06-21: Bounded cold-path preconditioner decomposition

Steps taken:

- Extended `profile_fixed_boundary.py` and
  `fixed_boundary_performance_decomposition.py` to expose setup sub-buckets and
  preconditioner refresh/seed/apply buckets.
- Tested a safe NumPy R/Z matrix-builder shortcut for host-controller updates.
  The shortcut is valid only for axisymmetric/stellsym (`lthreed=False`,
  `lasym=False`) updates; a synthetic 3D check showed the legacy NumPy
  representation is not output-equivalent to the current JAX independent
  toroidal-block path.
- Added a diagnostic `VMEC_JAX_RZ_MATRIX_ASSEMBLY_JIT=0` opt-out for the small
  R/Z matrix assembly JIT, then profiled it in a fresh child process.

Results obtained:

- `input.nfp4_QH_warm_start`, bounded two-iteration CPU run with
  `--finish-policy none --no-use-scan`:
  - default compiled R/Z matrix assembly: `1.282 s` wall,
    `1.047 s` solver total, `0.957 s` preconditioner seed.
  - unjitted R/Z matrix assembly opt-out: `1.897 s` wall,
    `1.674 s` solver total, `1.584 s` preconditioner seed.
- The fresh-process result disproves the earlier warm in-process experiment:
  the small matrix assembly JIT should remain enabled by default for 3D QH.
- Axisymmetric `input.solovev`, same bounded two-iteration CPU profile:
  `0.178 s` wall, `0.169 s` solver total, `0.077 s` preconditioner seed.
- The current cold 3D bottleneck is therefore not force evaluation or WOUT I/O;
  it is specifically first-call 3D preconditioner seed construction, dominated
  by R/Z preconditioner matrix compilation/build and lambda preconditioner
  setup.

Best next steps:

1. Target the 3D JAX preconditioner seed path directly: reduce graph size in
   `_compute_preconditioning_matrix`, precompute invariant radial/mode factors,
   and share the matrix assembly executable across equivalent `mpol/ntor/ns`
   rows.
2. Add a preconditioner microbenchmark that times matrix coefficient assembly,
   R/Z matrix assembly, lambda preconditioner, and R/Z apply separately on QH
   and one LASYM case.
3. Re-run the full README single-grid matrix only after the 3D preconditioner
   path has a measured improvement or is explicitly deferred.

Updated lane percentages:

- Performance benchmark/profiling harness: 81%.
- Fixed-boundary production differentiability: 82%.
- Free-boundary production differentiability: 78%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 72%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 90%.
- Docs/release hygiene: 89%.
- Overall: 73%.

### 2026-06-21: Strict README AD-vs-FD evidence gate

Steps taken:

- Re-ran the same-branch direct-coil free-boundary smoke report:
  `JAX_ENABLE_X64=1 python examples/optimization/free_boundary_QS_coil_optimization.py --smoke --provider circle --outdir outputs/pr20_ad_fd/qs_same_branch --write-same-branch-report --same-branch-report-mode vector --same-branch-report-ad-mode direct --same-branch-report-direction current-only --same-branch-report-vector-keys aspect,qs_total,mean_iota,lcfs_boundary_moment --max-evals 1 --max-iter 1 --vmec-max-iter 3 --same-branch-report-max-iter 3`.
- Regenerated `docs/_static/figures/readme_ad_fd_evidence.{png,csv,json}` from that report.
- Tightened the evidence renderer so all checked-in rows use a `1e-9`
  relative slope-error tolerance, including the branch-local direct-coil
  free-boundary rows.

Results obtained:

- The evidence panel now has ten rows:
  fixed-boundary aspect, iota, QS residual, smooth QI residual, `DMerc`,
  `D_R`, and four same-branch direct-coil free-boundary scalars (`aspect`,
  `qs_total`, `mean_iota`, `lcfs_boundary_moment`).
- All rows pass the stricter `1e-9` gate. The largest relative error is the
  branch-local free-boundary `qs_total` row at `2.57e-10`; `DMerc` is
  `3.55e-13` and `D_R` is `2.32e-12`.
- Focused tests passed:
  `tests/test_glasser_resistive_interchange.py::test_public_dmerc_and_glasser_objective_gradients_match_central_finite_difference`,
  `tests/test_glasser_resistive_interchange.py::test_profile_integral_mercier_and_glasser_gradients_match_central_finite_difference`,
  `tests/test_free_boundary_qs_coil_optimization_smoke.py::test_branch_local_scalar_report_adapter_records_gate_evidence`, and
  `tests/test_free_boundary_qs_coil_optimization_smoke.py::test_branch_local_scalar_report_adapter_records_failure_modes`.

Best next steps:

1. Promote the full benchmark/parity gate next: current branch vs `origin/main`
   full single-grid matrix, VMEC2000 rows, VMEC++ rows where converged, and WOUT
   parity on the selected cases.
2. Keep the adaptive free-boundary wording conservative: this evidence remains
   same-branch/fingerprint-gated, not arbitrary adaptive branch
   differentiation.
3. Continue the 3D preconditioner seed optimization after the full benchmark
   identifies whether the same QH bottleneck appears across the matrix.

Updated lane percentages:

- Performance benchmark/profiling harness: 82%.
- Fixed-boundary production differentiability: 86%.
- Free-boundary production differentiability: 82%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 72%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 91%.
- Docs/release hygiene: 90%.
- Overall: 75%.

### 2026-06-21: PR #20 single-grid benchmark and WOUT parity gate

Steps taken:

- Stopped an accidentally launched 36-row `examples/data` benchmark after it
  reached long timeouts; that was not the historical README single-grid matrix
  required by the PR #20 readiness gate.
- Ran the intended current-branch matrix:
  `PYTHONPATH=$PWD JAX_ENABLE_X64=1 python tools/diagnostics/example_runtime_memory_matrix.py --inputs-dir examples_single_grid/data --kind fixed --backend all --warm-runs 1 --jax-platforms cpu --runner-label current-cpu --vmec-exec ~/bin/xvmec2000 --timeout-s 1800 --vmec-timeout-s 1800 --outdir outputs/pr20_full_matrix_current_cpu_sg`.
- Ran the matching clean `origin/main` matrix in
  `/Users/rogeriojorge/local/tests/vmec_jax_main_perf` with the same command
  and `--outdir outputs/pr20_full_matrix_main_cpu_sg`.
- Compared the two matrices:
  `python tools/diagnostics/compare_runtime_memory_matrix.py --current outputs/pr20_full_matrix_current_cpu_sg/summary.json --baseline /Users/rogeriojorge/local/tests/vmec_jax_main_perf/outputs/pr20_full_matrix_main_cpu_sg/summary.json --csv-out docs/_static/figures/readme_runtime_compare_current_vs_main.csv --json-out docs/_static/figures/readme_runtime_compare_current_vs_main.json`.
- Regenerated the docs benchmark artifact:
  `python tools/diagnostics/readme_runtime_compare.py --cpu-summary outputs/pr20_full_matrix_current_cpu_sg/summary.json --figure-kind fixed --plot-mode runtime_memory --figure-out docs/_static/figures/readme_runtime_compare.png --csv-out docs/_static/figures/readme_runtime_compare.csv --json-out docs/_static/figures/readme_runtime_compare.json --table-out outputs/readme_runtime_table_pr20.md`.
- Ran the executable-backed WOUT parity benchmark:
  `PYTHONPATH=$PWD JAX_ENABLE_X64=1 python tools/diagnostics/converged_wout_parity_benchmark.py --nightly --vmec-exec ~/bin/xvmec2000 --case nfp4_QH_warm_start --case solovev --case ITERModel --case LandremanPaul2021_QA_lowres --output-dir outputs/pr20_wout_parity`.
- Copied the compact parity summary to
  `docs/_static/figures/pr20_wout_parity_summary.json`.
- Updated docs wording so the runtime/memory plot is docs-facing, not a
  README headline artifact.

Results obtained:

- Current branch single-grid matrix: 16/16 `vmec_jax` and VMEC2000 rows
  succeeded.
- VMEC++ succeeded on 9/16 rows and is recorded as unavailable/non-converged on
  the other 7 rows.
- Warm `vmec_jax` beat VMEC2000 on 7/16 rows; cold `vmec_jax` beat VMEC2000 on
  2/16 rows.
- Median current-branch warm runtime ratio vs VMEC2000: `1.51x` slower.
- Median current-branch cold runtime ratio vs VMEC2000: `3.76x` slower.
- Median current-branch peak-memory ratio vs VMEC2000: `4.90x` higher.
- Current-vs-`origin/main` comparator: 48 backend/case rows, 0 regressions
  under the configured `1.10x` runtime and `1.15x` peak-memory thresholds.
- WOUT parity: all four promoted rows passed. Worst reported relative RMS
  channel was `bsubvmnc` on `solovev` at `4.37e-5`; LPQA and QH warm-start
  residual RSS values matched VMEC2000 to the reported precision.

Best next steps:

1. Run local docs and size gates on the updated docs/artifacts.
2. Keep PR #20 in draft until any reviewer-requested benchmark presentation
   changes are handled, but the core benchmark/regression/parity evidence gate
   is now complete.
3. Resume performance work on first-call 3D preconditioner seed construction:
   add seed sub-buckets, then test a fused/cached R/Z coefficient-plus-assembly
   path while preserving the current JAX path for differentiability.
4. Continue the refactor plan after PR readiness: split the large
   fixed-boundary residual iteration module into setup, step, controller, and
   history/output seams.

Updated lane percentages:

- Performance benchmark/profiling harness: 95%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 86%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 74%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 95%.
- Overall: 82%.

### 2026-06-21: 3D preconditioner seed sub-bucket instrumentation

Steps taken:

- Added residual-timing buckets for
  `precond_refresh_seed_lambda_s` and
  `precond_refresh_seed_rz_matrices_s`.
- Wired those buckets through `profile_fixed_boundary.py` and
  `fixed_boundary_performance_decomposition.py`.
- Added unit coverage for the new timing schema and deterministic seed
  sub-bucket accumulation.
- Ran the QH warm-start bounded no-scan profile:
  `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python tools/diagnostics/profile_fixed_boundary.py --input examples/data/input.nfp4_QH_warm_start --iters 2 --solver-mode default --solver-device cpu --finish-policy none --no-use-scan --require-no-scan --no-warmup --simple-profile --vmec-timing --vmec-timing-detail --outdir outputs/precond_seed_subbucket_trace --json-out outputs/precond_seed_subbucket.json`.

Results obtained:

- QH profile wall time: `1.224 s`; solver total: `1.010 s`.
- Preconditioner total: `0.924 s`.
- Preconditioner seed: `0.922 s`.
- Lambda seed: `0.077 s`.
- R/Z matrix seed: `0.845 s`.
- Therefore the cold 3D preconditioner bottleneck is R/Z matrix construction,
  not lambda setup or preconditioner apply.

Best next steps:

1. Target `vmec_jax/preconditioner_1d_jax.py` R/Z matrix construction:
   identify coefficient construction versus matrix assembly costs inside
   `rz_preconditioner_matrices`.
2. Prototype a shape-keyed cached/JIT-fused R/Z coefficient-plus-assembly path
   that preserves tracer/JAX differentiability.
3. Keep the axisymmetric NumPy shortcut isolated; do not extend it to 3D until
   output-equivalence tests prove the representation matches the JAX
   independent-toroidal-block path.

Updated lane percentages:

- Performance benchmark/profiling harness: 96%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 86%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 76%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 95%.
- Overall: 83%.

### 2026-06-21: Promote full-JIT R/Z preconditioner matrix seed path

Steps taken:

- Added a guarded full-JIT R/Z coefficient-and-assembly builder in
  `vmec_jax/preconditioner_1d_jax.py`, preserving the previous assembly-only
  JIT as a fallback.
- Benchmarked the QH warm-start bounded no-scan cold profile in fresh CPU
  processes with `VMEC_JAX_RZ_MATRIX_FULL_JIT=0` and
  `VMEC_JAX_RZ_MATRIX_FULL_JIT=1`.
- Promoted the full-JIT builder to the default, with
  `VMEC_JAX_RZ_MATRIX_FULL_JIT=0` retained as a diagnostics opt-out.
- Added unit coverage for the new environment flag and updated performance
  documentation.

Results obtained:

- Default/eager coefficient path: `1.250 s` wall, `1.022 s` solver total,
  `0.936 s` preconditioner seed, and `0.861 s` R/Z matrix construction.
- Full-JIT coefficient-plus-assembly path: `0.583 s` wall, `0.320 s` solver
  total, `0.232 s` preconditioner seed, and `0.155 s` R/Z matrix construction.
- Short-trace residual parity was unchanged for the diagnostic row
  (`final_w=0.021577021484175438` in both profiles).
- A fresh default run without an explicit full-JIT flag confirmed the promoted
  path is active for normal users: `0.585 s` wall, `0.313 s` solver total,
  `0.227 s` preconditioner seed, and `0.150 s` R/Z matrix construction.
- Validation passed:
  `python -m ruff check vmec_jax/preconditioner_1d_jax.py tests/test_preconditioner_1d_jax_fast_helpers.py tools/diagnostics/profile_fixed_boundary.py docs/conf.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q tests/test_preconditioner_1d_jax_fast_helpers.py tests/test_solve_finish_cache_more_coverage.py::test_nonscan_reuses_preconditioner_seed_from_same_bcovar_refresh tests/test_solve_performance_instrumentation.py -q`;
  `python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50 --max-file-mib 2`;
  `LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html_pr20_performance`.

Best next steps:

1. Re-run the historical single-grid current-branch matrix only if reviewers
   want the docs-facing performance figure to include this newest speedup;
   otherwise leave PR-readiness benchmark evidence intact and documented.
2. Continue runtime work on the next bottleneck after the R/Z matrix seed:
   setup axis reset/boundary-profile setup for cold solves, and steady
   preconditioner/update dispatch for longer warm solves.

Updated lane percentages:

- Performance benchmark/profiling harness: 97%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 86%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 80%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 95%.
- Overall: 85%.

### 2026-06-21: Refresh current-branch matrix after R/Z full-JIT promotion

Steps taken:

- Re-ran the historical 16-row bundled single-grid fixed-boundary matrix on the
  current branch after the full-JIT R/Z seed promotion:
  `PYTHONPATH=$PWD JAX_ENABLE_X64=1 python tools/diagnostics/example_runtime_memory_matrix.py --inputs-dir examples_single_grid/data --kind fixed --backend all --warm-runs 1 --jax-platforms cpu --runner-label current-cpu --vmec-exec ~/bin/xvmec2000 --timeout-s 1800 --vmec-timeout-s 1800 --outdir outputs/pr20_full_matrix_current_cpu_sg_fulljit`.
- Reused the clean `origin/main` summary in
  `/Users/rogeriojorge/local/tests/vmec_jax_main_perf/outputs/pr20_full_matrix_main_cpu_sg/summary.json`
  for current-vs-main regression comparison.
- Regenerated:
  `docs/_static/figures/readme_runtime_compare.png`,
  `docs/_static/figures/readme_runtime_compare.csv`,
  `docs/_static/figures/readme_runtime_compare.json`,
  `docs/_static/figures/readme_runtime_compare_current_vs_main.csv`,
  and `docs/_static/figures/readme_runtime_compare_current_vs_main.json`.

Results obtained:

- Matrix completion: 16/16 `vmec_jax` rows and 16/16 VMEC2000 rows succeeded;
  VMEC++ was available on 9/16 rows.
- Current-vs-`origin/main`: 48 backend/case rows, 0 material regressions under
  the configured `1.10x` runtime and `1.15x` memory thresholds.
- Warm `vmec_jax` beat VMEC2000 on 7/16 rows; cold `vmec_jax` beat VMEC2000 on
  2/16 rows.
- Median warm runtime ratio vs VMEC2000 improved to `1.41x` slower.
- Median cold runtime ratio vs VMEC2000 improved to `2.93x` slower.
- Median peak process-memory ratio vs VMEC2000 improved to `4.44x` higher.
- Notable remaining blockers: low-latency axisymmetric rows are still dominated
  by startup/setup overhead, and the LASYM finite-beta row remains the largest
  memory ratio.

Best next steps:

1. Run docs/size/lint gates again after the regenerated benchmark artifacts.
2. Continue performance work on cold setup (`setup_axis_reset_s`,
   `setup_boundary_profiles_s`) and LASYM memory layout.
3. Keep the README without the runtime/memory plot; the regenerated matrix is
   now docs-facing evidence only.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 86%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 82%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 95%.
- Overall: 86%.

### 2026-06-22: PR #20 audit after benchmark/readiness refresh

Steps taken:

- Rechecked branch state after the regenerated full single-grid matrix,
  preconditioner speedup, WOUT parity evidence, and AD-vs-FD evidence pushes.
- Confirmed the working tree was clean before this plan update and that the
  active branch is `codex/differentiability-refactor-plan`.
- Queried PR #20 status with `gh pr view 20 --json ...`.
- Reran `tools/diagnostics/source_health.py --top 20 --top-functions 40` to
  identify the next maintainability blockers.
- Verified the README does not reference the runtime/memory benchmark figure;
  the full matrix remains docs-facing in `docs/performance.rst`.

Results obtained:

- PR #20 is currently `isDraft=false`; early CI shards passed and longer
  py3.11 coverage/physics shards were still running at the audit time.
- The largest production source-health blocker remains
  `vmec_jax/solvers/fixed_boundary/residual/iteration.py` (`3120` lines) and
  its `solve_fixed_boundary_residual_iter` function (`2645` lines).
- Other large production files remain below the immediate 2000-line warning
  threshold or are close enough to require later tranches:
  `optimization.py`, `preconditioner_1d_jax.py`, `vmec_forces.py`,
  `plotting.py`, `free_boundary.py`, `qi_optimization.py`, and
  `io/wout/minimal.py`.
- Oversized exact/free-boundary test files are also real maintainability debt,
  but the next production refactor should address the fixed-boundary iteration
  seam first because it controls runtime, differentiability, and API clarity.

Best next steps:

1. Profile the post-full-JIT QH warm-start path one level deeper around cold
   setup and axis/boundary/profile setup, then target the largest remaining
   measured setup bucket.
2. Start the fixed-boundary iteration refactor by extracting input/static
   setup, convergence policy, and history/output assembly from the giant
   residual iteration function before changing numerical kernels.
3. Keep PR #20 ready only if the user wants the evidence gates reviewed now;
   otherwise convert it back to draft before the larger refactor tranches.
4. Continue CI checks later and only react to concrete failed shards.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 86%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 82%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 95%.
- Overall: 86%.

### 2026-06-22: Reduce host flux-reconciliation startup cost

Steps taken:

- Profiled the post-full-JIT QH warm-start path with cProfile and VMEC timing:
  `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python tools/diagnostics/profile_fixed_boundary.py --input examples/data/input.nfp4_QH_warm_start --iters 2 --solver-mode default --solver-device cpu --finish-policy none --no-use-scan --require-no-scan --no-warmup --simple-profile --vmec-timing --vmec-timing-detail --cprofile-out outputs/post_fulljit_qh_cold_setup_profile.prof --cprofile-text-out outputs/post_fulljit_qh_cold_setup_profile.txt --outdir outputs/post_fulljit_qh_cold_setup_trace --json-out outputs/post_fulljit_qh_cold_setup.json`.
- Added a NumPy host implementation of VMEC `add_fluxes` iota smoothing in
  `_iotaf_from_iotas`, while keeping the JAX implementation for traced/autodiff
  inputs.
- Added a regression test proving traced inputs still use a differentiable JAX
  path.
- Reran the same fixed-boundary profile after the host fast path.

Results obtained:

- Before the host fast path, the short QH profile took `0.806 s` wall time;
  after the change it took `0.654 s`, with unchanged `final_w`.
- The solve body remained about `0.45 s`; the improvement is in host-side
  startup/post-solve reconciliation rather than force kernels.
- cProfile showed `finalize_flux_profiles_for_run` dropped from about
  `0.298 s` to about `0.150 s`.
- JAX compilation count in the cProfile table dropped from `30` to `19` calls,
  because host iota smoothing no longer uses JAX scatter/update for ordinary
  NumPy arrays.
- Validation passed:
  `python -m ruff check vmec_jax/energy.py tests/test_energy_fast_helpers.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q tests/test_energy_fast_helpers.py tests/test_fast_physics_kernels.py tests/test_driver_fast_reconstruction.py -q`;
  `git diff --check`;
  `python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50 --max-file-mib 2`;
  `LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs docs/_build/html_pr20_iotaf_fastpath`.
- A finite-beta short profile:
  `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python tools/diagnostics/profile_fixed_boundary.py --input examples/data/input.nfp4_QH_finite_beta --iters 2 --solver-mode default --solver-device cpu --finish-policy none --no-use-scan --require-no-scan --no-warmup --simple-profile --vmec-timing --vmec-timing-detail --outdir outputs/post_iotaf_numpy_finite_beta_trace --json-out outputs/post_iotaf_numpy_finite_beta.json`
  reported `6.83 s` wall time with `3.83 s` setup and `2.49 s` iteration
  loop. The dominant setup buckets were axis reset (`0.95 s`), index constants
  (`0.42 s`), and boundary profiles (`0.37 s`).

Best next steps:

1. Continue host startup profiling in `final_flux_profiles_from_state`; the
   next exposed cost is `vmec_pwint_from_trig`/quadrature table construction in
   the current-driven post-solve recomputation path.
2. Prioritize finite-beta setup staging and index-constant reuse before tuning
   tiny low-mode rows further; the finite-beta profile shows setup can exceed
   the iteration loop for short runs.
3. Keep traced/JAX paths covered whenever host NumPy shortcuts are added.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 86%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 83%.
- Refactor/API/examples: 41%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 95%.
- Overall: 86%.

### 2026-06-22: Fix PR #20 CI failures after readiness refresh

Steps taken:

- Pulled failed CI logs for `Fast Tests (py3.11 core coverage: rest)` and
  `Fast Tests (py3.11 exact coverage: remaining)`.
- Updated the README hygiene test to enforce the new product decision: the
  runtime/memory figure must be absent from the README and remain docs-facing.
- Made `tools/diagnostics/profile_fixed_boundary.py` robust when unit tests
  call `_summarize_run()` directly without first attaching
  `effective_finish_policy` in `main()`.
- Fixed dynamic replay JVP/VJP structure handling when a refreshed
  preconditioner matrix payload has the full matrix-key set while the cached
  trace carry has a minimal matrix-key set.
- Added cotangent padding for auxiliary replay cache dictionaries so VJP calls
  receive a cotangent pytree matching the differentiated step output.

Results obtained:

- The local reproduction of the failed core-rest shard now passes:
  `658 passed, 1 skipped`.
- The local reproduction of the failed exact-remaining shard now passes:
  `29 passed`.
- The direct failing tests now pass:
  `tests/test_docs_release_hygiene.py::test_root_readme_stays_concise_and_defers_extended_claims`,
  the four profiler summary tests in
  `tests/test_gpu_cpu_performance_profile.py`, and the two exact replay tests
  in `tests/test_discrete_adjoint_qh.py` / `tests/test_boundary_field.py`.

Best next steps:

1. Push these fixes and let CI rerun.
2. If CI is green, PR #20 can be reviewed as a readiness/refactor umbrella with
   the runtime plot intentionally absent from README.
3. Continue the next performance tranche on finite-beta setup staging and
   LASYM memory only after CI is green.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 83%.
- Refactor/API/examples: 42%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 87%.

### 2026-06-22: Split fixed-boundary profile/trig setup timing

Steps taken:

- Kept PR #20 in draft after user direction while continuing the active
  performance/refactor plan.
- Added profile-data and trig-table sub-buckets under the existing
  `setup_boundary_profiles` timing bucket:
  `setup_profile_data_s`, `setup_trig_tables_s`, and
  `setup_boundary_profiles_unattributed_s`.
- Wired the residual-loop timing accumulator through
  `build_residual_profile_setup` without changing default behavior when timing
  is disabled.
- Updated the compact fixed-boundary profiler output so these new setup costs
  are visible in terminal output and JSON provenance.
- Added focused helper/report tests for the new timing seam.

Results obtained:

- Focused validation passed:
  `python -m ruff check tools/diagnostics/profile_fixed_boundary.py vmec_jax/solvers/fixed_boundary/residual/runtime.py vmec_jax/solvers/fixed_boundary/residual/setup.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_runtime_helpers.py tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_performance_instrumentation.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_runtime_helpers.py tests/test_solve_residual_iter_setup_helpers.py tests/test_solve_performance_instrumentation.py -q`;
  `JAX_ENABLE_X64=1 python -m pytest -q tests/test_gpu_cpu_performance_profile.py -q`.
- Bounded finite-beta profile command:
  `JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python tools/diagnostics/profile_fixed_boundary.py --input examples/data/input.nfp4_QH_finite_beta --iters 2 --solver-mode default --solver-device cpu --finish-policy none --no-use-scan --require-no-scan --no-warmup --simple-profile --vmec-timing --vmec-timing-detail --outdir outputs/profile_setup_split_finite_beta_trace2 --json-out outputs/profile_setup_split_finite_beta2.json`.
- The final finite-beta stage reported `setup_boundary_profiles_s=0.408988`,
  `setup_profile_data_s=0.391885`, `setup_trig_tables_s=5.4e-06`, and
  `setup_boundary_profiles_unattributed_s=0.0171`.
- This narrows the next M3 target: profile-data construction dominates the
  boundary/profile setup bucket; trig-table construction is not a meaningful
  cold-path bottleneck for this row.

Best next steps:

1. Split profile-data construction further into pressure/current profile
   evaluation, lambda/iota smoothing, and WOUT-like object construction.
2. Attack `setup_axis_reset_s` next for finite-beta rows, since it remains
   about `0.94 s` in the bounded profile and includes a force evaluation.
3. Continue the larger M7 fixed-boundary residual split by moving setup
   orchestration into a dedicated module once the remaining setup sub-buckets
   are stable.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 84%.
- Refactor/API/examples: 43%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 87%.

### 2026-06-22: Enable host profile setup automatically for profile-heavy CPU decks

Steps taken:

- Used cProfile on the bounded QH finite-beta probe to inspect
  `build_wout_like_profiles_from_indata` internals.
- Found that the `setup_profile_data_s` cost came mostly from small JAX
  operations in mass/current profile setup, not from input parsing or trig
  tables.
- Verified the existing `VMEC_JAX_HOST_PROFILE_SETUP=1` path on the same
  finite-beta row and on the low-mode QH warm-start row.
- Added a pure config helper, `indata_has_profile_setup_work`, that detects
  finite-beta pressure profiles, current-driven profiles, explicit iota
  profiles, RFP mode, and non-default toroidal-flux profiles.
- Changed `resolve_host_profile_setup(..., profile_setup_env="auto")` so CPU
  solves keep the old default for simple decks but automatically use host
  profile setup for profile-heavy concrete decks. Traced/autodiff inputs still
  disable the NumPy patch inside `build_residual_profile_setup`.
- Added unit coverage for the new policy and profile-work detection.

Results obtained:

- With no env override, the bounded QH finite-beta profile now reports
  `setup_profile_data_s=0.00112` and `setup_boundary_profiles_s=0.01777`.
  Before the policy change, the same probe reported approximately
  `setup_profile_data_s=0.392` and `setup_boundary_profiles_s=0.409`.
- The low-mode QH warm-start row remains on the cheap default CPU path and
  still reports sub-millisecond `setup_profile_data_s`, with no useful benefit
  from forcing host profile setup.
- Focused validation passed:
  `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/config.py vmec_jax/solvers/fixed_boundary/residual/iteration.py tests/test_solve_residual_iter_config.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_residual_iter_config.py tests/test_solve_performance_instrumentation.py tests/test_solve_residual_iter_setup_helpers.py -q`.

Best next steps:

1. Attack `setup_axis_reset_s` on finite-beta rows. It remains about `0.94 s`
   and includes a force evaluation; this is now the largest setup bucket.
2. Continue reducing preconditioner seed/apply costs, especially
   `precond_refresh_seed_rz_matrices_s` and `precond_apply_s`.
3. Promote this profile-heavy CPU policy into the docs performance caveats only
   after one more full matrix confirms no broad regression.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 85%.
- Refactor/API/examples: 43%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 88%.

### 2026-06-22: Reuse setup axis force probe on strict no-reset branches

Steps taken:

- Surfaced setup-time axis reset branch provenance in solver diagnostics:
  `setup_axis_reset_applied`, `setup_axis_reset_done`,
  `setup_axis_force_probe_available`, and
  `setup_axis_force_probe_reused`.
- Extended the fixed-boundary profiler compact JSON and terminal timing output
  with the same branch evidence and `compute_forces_main_reuse_count`.
- Kept the VMEC-style setup axis force probe when no axis reset is applied.
- Reused that force payload for the first fixed-boundary iteration only under a
  strict predicate: iteration 1, no actual axis reset, fixed-boundary only, no
  edge residual, no free-boundary vacuum payload, no cached constraint
  preconditioner/tcon, `zero_m1=1`, and no debug dump index.
- Added helper/finalizer/profiler/timing-report tests for the new provenance and
  reuse counter.

Results obtained:

- Focused validation passed:
  `python -m ruff check vmec_jax/solvers/fixed_boundary/diagnostics/axis_reset.py vmec_jax/solvers/fixed_boundary/residual/iteration.py vmec_jax/solvers/fixed_boundary/residual/finalize.py vmec_jax/solvers/fixed_boundary/residual/runtime.py tools/diagnostics/profile_fixed_boundary.py tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_gpu_cpu_performance_profile.py tests/test_solve_performance_instrumentation.py tests/test_solve_residual_iter_runtime_helpers.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_axis_helpers_more_coverage.py tests/test_solve_residual_iter_finalize_helpers.py tests/test_gpu_cpu_performance_profile.py tests/test_solve_performance_instrumentation.py tests/test_solve_residual_iter_runtime_helpers.py -q`.
- Bounded QH finite-beta profile with the same command as the previous entry now
  reports `setup_axis_force_probe_reused=True`,
  `compute_forces_main_reuse_count=1`, and `compute_forces_s=0` for the final
  one-iteration stage.
- The same diagnostic retained identical residual scalars:
  `final_fsqr=0.07575699495805738`,
  `final_fsqz=0.04928241321537203`, and
  `final_fsql=0.04329198826597721`.
- Bounded finite-beta total wall time improved from `6.36 s` to `5.78 s`; final
  stage `solve_total_s` improved from `5.70 s` to `5.11 s`.

Best next steps:

1. Continue M3 with the now-dominant finite-beta costs:
   `precond_apply_s`, `precond_refresh_seed_rz_matrices_s`, and
   `update_state_s`.
2. Add a small parity/profile gate comparing first-iteration residual histories
   with and without setup-probe reuse if a debug opt-out flag is introduced.
3. Continue M7 by extracting a domain-named force-evaluation/reuse seam from
   the large residual loop, instead of leaving the predicate embedded in the
   2600-line controller.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 86%.
- Refactor/API/examples: 44%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 89%.

### 2026-06-22: Decouple NumPy force evaluation from CPU host-update assembly

Steps taken:

- Added `VMEC_JAX_NUMPY_FORCE_FAST_PATH` and
  `VMEC_JAX_NUMPY_FORCE_MAX_ITER` policy controls so CPU host-update assembly
  no longer forces NumPy force evaluation on long stages.
- Kept the default conservative for first-run CPU performance: host update
  remains enabled for moderate grids, NumPy force evaluation is used only for
  short stages (`max_iter <= 600`), and long stages use compiled JAX force
  kernels with host update/preconditioner assembly.
- Exposed `numpy_force_fast_path`, `numpy_force_fast_path_active`, and
  `numpy_force_fast_path_max_iter` in residual diagnostics and profiler JSON.
- Reprofiled the finite-beta QH row in three modes:
  host+NumPy force, JIT update, and host update + JAX force.

Results obtained:

- Full warmed finite-beta final-stage solve times:
  host+NumPy force `11.47 s`, JIT-update route `9.53 s`, new auto hybrid
  route `7.19 s`.
- Actual cold CLI-style finite-beta solve with new auto policy reports
  `10.00 s` total computational time, compared with ~`23.7 s` from the
  earlier JIT route and `14.92 s` from host+NumPy force.
- The printed convergence trace and final residual for
  `input.nfp4_QH_finite_beta` remain unchanged to displayed precision.

Best next steps:

1. Run the full single-grid benchmark matrix to confirm this hybrid policy
   improves or preserves other rows beyond finite-beta QH.
2. Attack the remaining long-stage hotspots: JAX force kernels
   (`compute_forces_s ~= 3.9 s`) and preconditioner apply
   (`precond_apply_s ~= 2.0 s`), using VMEC2000 `funct3d/bcovar/forces` and
   vmec++ `fourier_geometry/fourier_forces` as the locality reference.
3. Add a compact benchmark provenance table that records
   `host_update_assembly`, `numpy_force_fast_path_active`, and
   `numpy_preconditioner_apply` for each row.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 89%.
- Refactor/API/examples: 47%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 90%.

### 2026-06-22: Compare current README matrix against origin/main

Steps taken:

- Ran the same 16-row historical README CPU matrix in a clean
  `/Users/rogeriojorge/local/tests/vmec_jax_main_perf` worktree pinned to
  `origin/main`.
- Compared current PR branch versus `origin/main` using
  `tools/diagnostics/compare_runtime_memory_matrix.py`.

Results obtained:

- Baseline matrix:
  `/Users/rogeriojorge/local/tests/vmec_jax_main_perf/outputs/pr20_readme_matrix_main_cpu/summary.json`.
- Current-vs-main comparison:
  `outputs/pr20_readme_matrix_current_vs_main.json` and
  `outputs/pr20_readme_matrix_current_vs_main.csv`.
- The comparison reported `rows=32 regressions=0`; no row crossed the current
  runtime/memory regression thresholds.
- The PR branch is therefore not regressing the historical CPU matrix. The
  remaining performance lane is absolute runtime and memory versus VMEC2000,
  especially 3D force kernels and peak process memory.

Best next steps:

1. Keep PR #20 benchmark readiness focused on absolute VMEC2000 gaps and
   provenance/README presentation, not regression repair.
2. Profile `LandremanPaul2021_QA_lowres` and
   `LandremanPaul2021_QH_reactorScale_lowres` with detailed timing to isolate
   the remaining 3D kernel gap.
3. Profile `basic_non_stellsym_pressure` peak memory and LASYM array
   allocations to reduce the largest memory ratio.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 90%.
- Refactor/API/examples: 47%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 90%.

### 2026-06-22: Run current-branch historical README CPU matrix

Steps taken:

- Re-ran the 16 historical fixed-boundary README benchmark rows from
  `docs/_static/figures/readme_runtime_compare.csv` on the current PR branch.
- Used the CLI-style production policy explicitly:
  `tools/diagnostics/example_runtime_memory_matrix.py --backend both
  --solver-mode default --warm-runs 1 --jax-platforms cpu`.
- Ran VMEC++ on the same 16 rows with the local `vmecpp` CLI to identify which
  rows are supported and converged.

Results obtained:

- Current branch CPU matrix summary:
  `outputs/pr20_readme_matrix_current_cpu/summary.json`.
- Warm vmec_jax is now close to VMEC2000 for several bundled rows:
  `nfp4_QH_warm_start` `0.45 s` vs VMEC2000 `0.33 s`,
  `basic_non_stellsym_pressure` `1.34 s` vs `1.06 s`,
  `ITERModel` `0.98 s` vs `0.81 s`, and `cth_like_fixed_bdy`
  `0.29 s` vs `0.41 s` (vmec_jax faster).
- Harder 3D rows remain slower but within a smaller factor than earlier:
  Landreman-Paul QA/QH rows are roughly `1.45x` to `2.09x` VMEC2000 in warm
  runtime.
- Memory remains a clear open lane: vmec_jax CPU peak process memory is
  roughly `1.9x` to `6.3x` VMEC2000 on this matrix.
- VMEC++ is available locally and succeeds on 7/16 rows in this environment:
  `ITERModel`, `circular_tokamak`, `circular_tokamak_aspect_100`,
  `nfp4_QH_warm_start`, `purely_toroidal_field`,
  `shaped_tokamak_pressure`, and `solovev`. Other rows are unsupported or do
  not converge cleanly and should be shown as unavailable, not as failures.

Best next steps:

1. Run the same 16-row matrix against clean `origin/main` and compare cold,
   warm, memory, and convergence row-by-row.
2. Profile remaining runtime gaps on the Landreman-Paul 3D rows and memory
   gaps on `basic_non_stellsym_pressure`/large axisymmetric rows.
3. Refresh README/docs benchmark artifacts only after the current-vs-main
   comparison is complete and the README memory-plot policy is finalized.

Updated lane percentages:

- Performance benchmark/profiling harness: 99%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 90%.
- Refactor/API/examples: 47%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 90%.

### 2026-06-22: Add benchmark execution-policy provenance

Steps taken:

- Extended `tools/diagnostics/example_runtime_memory_matrix.py` so each
  vmec_jax row records `execution_policy_cold` and final/warm
  `execution_policy`, including host-update, NumPy-force, NumPy-preconditioner,
  strict-update JIT, and scan policy fields.
- Ran a one-case finite-beta QH benchmark slice with and without
  `--solver-mode default` to separate the public Python API default from the
  CLI-style production policy.

Results obtained:

- Bare `run_fixed_boundary(...)` benchmark slice:
  cold `20.46 s`, warm `9.55 s`, policy `host_update_assembly=False`.
- CLI-style/default-mode benchmark slice:
  cold `12.27 s`, warm `8.06 s`, policy `host_update_assembly=True`,
  `numpy_preconditioner_apply=True`, `numpy_force_fast_path=False`.
- The README benchmark matrix should use the CLI-style production path when it
  is meant to explain `vmec` command-line performance; Python API comparisons
  should be labeled separately.

Best next steps:

1. Regenerate the full README benchmark matrix with explicit
   `--solver-mode default` for the CLI-style performance claim.
2. Decide whether bare `run_fixed_boundary(...)` should adopt the production
   default mode, or stay conservative and require `solver_mode="default"` for
   CLI-parity performance.
3. Add the execution-policy fields to the detailed docs benchmark table, not to
   the compact README figure.

Updated lane percentages:

- Performance benchmark/profiling harness: 99%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 89%.
- Refactor/API/examples: 47%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 90%.

### 2026-06-22: Classify finite-beta CPU update policy and raise host-update cutoff

Steps taken:

- Added residual-iteration diagnostics for the resolved update/preconditioner
  execution policy: `host_update_assembly`, `jit_strict_update_enabled`, JIT
  work size, and NumPy preconditioner apply policy.
- Updated `tools/diagnostics/profile_fixed_boundary.py` so compact JSON
  summaries retain those policy fields.
- Profiled `examples/data/input.nfp4_QH_finite_beta` on CPU using the
  non-scan VMEC2000-style residual loop with detailed timing.
- Raised the public CPU host-update default cutoff from 1000 to 4096 work
  units so the bundled finite-beta QH final stage uses VMEC2000-like host
  update/preconditioner assembly on first-run CPU solves.
- Updated the driver policy tests to assert that the finite-beta row uses host
  update by default, while a much larger synthetic grid still routes to the
  strict-update JIT unless the user overrides the cutoff.

Results obtained:

- Short/cold diagnostic before the policy change: final-stage solve
  `5.13 s`, `host_update_assembly=False`, `numpy_preconditioner_apply=False`,
  `precond_apply_s=0.283`, `update_state_s=0.114`.
- Short/cold diagnostic after the policy change: final-stage solve `0.36 s`,
  `host_update_assembly=True`, `numpy_preconditioner_apply=True`,
  `precond_apply_s=0.00138`, `update_state_s=0.00031`.
- Full CLI-style finite-beta solve with the new default completed in
  `14.92 s` on this machine, improving the previously observed ~`23.7 s`
  first-run CPU behavior while preserving the convergence trace and final
  residual.
- A warmed full-profile comparison shows the opposite steady-state tradeoff:
  forcing the JIT-update route gives final-stage `solve_total_s=9.53`, whereas
  the host route gives `11.47`. This confirms the next performance target is a
  budget/cache-aware policy, not a single global cutoff.
- Targeted validation passed:
  `python -m ruff check vmec_jax/drivers/policy.py
  vmec_jax/solvers/fixed_boundary/residual/finalize.py
  tools/diagnostics/profile_fixed_boundary.py tests/test_driver_api.py` and
  `JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_driver_api.py::test_host_update_assembly_matches_jax_update_path
  tests/test_driver_api.py::test_host_update_assembly_matches_jax_update_path_lasym
  tests/test_driver_api.py::test_host_update_assembly_driver_default_env_override
  -q`.

Best next steps:

1. Implement a budget-aware CPU update policy seam that chooses host assembly
   for cold/short stages and JIT update for long warmed stages, with explicit
   provenance in benchmark JSON.
2. Continue profiling the remaining finite-beta full-run buckets:
   `compute_forces_s` (~8 s host route, ~3.9 s warmed JIT route),
   `precond_apply_s`, and host-control reductions.
3. Compare the full single-grid benchmark matrix with and without the raised
   cutoff before promoting PR #20 from draft.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 88%.
- Refactor/API/examples: 47%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 89%.

### 2026-06-22: Split preconditioner apply timing and compile m=1 RHS scaling

Steps taken:

- Split the VMEC2000 preconditioner apply bucket into:
  `precond_apply_scale_m1_rhs_s`, `precond_apply_rz_s`,
  `precond_apply_fused_payload_s`, `precond_apply_output_blocks_s`, and
  `precond_apply_sync_s`.
- Added these sub-buckets to the fixed-boundary profiler output and timing
  report tests.
- Added a small JIT-compiled channel scaler for the VMEC m=1 R/Z preconditioner
  RHS path, replacing the eager JAX slice-scaling calls in concrete JAX
  preconditioner applies. The NumPy host path is unchanged.
- Tested existing tridiagonal policy toggles on the promoted bounded QH
  finite-beta row before changing defaults.

Results obtained:

- Focused validation passed:
  `python -m ruff check vmec_jax/solvers/fixed_boundary/preconditioning/operators.py vmec_jax/solvers/fixed_boundary/residual/preconditioner_payload.py vmec_jax/solvers/fixed_boundary/residual/runtime.py tools/diagnostics/profile_fixed_boundary.py tests/test_solve_performance_instrumentation.py tests/test_solve_preconditioner_payload_helpers.py tests/test_solve_preconditioner_metric_helpers.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q tests/test_solve_preconditioner_payload_helpers.py tests/test_solve_preconditioner_metric_helpers.py tests/test_solve_performance_instrumentation.py -q`.
- Bounded QH finite-beta profile before the compiled m=1 scaler:
  `precond_apply_s=0.3593`,
  `precond_apply_scale_m1_rhs_s=0.1972`, and
  `precond_apply_rz_s=0.1379`.
- With the compiled m=1 scaler, the same row reports
  `precond_apply_s=0.2740`,
  `precond_apply_scale_m1_rhs_s=0.1127`, and
  `precond_apply_rz_s=0.1374`, with unchanged residual scalars.
- `VMEC_JAX_TRIDI_SOLVE=1` reduced `precond_apply_rz_s` to about `0.0766`,
  but increased setup/axis-reset costs and total wall time; it is not a default
  win for this row.
- `VMEC_JAX_TRIDI_PRECOMPUTE=1` also worsened total wall time for this bounded
  finite-beta row, so preconditioner tridiagonal defaults remain unchanged.

Best next steps:

1. Continue M3 by attacking `precond_refresh_seed_rz_matrices_s` and
   `update_state_s`, since R/Z apply policy toggles did not improve total wall
   for this row.
2. Promote the preconditioner sub-buckets into the performance decomposition
   table after one full-matrix rerun confirms they are stable.
3. Continue M7 by moving the preconditioner apply/reuse predicates into a
   smaller domain seam, keeping the residual controller focused on VMEC branch
   policy rather than payload mechanics.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 87%.
- Refactor/API/examples: 45%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 89%.

### 2026-06-22: Replace per-NFP QI wrappers with direct QP-to-QI examples

Steps taken:

- Replaced `examples/optimization/QI_optimization_nfp1.py` through
  `QI_optimization_nfp4.py` with self-contained scripts that follow the
  `QP_optimization.py` workflow:
  circular/minimal seed, deterministic `input.simple_seed`, visible QP
  objective tuple list, QP solve, visible QI objective tuple list, QI solve,
  result saving, and raw-seed-to-final plots.
- Removed the now-unused `examples/optimization/qi_minimal_seed_example_common.py`
  subprocess/JSON wrapper helper. The public per-NFP scripts no longer delegate
  to `QI_optimization.py`, use reference-family preconditioners, or expose
  environment/CLI overrides.
- Updated `README.md`, `examples/optimization/README.md`, and
  `docs/optimization.rst` so the documented public QI path is the simple
  QP-to-QI path, while the staged QI driver remains documented as a lower-level
  stress/robustness path.
- Updated `tests/test_optimization_examples.py` to validate the new scripts
  statically rather than importing them and accidentally launching full
  optimizations.
- Revisited VMEC2000 and vmec++ source code for the performance lane:
  VMEC2000 `eqsolve/evolve` calls `funct3d` in a tight in-place loop, turns on
  2D preconditioning only after a residual threshold, and stores timing at
  routine granularity; vmec++ organizes equivalent work into reusable Fourier
  basis, geometry, radial-profile, force, NESTOR, and flow-control objects with
  explicit partial-update flags.

Results obtained:

- Fast validation passed:
  `python -m py_compile examples/optimization/QI_optimization_nfp1.py
  examples/optimization/QI_optimization_nfp2.py
  examples/optimization/QI_optimization_nfp3.py
  examples/optimization/QI_optimization_nfp4.py`;
  `python -m ruff check tests/test_optimization_examples.py
  examples/optimization/QI_optimization_nfp1.py
  examples/optimization/QI_optimization_nfp2.py
  examples/optimization/QI_optimization_nfp3.py
  examples/optimization/QI_optimization_nfp4.py`;
  `python -m pytest -q tests/test_optimization_examples.py -q`;
  `SPHINX_FAST=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -b html docs
  docs/_build/html_qi_examples_fast`;
  `LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_qi_examples_full`.
- The new per-NFP defaults preserve the reviewed direct-mode budgets used by
  the minimal-seed lane: NFP1 uses `max_mode=3`, NFP2 uses `max_mode=5`, NFP3
  uses `max_mode=4`, and NFP4 uses `max_mode=3`.
- The source audit reinforces the next runtime target: move more hot-path
  mechanics into reusable, domain-named seams with shape-stable buffers and
  partial-update flags, matching VMEC2000/vmec++ locality while retaining the
  differentiable Python API.

Best next steps:

1. Run at least one low-budget per-NFP QI script on CPU/GPU after the next
   optimization window to verify the simple QP-to-QI path still reaches the
   documented basin without relying on deleted wrapper preconditioners.
2. Continue M3/M7 by extracting a `forces`/`state_update` seam from the large
   fixed-boundary residual controller, using VMEC2000 `evolve` and vmec++
   `fourier_geometry`/`fourier_forces` as the source-structure reference.
3. Continue performance profiling on `precond_refresh_seed_rz_matrices_s`,
   `update_state_s`, and cold trace/setup cost, because those are now the
   dominant bounded finite-beta costs after the setup-force and m=1 RHS scaling
   wins.

Updated lane percentages:

- Performance benchmark/profiling harness: 98%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 87%.
- Refactor/API/examples: 47%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 89%.

### 2026-06-22: Disable long spectral NumPy preconditioner apply by default

Steps taken:

- Profiled the historical README matrix representative
  `input.LandremanPaul2021_QA_lowres` after the current/main matrix comparison
  showed no PR regression but a persistent absolute gap to VMEC2000.
- Compared the default CPU host-update path with the NumPy R/Z preconditioner
  apply enabled against the compiled JAX preconditioner apply path by setting
  `VMEC_JAX_NUMPY_PRECOND_MAX_ITER=0` and
  `VMEC_JAX_NUMPY_PRECOND_MIN_MODES=999999`.
- Repeated the same controlled comparison on `input.nfp4_QH_finite_beta`, the
  finite-beta row that motivated the recent host-update policy work.
- Changed the default `VMEC_JAX_NUMPY_PRECOND_MIN_MODES` from `16` to `0`.
  The pure-NumPy preconditioner apply path remains available for short solves
  through `VMEC_JAX_NUMPY_PRECOND_MAX_ITER=240`, and advanced users can still
  opt into the spectral NumPy path by setting
  `VMEC_JAX_NUMPY_PRECOND_MIN_MODES` explicitly.
- Updated the policy tests and `docs/performance.rst` so they describe the new
  default as a short-solve policy, not a long-spectral policy.

Results obtained:

- `input.LandremanPaul2021_QA_lowres` final-stage controlled profile:
  default-before-change `solve_total_s=5.1146`, `preconditioner_s=1.3453`,
  `precond_apply_rz_s=1.1600`; compiled JAX preconditioner path
  `solve_total_s=4.1692`, `preconditioner_s=0.3640`,
  `precond_apply_rz_s=0.1783`.
- `input.nfp4_QH_finite_beta` final-stage controlled profile:
  default-before-change `solve_total_s=7.4904`, `preconditioner_s=2.2866`,
  `precond_apply_rz_s=1.9783`; compiled JAX preconditioner path
  `solve_total_s=5.6115`, `preconditioner_s=0.5649`,
  `precond_apply_rz_s=0.2599`.
- After the default change, the normal production profile uses the faster path:
  LP-QA final-stage `solve_total_s=4.1482` and finite-beta QH final-stage
  `solve_total_s=6.0958`. The convergence scalars are unchanged to the
  printed tolerance.

Best next steps:

1. Rerun the full README current CPU matrix with the new default and compare to
   the previous current matrix to quantify row-by-row wall-time improvements.
2. Continue the absolute-performance lane by profiling `compute_forces_s`,
   which is now the dominant per-iteration cost on LP-QA and finite-beta QH.
   The VMEC2000/vmec++ source audit points to reusable Fourier/geometry buffers
   and in-place force assembly as the next architectural target.
3. Keep the README benchmark artifact deferred until the presentation policy is
   finalized, but keep benchmark JSON/CSV provenance under ignored `outputs/`.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 91%.
- Refactor/API/examples: 47%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 90%.

### 2026-06-22: Fuse bcovar norm and metric-scale reductions

Steps taken:

- Reran the historical fixed-boundary README CPU matrix after the preconditioner
  policy change and compared it against the previous current-branch matrix.
- Spawned two read-only explorers to audit the force-loop architecture:
  one inspected VMEC2000/vmec++ force, Fourier, preconditioner, buffer, and
  profiling structure; the other inspected vmec_jax force-payload hotspots.
- Added `vmec_force_norms_scales_from_bcovar_dynamic`, a JAX-traceable helper
  that returns VMEC force norms plus R/Z and lambda metric preconditioner scales
  from one angular sweep over `bcovar`.
- Wired the default residual force-payload path to use the fused helper while
  retaining the previous separate `norms_func`/`scale_func` injection path for
  diagnostics and tests.
- Added a focused fused-vs-separate unit test, including an autodiff gradient
  sanity check through the fused helper.

Results obtained:

- Full current CPU matrix after the preconditioner policy change had no
  regressions versus the previous current matrix. Warm 3D row improvements:
  LP-QA lowres `9.08 s -> 7.20 s` (-20.8%),
  LP-QA lowres1 `8.86 s -> 7.17 s` (-19.1%),
  LP-QA reactor scale `13.16 s -> 10.66 s` (-19.0%), and
  LP-QH reactor scale `15.29 s -> 12.91 s` (-15.6%).
- The post-policy matrix also compares cleanly against `origin/main`:
  `python tools/diagnostics/compare_runtime_memory_matrix.py --current
  outputs/pr20_readme_matrix_current_cpu_post_precond/summary.json --baseline
  /Users/rogeriojorge/local/tests/vmec_jax_main_perf/outputs/pr20_readme_matrix_main_cpu/summary.json`
  reports `rows=32 regressions=0`.
- The post-policy README matrix now shows the large 3D rows at about
  `1.18x--1.21x` VMEC2000 for the reactor-scale rows. Tiny rows remain
  startup/setup dominated, and LASYM/memory-heavy rows remain separate open
  targets.
- Focused validation passed:
  `python -m ruff check vmec_jax/vmec_residue.py
  vmec_jax/solvers/fixed_boundary/residual/force_payload.py
  tests/test_vmec_residue_fast_helpers.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_vmec_residue_fast_helpers.py::test_dynamic_force_norm_guards_and_tree_roundtrip
  tests/test_vmec_residue_fast_helpers.py::test_dynamic_force_norms_scales_fused_matches_separate_helpers
  tests/test_solve_preconditioner_metric_helpers.py::test_metric_surface_precond_from_bcovar_jax_matches_scale_kernel
  tests/test_solve_residual_iter_force_payload_helpers.py -q`.
- LP-QA lowres controlled profile after the fused helper:
  final-stage `solve_total_s=4.1061`, `compute_forces_s=2.9979`,
  `preconditioner_s=0.3593`, final residual unchanged at
  `4.64e-13`. This is a smaller but real improvement over the post-policy
  profile (`solve_total_s=4.1482`, `compute_forces_s=3.0413`).
- Forcing the TOMNSP FFT path on CPU is a regression for this row:
  `VMEC_JAX_TOMNSPS_FFT=1` gives final-stage `solve_total_s=5.9379` and
  `compute_forces_s=4.8115`. The CPU default should remain the DFT/GEMM path;
  future transform work should target separable DFT buffers rather than CPU FFT.
- Explorer conclusions reinforce the next larger tranche:
  VMEC2000 and vmec++ use separable Fourier transforms, long-lived work arrays,
  radial streaming/scan structure, periodic preconditioner updates, and
  phase-level timers. vmec_jax should move toward a staged JIT force pipeline
  with domain-named seams rather than one all-array force function returning
  every debug intermediate.

Best next steps:

1. Commit the fused `bcovar` reduction tranche, then run a small current-matrix
   smoke if needed to verify no row-level behavior changed.
2. Start the larger force-kernel refactor with the lowest-risk seam:
   split default hot-path force aux from full debug/parity kernels so compiled
   solves do not carry full diagnostic payloads unless a dump/parity hook asks
   for them.
3. Plan the next architectural tranche around separable TOMNSP/Fourier kernels
   and shape-stable workspaces, with validation against VMEC2000 WOUT parity
   and AD-vs-FD gates before changing default math paths.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 91.5%.
- Refactor/API/examples: 48%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96%.
- Overall: 90.5%.

### 2026-06-22: Add opt-in compact residual force aux seam

Steps taken:

- Audited post-TOMNSP `VmecRZForceKernels` consumers in the residual loop,
  preconditioner refresh path, PTAU bad-Jacobian checks, free-boundary coupling,
  and diagnostics.
- Added `ResidualForceKernelAux`, a compact production-style payload containing
  only `bcovar`, `tcon`, PTAU geometry fields, and constraint baselines.
- Kept direct `evaluate_residual_force_from_state` diagnostics unchanged so
  full force-kernel fields remain available for parity/debug callers.
- Wired `make_residual_force_evaluator` so the compact aux can be selected with
  `compact_kernel_aux=True` or `VMEC_JAX_COMPACT_FORCE_AUX=1`, but left the
  default as the full payload because the first timing probe was not a wall-time
  win.
- Added unit coverage for compact field selection, default full-payload
  behavior, explicit compact opt-in, and env opt-in.

Results obtained:

- Focused validation passed:
  `python -m ruff check vmec_jax/solvers/fixed_boundary/residual/force_payload.py
  tests/test_solve_residual_iter_force_payload_helpers.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_solve_residual_iter_force_payload_helpers.py
  tests/test_solve_scan_math_helpers.py::test_ptau_minmax_jax_matches_host_and_returns_nan_for_missing_or_short_kernel -q`.
- Compact aux profile on LP-QA reduced peak RSS from about `908 MiB` to
  `847 MiB`, but worsened the final-stage wall time (`4.11 s -> 4.59 s` in the
  first compact run). Therefore it is an experimental memory diagnostic, not a
  production default.
- With compact aux default-off, residuals remain unchanged, but single-run
  timing remained noisy/slower than the earlier best fused profile. This tranche
  should be treated as a refactor/API seam, not a claimed runtime win.

Best next steps:

1. Do not promote compact aux by default until a staged force-kernel refactor
   makes it speed-neutral.
2. Move next to the larger force kernel architecture: separable DFT/TOMNSP
   buffers and staged JIT kernels with donation/remat options, validated against
   VMEC2000 parity.
3. Keep measuring memory separately from wall time so memory-saving knobs do not
   accidentally regress the public fast path.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 91.5%.
- Refactor/API/examples: 49%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96.5%.
- Overall: 90.7%.

### 2026-06-22: Split force-kernel profile phases into VMEC-comparable buckets

Steps taken:

- Replaced the ambiguous overlapping force profile stages with non-overlapping
  buckets: `m1_physical`, `bcovar`, `parity_extract`,
  `radial_force_assembly`, `constraint_finish`, and `force_total`.
- Added JAX profiler/Perfetto trace annotations with the same phase names so
  CPU/GPU traces can be matched to the printed VMEC-style timing summaries.
- Added a focused synthetic-force regression test that enables
  `VMEC_JAX_PROFILE_FORCE=1` and checks that the new phase names are emitted
  while the old overlapping names are not.

Results obtained:

- Focused validation passed:
  `python -m ruff check vmec_jax/vmec_forces.py
  tests/test_vmec_forces_synthetic_helpers.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_vmec_forces_synthetic_helpers.py::test_force_profile_phases_are_vmec_comparable
  tests/test_vmec_forces_synthetic_helpers.py::test_force_profile_log_respects_env
  tests/test_solve_residual_iter_force_payload_helpers.py -q`.
- A short QH single-grid profile smoke passed:
  `input.nfp4_QH_warm_start`, `--iters 20`, no warmup, no multigrid,
  `solve_total_s=0.198 s`, `compute_forces_s=0.014 s`,
  `final_w=7.30e-6` as expected for a deliberately under-converged
  20-iteration smoke.
- This tranche is an observability/refactor improvement. It does not change the
  force algebra and should not be counted as a runtime win by itself.

Best next steps:

1. Use the new non-overlapping force phases in CPU/GPU profiles to decide
   whether the next production speedup should target TOMNSP/Fourier synthesis,
   bcovar geometry, or constraint finish.
2. Prototype the separable TOMNSP/Fourier workspace path behind an opt-in flag,
   with WOUT parity and AD-vs-FD gates before changing the default math path.
3. Keep compact force aux opt-in until the staged force-kernel refactor makes
   the memory-saving path speed-neutral.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 91.6%.
- Refactor/API/examples: 49.5%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96.5%.
- Overall: 90.8%.

### 2026-06-22: Remove redundant TOMNSP theta stack materialization

Steps taken:

- Audited `tomnsps_rzl` after the force-phase split. The symmetric transform
  built `stack_all` before choosing the FFT/DFT path, and the FFT fused path
  rebuilt the same concatenation inside its branch.
- Moved the `stack_all` construction into the branches that actually need it:
  once in the FFT fused branch and once in the CPU-default DFT branch.
- Kept the CPU DFT/GEMM default unchanged because the previous
  `VMEC_JAX_TOMNSPS_FFT=1` experiment regressed the LP-QA final stage.

Results obtained:

- TOMNSP-focused validation passed:
  `python -m ruff check vmec_jax/vmec_tomnsp.py
  tests/test_vmec_tomnsp_branch_coverage.py tests/test_vmec_tomnsp_tables.py`;
  `JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_vmec_tomnsp_branch_coverage.py tests/test_vmec_tomnsp_tables.py -q`.
- A short QH single-grid profile smoke after the cleanup measured
  `solve_total_s=0.185 s`, `compute_forces_s=0.0125 s`, and the same
  `final_w=7.30e-6` for the 20-iteration under-converged diagnostic.
- This is a small materialization cleanup. It mainly reduces redundant FFT-path
  graph work and is not by itself sufficient to close the CPU/VMEC2000 gap.

Best next steps:

1. Use the new force phase labels to profile a long high-mode row and decide
   whether the dominant cost is TOMNSP, `bcovar`, or update/preconditioner.
2. Prototype a larger separable TOMNSP workspace only behind an opt-in flag;
   require TOMNSP parity tests, WOUT parity rows, and AD-vs-FD evidence before
   changing defaults.
3. Keep comparing current branch against `origin/main` on the full matrix after
   any default-path performance change.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 90%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 91.7%.
- Refactor/API/examples: 49.8%.
- VMEC2000/VMEC++ parity and physics gates: 96%.
- Docs/release hygiene: 96.5%.
- Overall: 90.9%.

### 2026-06-22: Fix fixed-boundary implicit edge-gradient sign and log force/bcovar phase evidence

Steps taken:

- Used the new force phase labels on `input.LandremanPaul2021_QA_lowres`.
  The bounded local profile completed under `outputs/pr20_force_phase_lpqa/`
  with `solve_total_s=11.753 s` and `compute_forces_s=4.235 s`.
- Enabled the existing `VMEC_JAX_PROFILE_BCOVAR=1` subphase logger on the same
  row. The profile completed under `outputs/pr20_bcovar_phase_lpqa/` with
  `solve_total_s=11.454 s` and `compute_forces_s=4.111 s`.
- Aggregated the phase logs. Force-phase means:
  `bcovar_done=0.0478 s`, `constraint_finish_done=0.0222 s`,
  `radial_force_assembly_done=0.0122 s`, `m1_physical_done=0.0094 s`.
  Bcovar-subphase means:
  `metric_done=0.0139 s`, `field_done=0.0098 s`,
  `odd_channels_done=0.0049 s`, `lambda_done=0.0029 s`,
  `parity_done=0.0028 s`.
- Added a fast fixed-boundary implicit scalar AD-vs-central-FD gate that
  exercises the energy implicit custom VJP direct boundary-parameter path.
- The new gate exposed a real sign bug: the implicit VJP added direct edge-row
  output cotangents before the final implicit-function negation, flipping the
  direct boundary sensitivity.
- Fixed `solve_fixed_boundary_state_implicit` so only the implicit
  `F_p^T H^{-1} ct` contribution receives the minus sign, while the direct
  edge-row projection is added with its physical sign.

Results obtained:

- Before the fix, the new scalar gate produced `AD=-2.4375` and
  `FD=+2.4375`.
- After the fix, focused validation passed:
  `python -m ruff check vmec_jax/implicit.py
  tests/test_implicit_differentiation_fast.py
  tests/test_glasser_resistive_interchange.py
  tests/test_finite_beta_helpers_unit.py`;
  `PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  -p no:cacheprovider
  tests/test_implicit_differentiation_fast.py::test_fixed_boundary_implicit_scalar_grad_matches_central_fd
  tests/test_implicit_differentiation_fast.py::test_fixed_boundary_backward_runs_hvp_and_direct_edge_cotangent
  tests/test_implicit_differentiation_fast.py::test_fixed_boundary_backward_zeros_inactive_edge_cotangents
  tests/test_glasser_resistive_interchange.py::test_glasser_d_r_gradient_matches_central_finite_difference
  tests/test_finite_beta_helpers_unit.py::test_mercier_terms_from_state_dmerc_and_d_r_ad_match_central_fd
  tests/test_finite_beta_helpers_unit.py::test_mercier_objective_wrappers_ad_match_central_fd -q`.
- The full fast implicit differentiation file passed:
  `PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  -p no:cacheprovider tests/test_implicit_differentiation_fast.py -q`.
- Existing `DMerc`/`D_R` derivative tests still pass after the sign fix.

Best next steps:

1. Promote the fixed-boundary evidence panel to include this direct boundary
   AD-vs-FD row, then keep the stricter `1e-9` threshold for rows where the
   scalar is smooth and synthetic enough to support it.
2. Target the next performance implementation at `bcovar` metric/field assembly
   and odd-channel synthesis rather than TOMNSP FFT flipping.
3. Add a similarly narrow free-boundary branch-local physical-scalar evidence
   gate using the existing same-branch report machinery.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 91.5%.
- Free-boundary production differentiability: 87%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 91.8%.
- Refactor/API/examples: 50%.
- VMEC2000/VMEC++ parity and physics gates: 96.2%.
- Docs/release hygiene: 96.5%.
- Overall: 91.1%.

### 2026-06-22: Harden README free-boundary AD-vs-FD evidence artifact contract

Steps taken:

- Audited the README AD-vs-FD evidence renderer and the same-branch
  direct-coil free-boundary report path.
- Tightened `tools/diagnostics/readme_ad_fd_evidence.py` so a supplied
  branch-local report must contain a passing physical-scalar gate with the
  promoted scalar set: `aspect`, `qs_total`, `mean_iota`, and
  `lcfs_boundary_moment`.
- Added fast synthetic tests that reject incomplete or failed branch-local
  evidence reports before they can be used to regenerate the public README
  differentiation panel.

Results obtained:

- Focused validation passed:
  `python -m ruff check tools/diagnostics/readme_ad_fd_evidence.py
  tests/test_readme_ad_fd_evidence.py`;
  `PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  -p no:cacheprovider tests/test_readme_ad_fd_evidence.py -q`.
- The stricter renderer accepted the current checked-in provenance report:
  `JAX_ENABLE_X64=1 python tools/diagnostics/readme_ad_fd_evidence.py
  --branch-local-report
  outputs/pr20_ad_fd/qs_same_branch/same_branch_complete_solve_report.json
  --figure-out outputs/pr20_ad_fd_contract/readme_ad_fd_evidence.png
  --csv-out outputs/pr20_ad_fd_contract/readme_ad_fd_evidence.csv
  --json-out outputs/pr20_ad_fd_contract/readme_ad_fd_evidence.json`,
  producing 10 passing rows.
- README/docs hygiene checks for concise README content and compact checked-in
  figures passed.

Best next steps:

1. Continue free-boundary phase-2 work with a narrow same-branch gate that
   includes an explicit accepted/rejected controller-slot fingerprint when the
   branch remains unchanged.
2. Keep public wording conservative: this gate protects branch-local,
   fingerprint-gated evidence and still does not claim arbitrary adaptive
   branch differentiation.
3. Return to CPU/GPU runtime work at the measured `bcovar` metric/field
   assembly hotspot once the next free-boundary gate is promoted.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 91.5%.
- Free-boundary production differentiability: 88%.
- Single-stage coil optimization: 86%.
- CPU/GPU runtime and memory footprint: 91.8%.
- Refactor/API/examples: 50%.
- VMEC2000/VMEC++ parity and physics gates: 96.2%.
- Docs/release hygiene: 96.7%.
- Overall: 91.3%.

### 2026-06-22: Promote accepted/rejected controller-slot fingerprint artifacts

Steps taken:

- Added a shared JSON-safe
  `direct_coil_accepted_trace_controller_slot_fingerprint` helper next to the
  existing controller-slot summary helper.
- Exposed the fingerprint in the same-branch adaptive full-loop seam report
  and in the single-stage coil optimization `accepted_rejected_controller_slot`
  artifact.
- Extended the mocked single-stage report test and the direct-coil trace
  fingerprint test to assert the accepted/rejected masks, trace-derived status
  source, and rejected-slot summary.

Results obtained:

- Focused validation passed:
  `python -m ruff check
  vmec_jax/solvers/free_boundary/adjoint/trace_metadata.py
  vmec_jax/solvers/free_boundary/adjoint/gate_reports.py
  vmec_jax/solvers/free_boundary/adjoint/facade.py
  vmec_jax/solvers/free_boundary/coil_optimization.py
  tests/test_free_boundary_qs_coil_optimization_smoke.py
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py`.
- The exact same-branch smoke subset passed:
  `PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  -p no:cacheprovider tests/test_free_boundary_qs_coil_optimization_smoke.py
  -k same_branch -q`.
- Targeted trace-fingerprint validation passed with the new helper included:
  `tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py::test_direct_coil_trace_fingerprint_detects_control_branch_changes`.

Best next steps:

1. Use the fingerprinted accepted/rejected slot artifact in the small
   coil-only QS derivative-proposal run so complete solves remain the
   acceptance authority but reviewers can inspect the exact branch-local replay
   slot layout.
2. Start the low-risk `bcovar` compact force-payload tranche identified by the
   performance audit, preserving the full public `VmecHalfMeshBcovar` return
   for diagnostics and WOUT/parity helpers.
3. Keep arbitrary adaptive branch differentiation deferred to the research
   differentiability plan until a true JAX-visible adaptive branch-selection
   loop exists and passes AD-vs-FD.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 91.5%.
- Free-boundary production differentiability: 89%.
- Single-stage coil optimization: 86.5%.
- CPU/GPU runtime and memory footprint: 91.8%.
- Refactor/API/examples: 50.3%.
- VMEC2000/VMEC++ parity and physics gates: 96.3%.
- Docs/release hygiene: 96.8%.
- Overall: 91.6%.

### 2026-06-22: Add compact bcovar payload for the force hot path

Steps taken:

- Added internal `VmecForceBcovarPayload` for production force/residual calls.
  The public `VmecHalfMeshBcovar` payload remains unchanged for diagnostics,
  parity helpers, and direct `bcovar` callers.
- Routed only `vmec_forces_rz_from_wout` production calls to
  `compact_force_payload=True`; `use_wout_bsup` reference/parity mode still
  returns the full bcovar payload.
- The compact payload keeps downstream force/residual fields (`jac`, metric
  tensors, `bsup*`, `bsub*`, `clmn/blmn`, `bsq`, `gij_b_*`, `lu_e/lv_e`,
  `lamscale`) and omits parity/debug-only intermediates such as
  `bsubu_parity_even`.
- Added a focused test asserting the production force path actually returns the
  compact payload and still exposes required downstream fields.

Results obtained:

- Focused validation passed:
  `python -m ruff check vmec_jax/vmec_bcovar.py vmec_jax/vmec_forces.py
  tests/test_forces_bcovar_wave12_coverage.py
  tests/test_vmec_forces_synthetic_helpers.py
  tests/test_force_norms_dynamic_parity.py`.
- Extended force/bcovar parity subset passed:
  `PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  -p no:cacheprovider tests/test_bcovar_lambda_axis_closure.py
  tests/test_wout_bcovar_forces_extra_coverage.py
  tests/test_force_norms_dynamic_parity.py
  tests/test_forces_bcovar_wave12_coverage.py
  tests/test_vmec_forces_synthetic_helpers.py -q`.
- Short QH smoke improved from the previous local diagnostic
  `solve_total_s=0.185 s`, `compute_forces_s=0.0125 s` to
  `solve_total_s=0.172 s`, `compute_forces_s=0.0112 s`.
- LP-QA profile improved from
  `solve_total_s=11.454 s`, `compute_forces_s=4.111 s`,
  `peak_rss=1770 MiB` to
  `solve_total_s=10.537 s`, `compute_forces_s=3.838 s`,
  `peak_rss=1723 MiB`.
- Bcovar subphase mean dropped from `bcovar_done=0.0353 s/call` to
  `0.0299 s/call` on the LP-QA profile.

Best next steps:

1. Deeper `bcovar` tranche: add a compact metric assembly mode that avoids
   computing half-even/odd metric parity channels and lambda derivative arrays
   when the force path cannot consume them.
2. Re-run the full README single-grid matrix after one more performance
   tranche, then compare against `origin/main` and VMEC2000/VMEC++ rows.
3. Keep WOUT/parity gates on the full payload and production force gates on the
   compact payload so performance changes do not silently reduce diagnostics.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 91.5%.
- Free-boundary production differentiability: 89%.
- Single-stage coil optimization: 86.5%.
- CPU/GPU runtime and memory footprint: 92.6%.
- Refactor/API/examples: 50.8%.
- VMEC2000/VMEC++ parity and physics gates: 96.5%.
- Docs/release hygiene: 96.8%.
- Overall: 92.0%.
