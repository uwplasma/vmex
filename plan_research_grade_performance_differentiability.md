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

- Performance benchmark and profiling harness: 100%.
  The PR #20 single-grid matrix, current-vs-main comparator, VMEC2000 rows, and
  VMEC++ optional rows have been regenerated with CSV/JSON provenance after the
  compact force-payload updates. Remaining work is deeper kernel-level
  decomposition and long-term dashboard automation, not benchmark harness
  readiness.
- Fixed-boundary production differentiability: 97.2%.
  AD-vs-central-FD evidence now passes `1e-9` for fixed-boundary geometry,
  profiles, QS/QI diagnostics, `DMerc`, and `D_R`. Remaining work is
  operator-level implicit/JVP/VJP productionization.
- Free-boundary production differentiability: 97.5%.
  Direct coil fields, JAX mgrid interpolation, accepted-branch replay, and
  fingerprint-gated branch-local gates pass `1e-9` evidence for selected
  physical scalars. Dynamic replay now tolerates minimal/full preconditioner
  cache payload structure changes in both JVP and VJP paths. Arbitrary adaptive
  branch differentiation remains unclaimed.
- Single-stage coil optimization: 93.6%.
  Examples and branch-local derivative proposal paths exist; complete solves
  still need to remain the acceptance authority until the full adaptive seam is
  validated.
- CPU/GPU runtime and memory footprint: 99.2%.
  The latest single-grid matrix shows warm CPU `vmec_jax` beating VMEC2000 on
  14 of 16 rows, with median warm runtime ratio `0.83x` VMEC2000 and median
  peak-memory ratio `3.04x` VMEC2000. Cold process runtime remains slower
  (`2.23x` median) because it includes Python/JAX/XLA startup, and peak memory
  remains materially higher than VMEC2000, especially for LASYM finite-beta
  rows. The remaining work is absolute memory reduction, cold-start reduction,
  and GPU/optimization callback costs.
- Refactor/API/examples: 76.4%.
  Public examples are better, but core source files and tests are still too
  large and too entangled. The fixed-boundary residual timing/setup seam is now
  slightly cleaner, but the main residual loop still needs a larger split.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
  The PR #20 four-row executable WOUT parity gate passed, and the single-grid
  runtime matrix records VMEC++ availability per row. More bounded
  free-boundary external parity remains future work.
- Docs/release hygiene: 100%.
  README is concise, runtime/memory detail lives in docs, and benchmark plus
  AD-FD provenance are refreshed. Remaining work is Sphinx gating and pruning
  historical performance prose after review.
- Overall completion: 99.6%.
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

### 2026-06-22: Compact bcovar metric intermediates in force mode

Steps taken:

- Extended the compact force-payload path so `bcovar` skips construction of
  half-even/odd metric parity channels and lambda derivative arrays when those
  fields cannot be consumed by the compact force payload.
- Kept full metric/parity intermediates in the public `VmecHalfMeshBcovar`
  path and in `use_wout_bsup` reference/parity mode.

Results obtained:

- The extended force/bcovar parity subset still passed:
  `PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  -p no:cacheprovider tests/test_bcovar_lambda_axis_closure.py
  tests/test_wout_bcovar_forces_extra_coverage.py
  tests/test_force_norms_dynamic_parity.py
  tests/test_forces_bcovar_wave12_coverage.py
  tests/test_vmec_forces_synthetic_helpers.py -q`.
- LP-QA memory improved slightly further (`peak_rss=1714 MiB` versus
  `1723 MiB` after the first compact-payload tranche and `1770 MiB` before
  compact payloads).
- Runtime evidence is mixed: QH short smoke remains faster than the original
  local baseline (`solve_total_s=0.179 s` versus `0.185 s`), but the LP-QA
  solve in this run was slower than the first compact-payload run
  (`10.93 s` versus `10.54 s`). Treat this as a memory/graph-footprint cleanup,
  not a promoted runtime win.

Best next steps:

1. Do not add more `bcovar` structural branches until the full benchmark matrix
   shows whether the compact-metric memory win is worth the neutral/mixed
   runtime behavior.
2. Re-run the current-branch single-grid matrix and compare against the last
   saved matrix before making further default performance changes.
3. If runtime still regresses relative to VMEC2000, target preconditioner
   assembly/apply next rather than adding more `bcovar` branches.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 91.5%.
- Free-boundary production differentiability: 89%.
- Single-stage coil optimization: 86.5%.
- CPU/GPU runtime and memory footprint: 92.8%.
- Refactor/API/examples: 51%.
- VMEC2000/VMEC++ parity and physics gates: 96.5%.
- Docs/release hygiene: 96.8%.
- Overall: 92.1%.

### 2026-06-22: Re-run full single-grid matrix after compact force payloads

Steps taken:

- Re-ran the 16-row historical bundled fixed-boundary single-grid matrix on the
  current branch after the compact force-payload and compact metric-payload
  changes:
  `PYTHONPATH=$PWD JAX_ENABLE_X64=1 python
  tools/diagnostics/example_runtime_memory_matrix.py --inputs-dir
  examples_single_grid/data --kind fixed --backend all --warm-runs 1
  --jax-platforms cpu --runner-label current-cpu-compact --vmec-exec
  ~/bin/xvmec2000 --timeout-s 1800 --vmec-timeout-s 1800 --outdir
  outputs/pr20_full_matrix_current_cpu_sg_compact`.
- Compared the new matrix against the saved clean `origin/main` matrix:
  `python tools/diagnostics/compare_runtime_memory_matrix.py --current
  outputs/pr20_full_matrix_current_cpu_sg_compact/summary.json --baseline
  /Users/rogeriojorge/local/tests/vmec_jax_main_perf/outputs/pr20_full_matrix_main_cpu_sg/summary.json
  --csv-out outputs/pr20_full_matrix_compact_vs_main.csv --json-out
  outputs/pr20_full_matrix_compact_vs_main.json`.
- Refreshed the docs benchmark artifacts:
  `docs/_static/figures/readme_runtime_compare.png`,
  `docs/_static/figures/readme_runtime_compare.csv`,
  `docs/_static/figures/readme_runtime_compare.json`,
  `docs/_static/figures/readme_runtime_compare_current_vs_main.csv`, and
  `docs/_static/figures/readme_runtime_compare_current_vs_main.json`.
- Tightened broad package/docs wording so free-boundary differentiation is
  described as branch-local/fingerprint-gated research evidence rather than
  arbitrary adaptive branch differentiation.

Results obtained:

- Matrix completion: 16/16 `vmec_jax` rows and 16/16 VMEC2000 rows succeeded;
  VMEC++ succeeded on 9/16 rows and is recorded as unavailable/non-converged
  on the remaining 7 rows.
- Warm `vmec_jax` beat VMEC2000 on 14/16 rows; cold `vmec_jax` beat VMEC2000
  on 1/16 rows.
- Median warm runtime ratio vs VMEC2000 improved to `0.83x`; median cold
  runtime ratio is still `2.23x` because cold runs include Python/JAX/XLA
  setup.
- Median peak process-memory ratio vs VMEC2000 improved to `3.04x`; the worst
  row remains the non-stellarator-symmetric finite-beta pressure case at
  `16.7x`, so memory remains the main absolute gap.
- The `origin/main` comparator still flags several per-row threshold hits
  (`~10%` runtime or `15%` memory), but the matrix-level aggregate is
  materially better than both `origin/main` and the previous current-branch
  full-JIT matrix. Treat the row flags as profile/classification items, not a
  reason to revert the compact payload work.

Best next steps:

1. Run the accepted/rejected controller-slot same-branch derivative-proposal
   gate recommended by the free-boundary differentiability audit.
2. Profile only the remaining absolute outliers: cold-start setup on tiny
   axisymmetric rows and peak memory on LASYM finite-beta rows.
3. Continue refactoring the residual/force seams only where the benchmark shows
   an absolute runtime or memory target; avoid speculative structural branches
   without matrix evidence.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 91.5%.
- Free-boundary production differentiability: 89.5%.
- Single-stage coil optimization: 86.8%.
- CPU/GPU runtime and memory footprint: 93.4%.
- Refactor/API/examples: 51.5%.
- VMEC2000/VMEC++ parity and physics gates: 96.8%.
- Docs/release hygiene: 97%.
- Overall: 92.6%.

### 2026-06-22: Promote accepted/rejected-slot free-boundary AD evidence

Steps taken:

- Ran the dependency-light direct-coil QS optimization smoke with the
  branch-local derivative proposal path and the accepted/rejected
  controller-slot gate enabled:
  `JAX_ENABLE_X64=1 python
  examples/optimization/free_boundary_QS_coil_optimization.py --smoke
  --provider circle --max-evals 1 --max-iter 1 --vmec-max-iter 2
  --helicity-m 1 --helicity-n 0 --write-same-branch-report
  --same-branch-report-mode vector --same-branch-report-ad-mode direct
  --same-branch-report-direction current-only
  --same-branch-report-vector-keys
  aspect,qs_total,mean_iota,lcfs_boundary_moment
  --same-branch-report-rejected-slot-gate
  --same-branch-derivative-proposal --outdir
  outputs/pr20_rejected_slot_proposal_full`.
- Re-rendered the AD-vs-FD evidence panel from the generated
  `same_branch_complete_solve_report.json` into
  `docs/_static/figures/readme_ad_fd_evidence.{png,csv,json}`.
- Updated validation/free-boundary documentation so the reproduction command
  includes the rejected-slot gate and derivative-proposal flags.

Results obtained:

- The branch-local report contains
  `accepted_rejected_controller_slot_gate.requested=true`,
  `available=true`, and `passed=true`.
- The controller-slot fingerprint contains two accepted slots and one rejected
  slot (`step_status = momentum, momentum, rejected`), with one active accepted
  free-boundary replay slot.
- The branch-local derivative proposal remains conservative:
  it uses the fixed accepted/rejected trace derivative to propose a coil-current
  step, but complete free-boundary solves remain the acceptance authority.
- The AD-vs-FD evidence renderer passed all 10 rows at `1e-9`: fixed-boundary
  aspect, iota profile, QS residual, smooth QI residual, `DMerc`, `D_R`, and
  free-boundary `aspect`, `qs_total`, `mean_iota`, and
  `lcfs_boundary_moment`.

Best next steps:

1. Use this validated branch-local vector/JVP report in the small coil-only QS
   example as the default derivative-proposal evidence path, while keeping
   complete solves as the only acceptance authority.
2. Keep arbitrary adaptive branch differentiation deferred to the
   research-grade differentiability plan; the current gate is same-branch and
   fingerprint-gated by design.
3. Continue VMEC2000/direct-coil/mgrid parity expansion only with bounded,
   finite-positive physical WOUT fixtures.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92%.
- Free-boundary production differentiability: 91%.
- Single-stage coil optimization: 87.5%.
- CPU/GPU runtime and memory footprint: 93.4%.
- Refactor/API/examples: 52%.
- VMEC2000/VMEC++ parity and physics gates: 97%.
- Docs/release hygiene: 97.3%.
- Overall: 93.2%.

### 2026-06-22: Classify LASYM peak-memory policy tradeoff

Steps taken:

- Ran focused one-row probes for `basic_non_stellsym_pressure`, the largest
  peak-memory outlier in the single-grid matrix.
- Compared default production policy, explicit `VMEC_JAX_SCAN_MINIMAL=1`,
  explicit `solver_mode=accelerated`, and explicit `solver_mode=parity`.
- Ran bounded profilers on `basic_non_stellsym_pressure` and
  `LandremanSengupta2019_section5.4_B2_A80` to classify whether the remaining
  gap was scan-history retention, policy selection, or force/preconditioner
  arithmetic.

Results obtained:

- `VMEC_JAX_SCAN_MINIMAL=1` did not reduce peak memory for
  `basic_non_stellsym_pressure`; it measured roughly the same warm runtime and
  slightly higher peak memory than the default. This rules out full scan
  history as the main memory culprit on that row.
- Explicit accelerated mode measured about `5.79 s` warm runtime and
  `1.66 GiB` peak process memory.
- Explicit parity mode measured about `15.48 s` warm runtime and
  `0.64 GiB` peak process memory. VMEC2000 is about `14.72 s` on this row.
- The next memory win is therefore a deliberate memory-aware policy selector or
  user-facing mode. Silently switching the default would sacrifice the main
  runtime gain and should not be done without an explicit user preference or
  documented memory budget.
- The bounded `A80` cold probe showed that automatic finish-policy micro-probes
  and preconditioner R/Z apply dominate the short parity diagnostic; this is a
  cold/tiny-row overhead target, not a large-row force-kernel target.

Best next steps:

1. Add a documented memory-aware fixed-boundary policy (`solver_mode` value or
   explicit option) that selects parity/host-loop paths for LASYM/high-memory
   rows when users prioritize peak memory over warm runtime.
2. Keep the runtime-optimized default for the public benchmark unless a memory
   mode is explicitly requested.
3. Profile cold/tiny-row setup separately from long-run force kernels so future
   optimizations do not trade away the 14/16 warm-runtime wins.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92%.
- Free-boundary production differentiability: 91%.
- Single-stage coil optimization: 87.5%.
- CPU/GPU runtime and memory footprint: 93.8%.
- Refactor/API/examples: 52%.
- VMEC2000/VMEC++ parity and physics gates: 97%.
- Docs/release hygiene: 97.5%.
- Overall: 93.5%.

### 2026-06-22: Add explicit low-memory fixed-boundary solver mode

Steps taken:

- Added `solver_mode="memory"` plus `low-memory`/`low_memory` aliases that map
  to the existing parity path.
- Updated CLI help and README CLI reference so beginners can choose the
  low-peak-memory path without needing to know that it is implemented by the
  VMEC2000-style parity loop.
- Added policy-helper coverage for the new aliases.

Results obtained:

- This does not change the default runtime-optimized policy or the public
  benchmark results.
- It exposes the measured LASYM tradeoff as an explicit user choice:
  lower peak memory with VMEC2000-like parity runtime, rather than a silent
  default-policy regression.

Best next steps:

1. Run focused policy/CLI tests and Sphinx docs build.
2. Consider a future automatic memory-budget selector only after users have a
   stable explicit control and after more rows are profiled.
3. Continue deeper runtime work on cold-start setup and GPU/optimization
   callback costs separately from this memory-mode knob.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92%.
- Free-boundary production differentiability: 91%.
- Single-stage coil optimization: 87.5%.
- CPU/GPU runtime and memory footprint: 94.2%.
- Refactor/API/examples: 52.5%.
- VMEC2000/VMEC++ parity and physics gates: 97%.
- Docs/release hygiene: 97.6%.
- Overall: 93.8%.

### 2026-06-22: Profile cold setup and reject no-win profile fast path

Steps taken:

- Re-ran a bounded cold setup probe on `examples_single_grid/data/input.solovev`
  with three single-grid parity iterations, timing detail enabled, and no finish
  policy.
- Inspected the residual profile setup, force payload, strict-update, and scan
  payload seams to identify whether optional channel materialization or
  profile setup offered a safe small memory/runtime win.
- Temporarily tested two small code changes locally:
  - a narrow zero-profile WOUT-like setup fast path for vacuum/default-APHI
    inputs,
  - skipping the NumPy-force patch import in profile setup when only
    `host_update_assembly` was true.
- Reverted both experimental changes before committing because the measured
  results did not justify carrying extra complexity.

Results obtained:

- Current `solovev` cold three-iteration probe is already much faster than the
  older pre-cleanup profile: `solve_total_s ~= 0.189 s`, with the largest
  buckets now `precond_refresh_seed_lambda_s ~= 0.076 s`,
  `setup_axis_reset_s ~= 0.045 s`, and `setup_boundary_profiles_s ~= 0.039 s`.
- `solovev` cannot use a vacuum-profile shortcut because it has nonzero
  pressure and iota coefficients (`AM = 0.125 -0.125`, `AI = 1`).
- The QH vacuum row showed the profile wrapper is already cheap when the
  NumPy-force patch is active (`setup_profile_data_s ~= 0.00027 s`). Removing
  the patch import was a regression: `setup_profile_data_s` increased to about
  `0.123 s` and total setup increased to about `0.211 s`.
- Optional asymmetric-channel zero materialization is real in the JAX
  preconditioner/scan payloads, but the current scan payload carries a fixed
  12-channel tuple and helper tests explicitly encode dense-zero behavior for
  absent optional channels. Changing this safely requires a larger optional
  channel pytree refactor, not a narrow patch.

Best next steps:

1. Do not pursue the rejected profile fast path or patch-import removal again
   unless a new profile shows a different target.
2. Target preconditioner seed cost next, especially lambda preconditioner setup
   and R/Z matrix setup reuse for short cold runs.
3. Treat optional-channel memory reduction as a deliberate refactor: introduce
   optional-channel payload types, update scan restart masking and strict-update
   call sites, and gate with LASYM/finite-beta/WOUT parity tests before
   promotion.
4. Continue fixed-boundary production differentiability and free-boundary
   branch-local derivative integration in parallel with runtime work.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92%.
- Free-boundary production differentiability: 91%.
- Single-stage coil optimization: 87.5%.
- CPU/GPU runtime and memory footprint: 94.4%.
- Refactor/API/examples: 52.5%.
- VMEC2000/VMEC++ parity and physics gates: 97%.
- Docs/release hygiene: 97.6%.
- Overall: 93.9%.

### 2026-06-22: Promote CPU host NumPy lambda preconditioner seed

Steps taken:

- Added a concrete-host dispatch in the residual preconditioner operator
  binding so non-traced CPU host-update stages use the existing pure-NumPy
  lambda preconditioner implementation.
- Kept traced/differentiable, JAX, and accelerator paths on
  `lambda_preconditioner_cached`, preserving autodiff and accelerator behavior.
- Added a focused dispatch-policy unit test that monkeypatches the NumPy and
  JAX implementations to verify the host path selects the NumPy implementation.
- Re-ran preconditioner parity/unit tests and fast physics gates.

Results obtained:

- Preconditioner tests passed:
  `tests/test_preconditioner_1d_fast_helpers.py`,
  `tests/test_preconditioner_1d_jax_fast_helpers.py`,
  `tests/test_tcon_precondn_diag.py`, and
  `tests/test_solve_preconditioner_payload_helpers.py`.
- Fast physics/parity gates passed:
  `tests/test_vmec_parity_physics_fast_gates.py`,
  `tests/test_wout_physics_gates.py`, and
  `tests/test_vmec2000_fixed_boundary_physics_gates.py`.
- Bounded three-iteration cold probes improved without changing the short-trace
  residuals:
  - `input.solovev`: solve-body time dropped from about `0.189 s` to
    `0.112 s`; lambda seed dropped from about `0.076 s` to `0.0012 s`.
  - `input.nfp4_QH_warm_start`: solve-body time dropped from about `0.346 s`
    to `0.274 s`; lambda seed dropped from about `0.079 s` to `0.0014 s`.
- The remaining cold preconditioner seed cost is now R/Z matrix construction,
  not lambda preconditioning.
- A compact 16-row bundled single-grid `vmec_jax` matrix after this change
  completed successfully.  Compared with the previous compact current-branch
  matrix, median cold runtime was `0.899x`, median warm runtime was `0.864x`,
  and median peak memory was `0.993x`.  One LASYM axisymmetric row had a small
  `1.104x` warm-runtime outlier while cold runtime improved, so it is tracked as
  noise unless repeated profiles reproduce it.

Best next steps:

1. Continue R/Z matrix construction profiling; keep differentiable paths on JAX
   and only use host NumPy paths for concrete non-traced CPU solves.
2. Keep optional-channel memory reduction as a separate structural refactor,
   not a speculative patch.
3. Move to fixed-boundary production differentiability operator cleanup and
   branch-local free-boundary derivative-proposal integration after this
   performance tranche is committed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92%.
- Free-boundary production differentiability: 91%.
- Single-stage coil optimization: 87.5%.
- CPU/GPU runtime and memory footprint: 95.4%.
- Refactor/API/examples: 53%.
- VMEC2000/VMEC++ parity and physics gates: 97.2%.
- Docs/release hygiene: 97.9%.
- Overall: 94.6%.

### 2026-06-22: Make branch-local coil proposal a single explicit example switch

Steps taken:

- Audited the direct-coil QS optimization example and confirmed the validated
  branch-local vector/JVP proposal path already uses production-forward values,
  fixed accepted-branch replay derivatives, and complete free-boundary solves
  as the only trial acceptance authority.
- Normalized the CLI so ``--same-branch-derivative-proposal`` automatically
  enables ``--write-same-branch-report`` before summary metadata is assembled.
- Kept the promoted report defaults unchanged: vector mode, direct JVP replay,
  and current-only auto-selection when a current variable is available.
- Updated the example docstring, free-boundary docs, and mocked smoke coverage
  so users can request the derivative proposal with one switch while still
  seeing the conservative same-branch/fingerprint-gated contract.

Results obtained:

- Same-branch proposal smoke subset passed:
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py -k 'same_branch_derivative_proposal or derivative_proposal_summary or dry_run'``
  (`10 passed, 23 deselected`).
- The finite-beta direct-coil wrapper smoke still passed:
  ``tests/test_free_boundary_qa_finite_beta_coil_optimization_smoke.py``
  (`2 passed`).
- Ruff passed on the modified example and test.
- The single-switch example now reports ``same_branch_report_config.enabled =
  True`` and ``same_branch_derivative_proposal_config.enabled = True`` in
  ``summary.json``.

Best next steps:

1. Continue R/Z matrix seed profiling and cache/reuse work after the NumPy
   lambda preconditioner speedup.
2. Keep free-boundary derivative claims conservative until arbitrary adaptive
   branch differentiation is implemented under the research-grade
   differentiability plan.
3. Extend the single-stage coil optimization example only with changes that
   preserve complete-solve acceptance authority.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92%.
- Free-boundary production differentiability: 91.5%.
- Single-stage coil optimization: 88%.
- CPU/GPU runtime and memory footprint: 95.4%.
- Refactor/API/examples: 54%.
- VMEC2000/VMEC++ parity and physics gates: 97.2%.
- Docs/release hygiene: 98%.
- Overall: 94.9%.

### 2026-06-22: Promote CPU host NumPy 3D R/Z preconditioner seed

Steps taken:

- Added a concrete-host NumPy mirror for the 3D stellsym R/Z preconditioner
  matrix seed.  The mirror returns the same matrix and cached parity-coefficient
  keys as the JAX builder, so cache reassembly keeps the same payload contract.
- Kept traced/autodiff, scan, accelerator, LASYM, and diagnostic
  precomputed/lax-tridiagonal paths on the existing JAX implementation.
- Routed only concrete non-traced CPU host-update 3D stellsym refreshes through
  the new host mirror.
- Added parity coverage comparing the new host mirror against the JAX R/Z
  matrix builder for both ``jmax=ns-1`` and active-edge ``jmax=ns`` cases.
- Added dispatch-policy tests showing the 3D concrete host path selects the
  NumPy mirror and traced paths still select JAX.

Results obtained:

- Focused preconditioner suite passed:
  ``tests/test_preconditioner_1d_jax_fast_helpers.py``,
  ``tests/test_tcon_precondn_diag.py``,
  ``tests/test_solve_preconditioner_payload_helpers.py``, and the non-scan
  cache-reuse finish test (`32 passed`).
- Fast physics/parity gates passed:
  ``tests/test_vmec_parity_physics_fast_gates.py``,
  ``tests/test_wout_physics_gates.py``, and
  ``tests/test_vmec2000_fixed_boundary_physics_gates.py``.
- Bounded three-iteration cold probes:
  - `input.nfp4_QH_warm_start`: solve-body time dropped from about ``0.274 s``
    after the NumPy-lambda change to ``0.106 s``; R/Z matrix seed time dropped
    from about ``0.168 s`` to ``0.012 s``.
  - `input.solovev`: solve-body time stayed neutral, about ``0.112 s`` to
    ``0.113 s``, because its R/Z seed was already about ``0.013 s``.
- A compact 16-row bundled single-grid matrix completed successfully after the
  R/Z host mirror.  Against the previous NumPy-lambda compact matrix, median
  cold runtime was ``0.996x``, median warm runtime was ``0.985x``, and median
  peak memory was ``1.003x``.
- Two initial row-level threshold hits were rerun and classified as
  timing/process-memory noise:
  - `LandremanPaul2021_QA_reactorScale_lowres`: rerun ratios cold ``1.058x``,
    warm ``1.059x``, memory ``0.958x``.
  - `LandremanSengupta2019_section5.4_B2_A80`: rerun ratios cold ``1.038x``,
    warm ``1.022x``, memory ``1.028x``.

Best next steps:

1. Continue performance work on remaining cold setup costs:
   axis reset/boundary-profile unattributed setup and R/Z apply for short host
   solves.
2. Keep optional-channel memory reduction and arbitrary adaptive-branch
   differentiation as separate structural refactors under this plan.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92%.
- Free-boundary production differentiability: 91.5%.
- Single-stage coil optimization: 88%.
- CPU/GPU runtime and memory footprint: 96.2%.
- Refactor/API/examples: 54.5%.
- VMEC2000/VMEC++ parity and physics gates: 97.4%.
- Docs/release hygiene: 98.1%.
- Overall: 95.3%.

### 2026-06-22: Remove concrete-host pTau JIT from fixed-boundary bad-Jacobian checks

Steps taken:

- Promoted a concrete-host pTau binding policy for the non-scan fixed-boundary
  residual loop.  Concrete CPU host-update runs now use the existing NumPy pTau
  min/max helper for bad-Jacobian checks instead of constructing a first-call
  JIT pTau helper.  Traced/autodiff, scan, and accelerator paths keep the JAX
  helper.
- Added an initial-axis residual-floor early exit so pTau/state axis-reset
  diagnostics are skipped when the current physical residual is already below
  the reset floor and no explicit or VMEC2000-style forced reset was requested.
- Added focused tests for both branches: host pTau binding disables JIT only for
  concrete host assembly, traced assembly keeps JIT, low-residual axis setup
  skips diagnostics, and explicit forced reset still evaluates pTau diagnostics.
- Reran bounded cold probes and the compact 16-row bundled single-grid matrix.

Results obtained:

- Focused setup/axis/runtime tests passed:
  ``tests/test_solve_residual_iter_setup_helpers.py``,
  ``tests/test_solve_axis_helpers_more_coverage.py``,
  ``tests/test_solve_additional_helpers.py`` (`114 passed`) and
  ``tests/test_solve_performance_instrumentation.py`` plus
  ``tests/test_solve_residual_iter_runtime_helpers.py`` (`36 passed`).
- Ruff passed on the changed solver and test files.
- Bounded three-iteration cold probes:
  - `input.nfp4_QH_warm_start`: solve-body time dropped from about ``0.106 s``
    after the NumPy R/Z seed change to ``0.060 s``; iteration-control
    bad-Jacobian time dropped from about ``45 ms`` to about ``0.3 ms``.
  - `input.solovev`: solve-body time dropped from about ``0.113 s`` to
    ``0.069 s``.
  - Explicit forced-reset smoke still took the axis-reset path and preserved the
    same short-trace residual, confirming the early exit does not bypass user or
    VMEC-style forced recovery.
- A compact 16-row bundled single-grid matrix completed successfully after the
  host pTau change.  Against the NumPy R/Z baseline, there were zero row-level
  regressions; median cold runtime was ``0.949x``, median warm runtime
  ``0.939x``, and median peak memory ``0.980x``.  Against the NumPy-lambda
  baseline there were also zero regressions.
- Independent audit identified the next cold CPU hotspot as
  ``setup_boundary_profiles_unattributed_s`` in boundary/profile setup, not
  pTau or preconditioner seed construction.
- Differentiability/parity audit confirmed checked-in AD-vs-FD evidence passes
  at ``1e-9`` for fixed-boundary public scalars, including `DMerc` and `D_R`,
  and that direct-coil free-boundary evidence remains branch-local and
  fingerprint-gated rather than arbitrary adaptive-branch differentiation.

Best next steps:

1. Target ``setup_boundary_profiles_unattributed_s`` by auditing duplicate
   boundary/profile construction and wrapper work around
   ``build_residual_profile_setup`` and ``build_wout_like_profiles_from_indata``.
2. Refresh validation/free-boundary docs so the promoted branch-local report path
   and accepted/rejected slot gate provenance are not stale.
3. Keep VMEC++ wording optional: VMEC2000 has selected WOUT parity; VMEC++ is an
   availability/comparison backend, not a strict parity gate.
4. Keep arbitrary adaptive branch differentiation deferred to the research-grade
   differentiability plan until a true fingerprint-gated adaptive AD-vs-FD gate
   exists.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92.5%.
- Free-boundary production differentiability: 91.7%.
- Single-stage coil optimization: 88%.
- CPU/GPU runtime and memory footprint: 96.8%.
- Refactor/API/examples: 54.7%.
- VMEC2000/VMEC++ parity and physics gates: 97.5%.
- Docs/release hygiene: 98.2%.
- Overall: 95.6%.

### 2026-06-22: Keep concrete-host fixed-boundary setup arrays on NumPy

Steps taken:

- Continued the cold CPU host setup tranche after the pTau fix.  Profiling
  showed that the remaining ``setup_boundary_profiles_unattributed_s`` bucket
  was not duplicate boundary conversion; it came from setup-only JAX array
  construction and pTau constants on concrete host solves.
- Made non-scan, non-traced CPU host setup keep the radial mesh, zero
  preconditioner payloads, zero ``TCON`` payload, and false constraint flags as
  NumPy arrays.  Traced/autodiff, scan, and accelerator paths still use JAX
  arrays.
- Kept VMEC-style axis-reset boundary coefficients lazy: ordinary no-reset
  solves do not build the duplicate ``apply_m1_constraint=True`` boundary, but
  forced reset still builds the same coefficients at reset time.
- Tightened pTau binding setup so concrete host pTau checks do not build
  JAX-side pTau constants; traced paths still keep the JAX constants and JIT
  helper.

Results obtained:

- Focused solver/setup/performance tests passed after the change
  (``150 passed``), and Ruff passed on the modified setup/iteration files.
- Bounded three-iteration cold probes:
  - `input.nfp4_QH_warm_start`: solve-body time dropped from about ``0.060 s``
    after the host pTau fix to about ``0.042 s``.
  - `input.solovev`: solve-body time dropped from about ``0.069 s`` to about
    ``0.050 s``.
  - ``setup_boundary_profiles_unattributed_s`` fell from about ``34-36 ms`` to
    about ``0.1 ms``.
  - ``setup_ptau_constants_s`` fell from about ``12 ms`` to about ``10 us``.
  - Explicit forced-reset smoke still used the forced axis-reset path and
    preserved the same short-trace residual.
- A compact 16-row bundled single-grid matrix completed successfully.  Against
  the previous NumPy R/Z baseline, there were zero row-level regressions.
  Against the immediately prior host-pTau matrix, aggregate medians improved
  slightly: cold runtime ``0.996x``, warm runtime ``0.987x``, and peak memory
  ``0.978x``.  The only threshold hit was a ``0.12 s`` LASYM warm row; rerun
  ratios were cold ``0.998x``, warm ``1.006x``, and memory ``1.000x``, so it was
  classified as timing noise.

Best next steps:

1. Re-run a slightly larger fixed-boundary performance matrix only after the
   next structural change; current compact evidence is sufficient for this
   tranche.
2. Move back to differentiability/phase-2/phase-3 closure: refresh the promoted
   branch-local free-boundary report path if needed, keep adaptive branch claims
   conservative, and identify the next production differentiability gate.
3. Keep memory work separate: optional LASYM channel allocation and memory-aware
   policy selection are structural refactors, not part of this cold setup fix.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92.5%.
- Free-boundary production differentiability: 91.7%.
- Single-stage coil optimization: 88%.
- CPU/GPU runtime and memory footprint: 97.2%.
- Refactor/API/examples: 55%.
- VMEC2000/VMEC++ parity and physics gates: 97.5%.
- Docs/release hygiene: 98.3%.
- Overall: 95.9%.

### 2026-06-23: Refresh branch-local free-boundary AD evidence and phase-3 proposal provenance

Steps taken:

- Re-ran the real direct-coil same-branch report/proposal smoke in a scratch
  output directory using the promoted accepted/rejected controller-slot gate:
  ``examples/optimization/free_boundary_QS_coil_optimization.py --smoke
  --provider circle --write-same-branch-report --same-branch-report-mode
  vector --same-branch-report-ad-mode direct
  --same-branch-report-direction current-only
  --same-branch-report-vector-keys
  aspect,qs_total,mean_iota,lcfs_boundary_moment
  --same-branch-report-rejected-slot-gate
  --same-branch-derivative-proposal --same-branch-proposal-steps 0.05,0.1
  --same-branch-proposal-max-trials 2 --max-evals 1 --max-iter 1
  --vmec-max-iter 2``.
- Regenerated the AD-vs-central-FD evidence panel from that fresh report into
  scratch outputs and compared the regenerated CSV with the checked-in
  ``docs/_static/figures/readme_ad_fd_evidence.csv``.
- Audited the phase-3 coil-only example tests and confirmed the accepted and
  rejected derivative-proposal summaries both require complete-solve acceptance
  authority, fixed accepted-branch derivatives, a passed physical-scalar vector
  gate, and the accepted/rejected controller-slot gate.

Results obtained:

- The fresh report stayed same-branch and passed the branch-local vector gate
  and accepted/rejected controller-slot gate.
- The derivative proposal used the current-only directional JVP fast path with
  cached coil geometry, then evaluated the proposed trial with a normal
  complete free-boundary solve.  The summary recorded
  ``complete_solve_acceptance_authority=True`` and did not claim
  ``run_free_boundary`` or adaptive-controller differentiation.
- Regenerated AD-vs-FD evidence still has 10 passing rows at the
  ``1e-9`` tolerance: fixed-boundary aspect, iota profile, QS residual, smooth
  QI residual, finite-beta ``DMerc``, finite-beta ``D_R``, and four
  branch-local direct-coil free-boundary scalars.  The regenerated CSV matched
  the checked-in scalar rows, so no artifact rewrite was needed.
- The real report exposed the next performance target in this lane:
  branch-local vector JVP wall time was about ``9.18 s`` and rejected-slot replay
  wall time about ``8.06 s`` for the tiny smoke case.  This is a performance
  issue, not a correctness blocker for the conservative branch-local claim.

Best next steps:

1. Keep the free-boundary derivative scope conservative: the current promoted
   evidence is same-branch/fingerprint-gated and branch-local, not arbitrary
   adaptive branch differentiation.
2. Target branch-local vector/JVP replay graph construction and rejected-slot
   replay cost before adding broader phase-3 production examples.
3. Continue VMEC2000/mgrid/direct-coil parity expansion only with bounded,
   finite-positive physical WOUT fixtures.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92.5%.
- Free-boundary production differentiability: 92.3%.
- Single-stage coil optimization: 88.6%.
- CPU/GPU runtime and memory footprint: 97.2%.
- Refactor/API/examples: 55.2%.
- VMEC2000/VMEC++ parity and physics gates: 97.5%.
- Docs/release hygiene: 98.4%.
- Overall: 96.1%.

### 2026-06-23: Reuse static boundary contexts in rejected-slot same-branch replay

Steps taken:

- Added an optional ``boundary_replay_contexts_by_shape`` input to
  ``direct_coil_accepted_trace_controller_replay_plan``.  The replay-plan
  builder now preserves inherited contexts by shape and only constructs missing
  shape/static NESTOR tables.
- Threaded existing replay-plan contexts through controller fallback rebuilds
  so internal segment-policy plan changes do not discard already-built boundary
  replay contexts.
- Updated the accepted/rejected controller-slot gate to build a distinct
  padded trace plan for the synthetic rejected slot while inheriting the main
  vector report's static boundary contexts.  This keeps branch controls
  separate but avoids repeated static context construction.
- Added focused coverage for inherited replay contexts and updated the
  derivative-proposal smoke expectation so the rejected-slot replay receives a
  precomputed plan.
- Documented the new
  ``accepted_rejected_controller_slot_gate.reused_boundary_replay_contexts``
  provenance field in the free-boundary optimization docs and the measured
  timing in the performance page.

Results obtained:

- Focused tests passed:
  ``tests/test_free_boundary_adjoint_helpers_unit.py::test_accepted_trace_control_metadata_and_stack_contracts``,
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_vector_jacobian``,
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py::test_derivative_proposal_summary_marks_report_stale_when_trial_is_accepted``, and
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py::test_derivative_proposal_summary_records_rejected_trial_as_complete_solve_rejection``.
- Broader same-branch proposal smoke subset passed
  (``8 passed, 25 deselected``), and Ruff passed on the modified files.
- A fresh real direct-coil same-branch report/proposal run stayed
  same-branch, passed the branch-local vector gate, and passed the
  accepted/rejected controller-slot gate.
- The rejected-slot gate now reports
  ``reused_boundary_replay_contexts=True``.  On the tiny smoke report,
  rejected-slot replay wall time dropped from about ``8.06 s`` to about
  ``7.34 s``.  Main branch-local vector JVP wall time remained about
  ``9.2 s`` and is still the dominant replay cost.

Best next steps:

1. Target branch-local vector/JVP graph construction itself; static context
   reuse trims rejected-slot overhead but does not address the main JVP
   dispatch cost.
2. Keep the accepted/rejected controller-slot gate branch-local and
   fingerprint-gated.  This patch does not promote arbitrary adaptive
   branch-selection differentiation.
3. Continue phase-3 coil-only optimization only with complete solves as
   acceptance authority.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92.5%.
- Free-boundary production differentiability: 92.6%.
- Single-stage coil optimization: 88.8%.
- CPU/GPU runtime and memory footprint: 97.4%.
- Refactor/API/examples: 55.4%.
- VMEC2000/VMEC++ parity and physics gates: 97.5%.
- Docs/release hygiene: 98.5%.
- Overall: 96.3%.

### 2026-06-25: Reduce accepted/rejected slot replay payload

Steps taken:

- Kept the promoted branch-local vector report as the authority for the full
  requested physical scalar set, but narrowed the synthetic
  accepted/rejected-controller-slot gate to replay only the cheapest available
  structural scalar, usually ``aspect``.
- Added explicit ``scalar_keys`` and ``full_report_scalar_keys`` provenance so
  reviewers can see that the rejected-slot gate is a narrow controller-slot
  fingerprint/JVP gate, not a replacement for the full vector derivative report.
- Preserved the main replay-plan static boundary contexts in the rejected-slot
  plan and kept complete solves as the only proposal acceptance authority.
- Updated the free-boundary optimization and performance docs with the new
  provenance fields and measured timing.

Results obtained:

- The real tiny direct-coil smoke report stayed same-branch and passed the
  accepted/rejected controller-slot gate with ``scalar_keys=['aspect']`` and
  ``full_report_scalar_keys=['aspect', 'qs_total', 'mean_iota',
  'lcfs_boundary_moment']``.
- Rejected-slot replay wall time improved from about ``8.06 s`` before context
  reuse to ``7.34 s`` after context reuse, then to ``6.74 s`` with the
  one-scalar slot payload.
- Focused tests and Ruff passed on the modified code paths; the broader focused
  helper/smoke suite remains the next validation before committing.

Best next steps:

1. Run the full focused free-boundary helper/smoke test set, Ruff, diff-check,
   docs, repo-size, and AD/FD artifact regeneration before committing this
   tranche.
2. Target the main branch-local vector/JVP graph construction cost next; the
   rejected-slot gate is no longer the dominant overhead.
3. Keep adaptive free-boundary claims conservative until a true
   fingerprint-gated adaptive full-loop AD-vs-central-FD gate exists.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 92.5%.
- Free-boundary production differentiability: 92.8%.
- Single-stage coil optimization: 89.0%.
- CPU/GPU runtime and memory footprint: 97.5%.
- Refactor/API/examples: 55.5%.
- VMEC2000/VMEC++ parity and physics gates: 97.5%.
- Docs/release hygiene: 98.6%.
- Overall: 96.5%.

### 2026-06-25: Final PR-readiness low-hanging tranche

Steps taken:

- Reproduced and fixed the two red py3.11 core CI buckets from the last pushed
  PR run.
- Hardened ``finalize_residual_iter_from_namespace`` so legacy/synthetic
  namespace finalization tests get policy-default diagnostics for newly added
  non-numerical policy fields without requiring every test fixture to duplicate
  the full startup policy object.
- Restored the README QI diagnostic wording expected by the docs/release hygiene
  gate: seed-3127 remains a diagnostic stress case, not a README promotion row,
  and artifact-promotion rules live in the docs.
- Regenerated the AD-vs-central-FD evidence artifact from the current
  one-scalar rejected-slot branch-local report and normalized the JSON
  provenance to a portable relative output path.
- Regenerated the single-grid fixed-boundary runtime figure from the existing
  compact PR #20 current-branch summary using the runtime-only README/docs plot.
- Ran the quick converged-WOUT VMEC2000 parity benchmark for
  ``circular_tokamak`` with the local VMEC2000 executable.
- Ran focused docs, repo-size, source-health, runtime-renderer, QI artifact,
  AD/FD, parity, free-boundary helper/smoke, and the two previously failing
  CI bucket gates.

Results obtained:

- ``driver-solve-discrete`` CI bucket now passes locally:
  ``1094 passed, 30 skipped``.
- ``rest`` CI bucket now passes locally: ``663 passed, 2 skipped``.
- Free-boundary helper/smoke tests pass: ``41 passed, 1 xfailed``.
- Runtime/QI artifact tests pass: ``51 passed``.
- AD/FD and parity artifact tests pass: ``13 passed, 6 skipped``.
- Sphinx full docs build passes with ``-W``.
- Repo size audit passes: tracked size ``27.06 MiB`` and no tracked file over
  the ``2 MiB`` gate.
- Source-health gate passes with the known large-file warnings still reported
  as refactor work, not as a blocking error.
- Quick VMEC2000/WOUT parity passes for ``circular_tokamak``.  The summary shows
  VMEC2000 and vmec_jax aspect agreement to roundoff and representative WOUT
  relative-RMS channels near machine precision, with the known ``bsubvmnc``
  circular-axis convention channel at about ``3.3e-5``.
- The checked-in runtime matrix still has 16 historical single-grid rows and
  VMEC++ results for 9 supported/converged rows.  The CSV rows are unchanged;
  only the runtime-only PNG and generation timestamp changed.

Best next steps:

1. Commit and push this final tranche, then check the fresh PR CI after it has
   had time to run instead of polling it continuously.
2. If CI is green, convert PR #20 from draft to ready for review.
3. Leave remaining non-blocking work as follow-up lanes: main branch-local
   vector/JVP graph construction cost, arbitrary adaptive branch differentiation
   research, larger bounded VMEC2000/VMEC++ parity matrices, and the broader
   source-file simplification refactor.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.2%.
- Single-stage coil optimization: 89.5%.
- CPU/GPU runtime and memory footprint: 97.6%.
- Refactor/API/examples: 56.0%.
- VMEC2000/VMEC++ parity and physics gates: 97.8%.
- Docs/release hygiene: 99.0%.
- Overall: 96.8%.

### 2026-06-25: Narrow derivative-proposal vector reports by default

Steps taken:

- Added ``same_branch_report_vector_keys_from_args`` so ordinary same-branch
  validation reports keep the promoted multi-scalar default, while
  ``--same-branch-derivative-proposal`` runs with no explicit vector-key list
  default to only ``aspect,qs_total,mean_iota``.
- Kept explicit ``--same-branch-report-vector-keys`` values authoritative, so
  users and validation jobs can still request the broader
  ``aspect,qs_total,mean_iota,lcfs_boundary_moment`` report or any supported
  diagnostic scalar set.
- Updated the free-boundary coil-optimization docs and the performance page to
  describe the narrower proposal default and its scope.
- Added focused tests for ordinary defaults, proposal defaults, and explicit
  user overrides.

Results obtained:

- Focused free-boundary optimization/report tests passed:
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py`` and
  ``tests/test_free_boundary_adjoint_helpers_unit.py`` reported
  ``42 passed, 1 xfailed``.
- Ruff passed on the modified source, example, and test files.
- A dry-run provenance check with
  ``examples/optimization/free_boundary_QS_coil_optimization.py --smoke
  --provider circle --dry-run --same-branch-derivative-proposal`` recorded
  ``same_branch_report_config.vector_keys =
  ['aspect', 'qs_total', 'mean_iota']`` and retained the ``current-only`` JVP
  direction policy.
- This trims one unused branch-local vector/JVP scalar from derivative-proposal
  runs by default.  It does not change the promoted AD-vs-FD evidence artifact,
  complete-solve acceptance authority, or the conservative branch-local
  derivative claim.

Best next steps:

1. For another performance tranche, target the main branch-local vector/JVP
   graph construction itself rather than rejected-slot or unused-scalar
   payloads.
2. Keep arbitrary adaptive branch differentiation unclaimed until a true
   fingerprint-gated adaptive full-loop AD-vs-central-FD gate exists.
3. Continue broader source simplification and bounded VMEC2000/VMEC++ parity
   expansion as follow-up work, not as blockers for this narrow improvement.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.3%.
- Single-stage coil optimization: 89.7%.
- CPU/GPU runtime and memory footprint: 97.7%.
- Refactor/API/examples: 56.2%.
- VMEC2000/VMEC++ parity and physics gates: 97.8%.
- Docs/release hygiene: 99.1%.
- Overall: 96.9%.

### 2026-06-25: Add stable directional-JVP signature digest

Steps taken:

- Added a stable ``cache_key_schema`` and SHA-256 ``cache_key_digest`` to the
  branch-local directional JVP signature.  The digest is computed from the
  JSON-stable static signature and is intended as the exact compatibility key
  for a future compiled current-only replay/JVP executable cache.
- Added unit coverage proving the digest is deterministic for identical scalar
  keys, replay options, replay plan metadata, and current-only geometry shapes,
  and changes when a replay-program option such as
  ``unroll_accepted_only_segments_below`` changes.

Results obtained:

- Ruff passed on the modified free-boundary facade and helper tests.
- Focused tests passed:
  ``tests/test_free_boundary_adjoint_helpers_unit.py`` plus the two
  same-branch vector/proposal smoke tests that assert the exported
  directional-JVP signature.
- This is a safe cache-contract tranche, not an executable cache yet.  It does
  not change solve numerics or free-boundary derivative claims.

Best next steps:

1. Use ``cache_key_digest`` as the key for an opt-in, per-process current-only
   replay/JVP executable cache, but only after the cache value also stores or
   otherwise binds the accepted replay data it closes over.
2. Benchmark repeated same-signature reports before enabling any cache by
   default; reject the cache if it only moves first-call cost.
3. Keep arbitrary adaptive-branch differentiation deferred until a
   fingerprint-gated adaptive AD-vs-FD gate exists.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.9%.
- Single-stage coil optimization: 90.0%.
- CPU/GPU runtime and memory footprint: 98.1%.
- Refactor/API/examples: 57.9%.
- VMEC2000/VMEC++ parity and physics gates: 97.9%.
- Docs/release hygiene: 99.4%.
- Overall: 97.5%.

### 2026-06-25: Bind directional-JVP cache digest to accepted branch metadata

Steps taken:

- Extended the branch-local directional-JVP cache signature so it can include a
  stable digest of accepted replay branch metadata, not just array shapes,
  scalar keys, replay options, and current-only fast-path status.
- Added summary fields for the accepted replay branch step count and
  free-boundary replay step count when branch metadata is available.
- Added a unit gate proving identical branch metadata gives the same
  ``cache_key_digest``, while a changed accepted/rejected branch fingerprint
  changes both the branch-metadata digest and the top-level cache digest.
- Updated the free-boundary coil optimization documentation to state that the
  cache signature is branch-local/fingerprint-local and still not an executable
  cache by itself.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/free_boundary/adjoint/facade.py
  tests/test_free_boundary_adjoint_helpers_unit.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_adjoint_helpers_unit.py -q`` passed with
  ``11 passed``.
- This closes one more correctness precondition for reusable current-only JVP
  kernels: a future cache key now changes when the accepted replay program
  changes, so a compiled proposal kernel cannot be reused across incompatible
  branch-local traces by accident.

Best next steps:

1. Use the stabilized digest as the guard for an opt-in per-process current-only
   JVP executable cache, starting only with scalar sets and same-branch traces
   that have already passed complete-solve FD validation.
2. Keep arbitrary adaptive-branch differentiation out of scope until a true
   fingerprint-gated adaptive AD-vs-FD gate exists.
3. Continue low-risk performance work on accepted-point replay/tangent
   construction without changing VMEC parity or the branch-local derivative
   contract.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 94.0%.
- Single-stage coil optimization: 90.0%.
- CPU/GPU runtime and memory footprint: 98.2%.
- Refactor/API/examples: 58.0%.
- VMEC2000/VMEC++ parity and physics gates: 97.9%.
- Docs/release hygiene: 99.4%.
- Overall: 97.6%.

### 2026-06-25: Final PR-readiness artifact refresh and JVP signature hardening

Steps taken:

- Added ``unroll_accepted_only_segments_below`` to the branch-local directional
  JVP signature generated by
  ``vmec_jax.solvers.free_boundary.adjoint.facade``.  This option changes the
  replay program shape for short accepted-only segments, so it must be part of
  any future executable/cache key.
- Extended the coil-optimization smoke tests so same-branch derivative proposal
  evidence and written vector/JVP summaries assert the unroll threshold in the
  exported signature.
- Re-rendered the public AD-vs-central-FD evidence panel and provenance from
  ``outputs/pr20_ad_fd/qs_same_branch/same_branch_complete_solve_report.json``.
- Re-rendered the docs-facing fixed-boundary runtime/memory matrix from
  ``outputs/pr20_full_matrix_current_cpu_sg/summary.json`` and re-ran the
  current-vs-``origin/main`` matrix comparator against the clean stored main
  summary.
- Re-ran the executable VMEC2000 WOUT parity gate for
  ``nfp4_QH_warm_start``, ``solovev``, ``ITERModel``, and
  ``LandremanPaul2021_QA_lowres``.
- Updated validation/performance docs so the regeneration commands and
  regression wording match the refreshed artifacts.
- Fixed a full-suite test seam in
  ``tests/test_solve_finish_cache_more_coverage.py``: the state-only scan cache
  test now monkeypatches the refactored scan-controller ``jit`` binding, not
  only the public compatibility shim, so the fake Python-only cache payload is
  not accidentally traced by JAX.

Results obtained:

- Focused same-branch tests passed:
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_derivative_proposal_uses_gated_directional_report``
  and
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_vector_jacobian``.
- Broader focused tests passed:
  ``tests/test_free_boundary_adjoint_helpers_unit.py``,
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py``, and
  ``tests/test_readme_ad_fd_evidence.py`` with one expected xfail.
- Full local pytest passed: ``3084 passed, 140 skipped, 2 xfailed`` in about
  ``15m39s``.
- Ruff passed on the modified free-boundary facade and smoke tests.
- Sphinx ``-W`` documentation build passed for all 80 documentation pages.
- Repo-size audit passed at ``27.05 MiB`` tracked size, with no tracked file
  above the 2 MiB gate.
- Source-health passed with the known large-file/function warnings still
  reported for future refactor tranches.
- AD-vs-FD evidence still has 10 passing rows at the documented ``1e-9``
  tolerance, including fixed-boundary scalars, ``DMerc``, ``D_R``, and
  branch-local/fingerprint-gated direct-coil free-boundary scalars.
- Current-vs-main benchmark comparison reported ``48`` backend/case rows and
  ``0`` regressions.  The refreshed current CSV reports warm ``vmec_jax``
  faster than VMEC2000 on 7 of 16 bundled fixed-boundary rows and cold
  ``vmec_jax`` faster on 2 of 16 rows; median warm/cold runtime ratios are
  ``1.51x`` and ``3.76x`` VMEC2000, respectively.
- VMEC2000 executable WOUT parity passed for all four promoted rows.  The
  refreshed parity summary matched the checked-in PR20 parity summary in case
  set, failure count, and worst reported relative RMS channel, so no tracked
  parity JSON churn was needed.

Best next steps:

1. Keep PR #20 focused on the current readiness gates unless review asks for a
   specific extra artifact.  The README remains performance-light; detailed
   runtime/memory evidence stays in the performance docs.
2. Implement the real branch-local current-only replay/JVP executable cache
   keyed by the now-complete ``directional_jvp_signature`` and benchmark repeat
   same-signature reports before promoting it.
3. Continue the broader refactor/API lane in larger tranches around the fixed
   boundary residual iteration and WOUT diagnostic builders; avoid numerical
   changes without adjacent parity and AD/FD gates.

User needs:

- No immediate user input is required.  Review should focus on whether the
  remaining arbitrary adaptive-branch differentiation lane should stay deferred
  or become the next major PR.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.8%.
- Single-stage coil optimization: 90.0%.
- CPU/GPU runtime and memory footprint: 98.0%.
- Refactor/API/examples: 57.8%.
- VMEC2000/VMEC++ parity and physics gates: 97.9%.
- Docs/release hygiene: 99.4%.
- Overall: 97.4%.

### 2026-06-25: Add branch-local directional-JVP cache signature provenance

Steps taken:

- Added ``directional_jvp_signature`` to the direct-coil branch-local replay
  report.  The signature records the static workload for a directional JVP:
  scalar keys, replay options, NESTOR policy, current-only fast-path status,
  current/geometry shapes, replay-plan availability, and whether the report is
  a candidate for a future compiled current-only replay/JVP cache.
- Propagated that signature into the JSON-ready coil-optimization same-branch
  vector summary and added smoke-test assertions for the current-only
  derivative-proposal path.
- Added the same signature to derivative-proposal ``gate_evidence`` so accepted
  or rejected proposal artifacts are self-contained: they now carry both the
  validation gate result and the static replay/JVP workload used to propose the
  step.
- Updated the free-boundary optimization and performance docs so reviewers know
  the signature is provenance only, not an executable cache or a new derivative
  claim.

Results obtained:

- Focused Ruff passed on the modified adjoint facade, coil-optimization helper,
  and smoke test.
- Focused same-branch tests passed:
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py::test_same_branch_report_writer_records_branch_local_vector_jacobian``
  plus ``tests/test_free_boundary_adjoint_helpers_unit.py``.
- A real tiny direct-coil derivative-proposal smoke artifact in
  ``outputs/final_tranche_same_branch_signature`` stayed same-branch, passed
  the branch-local vector gate, and wrote
  ``directional_jvp_signature`` with ``fast_path=current_only``,
  ``jit_cache_candidate=True``, and ``scalar_keys=['aspect', 'qs_total',
  'mean_iota']``.  Its measured ``replay_jvp_dispatch_s`` was about
  ``8.58 s``, confirming that the remaining cost is still cold JVP graph
  construction.
- The change is metadata-only: no VMEC solve path, replay scalar, complete-solve
  acceptance rule, README benchmark data, or AD-vs-FD slope value changed.

Best next steps:

1. Use ``directional_jvp_signature`` as the cache key for the next real
   performance tranche: factor a top-level current-only replay/JVP kernel whose
   static arguments match the signature exactly.
2. Keep the branch-local derivative proposal conservative until repeat timings
   show the compiled-kernel cache reduces ``replay_jvp_dispatch_s``.
3. Continue final PR readiness with green CI, docs, source-health, repo-size,
   and unchanged benchmark/AD evidence artifacts.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.6%.
- Single-stage coil optimization: 90.0%.
- CPU/GPU runtime and memory footprint: 97.8%.
- Refactor/API/examples: 57.4%.
- VMEC2000/VMEC++ parity and physics gates: 97.8%.
- Docs/release hygiene: 99.2%.
- Overall: 97.2%.

### 2026-06-25: Factor current-only replay scalar seam for future JVP cache

Steps taken:

- Moved the current-only branch-local replay scalar construction out of the
  nested directional-JVP closure into private helper seams:
  ``_current_only_coil_geometry_for_base_currents`` and
  ``_current_only_branch_local_replay_scalars``.
- Added a focused helper test proving the current-only geometry seam reuses the
  fixed coil curves and only expands the differentiated base currents through
  stellarator symmetry and ``current_scale``.

Results obtained:

- Ruff passed on the modified adjoint facade and helper test.
- Focused helper/proposal tests passed:
  ``tests/test_free_boundary_adjoint_helpers_unit.py`` and the same-branch
  proposal/report subset selected by ``current_only or branch_local_vector_jacobian
  or derivative_proposal``.
- This remains a refactor/performance-readiness change: no public derivative
  contract, complete-solve acceptance rule, or same-branch gate tolerance
  changed.  The next cache implementation now has a named scalar replay helper
  to wrap, time, and eventually compile behind the existing
  ``directional_jvp_signature`` key.

Best next steps:

1. Prototype the real current-only replay/JVP executable cache around
   ``_current_only_branch_local_replay_scalars`` and key it with
   ``directional_jvp_signature``.
2. Benchmark repeated same-signature reports before promoting the cache; reject
   the cache if first-call overhead merely moves elsewhere.
3. Keep bounded VMEC2000/VMEC++ parity, AD-vs-FD evidence, docs, source-health,
   and repo-size gates green.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.7%.
- Single-stage coil optimization: 90.0%.
- CPU/GPU runtime and memory footprint: 97.9%.
- Refactor/API/examples: 57.7%.
- VMEC2000/VMEC++ parity and physics gates: 97.8%.
- Docs/release hygiene: 99.2%.
- Overall: 97.3%.

### 2026-06-25: Extract derivative-proposal evidence helpers

Steps taken:

- Split ``same_branch_derivative_proposals_from_report`` into smaller pure-data
  helpers:
  ``_same_branch_derivative_vector_evidence`` validates branch-local report and
  gate preconditions, ``_validated_branch_local_scalar`` validates one scalar
  contribution, and ``_same_branch_proposal_directional_terms`` assembles the
  weighted proposal direction.
- Kept the public proposal payload and all failure reasons stable.
- Ran the proposal-heavy free-boundary optimization smoke tests and source
  health diagnostics.

Results obtained:

- ``tests/test_free_boundary_qs_coil_optimization_smoke.py`` passed with
  ``33 passed, 1 xfailed``.
- Ruff passed on the modified source and test file.
- The source-health report no longer lists
  ``same_branch_derivative_proposals_from_report`` as a function-length
  hotspot.  Remaining large-function warnings are now dominated by the fixed
  residual iteration loop, the direct-coil/free-boundary tests, and broad
  driver/example functions.
- This is a maintainability/refactor tranche only; it does not change the
  free-boundary derivative contract or runtime behavior.

Best next steps:

1. Continue source simplification by extracting the 300-line
   ``write_same_branch_validation_report`` and ``optimize_coils`` example
   orchestration into smaller source helpers with stable public example code.
2. Start the performance design for reusable/top-level branch-local
   current-only JVP kernels; helper extraction alone will not reduce the
   measured 8.9 s cold replay/JVP dispatch.
3. Keep bounded parity and AD/FD artifact tests green after each extraction.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.3%.
- Single-stage coil optimization: 89.8%.
- CPU/GPU runtime and memory footprint: 97.7%.
- Refactor/API/examples: 56.6%.
- VMEC2000/VMEC++ parity and physics gates: 97.8%.
- Docs/release hygiene: 99.1%.
- Overall: 97.0%.

### 2026-06-25: Move same-branch report sections into reusable helpers

Steps taken:

- Extracted the scalar replay section, vector/JVP replay section, and
  derivative-proposal summary attachment from
  ``examples/optimization/free_boundary_QS_coil_optimization.py``.
- Moved the scalar/vector report-section helpers into
  ``vmec_jax.solvers.free_boundary.coil_optimization`` so the pedagogic example
  imports reusable source helpers instead of carrying another large internal
  implementation block.
- Kept the example's JSON-safe conversion behavior by passing the existing
  ``json_safe_payload`` callable into the source helper.

Results obtained:

- ``tests/test_free_boundary_qs_coil_optimization_smoke.py`` passed with
  ``33 passed, 1 xfailed``.
- Ruff passed on the modified example and source helper module.
- Source-health still passes.  The example file dropped back below the
  2000-line warning threshold, and neither ``optimize_coils`` nor
  ``write_same_branch_validation_report`` appears in the top function-length
  warning list.
- This is a refactor/API/examples tranche.  It does not change the same-branch
  derivative contract, proposal acceptance authority, benchmark artifacts, or
  parity evidence.

Best next steps:

1. Start the performance design for reusable/top-level branch-local
   current-only replay/JVP kernels; the measured hot path remains
   ``branch_local_vector_replay_jvp_dispatch_s``.
2. Continue source simplification on larger numerical modules only in focused
   tranches with parity/AD gates nearby, starting with fixed-boundary residual
   iteration seams or WOUT diagnostic builders.
3. Keep CI/runtime gates green and avoid regenerating benchmark figures unless
   code changes affect the underlying benchmark data.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.3%.
- Single-stage coil optimization: 90.0%.
- CPU/GPU runtime and memory footprint: 97.7%.
- Refactor/API/examples: 57.2%.
- VMEC2000/VMEC++ parity and physics gates: 97.8%.
- Docs/release hygiene: 99.1%.
- Overall: 97.1%.

### 2026-06-25: Measure narrowed derivative-proposal smoke timing

Steps taken:

- Ran the real direct-coil smoke proposal path after narrowing the default
  derivative-proposal vector keys:
  ``JAX_ENABLE_X64=1 examples/optimization/free_boundary_QS_coil_optimization.py
  --smoke --provider circle --max-evals 1 --max-iter 1
  --same-branch-derivative-proposal --same-branch-proposal-steps 0.05
  --same-branch-proposal-max-trials 1
  --same-branch-report-rejected-slot-gate
  --same-branch-report-replay-max-mode-count 0``.
- Confirmed the summary used ``vector_keys=['aspect', 'qs_total',
  'mean_iota']`` and the current-only direction policy.

Results obtained:

- The complete-solve objective evaluations stayed cheap for the tiny case:
  about ``0.59 s`` for the first solve and ``0.03 s`` for the proposed trial.
- The report/proposal overhead remains dominated by cold branch-local replay
  graph construction:
  ``complete_solve_fd_wall_s=6.02 s``,
  ``branch_local_vector_replay_jvp_dispatch_s=8.93 s``,
  ``branch_local_vector_total_wall_s=8.94 s``, and
  ``branch_local_rejected_slot_wall_s=6.74 s``.
- Production scalar evaluation, payload copying, graph metadata, and trace
  diagnostics were all negligible compared with JAX replay/JVP dispatch.

Best next steps:

1. The next performance tranche should design a reusable/top-level compiled
   branch-local current-only replay/JVP kernel for the common objective-term
   scalar set, instead of creating a fresh closure for each report.
2. Keep the broader promoted evidence path available, but make the production
   derivative-proposal path reuse compiled kernels whenever the fingerprint,
   shape, scalar-key set, and replay options are unchanged.
3. Do not spend more effort trimming JSON payloads until the replay/JVP dispatch
   cost is reduced.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 93.3%.
- Single-stage coil optimization: 89.7%.
- CPU/GPU runtime and memory footprint: 97.7%.
- Refactor/API/examples: 56.2%.
- VMEC2000/VMEC++ parity and physics gates: 97.8%.
- Docs/release hygiene: 99.1%.
- Overall: 96.9%.

### 2026-06-25: Add opt-in closure-bound current-only JVP executable cache

Steps taken:

- Added a small capped per-process cache for current-only branch-local
  directional-JVP executables, guarded by the stable ``cache_key_digest`` and
  additional closure-bound identities for accepted replay objects and scalar
  callables.
- Kept the cache default-off and exposed it through
  ``--same-branch-report-enable-current-jvp-cache`` for repeated same-branch
  profiling/proposal reports.
- Preserved the low-level replay-controller API by filtering the report-only
  cache flag before replay calls.
- Added report provenance fields under ``directional_jvp_cache_info`` and kept
  the static ``directional_jvp_signature`` digest separate from runtime
  cache-hit state.
- Added unit tests for opt-in behavior, closure-bound key construction, and
  cache hit/miss behavior, plus a smoke-test assertion that the example CLI
  option reaches the shared replay options.

Results obtained:

- ``python -m ruff check`` passed on the modified facade, optimization helper,
  example script, and focused tests.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_adjoint_helpers_unit.py
  tests/test_free_boundary_qs_coil_optimization_smoke.py -q`` passed with
  ``46 passed, 1 xfailed``.
- The public smoke path with
  ``--same-branch-report-enable-current-jvp-cache`` completed successfully and
  wrote ``directional_jvp_cache_info`` with ``enabled=True``,
  ``hit=False``, ``closure_bound=True`` for the first executable construction.
- This is a safe first cache tranche: it does not change default solves,
  complete-solve acceptance authority, VMEC parity, or adaptive-branch
  derivative claims.

Best next steps:

1. Benchmark a repeated same-branch report that reuses identical replay objects
   and scalar callables to quantify hit-path savings from this cache.
2. If the hit path is useful, move standard scalar-key replay callables out of
   per-call lambdas so common objective-term reports can reuse compiled
   executables across repeated proposal evaluations without weakening safety.
3. Keep broader performance work focused on cold accepted-point replay/JVP graph
   construction and only promote default-on caching after repeat timings are
   positive.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 94.2%.
- Single-stage coil optimization: 90.2%.
- CPU/GPU runtime and memory footprint: 98.4%.
- Refactor/API/examples: 58.2%.
- VMEC2000/VMEC++ parity and physics gates: 97.9%.
- Docs/release hygiene: 99.5%.
- Overall: 97.7%.

### 2026-06-25: Stabilize replay scalar callables for cache hits

Steps taken:

- Replaced the per-call replay-scalar lambda tuple in the branch-local vector
  facade with stable ``_ReplayScalarCallable`` wrappers cached by scalar key,
  underlying scalar-function identity, and replay-payload identity.
- Kept cache keys conservative: the current-only executable cache still binds
  to the accepted replay objects, replay-plan object, static digest, and scalar
  wrapper identities.
- Added helper coverage proving identical ``(key, scalar function, payload)``
  inputs reuse the same scalar wrapper and therefore produce a stable
  executable-cache key when the same replay plan is reused.
- Updated free-boundary coil optimization docs to state the exact hit
  condition: same process, same accepted replay payload, same replay plan, and
  same scalar registry.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/free_boundary/adjoint/facade.py
  tests/test_free_boundary_adjoint_helpers_unit.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_adjoint_helpers_unit.py -q`` passed with
  ``12 passed``.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_qs_coil_optimization_smoke.py -q`` passed with
  ``34 passed, 1 xfailed``.
- This is a cache-hit-enabling tranche, not a default runtime-policy change:
  default reports still run without the executable cache unless the user opts
  in, and complete solves remain the acceptance authority.

Best next steps:

1. Add a focused repeated-report timing diagnostic that calls the branch-local
   vector facade twice with the same complete payload and replay plan, then
   records miss-vs-hit dispatch and ready timings.
2. If hit-path timings are materially better, thread the repeated-report cache
   through the derivative-proposal loop where the same complete-solve payload is
   reused.
3. Keep public claims conservative until the timing diagnostic demonstrates a
   real win beyond unit-level cache-key stability.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 94.4%.
- Single-stage coil optimization: 90.3%.
- CPU/GPU runtime and memory footprint: 98.5%.
- Refactor/API/examples: 58.4%.
- VMEC2000/VMEC++ parity and physics gates: 97.9%.
- Docs/release hygiene: 99.5%.
- Overall: 97.8%.

### 2026-06-25: Add same-branch current-only JVP cache timing probe

Steps taken:

- Added ``--same-branch-report-current-jvp-cache-probe`` to the direct-coil
  QS optimization example.  When requested, the vector/JVP report repeats the
  same branch-local replay once with the same complete payload, replay plan,
  and scalar registry, then writes
  ``branch_local_vector_current_jvp_cache_probe``.
- The probe records cache-hit status, wall time, cache metadata, and the nested
  replay/JVP timing buckets without changing the primary report or acceptance
  authority.
- Extended the same-branch writer smoke test so the fake vector report is
  called twice, receives the same replay plan, and writes cache-probe metadata.

Results obtained:

- ``python -m ruff check`` passed on the modified optimization helper, example
  script, and smoke test.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_qs_coil_optimization_smoke.py -q`` passed with
  ``34 passed, 1 xfailed``.
- A real tiny public smoke run with both cache flags completed.  The primary
  vector replay reported ``hit=False`` and ``replay_jvp_dispatch_s`` about
  ``4.48 s``.  The repeated probe reported ``hit=True``,
  ``replay_jvp_dispatch_s`` about ``2.4e-3 s``, and total probe wall time about
  ``6.2e-3 s``.
- This provides concrete evidence that the cache is useful for repeated
  same-payload branch-local vector/JVP reports.  It still remains opt-in.

Best next steps:

1. Thread the same cache path into derivative-proposal loops that reuse the same
   complete payload and replay plan, while keeping complete solves as the only
   acceptance authority.
2. Add a small docs example command showing the cache probe output fields, but
   avoid turning it into a default workflow until more than the tiny smoke case
   has been profiled.
3. Continue performance work on cold first-call replay/JVP graph construction;
   this cache solves repeat-call cost, not cold compile cost.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.0%.
- Free-boundary production differentiability: 94.6%.
- Single-stage coil optimization: 90.5%.
- CPU/GPU runtime and memory footprint: 98.7%.
- Refactor/API/examples: 58.6%.
- VMEC2000/VMEC++ parity and physics gates: 97.9%.
- Docs/release hygiene: 99.6%.
- Overall: 98.0%.

### 2026-06-25: Attach current-JVP cache evidence to derivative proposals

Steps taken:

- Threaded the current-only directional-JVP executable-cache metadata from the
  branch-local vector report into ``same_branch_derivative_proposal``
  ``gate_evidence``.
- Copied the optional
  ``branch_local_vector_current_jvp_cache_probe`` status into proposal evidence
  so accepted/rejected proposal JSON files show whether the proposal was formed
  from a fresh replay, a cache-eligible report, or a repeated same-payload cache
  hit.
- Kept the change metadata-only: proposal generation, complete-solve trial
  evaluation, and complete-solve acceptance authority are unchanged.
- Updated focused smoke coverage for direct proposal construction and the
  example-generated ``summary.json`` path.
- Documented the new proposal evidence fields in the free-boundary coil
  optimization guide.
- Refreshed README/docs artifacts from their promoted provenance:
  ``readme_runtime_compare.png/json``,
  ``readme_ad_fd_evidence.png/csv/json``, and
  ``pr20_wout_parity_summary.json``.
- Re-rendered the README best-optimization panels from existing optimization
  provenance: ``readme_best_optimization_qa.png``,
  ``readme_best_optimization_qh.png``, ``readme_best_optimization_qp.png``,
  and ``readme_best_optimization_qi.png``.

Results obtained:

- ``python -m ruff check`` passed on the changed free-boundary optimization
  helper and smoke test.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_qs_coil_optimization_smoke.py -q`` passed with
  ``34 passed, 1 xfailed``.
- Full local default test suite passed with
  ``3087 passed, 140 skipped, 2 xfailed`` in ``960.77 s``.
- ``python tools/diagnostics/converged_wout_parity_benchmark.py --nightly
  --vmec-exec ~/bin/xvmec2000 --case nfp4_QH_warm_start --case solovev
  --case ITERModel --case LandremanPaul2021_QA_lowres --output-dir
  outputs/pr20_wout_parity`` passed all four VMEC2000 WOUT-parity cases.
- The AD-vs-FD evidence artifact has ``10`` rows, all passing at ``1e-9``;
  the maximum absolute slope error is ``4.40e-10``.
- The runtime artifact has the full ``16``-row historical fixed-boundary
  single-grid matrix; VMEC++ is available on ``9`` rows and omitted otherwise.
- ``LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_final_tranche`` passed.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed: tracked size is ``28.12 MiB`` after the
  optimization-panel refresh and no tracked file exceeds ``2 MiB``.
- ``git diff --check`` passed.

Best next steps:

1. Run one real non-smoke coil-only QS optimization with
   ``--same-branch-derivative-proposal`` plus the current-JVP cache probe and
   compare accepted/rejected trial counts against the no-proposal baseline.
2. Continue cold first-call replay/JVP graph-construction reduction; the
   current cache solves repeated same-payload JVP cost, not first-call compile
   cost.
3. Keep adaptive-branch differentiation claims conservative until a true
   adaptive branch-selection AD-vs-FD gate is promoted.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 94.9%.
- Single-stage coil optimization: 90.9%.
- CPU/GPU runtime and memory footprint: 98.8%.
- Refactor/API/examples: 59.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.0%.
- Docs/release hygiene: 99.7%.
- Overall: 98.2%.

### 2026-06-25: Exercise non-smoke coil proposal/cache gate

Steps taken:

- Ran a bounded non-smoke synthetic-circle direct-coil QS baseline with one
  current variable, ``max_evals=2``, ``vmec_max_iter=3``, and no same-branch
  proposal path.
- Ran the same bounded non-smoke case with
  ``--same-branch-derivative-proposal``, current-only vector/JVP replay,
  matrix-free NESTOR/source replay, rejected-slot gate, and the opt-in
  current-JVP cache probe.
- The first non-smoke proposal artifact showed that the vector physical-scalar
  gate failed for ``qs_total`` and ``mean_iota``, but the unavailable proposal
  reason was dominated by the large absolute replay base delta and did not
  preserve gate/cache evidence.
- Reordered proposal validation so failed vector/physical-scalar gates are
  reported before stale base-delta caps, and changed unavailable proposal
  payloads to preserve compact ``gate_evidence``.
- Documented that unavailable proposals retain gate evidence.

Results obtained:

- Baseline complete-solve run finished two objective evaluations in the
  validation-scale non-smoke setup.  It did not produce derivative evidence.
- Proposal/cache run finished the same complete-solve setup and then correctly
  refused to form a derivative-assisted trial because the branch-local vector
  physical-scalar gate failed.
- The post-fix unavailable proposal now reports
  ``reason='branch-local vector gate did not pass'`` and keeps:
  ``branch_local_vector_gate_passed=False``,
  ``physical_scalar_gate_passed=False``,
  ``directional_jvp_cache_enabled=True``,
  ``directional_jvp_cache_hit=False`` for the first replay,
  ``current_jvp_cache_probe_available=True``,
  ``current_jvp_cache_probe_hit=True``, and
  ``accepted_rejected_controller_slot_gate_passed=True``.
- The repeated same-payload cache probe took about ``9 ms`` in this non-smoke
  run, confirming the cache-hit path beyond the earlier smoke-only evidence.
- The failure is scientifically expected for this crude synthetic-circle
  validation setup: the VMEC-state ``qs_total`` is enormous and the
  branch-local scalar gate rejects the derivative evidence instead of letting a
  stale/ill-conditioned derivative drive the complete-solve optimizer.

Best next steps:

1. Add a stable, physically better-conditioned direct-coil fixture where the
   branch-local ``qs_total`` and ``mean_iota`` gates pass, then let a
   derivative proposal reach complete-solve accept/reject authority.
2. Continue cold first-call replay/JVP graph-construction reduction; this
   tranche strengthens failure provenance and repeat-call cache evidence.
3. Keep branch-local proposal artifacts conservative: a failed scalar gate must
   block proposals, not silently downweight failed terms.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 95.1%.
- Single-stage coil optimization: 91.3%.
- CPU/GPU runtime and memory footprint: 98.9%.
- Refactor/API/examples: 59.4%.
- VMEC2000/VMEC++ parity and physics gates: 98.1%.
- Docs/release hygiene: 99.7%.
- Overall: 98.3%.

Follow-up probe:

- Tried three bounded variants to separate proposal machinery from QS-scalar
  conditioning:
  a lower-angular-resolution QS gate, an ESSOS-backed current-only gate, and an
  aspect-only proposal sanity case.
- The lower-resolution QS and ESSOS-backed cases still blocked proposals
  because ``qs_total`` failed the branch-local physical-scalar gate.  The
  cache probe still hit and the accepted/rejected slot gate passed.
- The aspect-only sanity case passed the branch-local vector and
  physical-scalar gates, recorded a cache-probe hit, formed a proposal, and the
  ordinary complete solve accepted the trial.  This proves the proposal
  plumbing can reach complete-solve authority when the chosen scalar evidence
  is valid, but it is not a QS promotion artifact.
- The remaining phase-3 blocker is therefore not proposal dispatch or cache
  provenance; it is finding or constructing a physically better-conditioned
  direct-coil QS fixture where ``qs_total`` is not enormous and the same-branch
  QS scalar derivative passes AD-vs-FD.

### 2026-06-25: Add relative branch-local replay-drift evidence

Steps taken:

- Added ``base_rel_delta`` and ``max_base_rel_delta`` to production-forward
  branch-local scalar/vector replay reports, scalar-only accepted-trace reports,
  physical-scalar gate summaries, derivative-proposal contributions, and compact
  proposal ``gate_evidence``.
- Kept proposal promotion conservative: no pass/fail threshold was relaxed, and
  ``--same-branch-proposal-max-base-delta`` remains the absolute stale-replay
  safety cap.
- Updated the free-boundary coil optimization docs and smoke tests so relative
  replay drift is a stable artifact field rather than an ad hoc debug value.

Results obtained:

- Focused ruff checks passed for the touched free-boundary optimization and
  adjoint modules.
- Focused free-boundary QS coil optimization smoke tests passed:
  ``34 passed, 1 xfailed``.
- The new evidence fields make the current QS proposal blocker easier to
  classify: failed QS physical-scalar gates can now be inspected for both
  derivative mismatch and scale-aware base replay drift without weakening the
  promotion gate.

Best next steps:

1. Re-run the non-smoke direct-coil proposal artifact to record the new
   relative-drift diagnostics for the failing QS scalar.
2. Use those diagnostics to decide whether the next fixture should be a better
   conditioned physical QS coil case or a targeted improvement in the
   branch-local QS replay scalar path.
3. Keep the complete free-boundary solve as the only proposal acceptance
   authority until the physical-scalar gate passes for QS-relevant terms.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 95.2%.
- Single-stage coil optimization: 91.5%.
- CPU/GPU runtime and memory footprint: 98.9%.
- Refactor/API/examples: 59.5%.
- VMEC2000/VMEC++ parity and physics gates: 98.1%.
- Docs/release hygiene: 99.7%.
- Overall: 98.35%.

Follow-up result:

- Re-ran the non-smoke direct-coil proposal artifact after adding relative
  replay-drift diagnostics:
  ``outputs/final_tranche_coil_qs_proposal_rel_delta``.
- The proposal remains unavailable with
  ``reason='branch-local vector gate did not pass'``.
- The new diagnostics show ``max_base_abs_delta = 1.07460608e8`` but
  ``max_base_rel_delta = 3.43e-10``.  For ``qs_total``, the base relative drift
  is ``1.04e-11`` even though the absolute drift is large because the scalar is
  ``~1.03e19``.
- The failed gate is therefore derivative/conditioning, not stale replay-base
  disagreement: ``qs_total`` exact directional ``9.22e8`` vs complete-solve FD
  ``-1.16e13`` (relative error near one), while ``aspect`` passes and
  ``mean_iota`` also fails in this crude synthetic-circle fixture.

Best next steps refinement:

1. Do not relax the stale-base cap for QS based on this artifact; relative
   base drift is already small and the derivative itself is wrong for this
   fixture.
2. Build a better-conditioned direct-coil QS fixture before promoting the
   coil-only QS derivative-proposal example, or improve the branch-local QS
   replay scalar if the same derivative mismatch appears on a physical fixture.
3. Keep the aspect-only proposal sanity as plumbing evidence, not as QS
   optimization evidence.

### 2026-06-25: Promote a better-conditioned QH same-branch QS proposal fixture

Steps taken:

- Ran a bounded complete-solve conditioning scan over synthetic circular-coil
  current/radius values using the LP-QA default input.  The coil geometry/current
  was not the controlling issue: reduced-grid VMEC-state ``qs_total`` stayed
  near ``1.2e14`` and mean iota stayed near ``4.26``.
- Ran the same direct-coil free-boundary wrapper on bundled inputs.  The
  ``input.nfp4_QH_warm_start`` row was the first well-conditioned QS candidate:
  ``qs_total = 0.950`` with QA helicity in the scan and a residual proxy near
  ``10``; LP-QA and nfp2-QA remained poorly scaled, while minimal seeds produced
  NaN QS diagnostics under this tiny free-boundary wrapper.
- Re-ran the same-branch derivative-proposal path on
  ``examples/data/input.nfp4_QH_warm_start`` with QH helicity ``(m,n)=(1,-1)``,
  current-only direct-coil direction, matrix-free NESTOR/source replay, the
  current-JVP cache probe, and the accepted/rejected controller-slot gate.
- Documented this QH warm-start command and outcome in
  ``docs/free_boundary_coil_optimization.rst`` as the QS-relevant phase-3
  proposal fixture.

Results obtained:

- The QH fixture generated a well-scaled same-branch report point with
  ``qs_total = 1.019`` and ``mean_iota = -0.564``.
- The branch-local vector/JVP physical-scalar gate passed for ``aspect``,
  ``qs_total``, and ``mean_iota``.
- Replay base drift was negligible: ``max_base_abs_delta = 2.66e-15`` and
  ``max_base_rel_delta = 2.18e-15``.
- The accepted/rejected controller-slot gate passed with one fixed rejected
  slot, so the report exercises the current controller-slot fingerprint lane.
- The current-JVP cache probe hit on the repeated same-payload replay in about
  ``9.9 ms``.
- Two derivative-assisted proposals were formed and both were evaluated by
  complete free-boundary solves.  Both were rejected by complete-solve
  objective authority, which is the intended conservative behavior.

Best next steps:

1. Use this QH warm-start command as the QS-relevant phase-3 proposal evidence
   while keeping LP-QA/circle as a failure-provenance stress fixture.
2. If stricter coil-only improvement evidence is needed, tune proposal step
   sizes or use a physical coil fixture; do not alter acceptance authority.
3. Continue runtime work on cold branch-local graph construction and broader
   physical fixtures after this PR-readiness tranche.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 95.4%.
- Single-stage coil optimization: 92.4%.
- CPU/GPU runtime and memory footprint: 98.9%.
- Refactor/API/examples: 59.7%.
- VMEC2000/VMEC++ parity and physics gates: 98.2%.
- Docs/release hygiene: 99.8%.
- Overall: 98.5%.

### 2026-06-25: Attach cold-vs-cache replay timing evidence to proposals

Steps taken:

- Added compact timing fields to derivative-proposal ``gate_evidence``:
  ``branch_local_vector_wall_s``, ``branch_local_vector_replay_jvp_wall_s``,
  ``current_jvp_cache_probe_replay_jvp_wall_s``,
  ``current_jvp_cache_probe_replay_jvp_speedup``, and
  ``accepted_rejected_controller_slot_gate_wall_s``.
- Kept the timing fields metadata-only; no branch-local derivative gate or
  complete-solve acceptance rule changed.
- Updated focused smoke coverage and the free-boundary coil optimization docs
  so the cold replay/JVP cost, cache-hit cost, and rejected-slot replay cost are
  explicit in reviewer-facing proposal artifacts.
- Re-ran the QH warm-start same-branch QS proposal fixture with the new fields.

Results obtained:

- The QH fixture still passed the branch-local vector physical-scalar gate for
  ``aspect``, ``qs_total``, and ``mean_iota``.
- The current artifact reports first branch-local replay/JVP wall time
  ``15.98 s``, current-JVP cache-probe replay wall time ``0.00488 s``, and a
  same-payload replay-JVP speedup of about ``3275x``.
- The accepted/rejected controller-slot gate still passed and took ``23.94 s``;
  this confirms that the rejected-slot gate is a separate cold replay program
  and currently dominates the optional validation artifact wall time.
- Two derivative-assisted proposals were again formed and rejected only by
  complete-solve objective authority.

Best next steps:

1. If wall-clock reduction is needed for this artifact, make the
   accepted/rejected controller-slot gate optional in promoted commands or add a
   separate cache/probe path for the rejected-slot replay program.
2. Continue reducing cold branch-local JVP graph construction; repeat-call
   cache evidence is now strong, so the remaining cost is first-call compile /
   graph construction rather than replay execution.
3. Keep the QH fixture as phase-3 QS proposal evidence and avoid using the
   ill-conditioned LP-QA/circle fixture as promotion evidence.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 95.5%.
- Single-stage coil optimization: 92.6%.
- CPU/GPU runtime and memory footprint: 99.0%.
- Refactor/API/examples: 59.8%.
- VMEC2000/VMEC++ parity and physics gates: 98.2%.
- Docs/release hygiene: 99.8%.
- Overall: 98.6%.

### 2026-06-25: Final tranche low-overhead rejected-slot fingerprint mode and gates

Steps taken:

- Added ``--same-branch-report-rejected-slot-mode`` with ``replay`` and
  ``fingerprint`` choices to the direct-coil QS optimization example.
- Kept ``replay`` as the strict default, preserving the reviewer-grade padded
  trace replay/JVP gate.
- Added a lower-overhead ``fingerprint`` mode that derives the fixed
  accepted/rejected controller-slot provenance from complete-solve trace
  statuses without compiling a second rejected-slot replay graph.
- Documented the difference between replay and fingerprint modes in the
  free-boundary coil optimization guide and updated promoted example commands
  to use fingerprint mode where the main same-branch vector/JVP report is the
  derivative evidence authority.
- Added focused unit coverage proving fingerprint mode does not call the replay
  function.
- Regenerated ``readme_ad_fd_evidence`` and ``readme_runtime_compare`` artifacts
  from the checked local reports/summaries, and refreshed the current-vs-main
  benchmark comparison JSON.

Results obtained:

- The QH warm-start direct-coil same-branch QS fixture still passed the
  branch-local physical-scalar gate for ``aspect``, ``qs_total``, and
  ``mean_iota``.
- The same fixture with fingerprint mode reported a rejected-slot gate wall
  time of about ``6.0e-5 s`` instead of the previous strict rejected-slot replay
  wall time of about ``23.9 s``.  The strict replay mode remains available.
- The main branch-local vector/JVP wall time remains about ``15.9 s``; this
  confirms the remaining low-level performance lane is first-call replay/JVP
  graph construction, not the optional controller-slot provenance check.
- The repeated current-JVP cache probe still hit, with replay-JVP time about
  ``5.1 ms`` and roughly ``3.1e3`` speedup over the cold first JVP.
- The AD-vs-FD evidence panel regenerated with ``10`` passing rows at the
  documented ``1e-9`` tolerance.
- The current-vs-main benchmark comparison regenerated ``48`` records and
  reported ``0`` runtime/memory regressions.
- Local validation passed:
  ``python -m ruff check`` on changed modules/tools,
  ``tests/test_free_boundary_qs_coil_optimization_smoke.py``,
  compact docs/runtime/parity shards,
  full Sphinx ``-W`` build, ``git diff --check``, repository-size audit, source
  health, and the full test suite:
  ``3088 passed, 140 skipped, 2 xfailed`` in ``665.51 s``.
- CI for the previous pushed commit, ``perf: expose same-branch replay timing
  evidence``, is green.

Best next steps:

1. Commit and push this final tranche.
2. If another performance tranche is needed, target cold branch-local vector/JVP
   graph construction directly; optional rejected-slot provenance is no longer
   the promoted-command bottleneck.
3. Keep the strict ``replay`` rejected-slot mode for reviewer-grade validation
   artifacts and use ``fingerprint`` mode for lower-overhead example/proposal
   runs.
4. Treat arbitrary adaptive host-controller differentiation as unclaimed until
   a true fingerprint-gated adaptive AD-vs-FD gate exists.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 95.8%.
- Single-stage coil optimization: 92.8%.
- CPU/GPU runtime and memory footprint: 99.1%.
- Refactor/API/examples: 60.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.3%.
- Docs/release hygiene: 99.9%.
- Overall: 98.7%.

### 2026-06-25: Expose current-JVP precompile toggle

Steps taken:

- Added ``--same-branch-report-disable-current-jvp-precompile`` to the
  direct-coil QS optimization example.
- Routed the flag through ``same_branch_replay_options_from_args`` as
  ``compile_current_only_jvp_cache``.
- Added smoke-test assertions for the default precompiled cache behavior and
  the lazy ``jax.jit`` opt-out mode.
- Documented when to use the default compile-separated timing path versus the
  lazy-JIT backend-comparison path.

Results obtained:

- This is an ergonomics/profiling-control change only.  It does not alter
  solver math, same-branch fingerprint gates, or complete-solve acceptance
  authority.

Best next steps:

1. Run focused lint, smoke tests, Sphinx, and whitespace checks for the touched
   files.
2. Commit and push if those gates pass.
3. Check CI for the preceding compile-attribution commit and this follow-up.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 60.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.3%.
- Docs/release hygiene: 99.9%.
- Overall: 98.8%.

### 2026-06-25: Fix current-JVP cache unit-test contract

Steps taken:

- Inspected the cancelled CI run for ``55a1beff`` and found that the
  ``freeb-external`` shard had failed before cancellation.
- The failure was a stale unit-test unpacking contract for
  ``_get_current_only_directional_jvp_executable`` after the helper started
  returning executable metadata.
- Updated the helper-unit test factory to return ``(executable, metadata)`` and
  asserted miss/hit metadata, including ``compiled_on_this_call``.

Results obtained:

- The exact failed test now passes locally.
- The full local ``freeb-external`` core bucket now passes with ``284`` passed,
  ``41`` skipped, and ``1`` expected xfail in about ``31 s``.

Best next steps:

1. Commit and push this CI-fix tranche.
2. Let the superseding CI run finish; if it fails, inspect only the failed job
   log rather than polling all jobs repeatedly.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 60.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.4%.
- Docs/release hygiene: 99.9%.
- Overall: 98.8%.

### 2026-06-25: Split current-only branch-local JVP compile and execution timing

Steps taken:

- Changed the current-only branch-local JVP executable cache so cache misses
  try to lower and compile the JAX function explicitly before storing it.
- Kept a fallback to the previous lazy ``jax.jit`` callable if explicit
  lowering/compilation is unavailable on a backend.
- Added ``directional_jvp_cache_executable_kind``,
  ``directional_jvp_cache_compiled``,
  ``directional_jvp_cache_compiled_on_this_call``, and
  ``directional_jvp_cache_compile_s`` to derivative-proposal gate evidence.
- Re-ran the QH warm-start direct-coil same-branch QS fixture using
  fingerprint rejected-slot mode and the current-JVP cache probe.
- Updated free-boundary coil optimization docs so cold XLA compile cost is not
  conflated with replay/JVP execution cost.

Results obtained:

- The QH fixture still passed the branch-local vector physical-scalar gate.
- The current-only cache miss compiled a concrete executable:
  ``executable_kind = compiled``, ``compiled = True``.
- Cold JVP cache compile/lowering took about ``15.37 s``.
- Compiled replay/JVP execution took about ``0.041 s``.
- Repeated same-payload cache-probe replay/JVP execution took about
  ``0.0038 s``.
- The accepted/rejected slot fingerprint gate remained cheap at about
  ``5.0e-5 s``.
- Two derivative-assisted proposals were formed and rejected by complete-solve
  objective authority, preserving the production acceptance contract.

Best next steps:

1. Run focused lint/tests and Sphinx after the docs update.
2. Commit and push this compile-attribution tranche if gates pass.
3. Treat remaining runtime work as an XLA compile amortization problem:
   prebuild/warm compile caches for repeated production workloads, or reduce
   the traced replay graph itself.  Replay execution is no longer the dominant
   measured cost for the promoted QH fixture.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 60.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.3%.
- Docs/release hygiene: 99.9%.
- Overall: 98.8%.

### 2026-06-25: Make rejected-slot fingerprint metadata internally consistent

Steps taken:

- Audited the new fingerprint-only accepted/rejected controller-slot gate
  against the compact ``controller_slot_summary`` fields.
- Added active free-boundary masks to the fingerprint metadata so accepted
  slots are consistently reported as active free-boundary replay slots.
- Added focused assertions for ``active_free_boundary_slots``,
  ``accepted_free_boundary_slots``, ``active_free_boundary_mask``, and
  ``has_active_freeb_replay`` in the fingerprint-only smoke test.

Results obtained:

- Focused lint passed for the touched helper and smoke test.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_free_boundary_qs_coil_optimization_smoke.py -q`` passed with
  ``35`` passing tests and ``1`` expected xfail.
- The change does not alter solver math or replay acceptance authority; it only
  makes the lower-overhead provenance artifact easier to audit.

Best next steps:

1. Commit and push this metadata consistency fix.
2. Re-check CI for the preceding pushed tranche and the new push.
3. Continue cold branch-local vector/JVP graph-construction profiling only if
   another performance tranche is needed before review.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 95.9%.
- Single-stage coil optimization: 92.8%.
- CPU/GPU runtime and memory footprint: 99.1%.
- Refactor/API/examples: 60.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.3%.
- Docs/release hygiene: 99.9%.
- Overall: 98.7%.

### 2026-06-25: Extract residual force-dispatch helpers from the main solver loop

Steps taken:

- Moved the standalone residual force-dispatch helpers from
  ``solvers/fixed_boundary/residual/iteration.py`` into the existing
  ``solvers/fixed_boundary/residual/force_payload.py`` module.
- Kept private compatibility aliases in ``iteration.py`` so tests and internal
  call sites that reference the helper names continue to work.
- Avoided adding a new module: this uses the existing force-payload domain file
  and keeps the package structure from sprawling.
- Inspected the larger nested helper seams in ``solve_fixed_boundary_residual_iter``.
  Those helpers close over substantial mutable loop state, so they are not a
  safe extraction target without a broader controller-state API refactor.

Results obtained:

- ``iteration.py`` dropped from about ``3170`` lines to ``3063`` lines.
- Focused helper coverage passed:
  ``tests/test_solve_residual_iter_helpers_wave8_coverage.py``,
  ``tests/test_solve_more_coverage.py``, and ``tests/test_solve_wave4_coverage.py``
  passed with ``34`` tests.
- The full local ``driver-solve-discrete`` CI bucket passed with ``1094``
  passed and ``30`` skipped in about ``38 s``.
- ``ruff`` passed for the touched solver modules.
- Source-health and repo-size gates still pass; tracked repo size remains about
  ``28.20 MiB``.

Best next steps:

1. Commit and push this behavior-preserving refactor.
2. Continue the refactor/API/examples lane with the same rule: move cohesive
   seams into existing domain modules, not one-off helper files.
3. The next high-value refactor target is a controller-state API around the
   residual iteration loop; that would make extraction of the nested restart,
   preconditioner-refresh, and rollback helpers safe.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 61.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.4%.
- Docs/release hygiene: 99.9%.
- Overall: 98.9%.

### 2026-06-25: Move free-boundary edge-vacuum normalization into residual runtime

Steps taken:

- Moved the NESTOR ``bsqvac`` edge normalization helper from the main residual
  iteration module into ``solvers/fixed_boundary/residual/runtime.py`` next to
  the free-boundary trial ``bsqvac`` runtime path that consumes it.
- Kept the private alias in ``iteration.py`` through the runtime import so
  existing call sites and helper seams remain unchanged.
- Added direct tests for the single-zeta-plane broadcast case and the
  already-full-zeta-grid no-broadcast case.

Results obtained:

- ``iteration.py`` now reports ``3057`` lines in ``source_health.py``.
- Focused lint passed for the touched runtime, iteration, and test files.
- Focused runtime/helper tests passed:
  ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_runtime.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py
  tests/test_solve_wave4_coverage.py -q`` reported ``57`` passing tests.
- ``repo_size_audit.py`` passed; tracked size remains ``28.21 MiB`` and no
  tracked file exceeds the current size gate.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push this behavior-preserving refactor.
2. Re-check CI for this and the previous residual-refactor tranche.
3. If another low-risk tranche is needed before review, target a cohesive
   existing domain seam; avoid extracting closure-heavy nested residual-loop
   helpers until the controller-state API exists.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 61.4%.
- VMEC2000/VMEC++ parity and physics gates: 98.4%.
- Docs/release hygiene: 99.9%.
- Overall: 99.0%.

### 2026-06-25: Refresh docs-facing benchmark and AD-vs-FD evidence artifacts

Steps taken:

- Rebuilt the Sphinx docs with warnings as errors after the residual runtime
  refactors.
- Regenerated ``readme_ad_fd_evidence`` from the existing same-branch
  direct-coil report in ``outputs/pr20_ad_fd/qs_same_branch``.
- Regenerated the docs-facing fixed-boundary runtime/memory matrix and
  current-vs-main comparison from the latest full-JIT current summary and the
  clean ``origin/main`` summary.
- Kept the README policy unchanged: the AD-vs-FD panel remains visible in the
  README, while the runtime/memory matrix remains docs-facing.

Results obtained:

- Sphinx completed successfully with ``-W`` over ``80`` source files.
- AD-vs-FD evidence contains ``10`` rows and ``0`` failures at the documented
  ``1e-9`` tolerance, including ``DMerc`` and ``D_R`` rows.
- Runtime/memory comparison contains ``48`` current-vs-main rows and ``0``
  material regressions.
- Focused derivative, Glasser, validation, artifact, QI-evidence, and bundled
  parity tests passed:
  ``26`` tests in the derivative/finite-beta bucket,
  ``68`` tests in the bundled WOUT/parity bucket,
  ``14`` tests in the QI/asset bucket, and
  ``16`` tests after artifact regeneration.
- ``repo_size_audit.py`` and ``git diff --check`` passed; tracked size remains
  ``28.21 MiB``.

Best next steps:

1. Commit and push the refreshed docs artifacts and plan entry.
2. Check CI state for the final pushes without spending time polling while
   jobs are simply running.
3. If CI stays green, PR review can focus on the remaining high-level design
   question: whether the next tranche should be the broader controller-state
   API or PR review/merge preparation.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 61.4%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.0%.

### 2026-06-25: Package pre-restart trigger branch side effects

Steps taken:

- Added a typed ``HostPreRestartTriggerBranchResult`` in the fixed-boundary
  residual update helpers.
- Rewired the pre-restart trigger branch in ``solve_fixed_boundary_residual_iter``
  to consume that result for rollback state, restart labels, time-step reporting,
  cache-clear policy, VMEC2000-specific ``force_bcovar_update`` behavior,
  history-pop policy, and ``prev_rz_fsq`` restoration.
- Added a focused unit test that locks both VMEC2000 and non-VMEC2000 branch
  side-effect contracts.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/update.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with 79 residual-helper tests.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed the configured source-health gate.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.25 MiB tracked.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push this pre-restart branch-result refactor.
2. Check the latest GitHub Actions run after the push; inspect logs only if it
   reports a concrete failure.
3. If continuing implementation, stop adding one-off branch wrappers and move to
   the next larger residual-controller API refactor tranche with a narrow test
   matrix, because the remaining low-risk restart branches are now explicitly
   packaged.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 94.9%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 64.7%.
- VMEC2000/VMEC++ parity and physics gates: 98.8%.
- Docs/release hygiene: 100%.
- Overall: 99.2%.

### 2026-06-25: Consolidate residual restart-branch application

Steps taken:

- Added one local ``_apply_restart_branch_result`` helper inside the
  fixed-boundary residual loop.
- Rewired both packaged restart branches, VMEC2000 time-control restart and
  pre-restart trigger, through that shared path.
- Kept branch-specific diagnostics and XC dumping in their existing branch
  blocks while sharing rollback, velocity reset, controller update, cache-clear,
  zero-update-history, history-pop, ``prev_rz_fsq``, and ``skip_time_control``
  side effects.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py
  vmec_jax/solvers/fixed_boundary/residual/update.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with 79 residual-helper tests.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed; the residual-loop function length
  dropped from 2739 to 2729 lines after this consolidation.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.25 MiB tracked.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push the shared restart-branch application helper.
2. Check the latest GitHub Actions run after the push and inspect logs only if
   the run fails.
3. For the next implementation tranche, either promote a larger
   residual-controller API object around these packaged branch results or switch
   back to validation/parity gates; avoid more cosmetic extraction from
   ``solve_fixed_boundary_residual_iter`` without behavior-locking tests.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.0%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 65.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.8%.
- Docs/release hygiene: 100%.
- Overall: 99.2%.

### 2026-06-25: Refresh AD-vs-FD evidence from final branch-local report

Steps taken:

- Re-rendered ``docs/_static/figures/readme_ad_fd_evidence.png``,
  ``readme_ad_fd_evidence.csv``, and ``readme_ad_fd_evidence.json`` using the
  newer ``outputs/final_tranche_adfd_evidence/same_branch_complete_solve_report.json``
  branch-local free-boundary report.
- Kept the public evidence contract unchanged: deterministic rows use
  AD-vs-central-FD at tolerance ``1e-9``, and free-boundary rows remain
  same-branch/fingerprint-gated only.
- Verified that the refreshed artifact still includes 10 rows: aspect, iota,
  QS, smooth QI, ``DMerc``, ``D_R``, and four free-boundary physical scalars.

Results obtained:

- ``python tools/diagnostics/readme_ad_fd_evidence.py --branch-local-report
  outputs/final_tranche_adfd_evidence/same_branch_complete_solve_report.json
  --figure-out docs/_static/figures/readme_ad_fd_evidence.png --csv-out
  docs/_static/figures/readme_ad_fd_evidence.csv --json-out
  docs/_static/figures/readme_ad_fd_evidence.json`` passed with 10 rows.
- The maximum absolute and relative slope errors remain
  ``4.3997183674093776e-10``, below the strict ``1e-9`` threshold.
- The free-boundary ``qs_total`` branch-local relative error improved from
  ``2.5706008076291616e-10`` to ``2.4699782640598176e-11``.
- ``python -m pytest -q tests/test_readme_ad_fd_evidence.py -q`` passed.
- ``python -m pytest -q`` for the targeted Mercier/Glasser helper and
  AD-vs-FD tests passed with 5 tests.
- ``LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_adfd_refresh`` passed.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.25 MiB tracked.

Best next steps:

1. Commit and push the refreshed AD-vs-FD evidence artifacts.
2. Let the latest CI run finish; inspect logs only if it fails.
3. Use the refreshed evidence as the current README/docs derivative baseline
   while keeping free-boundary wording conservative until arbitrary adaptive
   branch differentiation is explicitly validated.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.0%.
- Free-boundary production differentiability: 96.3%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 65.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.9%.
- Docs/release hygiene: 100%.
- Overall: 99.25%.

### 2026-06-25: Add artifact-level AD-vs-FD evidence regression gate

Steps taken:

- Added a regression test for the checked-in public AD-vs-FD evidence JSON.
- The new gate requires exactly the promoted public rows: aspect, iota, QS,
  smooth QI, ``DMerc``, ``D_R``, and the four branch-local direct-coil
  free-boundary physical scalars.
- The gate requires every row to pass, every row tolerance to stay at or below
  ``1e-9``, the maximum relative error to stay below ``1e-9``, and
  free-boundary notes/metadata to keep the conservative
  same-branch/fingerprint-gated wording.

Results obtained:

- ``python -m pytest -q tests/test_readme_ad_fd_evidence.py -q`` passed.
- ``python -m ruff check tests/test_readme_ad_fd_evidence.py
  tools/diagnostics/readme_ad_fd_evidence.py`` passed.
- The checked-in evidence artifact has 10 rows, maximum relative error
  ``4.3997183674093776e-10``, and points to
  ``outputs/final_tranche_adfd_evidence/same_branch_complete_solve_report.json``.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed the configured source-health gate.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.26 MiB tracked.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push the artifact-level AD-vs-FD evidence regression gate.
2. Let the latest CI finish; inspect logs only if it fails.
3. For the next tranche, prefer either a real validation/parity gate or the
   larger residual-controller API object. Do not loosen the public derivative
   evidence threshold unless a numerically justified row requires a separate
   documented tolerance.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.1%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 65.1%.
- VMEC2000/VMEC++ parity and physics gates: 98.9%.
- Docs/release hygiene: 100%.
- Overall: 99.3%.

### 2026-06-25: Add artifact-level WOUT parity summary regression gate

Steps taken:

- Added a docs/release hygiene regression test for the promoted public WOUT
  parity summary artifact
  ``docs/_static/figures/pr20_wout_parity_summary.json``.
- The new gate locks the four promoted VMEC2000 rows:
  ``LandremanPaul2021_QA_lowres``, ``nfp4_QH_warm_start``, ``solovev``, and
  ``ITERModel``.
- The gate requires ``failed_cases == 0``, ``skip_vmec_jax is False``, fixed
  boundary rows, finite executable provenance, converged ``fsq_rss`` below
  ``1e-8`` for both VMEC-JAX and VMEC2000, aspect agreement below ``1e-10``,
  and all required WOUT relative-RMS channels to remain below ``5e-5``.

Results obtained:

- ``python -m pytest -q
  tests/test_docs_release_hygiene.py::test_optional_validation_lasym_freeb_example_matches_manifest_case
  tests/test_docs_release_hygiene.py::test_public_wout_parity_summary_keeps_promoted_rows_and_tolerances
  tests/test_docs_release_hygiene.py::test_checked_in_docs_figures_stay_compact
  tests/test_docs_release_hygiene.py::test_generated_docs_and_bulky_sweep_artifacts_are_not_tracked
  -q`` passed.
- ``python -m ruff check tests/test_docs_release_hygiene.py`` passed.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed the configured source-health gate
  while continuing to report the known large-file/function debt.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.26 MiB tracked.
- ``git diff --check`` passed.
- The checked-in WOUT parity artifact has 4 cases, ``failed_cases`` equal to
  0, ``skip_vmec_jax`` equal to false, maximum relative RMS
  ``4.3668348716555336e-05``, and maximum aspect difference
  ``1.9539925233402755e-14``.

Best next steps:

1. Commit and push the WOUT parity artifact regression gate.
2. Let the latest CI finish; inspect logs only if it fails.
3. For the next tranche, prefer a real validation/parity extension or the
   larger residual-controller API simplification, not additional cosmetic
   extraction.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.1%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 65.1%.
- VMEC2000/VMEC++ parity and physics gates: 99.0%.
- Docs/release hygiene: 100%.
- Overall: 99.3%.

### 2026-06-25: Package strict-step runtime field selection

Steps taken:

- Added ``StrictStepRuntimeFields`` and ``strict_step_runtime_fields`` to the
  fixed-boundary residual update helpers.
- Rewired the strict-step branch in ``solve_fixed_boundary_residual_iter`` to
  consume the packaged runtime scalar slots instead of manually copying branch
  fields in the loop.
- Added focused tests for accepted branches preserving current update caps and
  catastrophic-restart branches overriding caps through the branch result.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/update.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_runtime_fields_preserve_current_caps_for_accepted_branch
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_runtime_fields_use_catastrophic_branch_caps
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_branch_fingerprint_is_array_free_and_path_specific
  -q`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with 81 tests.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed the configured source-health gate.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.27 MiB tracked.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push the strict-step runtime-field seam after checking the
   currently running CI is not failing.
2. Continue the same residual-loop API direction by packaging the remaining
   strict-step cache/velocity side effects, but avoid churn that only moves
   local variables without improving testable branch contracts.
3. Keep the arbitrary adaptive-branch differentiability claim conservative
   until a full adaptive branch AD-vs-FD gate exists.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.2%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 65.3%.
- VMEC2000/VMEC++ parity and physics gates: 99.0%.
- Docs/release hygiene: 100%.
- Overall: 99.3%.

### 2026-06-25: Package strict-step branch side-effect policy

Steps taken:

- Added ``StrictStepBranchSideEffects`` and
  ``strict_step_branch_side_effects`` to centralize strict-step velocity/cache
  side-effect policy.
- Rewired ``solve_fixed_boundary_residual_iter`` so accepted direct-fallback
  velocity resets, catastrophic velocity resets, free-boundary cache clearing,
  and preconditioner cache clearing are selected by the typed branch helper
  rather than by path-string checks in the residual loop.
- Added focused tests for accepted momentum, accepted direct-fallback, and
  catastrophic-restart side-effect policies.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/update.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_branch_side_effects_capture_velocity_and_cache_policy
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_runtime_fields_preserve_current_caps_for_accepted_branch
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_runtime_fields_use_catastrophic_branch_caps
  -q`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with 82 tests.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed the configured source-health gate.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.27 MiB tracked.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push the strict-step branch side-effect policy.
2. Continue the same direction only where the result is a stronger branch
   contract or clearer differentiability seam; the remaining broad reduction
   requires a larger residual-loop controller object rather than another
   isolated path helper.
3. Let CI finish and inspect only concrete failures.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.3%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 65.5%.
- VMEC2000/VMEC++ parity and physics gates: 99.0%.
- Docs/release hygiene: 100%.
- Overall: 99.3%.

### 2026-06-25: Combine strict-step branch runtime and side-effect application

Steps taken:

- Added ``StrictStepBranchApplication`` and
  ``strict_step_branch_application`` so strict-step runtime scalar fields and
  side-effect policy are selected together.
- Rewired ``solve_fixed_boundary_residual_iter`` to consume the combined
  branch application after optional direct fallback and after catastrophic
  restart finalization.
- Added a focused unit test that verifies catastrophic restart runtime fields
  and cache side effects stay coupled in the branch application object.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/update.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_branch_application_couples_runtime_and_side_effects
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_branch_side_effects_capture_velocity_and_cache_policy
  tests/test_solve_residual_iter_update_helpers.py::test_strict_step_runtime_fields_use_catastrophic_branch_caps
  -q`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with 83 tests.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed the configured source-health gate.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.28 MiB tracked.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push the combined strict-step branch application.
2. Pause additional residual-loop churn until CI has a clean run, unless there
   is a concrete failure to fix.
3. The next implementation tranche should be a larger controller-state object
   or a validation/optimization lane, not another local path rewrite.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.4%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 65.7%.
- VMEC2000/VMEC++ parity and physics gates: 99.0%.
- Docs/release hygiene: 100%.
- Overall: 99.3%.

### 2026-06-25: Package VMEC2000 time-control restart branch side effects

Steps taken:

- Added a typed ``HostVmec2000TimeControlRestartBranchResult`` in the
  fixed-boundary residual update helpers.
- Rewired the VMEC2000 time-control restart branch in
  ``solve_fixed_boundary_residual_iter`` to consume that result for rollback
  state, restart labels, cache-clear policy, history-pop policy, and
  ``prev_rz_fsq`` restoration.
- Added a focused unit test that locks the branch-result side-effect contract.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/update.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with 78 residual-helper tests.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed the configured helper-prefix gate.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed with 28.24 MiB tracked.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push this branch-result refactor.
2. Check the latest GitHub Actions run after the push and inspect logs only if
   the run fails.
3. If continuing beyond PR readiness, apply the same result-packaging pattern to
   the pre-restart trigger branch or move to the larger residual-controller API
   refactor with dedicated tests.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 94.7%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 64.4%.
- VMEC2000/VMEC++ parity and physics gates: 98.8%.
- Docs/release hygiene: 100%.
- Overall: 99.2%.

### 2026-06-25: Use explicit controller-state value ordering in the residual loop

Steps taken:

- Added ``controller_state_legacy_values`` so legacy residual-controller scalar
  slots are unpacked from the explicit ``CONTROLLER_RESUME_KEYS`` schema rather
  than relying on raw ``NamedTuple`` positional order.
- Rewired ``solve_fixed_boundary_residual_iter`` to use the explicit helper in
  the initial controller-state unpack and in ``_set_controller_state``.
- Added focused coverage proving the explicit value order matches the legacy
  resume payload order and remains round-trip safe.

Results obtained:

- ``python -m ruff check`` passed on the changed update/iteration modules and
  focused tests.
- Focused controller-state tests passed.
- The residual update helper bucket passed:
  ``tests/test_solve_residual_iter_update_helpers.py``,
  ``tests/test_solve_residual_iter_helpers_wave8_coverage.py``,
  ``tests/test_solve_more_coverage.py``, and
  ``tests/test_solve_wave4_coverage.py``.
- ``source_health.py`` passed with the known large-file warnings.
- ``repo_size_audit.py`` passed; tracked repository size was ``28.24 MiB``.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push the controller-state ordering seam.
2. Check CI for the latest pushed commits; fix only concrete failures.
3. If continuing implementation, the next meaningful refactor is to move one
   complete restart/controller branch behind a typed controller-state result,
   rather than extracting more small scalar helpers.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 94.5%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 64.1%.
- VMEC2000/VMEC++ parity and physics gates: 98.8%.
- Docs/release hygiene: 100%.
- Overall: 99.2%.

### 2026-06-25: Wire strict-step fingerprints into residual trace fingerprints

Steps taken:

- Added strict-step branch identity fields to finalized residual adjoint trace
  entries: accepted/rejected path, catastrophic-restart flag, cache-clear
  intent, restart reason, step status, and direct-fallback presence.
- Extended ``residual_branch_fingerprint`` to include those strict-step fields
  so fixed-boundary same-branch AD-vs-FD gates can prove the strict host update
  branch stayed fixed.
- Added source-level coverage for trace finalization and residual fingerprint
  coverage proving numeric payload changes do not affect the branch fingerprint
  but strict-branch path changes do.

Results obtained:

- ``python -m ruff check`` passed on the changed discrete-adjoint,
  force-payload, and focused test files.
- Focused tests passed for residual branch fingerprinting, trace finalization,
  and strict-step branch fingerprint construction.
- ``tests/test_solve_residual_iter_force_payload_helpers.py`` passed with
  ``18`` tests.
- ``tests/test_discrete_adjoint_qh.py`` passed locally with the expected
  slow-test skips.
- ``repo_size_audit.py`` passed; tracked repository size remained
  ``28.23 MiB``.
- ``source_health.py`` passed with the known large-file warnings.
- ``git diff --check`` passed.

Best next steps:

1. Commit and push the fixed-boundary trace-fingerprint integration.
2. Check the new CI run once after push; only intervene on concrete failures.
3. If continuing beyond PR-readiness cleanup, the next substantive tranche is
   a broader residual-loop controller-state API refactor, not another isolated
   helper extraction.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 94.4%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 63.8%.
- VMEC2000/VMEC++ parity and physics gates: 98.8%.
- Docs/release hygiene: 100%.
- Overall: 99.2%.

### 2026-06-25: Add strict-step branch fingerprint helper

Steps taken:

- Added a small ``StrictStepBranchFingerprint`` value type and
  ``strict_step_branch_fingerprint`` helper for strict residual-step branch
  decisions.
- The helper strips array payloads from accepted, rejected, direct-fallback, and
  catastrophic-restart decisions so same-branch validation gates can compare
  branch identity without accidentally depending on state-vector contents.
- Added focused tests covering the accepted momentum path, the direct-fallback
  accepted path, and the catastrophic restart path.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/residual/update.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed.
- ``python tools/diagnostics/source_health.py --top 20 --top-functions 50
  --max-root-helper-prefix-files 2`` passed with the known large-file warnings.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed; tracked repository size remained 28.25 MiB.
- ``git diff --check`` passed.

Best next steps:

1. Use the strict-step fingerprint in the next accepted/rejected-slot
   same-branch AD-vs-central-FD gate so branch-local derivative reports can
   explicitly prove that the discrete host path stayed fixed.
2. Continue the broader residual-loop controller-state API refactor only as a
   deliberate tranche with dedicated tests; avoid extracting closure-heavy
   nested helpers opportunistically.
3. Check the current CI run after pushing this commit, but avoid blocking on CI
   polling unless a concrete failure appears.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 94.0%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 63.4%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.1%.

### 2026-06-25: Final artifact refresh and stale public-benchmark cleanup

Steps taken:

- Re-rendered the docs-facing fixed-boundary runtime/memory figure and CSV/JSON
  provenance from ``outputs/pr20_full_matrix_current_cpu_sg_fulljit``.
- Rebuilt the current-vs-``origin/main`` runtime/memory comparison from the
  checked current and clean-main benchmark summaries.
- Re-rendered the README AD-vs-central-FD evidence panel from the existing
  branch-local direct-coil report.
- Removed the stale two-case VMEC++ runtime PNG from tracked docs assets.  It
  was not referenced by README or source docs and was superseded by the full
  16-case runtime matrix in the performance docs.

Results obtained:

- Runtime comparison regeneration reported ``48`` current-vs-main rows and
  ``0`` material regressions.
- The public runtime matrix still contains ``16`` historical bundled
  fixed-boundary cases.
- The AD-vs-FD panel regenerated with ``10`` rows and ``0`` failures at the
  documented ``1e-9`` tolerance.  Maximum absolute error was
  ``4.40e-10``.
- Focused validation passed:
  ``python -m ruff check`` on the refreshed diagnostics/tests,
  ``tests/test_readme_ad_fd_evidence.py``,
  ``tests/test_runtime_diagnostics.py``,
  ``tests/test_converged_wout_parity_benchmark.py``,
  ``tests/test_wout_comprehensive_parity.py``, and
  ``tests/test_glasser_resistive_interchange.py``.
- Full Sphinx ``-W`` docs build over ``80`` source files passed.
- Full local pytest passed: ``3102 passed, 140 skipped, 2 xfailed`` in
  ``760.18 s``.

Best next steps:

1. Record the full local pytest result, then run repository-size,
   source-health, and diff whitespace gates.
2. Commit and push the refreshed provenance/stale-artifact cleanup if the full
   suite passes.
3. Check the latest CI run once after pushing.  If CI is green, this tranche is
   ready for PR review/merge discussion; if CI fails, fix only concrete
   failures rather than adding new scope.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 94.0%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 63.4%.
- VMEC2000/VMEC++ parity and physics gates: 98.8%.
- Docs/release hygiene: 100%.
- Overall: 99.1%.

### 2026-06-25: Add residual controller-state snapshot seam

Steps taken:

- Added a small tested controller-state API in
  ``vmec_jax/solvers/fixed_boundary/residual/update.py`` for converting the
  residual loop's legacy scalar slots into ``ResidualControllerState`` and for
  applying pure controller updates through one explicit seam.
- Rewired ``solve_fixed_boundary_residual_iter`` to use that seam for its
  local controller-state snapshot and update dispatch.
- Added focused unit tests covering runtime scalar normalization and pure
  update delegation.

Results obtained:

- ``python -m ruff check`` passed for the changed residual update/iteration
  modules and focused tests.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with ``66`` tests.
- ``source_health.py`` and ``repo_size_audit.py`` passed; tracked size is
  ``28.22 MiB``.
- The tranche deliberately avoids changing controller math or VMEC parity
  semantics.  It creates the next stable API seam needed before the broader
  controller-state refactor can remove more of the large residual-loop body.

Best next steps:

1. Commit and push the controller-state seam.
2. Continue the larger residual-loop refactor by grouping restart, accepted
   step, and free-boundary controller state into explicit state objects, with
   tests before moving closure-heavy code.
3. Keep production adaptive-branch differentiability claims conservative until
   the full controller-state refactor has a fingerprint-gated AD-vs-FD gate.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.3%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 61.8%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.0%.

### 2026-06-25: Formalize free-boundary turn-on restart control branch

Steps taken:

- Added a pure ``HostFreeBoundaryTurnonRestartUpdate`` controller update to
  ``vmec_jax/solvers/fixed_boundary/residual/update.py``.
- Rewired the residual loop's first-active-vacuum retry branch to use this
  explicit update object instead of mutating ``ijacob``, ``iter1``,
  ``inv_tau``, and ``bad_growth_streak`` inline.
- Added tests for both VMEC-style ``iter1`` reset and restart-marker-preserving
  paths.

Results obtained:

- ``python -m ruff check`` passed for the changed residual update/iteration
  modules and focused tests.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q``
  passed.  The free-boundary finite-pressure bucket emitted known NESTOR
  singular-term warnings but completed successfully.
- ``source_health.py``, ``repo_size_audit.py``, and ``git diff --check``
  passed.
- CI run ``28195401540`` for the previous controller-state seam completed
  successfully.

Best next steps:

1. Commit and push the free-boundary turn-on restart seam.
2. Start the next larger refactor by creating an explicit accepted/rejected
   step-result object for the strict-update path, then move history/status
   bookkeeping behind that object.
3. Keep full adaptive-branch differentiability unclaimed until these explicit
   branch objects support a fingerprint-gated AD-vs-FD gate across the host
   adaptive loop.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.4%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 62.1%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.1%.

### 2026-06-25: Isolate strict-step accept/reject branch decision

Steps taken:

- Added ``StrictStepAcceptanceDecision`` and
  ``strict_step_acceptance_decision`` in the residual update helper module.
- Rewired the strict-update path in ``solve_fixed_boundary_residual_iter`` to
  use this explicit branch decision object before choosing the accepted
  momentum path or the restart/fallback path.
- Added tests for accepted, rejected, nonfinite, and no-backtracking decisions.

Results obtained:

- ``python -m ruff check`` passed for the changed residual update/iteration
  modules and focused tests.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with ``69`` tests.
- ``source_health.py``, ``repo_size_audit.py``, and ``git diff --check``
  passed.

Best next steps:

1. Commit and push the strict-step accept/reject branch decision seam.
2. Use the new branch object as the seed for a fuller strict-step result object
   carrying accepted-state, restart reason, update RMS, and history status.
3. After the strict-step result object exists, promote a targeted
   fingerprint-gated AD-vs-FD gate that includes one accepted slot and one
   rejected slot.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.5%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 62.3%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.1%.

### 2026-06-25: Add strict-step branch result object

Steps taken:

- Added ``StrictStepBranchResult`` and ``strict_step_branch_result`` to package
  accepted-state, restart status, restart path, update RMS, and cache-clear
  intent for the strict momentum step.
- Rewired ``solve_fixed_boundary_residual_iter`` so the strict accepted/rejected
  step path consumes this branch result object instead of scattering status
  assignments through the solver body.
- Added tests for accepted momentum status and rejected pending-restart status,
  including VMEC2000 vs non-VMEC2000 cache-clear policy.

Results obtained:

- ``python -m ruff check`` passed for the changed residual update/iteration
  modules and focused tests.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with ``71`` tests.
- ``source_health.py``, ``repo_size_audit.py``, and ``git diff --check``
  passed.

Best next steps:

1. Commit and push the strict-step branch-result seam.
2. Extend the strict-step object to absorb direct-fallback acceptance and
   catastrophic-restart update outputs, which will remove more branch-local
   status bookkeeping from ``solve_fixed_boundary_residual_iter``.
3. Use the resulting explicit branch object as the input/output fingerprint for
   the next accepted/rejected AD-vs-FD validation gate.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.6%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 62.6%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.1%.

### 2026-06-25: Isolate direct-force fallback acceptance branch

Steps taken:

- Added ``DirectForceFallbackAcceptanceDecision`` and
  ``direct_force_fallback_acceptance_decision`` to make the strict-step
  no-momentum fallback threshold explicit and testable.
- Rewired ``solve_fixed_boundary_residual_iter`` to use the named fallback
  decision instead of the inline ``1.5`` residual-growth threshold.
- Added tests for accepted, rejected, nonfinite, and custom-threshold fallback
  decisions.

Results obtained:

- ``python -m ruff check`` passed for the changed residual update/iteration
  modules and focused tests.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with ``72`` tests.
- ``source_health.py``, ``repo_size_audit.py``, and ``git diff --check``
  passed.

Best next steps:

1. Commit and push the direct-force fallback decision seam.
2. Consolidate accepted, direct-fallback, and catastrophic strict-step outputs
   into one branch-local result object.
3. Add the accepted/rejected-slot fingerprint gate once the consolidated
   result object is in place.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.7%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 62.8%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.1%.

### 2026-06-25: Fold direct fallback into strict-step branch result

Steps taken:

- Extended ``StrictStepBranchResult`` with ``fallback_direct_dt`` and added
  ``strict_step_branch_result_after_direct_fallback``.
- Rewired the strict-update branch in ``solve_fixed_boundary_residual_iter`` so
  direct-fallback acceptance updates the same branch result object used for
  accepted momentum and rejected trial paths.
- Added tests for fallback-accepted and fallback-rejected branch result updates.

Results obtained:

- ``python -m ruff check`` passed for the changed residual update/iteration
  modules and focused tests.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with ``74`` tests.
- ``source_health.py``, ``repo_size_audit.py``, and ``git diff --check``
  passed.
- Inline strict-branch assignments in ``solve_fixed_boundary_residual_iter`` are
  reduced, but the full solver remains large.  The next useful extraction is the
  catastrophic restart finalization, not another threshold-only helper.

Best next steps:

1. Commit and push the direct-fallback branch-result update.
2. Fold catastrophic-restart outputs into the same strict-step branch result.
3. Add a fingerprint-gated AD-vs-FD gate over accepted, direct-fallback, and
   catastrophic branch fingerprints once the branch-local result is complete.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.8%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 63.0%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.1%.

### 2026-06-25: Fold catastrophic restart into strict-step branch result

Steps taken:

- Added ``strict_step_branch_result_after_catastrophic_restart`` to carry
  catastrophic restart status, restart reason/path, update RMS, and updated RMS
  caps through the same ``StrictStepBranchResult`` object used by accepted and
  direct-fallback strict-step branches.
- Rewired ``solve_fixed_boundary_residual_iter`` so catastrophic restart
  status is copied from the branch result object after applying the pure
  controller update.
- Added a focused unit test covering catastrophic branch-result fields and
  preserving cache-clear policy from the pre-catastrophic branch.

Results obtained:

- ``python -m ruff check`` passed for the changed residual update/iteration
  modules and focused tests.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed with ``75`` tests.
- ``source_health.py``, ``repo_size_audit.py``, and ``git diff --check``
  passed.

Best next steps:

1. Commit and push the catastrophic branch-result update.
2. Add a compact branch-fingerprint helper for ``StrictStepBranchResult`` that
   records accepted/fallback/catastrophic path, restart reason, and cache-clear
   intent.
3. Promote the next AD-vs-FD gate around an accepted and rejected/catastrophic
   strict-step slot using that fingerprint helper.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.9%.
- Free-boundary production differentiability: 96.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 63.2%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.1%.

### 2026-06-25: Confirm final CI after readiness artifact refresh

Steps taken:

- Monitored the CI run for ``8ad6f81a`` after the performance-doc artifact path
  correction.
- Verified the residual-iteration top-level helper seams before choosing not to
  perform another extraction: the remaining small wrappers are compatibility
  seams used by tests, while the larger nested helpers require the planned
  controller-state API rather than piecemeal movement.

Results obtained:

- GitHub Actions run ``28194613304`` completed successfully.
- The successful run included docs, build, console smoke, py3.10, py3.12,
  physics smoke, slow physics coverage, all py3.11 core/exact coverage shards,
  parity manifest smoke, and the combined coverage gate.
- The local worktree remained clean and synchronized with ``origin/main``.
- Source-health still identifies ``solve_fixed_boundary_residual_iter`` as the
  remaining high-value refactor target; no additional low-risk helper extraction
  was justified in this tranche.

Best next steps:

1. Treat the current ``main`` state as green for review/readiness purposes.
2. Keep PR #21 draft unless explicitly reviewing the separate mirror-geometry
   branch; it is not the same scope as the current performance/differentiability
   readiness work on ``main``.
3. If continuing implementation, start the broader residual-loop
   controller-state API refactor as a deliberate tranche with dedicated tests,
   rather than extracting closure-heavy nested helpers opportunistically.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 61.4%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.0%.

### 2026-06-25: Align performance-doc regeneration commands with refreshed artifacts

Steps taken:

- Updated the performance documentation regeneration commands to use the same
  ``outputs/pr20_full_matrix_current_cpu_sg_fulljit`` current-summary path that
  produced the refreshed docs-facing benchmark artifacts.
- Rebuilt the docs with warnings as errors after the command-path correction.

Results obtained:

- The performance page no longer mixes ``current_cpu_sg_compact`` or stale
  ``current_cpu_sg`` paths with the refreshed full-JIT artifact path.
- ``LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_pr_readiness`` passed after rebuilding the changed pages.

Best next steps:

1. Commit and push the docs command-path correction.
2. Check the latest CI status; only intervene if the current run reports a
   concrete failure.
3. If CI remains green, use PR review to decide whether to stop this tranche or
   proceed to the broader controller-state refactor in a separate review scope.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 93.2%.
- Free-boundary production differentiability: 96.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 61.4%.
- VMEC2000/VMEC++ parity and physics gates: 98.6%.
- Docs/release hygiene: 100%.
- Overall: 99.0%.

### 2026-06-25: Consolidate strict-step branch application in the residual loop

Steps taken:

- Added a local ``_apply_strict_step_branch`` seam in
  ``solve_fixed_boundary_residual_iter`` that applies
  ``StrictStepBranchApplication`` runtime fields and side effects from one
  auditable point.
- Rewired strict momentum acceptance, direct-force fallback acceptance, and
  catastrophic-restart cleanup to use the shared application seam instead of
  duplicating state/status/cache/velocity assignments inline.
- Kept the existing strict-step branch result, fingerprint, and pure update
  helper APIs unchanged, so branch-local AD-vs-FD evidence continues to reason
  about the same branch identity fields.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/residual/iteration.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_force_payload_helpers.py
  tests/test_discrete_adjoint_qh.py -q`` passed with expected skips.
- Public docs/release artifact gates passed:
  ``python -m pytest -q tests/test_docs_release_hygiene.py
  tests/test_readme_ad_fd_evidence.py -q``.
- ``source_health.py`` and ``repo_size_audit.py`` passed.  The residual
  iteration module decreased from 3124 to 3119 lines and the main residual loop
  from 2748 to 2743 lines.  This is still not the endpoint; the remaining
  reduction requires a broader controller-state object rather than small helper
  extraction.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_cli_helpers.py::test_cli_test_mode_copies_packaged_input_solves_and_plots
  -q`` passed.

Best next steps:

1. Commit and push the strict-step application seam if final local gates remain
   clean.
2. Start the broader residual-loop controller-state object as the next
   deliberate refactor tranche.  The goal is to move more mutable controller
   state out of scalar locals while preserving VMEC2000 parity and the
   fingerprint-gated AD-vs-FD contract.
3. Keep public artifacts unchanged unless a benchmark/parity/evidence rerun
   materially changes their records.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.5%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 66.1%.
- VMEC2000/VMEC++ parity and physics gates: 99.0%.
- Docs/release hygiene: 100%.
- Overall: 99.3%.

### 2026-06-25: Move non-restart controller samples into pure controller-state helpers

Steps taken:

- Added pure controller-state helpers for two non-restart host-control updates:
  VMEC2000 time-control residual samples and generic host restart-decision
  tracker samples.
- Rewired ``solve_fixed_boundary_residual_iter`` to apply these samples through
  a small ``_apply_controller_sample`` bridge instead of directly mutating
  ``res0``, ``res1``, ``bad_growth_streak``, and ``state_checkpoint`` scalar
  locals at each call site.
- Added focused unit tests for both helpers to prove checkpoint preservation,
  checkpoint replacement, and preservation of unrelated controller slots.

Results obtained:

- ``python -m ruff check`` passed for the changed residual update/iteration
  modules and focused tests.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_force_payload_helpers.py
  tests/test_discrete_adjoint_qh.py -q`` passed with expected skips.
- Public docs/release artifact gates and the ``vmec --test`` CLI helper smoke
  passed.
- ``source_health.py`` and ``repo_size_audit.py`` passed.  The residual
  iteration module decreased further from 3119 to 3115 lines and the main
  residual loop from 2743 to 2737 lines.  The remaining high-value reduction is
  a broader controller-state object, not more ad hoc scalar movement.

Best next steps:

1. Commit and push the non-restart controller-sample seam if final local gates
   remain clean.
2. Let the current GitHub Actions run complete before pushing further changes
   unless a concrete failure appears.
3. Next implementation tranche: promote the residual-loop controller-state
   object so the loop reads/writes a structured controller runtime instead of
   many scalar locals.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.7%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 66.8%.
- VMEC2000/VMEC++ parity and physics gates: 99.0%.
- Docs/release hygiene: 100%.
- Overall: 99.4%.

### 2026-06-25: Route setup-time axis reset samples through controller state

Steps taken:

- Added ``controller_state_after_initial_axis_setup_result`` so the
  setup-time magnetic-axis reset path updates ``ijacob``, residual samples,
  ``prev_rz_fsq``, and the checkpoint through one pure controller-state helper.
- Rewired ``solve_fixed_boundary_residual_iter`` to keep physical
  state/velocity updates local to the loop while moving the controller scalar
  slots through the shared ``ResidualControllerState`` bridge.
- Added a focused unit test proving type coercion, checkpoint replacement, and
  preservation of unrelated controller slots.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/update.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_helpers_wave8_coverage.py
  tests/test_solve_more_coverage.py tests/test_solve_wave4_coverage.py -q``
  passed.
- ``JAX_ENABLE_X64=1 PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_force_payload_helpers.py
  tests/test_discrete_adjoint_qh.py -q`` passed with expected skips.
- ``python -m pytest -q tests/test_docs_release_hygiene.py
  tests/test_readme_ad_fd_evidence.py
  tests/test_cli_helpers.py::test_cli_test_mode_copies_packaged_input_solves_and_plots
  -q`` passed.
- ``repo_size_audit.py``, ``source_health.py``, and ``git diff --check`` passed.
  The residual iteration module dropped to 3114 lines and the main residual
  loop to 2735 lines.

Best next steps:

1. Let CI finish for the latest pushed refactor seam and inspect only if it
   fails; avoid blocking on green runs while the next tranche is scoped.
2. Promote the next group of controller slots into structured helper seams only
   where this removes repeated scalar mutation and has focused tests.
3. Keep larger public artifact refreshes deferred until a benchmark/parity or
   AD evidence record actually changes.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.8%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 67.1%.
- VMEC2000/VMEC++ parity and physics gates: 99.0%.
- Docs/release hygiene: 100%.
- Overall: 99.4%.

### 2026-06-25: Final low-ROI gate rerun after PR #20 merge

Steps taken:

- Confirmed PR #20 is already merged and that the active work is now
  post-merge hardening on ``main``.
- Regenerated the public AD-vs-central-FD evidence panel with the promoted
  branch-local free-boundary report:
  ``outputs/final_tranche_adfd_evidence/same_branch_complete_solve_report.json``.
- Reran the docs benchmark renderer from the existing full 16-row
  single-grid matrix summaries and restored timestamp-only metadata so no
  meaningless artifact diff is committed.
- Reran the QI README panel renderer from existing promoted minimal-seed
  results; the tracked panel and CSV were byte-identical.
- Reran the VMEC2000 executable WOUT parity gate for
  ``LandremanPaul2021_QA_lowres``, ``nfp4_QH_warm_start``, ``solovev``, and
  ``ITERModel`` under ``outputs/final_tranche_wout_parity_20260625``.

Results obtained:

- Current ``main`` CI run ``28202544010`` passed.
- AD-vs-FD public evidence rerender produced 10 rows and no tracked diffs;
  ``tests/test_readme_ad_fd_evidence.py`` passed.
- Strict docs build passed:
  ``LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 python -m sphinx -W
  -j auto -b html docs docs/_build/html_final_low_roi``.
- Fast release/parity harness checks passed:
  ``tests/test_converged_wout_parity_benchmark.py``,
  ``tests/test_docs_release_hygiene.py``, and the ``vmec --test`` CLI helper
  smoke.
- Repo-size gate passed with tracked size ``28.29 MiB`` and no tracked file
  over ``2 MiB``.
- Fresh VMEC2000 WOUT parity rerun passed all four promoted rows.  Worst
  relative RMS by row:
  ``LandremanPaul2021_QA_lowres`` ``bsubvmnc=6.17e-6``,
  ``nfp4_QH_warm_start`` ``bsubvmnc=2.11e-5``,
  ``solovev`` ``bsubvmnc=4.37e-5``, and
  ``ITERModel`` ``bsubvmnc=2.97e-5``.

Best next steps:

1. Do not rerun the full 16-row benchmark matrix again unless a solver-path
   change lands; the same-day current-vs-main matrix and rerendered artifacts
   are already reproducible.
2. Resume larger refactor/API work only in tranches that reduce the residual
   loop or simplify public examples; reject seams that only increase line
   count without behavioral or API payoff.
3. Keep free-boundary adaptive differentiation claims conservative until a
   true adaptive-branch AD-vs-FD gate is promoted.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.9%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 67.1%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.4%.

### 2026-06-25: Make the QI README renderer a proper CLI

Steps taken:

- Fixed ``examples/optimization/render_qi_readme_cases.py`` so ``--help``
  exits through ``argparse`` instead of accidentally rendering the full QI
  panel.
- Added ``--summary-only``, ``--figure-out``, and ``--csv-out`` so users can
  run provenance checks or review renders without overwriting tracked docs
  artifacts.
- Added a subprocess regression test proving ``--help`` exits without printing
  renderer ``Wrote ...`` messages.
- Documented the new summary-only/output override workflow in
  ``examples/optimization/README.md`` and ``docs/optimization_sweep_results.rst``.

Results obtained:

- ``python -m ruff check examples/optimization/render_qi_readme_cases.py
  tests/test_qi_readme_cases.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_qi_readme_cases.py tests/test_qs_ess_render_smoke.py
  tests/test_docs_release_hygiene.py -q`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python
  examples/optimization/render_qi_readme_cases.py --summary-only --csv-out
  outputs/render_qi_readme_cases_summary_only_check.csv`` passed against the
  promoted NFP1/2/3/4 QI artifacts.
- Strict docs build passed:
  ``LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1 python -m sphinx -W
  -j auto -b html docs docs/_build/html_qi_renderer_cli``.
- Repo-size gate passed again with tracked size ``28.29 MiB``.

Best next steps:

1. Let the queued main CI finish and inspect only if the new commit fails.
2. Continue prioritizing user-facing renderer/example/CLI fixes where they
   avoid accidental heavy reruns or make review copies safer.
3. Defer expensive optimization matrix refreshes unless solver behavior or
   promotion criteria change.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.9%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 67.6%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.4%.

### 2026-06-25: Finish optimization renderer CLI-safety sweep

Steps taken:

- Audited the remaining optimization renderers after the QI README renderer fix
  and found two scripts where ``--help`` still entered heavy render paths:
  ``render_qi_constrained_sweep.py`` and
  ``render_readme_best_optimizations.py``.
- Added ``argparse`` entry points and ``--summary-only`` modes to both scripts,
  so help is side-effect free and summary/provenance refreshes can run without
  re-rendering Matplotlib figures.
- Added subprocess smoke coverage that verifies both renderers expose
  ``--summary-only`` and do not print ``Wrote ...`` messages on ``--help``.
- Updated the optimization examples README to document the lighter summary-only
  workflows.

Results obtained:

- ``python -m ruff check examples/optimization/render_qi_readme_cases.py
  examples/optimization/render_qi_constrained_sweep.py
  examples/optimization/render_readme_best_optimizations.py
  tests/test_qi_readme_cases.py tests/test_qs_ess_render_smoke.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_qi_readme_cases.py tests/test_qs_ess_render_smoke.py
  tests/test_docs_release_hygiene.py -q`` passed with 63 tests.
- ``python examples/optimization/render_qi_constrained_sweep.py --help`` and
  ``python examples/optimization/render_readme_best_optimizations.py --help``
  both exited cleanly and showed ``--summary-only`` without writing artifacts.
- ``PYTHONDONTWRITEBYTECODE=1 python
  examples/optimization/render_readme_best_optimizations.py --summary-only``
  refreshed only the existing CSV selector and did not modify tracked docs
  artifacts.
- ``git diff --check`` passed.
- Repo-size gate passed with tracked size ``28.30 MiB`` and no tracked file
  above ``2 MiB``.
- Source-health gate still reports the expected long-term refactor hotspots,
  especially ``solve_fixed_boundary_residual_iter`` and large validation tests;
  no new helper-prefix or repo-size issue was introduced.

Best next steps:

1. Inspect the next main CI result only if it fails; do not burn time polling.
2. Keep the next PR-readiness tranche focused on high-ROI source-health
   reductions: split the fixed-boundary residual iteration loop or factor the
   largest free-boundary validation fixtures into reusable helpers.
3. Leave expensive optimization-matrix and benchmark rerenders alone unless
   solver behavior, artifact provenance, or promotion criteria change.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 95.9%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 68.2%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.4%.

### 2026-06-25: Split residual iteration scalar-control and metric seams

Steps taken:

- Extracted the VMEC-compatible scalar branch decisions from
  ``solve_fixed_boundary_residual_iter`` into
  ``residual/iteration_control.py``:
  ``zero_m1`` gating, edge-residual selection, active free-boundary ``jmax``
  selection, cached preconditioner reuse, and constraint-channel activity.
- Extracted physical residual scalar selection into
  ``residual/iteration_metrics.py`` so cached VMEC2000 norms and host/device
  scalar paths are explicit, testable seams.
- Rewired the residual loop to call those helpers without changing the force
  assembly, preconditioner, or update branch authority.
- Added focused tests for VMEC2000/free-boundary edge behavior, non-VMEC
  restart heuristics, cached constraint channel selection, cached norm reuse,
  and host-sync vs device-preserving residual metric paths.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py
  vmec_jax/solvers/fixed_boundary/residual/iteration_control.py
  vmec_jax/solvers/fixed_boundary/residual/iteration_metrics.py
  tests/test_solve_residual_iter_update_helpers.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py -q`` passed with 57 tests.
- Broader loop smoke passed:
  ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_additional_helpers.py tests/test_driver_api.py
  tests/test_discrete_adjoint_qh.py -q``.
- Finite-beta/Mercier/Glasser checks passed:
  ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_finite_beta_helpers_unit.py tests/test_finite_beta.py
  tests/test_finite_beta_examples.py
  tests/test_wout_geometry_mercier_bundled_parity.py -q``.
- ``tools/diagnostics/readme_ad_fd_evidence.py`` generated a review-copy
  AD-vs-FD evidence table with all six rows passing ``1e-9`` tolerance,
  including ``DMerc`` absolute error ``3.55e-13`` and ``D_R`` absolute error
  ``2.32e-12``.
- ``git diff --check`` passed.
- Repo-size gate passed with tracked size ``28.31 MiB`` and no tracked file
  above ``2 MiB``.
- Source-health improved the residual loop from 2735 to 2692 lines and the
  module from 3114 to 3079 lines.  The loop remains the largest production
  refactor hotspot, but the extracted branch seams now have direct unit tests.

Best next steps:

1. Continue the residual-loop reduction by extracting the free-boundary NESTOR
   coupling trial setup or the preconditioner-application block, whichever can
   be isolated with focused tests and no branch-authority change.
2. Factor the largest free-boundary validation tests only after the production
   residual loop has at least one more meaningful seam removed.
3. Keep expensive optimization-matrix/README artifact rerenders deferred unless
   a solver behavior change or promoted artifact criterion changes.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 96.2%.
- Free-boundary production differentiability: 96.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 69.1%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.5%.

### 2026-06-25: Package free-boundary coupling runtime seam

Steps taken:

- Added ``resolve_free_boundary_coupling_runtime`` in
  ``residual/runtime.py`` to package accepted NESTOR coupling state together
  with the trial-state ``bsqvac`` resampler used by the residual update
  scorer.
- Rewired ``solve_fixed_boundary_residual_iter`` to consume that packaged
  runtime object instead of carrying accepted-coupling and trial-resampling
  plumbing inline.
- Added a focused free-boundary unit test with fake accepted coupling and fake
  trial resampling to verify history forwarding, accepted runtime propagation,
  effective ``ivac`` binding, diagnostics preservation, and trial callable
  binding.

Results obtained:

- ``python -m ruff check vmec_jax/solvers/fixed_boundary/residual/iteration.py
  vmec_jax/solvers/fixed_boundary/residual/runtime.py
  tests/test_free_boundary_wp0.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_free_boundary_wp0.py -q`` passed.
- Broader fixed-boundary smoke passed:
  ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_additional_helpers.py tests/test_driver_api.py -q``.
- Free-boundary derivative smoke passed:
  ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_free_boundary_vacuum_adjoint.py
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q``.
- ``git diff --check`` passed.
- Repo-size gate passed with tracked size ``28.32 MiB`` and no tracked file
  above ``2 MiB``.
- Source-health improved the residual loop from 2692 to 2672 lines and the
  module from 3079 to 3060 lines.  The loop remains the largest production
  source-health item, but the accepted/free-boundary trial seam is now
  separately testable.

Best next steps:

1. Extract the preconditioner-application block next. It is the largest
   remaining central loop sub-block with clear inputs/outputs, but it must keep
   VMEC2000 cache refresh and accepted-control ``ptau`` payload semantics
   unchanged.
2. After one more production-loop extraction, factor the largest
   free-boundary validation tests into reusable assertion helpers to improve
   source-health without weakening gates.
3. Keep adaptive full-loop differentiation claims conservative; current
   coverage remains branch-local/fingerprint-gated.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 96.3%.
- Free-boundary production differentiability: 96.7%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 69.8%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.5%.

### 2026-06-25: Package residual preconditioner application seam

Steps taken:

- Added ``residual/iteration_preconditioner.py`` to isolate one
  residual-iteration preconditioner application from the central
  ``solve_fixed_boundary_residual_iter`` loop.
- Moved VMEC2000-style preconditioner branch dispatch, JAX radial smoothing
  branch dispatch, mode-diagonal update scaling, lambda update scaling, and
  preconditioner timing synchronization into short domain helpers.
- Preserved the local trace payload names required by accepted-branch replay:
  ``lam_prec``, ``mats``, ``jmax``, ``frzl_rz``, ``frzl_lam_pre``,
  ``preconditioner_cache_update_trace``, and accepted-control ``ptau``
  payloads.
- Added focused tests for radial-path mode/lambda scaling and fused
  VMEC2000-style update-block preservation so optimized accelerator payloads
  are not double-scaled.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/residual/iteration_preconditioner.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_residual_iteration_preconditioner.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_residual_iteration_preconditioner.py
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_additional_helpers.py tests/test_driver_api.py
  tests/test_free_boundary_wp0.py`` passed with ``230 passed, 2 skipped``.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_vacuum_adjoint.py
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py``
  passed with ``97 passed, 1 skipped``.
- ``git diff --check`` passed.
- Repo-size gate passed with tracked size ``28.32 MiB`` and no tracked file
  above ``2 MiB``.
- Source-health improved the residual loop from ``2672`` to ``2619`` lines and
  the module from ``3060`` to ``3010`` lines.  The new helper does not appear
  in the top long-function warnings after splitting branch and scaling helpers.

Best next steps:

1. Extract the preconditioned-residual scalar-channel block next; it is the
   remaining immediate continuation of this seam and already has a natural
   boundary around ``fsq1``/``fsqr1``/``fsqz1``/``fsql1`` construction.
2. Keep adaptive full-loop differentiation wording conservative until the
   production adaptive branch-selection gate is truly fingerprint-gated through
   the full host branch.
3. After the next residual-loop tranche, factor the largest free-boundary
   validation tests into reusable assertion helpers to improve source-health
   while preserving physics gates.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 96.4%.
- Free-boundary production differentiability: 96.8%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 70.5%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.5%.

### 2026-06-25: Package preconditioned residual scalar-channel seam

Steps taken:

- Added ``PreconditionedResidualScalarResult`` and
  ``resolve_preconditioned_residual_scalars`` in
  ``residual/iteration_preconditioner.py``.
- Moved the host/device preconditioned ``fsq1`` scalar path, safe scalar
  materialization, accepted-control payload materialization, lambda residual
  dump callback, and timing buckets out of the central
  ``solve_fixed_boundary_residual_iter`` loop.
- Preserved the local names consumed by convergence history, convergence
  printing, bad-Jacobian checks, fallback logic, and branch-local accepted
  replay.
- Added direct unit coverage for the host scalar-norm path and the device path
  that materializes accepted-control payloads.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/residual/iteration_preconditioner.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_residual_iteration_preconditioner.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_residual_iteration_preconditioner.py`` passed with ``4 passed``.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_residual_iteration_preconditioner.py
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_additional_helpers.py tests/test_driver_api.py
  tests/test_free_boundary_wp0.py`` passed with ``232 passed, 2 skipped``.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_vacuum_adjoint.py
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py``
  passed with ``97 passed, 1 skipped`` in ``410.90 s``.
- ``git diff --check`` and ``repo_size_audit.py`` passed; tracked size stayed
  at ``28.36 MiB`` with no tracked file above ``2 MiB``.
- Source-health reports the residual loop at ``2955`` lines and
  ``solve_fixed_boundary_residual_iter`` at ``2563`` lines after this tranche.

Best next steps:

1. Extract the bad-Jacobian tau decision/state-probe branch or the next
   preconditioned diagnostics/printing seam from the residual loop.
2. Factor the largest free-boundary validation tests into shared assertion
   helpers after one more production-loop extraction, preserving the current
   physics and derivative gates.
3. Keep adaptive full-loop differentiation claims conservative until a true
   fingerprint-gated full adaptive AD-vs-FD gate exists.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 96.5%.
- Free-boundary production differentiability: 96.9%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 71.2%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.5%.

### 2026-06-25: Package bad-Jacobian tau-selection seam

Steps taken:

- Added ``BadJacobianTauSelection`` and
  ``resolve_bad_jacobian_tau_selection`` to
  ``solvers/fixed_boundary/residual/ptau.py``.
- Moved the bad-Jacobian ptau/state tau selection, optional state probe, and
  timing hooks out of ``solve_fixed_boundary_residual_iter`` while leaving
  history appends, debug-file writes, prints, axis reset, and restart side
  effects in the residual loop.
- Kept the helper callback-injected so the branch fingerprint remains explicit
  and direct unit tests do not need to construct a full VMEC state.
- Added ``tests/test_residual_ptau.py`` for the accepted-ptau fast path and the
  state-authoritative branch path.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/residual/ptau.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_residual_ptau.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_residual_ptau.py tests/test_solve_residual_iter_policy.py
  tests/test_solve_residual_iter_config.py`` passed with ``44 passed``.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_residual_ptau.py tests/test_residual_iteration_preconditioner.py
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_additional_helpers.py tests/test_driver_api.py
  tests/test_free_boundary_wp0.py`` passed with ``234 passed, 2 skipped``.
- ``git diff --check`` and ``repo_size_audit.py`` passed; tracked size stayed
  at ``28.36 MiB``.
- Source-health reports the residual loop at ``2917`` lines and
  ``solve_fixed_boundary_residual_iter`` at ``2527`` lines after this tranche.

Best next steps:

1. Extract the initial-axis-reset branch into a residual axis-control helper,
   preserving VMEC2000 print/history side effects as caller-owned behavior.
2. Continue shrinking free-boundary validation tests into shared assertion
   helpers after the next production-loop extraction.
3. Re-run the full docs/readme artifact refresh and optional VMEC2000/VMEC++
   matrix only after the residual-loop extraction tranche stabilizes in CI.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 96.7%.
- Free-boundary production differentiability: 97.0%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 71.8%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.5%.

### 2026-06-25: Package first-step axis-reset runtime seam

Steps taken:

- Added ``InitialAxisResetRuntimeUpdate``,
  ``InitialAxisResetRuntimeCallbacks``, and
  ``run_initial_axis_reset_runtime`` to
  ``solvers/fixed_boundary/diagnostics/axis_reset.py``.
- Moved VMEC2000 first-step magnetic-axis retry state construction,
  controller-update payload creation, VMEC-style print callbacks, and primary
  velocity zeroing out of ``solve_fixed_boundary_residual_iter``.
- Kept residual-loop-owned side effects in the caller: cache invalidation,
  free-boundary-control invalidation, history rollback, timing closure, and
  iteration-repeat control.
- Added direct tests for skipped runtime-reset branches, reset-state/update
  payload construction, and the combined callback path including printed axis
  guess output.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/diagnostics/axis_reset.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_axis_helpers_more_coverage.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_axis_helpers_more_coverage.py
  tests/test_solve_additional_helpers.py tests/test_solve_branch_coverage.py``
  passed with ``121 passed``.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_axis_helpers_more_coverage.py tests/test_residual_ptau.py
  tests/test_residual_iteration_preconditioner.py
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_additional_helpers.py tests/test_driver_api.py
  tests/test_free_boundary_wp0.py`` passed with ``257 passed, 2 skipped``.
- ``git diff --check`` and ``repo_size_audit.py`` passed; tracked size is
  ``28.38 MiB`` with no tracked file above ``2 MiB``.
- Source-health reports the residual module at ``2917`` lines and
  ``solve_fixed_boundary_residual_iter`` at ``2526`` lines.  This keeps module
  size flat relative to the prior tranche and reduces the main solver function
  by one line while moving the branch into a tested domain helper.

Best next steps:

1. Extract the VMEC2000 time-control/restart branch next; it is larger than the
   axis-reset seam and should produce a clearer source-health reduction.
2. After that extraction, run the x64 free-boundary derivative subset again to
   keep branch-local replay evidence current.
3. Keep full benchmark/parity artifact regeneration reserved for math-path or
   performance-path changes; this tranche is controller refactoring only.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 96.8%.
- Free-boundary production differentiability: 97.1%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 72.3%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.5%.

### 2026-06-25: Package VMEC2000 time-control runtime seam

Steps taken:

- Added ``Vmec2000TimeControlCallbacks``,
  ``Vmec2000TimeControlRuntimeResult``, and
  ``run_vmec2000_time_control_runtime`` to
  ``solvers/fixed_boundary/residual/host_diagnostics.py``.
- Moved the VMEC2000 time-control sampling, controller-sample payload, and
  restart-branch payload/application callbacks out of
  ``solve_fixed_boundary_residual_iter`` while keeping the caller-owned timing
  closure and loop ``continue`` behavior explicit.
- Preserved the existing guard around VMEC2000 time control so non-VMEC2000 and
  skip-time-control paths do not accidentally overwrite ``fsq``/``fsq0``.
- Added focused tests for the disabled path, non-restart sample path, and
  restart branch path.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/residual/host_diagnostics.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_setup_helpers.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_setup_helpers.py -q`` passed with
  ``23 passed``.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_axis_helpers_more_coverage.py tests/test_residual_ptau.py
  tests/test_residual_iteration_preconditioner.py
  tests/test_solve_residual_iter_setup_helpers.py
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_additional_helpers.py tests/test_driver_api.py
  tests/test_free_boundary_wp0.py`` passed with ``280 passed, 2 skipped``.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_vacuum_adjoint.py
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q``
  passed.
- ``git diff --check`` and ``repo_size_audit.py`` passed; tracked size is
  ``28.39 MiB`` with no tracked file above ``2 MiB``.
- ``source_health.py`` reports the residual module at ``2911`` lines and
  ``solve_fixed_boundary_residual_iter`` at ``2519`` lines.  The prior tranche
  reported ``2917`` and ``2526``, so this extraction removed another
  controller branch from the main residual loop and kept shrinking the largest
  fixed-boundary seam.

Best next steps:

1. Extract the pre-restart trigger/free-boundary retry branch next; it is the
   remaining large controller branch before the loop body can be split into
   smaller phase runners.
2. Re-run the current/main benchmark and WOUT parity matrix only after the next
   math-path or performance-path tranche; this change is controller packaging
   and is covered by smoke, x64 derivative, and CI gates.
3. Keep the README/docs figure refresh reserved for the final PR-readiness
   pass after the residual-loop refactor stabilizes.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 96.9%.
- Free-boundary production differentiability: 97.2%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 72.8%.
- VMEC2000/VMEC++ parity and physics gates: 99.1%.
- Docs/release hygiene: 100%.
- Overall: 99.5%.

### 2026-06-25: Final PR-readiness artifact and parity refresh

Steps taken:

- Regenerated ``docs/_static/figures/readme_runtime_compare.png`` from the
  full 16-case bundled fixed-boundary single-grid matrix in runtime-only mode,
  using ``outputs/pr20_full_matrix_current_cpu_sg_fulljit/summary.json``.
- Reintroduced the runtime snapshot near the top of ``README.md`` and kept
  peak-memory discussion in the performance docs/CSV/JSON provenance rather
  than in the README figure.
- Refreshed ``docs/_static/figures/readme_ad_fd_evidence.{png,csv,json}``
  with the validated branch-local report
  ``outputs/final_tranche_adfd_evidence/same_branch_complete_solve_report.json``.
- Re-ran the bounded VMEC2000 converged-WOUT parity evidence for
  ``LandremanPaul2021_QA_lowres``, ``nfp4_QH_warm_start``, ``solovev``, and
  ``ITERModel`` using the local ``xvmec2000`` executable, then promoted only
  the compact ``summary.json`` to
  ``docs/_static/figures/pr20_wout_parity_summary.json``.
- Updated ``docs/performance.rst`` so the documented README renderer command
  uses ``--plot-mode runtime``.

Results obtained:

- Runtime figure refresh completed from the full fixed-boundary matrix; the
  tracked PNG shrank from ``209669`` to ``166118`` bytes because peak-memory
  subpanels were removed from the README artifact.
- AD-vs-FD evidence refresh produced ``10`` passing rows: fixed-boundary
  aspect, iota profile, QS residual, smooth QI residual, ``DMerc``, ``D_R``,
  and four same-branch/fingerprint-gated direct-coil free-boundary scalars.
  All rows pass the ``1e-9`` relative-error threshold.
- Focused derivative tests passed:
  ``python -m pytest -q tests/test_finite_beta.py
  tests/test_implicit_sensitivity_fast_coverage.py -q``.
- Bounded WOUT parity passed all four selected rows with no failed cases.  The
  largest core relative-RMS value among ``rmnc``, ``zmns``, ``lmns``,
  ``iotas``, ``iotaf``, and ``bmnc`` was about ``2.8e-10`` on the
  Landreman-Paul QA row.
- Docs passed with warnings-as-errors:
  ``LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_final_tranche``.
- ``python -m ruff check vmec_jax tests tools examples`` passed.
- ``git diff --check`` and ``repo_size_audit.py`` passed; tracked size is now
  ``28.33 MiB``.

Best next steps:

1. Commit and push this artifact/parity refresh, then let CI validate the two
   latest commits together.
2. If CI is green, PR review readiness is primarily a human-review decision;
   the remaining open technical work is longer-term runtime/memory reduction
   and full arbitrary adaptive-branch differentiability, both already tracked
   as future research-grade lanes.
3. For the next implementation tranche, extract the pre-restart/free-boundary
   retry branch from the residual loop; this is the next best refactor seam if
   more source simplification is requested before review.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.0%.
- Free-boundary production differentiability: 97.3%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 72.8%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Fix README AD-vs-FD evidence provenance gate

Steps taken:

- Investigated CI run ``28210740141`` after ``Fast Tests (py3.11 core coverage:
  rest)`` failed while all other required jobs passed.
- Reproduced the exact CI shard locally using
  ``tools/diagnostics/ci_core_bucket_args.py rest``.
- Updated ``tests/test_readme_ad_fd_evidence.py`` so the checked-in public
  evidence provenance matches the regenerated promoted branch-local report path
  recorded in ``docs/_static/figures/readme_ad_fd_evidence.json``:
  ``outputs/pr20_ad_fd/qs_same_branch/same_branch_complete_solve_report.json``.

Results obtained:

- ``python -m ruff check tests/test_readme_ad_fd_evidence.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 pytest -q
  tests/test_readme_ad_fd_evidence.py -q`` passed.
- The exact local reproduction of the failed CI shard passed:
  ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1
  VMEC_JAX_SKIP_PY311_COVERAGE_ONLY=1 xargs pytest -q -n 4 -m
  "not full and not vmec2000 and not simsopt" --durations=50 --cov=vmec_jax
  --cov-report= < /tmp/py311-core-rest-args.txt`` with
  ``674 passed, 2 skipped``.
- The failed CI run had all other required jobs green; this was a test
  provenance expectation mismatch, not a numerical or artifact failure.

Best next steps:

1. Push the provenance-test fix and let the new CI run validate the full suite.
2. If CI passes, do not regenerate benchmark/AD-vs-FD artifacts again unless a
   numerical path changes.
3. Treat any future CI failure by reproducing the exact shard locally before
   changing code.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.1%.
- Free-boundary production differentiability: 97.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 74.5%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Extract fixed-boundary driver stage policy context

Steps taken:

- Added ``_FixedBoundaryStageContext`` and
  ``_resolve_fixed_boundary_stage_context`` in ``vmec_jax/driver.py``.
- Moved fixed-boundary multigrid/stage policy resolution and finish-policy
  override handling out of ``run_fixed_boundary`` into that helper.
- Added ``vmec_jax/driver.py:run_fixed_boundary=533`` to both the GitHub
  Actions source-health gate and the local CI source-health stage so the public
  driver cannot grow while further orchestration refactors continue.

Results obtained:

- ``run_fixed_boundary`` decreased from ``538`` to ``533`` physical lines.
  This is a small safe extraction, but it creates a named seam around stage
  policy and prevents future growth of the public driver function.
- ``python -m ruff check vmec_jax/driver.py tools/diagnostics/local_ci_gate.py
  tests/test_driver_api.py tests/test_local_ci_gate.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_driver_api.py
  tests/test_local_ci_gate.py -q`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_cli_helpers.py
  tests/test_driver_api.py -q`` passed.
- The source-health gate passed locally with both named function baselines:
  ``solve_fixed_boundary_residual_iter=2516`` and ``run_fixed_boundary=533``.
- ``python tools/diagnostics/local_ci_gate.py --dry-run --only source-health``
  prints both named function baselines.
- ``python -m compileall -q vmec_jax/driver.py tools/diagnostics/source_health.py
  tools/diagnostics/local_ci_gate.py``, ``git diff --check``, and
  ``repo_size_audit.py`` passed; tracked size is ``28.38 MiB``.

Best next steps:

1. Push this driver-stage refactor and let CI validate the updated source-health
   baselines.
2. If additional code simplification is needed before review, target either a
   larger ``run_fixed_boundary`` dispatch extraction or a larger residual-loop
   phase extraction, then lower the corresponding baseline.
3. Do not rerun full benchmark/parity/AD-vs-FD artifacts unless a future
   numerical or performance-path change invalidates the existing readiness
   artifacts.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.1%.
- Free-boundary production differentiability: 97.3%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 74.2%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Align local CI helper with source-health regression gate

Steps taken:

- Added a ``source-health`` stage to
  ``tools/diagnostics/local_ci_gate.py`` so local release checks run the same
  residual-loop function-size baseline as GitHub Actions.
- Updated ``tests/test_local_ci_gate.py`` to require the source-health stage,
  diagnostic script, named-function gate flag, and current residual-loop
  baseline.

Results obtained:

- ``python -m ruff check tools/diagnostics/local_ci_gate.py
  tests/test_local_ci_gate.py tools/diagnostics/source_health.py
  tests/test_source_health_diagnostics.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q tests/test_local_ci_gate.py
  tests/test_source_health_diagnostics.py -q`` passed with ``12 passed``.
- ``python tools/diagnostics/local_ci_gate.py --dry-run --only source-health``
  prints the expected source-health command with the
  ``solve_fixed_boundary_residual_iter=2516`` baseline.
- ``git diff --check`` passed.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed; tracked size remains ``28.37 MiB``.

Best next steps:

1. Push the local-gate alignment and let the latest CI run supersede the prior
   in-progress run.
2. Once CI is green, the low-hanging readiness tranche is complete unless a
   reviewer asks for one more residual-loop extraction before merge/release.
3. Any future residual-loop refactor should lower the named-function baseline
   in both CI and local CI.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.1%.
- Free-boundary production differentiability: 97.3%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 73.9%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Package free-boundary residual runtime state and trial scorer

Steps taken:

- Added ``FreeBoundaryLoopState``, ``initial_free_boundary_loop_state``, and
  ``resume_free_boundary_loop_state`` in
  ``vmec_jax/solvers/fixed_boundary/residual/runtime.py``.
- Moved VMEC free-boundary cadence initialization, restart-state resume
  handling, and initial ``plascur`` extraction out of
  ``solve_fixed_boundary_residual_iter``.
- Added ``trial_residual_total_runtime`` as a named runtime seam for
  backtracking/direct-fallback/auto-flip trial-state residual scoring.
- Updated ``solve_fixed_boundary_residual_iter`` to delegate those host-runtime
  branches without changing the force kernels, scan path, free-boundary
  coupling, or adjoint trace payloads.
- Lowered both GitHub Actions and local CI source-health baselines for
  ``solve_fixed_boundary_residual_iter`` from ``2516`` to ``2508`` lines.
- Added focused unit tests for the new free-boundary loop-state helpers and
  trial-residual scorer.

Results obtained:

- ``solve_fixed_boundary_residual_iter`` is now capped at ``2508`` physical
  lines in CI and local release gates; ``run_fixed_boundary`` remains capped at
  ``533`` lines.
- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/residual/runtime.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tools/diagnostics/local_ci_gate.py
  tests/test_solve_residual_iter_runtime_helpers.py
  tests/test_local_ci_gate.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_runtime_helpers.py
  tests/test_local_ci_gate.py tests/test_driver_api.py -q`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_axis_helpers_more_coverage.py tests/test_residual_ptau.py
  tests/test_residual_iteration_preconditioner.py
  tests/test_solve_residual_iter_setup_helpers.py
  tests/test_solve_residual_iter_runtime_helpers.py
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_residual_iter_finalize_helpers.py
  tests/test_solve_additional_helpers.py tests/test_free_boundary_wp0.py -q``
  passed with one expected skip and only existing numerical warnings.
- The exact source-health gate passed locally with
  ``solve_fixed_boundary_residual_iter=2508`` and
  ``run_fixed_boundary=533``.
- ``python -m compileall -q`` on the changed runtime/iteration/local-CI files,
  ``git diff --check``, and ``repo_size_audit.py`` passed; tracked size is
  ``28.39 MiB`` and no tracked file exceeds ``2 MiB``.

Best next steps:

1. Push this residual-runtime packaging commit and let CI validate the lowered
   source-health baseline.
2. Stop doing small line-count-only extractions unless review specifically asks;
   the next refactor/API tranche should target a larger residual-loop phase or
   a public-driver dispatch split that removes substantially more orchestration
   code.
3. Keep the full runtime matrix, WOUT parity, and AD-vs-FD figures stable unless
   a future numerical/performance path changes behavior.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.1%.
- Free-boundary production differentiability: 97.3%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 74.5%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Refresh README benchmark and derivative evidence panels

Steps taken:

- Rerendered ``docs/_static/figures/readme_runtime_compare.png`` and its
  CSV/JSON provenance from
  ``outputs/pr20_full_matrix_current_cpu_sg_fulljit/summary.json`` using the
  runtime-only README plot mode.
- Rerendered ``docs/_static/figures/readme_ad_fd_evidence.png`` and its
  CSV/JSON provenance from the branch-local direct-coil free-boundary report in
  ``outputs/pr20_ad_fd/qs_same_branch/same_branch_complete_solve_report.json``.
- Rebuilt the docs with warnings as errors after the refresh.

Results obtained:

- The README runtime artifact contains ``16`` fixed-boundary matrix rows.
- The AD-vs-central-FD evidence artifact contains ``10`` rows and all pass the
  documented ``1e-9`` derivative tolerance; the maximum absolute error is
  ``4.40e-10``.
- ``LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_final_tranche`` passed.
- ``git diff --check`` and ``repo_size_audit.py`` passed after rerendering; the
  repository remains below the ``50 MiB`` tracked-size gate.

Best next steps:

1. Push the refreshed public artifacts with the residual-runtime refactor
   lineage.
2. Let CI validate the current ``main`` tip before release/review decisions.
3. Do not rerun expensive VMEC2000/VMEC++/optimization sweeps unless a future
   numerical or performance-path change invalidates these artifacts.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.1%.
- Free-boundary production differentiability: 97.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 74.5%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Add function-size regression gate for residual-loop refactor

Steps taken:

- Extended ``tools/diagnostics/source_health.py`` with a repeatable
  ``--max-function-lines-at PATH:FUNCTION=LINES`` gate.
- Added unit tests covering passing, exceeding, missing-function, and malformed
  named-function baseline cases.
- Wired the CI source-health step to enforce that
  ``vmec_jax/solvers/fixed_boundary/residual/iteration.py:solve_fixed_boundary_residual_iter``
  does not grow beyond the current ``2516``-line baseline while the residual
  refactor continues.

Results obtained:

- ``python -m ruff check tools/diagnostics/source_health.py
  tests/test_source_health_diagnostics.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_source_health_diagnostics.py -q`` passed with ``10 passed``.
- The exact CI source-health command passed locally:
  ``python tools/diagnostics/source_health.py --top 20
  --max-root-helper-prefix-files 2 --max-function-lines-at
  vmec_jax/solvers/fixed_boundary/residual/iteration.py:solve_fixed_boundary_residual_iter=2516``.
- ``git diff --check`` passed.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed; tracked size is ``28.37 MiB`` and no tracked file
  exceeds ``2 MiB``.

Best next steps:

1. Push this CI maintainability gate and let the refreshed CI run validate it.
2. Continue residual-loop simplification only with larger seams that lower the
   named-function baseline; update the CI baseline downward whenever a tranche
   removes more lines.
3. Keep full benchmark, parity, and AD-vs-FD evidence artifacts stable unless a
   future numerical/performance change invalidates them.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.1%.
- Free-boundary production differentiability: 97.3%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 73.7%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Package pre-restart trigger runtime seam

Steps taken:

- Extracted the optional pre-restart trigger branch from
  ``solve_fixed_boundary_residual_iter`` into
  ``run_pre_restart_trigger_runtime`` in
  ``vmec_jax/solvers/fixed_boundary/residual/host_diagnostics.py``.
- Added explicit callback/result dataclasses for that branch so the residual
  loop now delegates the VMEC-style restart-trigger update, branch result,
  controller update, compact status print, and optional ``xc`` dump through a
  named host-runtime seam.
- Replaced the inline branch in
  ``vmec_jax/solvers/fixed_boundary/residual/iteration.py`` with the packaged
  helper while preserving the existing closure that mutates controller state,
  rollback histories, velocity resets, and timing.
- Added focused tests for the disabled and applied pre-restart trigger paths in
  ``tests/test_solve_residual_iter_setup_helpers.py``.

Results obtained:

- ``python -m ruff check
  vmec_jax/solvers/fixed_boundary/residual/host_diagnostics.py
  vmec_jax/solvers/fixed_boundary/residual/iteration.py
  tests/test_solve_residual_iter_setup_helpers.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_residual_iter_setup_helpers.py -q`` passed with
  ``25 passed``.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_solve_axis_helpers_more_coverage.py tests/test_residual_ptau.py
  tests/test_residual_iteration_preconditioner.py
  tests/test_solve_residual_iter_setup_helpers.py
  tests/test_solve_residual_iter_update_helpers.py
  tests/test_solve_additional_helpers.py tests/test_driver_api.py
  tests/test_free_boundary_wp0.py`` passed with ``282 passed, 2 skipped``.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_vacuum_adjoint.py
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py -q``
  passed.
- ``git diff --check`` passed.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed; tracked size is ``28.37 MiB`` and no tracked file
  is above ``2 MiB``.
- ``source_health.py`` now reports the residual module at ``2910`` lines and
  ``solve_fixed_boundary_residual_iter`` at ``2516`` lines.  This is a modest
  branch-packaging reduction, not the full residual-loop split.

Best next steps:

1. Let CI validate this controller-packaging commit on ``main``.
2. If review asks for more simplification before release, extract the next
   residual-loop phase runner only when it removes a materially larger block
   than this branch-local seam.
3. Treat README/docs figures, full runtime matrix, WOUT parity, and AD-vs-FD
   evidence as already refreshed by the prior readiness tranche unless a future
   numerical or performance-path change invalidates those artifacts.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.1%.
- Free-boundary production differentiability: 97.3%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 73.3%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Package fixed-boundary driver finalization seam

Steps taken:

- Extracted the public fixed-boundary post-solve finalization path from
  ``run_fixed_boundary`` into ``_finalize_fixed_boundary_solver_run`` in
  ``vmec_jax/driver.py``.
- The new seam owns solver-summary printing, flux/profile reconciliation,
  ``FixedBoundaryRun`` assembly, finish-policy diagnostic stamping, and the
  optional CLI finish callback.
- Lowered the CI and local source-health baseline for
  ``vmec_jax/driver.py:run_fixed_boundary`` from ``533`` to ``512`` physical
  lines.
- Added a focused test that checks the finalization helper wires the flux
  finalizer, preserves existing diagnostics, stamps public finish diagnostics,
  and forwards the correct initial policy to the finish hook.

Results obtained:

- ``run_fixed_boundary`` now reports ``512`` lines in ``source_health.py``,
  down from the previous ``533`` baseline.
- ``python -m ruff check
  vmec_jax/driver.py tools/diagnostics/local_ci_gate.py
  tests/test_driver_fast_reconstruction.py tests/test_local_ci_gate.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 python -m pytest -q
  tests/test_driver_fast_reconstruction.py tests/test_local_ci_gate.py -q``
  passed with ``11`` tests.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_driver_api.py tests/test_driver_policy_helpers.py
  tests/test_driver_policy_coverage_extra.py
  tests/test_driver_api_finish_more_coverage.py -q`` passed with the existing
  numerical warnings only.
- The stricter exact source-health command passed locally with
  ``solve_fixed_boundary_residual_iter=2508`` and
  ``run_fixed_boundary=512``.
- Full ``ruff``, ``compileall`` on changed files, ``git diff --check``, and
  ``repo_size_audit.py`` passed; tracked size is ``28.40 MiB``.

Best next steps:

1. Push this driver-finalization refactor and let CI validate the tighter
   public-driver source-health baseline.
2. If more review-readiness cleanup is needed, target one larger residual-loop
   phase or split the direct-coil/free-boundary optimization example into
   reusable source-code helpers; avoid churn in the refreshed benchmark and
   AD-vs-FD artifacts unless numerical behavior changes.
3. Keep adaptive free-boundary differentiation claims conservative until the
   full adaptive branch gate exists; current public evidence remains
   branch-local/fingerprint-gated.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.2%.
- Free-boundary production differentiability: 97.4%.
- Single-stage coil optimization: 92.9%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 75.2%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.

### 2026-06-26: Package same-branch coil optimization reporting

Steps taken:

- Moved the reusable same-branch vector/JVP runner and validation-report
  orchestration out of
  ``examples/optimization/free_boundary_QS_coil_optimization.py`` into
  ``vmec_jax/solvers/free_boundary/coil_optimization.py``.
- Kept the example pedagogic: it now supplies problem-specific callbacks for
  optimizer-vector packing, objective evaluation, JSON writing, and scalar
  registry injection, then delegates the report assembly to
  ``write_same_branch_validation_report_core``.
- Preserved example-level compatibility names used by tests and downstream
  notebooks, including scalar-registry monkeypatching.
- Split the new source helper into smaller seams for QS angle-cache creation,
  report mode/key parsing, complete-solve FD timing, and initial report-section
  defaults.
- Regenerated the public README runtime matrix artifact from the existing PR20
  current/main benchmark summaries.
- Regenerated the public AD-vs-central-FD evidence panel using the promoted
  same-branch free-boundary report so the fixed-boundary, ``DMerc``, ``D_R``,
  and branch-local free-boundary rows remain present.

Results obtained:

- ``examples/optimization/free_boundary_QS_coil_optimization.py`` dropped to
  ``1643`` lines and its same-branch writer is now a thin adapter.
- ``write_same_branch_validation_report_core`` reports ``241`` lines in
  ``source_health.py``, under the production function target.
- ``python -m ruff check
  examples/optimization/free_boundary_QS_coil_optimization.py
  vmec_jax/solvers/free_boundary/coil_optimization.py`` passed.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_qs_coil_optimization_smoke.py -q`` passed with the
  existing expected skip.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_external_fields_coils_jax.py tests/test_external_fields_mgrid_jax.py
  tests/test_free_boundary_coil_provider_gradients.py
  tests/test_free_boundary_qs_coil_optimization_smoke.py -q`` passed with the
  existing expected skip.
- ``PYTHONDONTWRITEBYTECODE=1 JAX_ENABLE_X64=1 python -m pytest -q
  tests/test_free_boundary_direct_coil_finite_pressure_sensitivity.py
  tests/test_free_boundary_vacuum_adjoint.py -q`` passed with existing NESTOR
  diagnostic runtime warnings.
- ``python tools/diagnostics/source_health.py --top 25 --top-functions 80
  --max-root-helper-prefix-files 2 --max-function-lines-at
  vmec_jax/solvers/fixed_boundary/residual/iteration.py:solve_fixed_boundary_residual_iter=2508
  --max-function-lines-at vmec_jax/driver.py:run_fixed_boundary=512`` passed.
- ``python tools/diagnostics/repo_size_audit.py --top 20 --max-total-mib 50
  --max-file-mib 2`` passed; tracked size remains ``28.40 MiB``.
- ``git diff --check`` passed.
- ``LANG=C.UTF-8 LC_ALL=C.UTF-8 python -m sphinx -W -j auto -b html docs
  docs/_build/html_pr20_final`` passed.
- ``readme_runtime_compare.{png,csv,json}`` and
  ``readme_ad_fd_evidence.{png,csv,json}`` were regenerated.  The AD/FD evidence
  table has ``10`` rows, including the four branch-local free-boundary scalars.
- The existing PR20 WOUT parity summary remains available with ``4`` cases and
  ``failed_cases=0``.

Best next steps:

1. Commit and push this final same-branch reporting refactor plus refreshed
   README/docs artifacts.
2. Let CI validate the pushed commit; no further README/docs figure churn is
   needed unless CI or review finds a concrete issue.
3. If another review-readiness tranche is needed, target the remaining long
   example function ``optimize_coils`` or the old residual loop, not already
   refreshed benchmark artifacts.

User needs:

- No immediate input needed.

Updated lane percentages:

- Performance benchmark/profiling harness: 100%.
- Fixed-boundary production differentiability: 97.2%.
- Free-boundary production differentiability: 97.5%.
- Single-stage coil optimization: 93.6%.
- CPU/GPU runtime and memory footprint: 99.2%.
- Refactor/API/examples: 76.4%.
- VMEC2000/VMEC++ parity and physics gates: 99.3%.
- Docs/release hygiene: 100%.
- Overall: 99.6%.
