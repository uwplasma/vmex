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

- Performance benchmark and profiling harness: 95%.
  The PR #20 single-grid matrix, current-vs-main comparator, VMEC2000 rows, and
  VMEC++ optional rows are regenerated with CSV/JSON provenance. Remaining work
  is deeper kernel-level decomposition and long-term dashboard automation.
- Fixed-boundary production differentiability: 90%.
  AD-vs-central-FD evidence now passes `1e-9` for fixed-boundary geometry,
  profiles, QS/QI diagnostics, `DMerc`, and `D_R`. Remaining work is
  operator-level implicit/JVP/VJP productionization.
- Free-boundary production differentiability: 86%.
  Direct coil fields, JAX mgrid interpolation, accepted-branch replay, and
  fingerprint-gated branch-local gates pass `1e-9` evidence for selected
  physical scalars. Arbitrary adaptive branch differentiation remains
  unclaimed.
- Single-stage coil optimization: 86%.
  Examples and branch-local derivative proposal paths exist; complete solves
  still need to remain the acceptance authority until the full adaptive seam is
  validated.
- CPU/GPU runtime and memory footprint: 74%.
  The single-grid matrix shows no PR regression against `origin/main`, and warm
  CPU `vmec_jax` beats VMEC2000 on 7 of 16 rows. Memory remains materially
  higher than VMEC2000, and 3D preconditioner seed construction is the next
  CPU hotspot.
- Refactor/API/examples: 40%.
  Public examples are better, but core source files and tests are still too
  large and too entangled.
- VMEC2000/VMEC++ parity and physics gates: 96%.
  The PR #20 four-row executable WOUT parity gate passed, and the single-grid
  runtime matrix records VMEC++ availability per row. More bounded
  free-boundary external parity remains future work.
- Docs/release hygiene: 95%.
  README is concise, runtime/memory detail lives in docs, and benchmark plus
  AD-FD provenance are refreshed. Remaining work is Sphinx gating and pruning
  historical performance prose after review.
- Overall completion: 82%.
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

Status: 35%.

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

Status: 15%.

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

Status: 10%.

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

1. Finish M0 by committing this plan and the README benchmark removal.
2. Create the M1 VMEC2000/VMEC++ algorithm-to-kernel map using local source
   inspection and one profiled fixed-boundary row.
3. Add M2 profiling decomposition hooks around cold/warm `vmec_jax` runs without
   changing default user output.
4. Run the historical single-grid matrix once with the new decomposition and
   classify the largest runtime/memory gaps.
5. Start M7 by splitting `iteration.py` at stable seams while keeping
   compatibility wrappers and tests.
6. Promote M4 derivative evidence tolerance to `1e-9` where current artifacts
   show it is numerically stable; document exceptions.

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
