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

- Performance benchmark and profiling harness: 72%.
  Full README benchmark data exists, but the next gate is deeper decomposition
  into import/startup, XLA trace, XLA compile, steady solve, WOUT write, and
  optimizer callback costs.
- Fixed-boundary production differentiability: 82%.
  AD-vs-central-FD evidence exists for many scalars. Remaining work is
  operator-level implicit/JVP/VJP productionization and tighter `1e-9` evidence
  where finite-difference noise allows it.
- Free-boundary production differentiability: 78%.
  Direct coil fields, JAX mgrid interpolation, accepted-branch replay, and
  fingerprint-gated branch-local gates exist. Arbitrary adaptive branch
  differentiation remains unclaimed.
- Single-stage coil optimization: 86%.
  Examples and branch-local derivative proposal paths exist; complete solves
  still need to remain the acceptance authority until the full adaptive seam is
  validated.
- CPU/GPU runtime and memory footprint: 64%.
  Warm replay improved, but cold exact tape/forward-force cost and GPU-native
  derivative paths still dominate.
- Refactor/API/examples: 40%.
  Public examples are better, but core source files and tests are still too
  large and too entangled.
- VMEC2000/VMEC++ parity and physics gates: 90%.
  Existing gates are strong for selected cases. More bounded fixed/free-boundary
  rows and performance-regression parity gates are still needed.
- Docs/release hygiene: 88%.
  Docs are broad but still mirror historical work too much. They need a clearer
  "what is differentiable now" table and performance caveats separated from the
  README.
- Overall completion: 69%.
  PR #20 can be reviewed as a major milestone, but this plan defines the next
  phase before claiming final research-grade status.

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

Status: 28%.

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
