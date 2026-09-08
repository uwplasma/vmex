# Changelog

Release notes are mirrored from the GitHub releases, which carry each release
in full. A number appears here only where a committed artifact backs it, and
`benchmarks/INDEX.md` lists every benchmark artifact with its generator, the
revision it was measured at, and the pages that cite it.

## Unreleased

### Added

- Polish reports live progress: phase announcements, one row per Gauss-Newton
  iteration from inside the jitted solve, and a certificate verdict (#243, #244).
- `POLISH = AUTO` prices the solve and returns the equilibrium unpolished past
  `POLISH_BUDGET` (`--polish-budget`, default 3600 s); `POLISH = ON` never declines.
- Polish directives `POLISH_TOL`, `POLISH_DEGREE`, `POLISH_SPANS`,
  `POLISH_MAX_ITER` and `POLISH_FAIL`, with CLI flags and `solve_file` keywords (#243).

### Changed

- The gyrokinetic flux-tube `epsilon` is the field-line `|B|` modulation depth
  and `R0` the effective major radius, so GKX's minor radius is physical (#271).
- Optimization seed refinement is a deferred per-configuration executable, on the
  scalar, free-boundary implicit and mirror Newton-Krylov paths (#240, #241).

### Fixed

- Polish sizes its force sweep from the deck's own mode table and checkpoints the
  per-point kernel: W7-X standard certifies at 3.0 GiB, not 34 (`benchmarks/polish_memory_w7x.json`).
- Force-error reporting separates native accuracy from WOUT reconstruction; the
  corrected pair is `benchmarks/polish_force_error_2026-09-03.json` (#280, #282).

### Removed

- The `vmec_jax` compatibility package is gone; the project has been importable
  as `vmex` since 0.7.0 and the rename shim is no longer installed.
- `vmex.core.freeboundary_diff` is gone; import the prescribed-interface
  virtual-casing API from `vmex.core.virtual_casing` (or the `vmex` top level),
  which has re-exported every one of its names unchanged.
- The deprecated Boozer-transform aliases `boozer_bmnc_state` and
  `boozer_bmnc_high_order` are gone, together with `vmex.boozer_bmnc_high_order`;
  call `boozer_spectrum_state` / `boozer_spectrum_high_order`, whose signatures
  and return contracts are identical.
- `vmex.core.freeboundary_linear` is gone; the coupled free-boundary adjoint
  ships in `vmex.core.freeboundary_implicit`, whose edge Schur complement
  superseded this never-wired bordered-operator prototype.
- The unused helpers `vmex.core.wout.wout_field_names` and
  `vmex.core.virtual_casing.value_and_grad_bnormal` are gone; use
  `dataclasses.fields(WoutData)` and `jax.value_and_grad` on
  `PlasmaVacuumInterface.bnormal_objective` respectively.

## 0.8.1 - 2026-09-02

The cold-start performance release. Cold CLI, python, and optimization runs
are faster than every previous VMEX release, including v0.3.0, on every
deck measured, on x86 and arm64: 36-core x86 QA_lowres 43.1 s (v0.3.0) ->
32.4 s, solovev 12.1 -> 7.0 s; Apple M4 QA 22.3 s (v0.8.0) -> 12.4 s, li383
-> 4.0 s, solovev -> 3.1 s. At QA resolution the cold start compiles 343 XLA
programs (v0.8.0: 773; v0.3.0: 523).

- Eager setup, WOUT-export, stage-interpolation, and printout passes became
  module-level jitted lanes (#227, #230).
- The iteration body traces the funct3d chain once (halving every lane's
  compile), and the ns4 preconditioner refresh runs only on its VMEC2000
  cadence (#229).
- Print cadence, initial DELT, and ftol no longer key lane recompilation
  (#231); non-finite iterations are detected from the residual scalars
  with full classification on trip (#232).
- 3-D polishing no longer stalls in XLA constant folding: the Ruiz/probe
  jits stopped baking linearization residuals as constants (#234).
- CI pins compile budgets (lane HLO size, cold-solve program count) and
  disables the persistent compilation cache on ephemeral runners, whose
  eviction lock had been timing every lane out (#228, #233).
- Numerical statement: per-iteration physics unchanged; graph
  restructuring shifts XLA fusion, so trajectories can differ from v0.8.0 at
  1 ULP per iteration with identical iteration counts and converged
  geometry agreeing at 1e-12.

## 0.8.0 - 2026-08-30

- Certified force-balance polishing: `--polish`, `!@VMEX POLISH = AUTO`, or
  `solve_file(..., polish="auto")` lift a converged fixed-boundary state to
  axis-regular cubic B-splines and drive both physical force channels to
  zero on an overdetermined collocation grid with matrix-free SOLVAX
  Gauss-Newton steps; acceptance is an independent certificate (volume L2
  force error below 1e-2, radial-refinement stability within 1e-3, positive
  signed Jacobian) with tangent/adjoint derivatives through the polished
  root.
- Execution directives (`!@VMEX KEY = VALUE`, JSON `_vmex`, keywords, CLI)
  separate how to run from what to solve; VMEC2000 reads the same files.
- Implicit lane recompilation per optimization trial removed (steady
  campaign step 16.6 -> 10.8 s, warm value+gradient repeat 28.8 -> 3.0 s,
  bit-identical); polished solves reuse compiled programs (warm 52.9 ->
  22.8 s).
- `benchmarks/profile_workflows.py`: seventeen principal workflows in five
  timing regimes with compile counts and committed M4 baselines.
- CI attempt wall clock roughly halved by resharding; `tools/preflight.py`
  runs the static gates, guard tests, and diff-affected tests locally.
