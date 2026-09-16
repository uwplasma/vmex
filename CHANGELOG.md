# Changelog

Release notes are mirrored from the GitHub releases, which carry each release
in full. A number appears here only where a committed artifact backs it, and
`benchmarks/INDEX.md` lists every benchmark artifact with its generator, the
revision it was measured at, and the pages that cite it.

## 0.9.1 - 2026-09-16

Optimization gradients now compile on a machine that has a GPU, the coil
examples install from PyPI, and the exterior field sizes its source grid from
the boundary instead of a constant.

### Fixed

- **Optimization gradients could not be compiled on any machine with a GPU.**
  The implicit path is deliberately stood down to the CPU on an accelerator
  backend, and the host callback was pinned there — but JAX requires a pinned
  device to be in the enclosing computation's device assignment, and a jit
  compiled for `cuda:0` does not contain `cpu:0`. Every jitted
  `jax_value_and_grad` died in JAX's lowering with
  `ValueError: tuple.index(x): x not in tuple`, with no VMEX frame
  (jax-ml/jax#40722). The pin now applies only within one platform. On two RTX
  A4000s all three ways of asking now agree, peak device memory 0.16 GiB, warm
  value-and-gradient 1.3–1.6 s against 2.15 s on the same machine's CPU.
- The "When the GPU pays off" snippet referred to an undefined `runtime` (#157).

### Changed

- `max_fsq_ratio` defaults to `1e2`, not `1e6`. The implicit adjoint assumes
  `F = 0`, so differentiating a trial whose residual is 1e-6 against a 1e-12
  deck carries an O(norm(F)) error and lets a line search walk the design
  somewhere the solver cannot resolve; that is what stalled the finite-beta
  single stage (#361).
- **The coil examples install from PyPI.** ESSOS 0.17 carries uwplasma/ESSOS#58,
  so the `coils` extra pins `essos>=0.17` and the git-install instruction is
  gone from nine examples, five documentation pages and the examples README. A
  new `all` extra installs everything the examples use.
- **The exterior field sizes its source grid from the boundary.**
  `VmecExtender.from_wout` hard-coded `nphi=ntheta=32`, and the level schedule
  read a per-field-period sampling as a whole-torus count; on the shipped QA
  wout (`R0/a` = 15.9) the achieved error one minor radius out was 2.46e-02
  against a requested 1e-6. Sizing from the geometry gives 4.31e-07 at 0.116 s
  per call against 0.226 s — more accurate and no slower. A tokamak-like aspect
  ratio stays on the historical floor; explicit `nphi`, `ntheta` or `levels`
  are honored unchanged.
- `take_free_boundary_gradients.py` certifies against the second adjoint solver
  instead of a central difference, which on a free boundary has no usable step.
  The coupled GCROT and edge Schur adjoints agree to 1.6e-04.

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

### Added

- `vmex` reads DESC inputs and equilibria — `.h5` files and DESC input decks
  through the CLI and `VmecInput` — with round trips against DESC 0.17 (#300).
- Exterior-field accuracy is observable: `B_plasma_xyz` returns the accuracy the
  adaptive schedule achieved, a schedule failing its own self-test warns instead
  of silently returning its last level, and known-answer oracles cover the vacuum
  identity outside, the interior identity inside and Malhotra on-surface parity
  (#312).
- Every `OptimizationRecord` and benchmark row carries counters: descent
  iterations, refinement calls, steps and matvecs, Jacobian calls and columns,
  certifier iterations, adjoint calls and matvecs, and host wall time in each
  (#310, `benchmarks/optimization_counters_20260913.json`).
- The README shows a `--plot` summary for a QA and a QI stellarator (#344).

### Changed

- **The supported floor is Python 3.11, SciPy 1.16 and JAX 0.9.2.** SciPy added
  `least_squares(callback=)` in 1.16 and 1.16 needs Python 3.11, so a 3.10 install
  resolved SciPy 1.15.3 and every optimization example died at its
  `least_squares(` call. Python 3.10 also resolved JAX 0.6.2, which
  differentiates the QA residual wrongly — RBC(1,0) and ZBS(1,0) off by factors
  of 6 and 9 against finite differences, ending at QS total 5.29e-3 instead of
  3.72e-4. `import vmex` now checks the four versions before importing JAX and
  names the fix; jax 0.11 needs Python 3.12, so 3.11 caps at JAX 0.10.2 (#347).
- **The persistent compilation cache is off under jaxlib < 0.10 everywhere, not
  only on macOS.** jaxlib 0.9.2 segfaults loading a large cached executable on
  Linux too (reproduced without VMEX: cold exits 0, warm exits 139; 0.6.2,
  0.10.x and 0.11.1 clean). JAX 0.10+ restores it automatically;
  `VMEX_COMPILATION_CACHE=1` still forces it on (#347).
- **The refinement every optimizer trial pays is a Newton finish.** It runs
  through one raw block factorization instead of a stalling Krylov solve: on the
  QI example, refinement 333.0 s and 63,316 Krylov iterations -> 66.7 s, 153
  GMRES iterations and 20 factorizations, with the same 17,251 descent
  iterations, 0 failed trials and the final cost within 7.2e-7. The example runs
  565.1 -> 359.1 s wall and its least-squares phase 366.9 -> 127.5 s; the
  single-stage example 30.8 -> 22.3 s per trial. Decks where the Newton phase
  stalls replay the previous refinement and keep its anchor (#320, #338).
- **The implicit adjoint is solved exactly through the raw block
  factorization**, not 1,100 to 17,000 Krylov iterations: QI eager `jax.grad`
  203.9 -> 55.5 s, `from_loss` first value and gradient 190.2 -> 89.0 s. At a
  non-root anchor the scalar gradient moves by the formulation difference, up to
  1.1e-3 on QI (#330). The reverse Jacobian lane factors once per point: seed QA
  238.9 s, where the per-chunk path had not finished after 4,111 s (#335, #351).
- **The compiled lanes stopped recompiling.** A concrete
  `problem.jax_value_and_grad` returns the host lane's pair and shares its solve
  memo, warm-start stash and counters (JAX value and gradient 39 -> 1.5 s on QA,
  55 -> 1.6 s on QI; benchmark wall 95.3 -> 51.0 s and 117.5 -> 61.8 s). Solver
  executables are reused across trial seeds, the axis re-guess is traced, and
  the default-device context is out of the block lane's jit key (#319, #321,
  #325, #331).
- **The QI example converges.** Seeded from `input.QI_nfp2_initial` rather than a
  circular torus, with one `max_mode = 2` stage and a mirror-ratio penalty from
  1 % below the limit: 612 s, 0 failed trials (5 before), constructed QI 3.0e-3
  (0.789 before), limits met. It previously ran 1,456 s and then raised (#333).
- **The single-stage examples meet their stated targets or fail loudly.** The
  fixed-boundary example seeds at mean iota >= 0.3 and meets every target on a
  full run (min |iota| 0.4277, aspect 3.979, B.n RMS 0.80 %, an independent
  ns = 101 check converged). The free-boundary example floors the quantity it
  checks and reaches min |iota| 0.4271 in 614 s where it exited 1 at 0.4136 --
  at the cost of quasi-axisymmetry (7.3e-4 -> 4.5e-2) (#311, #352).
- **The CLI keeps the equilibrium when the final grid exhausts NITER.** It wrote
  no WOUT unless `LFULL3D1OUT` was set, discarding the run; `vmec.f` and
  `fileout.f` reach `wrout` on `more_iter_flag` either way, and `LFULL3D1OUT`
  governs threed1 fullness. The exit code stays the distinct `ier_flag = 2` and
  `wout.ier_flag` records 2, so the non-convergence stays visible in the file.
- The `--plot` summary resolves the effective ripple instead of aliasing it: the
  diagnostic NEO configuration was too coarse for the `mboz = 16` spectrum it is
  given, producing values 7-20x from converged NEO with dips to 1e-3 and 4e-6 of
  the reference. It is now within 3.4 % of NEO's converged settings (#354).
- The summary plot's force panel no longer reads 1 on converged vacuum
  equilibria: it plots `|J x B - grad p|` over the volume-averaged
  `|grad(B^2/2mu0)|` on `0.1 <= s <= 0.99` (DESC's normalization) instead of
  WOUT's `equif`, which is bounded by 1 and equals 1 without pressure or current.
  `equif` is unchanged and matches all nine VMEC2000 goldens to 5e-13 (#337).
- The persistent compilation cache keeps entries used within the last 24 hours
  when it trims to its 1,024-entry bound, up to four times it: the QI example's
  warm run no longer recompiles the 318 programs the trim evicted (#343).
- Polish reports live progress: phase announcements, a row per Gauss-Newton
  iteration from inside the jitted solve, and a certificate verdict (#243, #244).
- `POLISH = AUTO` prices the solve and returns the equilibrium unpolished past
  `POLISH_BUDGET` (`--polish-budget`, default 3600 s); `POLISH = ON` never
  declines. New directives `POLISH_TOL`, `POLISH_DEGREE`, `POLISH_SPANS`,
  `POLISH_MAX_ITER` and `POLISH_FAIL` carry CLI flags and `solve_file` keys (#243).
- The gyrokinetic flux-tube `epsilon` is the field-line `|B|` modulation depth and
  `R0` the effective major radius, so GKX's minor radius is physical (#271).
- Optimization seed refinement is a deferred per-configuration executable on the
  scalar, free-boundary implicit and mirror Newton-Krylov paths (#240, #241).

### Fixed

- The finite-beta single-stage example ran 19:37 and then raised: its final
  ns = 31/51/101 check burned all 20,000 iterations per rung. `max_fsq_ratio`'s
  1e6 default differentiated trials at 1e-6 against the deck's 1e-12, letting
  the line search walk somewhere unsolvable; requiring 1e2 keeps every trial at
  a root and the ladder converges (9.9e-15 at ns = 101) in 5:03 (#361).
- An AUTO ladder crossing the CPU/GPU work threshold raised `Received
  incompatible devices`: each rung placed its state on its own device but
  carried the previous rung's residual scalars through. Introduced by #321 and
  caught by the first GPU run since 2026-07-31 (#360).
- A `jacrev` fallback vmapped the GCROT adjoint over all 6,722 residual rows,
  asking for 47 GiB buffers and a 251 GB peak; it now pulls rows back in
  tangent-lane batches, with no intermediate above 64 MiB (from 9,138) (#328).
- A Jacobian retry rebinding the baselines dropped the stage's `use_fft` (#324).
- The single-stage coil pre-fit ran without an iteration cap; it stops at 200 (#342).
- The LASYM QH example lost its second stage's budget (#350).
- The CTH-like free-boundary examples raised a bare `MgridNotFoundError`; they
  now name the fetch that installs the file (#353).
- `test_use_fft_reaches_every_free_boundary_lane` spied on `_make_body` with a
  re-declared keyword list, raising `TypeError` once the vacuum steady lane was
  traced — broken since 2026-08-01 and run by no workflow. It now takes
  `**kwargs` and generated coils (#357).
- Polish sizes its force sweep from the deck's mode table and checkpoints the
  per-point kernel: W7-X certifies at 3.0 GiB, not 34 (`polish_memory_w7x.json`).
- Force-error reporting separates native accuracy from WOUT reconstruction (#280, #282).

### Removed

- The `vmec_jax` compatibility package is gone: the project has been importable
  as `vmex` since 0.7.0 and the shim is no longer installed.
- `vmex.core.freeboundary_diff` is gone; import the prescribed-interface
  virtual-casing API from `vmex.core.virtual_casing` (or the `vmex` top level),
  which re-exports every one of its names unchanged. `freeboundary_linear` is
  gone too: the coupled adjoint ships in `vmex.core.freeboundary_implicit`,
  whose edge Schur complement superseded that never-wired prototype.
- The deprecated Boozer aliases `boozer_bmnc_state` and `boozer_bmnc_high_order`
  are gone, with `vmex.boozer_bmnc_high_order`; call `boozer_spectrum_state` /
  `boozer_spectrum_high_order`, whose contracts are identical. The unused
  `wout.wout_field_names` and `virtual_casing.value_and_grad_bnormal` are gone;
  use `dataclasses.fields(WoutData)` and `jax.value_and_grad` on
  `PlasmaVacuumInterface.bnormal_objective`.

## 0.8.1 - 2026-09-02

The cold-start performance release: cold CLI, Python and optimization runs
faster than every previous release on every deck measured, from 773 XLA
programs at QA resolution to 343 (#227-#234). Full notes in the GitHub
release.
