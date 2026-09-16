# Changelog

Release notes are mirrored from the GitHub releases, which carry each release
in full. A number appears here only where a committed artifact backs it, and
`benchmarks/INDEX.md` lists every benchmark artifact with its generator, the
revision it was measured at, and the pages that cite it.

## 0.9.1 - 2026-09-16

Optimization gradients now compile on a machine that has a GPU, the coil
examples install from PyPI, and the exterior field sizes its source grid from
the boundary.

### Fixed

- **Optimization gradients could not be compiled on any machine with a GPU.**
  The implicit path is deliberately stood down to the CPU on an accelerator
  backend and the host callback was pinned there, but JAX requires a pinned
  device to be in the enclosing computation's device assignment and a jit
  compiled for `cuda:0` does not contain `cpu:0`. Every jitted
  `jax_value_and_grad` died in JAX's lowering with
  `ValueError: tuple.index(x): x not in tuple` and no VMEX frame
  (jax-ml/jax#40722). The pin now applies only within one platform. On two RTX
  A4000s all three ways of asking agree, peak device memory 0.16 GiB, warm
  value-and-gradient 1.3–1.6 s against 2.15 s on that machine's CPU.
- The "When the GPU pays off" snippet referred to an undefined `runtime` (#157).
  That page now also records a re-measurement: the GPU lost on both shipped
  decks, including the one `recommended_device` answers `gpu` for.

### Changed

- `max_fsq_ratio` defaults to `1e2`, not `1e6`. The implicit adjoint assumes
  `F = 0`, so differentiating a trial whose residual is 1e-6 against a 1e-12
  deck carries an O(norm(F)) error and lets a line search walk the design
  somewhere the solver cannot resolve; that is what stalled the finite-beta
  single stage (#361).
- **The coil examples install from PyPI.** ESSOS 0.17 carries uwplasma/ESSOS#58,
  so the `coils` extra pins `essos>=0.17` and the git-install instruction leaves
  nine examples, five documentation pages and the examples README. A new `all`
  extra installs everything the examples use.
- **The exterior field sizes its source grid from the boundary.**
  `VmecExtender.from_wout` hard-coded `nphi=ntheta=32`, and the level schedule
  read a per-field-period sampling as a whole-torus count; on the shipped QA
  wout (`R0/a` = 15.9) the achieved error one minor radius out was 2.46e-02
  against a requested 1e-6. Sizing from the geometry gives 4.31e-07 at 0.116 s
  per call against 0.226 s. A tokamak-like aspect ratio stays on the historical
  floor; explicit `nphi`, `ntheta` or `levels` are honored unchanged.
- `take_free_boundary_gradients.py` certifies against the second adjoint solver
  rather than a central difference, which on a free boundary has no usable step;
  the two adjoints agree to 1.6e-04.

### Removed

- Four files referenced by nothing in the tree: the QI sheet-current mgrid
  builder under tools, a repo census and a stale profile record under
  benchmarks, and the pre-commit config.

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
