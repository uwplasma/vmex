# Changelog

Release notes are mirrored from the GitHub releases, which carry each release
in full. A number appears here only where a committed artifact backs it, and
`benchmarks/INDEX.md` lists every benchmark artifact with its generator, the
revision it was measured at, and the pages that cite it.

## Unreleased

### Fixed

- **The interior field evaluated B at the wrong place.** The geometry table it
  is built from, `_state_field_spectra`, multiplied every odd-`m` coefficient
  of `rmnc` and `zmns` by an extra `sqrt(s)`. `wout_from_state` builds the same
  table from the same `R_cos_p` with `mode_scale` alone, and
  `m1_constrained_to_physical` has already returned physical amplitudes, so
  the factor was spurious. It is 1 at the boundary, which is why
  `surface_field_data_from_state` — which reads only `s_index = -1` — and
  every virtual-casing and exterior path through it were always correct, and 0
  on axis; in between it pulled each surface in towards the axis. On
  `li383_low_res` that put R 1.2 cm and Z 5.9 cm off at mid-radius, `|B|`
  1.0e-2 off relative to the wout's own `bmnc`, and left 28 of 126 adjacent
  surface pairs crossing each other by up to 6.5% of their spacing. The
  interior geometry now reproduces the wout table to 6.7e-16 m, `|B|` agrees
  with `bmnc` to 3.1e-4 — the half-mesh interpolation difference — and all 126
  pairs nest. `_state_field_spectra` had no test of its own: the only test
  that referenced it replaced it with a stub, so nothing compared it against
  the wout it is meant to mirror. One now does, on a one-iteration state,
  since the identity holds whether or not the solve converged.
- **A point well outside the plasma raised instead of returning NaN.** The
  stalled-inversion check added in #379 decided a point was interior from the
  `s` its own unconverged iteration stopped at. That carries no information:
  on the decks measured, a large fraction of points several minor radii out
  still finish at `s <= 1`. Querying the DSHAPE interior field three and eight
  times the minor radius past the boundary raised `VmecNumericalError` at 12
  and 14 of 32 points, against a documented quiet NaN; the existing test only
  probed 1.5 times out, which is the one regime that behaved. A stalled point
  is now classified against the boundary itself, sampled at the query's
  toroidal angle and shrunk 1% towards the axis so its finite resolution can
  only ever suppress a complaint, never invent one.
- **The inversion did not converge from its first guess.** It was seeded by
  equating the VMEC poloidal angle with the polar angle about the magnetic
  axis; on a shaped boundary the two differ enough that the undamped step
  leaves the basin of the root, pins against the `rho` clip where the
  determinant guard turns the angle step into noise, and wanders until the
  iteration cap stops it. It now starts from whichever of three candidates
  actually lands nearest the query point: the geometric guess, the best node
  of a coarse sweep of the forward map at the query's toroidal angle, and the
  map linearized about the axis. Seeded callers skip the sweep.
- **A query exactly on the magnetic axis returned NaN when its flux
  coordinates were given.** The chart collapses there, so such a point is
  displaced to a representative just off axis, but the iteration was still
  started from the caller's `s = 0` — the one place the Jacobian vanishes and
  the first step is set by the determinant guard instead of the geometry. It
  now starts from the representative's own coordinates.

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
  That page now carries a full re-measurement of the CPU-vs-GPU sweep
  (`benchmarks/gpu_a4000_2026-09-16.json`): on two RTX A4000s the GPU wins no
  cell, warm gain 0.17x to 0.83x across every shipped deck and every point of
  the synthetic size scan. The thresholds do not transfer between machines.

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
