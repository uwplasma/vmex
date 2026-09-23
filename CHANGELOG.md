# Changelog

Release notes are mirrored from the GitHub releases, which carry each release
in full. A number appears here only where a committed artifact backs it, and
`benchmarks/INDEX.md` lists every benchmark artifact with its generator, the
revision it was measured at, and the pages that cite it.

## Unreleased

### Fixed

- **Free-boundary values and gradients referred to different points.** VMEC's
  `ftol` does not place a free-boundary solve on the coupled plasma--vacuum
  root the adjoint differentiates; along weakly damped directions the
  converged state, and the objective read there, can sit far from it. Every
  certifiable solve is now Newton-anchored on that root (`refine_tol`,
  damped on Deuflhard's natural monotonicity test); one that cannot be
  anchored is the new status 3, never a silent pass.
- **Stalled free-boundary restarts ran to `max_iterations`.** A restart from
  the configuration's reference now gets the iterations the cold reference
  needed before the deterministic cold retry of #416.
- `VmecExtender.from_wout` raises on a wout that names an MGRID but records
  no coil currents, instead of extending with a zero coil field.
- The eager derivative accuracy check no longer warns "up to inf" far from
  the surface (virtual-casing-jax 0.0.8, now the `freeb` floor).

### Added

- `from_wout`/`from_state` accept `project_current` (off by default, #381).
- **The exterior field is accurate next to the plasma surface.** Eager
  `VmecExtender` calls switch each point the direct quadrature cannot resolve
  to a target-graded rule (`near_surface="auto"`, default): 1e-12 in B down to
  0.01 minor radii on the 2.5 % beta QA deck, where the default grid was off
  by order one. `with_graded_quadrature()` uses it everywhere, under `jit` too.

### Changed

- Direct-path derivatives use the closed-form layer kernels (same values to
  1e-12): first `B`..`gradgradgradB` calls 6.0 s -> 2.9 s, warm
  `gradgradgradB` 4-6x faster (`benchmarks/extender_ab_20260923.json`).

### Removed

- `VmecExtender.with_near_surface_continuation` and `near_surface_plan`
  (1.6-2.4 % error floor, 18.5 GB to prepare); use `with_graded_quadrature()`.

## 0.11.0 - 2026-09-21

See the GitHub release for this version in full.

### Changed

- **The interior field is built from the native VMEC form** (#403).
  `B^theta = (chi' - lambda_zeta)/sqrt(g)` and `B^zeta = (phi' + lambda_theta)/sqrt(g)`
  with `sqrt(g)` from the same `R`, `Z` series as the position, and a C2
  radial interpolant. On the breathing-circle oracle at ns = 41 the second
  derivative of B improves from 2.9e-2 to 2.2e-4 relative error and
  `|div B|/|grad B|` is at round-off (8.5e-18). Spectra without `lmns`,
  `phipf` and `chipf` keep the previous fitted path.
- **Minimum versions** (#410): `booz_xform_jax>=0.4.0` (its `Booz_xform`
  class rebuilds cached grids when the resolution changes and reports a
  correct `__version__`; the JAX kernel is unchanged from 0.3.0),
  `solvax>=0.21.0`, and `gkx>=1.8.0` for the turbulence extra.
- **Force-balance polishing is compared on one mesh** (#412). The example and
  README now state that the gain is concentrated near the axis (RMS force
  2.9e3 -> 61 N/m^3 for rho < 0.2) with a small volume-average change
  (2.27e-3 -> 1.91e-3), and that the summary plot's force panel cannot
  resolve it.

### Fixed

- **`surface_field_data_from_state` raised `TypeError`** for every caller
  after #403 added three keys to the live-state spectra (#409).
- **Free-boundary recovery depended on trial history** (#416). A shared
  cold-rebuild budget meant that, after unrelated rejected trials, a
  recoverable point returned the stalled restart instead of its converged
  root. Each stalled restart now gets exactly one deterministic cold retry.
- **Radial lifts accepted under-determined spline fits** (#414), inventing
  curvature in unsampled spans; they now raise.
- **Warm workflow profiles timed only the first stage** (#420); schema 2
  repeats every stage.

## 0.10.0 - 2026-09-20

See the GitHub release for this version in full.


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

Optimization gradients compile on a machine that has a GPU (the host-callback
pin now applies only within one platform; jax-ml/jax#40722), the coil examples
install from PyPI (`essos>=0.17`, new `all` extra), `max_fsq_ratio` defaults to
`1e2`, and the exterior field sizes its source grid from the boundary (4.31e-07
one minor radius out on the QA wout, against 2.46e-02 before). Full notes in
the GitHub release.

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
