# Changelog

Release notes are mirrored from the GitHub releases, which carry each release
in full. A number appears here only where a committed artifact backs it, and
`benchmarks/INDEX.md` lists every benchmark artifact with its generator, the
revision it was measured at, and the pages that cite it.

## 0.11.8 - 2026-10-06

- **`vmex --trace` converges without a step to choose.** The default
  integrator is per-particle error-controlled Dopri8 at tolerance `3e-7`
  (ESSOS 0.20 adaptive stepping); `--trace-tolerance` sets it and
  `--trace-timestep` becomes the first trial step. On twenty-one equilibria
  the worst energy error is `1.7e-4`, against up to `0.27` for the earlier
  fixed RK4 step, at about 2.6 times its cost (#554).
- The default Boozer mode cut is `2e-4`, safe on twenty equilibria, and every
  run reports its largest energy error and how to tighten it above `1e-3`
  (#553). The trace guide gives the mode-cut, timestep, `K` and integrator
  studies on the corrected tracer (#552).
- The optimization Jacobian is cached persistently (#549). New examples:
  low-bootstrap QA, a QI drift study and a DKX-bootstrap QI (#550).
- Floor `essos>=0.20.0`.

## 0.11.7 - 2026-10-05

- **Alpha tracing physics corrections; results traced with earlier releases
  change.** The Boozer toroidal flux is passed with the sign the
  guiding-centre equations expect (#517). With ESSOS >= 0.19.5: the CODATA
  alpha mass (0.76 % lighter), the pitch-angle scattering rate without its
  spurious factor of two, and finite collision coefficients at small speed.
- Birth sampling uses `|G + iota I| / B^2` with a fixed rejection bound, so
  equilibria with negative toroidal flux (CTH-like, HSX, LHD) trace (#529).
- Non-stellarator-symmetric equilibria trace, keeping their sine spectra
  (#524, #539). Optional fixed-step `--trace-method` integrators (#520).
  Stopped alphas are compacted out of the batch by default (#521).
- `vmex --turbulence`: GKX linear scan and nonlinear ITG run (#547).
- VMEC input decks reconstructed from WOUT files (#512).
- Optimizers refuse to start from, or cache, an uncertified equilibrium
  (#498, #541). The optimization examples print whether their targets are
  met (#500, #502, #503, #506, #507). `QA_optimization_alpha_losses.py`
  (#492).
- Floors: `essos>=0.19.5`, `booz_xform_jax>=0.4.3`, `neo-jax>=1.0.5`
  (#523, #538).

## 0.11.6 - 2026-09-28

- `vmex --trace` defaults to a Boozer mode cut of 1e-4, the cut the
  six-geometry study supports, and `--trace-mode-cut` sets it. The header
  states the cut and lists the flags that change the run (#496).
- An outdated or missing optional dependency (ESSOS, virtual-casing-jax,
  neo-jax, GKX) now fails with the `pip install -U` command that fixes it,
  and the import guard also covers `booz_xform_jax` and `solvax` (#496).
- No `RuntimeWarning` when no alpha is lost, and the terminal progress line
  is cleared properly (#496).

## 0.11.5 - 2026-09-28

- `vmex --trace` reports progress: the share of `tmax` traced, the elapsed
  time and an estimate of the time left, on stderr. The header states the
  ARIES-CS targets it scales to (#490).
- `--trace` defaults: 500 alphas, and a Boozer mode cut of 6e-4 instead of
  1e-3, which missed losses in precise quasisymmetry (#491). The tracer uses
  one CPU device per usable core. Floor `essos>=0.19.2`.
- `--trace` benchmarked against SIMPLE and SIMSOPT (#482).
- Fixed-boundary optimization rejects trials whose anchor stalls far from
  the root. Its adjoint is preconditioned with the anchor's block factors.
  Free-boundary restarts start from the reference state alone, and those that
  stall far from ftol go straight to the ladder (#483-#486).
- `MgridField` takes the interpolation order, and a deck's mgrid field can be
  built at either order (#461, #463, #479). The interior field's accuracy is
  documented against an exact equilibrium (#462).
- Turbulence objectives use GKX's forward-mode eigenpair rule (#480).
- `plan.md` is retired, and the README is shorter (#481, #487, #493).

## 0.11.4 - 2026-09-27

- Single-stage optimizations no longer stop early: a refinement restart is
  kept only when it certifies (#477).
- Coherent dependency floors: `solvax>=0.27.0` with `equinox>=0.13.3`,
  `booz_xform_jax>=0.4.1`, `virtual-casing-jax>=0.0.9`, `gkx>=2.4.1`,
  `essos>=0.19`; CI installs a stale environment to check them; DESC needs its
  own environment (#470).
- `vmex --trace`: about 30 s by default, Boozer tracing, summary plots,
  `--trace-birth volume`, `--collisional`; `--scale` targets ARIES-CS
  `a = 1.7044 m`, `<B> = 5.8646 T` (#472, #473).
- Mirrors: `MirrorInput` API and `&MIRROR` input decks; repeat solves drop
  from 38 s to 0.9 s through compiled-kernel reuse (#475).
- Fewer compilations: each optimization lane compiles once (#476).
- CI caching (#466).
- Native axisymmetric polish; the `--polish-*` flags and the old directives
  are removed (#448, #465).

## 0.11.3 - 2026-09-27

- `vmex --trace`: 1000 alphas over 1e-2 s at ARIES-CS size by default (scaled
  in memory; `--trace-no-scale` opts out), traced in Boozer coordinates
  (`essos.boozer`, ESSOS 0.19) in about 30 s on a 10-core laptop, with a
  converged step. New flags: `--trace-birth volume`, `--collisional`,
  `--trace-ne0` and `--trace-te0`. Output: `*_trace.json`/`.npz`, a summary
  figure and a 3-D loss map. Guide: `docs/howto/trace-alpha-particles.md`.
- `--scale` now targets the ARIES-CS wout's own `volavgB = 5.8646 T` and
  `Aminor_p = 1.7044 m` (was `|b0| = 5.7 T`, `1.7 m`, which matches no
  published convention and put the reference reactor-scale wouts 7-10 % high
  in field). `--scale-target axis` keeps Landreman & Paul (2022), Boozer
  `B00 = 5.7 T` on the axis and `a = 1.7 m` (`vmex.core.scaling.b00_axis`).

- The implicit fixed-boundary solve rehomes the forward state beside the
  parameters before refining it. On a GPU host with `JAX_PLATFORMS` set, the
  forward solve ran on the GPU while the callback's parameters stayed on the
  CPU, and every gradient failed with "Received incompatible devices".
- `vmex.solve_phiedge` finds the PHIEDGE whose free-boundary LCFS meets a
  target outboard radius, volume or user metric (bracketed secant over
  warm-started solves); example `examples/free_boundary_phiedge.py`, guide
  `docs/howto/match-phiedge.md`. `vmex.phiedge_root` attaches the
  implicit-function-theorem derivative of that PHIEDGE with respect to
  plasma and coil parameters, from one adjoint gradient.

- Long CPU runs no longer abort with "Failed to materialize symbols": vmex
  releases compiled executables before the process reaches
  `vm.max_map_count`. The compilation cache is one directory per machine, and
  `VMEX_COMPILATION_CACHE=disabled` overrides every cache variable.

## 0.11.2 - 2026-09-24

A compilation-cache directory set through `JAX_COMPILATION_CACHE_DIR` or
`VMEX_COMPILATION_CACHE_DIR` is now split by machine, like the default one
(#449). On clusters, a shared cache path was read by nodes with different CPU
features; XLA rejected the other node's executables ("Target machine feature
... is not supported on the host machine") and recompiled on every run.

## 0.11.1 - 2026-09-23

Free-boundary trials are anchored on the coupled root their gradients
differentiate (#432), the exterior field is accurate next to the plasma
surface, optimization results are the states the objective scored (#427), and
the optimization examples were re-tuned against a five-minute budget. Full
notes in the GitHub release.

## 0.11.0 - 2026-09-21

The interior field is built from the native VMEC form (#403), with a C2
radial interpolant; radial lifts reject under-determined spline fits (#414),
stalled restarts get one deterministic cold retry, and warm workflow profiles
time every stage (#420). Full notes in the GitHub release.

## 0.10.0 - 2026-09-20

The interior field's geometry table carried a spurious `sqrt(s)` on every
odd-`m` coefficient, which put `li383_low_res` surfaces 1.2 cm (R) and 5.9 cm
(Z) off at mid-radius; it now reproduces the wout table to 6.7e-16 m and all
126 adjacent surface pairs nest. Points well outside the plasma return NaN
instead of raising, the inversion starts from the best of three seeds, and a
query on the magnetic axis no longer returns NaN. Full notes in the GitHub
release.

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
