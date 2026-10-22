# Changelog

Release notes are mirrored from the GitHub releases, which carry each release
in full. A number appears here only where a committed artifact backs it; each
record under `benchmarks/` names the revision it was measured at.

## 0.12.0 - 2026-10-21

- **Faster optimization.** Implicit executables are keyed on problem structure
  and share one block factorization: QA stage compile 32-40 s to 24-27 s, a
  second stage of the same shape 24-38 s to 13-15 s, peak memory about 30 %
  lower (#599, #618). Adjoint tolerances are traced, so changing one no longer
  recompiles (#623). Rejected trials stop early (#608), the Jacobian reuses the
  refinement's block bands (#622), and multi-stage examples share one problem
  (#621). Geometry synthesis is separable, 10-15 % faster warm solves (#597).
- **Ballooning stability** is resolved along the field line (Richardson step,
  fourth-order radial reads) and matches COBRAVMEC within 1.7 %; the hard
  maximum is the default and the Jacobian is 6x cheaper (#589).
- **Free boundary:** opt-in Newton anchoring (`solve_free_boundary_newton`, #575),
  opt-in position control (`LPOSITION_CONTROL`, #570), a DEL-BSQ gate and a
  coil-gradient test for single-stage runs (#614).
- Opt-in radial preconditioner for the m = 0 lambda force,
  `lambda_m0_floor=1e-3` (#598). 3-D force polishing with prescribed current (#558).
- `boozer_spectrum_state(..., geometry=True)` returns the Boozer R, Z, nu and
  Jacobian spectra (#610). Tracing takes explicit energy, mass and charge (#540).
- JAX 0.12 support: the on-axis coil field stays differentiable (#620).
- Every example finishes in under five minutes on a laptop and prints
  progress (#562). README and docs restructured, with ballooning, polishing
  and MRX sections and a CLI reference (#605, #617, #540).
- CI runs all seven VMEC2000 golden-digest cases on every PR (#616); a
  packaging-triggered and weekly install check covers Linux, macOS and Windows
  (#588). `neo-jax>=1.0.6` (#588). About 880 fewer source lines (#601, #602).

## 0.11.11 - 2026-10-18

- **mgrid scaling matches VMEC2000.** EXTCUR multiplies every coil table; VMEX
  no longer divides by the raw coil current, which gave 5e-5 of the field on a
  raw mgrid. A deck NFP that differs from the mgrid NFP now stops the run (#591).
- A free-boundary boundary that leaves the mgrid box warns as VMEC2000 does,
  naming the side, the overshoot and the grid extents (#604). `piota` ignores
  BLOAT, as in VMEC2000; BLOAT other than 1 requires NCURR = 1.
- A PHIEDGE whose sign opposes the external toroidal field stops the run, as
  VMEC2000's vacuum.f does (#571).
- LASYM: the theta shift uses VMEC2000's `atan2` form (#606).
- WOUT `phi` is correct for non-uniform APHI; the namelist reader accepts
  `&END`, `$END` and a `/` before a comment; the summary plot works at NS = 3 (#593).
- `import vmex` no longer changes `jax.config`. `vmex.configure()` sets x64,
  the compilation cache and logging in one place; the CLI and the first core
  import call it, and the solver refuses to run in float32 (#607).
- CI runs in seven balanced lanes and the test suite runs jitted (#586, #580).
- README: VMEX against exact equilibria, with and without stellarator symmetry (#578).

## 0.11.10 - 2026-10-13

- **`vmex --trace` is about twice as fast** on large spectra with ESSOS 0.21:
  ARIES-CS, 500 alphas over 10 ms on 10 M4 cores, 111 s to 53 s (#581).
- `trace_alphas` and `vmex --trace` warn when the WOUT spectrum looks truncated
  (`spectral_truncation` > 3e-3; see the trace how-to for what to do about it).
- Floor `essos>=0.21.0` (0.20.1 brought per-pair Coulomb logarithms). `vj.essos_tracing_fields(wout, coils, wall=0.03)` gives ESSOS `Tracing`
  the VMEC field, tabulated exterior field and wall (`examples/vmex_essos_tracing_to_wall.py`).
- `VmecInteriorField` and the live-state field spectra work without stellarator
  symmetry (`LASYM = T`): the sine families `rmns`, `zmnc`, `lmnc` and the
  `B^u`/`B^v` sine tables are evaluated, and symmetric results are unchanged
  bit for bit. On the exact up-down asymmetric Solov'ev equilibrium at ns = 65
  the field gives B to 2e-7 and J to 3e-6-2e-5 at mid radius (4e-4 from WOUT).
- `examples/vmex_mrx_comparison.py` relaxes the 2.5 % beta QA deck with MRX
  from the VMEX ns = 65 field and compares the two (Poincare sections, iota,
  toroidal current, pressure). MRX is optional.
- `vmex --trace` runs on one GPU when several are visible (two A4000s took 2-3
  times longer than one); `trace_alphas(devices=jax.devices())` still splits.

## 0.11.9 - 2026-10-07

- On a GPU, every fused run of kernels is captured into a CUDA graph
  (`--xla_gpu_graph_min_graph_size=1` unless `XLA_FLAGS` is set): 8-11 %
  faster per solver iteration on an RTX A4000, results unchanged (#563).
- `vmex --neoclassical`: DKX neoclassical transport (#548).
- Optimization examples: dimensionless Mercier and resistive hinges for the
  bootstrap examples, a DESC-style ballooning hinge with a COBRAVMEC 3-D
  gate, and larger mode-2 budgets for the scalar and SciPy lanes
  (#555, #556, #557, #560).

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

- `vmex --trace` at ARIES-CS size in Boozer coordinates (ESSOS 0.19), with
  volume births, `--collisional` and summary figures; `--scale` targets the
  ARIES-CS `volavgB = 5.8646 T`, `Aminor_p = 1.7044 m`.
- `vmex.solve_phiedge` and `vmex.phiedge_root` match a free-boundary target
  through PHIEDGE, with its implicit derivative.
- Fixes: GPU-host implicit solves rehome the forward state beside the
  parameters; long CPU runs release executables before `vm.max_map_count`.

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

## 0.11.0 and earlier

See the [GitHub releases](https://github.com/uwplasma/vmex/releases).
