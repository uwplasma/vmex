# vmec-jax

[![PyPI version](https://img.shields.io/pypi/v/vmec-jax.svg)](https://pypi.org/project/vmec-jax/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/vmec-jax.svg)](https://github.com/conda-forge/vmec-jax-feedstock)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/uwplasma/vmec_jax/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/vmec_jax)](https://github.com/uwplasma/vmec_jax/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/vmec_jax/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/vmec_jax/actions/workflows/ci.yml)
[![Docs](https://img.shields.io/readthedocs/vmec-jax/latest?label=docs)](https://vmec-jax.readthedocs.io/en/latest/)

**vmec-jax** is a clean-room, JAX-native reimplementation of the
[VMEC2000](https://princetonuniversity.github.io/STELLOPT/VMEC) ideal-MHD
equilibrium code for stellarators and tokamaks. It reproduces VMEC2000
iteration-for-iteration on benchmark decks — and, unlike the Fortran
original, it is differentiable and runs on GPUs.

- **VMEC2000 parity.** The solver ports VMEC2000's algorithms
  constant-for-constant (steepest-descent moment method, radial
  preconditioner, spectral condensation, NESTOR vacuum solve). Benchmark
  decks converge in the *same* number of iterations and reproduce the
  plasma energy at machine precision.
- **Differentiable.** Gradients of fixed-boundary equilibrium outputs with
  respect to boundary shape and profile parameters by implicit
  differentiation of the converged fixed point — no finite differences, no
  unrolling — validated against central finite differences to ~1e-6 relative
  (see the gradient table in the docs). Free-boundary and coil-parameter
  derivatives are not yet supported by the implicit residual (roadmap).
- **Drop-in.** Reads VMEC2000 `input.*` namelists and VMEC++-style JSON,
  prints VMEC2000-format iteration output, and writes `wout_*.nc` files
  that load unchanged in simsopt and booz_xform.
- **Batteries included.** Plotting (`vmec --plot`), Boozer transform
  (`vmec --booz`), spline profiles, multigrid, hot restart, free boundary
  from mgrid files *or* directly from coils, typed zero-crash errors.

![Flux surfaces and boundary |B| of the bundled quick-start QH case](docs/_static/figures/readme_equilibrium_showcase.png)

*The bundled quick-start case (`vmec --test`): flux-surface cross sections
and the boundary `|B|` of a four-field-period quasi-helical stellarator,
plotted with the built-in `vmec_jax.core.plotting` helpers.*

## Install

```bash
pip install vmec-jax                        # PyPI
conda install -c conda-forge vmec-jax       # conda-forge
```

Development install from source:

```bash
git clone https://github.com/uwplasma/vmec_jax
cd vmec_jax && pip install -e .
```

## Quickstart

```bash
vmec --doctor     # check the installation and JAX backend
vmec --test       # solve the bundled QH case, write wout + plots
vmec input.X      # run any VMEC2000 input deck (or VMEC++-style JSON)
```

`vmec input.X` writes `wout_X.nc` next to the input (`--outdir` to
redirect). To try it on a real deck:

```bash
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/input.nfp4_QH_warm_start
vmec input.nfp4_QH_warm_start
```

Post-process any wout file, including ones written by VMEC2000:

```bash
vmec --plot wout_nfp4_QH_warm_start.nc     # surfaces, |B|, profiles, 3D
vmec --booz wout_nfp4_QH_warm_start.nc     # Boozer transform -> boozmn_*.nc
vmec --plot boozmn_nfp4_QH_warm_start.nc   # Boozer |B| contours + spectrum
```

## Parity with VMEC2000

![Iteration counts and plasma-energy agreement vs VMEC2000](docs/_static/figures/readme_parity.png)

vmec-jax is validated end-to-end against golden VMEC2000 (PARVMEC 9.0) runs:
the five fixture cases above converge in **exactly** the golden iteration
count — including DSHAPE's mid-run jacobian reset — and reproduce the plasma
energy `wb` to 1 part in 10¹⁵. Across the full benchmark suite below, the
iteration count matches VMEC2000 exactly on 12 of 13 rows (fixed and free
boundary); on the remaining row (Nuhrenberg–Zille QHS) vmec-jax converges in
*fewer* iterations (1681 vs 2829). Per-variable wout agreement and the full
test gates live in the [documentation](https://vmec-jax.readthedocs.io/en/latest/).

![Force residual vs iteration for vmec_jax, VMEC2000, and VMEC++](docs/_static/figures/readme_convergence.png)

*Parity is per-iteration, not just end-to-end: the total force residual
(`fsqr + fsqz + fsql`) of the quick-start QH case at ns=51, per iteration.
The vmec_jax trajectory lies exactly on top of VMEC2000's (both converge in
502 iterations); VMEC++ follows a near-identical path (501 iterations).
Traces: vmec_jax `SolveResult.fsq_history`, VMEC2000 `NSTEP=1` stdout,
VMEC++ wout `fsqt`.*

## Performance

![Wall-clock comparison against VMEC2000 and VMEC++](docs/_static/figures/readme_runtime_compare.png)

Full-solve wall-clock times on the bundled benchmark suite (Apple Silicon
CPU, single thread; `benchmarks/baseline.json`; reproduce with
`python benchmarks/run_baseline.py`):

- **Warm** (in-process, kernels compiled — the number that matters for
  optimization loops, parameter scans, and hot restarts): faster than
  VMEC2000 on 11 of 12 converged rows, typically 1.2–2.5x and up to 8x,
  including 2.1x on the heaviest deck (Nuhrenberg–Zille QHS, 116 s → 56 s).
  The one exception is the free-boundary row, where the vacuum solve is not
  yet tuned.
- **Cold** (fresh CLI process): pays a one-time 5–25 s JAX/XLA startup and
  compile cost, so small decks are slower than Fortran end-to-end; on the
  heaviest deck the cold CLI already beats VMEC2000 (98 s vs 116 s).
  Compiled executables are cached per solver structure, so repeated solves
  in one process — multigrid ladders, scans, optimizations — recompile
  nothing.
- **GPU** (violet, 2x RTX A4000 from `benchmarks/gpu_baseline.json`): at
  these problem sizes a fixed per-solve dispatch overhead dominates, so the
  CPU wins outright; per-iteration throughput favours the GPU by ~3x on the
  largest decks, and the default device policy picks CPU or GPU per stage
  accordingly.

## Features

| | vmec-jax | VMEC2000 | VMEC++ |
|---|:---:|:---:|:---:|
| Fixed-boundary equilibria | ✅ | ✅ | ✅ |
| Free boundary from an mgrid file | ✅ | ✅ | ✅ |
| Free boundary directly from coils (no mgrid) | ✅ | ❌ | ❌ |
| Free-boundary tokamaks (`ntor = 0`) | ✅ | ✅ | ❌ |
| Non-stellarator-symmetric (`LASYM = T`) | ✅ | ✅ | ❌ |
| Fixed-boundary fallback on missing mgrid | ✅ | ✅ | ❌ |
| Spline profiles (cubic / Akima) | ✅ | ✅ | ❌ |
| VMEC++-schema JSON input | ✅ | ❌ | ✅ |
| Hot restart from a previous state | ✅ | ❌ | ✅ |
| Typed zero-crash errors | ✅ | ❌ | ✅ |
| Boozer transform built in (`--booz`) | ✅ | ❌ | ❌ |
| Plotting built in (`--plot`) | ✅ | ❌ | ❌ |
| GPU execution | ✅ | ❌ | ❌ |
| Differentiable (implicit diff, fixed boundary) | ✅ | ❌ | ❌ |

## Python API

```python
from vmec_jax.core.input import VmecInput
from vmec_jax.core.multigrid import solve_multigrid
from vmec_jax.core.wout import wout_from_state, write_wout
from vmec_jax.core.plotting import plot_wout

inp = VmecInput.from_file("input.nfp4_QH_warm_start")
result = solve_multigrid(inp)          # full NS_ARRAY ladder, VMEC2000 numerics
print(result.converged, result.iterations, result.wmhd)

wout = wout_from_state(inp=inp, state=result.state, niter=result.iterations,
                       fsqr=result.fsqr, fsqz=result.fsqz, fsql=result.fsql)
write_wout("wout_nfp4_QH_warm_start.nc", wout)
plot_wout(wout, "figures/")
```

Optimization building blocks live in `vmec_jax.core.optimize`
(quasisymmetry residuals; aspect ratio, iota, mirror ratio, magnetic well
targets; a least-squares driver over boundary Fourier coefficients) with
complete QA/QH/QP/QI scripts in `examples/optimization/`, and
`vmec_jax.core.implicit` provides implicit-differentiation gradients of the
converged fixed-boundary equilibrium. The least-squares driver accepts
`jac="implicit"` to use those exact gradients (fixed boundary,
`LASYM = F`, implicit-differentiable objective terms); the default is
scipy finite differences.

## CLI reference

```text
vmec input.X             solve (INDATA or VMEC++ JSON), write wout_X.nc
vmec --plot wout_*.nc    diagnostic plots from a WOUT file
vmec --booz wout_*.nc    run booz_xform_jax, write boozmn_*.nc
vmec --plot boozmn_*.nc  Boozer contour/spectrum plots
vmec --test              run and plot the bundled quick-start case
vmec --doctor            installation and JAX backend diagnostics

options:
  --outdir PATH          directory for wout/boozmn/figure output
  --mode {cli,jit}       jitted blocks with live printing (cli, default)
                         or a single lax.while_loop (jit)
  --ftol F               override the final-stage FTOL_ARRAY tolerance
  --max-iter N           override the final-stage NITER_ARRAY cap
  --coils PATH           ESSOS-style coils file: drive an LFREEB = T deck
                         by direct Biot-Savart instead of an mgrid file
  --mbooz/--nbooz N      Boozer spectral resolution (default 32/32)
  --booz-surfaces S      Boozer surfaces ('all' or a list of s values)
  --quiet                silence the VMEC-style stdout
```

`vmec` follows the selected JAX backend: with CPU-only JAX it runs on the
CPU; with CUDA-enabled JAX it uses the GPU for the solver stages where that
is faster (`JAX_PLATFORMS=cpu|cuda` pins it explicitly).

## Documentation

Full documentation — installation, quickstart, theory and numerics with
equation-to-source cross-references, API reference, and
performance/validation notes — at
[vmec-jax.readthedocs.io](https://vmec-jax.readthedocs.io/en/latest/).

## License

MIT. If you use vmec-jax in published work, please cite this repository and
the original VMEC papers (Hirshman & Whitson, *Phys. Fluids* 1983;
Hirshman, van Rij & Merkel, *Comput. Phys. Commun.* 1986).
