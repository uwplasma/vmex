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
  plasma energy at machine precision. An optional **2D block
  preconditioner** cuts iterations 2.5–11x on stiff cases while leaving the
  default path byte-identical.
- **Differentiable.** Gradients of *fixed-boundary* equilibrium outputs with
  respect to boundary shape and profile parameters by implicit
  differentiation of the converged fixed point — no finite differences, no
  unrolling — validated against central finite differences to ~1e-6 relative
  (see the gradient table in the docs), with an O(1)-memory adjoint. **Free
  boundary** is differentiable end-to-end through the virtual-casing vacuum
  field (coil / `extcur` derivatives), finite-difference-validated.
- **Drop-in.** Reads VMEC2000 `input.*` namelists and VMEC++-style JSON,
  prints VMEC2000-format iteration output, and writes `wout_*.nc` files
  that load unchanged in simsopt and booz_xform.
- **Batteries included.** Plotting (`vmec --plot`), Boozer transform
  (`vmec --booz`), spline profiles, multigrid, hot restart, free boundary
  from mgrid files *or* directly from coils, near-axis (pyQSC/pyQIC) seeding,
  typed zero-crash errors — with the shared linear/adjoint solver layer
  factored out into [SOLVAX](https://pypi.org/project/solvax/).

![Flux surfaces, 3-D geometry, and Boozer |B| of the bundled quick-start QH case](docs/_static/figures/readme_equilibrium_showcase.png)

*The bundled quick-start case (`vmec --test`): flux-surface cross sections,
the 3-D plasma boundary coloured by `|B|`, and `|B|` in **Boozer coordinates**
on the last closed flux surface (the near-straight diagonal contours are the
signature of quasi-helical symmetry) for a four-field-period stellarator —
all from the built-in `vmec_jax.core.plotting` / `core.boozer` helpers.*

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
energy `wb` to 1 part in 10¹⁵. Across the full benchmark suite below (14 rows,
all at `ns ≥ 51`), the iteration count matches VMEC2000 exactly on 12 rows; on
the free-boundary CTH-like row it converges in 703 vs 642 iterations (a ~9%
tail), and on Nuhrenberg–Zille QHS it converges in *fewer* iterations (1681 vs
2829). Per-variable wout agreement and the full test gates live in the
[documentation](https://vmec-jax.readthedocs.io/en/latest/).

![Force residual vs iteration for vmec_jax, VMEC2000, and VMEC++](docs/_static/figures/readme_convergence.png)

*Parity is per-iteration, not just end-to-end: the total force residual
(`fsqr + fsqz + fsql`) of the quick-start QH case at ns=51, per iteration.
The vmec_jax trajectory lies exactly on top of VMEC2000's (both converge in
502 iterations); VMEC++ follows a near-identical path (501 iterations).
Traces: vmec_jax `SolveResult.fsq_history`, VMEC2000 `NSTEP=1` stdout,
VMEC++ wout `fsqt`.*

### Optional 2D preconditioner: fewer iterations on stiff cases

The default radial (1D) preconditioner reproduces VMEC2000 iteration-for-iteration.
For stiff decks — very high aspect ratio, strong finite-β coupling — an opt-in
**2D block preconditioner** (matrix-free Newton: a Jacobian-vector-product Hessian
on SOLVAX's GMRES) cuts the iteration count 2.5–11x. It is a strict add-on: the
default 1D path stays byte-identical.

![2D vs 1D preconditioner iteration counts on stiff cases](docs/_static/figures/readme_precond.png)

## Performance

![Wall-clock comparison against VMEC2000 and VMEC++](docs/_static/figures/readme_runtime_compare.png)

Full-solve wall-clock times on the bundled benchmark suite (Apple Silicon
CPU, single thread; `benchmarks/baseline.json`; reproduce with
`python benchmarks/run_baseline.py`):

- **Warm** (in-process, kernels compiled — the number that matters for
  optimization loops, parameter scans, and hot restarts): faster than
  VMEC2000 on 9 of the 13 converged rows, typically 1.3–2.2x and up to ~4x on
  the smallest decks, including 1.3x on the heaviest deck (Nuhrenberg–Zille
  QHS, which also converges in far fewer iterations). Of the four remaining
  rows two are dead heats (multigrid QA/QH) and two are the **free-boundary**
  rows: those now *converge* to VMEC2000 parity (fixed in R15 — they used to
  stall) but their warm wall is not yet faster than Fortran, since the vacuum
  (NESTOR) solve is not fully tuned. (These ratios were measured on a CPU
  shared with other load, so they are conservative lower bounds — on an idle
  machine the warm margins are larger; the warm/Fortran *ratio*, not the
  absolute seconds, is the comparable quantity.)
- **Cold** (fresh CLI process): pays a one-time 5–25 s JAX/XLA startup and
  compile cost, so an end-to-end CLI run is slower than Fortran — that
  overhead is a fixed toll, not physics. Compiled executables are cached per
  solver structure, so repeated solves in one process — multigrid ladders,
  scans, optimizations — recompile nothing, which is why the warm number is
  the one that matters inside a workflow.
- **GPU** (violet, 2x RTX A4000 from `benchmarks/gpu_baseline.json`): at
  these problem sizes a fixed per-solve dispatch overhead dominates, so the
  CPU wins outright; per-iteration throughput favours the GPU by ~3x on the
  largest decks, and the default device policy picks CPU or GPU per stage
  accordingly.
- **Memory.** Peak resident memory (0.6–1.5 GB, up to 3.3 GB on the largest
  multigrid deck) is dominated by the transient JAX/XLA *compile* working set,
  not the equilibrium data — the spectral state, DFT transform tensors, and
  solver carry together are a few MB, and a solve's warm runtime footprint is
  tens of MB. It is a per-process, per-resolution compile cost that amortizes
  across repeated solves. The optimization Jacobian is memory-bounded by
  column chunking (`jac_chunk_size="auto"`; DESC's knob) so it does not scale
  with the boundary-dof count, and the implicit-gradient compile was cut ~20%
  by factoring the residual and field pipelines into reusable compiled
  sub-computations.

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
| Differentiable fixed boundary (implicit diff, O(1) memory) | ✅ | ❌ | ❌ |
| Differentiable free boundary (virtual casing) | ✅ | ❌ | ❌ |
| 2D block preconditioner (stiff-case speedup) | ✅ | ❌ | ❌ |
| Near-axis (pyQSC / pyQIC) optimization seed | ✅ | ❌ | ❌ |

## Code size

vmec-jax delivers that superset of capabilities in roughly **half the code**,
and is the most densely documented of the three. Solver source only (tests,
language bindings, and vendored third-party excluded), counted with
[`pygount`](https://pypi.org/project/pygount/) 3.2:

| code base | language | files | code (SLOC) | comments / docstrings | doc-to-code |
|---|---|---:|---:|---:|---:|
| **vmec-jax** | Python | 36 | **11,789** | 5,532 | **0.47** |
| VMEC2000 (PARVMEC) | Fortran | 115 | 24,190 | 8,425 | 0.35 |
| VMEC++ | C++ / Python | 117 | 22,824 | 7,646 | 0.34 |

vmec-jax is under half the SLOC of VMEC2000 and about half of VMEC++, while
*adding* differentiability, GPU execution, direct-coil free boundary, and a
built-in Boozer transform — and it carries the highest comment/docstring
density of the three (reproduce with
`pygount --format=summary vmec_jax`).

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

From a circular-torus seed, staged `max_mode` continuation with ESS and
`jac="implicit"` (measured on an office 36-core CPU, quasisymmetry
residual = `QuasisymmetryRatioResidual.total`):

| class | nfp | helicity (m,n) | seed QS | achieved QS | max_mode | status |
|-------|-----|----------------|---------|-------------|----------|--------|
| QA | 2 | (1, 0)  | 2.04e-01 | **1.70e-04** | 2 | precise (>3 orders; aspect 6.00, iota 0.42) |
| QH | 4 | (1, −1) | 6.91e-01 | **5.83e-05** | 5 | precise (>4 orders; aspect 8.00, iota −1.22) |
| QP | 2 | (0, 1)  | 4.46e-01 | 9.4e-02 | 5 | basin-limited (documented QP caveat; same basin to `max_mode` 5) |
| QI | 1 | (0,1)→QI | 2.43 | 2.14e-02 | 3 | strong QP→QI (>2 orders); not precise — needs richer omnigenity residual |

![QA/QH/QP/QI optimization: seed vs optimized boundary and Boozer |B| on the LCFS](docs/_static/figures/readme_optimization.png)

*Each class starts from a near-circular torus (grey, dashed) and is shaped
into a quasi-symmetric stellarator (blue) by the least-squares driver; the
bottom row is `|B|` in Boozer coordinates on the LCFS (jet), whose contour
geometry reads off the symmetry family — horizontal for QA, diagonal for QH,
vertical for QP. `QS` is the quasisymmetry residual measured on the plotted
equilibrium; the table above lists the deepest values reached by the full
continuation campaign. Reproduce with
`python benchmarks/make_readme_figures.py --only optimization` from the decks
in `benchmarks/opt_decks/`.*

Implicit gradients are *essential*, not merely faster: for the helical (QH)
target the exact-axisymmetric seed is a saddle where finite differences stall,
and for QP the implicit path reaches a far better basin than FD. The implicit
Jacobian is **CPU-pinned by default** — it is launch-bound (one preconditioned
GMRES per boundary dof), so a `max_mode`-2 (24-dof) Jacobian evaluates in
~101 s on CPU versus >37 min hung in a single kernel-launch on the GPU. That
pin is what makes the deep QH `max_mode` 3→5 continuation (to precise QS
5.8e-5) tractable; the whole campaign is a multi-hour CPU run. The forward
solve is a host callback, so the small fixed-boundary solve does not benefit
from a GPU (cold solve ~2× faster on CPU); per stage the wall is dominated by
a one-time XLA compile of the implicit Jacobian.

## vmec-jax vs DESC

[DESC](https://desc-docs.readthedocs.io/) is the other JAX-native,
differentiable stellarator-equilibrium code, and the natural point of
comparison. The two solve *different* problems: DESC minimises the MHD force
in a global Zernike–Fourier basis (its own equilibrium), while vmec-jax
reproduces VMEC. Both are differentiable, GPU-capable, and built on JAX; most
of vmec-jax's distinct strengths follow from *being* VMEC:

| | vmec-jax | DESC |
|---|:---:|:---:|
| Iteration-for-iteration VMEC2000 parity + standard `wout_*.nc` | ✅ | — (a different equilibrium) |
| Drop-in VMEC2000 `input.*` / VMEC++ JSON, VMEC-format iteration prints | ✅ | — |
| Full VMEC namelist incl. non-symmetric (`LASYM = T`) surfaces | ✅ | partial |
| Free boundary via **NESTOR and** virtual casing | ✅ | virtual casing |
| JAX-native, differentiable, GPU | ✅ | ✅ |
| Implicit (adjoint) equilibrium gradients | ✅ O(1) memory, column-chunked | ✅ |
| High accuracy at *low* radial resolution | finite-difference radial grid | ✅ Zernike basis |
| Breadth of built-in objectives / optimizers | growing | ✅ mature |

Reach for vmec-jax when you want a differentiable code that *is* VMEC — the
same converged state, the same `wout`, the same conventions — dropped into an
existing VMEC-based workflow (simsopt, `booz_xform`, near-axis tooling), with
an O(1)-memory adjoint whose peak memory does not grow with the number of
design variables. Reach for DESC when you want its global spectral accuracy at
low radial resolution or its large, mature objective library. The comparison
is honest in both directions; the two codes are complementary.

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
