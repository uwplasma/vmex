# VMEX

VMEX is a JAX reimplementation of VMEC2000, the standard code for
three-dimensional ideal-MHD equilibria of stellarators and tokamaks. It reads
VMEC input decks, solves fixed- and free-boundary equilibria (NESTOR vacuum
field from an MGRID table or coils), reproduces VMEC2000's iterations and writes
`wout_*.nc` files that load unchanged in simsopt and booz_xform. Unlike the
Fortran original it differentiates the converged equilibrium through the
implicit function theorem. Its goal is fast, differentiable and
VMEC2000-compatible equilibria for stellarator optimization, including
single-stage design of the plasma boundary and the coils together.

```console
pip install vmex
vmex --test                        # solve, write and plot the bundled QH deck in ./vmex_test
vmex vmex_test/input.nfp4_QH_warm_start          # run any VMEC input deck
vmex --plot vmex_test/wout_nfp4_QH_warm_start.nc
```

The same solve from Python, followed by an implicit derivative:

```python
import jax
import vmex as vj
from vmex.core import implicit

inp = vj.VmecInput.from_file("vmex_test/input.nfp4_QH_warm_start")
result = vj.solve_multigrid(inp)               # converged equilibrium

p0 = implicit.params_from_input(inp)           # differentiable parameters
grad = jax.grad(lambda p: implicit.run(inp, p).wb)(p0)
```

This derivative describes the discrete equilibrium equations. Check equilibrium
and response convergence, then verify the observable under resolution refinement
and independently reconverged perturbations; see {doc}`tutorials/first-gradient`.

New here? {doc}`all-of-vmex` is the whole mental model on one page;
{doc}`installation` covers CPU/GPU installs and `vmex --doctor`.

## How it works

- **Equilibrium.** A stationary point of the ideal-MHD energy over nested flux
  surfaces, with `R`, `Z` and `λ` as Fourier series in the angles and finite
  differences in radius ({doc}`explanation/variational-problem`).
- **Solver.** VMEC2000's damped second-order Richardson force iteration, its
  radial preconditioner and its `NS_ARRAY` multigrid ladder
  ({doc}`explanation/iteration`).
- **Free boundary.** NESTOR's Green's-function vacuum solve coupled to the
  plasma iteration; virtual casing for the field outside the plasma
  ({doc}`explanation/nestor-vacuum`).
- **Derivatives.** The implicit function theorem at the converged root: one
  adjoint solve per scalar objective, derivatives of the discrete equations
  ({doc}`explanation/adjoint-gradients`).

## What has been measured

- **Parity.** CI holds six decks to VMEC2000 (`wb` to `1e-7`, harmonics and
  `iota` to `1e-5`). Six decks never run before agreed to `2.5e-10`, with
  identical iteration counts on five ({doc}`explanation/validation`).
- **Speed.** On those decks (VMEX 0.8.1, Apple M4 CPU) VMEX took 0.50–1.34
  times the VMEC2000 wall time with a warm compilation cache and 0.60–3.13
  times from a cold one ({doc}`reference/performance`).
- **Gradients.** CI compares adjoint gradients with central finite
  differences to `1e-6` (Solov'ev) and `2e-4` (3-D `li383`).
- **Limits.** Free-boundary derivatives are experimental (CPU by default), and
  3-D force-balance polishing has not certified; {doc}`reference/capabilities`
  is the support contract.

::::{grid} 2
:gutter: 3

:::{grid-item-card} Tutorials
:link: tutorials/index
:link-type: doc

Learn by doing: [your first equilibrium and its plots](tutorials/first-equilibrium.md),
[a first gradient](tutorials/first-gradient.md),
[a first optimization](tutorials/first-optimization.md).
:::

:::{grid-item-card} How-to guides
:link: howto/index
:link-type: doc

Task recipes: [run on GPU](howto/run-on-gpu.md),
[restart from a previous run](howto/restart-from-previous-run.md),
[free boundary](howto/free-boundary.md),
[optimization campaigns](howto/optimize-a-boundary.md),
[troubleshooting](howto/troubleshoot.md).
:::

:::{grid-item-card} Reference
:link: reference/index
:link-type: doc

Every [CLI flag](reference/cli.rst),
[input key and its VMEC2000 disposition](reference/vmec2000-compatibility.rst),
[output variable](reference/wout-file.rst) and
[objective](reference/objectives.rst); the
[capability contract](reference/capabilities.rst),
[performance records](reference/performance.rst) and the
[API](reference/api/basic.rst).
:::

:::{grid-item-card} Explanation
:link: explanation/index
:link-type: doc

The methods: [the equilibrium problem](explanation/variational-problem.rst),
[the solver](explanation/iteration.rst),
[NESTOR and virtual casing](explanation/nestor-vacuum.rst),
[adjoint gradients](explanation/adjoint-gradients.md), and
[what is validated](explanation/validation.md).
:::

::::

## Getting support

Bug reports, feature requests, and questions all go to
[GitHub issues](https://github.com/uwplasma/vmex/issues), which offers a
template for each; include the input file and the output of `vmex --doctor`.
{doc}`howto/troubleshoot` covers non-convergence, NaNs, and device placement
first. Contributions follow {doc}`project/contributing`.

```{figure} /_static/figures/readme_runtime_compare.webp
:alt: Runtime comparison of VMEX against VMEC2000 and VMEC++
:width: 95%

Benchmark-suite runtimes at ns = 201 from `benchmarks/baseline.json`
(Apple-Silicon CPU, VMEX 0.3.0, July 2026): VMEX cold and warm against
VMEC2000 and VMEC++. Warm solves reuse the compiled program, as in
optimization loops; the table is in {doc}`reference/performance`.
```

```{toctree}
:hidden:
:caption: Start here

all-of-vmex
installation
```

```{toctree}
:hidden:
:caption: Tutorials

tutorials/index
```

```{toctree}
:hidden:
:caption: How-to guides

howto/index
```

```{toctree}
:hidden:
:caption: Reference

reference/index
```

```{toctree}
:hidden:
:caption: Explanation

explanation/index
```

```{toctree}
:hidden:
:caption: Project

project/contributing
project/references
```
