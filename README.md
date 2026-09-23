# VMEX

[![PyPI version](https://img.shields.io/pypi/v/vmex.svg)](https://pypi.org/project/vmex/)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/vmex)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/vmex/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/vmex/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/vmex/branch/main/graph/badge.svg)](https://codecov.io/gh/uwplasma/vmex)
[![Docs](https://img.shields.io/readthedocs/vmex/latest?label=docs)](https://vmex.readthedocs.io/en/latest/)

VMEX is a JAX reimplementation of VMEC2000, the standard code for
three-dimensional ideal-MHD equilibria of stellarators and tokamaks. It reads
VMEC input decks, solves fixed- and free-boundary equilibria (NESTOR vacuum
field driven by an MGRID table or coils), reproduces VMEC2000's iteration
and writes standard `wout_*.nc` files. Because the solver is written in JAX,
VMEX also differentiates the converged equilibrium through the implicit
function theorem, so boundary, profile and coil parameters can be optimized
with derivatives of the discrete equilibrium equations.

The goal is fast, differentiable and VMEC2000-compatible equilibria for
stellarator optimization, including single-stage design, in which the plasma
boundary and the coils are optimized together. VMEX also computes Boozer
transforms, fields and their derivatives, quasisymmetry, quasi-isodynamicity
and stability diagnostics, and has a separate lane for open mirrors.

- **Use existing workflows:** VMEC input decks and `wout_*.nc` output, `NS_ARRAY` multigrid
  continuation, hot restart from a saved WOUT, and DESC inputs and outputs read without DESC.
- **Design with gradients:** implicit scalar adjoints and residual Jacobians for SciPy, JAXopt or
  Optax, with quasisymmetry, quasi-isodynamic, Mercier, ballooning, bootstrap and maximum-`J` objectives.
- **Inspect the physics:** Boozer transforms, the magnetic field and its first three spatial
  derivatives, effective ripple, and the `--plot` diagnostic summary.
- **Choose the hardware:** CPU or GPU equilibrium solves (optimization gradients default to CPU),
  reusable compilation and independent-case ensembles.
- **Connect coils:** ESSOS coil fields, NESTOR free boundary from an MGRID table or coils, and the
  virtual-casing exterior field of a finite-beta plasma.

![VMEX equilibria and diagnostics](docs/_static/figures/readme_equilibrium_showcase.webp)

The [capability reference](https://vmex.readthedocs.io/en/latest/reference/capabilities.html)
defines supported models and validation limits. The toroidal model assumes nested flux surfaces;
mirror and high-order polishing features have narrower validation scopes.

## Installation

```console
pip install vmex
vmex --doctor
vmex --test
```

`pip install vmex` is enough to solve, plot, restart, compute Boozer spectra and optimize with SciPy.
Its required dependencies include JAX (CPU), SciPy, netCDF4 and h5py, and two uwplasma packages that
the solver itself calls: [`solvax`](https://pypi.org/project/solvax/) for the linear solvers (the
radial preconditioner, polishing least squares, GMRES) and
[`booz_xform_jax`](https://pypi.org/project/booz_xform_jax/) for the Boozer transform. `vmex --doctor`
prints the interpreter, package versions and JAX devices; `vmex --test` solves a bundled QH deck
end to end and writes its figures to `./vmex_test/`.

Python 3.11, 3.12 and 3.13 are tested. JAX 0.11 needs Python 3.12 or newer, so a 3.11 environment
resolves an older JAX; 3.12 or newer is recommended.

### Optional packages for the full capabilities

Each extra installs one more package from PyPI and turns on the features that need it. Install all
of them with

```console
pip install "vmex[all]"
```

or pick what you need:

| Install | Adds | Enables |
|---|---|---|
| `pip install "vmex[coils]"` | `essos>=0.17` | ESSOS coil fields, `vmex --coils` free boundary, single-stage plasma and coil optimization, field-line and alpha-particle tracing |
| `pip install "vmex[freeb]"` | `virtual-casing-jax>=0.0.8` | the virtual-casing exterior field of the plasma (`VmecExtender`) |
| `pip install "vmex[neoclassical]"` | `neo-jax>=1.0.2` | effective ripple `ε_eff` from a WOUT or Boozer spectrum (`vmex.epsilon_effective_from_wout`) and the `--plot` ripple panel |
| `pip install "vmex[turbulence]"` | `gkx>=1.8.0` (with `jax>=0.10.1`) | gyrokinetic turbulence-proxy objectives (`vmex.core.turbulence`) |
| `pip install "vmex[optimizers]"` | `jaxopt`, `optax` | the JAXopt and Optax optimization drivers |
| `pip install "vmex[all]"` | all of the above | every example and documented workflow |

The same packages can be installed by name; the floors are the ones in `pyproject.toml`:

| Package | Minimum | Installed by | Command |
|---|---|---|---|
| `solvax` | 0.21.0 | `pip install vmex` | `pip install "solvax>=0.21.0"` |
| `booz_xform_jax` | 0.4.0 | `pip install vmex` | `pip install "booz_xform_jax>=0.4.0"` |
| `essos` | 0.17 | `vmex[coils]` | `pip install "essos>=0.17"` |
| `virtual-casing-jax` | 0.0.8 | `vmex[freeb]` | `pip install "virtual-casing-jax>=0.0.8"` |
| `neo-jax` | 1.0.2 | `vmex[neoclassical]` | `pip install "neo-jax>=1.0.2"` |
| `gkx` | 1.8.0 | `vmex[turbulence]` | `pip install "gkx>=1.8.0"` |
| `jaxopt`, `optax` | none | `vmex[optimizers]` | `pip install jaxopt optax` |

NESTOR free boundary from an MGRID table needs no extra. A feature whose package is missing raises
an `ImportError` that names the package to install; the core solver never imports them.

### GPU, conda-forge and source installs

VMEX does not force a GPU build of JAX, because the right wheel depends on the platform and the
CUDA or ROCm version. Install VMEX, then JAX for the accelerator following the
[JAX installation guide](https://docs.jax.dev/en/latest/installation.html) (for example
`pip install -U "jax[cuda13]"`), and read the [VMEX GPU guide](https://vmex.readthedocs.io/en/latest/howto/run-on-gpu.html)
before choosing a device. `conda install --channel conda-forge vmex` installs the core package; its
feedstock may lag PyPI, and the extras above come from pip. For development:

```console
git clone https://github.com/uwplasma/vmex
cd vmex
pip install -e ".[all,dev]"
```

The [installation guide](https://vmex.readthedocs.io/en/latest/installation.html) covers float64,
WSL2 and dependency details.

## Solve, plot and restart

With your own VMEC input file:

```console
vmex input.my_case --plot
vmex --plot wout_my_case.nc
vmex --booz wout_my_case.nc
vmex input.nearby --restart wout_my_case.nc
```

`--plot` writes six PNGs beside the input or in `--outdir`: the summary below, flux-surface cross-sections,
`|B|` in VMEC angles, radial profiles, Mercier stability and the 3-D LCFS. The summary adds Boozer `|B|`, a `J` map,
`D_R` and the DESC-normalized force balance (effective ripple needs `vmex[neoclassical]`). The QA and QI panels are
`vmex examples/data/input.nfp2_QA_finite_beta --plot` and `vmex examples/data/input.nfp4_QI_finite_beta --plot`.

![vmex --plot summary of the bundled finite-beta NFP=2 QA equilibrium](docs/_static/figures/readme_diagnostics_qa.webp)
![vmex --plot summary of the bundled finite-beta NFP=4 QI equilibrium](docs/_static/figures/readme_diagnostics_summary.webp)

VMEX follows the deck's `NS_ARRAY`, `FTOL_ARRAY` and `NITER_ARRAY`. In Python:

```python
import vmex as vj

inp = vj.VmecInput.from_file("input.my_case")
result = vj.solve_multigrid(inp, verbose=True)
print(result.converged, result.fsqr, result.fsqz, result.fsql)
vj.write_wout("wout_my_case.nc", vj.wout_from_result(inp, result))
```

Pass `initial_state=result.state` for a nearby solve, or `restart_from=` for a saved WOUT. The
[restart](https://vmex.readthedocs.io/en/latest/howto/restart-from-previous-run.html) and
[CLI](https://vmex.readthedocs.io/en/latest/reference/cli.html) guides cover resolution changes,
devices, profiles and output controls.

`vmex equilibrium.h5` reads DESC text inputs and HDF5/pickle outputs without installing DESC, using the final stage or equilibrium; it writes `input.equilibrium` and solves it to write `wout_equilibrium.nc`.
`--desc-tol 0` retains all boundary modes. The default 1% boundary tolerance does not guarantee magnetic-field accuracy. WOUT iota has the opposite sign to DESC.

## Differentiate and optimize

Compute a boundary/profile derivative without differentiating through every forward iteration:

```python
import jax
from vmex.core import implicit

params = implicit.params_from_input(inp)
gradient = jax.grad(lambda p: implicit.run(inp, p).aspect)(params)
```

For a simple boundary optimization, supply objectives and let SciPy choose the
steps. Each tuple is `(function, target, cost_weight)`:

```python
from scipy.optimize import least_squares
from vmex import optimize as opt

problem = opt.VmecProblem.from_tuples(
    inp, [(opt.aspect_ratio, 4.0, 1.0)], max_mode=1, use_ess=True)
fit = least_squares(
    problem.residual, problem.x0, jac=problem.residual_jac,
    x_scale=problem.scales, max_nfev=20)
problem.input_from_x(fit.x).to_indata("input.optimized")
equilibrium = problem.equilibrium_from_x(fit.x)
vj.write_wout("wout_optimized.nc", equilibrium.wout)
```

`problem.value_and_grad` supplies scalar objectives to BFGS/L-BFGS-B, `problem.jax_value_and_grad`
to JAX optimizers, and `problem.evaluate(x)` reports solve effort and derivative status. The
[gradient tutorial](https://vmex.readthedocs.io/en/latest/tutorials/first-gradient.html) and
[optimization guide](https://vmex.readthedocs.io/en/latest/howto/optimize-a-boundary.html) cover
convergence checks, constraints, scaling and finite-difference verification.

## How VMEX works

1. **Energy principle.** VMEX finds a stationary point of `W = ∫ (B²/2μ₀ + p/(γ−1)) dV` over
   nested flux surfaces, where `J × B = ∇p`. As in VMEC, the unknowns are the surface shapes `R`, `Z`
   and the angle function `λ`: Fourier series in both angles, finite differences in radius.
2. **Spectral force iteration.** Each iteration evaluates the Fourier-projected forces and takes a
   damped second-order Richardson step with VMEC2000's time-step control and restarts, in its
   order. A radial tridiagonal preconditioner per Fourier mode makes the step effective (a 2-D block
   preconditioner is opt-in), and `NS_ARRAY` runs a coarse-to-fine radial multigrid ladder. The
   solve has converged when `FSQR`, `FSQZ` and `FSQL` fall below `FTOL`.
3. **Free boundary.** NESTOR solves the exterior Neumann problem for the vacuum potential with
   Merkel's Green's-function method, driven by an MGRID table or a coil field; pressure balance
   across the boundary moves the plasma surface.
4. **Implicit derivatives.** At a root `F(x, p) = 0` of the force residual, `dx/dp = −(∂F/∂x)⁻¹ ∂F/∂p`.
   A scalar objective needs one adjoint linear solve for all parameters, and least-squares
   Jacobians factor the block-tridiagonal radial operator once. The forward iterations are not
   recorded, so memory does not grow with their number. The derivative assumes that the state is a
   root (on a fixed boundary VMEX first Newton-refines it to `refine_tol = 1e-10`), that `∂F/∂x` is
   invertible there, and that perturbed solves stay on the same branch. It is the derivative of the
   discrete equations, not of the continuum problem.
5. **Virtual casing.** `VmecExtender` adds the plasma currents' field, a surface integral over
   the boundary field, to the coil field outside the plasma; its error grows near the surface.

The [explanation pages](https://vmex.readthedocs.io/en/latest/explanation/index.html) derive each step.

## Results

- **VMEC2000 parity.** CI solves six decks against stored VMEC2000 output and requires `wb` within
  `1e-7` relative, `iota` and `R`, `Z` harmonics on three surfaces within `1e-5`, and iteration
  counts within ±25%. On six decks never run before (ITER, W7-X, HSX, ARIES-CS, ESTELL,
  Nührenberg–Zille), the worst relative difference from VMEC2000 was `2.5e-10`, and the iteration
  counts were identical on five and one apart on the sixth
  ([record](benchmarks/fresh_decks_vs_vmec2000_2026-09-02.md), VMEX 0.8.1).
- **Speed.** On those decks (Apple M4 CPU, one run each) VMEX took 0.50–1.34 times the VMEC2000
  wall time with a warm compilation cache and 0.60–3.13 times with it cleared; the difference is
  mostly XLA compilation, which a repeated solve in one process skips. In the older
  [ns = 201 table](https://vmex.readthedocs.io/en/latest/reference/performance.html) (VMEX 0.3.0)
  VMEC++ was faster than VMEX on the largest 3-D decks it completed. On two RTX A4000 GPUs no
  shipped deck or problem size ran faster than on the CPU.
- **Derivatives.** CI compares adjoint gradients with central finite differences: four Solov'ev
  gradients to `1e-6` relative and a 3-D `li383` boundary gradient to `2e-4`. One warm
  value-plus-Jacobian evaluation of a 48-parameter QA problem took 16.6 s on an Apple M4
  ([record](benchmarks/qa_optimization_startup_least_squares_m4.json), VMEX 0.7.0).
- **Not validated.** Free-boundary derivatives are experimental (CPU by default), 3-D force-balance
  polishing has not certified, and mirror beta above 10% is extended validation. The
  [validation record](docs/explanation/validation.md) lists every gate, tolerance and record.

![Force-residual traces for VMEX, VMEC2000 and VMEC++](docs/_static/figures/readme_convergence.webp)

The NFP=4 QH deck at ns = 51 in all three codes, from a trace recorded in July 2026
(`python benchmarks/make_readme_figures.py --only convergence`).

## What you can build

Every figure below is regenerated by the script named beside it, from decks in this repository.

### Quasisymmetric and quasi-isodynamic boundaries

![QA at nfp 2, QH at nfp 4 and QP at nfp 2](docs/_static/figures/readme_optimization.webp)

Boundary optimization for quasi-axisymmetry, quasi-helical symmetry and quasi-poloidal symmetry, each at its own
field-period count, from `examples/optimization/QA_optimization.py` and its QH and QP siblings.
`python examples/plot_optimized_families.py` draws the panel.

![QI at nfp 1, 2, 3 and 4](docs/_static/figures/readme_qi.webp)

Quasi-isodynamic designs at four field-period counts: the `|B|` contours close poloidally as `nfp`
rises. The QI scripts in `examples/optimization/` cover vacuum, finite-beta, bootstrap-consistent and
maximum-`J` variants, with scalar-adjoint, SciPy, JAXopt and Optax drivers.

### Free boundary with coils

![Free-boundary beta ramp and Shafranov shift](docs/_static/figures/readme_essos_beta_scan.webp)

NESTOR free boundary driven by ESSOS coils (or an MGRID table), ramping beta and tracking the
Shafranov shift: `python examples/free_boundary_essos_coils.py`. The exterior field is described
[below](#fields-coils-and-free-boundary).

### Single-stage plasma and coil design

`examples/optimization/single_stage_optimization.py` adjusts the plasma boundary and the coils
against one weighted objective, solving the equilibrium implicitly at every step;
`single_stage_free_boundary_optimization.py` couples them through a true free-boundary solve. Both
print final plasma and coil metrics; `single_stage_optimization.py` also states whether it met its
rotational-transform and normal-field targets. A lower penalty with unmet targets is not a design. The free-boundary gradient is exact at the root of the coupled residual, but the
free-boundary state is not yet Newton-refined onto it. A zero-beta free boundary also needs a nested
coil-field surface enclosing PHIEDGE; at an island chain VMEX, VMEC2000 and VMEC++ all fail to
converge ([not validated](docs/explanation/validation.md#what-is-not-validated)).

### Open mirrors and stellarator-mirror hybrids

![Fixed-boundary non-axisymmetric mirror](docs/_static/figures/mirror_fixed_boundary_3d.webp)

Fixed-boundary open mirrors, from `examples/mirror/mirror_fixed_boundary_nonaxisymmetric.py`.

![Free-boundary mirror beta scan](docs/_static/figures/mirror_free_boundary_beta_scan.webp)

The free-boundary mirror beta scan over the validated 0 to 10 percent range,
`examples/mirror/mirror_free_boundary_beta_scan.py` (needs `vmex[coils]`).

![Stellarator-mirror hybrid](docs/_static/figures/stellarator_mirror_hybrid.webp)

Periodic stellarator-mirror hybrids, `examples/mirror/stellarator_mirror_hybrid.py`, with a
quasi-isodynamic variant in `qi_mirror_hybrid_fourier_vs_bspline.py`. Hybrids and anisotropy are research scopes: see the
[mirror guide](https://vmex.readthedocs.io/en/latest/howto/mirror-machines.html) for what is validated.

### Running the examples

The examples live in the repository, not in the wheel. From a clone:

```console
git clone https://github.com/uwplasma/vmex
cd vmex
pip install -e ".[all]"
vmex examples/data/input.circular_tokamak --plot
python examples/take_gradients.py
```

| Application | Runnable starting point | Needs |
|---|---|---|
| Tokamak or stellarator equilibrium | `vmex examples/data/input.circular_tokamak --plot`; other decks in [examples/data](examples/data/) | core |
| QA, QH, QP or QI boundary design | [examples/optimization](examples/optimization/), including bootstrap, ballooning, Mercier and maximum-`J` variants | core |
| JAXopt and Optax drivers | `QI_optimization_jaxopt.py`, `QI_optimization_optax.py` | `vmex[optimizers]` |
| Asymmetric boundary design | [stellarator_asymmetry](examples/optimization/stellarator_asymmetry/) vacuum and finite-beta scripts | core |
| Single-stage plasma and coils | `single_stage_optimization.py`, `single_stage_free_boundary_optimization.py` | `vmex[coils]` |
| Fields and spatial derivatives | `python examples/vmex_get_B_gradB.py` | core |
| Exterior field from coils and plasma | `python examples/vmex_get_B_outside_plasma.py` | `vmex[coils,freeb]` |
| ESSOS coils and a free-boundary beta scan | `python examples/free_boundary_essos_coils.py` | `vmex[coils]` |
| Free boundary from an MGRID table | `python examples/free_boundary_mgrid.py` | core; first `python tools/fetch_assets.py --bundle reference-nc` |
| Experimental exterior tracing (unqualified topology) | `python examples/vmex_fieldline_tracing_finite_beta.py` | `vmex[coils,freeb]` |
| Effective ripple | `python examples/epsilon_effective.py` | `vmex[neoclassical]` |
| Independent-case ensembles | `python examples/parallel_ensemble_scan.py` | core |
| Open mirrors | `mirror/mirror_fixed_boundary_nonaxisymmetric.py`, `mirror/mirror_free_boundary_beta_scan.py` | core; the beta scan needs `vmex[coils]` |
| Research force-balance polishing | `python examples/force_balance_polishing.py` | core |

The optimization scripts expose resolutions, objective weights and iteration budgets near the top.
Inspect those settings before a research run; advanced coil examples need the optional dependencies
and versions in the [ESSOS guide](https://vmex.readthedocs.io/en/latest/howto/use-essos-fields-and-coils.html).

## Fields, coils and free boundary

The live equilibrium exposes Cartesian `B()`, `gradB()`, `gradgradB()` and
`gradgradgradB()`, with corresponding VJPs in the originating problem's degrees
of freedom. Use `set_points_xyz(...)` or `set_points_flux(...)` to select
interior evaluation points.

For an exterior field, `vj.VmecExtender.from_file("wout_my_case.nc",
external_field=coils.B)` combines the plasma's virtual-casing contribution with
the supplied coil field. The plasma part is a quadrature over a source grid on
the plasma surface, sampled by default from the boundary's aspect ratio, field
periods and requested digits. Its error grows rapidly near that surface, so at
the points where its error estimate misses the requested digits an eager call
switches to a target-graded quadrature, accurate to about 1e-12 of the field
down to 0.01 minor radii at a few milliseconds per point;
`with_graded_quadrature()` uses it everywhere, including under `jit`. Targets
must also stay away from coil filaments, and an MGRID field has a finite
tabulated domain. The exterior field-line example traces through the graded
field; a finite trace does not by itself establish magnetic topology.
See the [exterior-field explanation](https://vmex.readthedocs.io/en/latest/explanation/nestor-vacuum.html)
and [field and coil usage](https://vmex.readthedocs.io/en/latest/howto/use-essos-fields-and-coils.html).

Joint boundary/coil optimization and the boundary-Schur adjoint remain advanced
workflows with substantial solve costs; they require independent derivative and
final-constraint checks. Open mirrors support defined isotropic
fixed/free-boundary cases; the shipped free-boundary 0–10% beta range is the
supported range, while higher beta, anisotropy and periodic hybrids need further
validation. See the [mirror guide](https://vmex.readthedocs.io/en/latest/howto/mirror-machines.html).

## Accuracy and optional polishing

A small VMEC `FSQR/FSQZ/FSQL` means the discrete solve converged; it does not bound the continuous
force error `J × B − ∇p`. Optional spline-based polishing is off by default and remains a research
feature. It certifies on the bundled axisymmetric shaped, finite-pressure tokamak, where the gain
is real but not uniform; an accurate, affordable 3-D polished solve is still an open goal.

![Force error of a shaped finite-pressure tokamak before and after polishing](docs/_static/figures/readme_polish_before_after.webp)

`python examples/force_balance_polishing.py` (3 to 5 minutes on one CPU) writes both WOUT files on
the same 129-surface mesh and certifies each with the same independent oracle, so they differ only
in the polish. The near-axis error (`ρ < 0.2`) falls from 2.9e3 to 61 N m⁻³ and the edge error
2.5-fold, but the polished state is slightly worse for `0.6 ≲ ρ ≲ 0.8`, and the volume-averaged
`⟨|F|⟩/⟨|∇(B²/2μ₀)|⟩` falls only from 2.3e-3 to 1.9e-3. The written file keeps the native
certificate (`eps_F` 1.80e-3 native, 1.90e-3 read back). The `--plot` summary's force panel, a
finite-difference rebuild from the WOUT, reads about 5.8e-3 on both files and cannot show this.

From the command line, `vmex examples/data/input.shaped_tokamak_pressure_polished --polish auto`
runs the same polish; `AUTO` checks the estimated Gauss–Newton work against `--polish-budget` (an
admission estimate, not a timeout). `eps_F` is bounded above by 2 by construction and saturates in
vacuum; read the dimensional metrics with it. See the [validation record](docs/explanation/validation.md) for
the failed 3-D attempts and the [polishing reference](https://vmex.readthedocs.io/en/latest/explanation/high-order-force-balance.html)
for the method and certificate.

## Performance and parallel execution

JAX compiles each solve once per array structure and reuses the executable for matching shapes, so
measure first-call, cache-reload and warm costs separately, including refinement and gradients for
optimization. CPU and GPU performance depend on resolution and workload; the dated measurements are
in the [performance reference](https://vmex.readthedocs.io/en/latest/reference/performance.html)
and the [benchmark records](benchmarks/INDEX.md).

`vj.parallel.solve_ensemble(inputs, workers=None)` solves independent cases concurrently and returns
results in input order, each identical to solving that input alone; `workers=1` is the serial
baseline. Set worker and device budgets with the
[ensemble guide](https://vmex.readthedocs.io/en/latest/howto/parallel-ensembles.html). Multi-device
kernel and derivative tests do not yet establish a scalable distributed nonlinear equilibrium solve.

## Documentation, development and citation

Start with [your first equilibrium](https://vmex.readthedocs.io/en/latest/tutorials/first-equilibrium.html)
and [your first optimization](https://vmex.readthedocs.io/en/latest/tutorials/first-optimization.html),
then the [API](https://vmex.readthedocs.io/en/latest/reference/api/basic.html),
[VMEC compatibility](https://vmex.readthedocs.io/en/latest/reference/vmec2000-compatibility.html) and
[troubleshooting](https://vmex.readthedocs.io/en/latest/howto/troubleshoot.html) pages. For development,
`pip install -e ".[dev]"` and `python tools/preflight.py --static`. Report issues with the input deck
and `vmex --doctor` output. See [contributing](CONTRIBUTING.md), [citation](CITATION.cff),
[license](LICENSE) and the [plan](plan.md).
