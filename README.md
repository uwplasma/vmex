# vmec-jax

[![PyPI version](https://img.shields.io/pypi/v/vmec-jax.svg)](https://pypi.org/project/vmec-jax/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/uwplasma/vmec_jax/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/vmec_jax)](https://github.com/uwplasma/vmec_jax/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/vmec_jax/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/vmec_jax/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/vmec_jax/graph/badge.svg?branch=main)](https://codecov.io/gh/uwplasma/vmec_jax?branch=main)
[![Docs](https://img.shields.io/readthedocs/vmec-jax/latest?label=docs)](https://vmec-jax.readthedocs.io/en/latest/)
[![PyPI downloads](https://img.shields.io/pypi/dm/vmec-jax)](https://pypi.org/project/vmec-jax/)

End-to-end differentiable JAX implementation of **VMEC2000** for fixed-boundary
and free-boundary ideal-MHD equilibria.

## Install

```bash
pip install vmec-jax
```

Developer (editable) install:

```bash
git clone https://github.com/uwplasma/vmec_jax
pip install -e vmec_jax/
```

## Usage

Run the solver (VMEC2000-style CLI):

```bash
vmec_jax input.nfp4_QH_warm_start        # → wout_nfp4_QH_warm_start.nc
```

Generate diagnostic plots from any `wout_*.nc` (four-panel output, replicates `vmecPlot2.py`):

```bash
vmec_jax --plot wout_nfp4_QH_warm_start.nc           # saves in same directory
vmec_jax --plot wout_nfp4_QH_warm_start.nc --outdir figures/
```

From Python:

```python
import vmec_jax as vj

# Run a fixed-boundary solve
run = vj.run_fixed_boundary("input.nfp4_QH_warm_start")

# Run a free-boundary solve
freeb = vj.run_free_boundary("input.cth_like_free_bdy_lasym_small")

# Plot any wout file (produces *_VMECparams.pdf, *_poloidal_plot.png, *_VMECsurfaces.pdf, *_VMEC_3Dplot.png)
vj.plot_wout("wout_nfp4_QH_warm_start.nc", outdir="figures/")
```

Run tests:

```bash
pytest -q
```

## Showcase (single-grid)

All figures below use the same **single-grid** run settings: `NS_ARRAY=151`, `NITER_ARRAY=5000`, `FTOL_ARRAY=1e-14`, `NSTEP=500`.

<table>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_cross_sections.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_cross_sections.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center"><code>ITERModel</code> cross-section (VMEC2000 vs vmec_jax)</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> cross-section (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_iota.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_iota.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center"><code>ITERModel</code> iota (VMEC2000 vs vmec_jax)</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> iota (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_3d.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_3d.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center"><code>ITERModel</code> 3D LCFS</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> 3D LCFS</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_bmag_surface.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_bmag_surface.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center"><code>ITERModel</code> |B| on LCFS</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> |B| on LCFS</td>
  </tr>
</table>

<p align="center">
  <img src="docs/_static/figures/readme_fsq_trace_single_grid.png" width="860" />
</p>

<p align="center">
  <img src="docs/_static/figures/readme_runtime_compare.png" width="860" />
</p>

**Cold vs warm runtime**: the *cold* bar includes XLA JIT compilation on the first call (one-time cost per process); the *warm* bar is the steady-state solve time for subsequent calls in the same process. VMEC2000 has no compilation overhead — it is always "cold". The warm vmec_jax time is the fair comparison for repeated solves (e.g., in an optimization loop). vmec_jax automatically caches compiled XLA kernels to disk (`~/.cache/vmec_jax/jax_cache`), so after the first run cold starts also approach warm speed.

## Quasi-helical symmetry optimization (discrete-adjoint)

`examples/optimization/qh_fixed_resolution_jax.py` demonstrates an end-to-end
fixed-boundary QH optimization using the built-in **exact discrete-adjoint Jacobian**
— no finite differences, no SIMSOPT dependency.

> **Discrete adjoint**: rather than perturbing each boundary DOF separately (finite
> differences), vmec_jax records a *checkpoint tape* of the VMEC iteration and
> propagates all parameter tangents through it in one batched forward pass
> (`jax.vmap(jax.jvp(...))`).  The Jacobian is exact (machine precision) and its
> cost is roughly 1–2 forward solves regardless of the number of DOFs — vs.
> *n*_DOFs forward solves for finite differences.
> → [Detailed explanation](docs/discrete_adjoint.rst) · [SIMSOPT comparison](docs/simsopt_comparison.rst)

```bash
python examples/optimization/qh_fixed_resolution_jax.py   # MAX_MODE=2 by default
```

When `max_mode` exceeds the modes present in the input file, vmec_jax automatically
extends the boundary to include the requested harmonics at zero amplitude
(`vj.extend_boundary_for_max_mode`), matching SIMSOPT's `fixed_range()` behaviour.
All runs use consistent VMEC resolution `mpol = ntor = 5` so the initial QS metric
is normalised identically across `max_mode` values.

| `max_mode` | DOFs | QS initial | QS final | Reduction | Wall time ¹ |
|:----------:|:----:|:----------:|:--------:|:---------:|:-----------:|
| 1          |  8   |   0.303    |  0.213   |  30 %     | ~124 s      |
| 2          | 24   |   0.303    |  **0.008** | **97 %** | ~323 s    |

¹ Wall time on Apple M-series (warm-cache subsequent runs are faster).

With only 8 DOFs (`max_mode=1`) the boundary deformation space is too limited
to reach a deep quasi-helical minimum.  `max_mode=2` (24 DOFs) achieves a
97 % reduction because the higher harmonics give the optimizer room to reshape
the boundary helically.

**vmec_jax vs SIMSOPT**: vmec_jax uses an exact discrete-adjoint Jacobian
(one batched JVP pass ≈ 1–2 forward solves regardless of DOF count) while
SIMSOPT + VMEC2000 uses finite differences (*n*_DOFs × 1 forward solve per
Jacobian).  For a detailed comparison of algorithms, runtimes, and memory,
see [docs/simsopt_comparison.rst](docs/simsopt_comparison.rst).

<table>
  <tr>
    <th align="center">max_mode = 1 &nbsp;(8 DOFs, 30 % QS reduction)</th>
    <th align="center">max_mode = 2 &nbsp;(24 DOFs, 97 % QS reduction)</th>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode2/boundary_comparison.png" /></td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode2/bmag_surface.png" /></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/_static/figures/qh_opt/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qh_opt/mode2/objective_history.png" /></td>
  </tr>
</table>

The |B| contour plots show quasi-helical alignment after optimization: contour lines
become increasingly helical (aligned with *m θ − n φ* = const). The ζ axis spans
one field period (0 → 2π/nfp).

Regenerate plots after running the optimization:

```bash
python examples/optimization/plot_qh_optimization_results.py --output-dir results/qh_opt
```

## Quasi-axisymmetric optimization (fixed-boundary)

`examples/optimization/qa_fixed_resolution_jax_ess.py` optimizes an nfp=2 QA
equilibrium for aspect ratio, mean iota, and QA symmetry residuals.

```bash
python examples/optimization/qa_fixed_resolution_jax_ess.py   # MAX_MODE=2 by default
```

When `max_mode` exceeds the modes in the input file, vmec_jax automatically extends
the boundary to include those harmonics at zero amplitude (`vj.extend_boundary_for_max_mode`).
All runs use consistent VMEC resolution `mpol = ntor = 5`.
Objectives: aspect ratio (target 6.0) + mean iota (target 0.41) + QA symmetry residuals.
The optimization history shows three panels: QS residuals, aspect ratio, and mean iota.

| `max_mode` | DOFs | Aspect initial → final | Mean iota initial → final | Wall time ¹ |
|:----------:|:----:|:----------------------:|:-------------------------:|:-----------:|
| 1          |  8   | 5.0 → **6.0** ✓        | 0 → 0 (axisymmetric DOFs) | ~23 s       |
| 2          | 24   | 5.0 → 5.51             | 0 → **0.14** (3D modes)   | ~608 s      |

¹ Wall time on Apple M-series (warm-cache subsequent runs are faster).

With 8 DOFs (`max_mode=1`) only axisymmetric (`n=0`) harmonics are free, so
the optimizer hits the aspect ratio target (5.0 → 6.0) but cannot generate
rotational transform — iota stays at 0.  `max_mode=2` (24 DOFs) unlocks 3D
modes that generate iota (0 → 0.14 toward target 0.41) while partially
improving aspect ratio, at the cost of introducing mild QS breaking.

<table>
  <tr>
    <th align="center">max_mode = 1 &nbsp;(8 DOFs, aspect hit target)</th>
    <th align="center">max_mode = 2 &nbsp;(24 DOFs, iota 0→0.14)</th>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qa_opt/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode2/boundary_comparison.png" /></td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qa_opt/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode2/bmag_surface.png" /></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/_static/figures/qa_opt/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qa_opt/mode2/objective_history.png" /></td>
  </tr>
</table>

## Performance vs parity

- Default runs select the fastest stable path for each input automatically.
- Use `--parity` (or `performance_mode=False` in Python) to force the conservative VMEC2000 loop.
- Use `--solver-mode accelerated` to force the optimized fixed-boundary controller.

Details, profiling guidance, and parity methodology:

- `docs/performance.rst`
- `docs/validation.rst`
- `tools/diagnostics/parity_manifest.toml` + `tools/diagnostics/parity_sweep_manifest.py`

## CLI reference

```
vmec_jax input.*                run the equilibrium solver → wout_*.nc
vmec_jax --plot wout.nc         generate diagnostic plots (4 output files)
vmec_jax --parity input.*       force conservative VMEC2000 loop
vmec_jax --help                 full option list
```

## VMEC++ notes

The current runtime benchmark compares vmec_jax against VMEC2000. VMEC++ is not included in this benchmark.

When VMEC++ is available, it can be added to the runtime plot via `--cpu-summary` entries with `backend=vmecpp`. Some inputs are not supported or do not converge under the same single-grid settings:

VMEC++ unsupported inputs (`lasym=True`):

- `LandremanSenguptaPlunk_section5p3_low_res`
- `basic_non_stellsym_pressure`
- `cth_like_free_bdy_lasym_small`
- `up_down_asymmetric_tokamak`

VMEC++ known non-convergence on these `lasym=False` cases under the same single-grid settings:

- `DIII-D_lasym_false`
- `LandremanPaul2021_QA_reactorScale_lowres`
- `LandremanPaul2021_QH_reactorScale_lowres`
- `LandremanSengupta2019_section5.4_B2_A80`
- `cth_like_fixed_bdy`

## CLI output and `NSTEP`

The VMEC-style iteration loop prints every `NSTEP` iterations. Larger `NSTEP` means fewer print callbacks and faster runs.

To disable live printing:

```bash
export VMEC_JAX_SCAN_PRINT=0
```

Quiet runs (`--quiet` or `verbose=False`) default the scan path to minimal history
mode to reduce host/device traffic. Override with:

```bash
export VMEC_JAX_SCAN_MINIMAL=0  # keep full scan diagnostics even when quiet
```
