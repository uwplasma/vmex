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

## Choosing CPU or GPU

`vmec_jax` follows the JAX backend you select. If you installed CPU-only JAX,
runs use CPU. If you installed GPU-enabled JAX and select a GPU backend, runs
use GPU; vmec_jax does not silently force those runs back to CPU.

```bash
# Check what JAX will use.
python -c "import jax; print(jax.default_backend()); print(jax.devices())"

# Force CPU for one command.
JAX_PLATFORMS=cpu vmec_jax input.nfp4_QH_warm_start

# Force an accelerator backend after installing GPU-enabled JAX.
JAX_PLATFORM_NAME=gpu vmec_jax input.nfp4_QH_warm_start

# For NVIDIA CUDA specifically, this is also valid.
JAX_PLATFORMS=cuda vmec_jax input.nfp4_QH_warm_start
```

From Python, leave `solver_device` unset to inherit JAX's default backend, or
pass `solver_device="cpu"` / `solver_device="gpu"` explicitly:

```python
import vmec_jax as vj

run_gpu = vj.run_fixed_boundary("input.nfp4_QH_warm_start", solver_device="gpu")
run_cpu = vj.run_fixed_boundary("input.nfp4_QH_warm_start", solver_device="cpu")
```

For GPU runs, vmec_jax defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false` before
JAX import so the allocator grows on demand. This avoids GPU memory contention
between optimization workers and was faster in the exact-Jacobian GPU profile.
Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` before import if you explicitly want
JAX's default preallocation behavior.

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

**Cold vs warm runtime**: the *cold* bar includes XLA JIT compilation on the first call (one-time cost per process); the *warm* bar is the steady-state solve time for subsequent calls in the same process. VMEC2000 has no compilation overhead, so it is always effectively cold. `vmec_jax` enables JAX's persistent compilation cache by default under `~/.cache/vmec_jax/jax_cache` so repeated cold-process runs can reuse compiled kernels; set `VMEC_JAX_COMPILATION_CACHE=0` to disable it or `VMEC_JAX_COMPILATION_CACHE_DIR=/path/to/cache` to choose a different location.

## Quasi-helical symmetry optimization (discrete-adjoint)

`examples/optimization/qh_fixed_resolution_jax.py` demonstrates an end-to-end
fixed-boundary QH optimization using the built-in **exact discrete-adjoint Jacobian**
— no finite differences, no SIMSOPT dependency.

The script is intentionally written in the same teaching style as SIMSOPT's
`QH_fixed_resolution.py`: choose the VMEC resolution directly in Python, choose
the active boundary coefficients directly, build the objective blocks directly
in the script, and choose the outer optimizer explicitly. Nothing relies on a
hidden SIMSOPT wrapper layer.

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

Key top-level controls in the script:

- `VMEC_MPOL`, `VMEC_NTOR`: solver resolution
- `MAX_MODE`: boundary parameterization richness
- `OBJECTIVE_TUPLES`: explicit aspect + QS residual blocks
- `METHOD`: `"gauss_newton"` or `"scipy"`
- `SCIPY_TR_SOLVER`: SciPy trust-region linear solver (`"lsmr"` by default for the QA/QH examples)
- `USE_MODE_CONTINUATION`: staged solves for higher-mode runs
- `USE_ESS`, `ALPHA`: optional exponential spectral scaling

When `max_mode` exceeds the modes present in the input file, vmec_jax automatically
extends the boundary to include the requested harmonics at zero amplitude
(`vj.extend_boundary_for_max_mode`), matching SIMSOPT's `fixed_range()` behaviour.
All runs use consistent VMEC resolution `mpol = ntor = 5` so the initial QS metric
is normalised identically across `max_mode` values.

| `max_mode` | DOFs | Policy | QS initial | QS final | Reduction | Objective final | Wall time ¹ |
|:----------:|:----:|:------:|:----------:|:--------:|:---------:|:---------------:|:-----------:|
| 1          |  8   | continuation, no ESS | 0.303 | 0.214 | 30 % | `0.216` | ~133 s |
| 2          | 24   | continuation, no ESS | 0.303 | `3.19e-3` | 99 % | `3.19e-3` | ~746 s |
| 3          | 48   | continuation + ESS | 0.303 | **`9.51e-4`** | **99.7 %** | **`9.51e-4`** | ~952 s |

¹ Wall time on Apple M-series (warm-cache subsequent runs are faster).

With only 8 DOFs (`max_mode=1`) the boundary deformation space is too limited
to reach a deep quasi-helical minimum. `max_mode=2` already gives a strong QH
solution, and the current `max_mode=3` continuation+ESS run improves it further
on the exact standalone path.

**vmec_jax vs SIMSOPT**: vmec_jax uses an exact discrete-adjoint Jacobian
(one batched JVP pass ≈ 1–2 forward solves regardless of DOF count) while
SIMSOPT + VMEC2000 uses finite differences (*n*_DOFs × 1 forward solve per
Jacobian).  For a detailed comparison of algorithms, runtimes, and memory,
see [docs/simsopt_comparison.rst](docs/simsopt_comparison.rst).

<table>
  <tr>
    <th align="center">max_mode = 1 &nbsp;(8 DOFs, 30 % QS reduction)</th>
    <th align="center">max_mode = 2 &nbsp;(24 DOFs, 99 % QS reduction)</th>
    <th align="center">max_mode = 3 &nbsp;(48 DOFs, 99.7 % QS reduction)</th>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode2/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode3/boundary_comparison.png" /></td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode2/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode3/bmag_surface.png" /></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/_static/figures/qh_opt/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qh_opt/mode2/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qh_opt/mode3/objective_history.png" /></td>
  </tr>
</table>

The |B| contour plots show quasi-helical alignment after optimization: contour lines
become increasingly helical (aligned with *m θ − n φ* = const). The ζ axis spans
one field period (0 → 2π/nfp).

The current exact standalone path keeps improving through `max_mode=3`, with
the 48-DOF continuation+ESS run reaching `~9.5e-4` total objective and QS.

Regenerate plots after running the optimization:

```bash
python examples/optimization/plot_qh_optimization_results.py --output-dir results/qh_opt
```

## Quasi-axisymmetric optimization (fixed-boundary)

`examples/optimization/qa_fixed_resolution_jax_ess.py` optimizes an nfp=2 QA
equilibrium for aspect ratio, mean iota, and QA symmetry residuals.

Like the QH script, it exposes the problem construction directly in Python:
VMEC resolution, active boundary DOFs, the three objective blocks, weights,
continuation policy, ESS settings, and the outer optimizer are all top-level
variables in the file.

```bash
python examples/optimization/qa_fixed_resolution_jax_ess.py   # MAX_MODE=2 by default
```

When `max_mode` exceeds the modes in the input file, vmec_jax automatically extends
the boundary to include those harmonics at zero amplitude (`vj.extend_boundary_for_max_mode`).
All runs use consistent VMEC resolution `mpol = ntor = 5`.
Objectives: aspect ratio (target 6.0) + mean iota (target 0.41) + QA symmetry residuals.
The current standalone QA path uses exact residuals + exact discrete-adjoint
Jacobians with `scipy.optimize.least_squares`. For `max_mode > 1`, the script
can use staged mode continuation: it solves the lower-mode QA problem first,
then lifts that solution into the richer boundary space before running the final
stage. The full sweep also tests direct-start and ESS variants.

| `max_mode` | DOFs | Policy | Eval used | Aspect final | Mean iota final | QS final | Objective final | Wall time ¹ |
|:----------:|:----:|:------:|:---------:|:------------:|:---------------:|:--------:|:---------------:|:-----------:|
| 1          |  8   | input deck, no ESS | 27 | 6.0024 | 0.3942 | `9.04e-3` | `9.29e-3` | ~315 s |
| 2          | 24   | continuation, no ESS | 52 | **6.0000** | 0.4095 | `1.46e-4` | `1.46e-4` | ~801 s |
| 3          | 48   | continuation, no ESS | 64 | **6.0000** | **0.4099** | **`7.61e-6`** | **`7.62e-6`** | ~1150 s |

¹ Wall time on Apple M-series.

On the latest fresh standalone rerun, staged continuation is decisive for QA.
Direct-start `max_mode=3` stays in a poor basin, while continuation reaches a
deep QA minimum; the best displayed QA run is the no-ESS `max_mode=3`
continuation case.

<table>
  <tr>
    <th align="center">max_mode = 1 &nbsp;(8 DOFs, exact SciPy + adjoint)</th>
    <th align="center">max_mode = 2 &nbsp;(24 DOFs, exact SciPy + adjoint, continuation)</th>
    <th align="center">max_mode = 3 &nbsp;(48 DOFs, exact SciPy + adjoint, continuation)</th>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qa_opt/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode2/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode3/boundary_comparison.png" /></td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qa_opt/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode2/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode3/bmag_surface.png" /></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/_static/figures/qa_opt/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qa_opt/mode2/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qa_opt/mode3/objective_history.png" /></td>
  </tr>
</table>

## QA/QH optimization policy sweep

The full benchmark panel below compares continuation versus direct-start mode
expansion, with and without ESS, for QA and QH at `max_mode = 1, 2, 3`.
Wall time is printed in each objective-history subplot. The renderer adds one
set of rows per available backend label, so CPU and GPU runs can be reported in
the same panel as soon as both result sets exist. By default, exact-optimizer
workers inherit the active JAX backend. Use `JAX_PLATFORMS=cpu` or
`--worker-jax-platforms cpu` only when you intentionally want CPU-only workers.

```bash
JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation
JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct
JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy continuation
JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy direct
python examples/optimization/render_qs_ess_publication_panel.py
```

<p align="center">
  <img src="docs/_static/figures/qs_ess_publication_panel_full.png" width="980" />
</p>

## Performance vs parity

- Default runs select the fastest stable path for each input automatically.
- Use `--parity` (or `performance_mode=False` in Python) to force the conservative VMEC2000 loop.
- Use `--solver-mode accelerated` to force the optimized fixed-boundary controller.
- For GPU benchmarking, compare both first-process and cache-warm timings; the first GPU process pays XLA compilation, while later processes reuse the persistent cache automatically.

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
