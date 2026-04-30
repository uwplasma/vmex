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

QI optimization uses `booz_xform_jax` for the differentiable Boozer transform:

```bash
pip install "vmec-jax[qi]"
```

Developer (editable) install:

```bash
git clone https://github.com/uwplasma/vmec_jax
pip install -e "vmec_jax[qi]"
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

`vmec_jax` enables JAX's persistent compilation cache by default, but its
default cache path is machine/CPU-feature scoped to avoid reusing CPU AOT
executables compiled on a different host. Set `VMEC_JAX_COMPILATION_CACHE=0` to
disable the persistent cache or `VMEC_JAX_COMPILATION_CACHE_DIR=/path/to/cache`
to choose a custom location.

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

**Cold vs warm runtime**: the *cold* bar includes XLA JIT compilation on the first call (one-time cost per process); the *warm* bar is the steady-state solve time for subsequent calls in the same process. VMEC2000 has no compilation overhead, so it is always effectively cold. `vmec_jax` enables JAX's persistent compilation cache by default under `~/.cache/vmec_jax/jax_cache/<machine-fingerprint>` so repeated cold-process runs on the same host can reuse compiled kernels without sharing CPU AOT executables across incompatible machines; set `VMEC_JAX_COMPILATION_CACHE=0` to disable it or `VMEC_JAX_COMPILATION_CACHE_DIR=/path/to/cache` to choose a different location.

## Optimization Internals

The fixed-boundary optimization examples expose the problem construction
directly in Python: VMEC resolution, active boundary coefficients, objective
blocks, weights, continuation policy, ESS scaling, and the outer optimizer are
all top-level variables in the scripts.  No SIMSOPT wrapper layer is required.

For a boundary parameter vector `x`, vmec_jax solves the VMEC residual
`F(y, x) = 0` for the equilibrium state `y`, then differentiates objective
residuals `r(y(x), x)` with the exact discrete-adjoint/tape path.  Instead of
finite-differencing each boundary DOF, vmec_jax records a checkpoint tape of
the nonlinear VMEC iteration and replays it with JAX JVP/VJP rules.  The dense
least-squares Jacobian used by the examples is exact to machine precision and
has cost comparable to a small number of forward solves, not one VMEC solve per
boundary DOF.

Details: [discrete adjoint](docs/discrete_adjoint.rst),
[optimization guide](docs/optimization.rst), and
[SIMSOPT comparison](docs/simsopt_comparison.rst).

## Individual Fixed-Boundary Optimization Scripts

The standalone examples teach the same workflow as the SIMSOPT examples without
depending on SIMSOPT: choose VMEC resolution, active boundary coefficients,
objective terms, weights, continuation policy, ESS scaling, and optimizer
directly in Python.  They are intentionally plain scripts with top-level
variables, not argparse wrappers.

```bash
python examples/optimization/qa_fixed_resolution_jax_ess.py
python examples/optimization/qh_fixed_resolution_jax.py
python examples/optimization/qp_fixed_resolution_jax_ess.py
python examples/optimization/qi_fixed_resolution_jax_ess.py
```

Key top-level controls are `VMEC_MPOL`, `VMEC_NTOR`, `MAX_MODE`,
`OBJECTIVES`, `METHOD`, `USE_MODE_CONTINUATION`, `USE_ESS`, and `ALPHA`.
When `MAX_MODE` exceeds the modes present in the input file, vmec_jax extends
the boundary with `vj.extend_boundary_for_max_mode`, matching SIMSOPT's
`fixed_range()` workflow.

Add a new target by appending an objective term:

```python
OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    abs_mean_iota_floor_objective(TARGET_ABS_IOTA_MIN, IOTA_WEIGHT),
    quasisymmetry_objective(helicity_m=1, helicity_n=-1, surfaces=SURFACES, weight=QS_WEIGHT),
    ObjectiveTerm("custom", lambda ctx, state: your_metric(ctx, state), target=0.0, weight=0.1),
]
```

For current QA/QH/QP/QI results, use the policy sweep below. It is the only
optimization benchmark table shown in the README so stale single-script results
do not drift from the generated sweep artifacts.

## Finite-beta Stage-one Optimization

The finite-beta examples reproduce the VMEC-only stage-one finite-beta
workflows without SIMSOPT and without coils.  They use bundled finite-pressure
and current-driven input decks and add differentiable residuals for aspect
ratio, iota bounds, volume-averaged field proxy, total beta, plus the field
quality objective (QA/QH quasisymmetry or QI).

```bash
python examples/optimization/qa_optimization_finite_beta.py
python examples/optimization/qh_optimization_finite_beta.py
python examples/optimization/qi_optimization_finite_beta.py
```

The scripts save `input.initial`, `input.final`, `wout_initial.nc`,
`wout_final.nc`, and `history.json`.  Full differentiable Mercier `DMerc` and
Redl bootstrap-current mismatch residuals are the next finite-beta extensions;
the current examples keep the stage-one structure and current-profile support
in place so those terms can be added without changing the user workflow.
The QI script exposes `QI_MBOZ`, `QI_NBOZ`, `QI_NPHI`, `QI_NALPHA`, and
`QI_N_BOUNCE` at the top; the defaults are first-run diagnostic settings, and
should be increased for final research-quality QI refinements.

## QA/QH/QP/QI Optimization Policy Sweep

The panel below compares the exact standalone optimizer on CPU and GPU for
four targets: QA, QH, QP, and QI. It includes the complete
stellarator-symmetric matrix and the currently available partial LASYM lanes.
Columns increase the boundary space from `max_mode = 1` to `max_mode = 4`.
Rows compare staged mode continuation against direct-start mode expansion.
Blue curves use unscaled boundary DOFs; orange curves use ESS with
`alpha = 2.5`. Solid lines met the optimizer success criterion; dashed lines
mark stopped, failed, or budgeted lanes. Timeout/OOM details are recorded in
the summary tables.

<p align="center">
  <img src="docs/_static/figures/qs_ess_objective_panel_all_policies.png" width="980" />
</p>

The QA input includes `1e-5` seeds for the mode-1 boundary terms so the iota
residual has a useful direction. With the corrected bounded solve budgets, QA
continuation reaches the target-iota basin on both CPU and GPU. Direct QA with
ESS also reaches `iota ~= 0.409`; direct QA without ESS now leaves the zero-iota
branch for modes 2 and 3, but direct high-mode starts remain weak for mode 4.

QH and QP use the quasisymmetry residual with different helicities. QI uses
`vmec_jax.quasi_isodynamic`, a smooth Boozer-space residual built through
`booz_xform_jax`. The QI rows first run a same-mode QP preseed and then refine
with the QI residual; this avoids the QH warm-start basin and gives visibly
non-QH `|B|` contours while keeping the objective differentiable.

Final CPU states for the continuation and direct-start policies are shown
below. The `|B|` panels use line contours on the LCFS, with a separate colorbar
for each panel because the field ranges are not identical across aspect-ratio
changes.

<p align="center">
  <img src="docs/_static/figures/qs_ess_final_state_atlas_continuation.png" width="980" />
</p>

<p align="center">
  <img src="docs/_static/figures/qs_ess_final_state_atlas_direct.png" width="980" />
</p>

Recreate the full CPU/GPU sweep:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both
PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both
PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both
```

The default per-case timeout is 1200 s. GPU sweeps use exact/replay callbacks
with calibrated optimizer budgets (`inner_max_iter = trial_max_iter = 180`,
`ftol = trial_ftol = 1e-9` for deck-controlled QA/QH cases) so production
sweeps have enough room to converge high-mode/LASYM cases while still bounding
runaway rows. Add `--diagnostic-budgets` only when you explicitly want bounded
quick-look GPU diagnostics, and use `--case-timeout-s 0` only for an unbounded
local diagnostic run.

Recreate just the CPU direct-start rows:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both
```

Render the README/docs panels and tables:

```bash
PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py
```

To run the non-stellarator-symmetric matrix, append
`--stellarator-asymmetric`. This sets `LASYM = T` in memory, optimizes
`RBC/ZBS/RBS/ZBC`, seeds zero asymmetric `RBS/ZBC` modes with `1e-7`, and
writes separate LASYM outputs under `results/qs_ess_sweep/<backend>/asymmetric/`.
The renderer then creates additional `*_asymmetric_*` objective, atlas, summary,
and publication panels.

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both --stellarator-asymmetric
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both --stellarator-asymmetric
PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both --stellarator-asymmetric
PYTHONPATH=. JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3,4 --ess both --stellarator-asymmetric
PYTHONPATH=. python examples/optimization/render_qs_ess_publication_panel.py
```

The LASYM panels are published as a partial 1200 s snapshot. This is useful
because the failures are informative: current mode-4 GPU LASYM lanes include
timeout and GPU-memory limits in the exact tangent replay path. In the frozen
snapshot used here, the partial LASYM table contains 13 CPU rows and 61 GPU
rows. The CPU subset has 6 successful rows, 6 crashed rows, and 1 budgeted
stop; the GPU subset has 19 successful rows, 10 crashed rows, and 32 budgeted
stops.

<p align="center">
  <img src="docs/_static/figures/qs_ess_objective_panel_asymmetric_all_policies.png" width="980" />
</p>

<p align="center">
  <img src="docs/_static/figures/qs_ess_summary_tables_asymmetric_all_policies.png" width="980" />
</p>

For NVIDIA-only JAX installations, `JAX_PLATFORMS=cuda` is also valid. Do not
use `JAX_PLATFORMS=gpu`: some JAX versions interpret that as both CUDA and ROCm
and fail if ROCm is not installed.

Run individual examples by editing top-level variables in each script:

```bash
python examples/optimization/qa_fixed_resolution_jax_ess.py
python examples/optimization/qh_fixed_resolution_jax.py
python examples/optimization/qp_fixed_resolution_jax_ess.py
python examples/optimization/qi_fixed_resolution_jax_ess.py
```

More figures, CSV/JSON summaries, and reproduction notes are in
[docs/optimization_sweep_results.rst](docs/optimization_sweep_results.rst).

Best row per backend/problem/policy in the plotted symmetric sweep:

| Backend | Problem | Policy | Best max_mode | ESS | Status | Final J | Aspect | Iota | nfev | Wall min |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| CPU | QA | continuation | 4 | yes | stopped | 2.84e-05 | 5.999 | 0.4100 | 79 | 19.8 |
| CPU | QA | direct | 3 | yes | ok | 3.13e-05 | 6.000 | 0.4102 | 51 | 19.3 |
| CPU | QH | continuation | 4 | yes | ok | 5.87e-04 | 7.000 | -1.2182 | 46 | 18.6 |
| CPU | QH | direct | 3 | yes | ok | 3.27e-03 | 6.999 | - | 20 | 9.2 |
| CPU | QP | continuation | 4 | no | ok | 3.65e-02 | 7.002 | -0.4218 | 51 | 5.0 |
| CPU | QP | direct | 2 | yes | ok | 3.74e-02 | 7.004 | -0.4037 | 30 | 1.1 |
| CPU | QI | continuation | 4 | no | ok | 5.20e-03 | 7.002 | -0.4148 | 80 | 4.5 |
| CPU | QI | direct | 2 | yes | ok | 4.90e-03 | 7.001 | -0.5808 | 31 | 1.4 |
| GPU | QA | continuation | 4 | no | ok | 9.74e-05 | 6.000 | 0.4100 | 94 | 19.4 |
| GPU | QA | direct | 4 | yes | stopped | 6.77e-05 | 6.000 | 0.4100 | 60 | 15.1 |
| GPU | QH | continuation | 4 | yes | ok | 9.38e-04 | 7.000 | -1.2528 | 91 | 19.5 |
| GPU | QH | direct | 4 | yes | ok | 1.30e-03 | 7.000 | -1.2280 | 20 | 5.0 |
| GPU | QP | continuation | 4 | no | ok | 6.46e-02 | 7.011 | -0.4012 | 87 | 14.8 |
| GPU | QP | direct | 2 | yes | ok | 3.72e-02 | 7.005 | -0.4028 | 30 | 4.4 |
| GPU | QI | continuation | 4 | no | ok | 1.66e-03 | 7.000 | -0.4096 | 93 | 17.1 |
| GPU | QI | direct | 3 | yes | ok | 2.72e-03 | 6.999 | -0.4025 | 45 | 7.4 |

The generated CSV includes the complete 128-row symmetric CPU/GPU table plus
the current partial LASYM rows. Filter `stellarator_asymmetric=False` for the
symmetric benchmark subset:
[`docs/_static/figures/qs_ess_summary_all.csv`](docs/_static/figures/qs_ess_summary_all.csv).

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
