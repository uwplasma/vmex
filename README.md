# vmec-jax

[![PyPI version](https://img.shields.io/pypi/v/vmec-jax.svg)](https://pypi.org/project/vmec-jax/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/vmec-jax.svg)](https://github.com/conda-forge/vmec-jax-feedstock)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/uwplasma/vmec_jax/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/vmec_jax)](https://github.com/uwplasma/vmec_jax/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/vmec_jax/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/vmec_jax/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/vmec_jax/graph/badge.svg?branch=main)](https://codecov.io/gh/uwplasma/vmec_jax?branch=main)
[![Docs](https://img.shields.io/readthedocs/vmec-jax/latest?label=docs)](https://vmec-jax.readthedocs.io/en/latest/)
[![PyPI downloads](https://img.shields.io/pypi/dm/vmec-jax)](https://pypi.org/project/vmec-jax/)

End-to-end differentiable JAX implementation of **VMEC2000** for fixed-boundary
and free-boundary ideal-MHD equilibria.

## Release notes

### v0.0.9

- Default fixed-boundary production solves now use the VMEC-control non-scan
  loop on CPU and GPU, matching the profiled QH/QA/QI paths and the available
  partial LASYM profiling lanes.
- GPU exact-Jacobian replay uses the profiled dense-column chunking policy for
  larger fixed-boundary optimizations, reducing the observed QH mode-2 replay
  callback from about 42 s to about 18 s on the `office` RTX A4000 profile.
- Fixed-boundary profiling tools now report effective optimizer, solver, replay,
  and finish-budget settings so CPU/GPU regressions are easier to attribute.
- CI action versions were refreshed for the Node 24 runtime, and the PyPI
  release workflow still rejects tags that do not match `pyproject.toml`.

## Install

### From PyPI

```bash
pip install vmec-jax
```

QI optimization uses `booz_xform_jax` for the differentiable Boozer transform:

```bash
pip install "vmec-jax[qi]"
```

### From conda-forge

`vmec-jax` can be installed as a conda package from [conda-forge](https://github.com/conda-forge/vmec-jax-feedstock) into a particular project with [Pixi](https://pixi.prefix.dev/)

```
pixi add vmec-jax
```

or into a conda environment with [conda](https://docs.conda.io/projects/conda/)

```
conda install --channel conda-forge vmec-jax
```

### From source

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

For optimization sweep commands, `--backend-label` is only the output
directory/table label.  Select the actual JAX process backend with
`JAX_PLATFORMS=cpu`, `JAX_PLATFORM_NAME=gpu`, or `JAX_PLATFORMS=cuda`, and use
`--solver-device cpu` / `--solver-device gpu` when a worker should force the
solver device.  Trust the recorded `jax_backend`, `jax_device_kind`,
`solver_device`, and `jax_platforms` fields in `case_result.json` or generated
CSV files over the output directory name.

For production fixed-boundary solves, the default policy selects the profiled
VMEC-control non-scan path.  This is the most stable public path in the current
validation matrix, not a guarantee that every input is runtime-optimal.  The
scan loop remains available for explicit fast-mode experiments with
`use_scan=True` from Python or `--fast`/`--solver-mode accelerated` from the CLI.

For GPU runs, vmec_jax defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false` before
JAX import so the allocator grows on demand. This avoids GPU memory contention
between optimization workers and was faster in the exact-Jacobian GPU profile.
Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` before import if you explicitly want
JAX's default preallocation behavior.

`vmec_jax` enables JAX's persistent compilation cache automatically for
accelerator-selected runs, including runs where `CUDA_VISIBLE_DEVICES` or the
ROCm equivalents expose an accelerator before import. CPU cache use is explicit
opt-in because XLA:CPU AOT cache hits can emit host-feature mismatch errors on
some JAX versions. Set
`VMEC_JAX_COMPILATION_CACHE=1` to enable the default cache for CPU runs, set
`VMEC_JAX_COMPILATION_CACHE=0` to disable it, or set
`VMEC_JAX_COMPILATION_CACHE_DIR=/path/to/cache` to choose a custom location.
The default cache path is scoped by machine, CPU features, Python version, and
JAX/JAXLIB versions.

For the current small/medium fixed-boundary examples, CPU is often faster after
JIT warmup. GPU support is production-enabled and useful to profile, but the
exact optimizer defaults accepted-point Jacobians to the discrete-adjoint tape
path on both CPU and GPU. The scan exact path is an explicit diagnostic override
via `VMEC_JAX_OPT_EXACT_PATH=scan`; relaxed trial residuals use the scan forward
path by default. See the performance guide for current CPU/GPU timings and
profiling commands.

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

**Cold vs warm runtime**: the *cold* bar includes XLA JIT compilation on the first call (one-time cost per process); the *warm* bar is the steady-state solve time for subsequent calls in the same process. VMEC2000 has no compilation overhead, so it is always effectively cold. `vmec_jax` uses JAX's persistent compilation cache automatically for accelerator-selected runs under `~/.cache/vmec_jax/jax_cache/<machine-fingerprint>`. CPU cache use is opt-in with `VMEC_JAX_COMPILATION_CACHE=1` to avoid XLA:CPU AOT host-feature mismatch warnings on some JAX versions.

The current fixed-boundary CPU matrix is intentionally shown as a reality
check: warm `vmec_jax` beats VMEC2000 on 1 of 16 bundled fixed-boundary rows
(`circular_tokamak_aspect_100`, 1.33x), while the median warm single-solve row
is still about 4.4x slower than VMEC2000 on this host.  The exact-adjoint
optimization path can still win at the workflow level because it avoids
finite-difference VMEC subprocess columns, but single-solve CPU speed remains
an open performance lane.  The plotted rows are exported in
`docs/_static/figures/readme_runtime_compare.csv` and `.json`.

## Best Stellarator-Symmetric Optimizations

The fixed-boundary examples solve VMEC equilibria and differentiate the
objective with the exact discrete-adjoint/tape path. The README shows one
current `LASYM = F` result per target at aspect ratio near 5; the full CPU/GPU
matrix, partial LASYM panels, finite-beta examples, QI robustness notes, and
detailed tables live in the [optimization guide](docs/optimization.rst) and
[optimization sweep results](docs/optimization_sweep_results.rst).

Each panel shows the original deck LCFS, final LCFS, per-stage objective
history, and initial/final outer-surface `|B|` line contours in Boozer
coordinates computed with `booz_xform_jax`. The QI best row starts from the
bundled NFP=2 QI seed, which is already QI-like; the next section shows a
farther seed-3127 robustness lane.

| Target | Backend | Policy | max_mode | ESS | QP preseed | Final J | QI legacy | Mirror | Elong. | Aspect | Iota | Wall time |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| QA | CPU | continuation | 3 | yes |  | 2.33e-04 |  |  |  | 5.000 | 0.4200 | 6.3 min |
| QH | CPU | continuation | 3 | yes |  | 9.68e-03 |  |  |  | 4.999 | -1.6595 | 4.0 min |
| QP | CPU | continuation | 3 | no |  | 6.76e-02 |  |  |  | 5.019 | -0.6255 | 3.7 min |
| QI | CPU | continuation | 3 | yes | no | 2.17e-03 | 2.17e-03 | 0.211 | 4.30 | 5.001 | -0.5494 | 11.3 min |

<p align="center">
  <img src="docs/_static/figures/readme_best_optimization_qa.png" width="980" />
</p>

<p align="center">
  <img src="docs/_static/figures/readme_best_optimization_qh.png" width="980" />
</p>

<p align="center">
  <img src="docs/_static/figures/readme_best_optimization_qp.png" width="980" />
</p>

<p align="center">
  <img src="docs/_static/figures/readme_best_optimization_qi.png" width="980" />
</p>

Recreate these four panels:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qa --modes 3 --ess on --rerun
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qh --modes 3 --ess on --rerun
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qp --modes 3 --ess off --rerun
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qi --modes 3 --ess on --qi-qp-preseed off --rerun
PYTHONPATH=. python examples/optimization/render_readme_best_optimizations.py
```

## Optimization from Different Initial Conditions

The repository also tracks stress tests from less tailored seeds. These are
failure-revealing regression artifacts, not seed-robustness evidence. In the
current checked-in minimal-seed summary, only `qh_nfp4` is `status=ok`; QI rows
are partial timeout records and QP is incomplete. The dedicated QI coverage
figure below includes the `input.QI_stel_seed_3127` far-seed lane; the docs
contain the longer discussion, gates, and landscape diagnostics.
Common minimal-seed QA/QH/QP/QI runs start from
`examples/data/input.minimal_seed_nfp*`, which contain only `RBC(0,0)`,
`RBC(0,1)`, and `ZBS(0,1)` before optimization-time helicity hints are added.
Those common-minimal rows are a failure-revealing regression lane; inspect the
generated `status`, `success`, `crashed`, and `message` columns before using
them as optimization evidence.

<p align="center">
  <img src="docs/_static/figures/readme_qi_optimization_cases.png" width="980" />
</p>

Recreate the seed-stress artifacts:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu VMEC_JAX_QI_RUN_CASE=qi_stel_seed_3127 \
  VMEC_JAX_QI_OUTPUT_DIR=results/qi_opt/ess/qi_stel_seed_3127_current_public_final \
  python examples/optimization/QI_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_minimal_seed_showcase.py --cases all --backend-label cpu --solver-device cpu --worker-jax-platforms cpu --policy continuation --max-mode 3 --ess on --max-nfev 30 --continuation-nfev 20 --inner-max-iter 120 --trial-max-iter 120 --inner-ftol 1e-9 --trial-ftol 1e-9 --case-timeout-s 1200 --rerun
PYTHONPATH=. python examples/optimization/render_qi_readme_cases.py
PYTHONPATH=. python examples/optimization/render_minimal_seed_showcase.py
```

## Performance vs parity

- Default runs select the profiled stable path; they do not guarantee the
  fastest backend or iteration policy for every input.
- Use `--parity` (or `performance_mode=False` in Python) to force the conservative VMEC2000 loop.
- Use `--solver-mode accelerated` to force the optimized fixed-boundary controller.
- For GPU benchmarking, separate raw solver throughput from public policy overhead. For example, use `tools/diagnostics/profile_fixed_boundary.py --no-auto-cli-policy --solver-mode accelerated --no-multigrid --use-scan --solver-device gpu`.
- Compare both first-process and in-process warm timings. The first GPU process pays XLA/runtime setup; persistent cache effectiveness depends on the JAX version, backend, and machine features.

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
