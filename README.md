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
  loop on CPU and GPU, matching the latest QH/QA/QI/LASYM profiling results.
- GPU exact-Jacobian replay uses the profiled dense-column chunking policy for
  larger fixed-boundary optimizations, reducing the observed QH mode-2 replay
  callback from about 42 s to about 18 s on the `office` RTX A4000 profile.
- Fixed-boundary profiling tools now report effective optimizer, solver, replay,
  and finish-budget settings so CPU/GPU regressions are easier to attribute.
- CI action versions were refreshed for the Node 24 runtime, and the PyPI
  release workflow still rejects tags that do not match `pyproject.toml`.

## Install

## From PyPI

```bash
pip install vmec-jax
```

QI optimization uses `booz_xform_jax` for the differentiable Boozer transform:

```bash
pip install "vmec-jax[qi]"
```

## From conda-forge

`vmec-jax` can be installed as a conda package from [conda-forge](https://github.com/conda-forge/vmec-jax-feedstock) into a particular project with [Pixi](https://pixi.prefix.dev/)

```
pixi add vmec-jax
```

or into a conda environment with [conda](https://docs.conda.io/projects/conda/)

```
conda install --channel conda-forge vmec-jax
```

## From source

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

For production fixed-boundary solves, the auto-selected CPU/GPU policy uses the
VMEC-control non-scan loop because it is faster for converged equilibria on the
current benchmark set. The scan loop remains available for explicit fast-mode
experiments with `use_scan=True` from Python or `--fast`/`--solver-mode
accelerated` from the CLI.

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

The fixed-boundary optimization examples solve VMEC equilibria and differentiate
the objective with the exact discrete-adjoint/tape path. The README only shows
one current best `LASYM = F` result for each target; the full CPU/GPU policy
matrix, LASYM panels, finite-beta examples, QI constraint sweep, and all tables
live in the
[optimization guide](docs/optimization.rst) and
[optimization sweep results](docs/optimization_sweep_results.rst).

Each row below shows the original deck LCFS before any `max_mode=1`
optimization work, the final LCFS, per-stage objective history, and the final
outer-surface `|B|` in Boozer coordinates computed with `booz_xform_jax`.
This sweep uses NFP=2 seeds for QA/QP/QI and the standard bundled NFP=4 warm
start for QH.  The current objective priority is primary symmetry/QI quality
and rotational-transform control.  QA follows the reference omnigenity QA deck
with aspect ratio near 5 and signed mean iota target 0.42; QH/QP/QI also use
`abs(mean_iota) >= 0.41`; QI now uses a higher aspect-ratio target of 10 to
make precise QI with acceptable mirror ratio and elongation less
overconstrained.  `LgradB` remains available as an optional script-level term,
but it is not active in the default README examples or best-row selection.

For stress-testing robustness from a common far-from-goal seed, the repository
also ships `examples/data/input.minimal_seed_nfp1` through
`examples/data/input.minimal_seed_nfp4`.  These are generated by
`vj.minimal_fixed_boundary_indata(nfp=...)` and contain only `RBC(0,0)`,
`RBC(0,1)`, and `ZBS(0,1)` as nonzero boundary coefficients; the optimization
policy, not the seed file, must introduce the QA/QH/QP/QI structure.

When the common minimal-seed lane uses deterministic target-helicity seeding,
keep that perturbation in the optimization setup rather than regenerating the
raw input decks: use tiny `1e-5` `RBC/ZBS` mode-1 hints, leave already
nonzero minimal modes unchanged, and let the active `max_mode` projection drop
any inactive hint modes.  The current deterministic hint set is
`RBC(1,0)`, `ZBS(1,0)`, `RBC(-1,1)`, `ZBS(-1,1)`, `RBC(1,1)`, and
`ZBS(1,1)` in VMEC input-index convention.
QA and QP common-seed production rows also use a per-run reference-family
preseed, without modifying the raw input decks: QA blends the active low-order
RBC/ZBS space 25% toward `input.nfp2_QA_omnigenity`, and QP blends 25% toward
`input.nfp2_QI`.  This is explicit provenance in `showcase_case.json` and is
used to escape the zero-transform branch before local exact optimization.

The bounded common-seed production stress test is documented in the
[optimization guide](docs/optimization.rst); the saved panel there is a
regression/stress artifact and should not be read as the best-row result.  In
that lane, QI cases are routed through `examples/optimization/qi_staged_runner.py`
into the staged `QI_optimization.py` policy; stale pre-dispatch QI rows under
`.../qi_nfp*/continuation/qp_preseed/...`, and QA/QP rows without reference
preseed metadata, should be regenerated before use.

The QP and QI rows both start from the bundled NFP=2 QI seed.  QP is a
quasi-poloidal-symmetry target using that same input deck; the current best QI
row uses the dedicated mirror-aware `QI_optimization.py` lane at `max_mode=3`
without a QP preseed.
The bundled NFP=2 seed is projected to each active `max_mode`, so
`max_mode=1` zeroes the seed's mode-2 boundary harmonics before optimizing.
For QI, the listed wall time includes all repeated stages using the same
constrained least-squares residual definition.

| Target | Backend | Policy | max_mode | ESS | QP preseed | Final J | QI legacy | Mirror | Elong. | Aspect | Iota | Wall time |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| QA | CPU | continuation | 3 | yes |  | 2.33e-04 |  |  |  | 5.000 | 0.4200 | 6.3 min |
| QH | CPU | continuation | 3 | yes |  | 9.68e-03 |  |  |  | 4.999 | -1.6595 | 4.0 min |
| QP | CPU | continuation | 3 | no |  | 6.76e-02 |  |  |  | 5.019 | -0.6255 | 3.7 min |
| QI | CPU | qi_default | 3 | yes | no | 1.17e-02 | 3.09e-04 | 0.225 | 6.43 | 9.999 | -0.5043 | 10.1 min |

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

The dedicated `QI_optimization.py` coverage figure tracks the two bundled QI
inputs used by the README/docs lane.  It is rendered from existing reviewed
outputs and uses Boozer `|B|` line contours only.  The objective panel
concatenates all recorded stages as best-so-far, stage-normalized objectives,
with dashed separators where the objective definition or weights changed.  For
the seed-3127 lane, the inset is a boundary-reference interpolation scan, not
an optimizer trajectory.

| QI input | Output/provenance | Final J | QI smooth | QI legacy | Mirror | Elong. | Aspect | Iota | CPU time |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `examples/data/input.nfp2_QI` | `results/qi_opt/ess/nfp2_qi` | 1.17e-02 | 1.13e-03 | 3.09e-04 | 0.225 / 0.30 | 6.43 / 8.2 | 9.999 / 10.0 | -0.5043 | 14.7 min |
| `examples/data/input.QI_stel_seed_3127` | `results/qi_opt/ess/qi_stel_seed_3127_current_public_final` | 1.12e-01 | 4.32e-03 | 1.16e-03 | 0.316 / 0.35 | 3.91 / 8.0 | 3.465 / 4.0 | -1.0366 | 6.3 min |

<p align="center">
  <img src="docs/_static/figures/readme_qi_optimization_cases.png" width="980" />
</p>

Recreate the four displayed runs:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qa --modes 3 --ess on
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qh --modes 3 --ess on
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qp --modes 3 --ess off
PYTHONPATH=. JAX_PLATFORMS=cpu VMEC_JAX_QI_RUN_CASE=nfp2_qi python examples/optimization/QI_optimization.py
PYTHONPATH=. python examples/optimization/render_readme_best_optimizations.py
PYTHONPATH=. python examples/optimization/render_qi_readme_cases.py
```

The sweep driver skips existing successful `case_result.json` rows by default;
append `--rerun` to a sweep command when you need to overwrite local artifacts
and reproduce a row from scratch.
The constrained-QI matrix is a separate sweep artifact; regenerate it with the
QI preseed/no-preseed commands in `docs/optimization_sweep_results.rst` before
running `render_qi_constrained_sweep.py`.

For QI seed-robustness probes, set `VMEC_JAX_QI_RUN_CASE=qi_stel_seed_3127`
when running `examples/optimization/QI_optimization.py`, or change the
top-level `RUN_CASE` to `nfp1_qi`, `nfp2_qi`, `qi_stel_seed_3127`,
`nfp4_qh_warm_to_qi`, or a new `QI_CASES` entry for another VMEC input deck.
The NFP=4 QH-warm case is currently a diagnostic stress test: it exercises the
same machinery, but it is not yet a validated route to a precise NFP=4 QI state.
Before promoting such a result,
run
`examples/optimization/audit_qi_seed_suitability.py --quick` and check the
legacy QI, smooth QI, mirror ratio, elongation, iota, and Boozer `|B|`
line-contour diagnostics. For the `qi_stel_seed_3127` far-seed lane, use the
same gates as the optimization case: `--smooth-qi-max 5e-3 --legacy-qi-max 2e-3`.
Use the prefine manifest path for reviewed expensive probes rather than
launching ad hoc far-seed jobs.

The `input.QI_stel_seed_3127` robustness lane is intentionally harder than the
default NFP=2 QI seed.  Purely local boundary moves still get trapped, but the
current `QI_optimization.py` case now includes a deterministic same-NFP
reference-family preconditioner: it interpolates the seed boundary toward the
bundled NFP=3 QI reference, audits each candidate with the independent
smooth/legacy QI, mirror, elongation, aspect, and iota gates, and then starts
local QI cleanup from the lowest-mirror accepted non-endpoint candidate when
one exists.
That candidate is recorded as the accepted baseline, so later cleanup stages
cannot replace it unless exact diagnostics improve.
For this far-seed case the legacy Goodman-style QI gate is `2e-3`, while
the smooth differentiable proxy gate is `5e-3` because it is the optimization
surrogate and is more conservative on the six-surface audit.
Guarded local cleanup can now use anisotropic boundary stages, for example
unlocking `max_m=1, max_n=4` before the full `max_m=max_n=4` boundary. The
script promotes such stages only when independent exact diagnostics improve,
so a mirror-heavy local solve cannot replace a precise-QI baseline if it
damages legacy QI.
The diagnostic below scans two boundary coefficients around the raw seed and
shows why this larger global-to-local move is needed.

<p align="center">
  <img src="docs/_static/figures/qi_seed3127_landscape_rc01_zs01.png" width="980" />
</p>

Recreate that landscape plot:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python tools/diagnostics/qi_landscape_scan.py \
  --input examples/data/input.QI_stel_seed_3127 \
  --output-dir results/diagnostics/qi_landscape_seed3127 \
  --max-mode 3 --min-vmec-mode 6 --dofs rc01,zs01 --points 3 \
  --span 0.03 --span2 0.03 --surfaces 0.35,0.65 \
  --nphi 51 --nalpha 11 --n-bounce 15 \
  --mirror-ntheta 32 --mirror-nphi 32 \
  --elongation-ntheta 24 --elongation-nphi 8
```

The landscape command defaults to trial solves for speed. Add `--exact-solve`
before using the scanned QI, mirror, elongation, or iota values as promotion
evidence.

Run the current reference-family preconditioner directly:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu VMEC_JAX_QI_RUN_CASE=qi_stel_seed_3127 \
  python examples/optimization/QI_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python tools/diagnostics/qi_boundary_interpolation_scan.py \
  --seed-input examples/data/input.QI_stel_seed_3127 \
  --reference-input examples/data/input.nfp3_QI_fixed_resolution_final \
  --out-root results/diagnostics/qi_seed3127_boundary_interpolation \
  --lambdas 0.994,0.995,0.996,0.997,0.998,0.999,1.0,1.001,1.002 \
  --max-mode 4 --max-iter 80 --target-aspect 4.0 \
  --surfaces 0.1,0.28,0.46,0.64,0.82,1.0 \
  --mboz 18 --nboz 18 --nphi 151 --nalpha 31 --n-bounce 51 \
  --smooth-qi-max 5e-3 --legacy-qi-max 2e-3 \
  --max-mirror-ratio 0.35 --max-elongation 8.0
```

Regenerate the README panels and the compact CSV used for the table:

```bash
PYTHONPATH=. python examples/optimization/render_readme_best_optimizations.py
```

## Performance vs parity

- Default runs select the fastest stable path for each input automatically.
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
