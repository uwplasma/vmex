# vmec-jax

[![PyPI version](https://img.shields.io/pypi/v/vmec-jax.svg)](https://pypi.org/project/vmec-jax/)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/vmec-jax.svg)](https://github.com/conda-forge/vmec-jax-feedstock)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/uwplasma/vmec_jax/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/vmec_jax)](https://github.com/uwplasma/vmec_jax/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/vmec_jax/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/vmec_jax/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/vmec_jax/graph/badge.svg?branch=main)](https://codecov.io/gh/uwplasma/vmec_jax?branch=main)
[![Docs](https://img.shields.io/readthedocs/vmec-jax/latest?label=docs)](https://vmec-jax.readthedocs.io/en/latest/)
[![PyPI downloads](https://img.shields.io/pypi/dm/vmec-jax)](https://pypi.org/project/vmec-jax/)

JAX implementation of **VMEC2000** for fixed-boundary and VMEC-compatible
free-boundary ideal-MHD equilibria. Supported promoted paths are differentiable;
full free-boundary vacuum/NESTOR adjoints remain a research lane.

## Install

From PyPI:

```bash
pip install vmec-jax
```

PyPI and conda-forge can lag the repository tags. If you need an exact release,
check the package-index version before installing or pinning it.

The plain package includes plotting support (`matplotlib`) and the
differentiable Boozer transform dependency (`booz_xform_jax`) used by the QI
examples, so there is no separate plotting or QI extra to install.

From conda-forge use `pixi add vmec-jax` or
`conda install --channel conda-forge vmec-jax`. For development, clone the
repository and run `pip install -e .`.

## Quick Start

Run the solver with the VMEC2000-style CLI:

```bash
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/input.nfp4_QH_warm_start
vmec_jax input.nfp4_QH_warm_start
```

Plot any `wout_*.nc` file:

```bash
vmec_jax --plot wout_nfp4_QH_warm_start.nc
vmec_jax --plot wout_nfp4_QH_warm_start.nc --outdir figures/
```

Use the Python API:

```python
import vmec_jax as vj

fixed = vj.run_fixed_boundary("input.nfp4_QH_warm_start")
vj.plot_wout("wout_nfp4_QH_warm_start.nc", outdir="figures/")
```

For the bundled small free-boundary example, download both the input deck and
its magnetic grid into the same folder:

```bash
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/input.cth_like_free_bdy_lasym_small
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/mgrid_cth_like_lasym_small.nc
vmec_jax input.cth_like_free_bdy_lasym_small
```

## Single-Stage Free-Boundary Coil Optimization

The research branch `feature/freeb-essos-coil-single-stage` adds the first
coil-aware free-boundary lane. The compatibility path still uses VMEC-style
`mgrid` files; the new research path samples the external field directly from
differentiable JAX Biot-Savart coils so coil currents and Fourier coefficients
can be optimization variables:

```text
coil Fourier dofs/currents -> direct Biot-Savart field -> free-boundary VMEC
-> wout/proxy diagnostics -> coil-only objective update
```

Current status: direct-coil finite-pressure coupling is validated at low
resolution against generated-`mgrid` backends, while VMEC2000 WOUT promotion and
Boozer/QS full-loop gradients remain optional/phase-2 promotion work.

![Direct-coil free-boundary architecture](docs/_static/figures/freeb_single_stage_architecture.png)<br>
![Finite-pressure beta scan](docs/_static/figures/freeb_single_stage_beta_scan.png)<br>
![Direct-coil provider parity](docs/_static/figures/freeb_single_stage_provider_parity.png)<br>
![Direct-coil CPU/GPU benchmark matrix](docs/_static/figures/freeb_single_stage_benchmark_matrix.png)

Run the low-resolution direct-coil/generated-`mgrid` scan:

```bash
export ESSOS_ROOT=/path/to/ESSOS_mgrid_pr
export ESSOS_INPUT_DIR=$ESSOS_ROOT/examples/input_files
PYTHONPATH=.:$ESSOS_ROOT:$PYTHONPATH \
  python examples/free_boundary_essos_coils_beta_scan.py \
  --outdir results/free_boundary_essos_coils_beta_scan_readme \
  --activate-fsq 1e99
```

Run the bounded provider/solve/gradient benchmark matrix before rendering the
README figures:

```bash
python tools/benchmarks/bench_freeb_direct_coil_matrix.py \
  --quick \
  --out results/bench_freeb_direct_coil_matrix/summary.json
```

Then render the figures:

```bash
python tools/diagnostics/render_freeb_single_stage_readme.py \
  --summary results/free_boundary_essos_coils_beta_scan_readme/summary.json \
  --benchmark-summary results/bench_freeb_direct_coil_matrix/summary.json \
  --outdir docs/_static/figures
```

Run the phase-1 coil-only validation example without ESSOS assets:

```bash
python examples/optimization/free_boundary_QS_coil_optimization.py \
  --smoke --provider circle --max-evals 1 --max-iter 1 --vmec-max-iter 1 \
  --pressure-scale 100 --activate-fsq 1e99 \
  --outdir results/free_boundary_QS_coil_optimization_circle_smoke
```

Detailed caveats and optional VMEC2000 diagnostics are in
`docs/free_boundary_coil_optimization.rst`.

## Backend Selection

`vmec_jax` follows the selected JAX backend. If CPU-only JAX is installed, runs
use CPU. If GPU-enabled JAX is installed and selected, runs use the accelerator;
`vmec_jax` does not silently force those runs back to CPU.

```bash
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
JAX_PLATFORMS=cpu vmec_jax input.nfp4_QH_warm_start
JAX_PLATFORM_NAME=gpu vmec_jax input.nfp4_QH_warm_start
JAX_PLATFORMS=cuda vmec_jax input.nfp4_QH_warm_start
```

From Python, leave `solver_device` unset to inherit JAX's default backend, or
pass `solver_device="cpu"` / `solver_device="gpu"` explicitly.

## Optimization Examples

Editable optimization examples live in `examples/optimization/`. Start with
`examples/optimization/README.md` for workflow anatomy, then use
`docs/optimization.rst` for the full method guide,
`docs/optimization_sweep_results.rst` for generated sweep tables/figures, and
`docs/piecewise_omnigenous_plan.rst` for the pwO planning and acceptance gates.

The README keeps only compact best-current stellarator-symmetric QA/QH/QP/QI
rows. Extended LASYM, finite-beta, QI provenance, minimal-seed, CPU/GPU sweep,
and artifact-promotion details live in the docs.

| Target | Backend | Policy | max_mode | ESS | Final J | QI legacy | Mirror | Aspect | Iota | Wall time |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| QA | CPU | continuation | 3 | yes | 4.35e-04 |  |  | 6.000 | 0.4200 | 5.4 min |
| QH | CPU | continuation | 3 | yes | 1.90e-03 |  |  | 6.000 | -1.2053 | 3.9 min |
| QP | CPU | continuation | 3 | no | 5.38e-02 |  |  | 6.015 | -0.6724 | 3.9 min |
| QI | CPU | qi_default | 3 | yes | 1.37e-02 | 4.31e-04 | 0.272 | 6.002 | -0.5690 | 10.9 min |

Metric definitions and policy details are in `docs/optimization.rst`; the
README table is only the current compact promotion snapshot.

![QA optimization](docs/_static/figures/readme_best_optimization_qa.png)
![QH optimization](docs/_static/figures/readme_best_optimization_qh.png)
![QP optimization](docs/_static/figures/readme_best_optimization_qp.png)
![QI optimization](docs/_static/figures/readme_best_optimization_qi.png)

### QI from different NFP inputs

The same `QI_optimization.py` workflow can be run from reviewed NFP 1, 2, 3,
and 4 inputs by changing the input variables at the top of the script. The
panel is case-gated rather than a uniform aspect-ratio promotion table; full
provenance and limitations are in the docs.

![QI optimization from NFP seeds](docs/_static/figures/readme_qi_optimization_cases.png)

Reproduction commands, artifact-promotion rules, QI NFP coverage, and full
sweep publication requirements are documented in `docs/optimization.rst`; these
case-specific artifacts are not aspect-6 README best-row promotion evidence.

## Performance, Validation, Release

- Performance notes and benchmark caveats: `docs/performance.rst`
- Validation and VMEC2000 parity status: `docs/validation.rst`
- Testing and coverage strategy: `docs/testing_strategy.rst`
- Release checklist and CI gates: `docs/release_checklist.rst`
- Latest documented local rerun snapshot: `outputs/rerun_20260525_123334`.
- Latest repository release tag:
  [`v0.0.13`](https://github.com/uwplasma/vmec_jax/releases/tag/v0.0.13).
- Package indexes may lag tags; verify PyPI/conda-forge before advertising.
- Re-check green `main` CI and record the required `95%` coverage gate before
  tagging.

## CLI Reference

```text
vmec_jax input.*           run the equilibrium solver and write wout_*.nc
vmec_jax --plot wout.nc    generate diagnostic plots
vmec_jax --parity input.*  force the conservative VMEC2000-style loop
vmec_jax --help            show the full option list
```
