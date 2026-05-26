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

From conda-forge:

```bash
pixi add vmec-jax
conda install --channel conda-forge vmec-jax
```

Developer install from source:

```bash
git clone https://github.com/uwplasma/vmec_jax
cd vmec_jax
pip install -e .
```

The repository intentionally keeps generated WOUT fixtures and large optional
validation assets out of git.  A source clone contains the VMEC input decks and
small magnetic grids needed for ordinary examples; run the inputs to generate
new `wout_*.nc` files.  If you need the full released WOUT/reference bundle for
CI-style validation or docs regeneration, download it with:

```bash
python tools/fetch_assets.py --list
python tools/fetch_assets.py
```

## Quick Start

For a first run after `pip install vmec-jax`, use the bundled test case:

```bash
vmec_jax --test
```

This copies the packaged `input.nfp4_QH_warm_start` into `vmec_jax_test/`,
runs the solver, writes `wout_nfp4_QH_warm_start.nc`, and automatically plots
the WOUT file into `vmec_jax_test/figures/`. The terminal output also prints the
equivalent manual commands so new users can repeat each step themselves.

To run the same workflow manually with an input downloaded from the repository:

```bash
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/input.nfp4_QH_warm_start
vmec_jax input.nfp4_QH_warm_start
```

Plot the `wout_*.nc` file produced by that run:

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

### Direct-Coil Free-Boundary Research Lane

The `feature/freeb-essos-coil-single-stage` branch adds a JAX-native
external-field provider so free-boundary solves can sample differentiable
Biot-Savart coils directly, while the existing `mgrid` path remains the
VMEC2000-compatibility backend. Phase-1 generated WOUTs, magnetic grids, and
reviewer plots are kept out of git.

```bash
python examples/free_boundary_direct_coils_forward.py \
  --max-iter 4 \
  --outdir results/free_boundary_direct_coils_forward
```

With ESSOS on `PYTHONPATH`, run the finite-pressure ESSOS-coil beta scan. The
matched `mgrid` lane requires an ESSOS checkout with `Coils.to_mgrid`. Add
`--skip-mgrid-runs` for a direct-coil-only run with released ESSOS:

```bash
export ESSOS_ROOT=/path/to/ESSOS_mgrid_pr
PYTHONPATH=.:$ESSOS_ROOT:$PYTHONPATH \
  python examples/free_boundary_essos_coils_beta_scan.py \
  --input examples/data/input.LandremanPaul2021_QA_lowres \
  --phiedge=-0.025 --betas 0 1 2 --pressure-continuation \
  --ns-array 16,31 --niter-array 600,1200 --ftol-array 1e-8,1e-8 \
  --max-iter 1200 --activate-fsq 1e99
```

Long high-resolution pressure scans can be resumed without repeating completed
beta points by rerunning the same command with `--resume-existing`; existing
`wout_*_beta_*.nc` files are reused as pressure-continuation seeds when their
residuals satisfy the promotion threshold.

The DIII-D reference scan reaches final `ns=101`, `FTOL=1e-12`, and actual
WOUT betas through 2.18%. The LP-QA stellarator pressure-continuation lane now
promotes for both generated-`mgrid` and direct differentiable coil providers,
with actual WOUT betas above 1%. Full nonlinear exact-adjoint gradients through
the free-boundary iteration remain phase-2 work. See
`docs/free_boundary_coil_optimization.rst` for reviewer plots, VMEC2000
comparisons, benchmarks, limitations, and the full-solve adjoint plan.

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
`examples/optimization/README.md`, then use `docs/optimization.rst`,
`docs/optimization_sweep_results.rst`, and
`docs/piecewise_omnigenous_plan.rst` for the full method guide, sweep tables,
pwO plan, LASYM panels, finite-beta cases, and QI provenance.

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
and 4 inputs by changing the input variables at the top of the script. This is
case-specific evidence, not an aspect-6 README best-row promotion; full
provenance and limitations are in the docs.

![QI optimization from NFP seeds](docs/_static/figures/readme_qi_optimization_cases.png)

Reproduction commands, artifact-promotion rules, and full sweep requirements
are documented in `docs/optimization.rst` and
`docs/optimization_sweep_results.rst`.

## Performance, Validation, Release

- Performance notes and benchmark caveats: `docs/performance.rst`
- Validation and VMEC2000 parity status: `docs/validation.rst`
- Testing and coverage strategy: `docs/testing_strategy.rst`
- Release checklist and CI gates: `docs/release_checklist.rst`
- Latest repository release tag:
  [`v0.0.13`](https://github.com/uwplasma/vmec_jax/releases/tag/v0.0.13)
- Package indexes may lag the repository tag; verify PyPI/conda-forge before
  advertising a package-index release.
- Required fast coverage gate is `95%`; record the current CI/local coverage
  result from the release-candidate commit in the release notes.

## CLI Reference

```text
vmec_jax input.*           run the equilibrium solver and write wout_*.nc
vmec_jax --plot wout.nc    generate diagnostic plots
vmec_jax --parity input.*  force the conservative VMEC2000-style loop
vmec_jax --help            show the full option list
```
