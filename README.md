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

QI optimization uses `booz_xform_jax` for the differentiable Boozer transform:

```bash
pip install "vmec-jax[qi]"
```

From conda-forge:

```bash
pixi add vmec-jax
conda install --channel conda-forge vmec-jax
```

Developer install from source:

```bash
git clone https://github.com/uwplasma/vmec_jax
cd vmec_jax
pip install -e ".[qi]"
```

## Quick Start

Run the solver with the VMEC2000-style CLI:

```bash
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
freeb = vj.run_free_boundary("input.cth_like_free_bdy_lasym_small")
vj.plot_wout("wout_nfp4_QH_warm_start.nc", outdir="figures/")
```

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

## Best Optimization Examples

Editable optimization examples live in `examples/optimization/`. Start with
`examples/optimization/README.md` for workflow anatomy, then use
`docs/optimization.rst` for the full method guide,
`docs/optimization_sweep_results.rst` for generated sweep tables/figures, and
`docs/piecewise_omnigenous_plan.rst` for the pwO planning and acceptance gates.

The README intentionally shows only the best current stellarator-symmetric
QA/QH/QP/QI rows. Each panel contains initial/final 3D LCFS views, objective
history over all stages, and initial/final Boozer `|B|` line contours. Extended
policy discussion, LASYM panels, finite-beta examples, QI case coverage,
failure modes, partial CPU/GPU sweep snapshots, and full-matrix artifact
requirements live in the docs.

| Target | Backend | Policy | max_mode | ESS | Final J | QI legacy | Mirror | Aspect | Iota | Wall time |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| QA | CPU | continuation | 3 | yes | 4.35e-04 |  |  | 6.000 | 0.4200 | 5.4 min |
| QH | CPU | continuation | 3 | yes | 1.90e-03 |  |  | 6.000 | -1.2053 | 3.9 min |
| QP | CPU | continuation | 3 | no | 5.38e-02 |  |  | 6.015 | -0.6724 | 3.9 min |
| QI | CPU | qi_default | 3 | yes | 1.37e-02 | 4.31e-04 | 0.272 | 6.002 | -0.5690 | 10.9 min |

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

Reproduce the compact README rows and panels with the individual optimization
scripts and renderer:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QA_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QH_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QP_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QI_optimization.py
PYTHONPATH=. python examples/optimization/render_readme_best_optimizations.py
```

Full sweep reproduction targets, generated result tables, QI multi-NFP
coverage, and publication asset requirements are in
`docs/optimization_sweep_results.rst`. The compact README panels remain the
reviewed `LASYM = F` best rows only.

## QI From Multiple NFP Inputs

The same `examples/optimization/QI_optimization.py` workflow is exercised on
case-gated NFP 1, 2, 3, and 4 QI input lanes. The panel below is coverage
evidence for those configured cases, not a seed-robustness claim and not a
blanket guarantee that arbitrary VMEC inputs reach QI without tuning. The
initial panels use the source-input WOUTs; final panels use the accepted
case-gated outputs. Provenance, gates, and case-specific targets are documented
in `docs/optimization_sweep_results.rst`; these rows are not aspect-6 README
best-row promotion evidence.
The panel below is regenerated with:

```bash
PYTHONPATH=. python examples/optimization/render_qi_readme_cases.py
```

<p align="center">
  <img src="docs/_static/figures/readme_qi_optimization_cases.png" width="980" />
</p>

## Performance, Validation, Release

- Performance notes and benchmark caveats: `docs/performance.rst`
- Validation and VMEC2000 parity status: `docs/validation.rst`
- Testing and coverage strategy: `docs/testing_strategy.rst`
- Release checklist and CI gates: `docs/release_checklist.rst`
- Latest published release:
  [`v0.0.11`](https://github.com/uwplasma/vmec_jax/releases/tag/v0.0.11)
- Release-candidate CI baseline: re-check the newest completed green `main`
  run with `gh run list --repo uwplasma/vmec_jax --branch main --workflow CI
  --limit 5` before tagging.
- Required fast coverage gate is currently `90%`; `95%` remains the next
  ratchet and should be claimed only from a fresh coverage run on the
  release-candidate commit.

## CLI Reference

```text
vmec_jax input.*           run the equilibrium solver and write wout_*.nc
vmec_jax --plot wout.nc    generate diagnostic plots
vmec_jax --parity input.*  force the conservative VMEC2000-style loop
vmec_jax --help            show the full option list
```
