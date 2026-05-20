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
`docs/optimization.rst` for the full method guide and
`docs/optimization_sweep_results.rst` for generated sweep tables, figures,
minimal-seed stress cases, QI robustness notes, and reproduction commands.

The panels below show the current stellarator-symmetric examples used for the
README: initial LCFS, final LCFS, objective history, and initial/final Boozer
`|B|` contours on the outer surface. Extended policy discussion, LASYM panels,
finite-beta examples, QI seed robustness, failure modes, and full CPU/GPU sweep
tables live in the docs.

| Target | Backend | Policy | max_mode | ESS | Final J | QI legacy | Mirror | Aspect | Iota | Wall time |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| QA | CPU | continuation | 3 | yes | 2.33e-04 |  |  | 5.000 | 0.4200 | 6.3 min |
| QH | CPU | continuation | 3 | yes | 9.68e-03 |  |  | 4.999 | -1.6595 | 4.0 min |
| QP | CPU | continuation | 3 | no | 6.76e-02 |  |  | 5.019 | -0.6255 | 3.7 min |
| QI | CPU | continuation | 3 | yes | 2.17e-03 | 2.17e-03 | 0.211 | 5.001 | -0.5494 | 11.3 min |

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

Reproduction commands for these panels live in
`docs/optimization_sweep_results.rst`.

QI remains a staged validation area rather than a completed seed-robustness
claim across all NFPs. Run the representative QI example with:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu VMEC_JAX_QI_RUN_CASE=nfp2_qi python examples/optimization/QI_optimization.py
```

NFP=4 QI is tracked as a stress/validation lane, not completed robustness
evidence. Full examples and caveats are in `docs/optimization.rst`.

## Performance, Validation, Release

- Performance notes and benchmark caveats: `docs/performance.rst`
- Validation and VMEC2000 parity status: `docs/validation.rst`
- Testing and coverage strategy: `docs/testing_strategy.rst`
- Release checklist and CI gates: `docs/release_checklist.rst`
- Latest published release: [`v0.0.11`](https://github.com/uwplasma/vmec_jax/releases/tag/v0.0.11)
- Current release hygiene baseline: latest known green `main` CI at `7030eaf`,
  validated local required coverage `86.871%` against the `85%` gate; `90%` and
  `95%` coverage ratchets remain future targets.

## CLI Reference

```text
vmec_jax input.*           run the equilibrium solver and write wout_*.nc
vmec_jax --plot wout.nc    generate diagnostic plots
vmec_jax --parity input.*  force the conservative VMEC2000-style loop
vmec_jax --help            show the full option list
```
