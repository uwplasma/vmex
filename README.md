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

The plain package includes plotting support (`matplotlib`) and the
differentiable Boozer transform dependency (`booz_xform_jax`) used by the QI
examples, so there is no separate plotting or QI extra to install.

From conda-forge (the feedstock can lag PyPI by a release):

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

The README intentionally keeps only the compact best current
stellarator-symmetric QA/QH/QP/QI rows. Extended policy discussion, LASYM
panels, finite-beta examples, QI NFP 1/2/3/4 coverage, minimal-seed status,
failure modes, partial CPU/GPU sweep snapshots, and full-matrix artifact
requirements live in the docs.

| Target | Backend | Policy | max_mode | ESS | Final J | QI legacy | Mirror | Aspect | Iota | Wall time |
|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| QA | CPU | continuation | 3 | yes | 4.35e-04 |  |  | 6.000 | 0.4200 | 5.4 min |
| QH | CPU | continuation | 3 | yes | 1.90e-03 |  |  | 6.000 | -1.2053 | 3.9 min |
| QP | CPU | continuation | 3 | no | 5.38e-02 |  |  | 6.015 | -0.6724 | 3.9 min |
| QI | CPU | qi_default | 3 | yes | 1.37e-02 | 4.31e-04 | 0.272 | 6.002 | -0.5690 | 10.9 min |

Reviewed best-row panel assets are checked in as
`docs/_static/figures/readme_best_optimization_qa.png`,
`docs/_static/figures/readme_best_optimization_qh.png`,
`docs/_static/figures/readme_best_optimization_qp.png`, and
`docs/_static/figures/readme_best_optimization_qi.png`.

![QA optimization](docs/_static/figures/readme_best_optimization_qa.png)
![QH optimization](docs/_static/figures/readme_best_optimization_qh.png)
![QP optimization](docs/_static/figures/readme_best_optimization_qp.png)
![QI optimization](docs/_static/figures/readme_best_optimization_qi.png)

Refresh the checked-in compact README rows and panels from the reviewed artifact
bundle with:

```bash
PYTHONPATH=. python examples/optimization/render_readme_best_optimizations.py
```

To promote newly run optimizations into those rows, first run the individual
scripts, inspect the outputs, and copy the accepted `history.json`,
`wout_original.nc`, and `wout_final.nc` artifacts into
`docs/_static/readme_best_cases/<case>/` before rerunning the renderer:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QA_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QH_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QP_optimization.py
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QI_optimization.py
```

Additional checked-in optimization assets include case-specific, case-gated QI
NFP 1/2/3/4 panel/CSV rows and the minimal-seed showcase objective/state panels
plus CSV. They are
documented in `docs/optimization_sweep_results.rst` as status artifacts, not as
aspect-6 README best-row promotion or global seed-robustness evidence.

![QI NFP coverage](docs/_static/figures/readme_qi_optimization_cases.png)

Each row in `readme_qi_optimization_cases.png` is produced by the same editable
script. To run one case, open `examples/optimization/QI_optimization.py`, set
`RUN_CASE` at the top to one of `nfp1_qi`, `nfp2_qi`, `nfp3_qi`, or `nfp4_qi`,
and run:

```bash
PYTHONPATH=. JAX_PLATFORMS=cpu python examples/optimization/QI_optimization.py
```

That script intentionally exposes the full workflow instead of hiding it behind
one high-level wrapper: it selects the seed VMEC input, optionally builds the
simple or target-helicity seed, runs the configured preconditioner/local stages,
assembles the QI residual blocks and weights, calls the least-squares optimizer,
writes the final VMEC input/wout/history artifacts, prints the final QI,
mirror, elongation, aspect-ratio, and iota metrics from the result object, and
renders the 3D, Boozer-`|B|`, and objective-history plots. To rerender the
checked-in NFP panel after updating cases, run:

```bash
PYTHONPATH=. python examples/optimization/render_qi_readme_cases.py
```

## Performance, Validation, Release

- Performance notes and benchmark caveats: `docs/performance.rst`
- Validation and VMEC2000 parity status: `docs/validation.rst`
- Testing and coverage strategy: `docs/testing_strategy.rst`
- Release checklist and CI gates: `docs/release_checklist.rst`
- Latest published release:
  [`v0.0.12`](https://github.com/uwplasma/vmec_jax/releases/tag/v0.0.12)
- Release-candidate CI baseline: re-check the newest completed green `main`
  run with `gh run list --repo uwplasma/vmec_jax --branch main --workflow CI
  --limit 5` before tagging.
- Required fast coverage gate is now `95%`; the latest local CI-equivalent
  run reached `95.10%` on this development batch.

## CLI Reference

```text
vmec_jax input.*           run the equilibrium solver and write wout_*.nc
vmec_jax --plot wout.nc    generate diagnostic plots
vmec_jax --parity input.*  force the conservative VMEC2000-style loop
vmec_jax --help            show the full option list
```
