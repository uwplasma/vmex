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
workflows, with free-boundary support, VMEC2000-compatible `mgrid` workflows,
and direct-coil research paths. Full adaptive free-boundary solve adjoints
remain in development.

## Install

From PyPI:

```bash
pip install vmec-jax
```

The plain package includes plotting support and `booz_xform_jax`; no separate
extra is needed. If bare `pip` does not install into the Python you intend to
use, retry with `python -m pip install vmec-jax`.

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

Large optional validation assets stay out of git; inspect released bundles with
`python tools/fetch_assets.py --list`.

## Quick Start

For a first run after `pip install vmec-jax`, use the bundled test case:

```bash
vmec --doctor
vmec --test
```

`vmec --doctor` prints Python, pip, package, and JAX backend diagnostics.
`vmec --test` copies `input.nfp4_QH_warm_start`, runs with `FTOL_ARRAY = 1e-12`,
writes WOUT and plots under `vmec_jax_test/`, and prints equivalent manual
commands. The canonical executable is `vmec`; old aliases remain supported.

To run the same workflow manually with an input downloaded from the repository:

```bash
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/input.nfp4_QH_warm_start
vmec input.nfp4_QH_warm_start
```

```bash
vmec --plot wout_nfp4_QH_warm_start.nc
vmec --plot wout_nfp4_QH_warm_start.nc --outdir figures/
```

Run Boozer coordinates with bundled `booz_xform_jax`; by default `--booz` uses
`mbooz = 32`, `nbooz = 32`, and all VMEC surfaces:

```bash
vmec --booz input.nfp4_QH_warm_start
vmec --booz --plot input.nfp4_QH_warm_start
vmec --booz wout_nfp4_QH_warm_start.nc
vmec --plot boozmn_nfp4_QH_warm_start.nc
```

Use the Python API:

```python
import vmec_jax as vj

run = vj.run_fixed_boundary("input.nfp4_QH_warm_start")
wout_path = "wout_nfp4_QH_warm_start.nc"
vj.write_wout_from_fixed_boundary_run(wout_path, run, include_fsq=True)
vj.plot_wout(wout_path, outdir="figures/")
boozmn = vj.run_booz_xform(wout_path, mbooz=32, nbooz=32)
vj.plot_boozmn(boozmn, outdir="figures/")
```

VMEC pressure, iota, and current profiles can be polynomial coefficients or
tabulated splines. The bundled spline deck uses `PMASS_TYPE`/`PIOTA_TYPE =
"cubic_spline"` with `*_AUX_S/F`; finite-beta decks use `PCURR_TYPE =
"cubic_spline_ip"` with `AC_AUX_S/F`. The same syntax supports `akima_spline`
and `line_segment`:

```bash
python examples/profile_input_examples.py
vmec examples/data/input.profile_splines --plot
vmec examples/data/input.nfp4_QH_finite_beta
```

`examples/profile_input_examples.py` writes editable polynomial and spline decks
under `examples/outputs/profile_inputs/` and prints the matching `vmec` commands.

For the bundled small free-boundary example, download both the input deck and
its magnetic grid into the same folder:

```bash
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/input.cth_like_free_bdy_lasym_small
curl -L -O https://raw.githubusercontent.com/uwplasma/vmec_jax/main/examples/data/mgrid_cth_like_lasym_small.nc
vmec input.cth_like_free_bdy_lasym_small
```

### Direct-Coil Free-Boundary Research Lane

The direct-coil free-boundary lane samples differentiable Biot-Savart coils
directly while keeping the existing `mgrid` path for VMEC2000 compatibility.
The finalized single-stage optimization lane recomputes a complete direct-coil
free-boundary solve for each accepted objective point. Current coil-only
examples validate cheap VMEC-state QS/aspect/iota proxies and same-branch,
fingerprint-gated branch-local derivatives; complete solves remain the
acceptance authority. They do not claim production differentiation through
adaptive accepted/rejected host-controller branch changes or full
coil-to-Boozer adjoints.

```bash
python examples/free_boundary_direct_coils_forward.py \
  --max-iter 4 \
  --outdir results/free_boundary_direct_coils_forward
```

ESSOS direct-coil, generated-mgrid, finite-beta scan, and coil-only QS
optimization commands are documented in `docs/free_boundary_coil_optimization.rst`.

![DIII-D finite-beta mgrid free-boundary scan](docs/_static/figures/freeb_diiid_mgrid_beta_ns101_panel.png)

![LP-QA direct-coil finite-beta free-boundary scan](docs/_static/figures/freeb_lpqa_direct_coil_beta_ns101_panel.png)

## Backend Selection

`vmec_jax` follows the selected JAX backend. If CPU-only JAX is installed, runs
use CPU. If GPU-enabled JAX is installed and selected, runs use the accelerator;
`vmec_jax` does not silently force those runs back to CPU.
Install or upgrade GPU-enabled JAX using the official JAX installation matrix:
https://docs.jax.dev/en/latest/installation.html

```bash
python -c "import jax; print(jax.default_backend()); print(jax.devices())"
JAX_PLATFORMS=cpu vmec input.nfp4_QH_warm_start
JAX_PLATFORM_NAME=gpu vmec input.nfp4_QH_warm_start
JAX_PLATFORMS=cuda vmec input.nfp4_QH_warm_start
```

From Python, leave `solver_device` unset to inherit JAX's default backend, or
pass `solver_device="cpu"` / `solver_device="gpu"` explicitly.

## Optimization Examples

Editable optimization examples live in `examples/optimization/`. Start with
`examples/optimization/README.md`, then use `docs/optimization.rst`,
`docs/optimization_sweep_results.rst`, and `docs/piecewise_omnigenous_plan.rst`.

The compact panels show QA/QH/QP common-minimal-seed runs and the QI NFP1/2/3/4
minimal-seed policy. For QI, the raw input is always a circular-torus-like
`input.minimal_seed_nfp*` deck; optimization-time helicity hints and same-NFP
reference-family stages are recorded as artifacts, not hidden seed edits. Full
numeric tables, caveats, LASYM panels, artifact-promotion rules live in the docs,
with historical `readme_best_optimization_qa.png`, `readme_best_optimization_qh.png`,
`readme_best_optimization_qp.png`, and `readme_best_optimization_qi.png` archived there.

![Common minimal-seed QA/QH/QP states](docs/_static/figures/minimal_seed_showcase_state_panel.png)

Reproduce the minimal-seed optimization rows with:

```bash
PYTHONPATH=. JAX_PLATFORMS=cuda python3 examples/optimization/generate_minimal_seed_showcase.py \
  --cases qa_nfp2,qa_nfp3,qh_nfp3,qh_nfp4,qp_nfp2,qp_nfp3,qi_nfp1,qi_nfp2,qi_nfp3,qi_nfp4 \
  --backend-label gpu \
  --solver-device gpu --worker-jax-platforms cuda --policy continuation --max-mode 5 --ess on \
  --max-nfev 70 --continuation-nfev 20 --inner-max-iter 550 --inner-ftol 1e-10 \
  --trial-max-iter 550 --trial-ftol 1e-10 --ess-alpha 1.2 --case-timeout-s 7200 --rerun
PYTHONPATH=. python examples/optimization/render_minimal_seed_showcase.py --publication-matrix
PYTHONPATH=. python examples/optimization/render_qi_readme_cases.py
```

Run individual editable examples with `python examples/optimization/QA_optimization.py`,
`QH_optimization.py`, `QP_optimization.py`, or `QI_optimization.py`. The
`QI_optimization_seed.py` seed-3127 preset is retained as a diagnostic stress
case, not as a README promotion row. Full provenance and artifact rules are in
`docs/optimization.rst` and `docs/optimization_sweep_results.rst`.

## Performance, Validation, Release

- Performance notes: `docs/performance.rst`; validation, coverage, and release
  gates: `docs/validation.rst`, `docs/testing_strategy.rst`, and
  `docs/release_checklist.rst`.
- Latest repository release tag:
  [`v0.0.14`](https://github.com/uwplasma/vmec_jax/releases/tag/v0.0.14).

## CLI Reference

```text
vmec input.*           run the equilibrium solver and write wout_*.nc
vmec --plot wout.nc    generate VMEC diagnostic plots from a WOUT file
vmec --booz wout.nc    run booz_xform_jax and write boozmn_*.nc
vmec --plot boozmn.nc  generate Boozer contour and spectrum plots
vmec --parity input.*  force the conservative VMEC2000-style loop
vmec --help            show the full option list
```
