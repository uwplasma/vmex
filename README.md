# vmec-jax

Laptop-friendly, end-to-end differentiable (JAX) rewrite of **VMEC2000**, focusing on **fixed-boundary** first.

## Scope (current)

- Fixed boundary only (free boundary deferred).
- Axisymmetric focus for end-to-end parity: `ntor=0`, `nfp=1`, `lasym=False`.
- Non-axisymmetric and `lasym=True` end-to-end parity are deferred, though many kernels are exercised on bundled 3D reference `wout` files.

## Quickstart

Run the end-to-end showcase (recommended):

```bash
python examples/showcase_axisym_input_to_wout.py --suite
```

Run tests:

```bash
pytest -q
```

## Snapshot figures

Generated from the bundled `shaped_tokamak_pressure` case:

```bash
python examples/showcase_axisym_input_to_wout.py --case shaped_tokamak_pressure --max-iter 240 --use-input-niter --emit-readme-figures --no-verbose
```

<table>
  <tr>
    <td><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_surfaces.png" width="420" /></td>
    <td><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_bmag_lcfs.png" width="420" /></td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_residual.png" width="860" /></td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_lcfs_3d_bmag.png" width="860" /></td>
  </tr>
</table>

## Parity status (VMEC2000)

Parity work is tracked in two layers:

- **Kernel parity on reference states (solver-free):** reconstruct intermediate quantities from a *reference* `wout` state and compare to the quantities stored in that same `wout`. This isolates conventions and avoids solver noise.
- **End-to-end solve parity:** run a nonlinear fixed-boundary solve from `input.*` and compare the final `wout` to the VMEC2000 reference. This depends on the update loop (preconditioning, time-step control, triggers), and is still in progress.

Reproduce the current kernel-parity snapshot table:

```bash
python examples/validation/pipeline_parity_summary.py
```

Current kernel-parity snapshot (solver-free, bundled reference states):

| Variable | circular_tokamak | shaped_tokamak_pressure | solovev | li383_low_res |
|---| :--: | :--: | :--: | :--: |
| sqrtg | 3.10e-15 | 1.24e-14 | 2.19e-15 | 2.10e-14 |
| bsupu | 2.46e-15 | 1.13e-14 | 2.18e-15 | 2.25e-14 |
| bsupv | 3.06e-15 | 1.32e-14 | 2.27e-15 | 2.09e-14 |
| bsubu | 7.20e-07 | 4.57e-05 | 2.41e-05 | 7.55e-02 |
| bsubv | 1.24e-05 | 2.59e-05 | 3.11e-05 | 1.13e-02 |
| abs(B) | 3.09e-15 | 1.25e-14 | 2.20e-15 | 2.16e-14 |
| bsq = 0.5*B^2 + p | 6.17e-15 | 2.64e-14 | 4.57e-15 | 4.23e-14 |
| fsqr | 4.05e-09 | 3.51e-08 | 3.64e-07 | 4.34e+03 |
| fsqz | 5.78e-10 | 2.73e-08 | 2.69e-07 | 8.31e+03 |
| fsql | 1.93e-10 | 6.14e-11 | 6.42e-07 | 1.11e+05 |
| fsq_total | 4.82e-09 | 7.88e-09 | 2.56e-07 | 6.70e+03 |

Interpretation:
- Axisymmetric cases are at floating-point parity for geometry, ``bsup*``, and ``abs(B)``.
- Remaining known gap: 3D ``bsub*`` (and the resulting scalar residuals) on some ``nfp>1`` cases.

Reproduce scalar residual parity (`fsqr/fsqz/fsql`) on reference states:

```bash
python examples/validation/getfsq_parity_cases.py --solve-metric
```

Reproduce the short end-to-end solve snapshot:

```bash
python examples/validation/end_to_end_solve_parity_summary.py --use-input-niter --fast
```

This is a quick sanity run (reduced cases and resolution). For a full parity snapshot, drop `--fast` and increase `--max-iter`, but expect longer runtimes.

## Benchmark (runtime + residual traces)

This script compares a *fixed iteration budget* across `vmec_jax` and (optionally) `vmec2000` via the `vmec` Python extension:

```bash
python examples/validation/benchmark_fixed_boundary_runtime_and_residuals.py --iters 5 --cases circular_tokamak --ns-override 9 --disable-jit --no-warmup
```

To also run the external VMEC2000 backend (if installed):

```bash
python examples/validation/benchmark_fixed_boundary_runtime_and_residuals.py --iters 5 --cases circular_tokamak --ns-override 9 --disable-jit --no-warmup --run-vmec2000
```

The quick settings above keep runs under ~30s; increase `--iters` and `--cases` (and drop `--disable-jit/--no-warmup`) for higher-fidelity traces.

<table>
  <tr>
    <td><img src="docs/_static/figures/bench_fixed_boundary_runtime.png" width="420" /></td>
    <td><img src="docs/_static/figures/bench_fixed_boundary_residual.png" width="420" /></td>
  </tr>
</table>

## External VMEC2000 runs (optional)

If you have the VMEC2000 Python extension installed (`vmec` + `mpi4py` + `netCDF4`), you can run VMEC2000 on an input and compare against bundled references:

```bash
python tools/diagnostics/external_vmec_driver_compare.py --case circular_tokamak
```

## Installation

Create an environment with Python >= 3.10.

Regular users (non-editable install):

```bash
python -m pip install -U pip
python -m pip install .
```

Developers (editable install):

```bash
python -m pip install -e .
```

Recommended extras:

```bash
# JAX runtime (CPU)
python -m pip install ".[jax]"

# Read VMEC2000 `wout_*.nc` reference files
python -m pip install ".[netcdf]"

# Publication-ready figures in examples
python -m pip install ".[plots]"

# Build docs locally
python -m pip install ".[docs]"

# Dev tools
python -m pip install -e ".[dev]"
```

VMEC is typically run in float64. Enable x64 for JAX:

```bash
export JAX_ENABLE_X64=1
```

## Documentation

Sphinx docs live in `docs/`. Build locally:

```bash
LANG=C LC_ALL=C python -m sphinx -b html docs docs/_build/html
```
