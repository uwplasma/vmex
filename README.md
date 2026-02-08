# vmec-jax

Laptop-friendly, end-to-end differentiable (JAX) rewrite of **VMEC2000**, focusing on **fixed-boundary** first.

`vmec_jax` targets:
- VMEC-compatible `input.* -> wout_*.nc` (parity-first development),
- clean Python API with JAX transforms (grad/JIT/vmap),
- explicit, documented numerics (VMEC2000 and VMEC++ conventions).

Current scope:
- Fixed-boundary only.
- Symmetric configurations only (`lasym=False`) for now.

## Quickstart (recommended)

The minimal workflow is:
1. Load `input.*`
2. Run fixed-boundary
3. Write `wout_*.nc`
4. Plot and compare to a reference `wout`

The canonical example is:

```bash
python examples/showcase_axisym_input_to_wout.py
```

## Fixed-Boundary Snapshot (Current)

These figures were generated from the bundled `shaped_tokamak_pressure` case using:

```bash
python examples/showcase_axisym_input_to_wout.py --case shaped_tokamak_pressure --max-iter 5
```

<table>
  <tr>
    <td><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_surfaces.png" width="420" /></td>
    <td><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_bmag_lcfs.png" width="420" /></td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_lcfs_3d_bmag.png" width="860" /></td>
  </tr>
</table>
<!-- end snapshot table -->

## Fixed-Boundary Benchmark (Runtime + Residual Traces)

These figures compare a *fixed iteration budget* across `vmec2000` (Fortran via `vmec` Python extension),
`vmecpp`, and `vmec_jax` on 4 bundled inputs (2 axisymmetric, 2 3D). Reproduce via:

```bash
python examples/validation/benchmark_fixed_boundary_runtime_and_residuals.py --iters 10
```

Default cases (inputs live under `examples/data/`):
- `circular_tokamak`
- `vmecpp_solovev`
- `cth_like_fixed_bdy`
- `nfp4_QH_warm_start`

Notes:
- `vmec_jax` runtime excludes JAX compilation time (per-case warmup run).
- Residual traces use VMEC-style `fsq_total` quantities and are not normalized to O(1).

<table>
  <tr>
    <td><img src="docs/_static/figures/bench_fixed_boundary_runtime.png" width="420" /></td>
    <td><img src="docs/_static/figures/bench_fixed_boundary_residual.png" width="420" /></td>
  </tr>
</table>
<!-- end benchmark table -->

## Fixed-Boundary Pipeline (VMEC2000/VMEC++ Numerics)

The fixed-boundary iteration in VMEC is not "one kernel"; it is a pipeline of discrete conventions.
The high-ROI parity strategy in `vmec_jax` is to validate each stage against VMEC outputs before iterating.

1. Parse input: `VmecConfig`, `INDATA` (boundary, profiles, resolution, symmetry flags).
2. Build grids and modes: `AngleGrid`, `ModeTable`, `HelicalBasis`, Nyquist mode table.
3. Boundary representation: `RBC/ZBS` (and `RBS/ZBC` when `lasym=True`, deferred).
4. Initial guess: `state0` (`Rcos/Rsin/Zcos/Zsin/Lcos/Lsin`) with VMEC axis rules.
5. Real-space synthesis: `R,Z,L` and derivatives (`Ru,Zu,Rv,Zv,Lu,Lv`) with VMEC parity rules.
6. Half-mesh Jacobian-like fields: `sqrtg`, `r12`, `ru12`, `zu12`, `rs`, `zs`, `tau`.
7. Half-mesh metric elements: `guu`, `guv`, `gvv` (including cylindrical `R^2` in `gvv`).
8. Flux functions and lambda scaling: `phipf`, `chipf -> chips`, `lamscale`, `overg`.
9. Contravariant magnetic field: `bsupu`, `bsupv` (VMEC's staggered, flux-corrected conventions).
10. Covariant magnetic field: `bsubu`, `bsubv` via metric products.
11. Magnetic + pressure scalar: `bsq = 0.5*|B|^2 + p`.
12. Constraint-force plumbing: `tcon`, `alias` (m=1 rotation, constraint mixing).
13. Spectral force blocks: `tomnsps` transforms, mode-by-mode forces.
14. Residual scalars: `fsqr`, `fsqz`, `fsql` ("Step-10 / getfsq" scalars).
15. Preconditioners and update loop: radial and lambda preconditioners, `dt_eff`, edge-force inclusion triggers.
16. Convergence and output: write `wout_*.nc` (geometry, fields, scalars, diagnostics).

See `docs/validation.rst` for definitions and deeper references to VMEC2000/VMEC++ source blocks.

## 4-Case Pipeline Parity Snapshot (No Solve)

This table compares `vmec_jax` pipeline quantities reconstructed from *reference wout states* against the same
quantities stored in the reference `wout_*.nc` (fast, solver-free).

Reproduce via:

```bash
python examples/validation/pipeline_parity_summary.py
```

| Variable | circular_tokamak | shaped_tokamak_pressure | vmecpp_solovev | li383_low_res |
|---| :--: | :--: | :--: | :--: |
| sqrtg | 3.10e-15 | 1.24e-14 | 2.19e-15 | 2.10e-14 |
| bsupu | 2.46e-15 | 1.13e-14 | 2.18e-15 | 2.25e-14 |
| bsupv | 3.06e-15 | 1.32e-14 | 2.27e-15 | 2.09e-14 |
| bsubu | 7.20e-07 | 4.57e-05 | 2.41e-05 | 7.55e-02 |
| bsubv | 1.24e-05 | 2.59e-05 | 3.11e-05 | 1.13e-02 |
| abs(B) | 3.09e-15 | 1.25e-14 | 2.20e-15 | 2.16e-14 |
| bsq = 0.5*B^2 + p | 6.17e-15 | 2.64e-14 | 4.57e-15 | 4.23e-14 |
| fsqr (step10) | 4.05e-09 | 3.51e-08 | 3.64e-07 | 4.34e+03 |
| fsqz (step10) | 5.78e-10 | 2.73e-08 | 2.69e-07 | 8.31e+03 |
| fsql (step10) | 1.93e-10 | 6.14e-11 | 6.42e-07 | 1.11e+05 |
| fsq_total (step10) | 4.82e-09 | 7.88e-09 | 2.56e-07 | 6.70e+03 |

Interpretation:
- Axisymmetric cases are essentially at floating-point parity for geometry, `bsup*`, and `abs(B)`.
- Remaining known gap: 3D `bsub*` and the resulting step-10 scalars on some `nfp>1` cases (e.g. `li383_low_res`).

## End-to-End Solve Snapshot (Input -> Solve -> Compare)

The table above is solver-free. It demonstrates that many **kernel conventions** match VMEC on *reference states*,
but it does not imply that `vmec_jax` already produces the same converged equilibrium `wout_*.nc` starting from
only `input.*`.

This table runs a short fixed-boundary solve and compares a few end-to-end outputs against bundled VMEC references.
Reproduce via:

```bash
python examples/validation/end_to_end_solve_parity_summary.py --max-iter 2
```

| Case | input | ns | mpol | ntor | nfp | solver | max_iter | ftol | fsq_total(ref) | fsq_total(new) | rmnc relRMS | zmns relRMS |
|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
| circular_tokamak | `input.circular_tokamak` | 17 | 8 | 0 | 1 | vmecpp_iter | 2 | 1.00e-10 | 2.14e-20 | 7.50e-02 | 1.39e-02 | 1.19e-02 |
| shaped_tokamak_pressure | `input.shaped_tokamak_pressure` | 51 | 12 | 0 | 1 | vmecpp_iter | 2 | 1.00e-10 | 1.08e-20 | 3.70e-01 | 1.11e-02 | 3.07e-02 |
| vmecpp_solovev | `input.vmecpp_solovev` | 11 | 6 | 0 | 1 | vmecpp_iter | 2 | 1.00e-10 | 1.70e-10 | 1.29e-01 | 1.92e-03 | 6.66e-03 |
| li383_low_res | `input.li383_low_res` | 16 | 4 | 3 | 3 | vmecpp_iter | 2 | 1.00e-10 | 1.06e-06 | 3.40e+07 | 5.43e-02 | 8.45e-02 |

Interpretation:
- Full end-to-end fixed-boundary parity is **not yet achieved**. The solver iteration and convergence behavior is still being aligned with VMEC++/VMEC2000.
- `vmec_jax` currently writes *minimal* `wout_*.nc` outputs for solver runs; Nyquist outputs (`gmnc`, `bsup*`, `bsub*`, `bmnc`) are not fully populated yet.

## Remaining Parity Assertions

- VMEC-quality fixed-boundary convergence (iteration history parity, stopping criteria, and edge-force triggers).
- 3D `bsub*` parity for `nfp>1` (drives `getfsq` scalars in the converged regime).
- `lasym=True` (up-down / stellarator asymmetry), including `tomnspa` conventions.
- Free-boundary VMEC (planned).
- Parallelization (planned).

## External baselines (VMEC2000 + VMEC++)

Two external baselines are supported for cross-checks when installed locally:

- **VMEC2000 (Fortran)** via its Python extension (`vmec`) and MPI driver.
- **VMEC++** via the `vmecpp` Python API.

Use the helper script:

```bash
# VMEC2000 (requires vmec python extension + mpi4py + netCDF4 + system libnetcdf)
python examples/validation/external_vmec_driver_compare.py --backend vmec2000 --case circular_tokamak

# VMEC++ (requires vmecpp + netCDF4)
python examples/validation/external_vmec_driver_compare.py --backend vmecpp --case circular_tokamak
```

The script runs the external code, writes a `wout_*.nc`, compares key fields to
the bundled references, and can optionally compute vmec_jax B-field parity
metrics. See `docs/validation.rst` for details and troubleshooting notes.

## Installation

Create an environment with Python >= 3.10.

Regular users (non-editable install):

```bash
git clone https://github.com/uwplasma/vmec_jax.git
cd vmec_jax
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

## Quickstart

Run a small validated workflow (inputs + reference `wout` files are bundled under `examples/data/`):

```bash
python examples/showcase_axisym_input_to_wout.py
python examples/tutorial/00_parse_and_boundary.py examples/data/input.circular_tokamak --out boundary.npz --verbose
```

## Examples

Examples are organized into:
- `examples/showcase_axisym_input_to_wout.py`: minimal "input -> wout + plots + parity" entrypoint.
- `examples/tutorial/`: minimal low-level kernel demos.
- `examples/validation/`: parity checks vs bundled `wout_*.nc`.
- `examples/visualization/`: plotting + VTK export.
- `examples/data/`: bundled regression inputs + reference `wout` files.
- `tools/diagnostics/`: developer-only parity and debugging scripts.

ParaView export (VTK surface fields + field lines):

```bash
python examples/visualization/vtk_field_and_fieldlines.py examples/data/input.li383_low_res --hi-res --outdir vtk_out
```

## Documentation

Sphinx docs live in `docs/`.

Build locally:

```bash
LANG=C LC_ALL=C python -m sphinx -b html docs docs/_build/html
```

## Testing

```bash
pytest -q
```

If `netCDF4` is not installed, tests requiring `wout_*.nc` I/O are skipped.

## Contributing

Contributions are welcome. Practical ways to help:
- add parity regressions vs VMEC2000 (new cases, tighter tolerances),
- improve kernels (correctness-first; then JIT/vmap performance),
- expand documentation (derivations, conventions, and references),
- add examples that demonstrate differentiability and optimization workflows.

See `docs/contributing.rst` for style and workflow.

## License

MIT. See `LICENSE`.

## References / background

See `docs/references.rst` and the original VMEC literature for algorithmic context.

## Roadmap / step log

The detailed step-by-step porting log and current parity status live in `CODEX_RESUME.md` and `PORTING_NOTES.md`.
