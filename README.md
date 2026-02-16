# vmec-jax

Laptop-friendly, end-to-end differentiable (JAX) rewrite of **VMEC2000**, focusing on **fixed-boundary** first.

<table>
  <tr>
    <td><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_lcfs_3d_bmag.png" width="420" /></td>
    <td><img src="docs/_static/figures/n3are_compare_3d.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Tokamak (shaped_tokamak_pressure)</td>
    <td align="center">Stellarator (n3are_R7.75B5.7_lowres)</td>
  </tr>
</table>

## Scope (current)

- Fixed boundary only (free boundary deferred).
- Axisymmetric end-to-end parity is stable (`ntor=0`, `nfp=1`, `lasym=False`).
- Non-axisymmetric parity: QA/QH stage-1 + stage-2 trace parity (10 iters, `rtol=1e-3`) is now passing; remaining 3D cases still in progress.

## Quickstart

Run the end-to-end showcase (recommended):

```bash
python examples/showcase_axisym_input_to_wout.py --suite
```

CLI (VMEC2000-style executable):

```bash
vmec_jax examples/data/input.circular_tokamak
```

This writes `wout_circular_tokamak.nc` next to the input file and prints the
VMEC2000-style screen table by default. Use `--quiet` to silence output, or
`--outdir` / `--output` to control where the `wout_*.nc` file is written. For
short debug runs, pass `--max-iter` and `--no-multigrid` (single grid).

By default the solver prints the VMEC2000-style per-iteration **screen** table
(FSQR/FSQZ/FSQL, RAX, DELT, WMHD). Pass ``--no-verbose`` to silence it.

Legacy `vmecPlot2.py` compatibility (NetCDF3 `wout` output):

```bash
python examples/showcase_axisym_input_to_wout.py --case circular_tokamak --max-iter 5 --no-vmec2000-trace
python vmecPlot2.py examples/outputs/showcase/circular_tokamak/wout_circular_tokamak_vmec_jax.nc /tmp/vmecplot2_jax
```

Run tests:

```bash
pytest -q
```

Profiling (fixed-boundary iterations):

```bash
python tools/diagnostics/profile_fixed_boundary.py --input examples/data/input.ITERModel --iters 3 --use-scan
python tools/diagnostics/profile_fixed_boundary.py --input examples/data/input.ITERModel --iters 3 --use-scan --simple-profile
```

The first command attempts a TensorBoard trace (requires a compatible TensorFlow install). Use `--simple-profile` to fall back to a timing-only run without TensorBoard.
`--use-scan` enables the fast ``lax.scan`` iteration path (no VMEC2000 control logic), which is ideal for performance profiling but not for per-iteration parity.
You can also select the scan path directly via solver `vmec2000_iter_fast` (alias `vmec2000_scan`).
Set `VMEC_JAX_USE_SCAN=1` to force scan mode for VMEC-style runs without changing code.

VMEC2000 integration parity (requires the VMEC2000 executable):

```bash
VMEC2000_INTEGRATION=1 pytest -k vmec2000_exec_qa_regression
```

Note: `vmec_jax` enables JAX 64-bit in the fixed-boundary driver for parity. Set `JAX_ENABLE_X64=0` to prioritize speed.
For faster fixed-boundary solves in Python, the force/residual pipeline is JIT-compiled by default. Pass `jit_forces=False` to `run_fixed_boundary(...)` to disable it. Debug dump env vars automatically disable JIT.
For best performance, `VMECStatic` now precomputes VMEC real-space phase stacks. Set `VMEC_JAX_CACHE_VMEC_PHASE=0` to skip the extra cached tensors if you need to minimize memory.
To reduce repeat JIT compilation time across runs, set `VMEC_JAX_COMPILATION_CACHE_DIR=/path/to/cache` (or `JAX_COMPILATION_CACHE_DIR`) to enable the JAX compilation cache.
The fixed-boundary update also precomputes dense (m,n)->signed maps per solve to reduce scatter-heavy updates during iterations.
Scan mode batches the Z/L sin-block conversions into one matmul-based mapping to reduce kernel count.
Axis/edge enforcement now uses concatenation instead of scatter updates to keep the scan loop lighter.
Initial-guess axis blending updates all m=0 columns in one vectorized step to reduce startup overhead.
Mode scaling factors (1/(mscale*nscale)) are cached in `VMECStatic` to avoid repeated table gathers in the initial guess.
Lambda gauge enforcement uses a boolean mask instead of scatter updates in the iteration loop.
Axis m=0 masks are reused from `VMECStatic` to avoid per-iteration reconstruction.

## Snapshot figures

Generated from:

- Tokamak: bundled `shaped_tokamak_pressure` case (single-grid parity run).
- Stellarator: bundled `n3are_R7.75B5.7_lowres` side-by-side VMEC2000/vmec_jax diagnostics.

```bash
python examples/showcase_axisym_input_to_wout.py \
  --case shaped_tokamak_pressure \
  --max-iter 10 \
  --emit-readme-figures \
  --vmec2000-timeout 60 \
  --vmec2000-nstep 1

python tools/diagnostics/n3are_vmec_vs_vmecjax.py \
  --solve --solver vmec2000_iter --max-iter 10 \
  --outdir docs/_static/figures
```

<table>
  <tr>
    <td><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_surfaces.png" width="420" /></td>
    <td><img src="docs/_static/figures/n3are_compare_cross_sections.png" width="420" /></td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_bmag_lcfs.png" width="420" /></td>
    <td><img src="docs/_static/figures/n3are_compare_bmag_surface.png" width="420" /></td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_lcfs_3d_bmag.png" width="420" /></td>
    <td><img src="docs/_static/figures/n3are_compare_3d.png" width="420" /></td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/_static/figures/showcase_shaped_tokamak_pressure_residual.png" width="860" /></td>
  </tr>
</table>

Interpretation of the snapshot figures:

- The residual trace overlay is a **per-iteration VMEC2000 executable trace** (dashed) from `threed1.*`. The default single-grid run (`ns=13`, 10 iterations) overlays VMEC2000 and vmec_jax within ~1e-3 rtol (often tighter).
- The LCFS `|B|` panel now uses **vmecPlot2-style grids** (theta/zeta resolution and toroidal-angle conventions) for both VMEC2000 and vmec_jax. Differences here reflect solver parity, not plotting.

## Parity status (VMEC2000)

Parity work is tracked in two layers:

- **Kernel parity on reference states (solver-free):** reconstruct intermediate quantities from a *reference* `wout` state and compare to the quantities stored in that same `wout`. This isolates conventions and avoids solver noise.
- **End-to-end solve parity:** run a nonlinear fixed-boundary solve from `input.*` and compare the final `wout` to the VMEC2000 reference. This depends on the update loop (preconditioning, time-step control, triggers), and is still in progress.

Reproduce the current kernel-parity snapshot table:

```bash
python examples/validation/pipeline_parity_summary.py
```

Current kernel-parity snapshot (solver-free, bundled reference states):

| Variable | circular_tokamak | purely_toroidal_field | shaped_tokamak_pressure | solovev |
|---| :--: | :--: | :--: | :--: |
| sqrtg | 3.10e-15 | 2.18e-14 | 1.24e-14 | 2.19e-15 |
| bsupu | 2.45e-15 | 2.57e-14 | 1.13e-14 | 2.17e-15 |
| bsupv | 3.08e-15 | 2.68e-14 | 1.32e-14 | 2.23e-15 |
| bsubu | 7.20e-07 | 1.27e-03 | 4.57e-05 | 2.41e-05 |
| bsubv | 1.24e-05 | 3.65e-06 | 2.59e-05 | 3.11e-05 |
| abs(B) | 3.09e-15 | 2.47e-14 | 1.25e-14 | 2.16e-15 |
| bsq = 0.5*B^2 + p | 6.20e-15 | 5.36e-14 | 2.64e-14 | 4.49e-15 |
| fsqr | 5.63e-09 | 1.42e-04 | 3.36e-08 | 3.64e-07 |
| fsqz | 1.15e-10 | 2.45e-04 | 2.67e-08 | 2.69e-07 |
| fsql | 2.17e-10 | 7.97e-09 | 6.21e-11 | 6.42e-07 |
| fsq_total | 5.73e-09 | 6.73e-05 | 6.90e-09 | 2.56e-07 |

Interpretation:
- Axisymmetric cases are at floating-point parity for geometry, ``bsup*``, and ``abs(B)``.
- Axisymmetric tomnsps/gc blocks (including lambda-force ``blmn/clmn``) match VMEC2000 to ~1e-11 abs on reduced grids; scalar residuals now match VMEC2000 at ~1e-7 or better on the standard suite, with the purely-toroidal-field case still the largest scalar residual gap (~1e-4).
- The VMEC-style update loop uses scalxc-weighted forces, and ``xc``/``v`` dumps match VMEC2000 at iter 1 in reduced-grid parity runs.
- The default benchmark path (10 iterations, ``ns=13``) now overlays VMEC2000 and vmec_jax traces for all 4 axisymmetric cases (`circular_tokamak`, `purely_toroidal_field`, `shaped_tokamak_pressure`, `solovev`).
- Non-axisymmetric parity hardening is wired into a batch comparator (`tools/diagnostics/nonaxis_parity_batch.py`) over Simsopt `input.*` files.
- Latest full-grid multigrid status at `rtol=1e-3`, `max_iter=10` (VMEC2000 exec comparator):
  - **Pass:** `input.qa_signgs1` (QA, NFP=2, NTOR=6) matches per-iteration trace through stage 1 (ns=16) iter 1 and stage 2 (ns=50) iters 1-9.
  - **Pass:** `LandremanPaul2021_QH_reactorScale_lowres` matches per-iteration trace through stage 1 (ns=12) iter 1 and stage 2 (ns=50) iters 1-9.
  - **Fail (stage 1 iter 1):** `LandremanPaul2021_QA_lowres`, `li383_low_res`, `n3are_R7.75B5.7_lowres`.
  - **Axisymmetric control:** `circular_tokamak` passes multigrid trace parity (ns=10/17, 10 iters).
- Remaining known gap: close QA_lowres/n3are/li383 early-iteration nonlinear mismatches (lambda-force/gc/tomnsps path) before extending long multigrid traces.

Full-grid parity snapshot (VMEC2000 exec comparator, `rtol=1e-3`, `max_iter=10`):

| Case | Input | Stages (ns, niter) | Status | fsq_total (VMEC/JAX) | runtime_s | Notes |
|---|---|---|---|---|---|---|
| circular_tokamak | `examples/data/input.circular_tokamak` | `[(10,5),(17,5)]` | PASS | `2.765e-02 / 2.765e-02` | 24.8 | Axisymmetric control |
| QA signgs1 | `/Users/rogeriojorge/local/test/input.qa_signgs1` | `[(16,1),(50,9)]` | PASS | `5.267e-01 / 5.267e-01` | 44.5 | Full dumps clean; wout parity `rmnc` relRMS ~8.8e-2, `zmns` relRMS ~3.2e-1 |
| LandremanPaul2021_QA_lowres | `simsopt/tests/test_files/input.LandremanPaul2021_QA_lowres` | `[(16,1),(50,1),(75,8)]` | FAIL (stage1 iter1) | `1.630e+00 / 9.245e+01` | 66.7 | fsq mismatch at iter 1 |
| LandremanPaul2021_QH_reactorScale_lowres | `simsopt/tests/test_files/input.LandremanPaul2021_QH_reactorScale_lowres` | `[(12,1),(50,9)]` | PASS | `5.591e+00 / 5.591e+00` | 56.8 | Wout parity: `rmnc` relRMS ~7.8e-2, `zmns` relRMS ~3.1e-1 |
| li383_low_res | `simsopt/tests/test_files/input.li383_low_res` | `[(16,10)]` | FAIL (stage1 iter1) | `1.489e-01 / 3.726e-01` | 23.9 | fsq mismatch at iter 1 |
| n3are_R7.75B5.7_lowres | `simsopt/tests/test_files/input.n3are_R7.75B5.7_lowres` | `[(16,4),(49,3),(100,3)]` | FAIL (stage1 iter1) | `3.192e+01 / 1.085e+13` | 65.3 | fsq blow-up at iter 1 |

Iteration trace parity (VMEC2000 executable, reduced grid):

- Single-grid axisym cases match ``fsq*`` and preconditioned scalars at machine precision for the first **10 iterations** at `--single-ns 13`.
- Full-grid multigrid axisymmetric traces are validated in the 10-iteration benchmark overlay with matching VMEC2000/vmec_jax lines for all 4 cases.
- ``up_down_asymmetric_tokamak`` (``lasym=True``) shows large bcovar/force-kernel mismatches at iter 1; nonlinear trace diverges. This is the current top lasym parity blocker.

Notes on the snapshot figures:

- The residual trace overlay uses the **VMEC2000 executable** (`xvmec2000`) per-iteration `threed1.*` table (dashed line). If the executable is not available, the plot falls back to a flat reference line at final `fsq_total`.
- The `|B|` LCFS panel uses the *same* vmecPlot2-style evaluation path for VMEC2000 and vmec_jax. Differences here reflect end-to-end solve mismatch (not a plotting artifact). For a fast single-grid parity check, use `--single-ns 13`.

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

This script compares a *fixed iteration budget* across `vmec_jax` and the **VMEC2000 executable** (`xvmec2000`). The current README figures were generated with the parity-first default reduced grid (`ns=13`) and a 10-iteration budget:

```bash
python examples/validation/benchmark_fixed_boundary_runtime_and_residuals.py \
  --iters 10 \
  --cases circular_tokamak shaped_tokamak_pressure solovev purely_toroidal_field \
  --run-vmec2000 --vmec2000-timeout 60
```

The quick settings above keep runs under ~60s per case. Increase `--iters` and/or pass larger `--ns-override`/`--vmec2000-ns-override` for longer and higher-resolution traces.

<table>
  <tr>
    <td><img src="docs/_static/figures/bench_fixed_boundary_runtime.png" width="420" /></td>
    <td><img src="docs/_static/figures/bench_fixed_boundary_residual.png" width="420" /></td>
    <td><img src="docs/_static/figures/bench_fixed_boundary_objective.png" width="420" /></td>
  </tr>
</table>

## External VMEC2000 runs (optional)

If you have the VMEC2000 Python extension installed (`vmec` + `mpi4py` + `netCDF4`), you can run VMEC2000 on an input and compare against bundled references:

```bash
python tools/diagnostics/external_vmec_driver_compare.py --case circular_tokamak
```

For per-iteration trace parity against the VMEC2000 executable (single grid, quick run):

```bash
python tools/diagnostics/vmec2000_exec_stage_trace_compare.py --case circular_tokamak --max-iter 30 --vmec-nstep 1 --single-ns 13 --dump-level lite --vmec-timeout 60
python tools/diagnostics/vmec2000_exec_stage_trace_compare.py --case nfp4_QH_warm_start --max-iter 10 --single-ns 16 --vmec-timeout 60 --rtol 1e-3
python tools/diagnostics/nonaxis_parity_batch.py --max-cases 8 --single-ns 13 --max-iter 1 --vmec-timeout 60
```

This uses a reduced grid to stay under ~1 minute; increase `--max-iter`/`--single-ns` for deeper parity checks.

To scan internal force-block parity (tomnsps + gc) and stop at the first mismatch:

```bash
python tools/diagnostics/vmec2000_exec_internal_scan.py --case circular_tokamak --single-ns 17 --iter-start 1 --iter-stop 5
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
