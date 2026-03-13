# vmec-jax

End-to-end differentiable JAX implementation of **VMEC2000** for fixed-boundary
and free-boundary ideal-MHD equilibria.

<table>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_cross_sections.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_cross_sections.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Axisymmetric: optimized fixed-boundary cross-section (VMEC2000 vs vmec_jax)</td>
    <td align="center">LandremanPaul2021_QA_lowres: optimized fixed-boundary cross-section (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_3d.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_3d.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Axisymmetric: optimized fixed-boundary 3D LCFS (VMEC2000 vs vmec_jax)</td>
    <td align="center">LandremanPaul2021_QA_lowres: optimized fixed-boundary 3D LCFS (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_bmag_surface.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_bmag_surface.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Axisymmetric: optimized fixed-boundary |B| on LCFS (VMEC2000 vs vmec_jax)</td>
    <td align="center">LandremanPaul2021_QA_lowres: optimized fixed-boundary |B| on LCFS (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_iota.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_iota.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Axisymmetric: optimized fixed-boundary iota (VMEC2000 vs vmec_jax)</td>
    <td align="center">LandremanPaul2021_QA_lowres: optimized fixed-boundary iota (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/_static/figures/readme_fsq_trace.png" width="860" /></td>
  </tr>
  <tr>
    <td align="center" colspan="2">Optimized fixed-boundary fsq_total trace (VMEC2000 vs vmec_jax) for shaped tokamak + LandremanPaul2021_QA_lowres cases</td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/_static/figures/readme_runtime_compare.png" width="860" /></td>
  </tr>
  <tr>
    <td align="center" colspan="2">Bundled fixed/free runtime comparison: VMEC2000 vs vmec_jax CPU on a reference CPU host</td>
  </tr>
</table>

## What it is

- VMEC2000-parity solver for fixed-boundary and free-boundary equilibria.
- Supports axisymmetric and non-axisymmetric configurations, with `lasym=False` and `lasym=True` for stellarator symmetry/asymmetry and up-down symmetry/asymmetry.
- Default CLI path is the same across all supported branches: `vmec_jax input.name`.
- `wout_*.nc` outputs, iteration diagnostics, and manifest-based parity sweeps are built around VMEC2000-compatible workflows.
- JAX-native kernels for geometry, transforms, and residual assembly.
- Differentiable optimization workflows are available through the Python API and bundled examples.

## Quickstart

Install (editable) and run the showcase:

```bash
python -m pip install -e .
python examples/showcase_axisym_input_to_wout.py --suite
```

CLI (VMEC2000-style executable):

```bash
vmec_jax examples/data/input.circular_tokamak
```

For fixed-boundary inputs, the default CLI path now uses the optimized
controller: it tries the fast final-grid scan route first, then escalates to
staged continuation and strict parity finishing only when the input structure
and residual history require it. Pass `--parity` to force the conservative
VMEC2000 loop. Pass `--solver-mode accelerated` to request the optimized track
explicitly.

Python driver comparison (reference track vs optimized CLI-style track):

```bash
python examples/fixed_boundary_driver_tracks.py \
  examples/data/input.circular_tokamak \
  --quiet --json
```

Run tests:

```bash
pytest -q
```

Optimization tutorials (differentiable boundary tuning):

```bash
python examples/optimization/optimize_bmag_volume.py --case circular_tokamak --opt-steps 3
python examples/optimization/explicit_target_iota_volume.py --case circular_tokamak --opt-steps 3
python examples/optimization/implicit_target_iota_volume.py --case circular_tokamak --opt-steps 3
```

## Performance vs parity

- Default runs enable the scan-based fast loop (`performance_mode=True`) with a parity guard.
- LASYM fixed-boundary stages now use a timed scan/non-scan probe on CPU and a short parity-only probe on accelerators, so the default GPU path keeps the scan fast path without paying the full non-scan timing cost.
- Quiet accelerator scan runs now use backend-aware larger chunks, capped to the remaining iterations, to reduce host/device launch overhead without changing solver parity.
- Use `--parity` or `performance_mode=False` to force the conservative parity path.
- Use `--solver-mode accelerated` to force the experimental accelerated
  fixed-boundary path, which skips parity-oriented scan probes and is judged by
  final residual/output quality rather than iteration-trace parity.
- In the current branch, accelerated fixed-boundary solves default to a single
  final-grid stage unless the caller explicitly requests `multigrid=True`. When
  staged inputs provide `NITER_ARRAY`, the accelerated single-grid path now
  carries the total staged iteration budget forward instead of silently falling
  back to `NITER`, and the CLI can automatically retry that staged schedule if
  the first final-grid solve misses the target.
- The optimized CLI controller is therefore layered:
  fast final-grid accelerated attempt first, then input-driven staged follow-up
  for explicit `NS_ARRAY` / `NITER_ARRAY`, then strict parity finish blocks
  only if the staged route still has not closed.
- On the optimized non-autodiff path, non-verbose runs now keep lighter
  iteration histories by default, and ordinary free-boundary runs avoid extra
  `scalpot` axis-diagnostic synthesis unless dump env vars are explicitly
  enabled.
- The current GPU path is fastest when the solve can stay on the scan fast path. Many of the slow GPU benchmark rows are parity-path solves, especially free-boundary cases, where VMEC2000-style restart logic, Jacobian checks, and cadence control still run as a host-controlled loop around many short float64 kernels.
- That means the GPU often sees too little work per launch to amortize host/device overhead, while the CPU benefits from lower launch latency and efficient float64 execution on these moderate-size grids. This is an implementation limit of the current parity path, not a claim that the underlying physics is inherently CPU-only.
- The accelerated-mode comparison harness lives at `tools/diagnostics/benchmark_accelerated_mode.py`.
- The parity-vs-optimized Python driver example lives at
  `examples/fixed_boundary_driver_tracks.py`.
- Details and profiling guidance live in `docs/performance.rst`.
- Merge scope and review criteria for the accelerated branch live in
  `docs/accelerated_merge_readiness.rst`.
- Parity methodology and current status live in `docs/validation.rst`.
- The cross-case parity matrix (fixed/free boundary, axisym/non-axisym, `lasym=False/True`)
  is maintained in `tools/diagnostics/parity_manifest.toml` and executed with
  `tools/diagnostics/parity_sweep_manifest.py`.

### Live NSTEP printing

By default, the VMEC2000-style iteration loop (scan or non-scan) prints every
`NSTEP` iterations using JAX's debug callback (differentiable). This keeps the
output VMEC-like while avoiding explicit host/device syncs in Python.

To disable live printing, set:

```bash
export VMEC_JAX_SCAN_PRINT=0
```

If you want minimal overhead, increase `NSTEP` in your input file. Larger
`NSTEP` means fewer host callbacks and faster runs.

Quiet runs (`--quiet` or `verbose=False`) default the scan path to a minimal
history mode (only `fsqr/fsqz/fsql` and `w_history` are kept) to reduce
host/device traffic. You can override this with:

```bash
export VMEC_JAX_SCAN_MINIMAL=0  # keep full scan diagnostics even when quiet
```

## When to use vmec_jax

- Use `vmec_jax` for fixed-boundary and free-boundary production runs, autodiff, rapid parameter sweeps, and JAX-native optimization workflows.
- Use the VMEC2000 executable as an optional parity reference or regression oracle, not as an operational requirement.

## Reproduce figures

Recreate the shaped-tokamak + LandremanPaul2021_QA_lowres VMEC2000 vs vmec_jax optimized panels shown above (single-plane cross-sections, |B| on LCFS, iota overlays, plus the fsq_total trace):

```bash
python tools/diagnostics/qh_vmec_vs_vmecjax.py \
  --input examples/data/input.shaped_tokamak_pressure \
  --wout-ref examples/data/wout_shaped_tokamak_pressure_reference.nc \
  --solve --solver vmec2000_iter --solver-mode accelerated \
  --cli-fixed-boundary-mode --jax-title "vmec_jax optimized" \
  --phi 0.0 --n-surfaces 31 \
  --prefix axisym --outdir docs/_static/figures

python tools/diagnostics/qh_vmec_vs_vmecjax.py \
  --input examples/data/input.LandremanPaul2021_QA_lowres \
  --wout-ref examples/data/wout_LandremanPaul2021_QA_lowres_reference.nc \
  --solve --solver vmec2000_iter --solver-mode accelerated \
  --cli-fixed-boundary-mode --jax-title "vmec_jax optimized" \
  --phi 0.0 --n-surfaces 31 \
  --prefix qa --outdir docs/_static/figures

python tools/diagnostics/readme_fsq_trace.py \
  --axisym-input examples/data/input.shaped_tokamak_pressure \
  --stellarator-input examples/data/input.LandremanPaul2021_QA_lowres \
  --niter 1800 --ftol 1e-13 --solver-mode accelerated \
  --outdir docs/_static/figures

python tools/diagnostics/example_runtime_memory_matrix.py \
  --backend both \
  --runner-label cpu \
  --jax-platforms cpu \
  --vmec-exec /path/to/xvmec2000 \
  --solver-mode accelerated \
  --cli-fixed-boundary-mode \
  --warm-runs 1 \
  --outdir outputs/fixed_runtime_vmec2000_accel_cpu_warm

python tools/diagnostics/example_runtime_memory_matrix.py \
  --kind freeb \
  --backend both \
  --include-external-diiid \
  --runner-label cpu \
  --jax-platforms cpu \
  --vmec-exec /path/to/xvmec2000 \
  --warm-runs 1 \
  --outdir outputs/free_runtime_vmec2000_cpu_warm

python tools/diagnostics/readme_runtime_compare.py \
  --cpu-summary outputs/fixed_runtime_vmec2000_accel_cpu_warm/summary.json \
               outputs/free_runtime_vmec2000_cpu_warm/summary.json \
  --outdir docs/_static/figures \
  --table-out outputs/readme_runtime_table.md \
  --figure-kind all \
  --plot-mode runtime
```

The exact numbers in the checked-in benchmark table will vary by machine. The
README runtime figure intentionally uses warmed fixed-boundary optimized-CLI
runs so it reflects steady-state solve cost rather than cold JAX startup
overhead. The top-level README plot is CPU-only on this branch because the GPU
path is still under active optimization and is not yet a broadly faster default
story.

## Documentation

- `docs/quickstart.rst`: getting started
- `docs/validation.rst`: parity workflow and regression tests
- `docs/free_boundary_plan.rst`: VMEC2000-aligned free-boundary implementation plan
- `docs/performance.rst`: profiling and performance knobs
- `docs/algorithms.rst`: algorithmic overview
- `docs/equations.rst`: equations and conventions

## Bundled Runtime Snapshot

Measured on 2026-03-12 using warmed serial runs on the same CPU host. The top
runtime plot now combines:

- fixed-boundary optimized CLI runs from
  `outputs/readiness_fixed_all_20260312_r3/summary.json`
- free-boundary default-path runs from
  `outputs/readiness_freeb_all_20260312_r2/summary.json`

Current checked-in summary:

- 21 total cases were exercised across fixed/free, axisymmetric /
  non-axisymmetric, and `lasym=False/True`,
- all 21 completed with `converged=True`,
- the bundled `cth_like_free_bdy_lasym_small` example was replaced with a
  convergent `lasym=True` free-boundary CTH-like fixture so the shipped matrix
  no longer has a free-boundary holdout,
- ordinary CLI and Python `run_fixed_boundary(...)` calls now choose the fast
  optimized fixed-boundary controller automatically when autodiff is not in
  play,
- the runtime plot is sorted by best VMEC2000-relative CPU speedup first
  (closest-to-VMEC2000 rows at the top),
- the free-boundary DIII-D rows are still much slower than VMEC2000 on CPU
  even after the latest optimization pass, but `DIII-D_lasym_false` improved
  materially on this branch from about `173.82s` to about `121.93s` warmed.
- the representative `cth_like_free_bdy` free-boundary case is improving on
  this branch and now runs in about `7.69s` warmed on CPU, but that is still
  well behind VMEC2000's `1.71s`.

Representative warmed CPU VMEC2000-vs-`vmec_jax` points:

| Example | VMEC2000 runtime | vmec_jax CPU runtime | Relative result |
| --- | ---: | ---: | --- |
| LandremanPaul2021_QH_reactorScale_lowres | 41.90s | 44.83s | close, VMEC2000 faster |
| LandremanPaul2021_QA_reactorScale_lowres | 36.80s | 44.71s | close, VMEC2000 faster |
| LandremanPaul2021_QA_lowres | 24.91s | 33.05s | close, VMEC2000 faster |
| cth_like_free_bdy_lasym_small | 3.44s | 12.81s | converged, VMEC2000 faster |
| DIII-D_lasym_false | 18.30s | 121.93s | converged, VMEC2000 faster |

## Accelerated Branch Reassessment

The optimized non-autodiff fixed-boundary track is now the intended default
user path for the branch:

- it is selected automatically by both the CLI and ordinary Python
  `run_fixed_boundary(...)` calls,
- it keeps convergence on the bundled fixed-boundary matrix,
- and the conservative parity / implicit-differentiation paths remain
  available when exact VMEC-style control or autodiff-specific behavior is the
  priority.

The bundled Python driver example shows the intended user flow on
`input.circular_tokamak`: parity `28.863s` vs optimized CLI-style `3.445s`,
both at `fsq_total ~ 2e-14`:

```bash
python examples/fixed_boundary_driver_tracks.py \
  examples/data/input.circular_tokamak \
  --quiet --json
```

Same-host CPU/GPU comparison remains mixed. The current branch is ready for a
PR that makes this optimized non-autodiff path the default `vmec_jax`
experience on `main`, but not for claims that it is broadly faster than
VMEC2000 on CPU or that it replaces the parity / implicit-differentiation
paths.
