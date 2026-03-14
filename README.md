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
    <td colspan="2"><img src="docs/_static/figures/readme_fsq_trace_single_grid.png" width="860" /></td>
  </tr>
  <tr>
    <td align="center" colspan="2">Single-grid fixed-boundary fsq_total trace (VMEC2000 vs vmec_jax) for ITERModel + LandremanPaul2021_QA_lowres. Run settings: NS_ARRAY=151, NITER_ARRAY=5000, FTOL_ARRAY=1e-14, NSTEP=500. vmec_jax was run as <code>vmec_jax &lt;inputfile&gt;</code> (no flags); NSTEP was set to 1 in a temporary copy only to record the per-iteration trace.</td>
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
- Default CLI path is `vmec_jax input.name`.
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
- Use `--solver-mode accelerated` to force the optimized fixed-boundary path
  explicitly, which skips parity-oriented scan probes and is judged by final
  residual/output quality rather than iteration-trace parity.
- Accelerated fixed-boundary solves default to a single
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
- Implementation notes and merge rationale for the optimized controller live in
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
