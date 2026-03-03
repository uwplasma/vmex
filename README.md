# vmec-jax

Laptop-friendly, end-to-end differentiable (JAX) rewrite of **VMEC2000**, focusing on **fixed-boundary** first.

<table>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_cross_sections.png" width="420" /></td>
    <td><img src="docs/_static/figures/n3are_compare_cross_sections.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Axisymmetric: cross-section (VMEC2000 vs vmec_jax)</td>
    <td align="center">Stellarator (n3are): cross-section (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_3d.png" width="420" /></td>
    <td><img src="docs/_static/figures/n3are_compare_3d.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Axisymmetric: 3D LCFS (VMEC2000 vs vmec_jax)</td>
    <td align="center">Stellarator (n3are): 3D LCFS (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_bmag_surface.png" width="420" /></td>
    <td><img src="docs/_static/figures/n3are_compare_bmag_surface.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Axisymmetric: |B| on LCFS (VMEC2000 vs vmec_jax)</td>
    <td align="center">Stellarator (n3are): |B| on LCFS (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_iota.png" width="420" /></td>
    <td><img src="docs/_static/figures/n3are_compare_iota.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center">Axisymmetric: iota (VMEC2000 vs vmec_jax)</td>
    <td align="center">Stellarator (n3are): iota (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/_static/figures/readme_fsq_trace.png" width="860" /></td>
  </tr>
  <tr>
    <td align="center" colspan="2">fsq_total trace (VMEC2000 vs vmec_jax) for axisymmetric + n3are cases</td>
  </tr>
  <tr>
    <td colspan="2"><img src="docs/_static/figures/readme_runtime_compare.png" width="860" /></td>
  </tr>
  <tr>
    <td align="center" colspan="2">Runtime comparison (VMEC2000 vs vmec_jax, cold + warm JIT)</td>
  </tr>
</table>

## What it is

- Laptop-friendly, end-to-end differentiable (JAX) rewrite of VMEC2000 (fixed boundary first).
- Fixed-boundary parity solver for axisymmetric and non-axisymmetric cases, including `lasym=False` and `lasym=True`.
- Current fixed-boundary parity target is met at `rtol=1e-3` (with axis masking for cancellation-limited near-axis channels).
- JAX-native kernels for geometry, transforms, and residual assembly.
- Free-boundary WP0/WP1 + WP2 scaffold is implemented (typed config/state, mgrid validation + interpolation, VMEC-style `ivac/ivacskip`, boundary `Bu/Bv/B^u/B^v/bsqvac`, edge `bsq` coupling, and dual NESTOR models: VMEC2000-like dense integral assembly + fast spectral fallback).
- Next major milestone: free-boundary parity.

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

Default CLI runs use the scan-based fast loop.
Pass `--parity` to use the VMEC2000 parity loop (time-step control + restarts).

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
- Use `--parity` or `performance_mode=False` to force the conservative parity path.
- Details and profiling guidance live in `docs/performance.rst`.
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

## When to use vmec_jax vs VMEC2000

- Use `vmec_jax` for autodiff, rapid parameter sweeps, and JAX-native pipelines.
- Use VMEC2000 when you need mature free-boundary workflows today.

## Reproduce figures

Recreate the axisym + n3are VMEC2000 vs vmec_jax panels shown above (single-plane cross-sections, |B| on LCFS, iota overlays, plus the fsq_total trace):

```bash
python tools/diagnostics/qh_vmec_vs_vmecjax.py   --input examples/data/input.shaped_tokamak_pressure   --wout-ref examples/data/wout_shaped_tokamak_pressure_reference.nc   --use-wout-state --jax-title vmec_jax   --phi 0.0 --n-surfaces 31   --prefix axisym --outdir docs/_static/figures

python tools/diagnostics/qh_vmec_vs_vmecjax.py   --input examples/data/input.n3are_R7.75B5.7_lowres   --wout-ref examples/data/wout_n3are_R7.75B5.7_lowres.nc   --use-wout-state --jax-title vmec_jax   --phi 0.0 --n-surfaces 31   --prefix n3are --outdir docs/_static/figures

python tools/diagnostics/readme_fsq_trace.py   --axisym-input examples/data/input.shaped_tokamak_pressure   --stellarator-input examples/data/input.n3are_R7.75B5.7_lowres   --niter 250 --ftol 1e-14   --outdir docs/_static/figures

python tools/diagnostics/readme_runtime_compare.py   --axisym-input examples/data/input.shaped_tokamak_pressure   --stellarator-input examples/data/input.n3are_R7.75B5.7_lowres   --niter 250 --ftol 1e-14   --outdir docs/_static/figures
```

## Documentation

- `docs/quickstart.rst`: getting started
- `docs/validation.rst`: parity workflow and regression tests
- `docs/free_boundary_plan.rst`: VMEC2000-aligned free-boundary implementation plan
- `docs/performance.rst`: profiling and performance knobs
- `docs/algorithms.rst`: algorithmic overview
- `docs/equations.rst`: equations and conventions
