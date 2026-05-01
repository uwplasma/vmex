# vmec-jax

End-to-end differentiable JAX implementation of **VMEC2000** for fixed-boundary
and free-boundary ideal-MHD equilibria.

## Showcase (single-grid)

All figures below use the same **single-grid** run settings: `NS_ARRAY=151`, `NITER_ARRAY=5000`, `FTOL_ARRAY=1e-14`, `NSTEP=500`.

<table>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_cross_sections.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_cross_sections.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center"><code>ITERModel</code> cross-section (VMEC2000 vs vmec_jax)</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> cross-section (VMEC2000 vs vmec_jax)</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_iota.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_iota.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center"><code>ITERModel</code> iota (VMEC2000 vs vmec_jax vs VMEC++)</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> iota (VMEC2000 vs vmec_jax vs VMEC++)</td>
  </tr>
</table>

<p align="center">
  <img src="docs/_static/figures/readme_fsq_trace_single_grid.png" width="860" />
</p>

<p align="center">
  <img src="docs/_static/figures/readme_runtime_compare.png" width="860" />
</p>

<details>
  <summary>More visuals (single-grid)</summary>
  <br/>
  <table>
    <tr>
      <td><img src="docs/_static/figures/axisym_compare_3d.png" width="420" /></td>
      <td><img src="docs/_static/figures/qa_compare_3d.png" width="420" /></td>
    </tr>
    <tr>
      <td align="center"><code>ITERModel</code> 3D LCFS (VMEC2000 vs vmec_jax)</td>
      <td align="center"><code>LandremanPaul2021_QA_lowres</code> 3D LCFS (VMEC2000 vs vmec_jax)</td>
    </tr>
    <tr>
      <td><img src="docs/_static/figures/axisym_compare_bmag_surface.png" width="420" /></td>
      <td><img src="docs/_static/figures/qa_compare_bmag_surface.png" width="420" /></td>
    </tr>
    <tr>
      <td align="center"><code>ITERModel</code> |B| on LCFS (VMEC2000 vs vmec_jax)</td>
      <td align="center"><code>LandremanPaul2021_QA_lowres</code> |B| on LCFS (VMEC2000 vs vmec_jax)</td>
    </tr>
  </table>
</details>

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
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/showcase_axisym_input_to_wout.py --suite
```

If you want the bundled reference outputs and mgrid files, fetch the assets once:

```bash
python tools/fetch_assets.py
```

Lightweight clone (keeps full history, downloads blobs lazily):

```bash
git clone --filter=blob:none https://github.com/uwplasma/vmec_jax
```

Note: the repo history was rewritten on 2026-03-16 to remove large assets from
all commits. If you cloned before that date, please re-clone (or prune and
reset) to get the smaller history.

CLI (VMEC2000-style executable):

```bash
vmec_jax examples/data/input.circular_tokamak
```

Sanity check (verifies the console script is wired to the right interpreter):

```bash
vmec_jax --help
```

If the `vmec_jax` command is not found or raises `ModuleNotFoundError`, make sure
you installed with the same interpreter and use the module entrypoint:

```bash
python -m pip install -e .
python -m vmec_jax examples/data/input.circular_tokamak
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

Full test suite (requires netCDF assets):

```bash
python tools/fetch_assets.py
RUN_FULL=1 pytest -q
```

Advanced optimization examples live in `examples/optimization/`. They are
intended as deeper workflow templates rather than README quickstarts, so use
the fixed-boundary driver example above as the validated copy/paste entry point
and then adapt the optimization scripts for your target objective. The simplest
starting point is:

```bash
python examples/optimization/target_iota_aspect_volume.py --opt-steps 2
```

That example keeps the boundary parameterization small (`max |m|,|n| <= 1`),
targets equilibrium volume, aspect ratio, and mean iota, and defaults to the
bundled current-driven `cth_like_fixed_bdy` case so the iota channel is active.

## Performance vs parity

- Default runs aim for VMEC2000-compatible behavior while selecting the fastest stable path for the input.
- Use `--parity` (or `performance_mode=False` in Python) to force the conservative VMEC2000 loop.
- Use `--solver-mode accelerated` to force the optimized fixed-boundary controller explicitly.

Details, profiling guidance, and parity methodology:

- `docs/performance.rst`
- `docs/validation.rst`
- `tools/diagnostics/parity_manifest.toml` + `tools/diagnostics/parity_sweep_manifest.py`

## VMEC++ notes

The runtime plot includes VMEC++ (green) for context. Some inputs are not supported or do not converge under the same single-grid settings.

VMEC++ unsupported inputs in this benchmark (`lasym=True`):

- `LandremanSenguptaPlunk_section5p3_low_res`
- `basic_non_stellsym_pressure`
- `cth_like_free_bdy_lasym_small`
- `up_down_asymmetric_tokamak`

VMEC++ failed to converge (non-zero exit) on these `lasym=False` cases under the same single-grid settings:

- `DIII-D_lasym_false`
- `LandremanPaul2021_QA_reactorScale_lowres`
- `LandremanPaul2021_QH_reactorScale_lowres`
- `LandremanSengupta2019_section5.4_B2_A80`
- `cth_like_fixed_bdy`

## CLI output and `NSTEP`

The VMEC-style iteration loop prints every `NSTEP` iterations. Larger `NSTEP` means fewer print callbacks and faster runs.

To disable live printing, set:

```bash
export VMEC_JAX_SCAN_PRINT=0
```

Quiet runs (`--quiet` or `verbose=False`) default the scan path to a minimal
history mode (only `fsqr/fsqz/fsql` and `w_history` are kept) to reduce
host/device traffic. You can override this with:

```bash
export VMEC_JAX_SCAN_MINIMAL=0  # keep full scan diagnostics even when quiet
```

## When to use vmec_jax

- Use `vmec_jax` for fixed-boundary and free-boundary production runs, autodiff, rapid parameter sweeps, and JAX-native optimization workflows.
- Use the VMEC2000 executable as an optional parity reference or regression oracle, not as an operational requirement.
