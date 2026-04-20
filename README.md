# vmec-jax

Install from PyPI:

```bash
pip install vmec-jax
```

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
    <td align="center"><code>ITERModel</code> iota (VMEC2000 vs vmec_jax)</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> iota (VMEC2000 vs vmec_jax)</td>
  </tr>
</table>

<p align="center">
  <img src="docs/_static/figures/readme_fsq_trace_single_grid.png" width="860" />
</p>

<p align="center">
  <img src="docs/_static/figures/readme_runtime_compare.png" width="860" />
</p>

**Cold vs warm runtime**: the *cold* bar includes XLA JIT compilation on the first call (one-time cost per process); the *warm* bar is the steady-state solve time for all subsequent calls in the same process, with the compiled kernels already in-memory. VMEC2000 is a pre-compiled Fortran binary and therefore has no compilation overhead — it is always effectively "cold". The warm vmec_jax time is the fair comparison for repeated solves (e.g., in an optimization loop). Starting from v0.2, vmec_jax automatically caches compiled XLA kernels to disk (`~/.cache/vmec_jax/jax_cache`), so that *cold* runs in a fresh process on the same machine after the first invocation benefit from the on-disk cache and approach warm-run speed.

## More visuals (single-grid)

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

## What it is

- VMEC2000-parity solver for fixed-boundary and free-boundary equilibria.
- Supports axisymmetric and non-axisymmetric configurations, with `lasym=False` and `lasym=True` for stellarator symmetry/asymmetry and up-down symmetry/asymmetry.
- Default CLI path is `vmec_jax input.name`.
- `wout_*.nc` outputs, iteration diagnostics, and manifest-based parity sweeps are built around VMEC2000-compatible workflows.
- JAX-native kernels for geometry, transforms, and residual assembly.
- Differentiable optimization workflows are available through the Python API and bundled examples.

## Quickstart

Install directly from PyPI:

```bash
pip install vmec-jax
```

Install and run the showcase:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/showcase_axisym_input_to_wout.py --suite
```

If you want a release-style non-editable install instead, use:

```bash
python -m pip install .
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

Generate diagnostic plots from any `wout_*.nc` file (replicates `vmecPlot2.py`):

```bash
vmec_jax --plot wout_circular_tokamak.nc
# Writes: wout_circular_tokamak_VMECparams.pdf  (9-panel: iota, pressure, buco, bvco, jcuru, jcurv, DMerc, |B|)
#         wout_circular_tokamak_poloidal_plot.png  (R-Z cross-sections at multiple toroidal angles)
#         wout_circular_tokamak_VMECsurfaces.pdf   (nested flux-surface cross-sections)
#         wout_circular_tokamak_VMEC_3Dplot.png    (3-D LCFS surface coloured by |B|)

# Optional: specify output directory
vmec_jax --plot wout_nfp4_QH.nc --outdir figures/
```

Or from Python:

```python
import vmec_jax as vj
vj.plot_wout("wout_nfp4_QH.nc", outdir="figures/")
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

## Quasi-helical symmetry optimization (discrete-adjoint)

`examples/optimization/qh_fixed_resolution_jax.py` demonstrates an
end-to-end fixed-boundary QH optimization using the built-in exact
discrete-adjoint Jacobian — no finite differences, no SIMSOPT dependency.
Input parameters sit at the top of the script as variables (no argparse),
mirroring the SIMSOPT workflow:

```bash
python examples/optimization/qh_fixed_resolution_jax.py   # uses MAX_MODE=2, MAX_NFEV=15
```

Quasisymmetry objective reduced from **0.311 → 0.212** (32 %) at `max_mode=1`
and from **0.311 → 0.055** (82 %) at `max_mode=2` — **dramatically better than
the classic SIMSOPT + VMEC2000 + finite-differences reference of 0.214** at the
same `max_mode=1` budget. The table below shows results for three mode-number
cutoffs:

| `max_mode` | DOFs | Wall time | QS initial | QS final | Reduction |
|:----------:|:----:|:---------:|:----------:|:--------:|:---------:|
| 1          | 4    | ~33 s     | 0.311      | 0.212    | 32 %      |
| 2          | 14   | ~63 s     | 0.311      | 0.055    | 82 %      |
| 3          | 14 ¹ | ~68 s     | 0.311      | 0.055    | 82 %      |

¹ Same parameter space as `max_mode=2` for this input (starting boundary has no
non-zero modes beyond `|m|=1`).

### max_mode = 1 — 32 % QS reduction in 33 s

<table>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/boundary_comparison.png" width="540" /></td>
    <td><img src="docs/_static/figures/qh_opt/objective_history.png" width="340" /></td>
  </tr>
  <tr>
    <td align="center">3-D LCFS coloured by |B| — initial (left) vs optimized (right)</td>
    <td align="center">Objective and aspect ratio per Jacobian evaluation</td>
  </tr>
</table>

<p align="center">
  <img src="docs/_static/figures/qh_opt/bmag_surface.png" width="700" />
</p>
<p align="center">|B| contour lines on LCFS — initial (top) vs optimized (bottom).
The contours show quasi-helical alignment: |B| depends mainly on <em>m θ − n φ</em>;
the optimized contours are clearly more helically closed.</p>

### max_mode = 2 — 82 % QS reduction in 63 s

<table>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/mode2/boundary_comparison.png" width="540" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode2/objective_history.png" width="340" /></td>
  </tr>
  <tr>
    <td align="center">3-D LCFS coloured by |B| — initial (left) vs optimized (right)</td>
    <td align="center">Objective and aspect ratio per Jacobian evaluation</td>
  </tr>
</table>

<p align="center">
  <img src="docs/_static/figures/qh_opt/mode2/bmag_surface.png" width="700" />
</p>

### max_mode = 3 — 82 % QS reduction in 68 s (same DOFs as max_mode=2)

<table>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/mode3/boundary_comparison.png" width="540" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode3/objective_history.png" width="340" /></td>
  </tr>
  <tr>
    <td align="center">3-D LCFS coloured by |B| — initial (left) vs optimized (right)</td>
    <td align="center">Objective and aspect ratio per Jacobian evaluation</td>
  </tr>
</table>

<p align="center">
  <img src="docs/_static/figures/qh_opt/mode3/bmag_surface.png" width="700" />
</p>

Plots are regenerated by running the plotting helper after the optimization:

```bash
python examples/optimization/plot_qh_optimization_results.py \
    --output-dir results/qh_opt
```

## Performance vs parity

- Default runs aim for VMEC2000-compatible behavior while selecting the fastest stable path for the input.
- Use `--parity` (or `performance_mode=False` in Python) to force the conservative VMEC2000 loop.
- Use `--solver-mode accelerated` to force the optimized fixed-boundary controller explicitly.

Details, profiling guidance, and parity methodology:

- `docs/performance.rst`
- `docs/validation.rst`
- `tools/diagnostics/parity_manifest.toml` + `tools/diagnostics/parity_sweep_manifest.py`

## VMEC++ notes

The current runtime benchmark compares vmec_jax against VMEC2000. VMEC++ is not included in this benchmark.

When VMEC++ is available, it can be added to the runtime plot via `--cpu-summary` entries with `backend=vmecpp`. Some inputs are not supported or do not converge under the same single-grid settings:

VMEC++ unsupported inputs (`lasym=True`):

- `LandremanSenguptaPlunk_section5p3_low_res`
- `basic_non_stellsym_pressure`
- `cth_like_free_bdy_lasym_small`
- `up_down_asymmetric_tokamak`

VMEC++ known non-convergence on these `lasym=False` cases under the same single-grid settings:

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
