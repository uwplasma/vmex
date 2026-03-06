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
    <td align="center" colspan="2">Bundled-example runtime and memory ratios vs VMEC2000 (local CPU and office GPU)</td>
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

python tools/diagnostics/example_runtime_memory_matrix.py \
  --backend both \
  --runner-label cpu \
  --jax-platforms cpu \
  --vmec-exec /Users/rogeriojorge/local/test/STELLOPT/VMEC2000/Release/xvmec2000 \
  --outdir outputs/example_runtime_memory_matrix_cpu_20260306

ssh office 'source /home/rjorge/venvs/vmec_jax_gpu_bench/bin/activate && \
  cd /home/rjorge/vmec_jax_gpu_bench && \
  python tools/diagnostics/example_runtime_memory_matrix.py \
    --backend vmec_jax \
    --runner-label gpu \
    --jax-platforms cuda,cpu \
    --outdir /home/rjorge/vmec_jax_gpu_bench/outputs/example_runtime_memory_matrix_gpu_20260306'

ssh office 'source /home/rjorge/venvs/vmec_jax_gpu_bench/bin/activate && \
  cd /home/rjorge/vmec_jax_gpu_bench && \
  CUDA_VISIBLE_DEVICES=1 python tools/diagnostics/example_runtime_memory_matrix.py \
    --ids DIII-D_lasym_false,cth_like_free_bdy,cth_like_free_bdy_lasym_small \
    --backend vmec_jax \
    --runner-label gpu \
    --jax-platforms cuda,cpu \
    --outdir /home/rjorge/vmec_jax_gpu_bench/outputs/example_runtime_memory_matrix_gpu_freeb_20260306_rerun'

scp office:/home/rjorge/vmec_jax_gpu_bench/outputs/example_runtime_memory_matrix_gpu_20260306/summary.json \
  outputs/example_runtime_memory_matrix_gpu_20260306_summary.json
scp office:/home/rjorge/vmec_jax_gpu_bench/outputs/example_runtime_memory_matrix_gpu_freeb_20260306_rerun/summary.json \
  outputs/example_runtime_memory_matrix_gpu_freeb_20260306_rerun_summary.json

python tools/diagnostics/readme_runtime_compare.py \
  --cpu-summary outputs/example_runtime_memory_matrix_cpu_20260306/summary.json \
  --gpu-summary outputs/example_runtime_memory_matrix_gpu_20260306_summary.json \
    outputs/example_runtime_memory_matrix_gpu_freeb_20260306_rerun_summary.json \
  --outdir docs/_static/figures \
  --table-out outputs/readme_runtime_table_20260306.md
```

The README GPU rows were measured on the `office` benchmark clone with JAX CUDA
enabled. The free-boundary GPU rows used local copies of `mgrid_d3d_ef.nc` and
`mgrid_cth_like.nc` with the cloned input files rewritten to those local paths.

## Documentation

- `docs/quickstart.rst`: getting started
- `docs/validation.rst`: parity workflow and regression tests
- `docs/free_boundary_plan.rst`: VMEC2000-aligned free-boundary implementation plan
- `docs/performance.rst`: profiling and performance knobs
- `docs/algorithms.rst`: algorithmic overview
- `docs/equations.rst`: equations and conventions

## Bundled Example Benchmarks

Measured on 2026-03-06 using the default `run_fixed_boundary(input, verbose=False)`
path. `VMEC2000` and `vmec_jax` CPU were run locally on an Apple M2 (8 GiB RAM).
`vmec_jax` GPU was run on `office` with dual RTX A4000 GPUs.

| Example | Boundary | Topology | LASYM | VMEC2000 runtime | VMEC2000 memory | vmec_jax CPU runtime | vmec_jax CPU memory | vmec_jax GPU runtime | vmec_jax GPU memory |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DIII-D_lasym_false | free | axisym | false | 14.37s | 0.07 GiB | 428.24s | 7.36 GiB | 1602.31s | 6.23 GiB |
| ITERModel | fixed | axisym | false | 0.90s | 0.07 GiB | 5.83s | 0.91 GiB | 68.66s | 1.81 GiB |
| LandremanPaul2021_QA_lowres | fixed | non-axisym | false | 23.89s | 0.07 GiB | 16.79s | 1.84 GiB | 131.89s | 2.85 GiB |
| LandremanPaul2021_QA_lowres1 | fixed | non-axisym | false | 15.37s | 0.07 GiB | 14.86s | 1.82 GiB | 123.45s | 2.84 GiB |
| LandremanSengupta2019_section5.4_B2_A80 | fixed | axisym | false | 0.24s | 0.07 GiB | 3.90s | 0.70 GiB | 44.38s | 1.60 GiB |
| LandremanSenguptaPlunk_section5p3_low_res | fixed | axisym | true | 0.69s | 0.07 GiB | 46.77s | 4.07 GiB | 226.18s | 4.10 GiB |
| basic_non_stellsym_pressure | fixed | non-axisym | true | 2.02s | 0.07 GiB | 29.73s | 3.22 GiB | 223.36s | 3.90 GiB |
| circular_tokamak | fixed | axisym | false | 0.29s | 0.07 GiB | 5.55s | 1.18 GiB | 60.72s | 2.13 GiB |
| circular_tokamak_aspect_100 | fixed | axisym | false | 2.36s | 0.07 GiB | 9.64s | 1.58 GiB | 104.44s | 2.49 GiB |
| cth_like_fixed_bdy | fixed | axisym | false | 0.81s | 0.07 GiB | 2.43s | 0.54 GiB | 26.46s | 1.42 GiB |
| cth_like_free_bdy | free | non-axisym | false | 2.48s | 0.07 GiB | 41.83s | 1.64 GiB | 155.79s | 2.30 GiB |
| cth_like_free_bdy_lasym_small | free | non-axisym | true | 0.63s | 0.07 GiB | 37.59s | 1.47 GiB | 103.53s | 1.97 GiB |
| li383_low_res | fixed | axisym | false | 0.29s | 0.07 GiB | 3.81s | 0.99 GiB | 38.87s | 1.94 GiB |
| n3are_R7.75B5.7_lowres | fixed | axisym | false | 9.54s | 0.07 GiB | 160.06s | 6.50 GiB | 710.51s | 6.16 GiB |
| nfp4_QH_warm_start | fixed | non-axisym | false | 0.55s | 0.07 GiB | 5.14s | 1.32 GiB | 54.84s | 2.33 GiB |
| purely_toroidal_field | fixed | axisym | false | 3.21s | 0.07 GiB | 9.87s | 1.59 GiB | 104.91s | 2.49 GiB |
| shaped_tokamak_pressure | fixed | axisym | false | 0.79s | 0.07 GiB | 5.66s | 0.90 GiB | 48.58s | 1.76 GiB |
| solovev | fixed | axisym | false | 0.16s | 0.07 GiB | 2.08s | 0.48 GiB | 18.80s | 1.38 GiB |
| up_down_asymmetric_tokamak | fixed | axisym | true | 0.74s | 0.07 GiB | 6.72s | 0.89 GiB | 52.05s | 1.77 GiB |
