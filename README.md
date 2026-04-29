# vmec-jax

[![PyPI version](https://img.shields.io/pypi/v/vmec-jax.svg)](https://pypi.org/project/vmec-jax/)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://github.com/uwplasma/vmec_jax/blob/main/pyproject.toml)
[![License](https://img.shields.io/github/license/uwplasma/vmec_jax)](https://github.com/uwplasma/vmec_jax/blob/main/LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/uwplasma/vmec_jax/ci.yml?branch=main&label=ci)](https://github.com/uwplasma/vmec_jax/actions/workflows/ci.yml)
[![Coverage](https://codecov.io/gh/uwplasma/vmec_jax/graph/badge.svg?branch=main)](https://codecov.io/gh/uwplasma/vmec_jax?branch=main)
[![Docs](https://img.shields.io/readthedocs/vmec-jax/latest?label=docs)](https://vmec-jax.readthedocs.io/en/latest/)
[![PyPI downloads](https://img.shields.io/pypi/dm/vmec-jax)](https://pypi.org/project/vmec-jax/)

End-to-end differentiable JAX implementation of **VMEC2000** for fixed-boundary
and free-boundary ideal-MHD equilibria.

## Install

```bash
pip install vmec-jax
```

QI optimization uses `booz_xform_jax` for the differentiable Boozer transform:

```bash
pip install "vmec-jax[qi]"
```

Developer (editable) install:

```bash
git clone https://github.com/uwplasma/vmec_jax
pip install -e "vmec_jax[qi]"
```

## Choosing CPU or GPU

`vmec_jax` follows the JAX backend you select. If you installed CPU-only JAX,
runs use CPU. If you installed GPU-enabled JAX and select a GPU backend, runs
use GPU; vmec_jax does not silently force those runs back to CPU.

```bash
# Check what JAX will use.
python -c "import jax; print(jax.default_backend()); print(jax.devices())"

# Force CPU for one command.
JAX_PLATFORMS=cpu vmec_jax input.nfp4_QH_warm_start

# Force an accelerator backend after installing GPU-enabled JAX.
JAX_PLATFORM_NAME=gpu vmec_jax input.nfp4_QH_warm_start

# For NVIDIA CUDA specifically, this is also valid.
JAX_PLATFORMS=cuda vmec_jax input.nfp4_QH_warm_start
```

From Python, leave `solver_device` unset to inherit JAX's default backend, or
pass `solver_device="cpu"` / `solver_device="gpu"` explicitly:

```python
import vmec_jax as vj

run_gpu = vj.run_fixed_boundary("input.nfp4_QH_warm_start", solver_device="gpu")
run_cpu = vj.run_fixed_boundary("input.nfp4_QH_warm_start", solver_device="cpu")
```

For GPU runs, vmec_jax defaults `XLA_PYTHON_CLIENT_PREALLOCATE=false` before
JAX import so the allocator grows on demand. This avoids GPU memory contention
between optimization workers and was faster in the exact-Jacobian GPU profile.
Set `XLA_PYTHON_CLIENT_PREALLOCATE=true` before import if you explicitly want
JAX's default preallocation behavior.

## Usage

Run the solver (VMEC2000-style CLI):

```bash
vmec_jax input.nfp4_QH_warm_start        # → wout_nfp4_QH_warm_start.nc
```

Generate diagnostic plots from any `wout_*.nc` (four-panel output, replicates `vmecPlot2.py`):

```bash
vmec_jax --plot wout_nfp4_QH_warm_start.nc           # saves in same directory
vmec_jax --plot wout_nfp4_QH_warm_start.nc --outdir figures/
```

From Python:

```python
import vmec_jax as vj

# Run a fixed-boundary solve
run = vj.run_fixed_boundary("input.nfp4_QH_warm_start")

# Run a free-boundary solve
freeb = vj.run_free_boundary("input.cth_like_free_bdy_lasym_small")

# Plot any wout file (produces *_VMECparams.pdf, *_poloidal_plot.png, *_VMECsurfaces.pdf, *_VMEC_3Dplot.png)
vj.plot_wout("wout_nfp4_QH_warm_start.nc", outdir="figures/")
```

Run tests:

```bash
pytest -q
```

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
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_3d.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_3d.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center"><code>ITERModel</code> 3D LCFS</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> 3D LCFS</td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/axisym_compare_bmag_surface.png" width="420" /></td>
    <td><img src="docs/_static/figures/qa_compare_bmag_surface.png" width="420" /></td>
  </tr>
  <tr>
    <td align="center"><code>ITERModel</code> |B| on LCFS</td>
    <td align="center"><code>LandremanPaul2021_QA_lowres</code> |B| on LCFS</td>
  </tr>
</table>

<p align="center">
  <img src="docs/_static/figures/readme_fsq_trace_single_grid.png" width="860" />
</p>

<p align="center">
  <img src="docs/_static/figures/readme_runtime_compare.png" width="860" />
</p>

**Cold vs warm runtime**: the *cold* bar includes XLA JIT compilation on the first call (one-time cost per process); the *warm* bar is the steady-state solve time for subsequent calls in the same process. VMEC2000 has no compilation overhead, so it is always effectively cold. `vmec_jax` enables JAX's persistent compilation cache by default under `~/.cache/vmec_jax/jax_cache` so repeated cold-process runs can reuse compiled kernels; set `VMEC_JAX_COMPILATION_CACHE=0` to disable it or `VMEC_JAX_COMPILATION_CACHE_DIR=/path/to/cache` to choose a different location.

## Optimization Internals

The fixed-boundary optimization examples expose the problem construction
directly in Python: VMEC resolution, active boundary coefficients, objective
blocks, weights, continuation policy, ESS scaling, and the outer optimizer are
all top-level variables in the scripts.  No SIMSOPT wrapper layer is required.

For a boundary parameter vector `x`, vmec_jax solves the VMEC residual
`F(y, x) = 0` for the equilibrium state `y`, then differentiates objective
residuals `r(y(x), x)` with the exact discrete-adjoint/tape path.  Instead of
finite-differencing each boundary DOF, vmec_jax records a checkpoint tape of
the nonlinear VMEC iteration and replays it with JAX JVP/VJP rules.  The dense
least-squares Jacobian used by the examples is exact to machine precision and
has cost comparable to a small number of forward solves, not one VMEC solve per
boundary DOF.

Details: [discrete adjoint](docs/discrete_adjoint.rst),
[optimization guide](docs/optimization.rst), and
[SIMSOPT comparison](docs/simsopt_comparison.rst).

## Quasi-helical Symmetry Optimization

`examples/optimization/qh_fixed_resolution_jax.py` demonstrates an end-to-end
fixed-boundary QH optimization using the built-in **exact discrete-adjoint Jacobian**
— no finite differences, no SIMSOPT dependency.

The script is intentionally written in the same teaching style as SIMSOPT's
`QH_fixed_resolution.py`: choose the VMEC resolution directly in Python, choose
the active boundary coefficients directly, build an `OBJECTIVES` list, then run
the visible setup → continuation stages → optimizer → save/plot workflow. It is
a standalone script, not an argparse entry point or a hidden wrapper call.

```bash
python examples/optimization/qh_fixed_resolution_jax.py   # edit MAX_MODE at the top
```

Key top-level controls in the script:

- `VMEC_MPOL`, `VMEC_NTOR`: solver resolution
- `MAX_MODE`: boundary parameterization richness
- `OBJECTIVES`: explicit aspect + QS residual blocks
- `METHOD`: `"gauss_newton"` or `"scipy"`
- `SCIPY_TR_SOLVER`: SciPy trust-region linear solver (`"lsmr"` by default for the QA/QH examples)
- `USE_MODE_CONTINUATION`: staged solves for higher-mode runs
- `USE_ESS`, `ALPHA`: optional exponential spectral scaling

Add a new target by appending an objective term:

```python
OBJECTIVES = [
    aspect_objective(TARGET_ASPECT, ASPECT_WEIGHT),
    quasisymmetry_objective(helicity_m=1, helicity_n=-1, surfaces=SURFACES, weight=QS_WEIGHT),
    ObjectiveTerm("custom", lambda ctx, state: your_metric(ctx, state), target=0.0, weight=0.1),
]
```

When `max_mode` exceeds the modes present in the input file, vmec_jax automatically
extends the boundary to include the requested harmonics at zero amplitude
(`vj.extend_boundary_for_max_mode`), matching SIMSOPT's `fixed_range()` behaviour.
All runs use consistent VMEC resolution `mpol = ntor = 5` so the initial QS metric
is normalised identically across `max_mode` values.

| `max_mode` | DOFs | Policy | QS initial | QS final | Reduction | Objective final | Wall time ¹ |
|:----------:|:----:|:------:|:----------:|:--------:|:---------:|:---------------:|:-----------:|
| 1          |  8   | continuation, no ESS | 0.303 | 0.214 | 30 % | `0.216` | ~2.2 min |
| 2          | 24   | continuation, no ESS | 0.303 | `3.72e-3` | 98.8 % | `3.72e-3` | ~8.5 min |
| 3          | 48   | continuation, no ESS | 0.303 | **`1.37e-3`** | **99.5 %** | **`1.37e-3`** | ~10.7 min |

¹ Wall time on Apple M-series (warm-cache subsequent runs are faster).

With only 8 DOFs (`max_mode=1`) the boundary deformation space is too limited
to reach a deep quasi-helical minimum. `max_mode=2` already gives a strong QH
solution, and the current `max_mode=3` continuation run improves it further.

**vmec_jax vs SIMSOPT**: vmec_jax uses an exact discrete-adjoint Jacobian
(one batched JVP pass ≈ 1–2 forward solves regardless of DOF count) while
SIMSOPT + VMEC2000 uses finite differences (*n*_DOFs × 1 forward solve per
Jacobian).  For a detailed comparison of algorithms, runtimes, and memory,
see [docs/simsopt_comparison.rst](docs/simsopt_comparison.rst).

<table>
  <tr>
    <th align="center">max_mode = 1 &nbsp;(8 DOFs, 30 % QS reduction)</th>
    <th align="center">max_mode = 2 &nbsp;(24 DOFs, 99 % QS reduction)</th>
    <th align="center">max_mode = 3 &nbsp;(48 DOFs, 99.7 % QS reduction)</th>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode2/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode3/boundary_comparison.png" /></td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qh_opt/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode2/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qh_opt/mode3/bmag_surface.png" /></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/_static/figures/qh_opt/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qh_opt/mode2/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qh_opt/mode3/objective_history.png" /></td>
  </tr>
</table>

The |B| contour plots show quasi-helical alignment after optimization: contour lines
become increasingly helical (aligned with *m θ − n φ* = const). The ζ axis spans
one field period (0 → 2π/nfp).

The current exact standalone path keeps improving through `max_mode=3`, with
the 48-DOF continuation run reaching `~1.4e-3` total objective and QS.

Regenerate plots after running the optimization:

```bash
python examples/optimization/plot_qh_optimization_results.py --output-dir results/qh_opt
```

## Quasi-axisymmetric Optimization

`examples/optimization/qa_fixed_resolution_jax_ess.py` optimizes an nfp=2 QA
equilibrium for aspect ratio, mean iota, and QA symmetry residuals.

Like the QH script, it exposes the problem construction directly in Python:
VMEC resolution, active boundary DOFs, the three objective blocks, weights,
continuation policy, ESS settings, and the outer optimizer are all top-level
variables in the file.

```bash
python examples/optimization/qa_fixed_resolution_jax_ess.py   # edit MAX_MODE at the top
```

When `max_mode` exceeds the modes in the input file, vmec_jax automatically extends
the boundary to include those harmonics at zero amplitude (`vj.extend_boundary_for_max_mode`).
All runs use consistent VMEC resolution `mpol = ntor = 5`.
Objectives: aspect ratio (target 6.0) + mean iota (target 0.41) + QA symmetry residuals.
The current standalone QA path uses exact residuals + exact discrete-adjoint
Jacobians with `scipy.optimize.least_squares`. For `max_mode > 1`, the script
can use staged mode continuation: it solves the lower-mode QA problem first,
then lifts that solution into the richer boundary space before running the final
stage. The full sweep also tests direct-start and ESS variants.

| `max_mode` | DOFs | Policy | Eval used | Aspect final | Mean iota final | QS final | Objective final | Wall time ¹ |
|:----------:|:----:|:------:|:---------:|:------------:|:---------------:|:--------:|:---------------:|:-----------:|
| 1          |  8   | input deck, no ESS | 27 | 6.0024 | 0.3942 | `9.04e-3` | `9.29e-3` | ~315 s |
| 2          | 24   | continuation, no ESS | 52 | **6.0000** | 0.4095 | `1.46e-4` | `1.46e-4` | ~801 s |
| 3          | 48   | continuation, no ESS | 64 | **6.0000** | **0.4099** | **`7.61e-6`** | **`7.62e-6`** | ~1150 s |

¹ Wall time on Apple M-series.

On the latest fresh standalone rerun, staged continuation is decisive for QA.
Direct-start `max_mode=3` stays in a poor basin, while continuation reaches a
deep QA minimum; the best displayed QA run is the no-ESS `max_mode=3`
continuation case.

<table>
  <tr>
    <th align="center">max_mode = 1 &nbsp;(8 DOFs, exact SciPy + adjoint)</th>
    <th align="center">max_mode = 2 &nbsp;(24 DOFs, exact SciPy + adjoint, continuation)</th>
    <th align="center">max_mode = 3 &nbsp;(48 DOFs, exact SciPy + adjoint, continuation)</th>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qa_opt/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode2/boundary_comparison.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode3/boundary_comparison.png" /></td>
  </tr>
  <tr>
    <td><img src="docs/_static/figures/qa_opt/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode2/bmag_surface.png" /></td>
    <td><img src="docs/_static/figures/qa_opt/mode3/bmag_surface.png" /></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/_static/figures/qa_opt/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qa_opt/mode2/objective_history.png" /></td>
    <td align="center"><img src="docs/_static/figures/qa_opt/mode3/objective_history.png" /></td>
  </tr>
</table>

## Quasi-poloidal Symmetry Optimization

`examples/optimization/qp_fixed_resolution_jax_ess.py` uses the same exact
fixed-boundary optimizer with helicity `(M, N) = (0, -1)`.  It starts from the
QH warm-start input and targets aspect ratio 7 with an absolute-iota lower
bound of 0.31.  Edit the top-level variables in the script to choose
`MAX_MODE`, `USE_ESS`, `USE_MODE_CONTINUATION`, and the VMEC/optimizer budgets.

```bash
python examples/optimization/qp_fixed_resolution_jax_ess.py
```

In the current CPU sweep, QP is the least mature of the three quasisymmetry
examples: `max_mode=2` direct-start/no-ESS gives the best objective in the
small benchmark, while `max_mode=3` benefits from ESS but does not yet improve
monotonically over mode 2.  The full policy matrix is kept in the docs so this
regression remains visible instead of hidden.

| `max_mode` | Best current policy | Objective final | QS/QP final | Aspect final | Iota final | Wall time |
|:----------:|:--------------------|:---------------:|:-----------:|:------------:|:----------:|:---------:|
| 1 | direct or continuation, no ESS | `0.600` | `0.591` | 7.089 | -0.308 | ~0.5 min |
| 2 | direct, no ESS | **`4.60e-2`** | **`4.60e-2`** | 7.006 | -0.583 | ~0.7 min |
| 3 | direct + ESS | `9.60e-2` | `9.25e-2` | 7.057 | -0.309 | ~0.8 min |

## Quasi-isodynamic Optimization

`examples/optimization/qi_fixed_resolution_jax_ess.py` uses
`vmec_jax.quasi_isodynamic`, a smooth Boozer-space QI residual evaluated through
`booz_xform_jax`.  The documented workflow first runs a same-mode QP preseed
and then refines with the QI objective; this avoids leaving the solution in the
QH warm-start basin and gives visibly non-QH `|B|` contours.

```bash
python examples/optimization/qi_fixed_resolution_jax_ess.py
```

The current QI objective is already differentiable end-to-end through VMEC and
Boozer-space post-processing.  Several cases stop by the configured `max_nfev`
before satisfying SciPy's convergence criterion, so the table reports the best
objective values rather than claiming all rows are fully converged.

| `max_mode` | Best current policy | Objective final | QI final | Aspect final | Wall time |
|:----------:|:--------------------|:---------------:|:--------:|:------------:|:---------:|
| 1 | direct or continuation, no ESS | `1.19e-2` | `1.19e-2` | 7.001 | ~0.7 min |
| 2 | continuation, no ESS | **`6.85e-4`** | **`6.85e-4`** | 7.000 | ~1.4 min |
| 3 | continuation + ESS | `1.25e-3` | `1.25e-3` | 7.001 | ~1.9 min |

## Finite-beta Stage-one Optimization

The finite-beta examples reproduce the VMEC-only stage-one finite-beta
workflows without SIMSOPT and without coils.  They use bundled finite-pressure
and current-driven input decks and add differentiable residuals for aspect
ratio, iota bounds, volume-averaged field proxy, total beta, plus the field
quality objective (QA/QH quasisymmetry or QI).

```bash
python examples/optimization/qa_optimization_finite_beta.py
python examples/optimization/qh_optimization_finite_beta.py
python examples/optimization/qi_optimization_finite_beta.py
```

The scripts save `input.initial`, `input.final`, `wout_initial.nc`,
`wout_final.nc`, and `history.json`.  Full differentiable Mercier `DMerc` and
Redl bootstrap-current mismatch residuals are the next finite-beta extensions;
the current examples keep the stage-one structure and current-profile support
in place so those terms can be added without changing the user workflow.
The QI script exposes `QI_MBOZ`, `QI_NBOZ`, `QI_NPHI`, `QI_NALPHA`, and
`QI_N_BOUNCE` at the top; the defaults are first-run diagnostic settings, and
should be increased for final research-quality QI refinements.

## QA/QH/QP/QI Optimization Policy Sweep

The panel below compares the exact standalone optimizer on CPU and GPU for
four targets: QA, QH, QP, and QI. Columns increase the boundary space from
`max_mode = 1` to `max_mode = 3`. Rows compare staged mode continuation against
direct-start mode expansion. Blue curves use unscaled boundary DOFs; orange
curves use ESS with `alpha = 2.5`. Solid lines met the optimizer success
criterion; dashed lines reached the configured `max_nfev` before satisfying
the optimizer convergence tolerances, not wall-clock timeouts.

<p align="center">
  <img src="docs/_static/figures/qs_ess_objective_panel_all_policies.png" width="980" />
</p>

The QA input includes `1e-5` seeds for the mode-1 boundary terms so the iota
residual has a useful direction. With the corrected bounded solve budgets, QA
continuation reaches the target-iota basin on both CPU and GPU. Direct QA with
ESS also reaches `iota ~= 0.409`; direct QA without ESS now leaves the zero-iota
branch, but remains a weak policy for `max_mode=3`.

QH and QP use the quasisymmetry residual with different helicities. QI uses
`vmec_jax.quasi_isodynamic`, a smooth Boozer-space residual built through
`booz_xform_jax`. The QI rows first run a same-mode QP preseed and then refine
with the QI residual; this avoids the QH warm-start basin and gives visibly
non-QH `|B|` contours while keeping the objective differentiable.

Final CPU states for the continuation and direct-start policies are shown
below. The `|B|` panels use line contours on the LCFS, with a separate colorbar
for each panel because the field ranges are not identical across aspect-ratio
changes.

<p align="center">
  <img src="docs/_static/figures/qs_ess_final_state_atlas_continuation.png" width="980" />
</p>

<p align="center">
  <img src="docs/_static/figures/qs_ess_final_state_atlas_direct.png" width="980" />
</p>

Recreate the full CPU/GPU sweep:

```bash
JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3 --ess both
JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both
JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy continuation --problems qa,qh,qp,qi --modes 1,2,3 --ess both
JAX_PLATFORM_NAME=gpu python examples/optimization/generate_qs_ess_sweep.py --backend-label gpu --solver-device gpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both
```

The default per-case timeout is 600 s. GPU sweeps use exact/replay callbacks
with calibrated optimizer budgets (`inner_max_iter = trial_max_iter = 120`,
`ftol = trial_ftol = 1e-8` for deck-controlled QA/QH cases) so production
sweeps do not differentiate through 1500 strict VMEC iterations at every
accepted point. Add `--diagnostic-budgets` only when you explicitly want the
older bounded quick-look GPU diagnostics, and use `--case-timeout-s 0` only for
an unbounded local diagnostic run.

Recreate just the CPU direct-start rows:

```bash
JAX_PLATFORMS=cpu python examples/optimization/generate_qs_ess_sweep.py --backend-label cpu --solver-device cpu --policy direct --problems qa,qh,qp,qi --modes 1,2,3 --ess both
```

Render the README/docs panels and tables:

```bash
python examples/optimization/render_qs_ess_publication_panel.py
```

Run individual examples by editing top-level variables in each script:

```bash
python examples/optimization/qa_fixed_resolution_jax_ess.py
python examples/optimization/qh_fixed_resolution_jax.py
python examples/optimization/qp_fixed_resolution_jax_ess.py
python examples/optimization/qi_fixed_resolution_jax_ess.py
```

More figures, CSV/JSON summaries, and reproduction notes are in
[docs/optimization_sweep_results.rst](docs/optimization_sweep_results.rst).

CPU wall-time summary for the plotted runs:

| Backend | Problem | Policy | max_mode | ESS | Status | Final J | Aspect | Iota | nfev | Wall min |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| CPU | QA | continuation | 1 | no | ok | 9.29e-03 | 6.002 | 0.3940 | 20 | 2.5 |
| CPU | QA | continuation | 1 | yes | ok | 9.29e-03 | 6.002 | 0.3940 | 20 | 2.7 |
| CPU | QA | continuation | 2 | no | ok | 2.66e-04 | 6.000 | 0.4090 | 45 | 7.2 |
| CPU | QA | continuation | 2 | yes | ok | 1.56e-04 | 6.000 | 0.4099 | 48 | 8.5 |
| CPU | QA | continuation | 3 | no | ok | 7.62e-06 | 6.000 | 0.4099 | 64 | 19.2 |
| CPU | QA | continuation | 3 | yes | ok | 2.16e-05 | 6.000 | 0.4099 | 71 | 25.1 |
| CPU | QH | continuation | 1 | no | ok | 2.16e-01 | 7.049 | - | 9 | 2.2 |
| CPU | QH | continuation | 1 | yes | ok | 2.16e-01 | 7.049 | - | 9 | 2.3 |
| CPU | QH | continuation | 2 | no | ok | 3.72e-03 | 7.001 | - | 28 | 8.5 |
| CPU | QH | continuation | 2 | yes | ok | 4.32e-03 | 7.000 | - | 29 | 6.3 |
| CPU | QH | continuation | 3 | no | ok | 1.37e-03 | 7.000 | - | 32 | 10.7 |
| CPU | QH | continuation | 3 | yes | ok | 1.38e-03 | 7.000 | - | 33 | 8.1 |
| CPU | QP | continuation | 1 | no | stopped | 6.00e-01 | 7.089 | -0.3083 | 20 | 0.5 |
| CPU | QP | continuation | 1 | yes | stopped | 6.00e-01 | 7.089 | -0.3083 | 20 | 0.5 |
| CPU | QP | continuation | 2 | no | stopped | 2.97e-01 | 7.077 | -0.3097 | 28 | 0.9 |
| CPU | QP | continuation | 2 | yes | stopped | 4.43e-01 | 7.087 | -0.3102 | 28 | 0.8 |
| CPU | QP | continuation | 3 | no | ok | 3.20e-01 | 7.077 | -0.3023 | 26 | 1.0 |
| CPU | QP | continuation | 3 | yes | stopped | 2.74e-01 | 7.063 | -0.3105 | 36 | 1.4 |
| CPU | QI | continuation | 1 | no | ok | 1.19e-02 | 7.001 | - | 28 | 0.7 |
| CPU | QI | continuation | 1 | yes | ok | 1.19e-02 | 7.001 | - | 28 | 0.7 |
| CPU | QI | continuation | 2 | no | stopped | 6.85e-04 | 7.000 | - | 40 | 1.4 |
| CPU | QI | continuation | 2 | yes | ok | 2.37e-02 | 7.009 | - | 32 | 0.9 |
| CPU | QI | continuation | 3 | no | stopped | 1.18e-02 | 6.996 | - | 38 | 1.6 |
| CPU | QI | continuation | 3 | yes | stopped | 1.25e-03 | 7.001 | - | 48 | 1.9 |
| CPU | QA | direct | 1 | no | ok | 9.29e-03 | 6.002 | 0.3940 | 20 | 2.5 |
| CPU | QA | direct | 1 | yes | ok | 9.29e-03 | 6.002 | 0.3940 | 20 | 2.7 |
| CPU | QA | direct | 2 | no | ok | 4.50e-04 | 5.999 | 0.4066 | 18 | 18.6 |
| CPU | QA | direct | 2 | yes | stopped | 1.58e-04 | 6.000 | 0.4095 | 40 | 14.9 |
| CPU | QA | direct | 3 | no | ok | 1.76e-02 | 6.007 | 0.3228 | 24 | 1.4 |
| CPU | QA | direct | 3 | yes | stopped | 1.46e-04 | 6.000 | 0.4093 | 24 | 1.6 |
| CPU | QH | direct | 1 | no | ok | 2.16e-01 | 7.049 | - | 9 | 2.2 |
| CPU | QH | direct | 1 | yes | ok | 2.16e-01 | 7.049 | - | 9 | 2.3 |
| CPU | QH | direct | 2 | no | ok | 3.45e-03 | 7.001 | - | 28 | 10.2 |
| CPU | QH | direct | 2 | yes | ok | 4.00e-03 | 7.001 | - | 20 | 5.6 |
| CPU | QH | direct | 3 | no | ok | 4.29e-03 | 6.999 | - | 15 | 9.5 |
| CPU | QH | direct | 3 | yes | ok | 3.27e-03 | 6.999 | - | 20 | 9.2 |
| CPU | QP | direct | 1 | no | stopped | 6.00e-01 | 7.089 | -0.3083 | 20 | 0.5 |
| CPU | QP | direct | 1 | yes | stopped | 6.00e-01 | 7.089 | -0.3083 | 20 | 0.5 |
| CPU | QP | direct | 2 | no | ok | 4.60e-02 | 7.006 | -0.5828 | 19 | 0.7 |
| CPU | QP | direct | 2 | yes | stopped | 5.67e-02 | 7.013 | -0.3097 | 20 | 0.7 |
| CPU | QP | direct | 3 | no | ok | 5.35e-01 | 7.064 | -1.1401 | 17 | 0.7 |
| CPU | QP | direct | 3 | yes | ok | 9.60e-02 | 7.057 | -0.3092 | 20 | 0.8 |
| CPU | QI | direct | 1 | no | ok | 1.19e-02 | 7.001 | - | 28 | 0.7 |
| CPU | QI | direct | 1 | yes | ok | 1.19e-02 | 7.001 | - | 28 | 0.7 |
| CPU | QI | direct | 2 | no | stopped | 3.93e-03 | 7.000 | - | 31 | 1.3 |
| CPU | QI | direct | 2 | yes | ok | 5.39e-03 | 7.004 | - | 32 | 1.1 |
| CPU | QI | direct | 3 | no | stopped | 1.46e-02 | 7.018 | - | 29 | 1.4 |
| CPU | QI | direct | 3 | yes | ok | 4.21e-03 | 7.003 | - | 28 | 1.1 |

GPU quick-look diagnostic wall-time summary for the plotted runs:

| Backend | Problem | Policy | max_mode | ESS | Status | Final J | Aspect | Iota | nfev | Wall min |
|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| GPU | QA | continuation | 1 | no | stopped | 9.19e-03 | 6.002 | 0.3950 | 12 | 2.3 |
| GPU | QA | continuation | 1 | yes | stopped | 9.19e-03 | 6.002 | 0.3950 | 12 | 2.3 |
| GPU | QA | continuation | 2 | no | stopped | 1.19e-03 | 6.007 | 0.4071 | 20 | 3.7 |
| GPU | QA | continuation | 2 | yes | stopped | 8.19e-04 | 6.000 | 0.4054 | 20 | 4.0 |
| GPU | QA | continuation | 3 | no | stopped | 1.38e-04 | 6.000 | 0.4088 | 30 | 5.7 |
| GPU | QA | continuation | 3 | yes | stopped | 3.92e-04 | 6.003 | 0.4072 | 30 | 5.8 |
| GPU | QH | continuation | 1 | no | stopped | 1.71e-01 | 7.027 | - | 5 | 1.2 |
| GPU | QH | continuation | 1 | yes | stopped | 1.71e-01 | 7.027 | - | 5 | 0.6 |
| GPU | QH | continuation | 2 | no | stopped | 5.31e-02 | 7.012 | - | 7 | 2.2 |
| GPU | QH | continuation | 2 | yes | stopped | 5.58e-02 | 7.011 | - | 7 | 1.4 |
| GPU | QH | continuation | 3 | no | stopped | 6.51e-02 | 7.010 | - | 9 | 2.3 |
| GPU | QH | continuation | 3 | yes | stopped | 5.15e-02 | 7.003 | - | 9 | 1.8 |
| GPU | QP | continuation | 1 | no | stopped | 6.87e-01 | 7.238 | -0.3151 | 5 | 1.3 |
| GPU | QP | continuation | 1 | yes | stopped | 6.87e-01 | 7.238 | -0.3151 | 5 | 0.6 |
| GPU | QP | continuation | 2 | no | stopped | 5.16e-01 | 7.060 | -0.6073 | 7 | 2.3 |
| GPU | QP | continuation | 2 | yes | stopped | 4.21e-01 | 7.405 | -0.3375 | 7 | 1.6 |
| GPU | QP | continuation | 3 | no | stopped | 6.27e-01 | 7.159 | -0.8721 | 9 | 2.0 |
| GPU | QP | continuation | 3 | yes | stopped | 4.78e-01 | 7.151 | -0.3198 | 9 | 1.8 |
| GPU | QI | continuation | 1 | no | stopped | 4.70e-02 | 7.130 | - | 10 | 1.1 |
| GPU | QI | continuation | 1 | yes | stopped | 4.70e-02 | 7.130 | - | 10 | 1.2 |
| GPU | QI | continuation | 2 | no | stopped | 3.44e-02 | 7.032 | - | 12 | 2.1 |
| GPU | QI | continuation | 2 | yes | stopped | 1.37e-02 | 7.002 | - | 12 | 2.2 |
| GPU | QI | continuation | 3 | no | stopped | 3.86e-02 | 6.977 | - | 14 | 2.5 |
| GPU | QI | continuation | 3 | yes | stopped | 7.35e-03 | 6.995 | - | 14 | 2.5 |
| GPU | QA | direct | 1 | no | stopped | 9.19e-03 | 6.002 | 0.3950 | 12 | 2.3 |
| GPU | QA | direct | 1 | yes | stopped | 9.19e-03 | 6.002 | 0.3950 | 12 | 2.3 |
| GPU | QA | direct | 2 | no | stopped | 4.12e-02 | 6.110 | 0.3509 | 12 | 4.0 |
| GPU | QA | direct | 2 | yes | stopped | 6.61e-04 | 6.006 | 0.4099 | 12 | 4.2 |
| GPU | QA | direct | 3 | no | stopped | 6.81e-02 | 5.951 | 0.1732 | 12 | 2.8 |
| GPU | QA | direct | 3 | yes | stopped | 1.43e-04 | 6.000 | 0.4084 | 24 | 4.7 |
| GPU | QH | direct | 1 | no | stopped | 1.71e-01 | 7.027 | - | 5 | 0.7 |
| GPU | QH | direct | 1 | yes | stopped | 1.71e-01 | 7.027 | - | 5 | 0.6 |
| GPU | QH | direct | 2 | no | stopped | 5.56e-02 | 7.001 | - | 4 | 1.0 |
| GPU | QH | direct | 2 | yes | stopped | 4.08e-02 | 7.012 | - | 5 | 1.3 |
| GPU | QH | direct | 3 | no | stopped | 9.78e-02 | 7.051 | - | 4 | 1.0 |
| GPU | QH | direct | 3 | yes | stopped | 4.83e-02 | 7.011 | - | 5 | 1.4 |
| GPU | QP | direct | 1 | no | stopped | 6.87e-01 | 7.238 | -0.3151 | 5 | 0.6 |
| GPU | QP | direct | 1 | yes | stopped | 6.87e-01 | 7.238 | -0.3151 | 5 | 0.6 |
| GPU | QP | direct | 2 | no | stopped | 1.23e+00 | 6.918 | -1.2527 | 4 | 1.0 |
| GPU | QP | direct | 2 | yes | stopped | 4.40e-01 | 7.110 | -0.8996 | 5 | 1.3 |
| GPU | QP | direct | 3 | no | stopped | 1.10e+00 | 7.160 | -1.0789 | 4 | 1.0 |
| GPU | QP | direct | 3 | yes | stopped | 6.62e-01 | 7.092 | -1.1608 | 5 | 1.2 |
| GPU | QI | direct | 1 | no | stopped | 4.70e-02 | 7.130 | - | 10 | 1.8 |
| GPU | QI | direct | 1 | yes | stopped | 4.70e-02 | 7.130 | - | 10 | 1.2 |
| GPU | QI | direct | 2 | no | stopped | 3.36e-02 | 7.012 | - | 9 | 2.0 |
| GPU | QI | direct | 2 | yes | stopped | 6.10e-03 | 6.985 | - | 10 | 2.0 |
| GPU | QI | direct | 3 | no | stopped | 5.65e-02 | 6.973 | - | 9 | 1.9 |
| GPU | QI | direct | 3 | yes | stopped | 1.46e-02 | 6.962 | - | 10 | 1.8 |

## Performance vs parity

- Default runs select the fastest stable path for each input automatically.
- Use `--parity` (or `performance_mode=False` in Python) to force the conservative VMEC2000 loop.
- Use `--solver-mode accelerated` to force the optimized fixed-boundary controller.
- For GPU benchmarking, compare both first-process and cache-warm timings; the first GPU process pays XLA compilation, while later processes reuse the persistent cache automatically.

Details, profiling guidance, and parity methodology:

- `docs/performance.rst`
- `docs/validation.rst`
- `tools/diagnostics/parity_manifest.toml` + `tools/diagnostics/parity_sweep_manifest.py`

## CLI reference

```
vmec_jax input.*                run the equilibrium solver → wout_*.nc
vmec_jax --plot wout.nc         generate diagnostic plots (4 output files)
vmec_jax --parity input.*       force conservative VMEC2000 loop
vmec_jax --help                 full option list
```

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

To disable live printing:

```bash
export VMEC_JAX_SCAN_PRINT=0
```

Quiet runs (`--quiet` or `verbose=False`) default the scan path to minimal history
mode to reduce host/device traffic. Override with:

```bash
export VMEC_JAX_SCAN_MINIMAL=0  # keep full scan diagnostics even when quiet
```
