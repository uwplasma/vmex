# Solve mirror and hybrid equilibria

`vmex.mirror` solves open-mirror equilibria (fixed and free boundary) and
closed stellarator-mirror hybrids in a spline-native basis. This page is the
run recipes; the theory, gate evidence, and lane status are in
{doc}`/explanation/mirror-geometry`, and the `mout_*.nc` output format is
{doc}`/reference/wout-file`.

## Solve a fixed-boundary mirror from one radius

The one-call entry point solves an axisymmetric open mirror from a single
LCFS radius, picking a default boundary and profiles for the requested
resolution:

```python
from vmex.mirror import MirrorConfig, MirrorResolution, solve_fixed_boundary_from_radius

config = MirrorConfig(resolution=MirrorResolution(ns=7, mpol=4, nxi=17))
result = solve_fixed_boundary_from_radius(0.3, config)
```

`MirrorResolution` takes `ns` (radial surfaces), `mpol` (largest represented
poloidal Fourier mode; the collocation size is the read-only
`ntheta = 2*mpol + 1`), and `nxi` (axial nodes); the returned
`SplineMirrorSolveResult` carries the converged coefficients and the
variational, weak, and pointwise-force residuals.

For a shaped boundary, build `SplineMirrorBoundary` / `SplineMirrorState` /
`SplineMirrorDiscretization` directly and call
{func}`vmex.mirror.solve_fixed_boundary`.

## Run the shipped examples

Five examples live in the repository's `examples/mirror/` directory (they are
not part of the installed wheel, so run them from a source checkout). Each has
editable inputs at its top and takes no command-line arguments:

```console
python examples/mirror/mirror_fixed_boundary_nonaxisymmetric.py   # rotating-ellipse fixed boundary
python examples/mirror/mirror_free_boundary_beta_scan.py          # axisymmetric free-boundary beta scan (needs ESSOS)
python examples/mirror/stellarator_mirror_hybrid.py               # periodic B-spline racetrack hybrid
python examples/mirror/qi_mirror_hybrid_fourier_vs_bspline.py     # QI-mirror hybrid: Fourier vs B-spline
python examples/mirror/pleiades_mirror_reference.py               # Pleiades reference data (external checkout)
```

The last one regenerates `examples/data/pleiades_two_coil_beta_reference.csv`
and needs `PLEIADES_ROOT` at its top set to a Pleiades checkout at the commit
named in its docstring; it exits
with a message otherwise.

The first checks every convergence gate for the rotating ellipse and the
axisymmetric mirror, differentiates rotating-ellipse volume against two fully
reconverged solves, and writes MOUT plus 3-D, cross-section, `|B|`, residual,
symmetry, and analytic-direction figures. The figures that the README and
these pages embed (the paired 3-D view, the beta-scan composite, and the
hybrid panel) are written by these scripts straight into
`docs/_static/figures/` as lossless WebP, so re-running a script reproduces
the committed figure; `docs/_static/figures/figures.json` records each one.

## Plot the results

Open-mirror solves write mirror-native `mout_*.nc` files, which plot with

```console
vmex --plot mout_example.nc
```

rendering horizontal 3D, coil, cap-to-cap field-line, `|B|`, pressure,
cross-section, and residual figures.

## Run a free-boundary beta scan

```console
python examples/mirror/mirror_free_boundary_beta_scan.py
```

The script solves the beta points 0, 1, 3, 10, 25, 50, and 80% (`BETAS` at
its top) and writes one MOUT per state, a compact JSON summary, restart files, and per-state figures under
`results/mirror_free_boundary_beta_scan/`, and the beta-scan composite under
`docs/_static/figures/`. The example's two ESSOS loops are
sized to the plasma: radius 0.5 m at z = +/-1.0 m carrying 3.72e5 A each,
which keeps the central vacuum field of the recorded benchmark geometry
(about 0.0836 T) with a deeper mirror well. Only the first four points (0--10%)
are in the supported lane; 25, 50, and 80% are extended validation
({doc}`/reference/capabilities`).

External fields enter as an ESSOS/SIMSOPT Biot-Savart object, any
vectorized `xyz -> B` callable, or a shared
{class}`~vmex.core.mgrid.MgridField`; coil geometry stays in ESSOS. Field
callables that capture committed arrays should use `jax.tree_util.Partial`
(or another registered pytree) so VMEX can relocate the captured leaves; an
ordinary Python closure is opaque and pins its arrays' placement.

### Resume an interrupted scan

The example writes one compressed `.npz` hot-start per beta point by default
(`SAVE_RESTARTS = True`;
{func}`vmex.mirror.output.save_free_boundary_restart`). To resume, set
`RESTART_FROM` and trim `BETAS` to the unfinished suffix;
{func}`vmex.mirror.output.load_free_boundary_restart` checks the schema and
coefficient shapes before returning the boundary, plasma state, and
calibrated mass scale. The original beta-zero boundary remains the
pressure-profile reference.

## Build a hybrid

```python
import jax

jax.config.update("jax_enable_x64", True)

from vmex.mirror import (
    MirrorConfig,
    MirrorResolution,
    build_stellarator_mirror_hybrid,
    solve_fixed_boundary,
)

resolution = MirrorResolution(ns=5, mpol=2, nxi=4)
config = MirrorConfig(resolution=resolution, ftol=1.0e-12, max_iterations=1500)
setup = build_stellarator_mirror_hybrid(
    resolution,
    coefficient_count=32,
    axis_coefficient_count=16,  # freeze the leg-return junction
    semi_major=0.45,
    semi_minor=0.45,  # circular section
    axial_flux_derivative=0.02,
    quadrature_order=3,
)
result = solve_fixed_boundary(
    setup.initial_state,
    setup.boundary,
    setup.discretization,
    config,
    axial_flux_derivative=0.02,
    solve_lambda=True,
    axis=setup.axis,
)
print(result.evaluated.converged, float(result.evaluated.variational.maximum))
```

The returned `StellaratorMirrorSetup` carries `initial_state`, `boundary`,
`discretization`, and the closed `axis`; pass `axis=setup.axis` so the solve
uses the periodic geometry. Add `current_derivative=...` for a finite axial
current (rotational transform) and `section_turns=...` to the builder for a
rotating elliptical section, as `examples/mirror/stellarator_mirror_hybrid.py`
does.

`build_stellarator_mirror_hybrid` constructs the periodic B-spline racetrack
(two exactly straight mirror legs, two stellarator returns); pass
`axis_coefficient_count` to freeze the leg-return junction while refining the
solve basis — the contract under which the circular-section lane converges
monotonically ({doc}`/explanation/mirror-geometry`). `build_qi_mirror_hybrid`
splices straight legs into a QI stellarator axis instead;
`examples/mirror/qi_mirror_hybrid_fourier_vs_bspline.py` runs that construction end
to end.

## Pick the device

Mirror fixed/free-boundary solves and beta scans expose the same `device=`
contract as the toroidal core. The solver control flow runs in SciPy and calls
back into JAX for every value, JVP, and VJP, so the mirror-specific
`device="auto"` policy
(`vmex.core.device.resolve_mirror_device`) selects CPU unless you have
chosen a JAX placement yourself. Explicit `device="cpu"`/`"gpu"` always wins and `device=None` follows
ordinary JAX placement; no environment variable is required.

## Differentiate a mirror equilibrium

`spline_fixed_boundary_adjoint` (scalar diagnostics, reverse) and
`spline_fixed_boundary_tangent` (forward) differentiate through the converged
coefficient residual; `free_boundary_adjoint` covers the axisymmetric
free-boundary lane through the 10% beta ceiling (validated to 1.08e-10
relative against a reconverged finite difference,
`benchmarks/mirror_free_boundary_axisymmetric.json`). Scope and validation
evidence: {doc}`/explanation/mirror-geometry`.
