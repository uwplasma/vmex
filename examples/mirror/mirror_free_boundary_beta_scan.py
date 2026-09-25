#!/usr/bin/env python
"""Solved two-coil free-boundary mirror beta scan and physics plots.

Two circular ESSOS coils make an axisymmetric vacuum mirror. The script
continues the coupled plasma-boundary-vacuum equilibrium through the beta
points in ``BETAS`` (0, 10, 50 and 80 %), each solved to the residual
tolerance ``FTOL``; no prescribed finite-beta boundary is plotted. The points
through 10 % are the supported lane and must pass the strong-force gate; 50
and 80 % are extended validation outside the supported model range. Each
point costs about a minute on a laptop CPU, so add intermediate points (1, 3,
25 %, ...) only when the extra minutes are wanted. It writes one MOUT and
restart file per point, a JSON summary, the mirror ratios, per-state figures
for three points, and the beta-scan composite.

The composite the docs embed is written straight into
``docs/_static/figures`` as lossless WebP, so re-running this script
reproduces the committed bytes; ``VMEX_EXAMPLES_CI=1`` sends it to
``OUTPUT_DIR`` instead. The coils need ESSOS (``pip install "vmex[coils]"``);
without it the script exits with that message.
"""

import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from vmex.mirror import (
    MirrorBoundary,
    MirrorConfig,
    MirrorResolution,
    SplineMirrorDiscretization,
    mirror_ratio_diagnostics,
    mout_from_result,
    plot_mout,
    solve_beta_scan,
    write_mout,
)
from vmex.mirror.output import (
    FreeBoundaryRestart,
    load_free_boundary_restart,
    plot_axisymmetric_beta_scan_summary,
    save_free_boundary_restart,
    summarize_axisymmetric_beta_scan,
)

try:
    from essos.coils import Coils, Curves
    from essos.fields import BiotSavart
except ModuleNotFoundError:
    raise SystemExit("This example needs ESSOS: pip install 'vmex[coils]'") from None

# Requested central beta of each point; the scan continues from vacuum:
BETAS = np.asarray([0.0, 0.10, 0.50, 0.80])

# Largest beta of the supported lane, and its strong-force gate:
SUPPORTED_BETA_MAX = 0.10
STRONG_FORCE_GATE = 5.0e-2

# Radial surfaces, axial samples, and axial spline elements:
NS = 7
NXI = 13
SPLINE_ELEMENTS = 7

# Exterior (vacuum) boundary-integral resolution:
EXTERIOR_NTHETA = 12
EXTERIOR_ORDER = 6
EXTERIOR_SPECTRAL_SIDE_DENSITY = True

# Force tolerance and iteration budget per point:
FTOL = 1.0e-12
MAX_ITERATIONS = 2000

# Axial extent of the modelled grid [m]:
Z_MIN, Z_MAX = -0.8, 0.8

# Two circular coils sized to the plasma [m, m, A]: vacuum field on axis at
# the midplane B(0) ~ 0.0836 T:
COIL_RADIUS = 0.5
COIL_SEPARATION = 2.0
COIL_CURRENT = 3.72e5

# Vacuum plasma radius at the midplane [m]:
CENTER_RADIUS = 0.25

# Directory for the MOUT, restart, JSON and per-state figure files:
OUTPUT_DIR = Path("results/mirror_free_boundary_beta_scan")

# Write one hot-start .npz per beta point; to resume, set RESTART_FROM to one
# of them (e.g. OUTPUT_DIR / "beta_010p0pct.npz") and trim BETAS to the rest:
SAVE_RESTARTS = True
RESTART_FROM = None

# VMEX_EXAMPLES_CI=1 is the smoke pass the test suite runs: vacuum, the
# supported endpoint and two extended points on a coarse grid, with the
# composite kept out of the docs tree:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = REPO_ROOT / "docs" / "_static" / "figures"
if ci_smoke:
    BETAS = np.asarray([0.0, 0.10, 0.25, 0.50])
    NS, NXI, SPLINE_ELEMENTS = 5, 7, 4
    EXTERIOR_NTHETA = 8
    MAX_ITERATIONS = 500
    FIGURE_DIR = OUTPUT_DIR

###############################################################################
# End of input parameters.
###############################################################################

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

### Coils and vacuum field ####################################################

coil_dofs = np.zeros((2, 3, 3))
coil_dofs[:, 0, 2] = COIL_RADIUS
coil_dofs[:, 1, 1] = COIL_RADIUS
coil_dofs[:, 2, 0] = np.asarray([-0.5, 0.5]) * COIL_SEPARATION
coils = Coils(
    Curves(jnp.asarray(coil_dofs), n_segments=128, nfp=1, stellsym=False),
    jnp.full(2, COIL_CURRENT),
)
biot_savart = BiotSavart(coils)


def external_field(points):
    """Evaluate the ESSOS field on an arbitrary array of Cartesian points."""

    points = jnp.asarray(points)
    return jax.vmap(biot_savart.B)(points.reshape(-1, 3)).reshape(points.shape)


### Grid and vacuum boundary ##################################################

config = MirrorConfig(
    resolution=MirrorResolution(ns=NS, mpol=0, nxi=NXI),
    z_min=Z_MIN,
    z_max=Z_MAX,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
source_grid = config.build_grid()
discretization = SplineMirrorDiscretization.build_cgl(config, elements=SPLINE_ELEMENTS)
grid = discretization.grid
initial_restart = None if RESTART_FROM is None else load_free_boundary_restart(RESTART_FROM, discretization)
z = jnp.asarray(grid.z)
coil_z = 0.5 * COIL_SEPARATION
vacuum_axis_field = sum(
    4.0e-7 * jnp.pi * COIL_CURRENT * COIL_RADIUS**2 / (2.0 * (COIL_RADIUS**2 + (z - position) ** 2) ** 1.5)
    for position in (-coil_z, coil_z)
)
center = int(np.argmin(np.abs(grid.z)))
axial_flux_derivative = 0.5 * vacuum_axis_field[center] * CENTER_RADIUS**2
initial_boundary = discretization.fit_boundary(
    MirrorBoundary.from_axis_field(
        axial_flux_derivative,
        vacuum_axis_field,
        grid,
    ),
    source_grid,
)
### Solve the beta scan #######################################################

print(f"Solving {BETAS.size} beta points at ns={NS}, nxi={NXI}, ftol={FTOL:.0e}")
results = solve_beta_scan(
    initial_boundary,
    discretization,
    config,
    external_field,
    jnp.asarray(BETAS),
    axial_flux_derivative=axial_flux_derivative,
    reference_field=float(vacuum_axis_field[center]),
    initial_restart=initial_restart,
    exterior_ntheta=EXTERIOR_NTHETA,
    exterior_order=EXTERIOR_ORDER,
    exterior_spectral_side_density=EXTERIOR_SPECTRAL_SIDE_DENSITY,
)
### Save the states and the summary ###########################################

gamma = np.asarray(coils.gamma)
if SAVE_RESTARTS:
    for beta, result in zip(BETAS, results, strict=True):
        label = f"beta_{100 * beta:05.1f}pct".replace(".", "p")
        save_free_boundary_restart(OUTPUT_DIR / label, FreeBoundaryRestart.from_result(result))
for beta, result in zip(BETAS, results, strict=True):
    label = f"beta_{100 * beta:05.1f}pct".replace(".", "p")
    write_mout(
        OUTPUT_DIR / f"mout_mirror_{label}.nc",
        mout_from_result(
            result,
            grid,
            config,
            axial_flux_derivative=axial_flux_derivative,
            coil_xyz=gamma,
        ),
    )
diagnostics = summarize_axisymmetric_beta_scan(
    results,
    jnp.asarray(BETAS),
    grid,
    reference_field=float(vacuum_axis_field[center]),
)


summary = [
    {key: float(value) for key, value in vars(item).items()}
    | {
        "variational_max": float(result.variational_max),
        "pointwise_force_rms": float(result.plasma_force.normalized_rms),
        "supported_lane": bool(
            item.requested_beta <= SUPPORTED_BETA_MAX
            and float(result.plasma_force.normalized_rms) < STRONG_FORCE_GATE
        ),
        "model_supported_beta_range": bool(item.requested_beta <= SUPPORTED_BETA_MAX),
        "passes_strong_force_gate": bool(
            float(result.plasma_force.normalized_rms) < STRONG_FORCE_GATE
        ),
    }
    for item, result in zip(diagnostics, results, strict=True)
]
(OUTPUT_DIR / "beta_scan_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
for row in summary:
    if row["model_supported_beta_range"]:
        assert row["passes_strong_force_gate"], f"supported beta point failed the force gate: {row}"

### Plot ######################################################################

middle_beta = min(0.10, 0.5 * float(BETAS[-1]))
display_indices = sorted({0, int(np.argmin(np.abs(BETAS - middle_beta))), len(BETAS) - 1})
for index in display_indices:
    label = f"beta_{100 * BETAS[index]:05.1f}pct".replace(".", "p")
    plot_mout(
        OUTPUT_DIR / f"mout_mirror_{label}.nc",
        OUTPUT_DIR,
        name=f"mirror_{label}",
    )

# One definition of "mirror ratio" across the mirror lane: R_m,axis is the
# max/min of |B| on the axis over the |B| well bounded by two maxima, and
# R_m,LCFS is reported separately.  Here the |B| maxima sit at the two ends of
# the modelled grid, since the coils lie outside it, so L_mirror,B is the grid
# length rather than the coil separation.
ratios = mirror_ratio_diagnostics(
    np.asarray(vacuum_axis_field),
    np.asarray(grid.z),
    lcfs_field_strength=np.sqrt(np.asarray(results[0].plasma_b_squared[-1, 0])),
    axis_curvature=np.zeros(grid.nxi),
)
(vacuum_well,) = ratios.wells
mirror_ratio = vacuum_well.mirror_ratio
(OUTPUT_DIR / "mirror_ratios.json").write_text(json.dumps(ratios.summary(), indent=2) + "\n")
radius_expansion = 100.0 * (summary[-1]["center_radius"] / summary[0]["center_radius"] - 1.0)
field_reduction = 100.0 * (1.0 - summary[-1]["diamagnetic_field_ratio"])
final_gate = (
    "its independent force gate fails"
    if not summary[-1]["passes_strong_force_gate"]
    else "beyond the supported model range"
)
caption = (
    f"Two ESSOS loops (radius {COIL_RADIUS} m at z = +/-{0.5 * COIL_SEPARATION} m, "
    f"{COIL_CURRENT:.3g} A each) give vacuum B(0) = {float(vacuum_axis_field[center]):.4f} T, "
    f"vacuum R_m,axis = {mirror_ratio:.2f} over L_mirror,B = {vacuum_well.mirror_length:.2f} m, "
    f"and R_m,LCFS = {ratios.lcfs_mirror_ratio:.2f} at beta = 0. "
    f"Betas through {100 * SUPPORTED_BETA_MAX:g}% pass the "
    f"strong-force gate; the {100 * float(BETAS[-1]):g}% validation continuation expands the center radius by "
    f"{radius_expansion:.2f}% and lowers the on-axis field by {field_reduction:.2f}% ({final_gate})."
)
composite = plot_axisymmetric_beta_scan_summary(
    [
        (
            f"beta = {100 * beta:g}%",
            OUTPUT_DIR / f"mout_mirror_{f'beta_{100 * beta:05.1f}pct'.replace('.', 'p')}.nc",
            bool(row["supported_lane"]),
        )
        for beta, row in zip(BETAS, summary, strict=True)
    ],
    FIGURE_DIR,
    display=tuple(display_indices),
    name="mirror_free_boundary_beta_scan",
    strong_force_gate=STRONG_FORCE_GATE,
    image_format="webp",
)

# The figure stays clean (short title + panel labels only); the coil geometry,
# vacuum field, mirror ratio, and beta observables are reported here and in
# docs/explanation/mirror-geometry.rst.
print(caption)
print(json.dumps(summary, indent=2))
print(f"Wrote solved-state 3D, cross-section, |B|, and summary plots in {OUTPUT_DIR}")
print(f"Wrote beta-scan composite figure: {composite}")
