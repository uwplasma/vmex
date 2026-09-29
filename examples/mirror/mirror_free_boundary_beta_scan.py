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

The composite the docs embed is written to ``OUTPUT_DIR`` as lossless
WebP; copying it over the committed one in ``docs/_static/figures``
reproduces those bytes. The loops are :class:`vmex.mirror.CircularCoils` (the exact
elliptic-integral field); any ESSOS, SIMSOPT or mgrid field can replace them
through ``solve_mirror_beta_scan(..., external_field=...)``.
"""

import json
import os
from pathlib import Path

import jax
import numpy as np

from vmex.mirror import (
    CircularCoils,
    MirrorInput,
    SplineMirrorDiscretization,
    mirror_ratio_diagnostics,
    plot_mout,
    solve_mirror_beta_scan,
)
from vmex.mirror.output import (
    FreeBoundaryRestart,
    load_free_boundary_restart,
    plot_axisymmetric_beta_scan_summary,
    save_free_boundary_restart,
    summarize_axisymmetric_beta_scan,
)

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
# supported endpoint and two extended points on a coarse grid:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
if ci_smoke:
    BETAS = np.asarray([0.0, 0.10, 0.25, 0.50])
    NS, NXI, SPLINE_ELEMENTS = 5, 7, 4
    EXTERIOR_NTHETA = 8
    MAX_ITERATIONS = 500

###############################################################################
# End of input parameters.
###############################################################################

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

### Set up the equilibrium ####################################################

coils = CircularCoils(
    np.full(2, COIL_RADIUS),
    np.asarray([-0.5, 0.5]) * COIL_SEPARATION,
    np.full(2, COIL_CURRENT),
)
center_field = float(coils.axis_field(0.0))
inp = MirrorInput(
    ns=NS,
    nxi=NXI,
    elements=SPLINE_ELEMENTS,
    z_min=Z_MIN,
    z_max=Z_MAX,
    phiedge=np.pi * CENTER_RADIUS**2 * center_field,
    lfreeb=True,
    coil_radius=coils.radius,
    coil_z=coils.z,
    coil_current=coils.current,
    exterior_ntheta=EXTERIOR_NTHETA,
    exterior_order=EXTERIOR_ORDER,
    ftol=FTOL,
    niter=MAX_ITERATIONS,
)

### Solve the beta scan #######################################################

print(f"Solving {BETAS.size} beta points at ns={NS}, nxi={NXI}, ftol={FTOL:.0e}")
restart = None
if RESTART_FROM is not None:
    discretization = SplineMirrorDiscretization.build_cgl(inp.config, elements=SPLINE_ELEMENTS)
    restart = load_free_boundary_restart(RESTART_FROM, discretization)
solutions = solve_mirror_beta_scan(inp, BETAS, initial_restart=restart)
results = [solution.result for solution in solutions]
grid = solutions[0].discretization.grid
vacuum_axis_field = np.asarray(coils.axis_field(grid.z))

### Save the states and the summary ###########################################

labels = [f"beta_{100 * beta:05.1f}pct".replace(".", "p") for beta in BETAS]
for label, solution in zip(labels, solutions, strict=True):
    if SAVE_RESTARTS:
        save_free_boundary_restart(OUTPUT_DIR / label, FreeBoundaryRestart.from_result(solution.result))
    solution.write_mout(OUTPUT_DIR / f"mout_mirror_{label}.nc")
diagnostics = summarize_axisymmetric_beta_scan(
    results, BETAS, grid, reference_field=center_field, axial_flux_derivative=solutions[0].axial_flux_derivative
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
    f"Two circular loops (radius {COIL_RADIUS} m at z = +/-{0.5 * COIL_SEPARATION} m, "
    f"{COIL_CURRENT:.3g} A each) give vacuum B(0) = {center_field:.4f} T, "
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
    OUTPUT_DIR,
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
