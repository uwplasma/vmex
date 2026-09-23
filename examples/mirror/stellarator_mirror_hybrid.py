#!/usr/bin/env python
"""Solve and plot a periodic B-spline stellarator-mirror hybrid.

The closed racetrack has two exactly straight mirror legs joined by two
stellarator returns, represented in the spline-native ``vmex.mirror`` basis
rather than in Fourier modes. A weak axial current and an elliptical section
that turns continuously around the circuit give the field its rotational
transform. The script prints the convergence and force diagnostics, the
transform from a traced field line, and writes ``summary.json``.

The panel figure the docs embed is written straight into
``docs/_static/figures`` as lossless WebP, so re-running this script
reproduces the committed bytes; ``VMEX_EXAMPLES_CI=1`` sends it to
``OUTPUT_DIR`` instead. Run it from a source checkout with VMEX installed.
"""

import json
import os
from pathlib import Path

import jax

from vmex.mirror import (
    MirrorConfig,
    MirrorResolution,
    build_stellarator_mirror_hybrid,
    plot_stellarator_mirror_hybrid,
    solve_fixed_boundary,
    trace_closed_field_line,
)

# Mirror resolution: radial surfaces, largest poloidal mode, axial grid size:
NS, MPOL, NXI = 5, 4, 4

# B-spline controls along the closed axis and their quadrature order:
SPLINE_COEFFICIENTS = 32
SPLINE_QUADRATURE_ORDER = 3

# Racetrack geometry [m]: straight-leg length, return radius, section semi-axes:
STRAIGHT_LENGTH = 8.0
RETURN_RADIUS = 2.5
SEMI_MAJOR = 0.45
SEMI_MINOR = 0.25

# Full 2*pi turns of the elliptical section around the circuit, on top of the
# return-only 90-degree rotation. The legs keep an exactly straight axis; only
# the ellipse they carry rotates. Two turns lift the transform at s = 0.75 from
# the return-only 0.085 to 0.141; 0 gives the return-only rotation:
SECTION_TURNS = 2

# Axial flux derivative and a weak axial-current derivative (the transform):
AXIAL_FLUX_DERIVATIVE = 0.02
CURRENT_DERIVATIVE = 0.002

# Force tolerance and iteration budget:
FTOL = 1.0e-12
MAX_ITERATIONS = 1000

# Directory for summary.json (and the figure under VMEX_EXAMPLES_CI=1):
OUTPUT_DIR = Path("results/stellarator_mirror_hybrid")

# VMEX_EXAMPLES_CI=1 is the smoke pass the test suite runs; it keeps the
# committed docs figure untouched:
ci_smoke = os.environ.get("VMEX_EXAMPLES_CI") == "1"
REPO_ROOT = Path(__file__).resolve().parents[2]
FIGURE_DIR = OUTPUT_DIR if ci_smoke else REPO_ROOT / "docs" / "_static" / "figures"

###############################################################################
# End of input parameters.
###############################################################################

jax.config.update("jax_enable_x64", True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

### Build the hybrid ##########################################################

resolution = MirrorResolution(ns=NS, mpol=MPOL, nxi=NXI)
config = MirrorConfig(
    resolution=resolution,
    ftol=FTOL,
    max_iterations=MAX_ITERATIONS,
)
setup = build_stellarator_mirror_hybrid(
    resolution,
    coefficient_count=SPLINE_COEFFICIENTS,
    straight_length=STRAIGHT_LENGTH,
    return_radius=RETURN_RADIUS,
    semi_major=SEMI_MAJOR,
    semi_minor=SEMI_MINOR,
    section_turns=SECTION_TURNS,
    axial_flux_derivative=AXIAL_FLUX_DERIVATIVE,
    quadrature_order=SPLINE_QUADRATURE_ORDER,
)

### Solve the equilibrium #####################################################

result = solve_fixed_boundary(
    setup.initial_state,
    setup.boundary,
    setup.discretization,
    config,
    axial_flux_derivative=AXIAL_FLUX_DERIVATIVE,
    current_derivative=CURRENT_DERIVATIVE,
    solve_lambda=True,
    axis=setup.axis,
    require_convergence=True,
)

### Plot, trace and save ######################################################

figure = plot_stellarator_mirror_hybrid(result, setup, FIGURE_DIR, image_format="webp")
field_line = trace_closed_field_line(
    result.evaluated.energy.field,
    setup.discretization,
    radial_index=NS - 2,
    turns=2,
)
summary = {
    "converged": result.evaluated.converged,
    "iterations": result.evaluated.iterations,
    "variational_max": float(result.evaluated.variational.maximum),
    "staggered_weak_max": float(result.evaluated.staggered_weak_force.maximum),
    "strong_force_normalized_rms": float(result.evaluated.force.normalized_rms),
    "strong_force_axis_rms": float(result.evaluated.force.axis_normalized_rms),
    "strong_force_first_row_rms": float(result.evaluated.force.first_row_normalized_rms),
    "strong_force_bulk_rms": float(result.evaluated.force.bulk_normalized_rms),
    "strong_force_components": [float(value) for value in result.evaluated.force.component_rms],
    "normalized_divergence_rms": float(result.evaluated.normalized_divergence_rms),
    "volume": float(result.evaluated.energy.geometry.volume),
    "axis_length": float(setup.axis.arc_length),
    "axis_closure_error": float(setup.axis.closure_error),
    "frame_closure_error": float(setup.axis.frame_closure_error),
    "section_turns": SECTION_TURNS,
    "iota_at_s_0p75": float(field_line.iota),
    "figure": str(figure),
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
