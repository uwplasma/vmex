#!/usr/bin/env python
"""Solve an axisymmetric fixed-boundary mirror and check it against the exact field.

The boundary is a flux surface of an exact vacuum mirror whose on-axis field is
``B0 * (1 + (R_m - 1) * z**2 / L**2)``
(:class:`vmex.mirror.analytic.AxisymmetricPolynomialMirror`). With no pressure
the solved equilibrium must reproduce that field, so the script prints the
error of the solved ``|B|`` on the axis and on the boundary and of the mirror
ratio, then writes ``mout_*.nc``, its figures and ``summary.json``. Raise
``PRES_SCALE`` for a finite-pressure equilibrium in the same boundary; the
comparison then measures the diamagnetic change instead of an error.
"""

import json
from pathlib import Path

import jax
import numpy as np

from vmex.mirror import MirrorInput, plot_mout, solve_mirror
from vmex.mirror.analytic import AxisymmetricPolynomialMirror

# Exact vacuum mirror: midplane field [T], half-length [m], and the on-axis
# mirror ratio B(L)/B(0):
CENTER_FIELD = 1.0
HALF_LENGTH = 1.0
MIRROR_RATIO = 1.5

# Plasma radius at the midplane [m] and boundary stations along z:
MIDPLANE_RADIUS = 0.12
BOUNDARY_STATIONS = 25

# Pressure p(s) = PRES_SCALE * (1 - s) [Pa]; 0 is the vacuum benchmark:
PRES_SCALE = 0.0

# Resolution: radial surfaces and axial B-spline elements:
NS = 7
ELEMENTS = 6

# Force tolerance and iteration budget:
FTOL = 1.0e-12
NITER = 1000

# Directory for the MOUT file, its figures and summary.json:
OUTPUT_DIR = Path("results/mirror_fixed_boundary_axisymmetric")

###############################################################################
# End of input parameters.
###############################################################################

jax.config.update("jax_enable_x64", True)

### Set up the equilibrium ####################################################

field = AxisymmetricPolynomialMirror(
    center_field=CENTER_FIELD,
    half_length=HALF_LENGTH,
    mirror_strength=MIRROR_RATIO - 1.0,
)
zb = np.linspace(-HALF_LENGTH, HALF_LENGTH, BOUNDARY_STATIONS)
inp = MirrorInput(
    ns=NS,
    elements=ELEMENTS,
    z_min=-HALF_LENGTH,
    z_max=HALF_LENGTH,
    phiedge=2.0 * np.pi * float(field.poloidal_flux(MIDPLANE_RADIUS, 0.0)),
    zb=zb,
    rbc=[field.boundary_radius(MIDPLANE_RADIUS, zb)],
    pres_scale=PRES_SCALE,
    am=[1.0, -1.0],
    ftol=FTOL,
    niter=NITER,
)

### Solve the equilibrium #####################################################

solution = solve_mirror(inp)
summary = solution.summary()

### Compare with the exact field ##############################################

mod_b = solution.mod_b()
z = np.asarray(solution.discretization.grid.z)
boundary = np.asarray(solution.discretization.evaluate_boundary(solution.boundary).radius_scale)[0]
boundary_points = np.stack((boundary, 0.0 * z, z), axis=-1)
exact_boundary_field = np.linalg.norm(np.asarray(jax.vmap(field.field)(boundary_points)), axis=-1)
summary |= {
    "axis_field_max_relative_error": float(np.max(np.abs(mod_b[0, 0] / field.axis_field(z) - 1.0))),
    "boundary_field_max_relative_error": float(np.max(np.abs(mod_b[-1, 0] / exact_boundary_field - 1.0))),
    "mirror_ratio_relative_error": abs(summary["R_m_axis"][0] / MIRROR_RATIO - 1.0),
}
print(f"converged = {summary['converged']} after {summary['iterations']} iterations; "
      f"variational force = {summary['variational_max']:.2e}")
print(f"solved R_m = {summary['R_m_axis'][0]:.6f} (exact {MIRROR_RATIO}); relative errors: "
      f"R_m {summary['mirror_ratio_relative_error']:.1e}, "
      f"axis |B| {summary['axis_field_max_relative_error']:.1e}, "
      f"boundary |B| {summary['boundary_field_max_relative_error']:.1e}")

### Save and plot #############################################################

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
mout_path = solution.write_mout(OUTPUT_DIR / "mout_mirror_axisymmetric.nc")
print(f"Wrote {mout_path}")
for path in plot_mout(mout_path, OUTPUT_DIR, name="mirror_axisymmetric").values():
    print(f"Wrote {path}")
summary_path = OUTPUT_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"Wrote {summary_path}")
