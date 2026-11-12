#!/usr/bin/env python
"""Field of a mirror plasma outside its boundary, and field lines through it.

A fixed-boundary mirror equilibrium gives the total field on its boundary: the side wall and the two
end planes. Treating that closed surface as a virtual-casing surface
(:class:`virtual_casing_jax.mirror.MirrorExteriorField`) splits the total field into the field of the
currents inside the surface (the plasma) and the field of the sources outside it (the coils). The script
evaluates the plasma field outside the plasma and, with ``branch="external"``, the coil field inside,
then traces field lines of the exact vacuum mirror field plus the plasma field. At zero pressure the
plasma field outside vanishes up to the discretization error of the equilibrium (about 1e-5 T at the
default resolution, against 5e-4 T at ``PRES_SCALE = 2e4``). Anisotropic-pressure equilibria are not covered.

Requires a ``virtual-casing-jax`` release that includes ``virtual_casing_jax.mirror``.
"""

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
from virtual_casing_jax.mirror import MirrorExteriorField, mirror_panels

from vmex.mirror import MirrorInput, solve_mirror
from vmex.mirror.analytic import AxisymmetricPolynomialMirror

# Exact vacuum mirror: midplane field [T], half-length [m], on-axis mirror ratio:
CENTER_FIELD = 1.0
HALF_LENGTH = 1.0
MIRROR_RATIO = 1.5

# Plasma radius at the midplane [m] and boundary stations along z:
MIDPLANE_RADIUS = 0.12
BOUNDARY_STATIONS = 25

# Pressure p(s) = PRES_SCALE * (1 - s) [Pa]; 0 is the vacuum check:
PRES_SCALE = 2.0e4

# Resolution: radial surfaces and axial B-spline elements:
NS = 7
ELEMENTS = 6

# Field-line launch radii at the midplane [m] and integration:
LAUNCH_RADII = (0.2, 0.3, 0.4)
LINE_STEPS = 80

# Directory for the figure and summary.json:
OUTPUT_DIR = Path("results/mirror_exterior_field")

###############################################################################
# End of input parameters.
###############################################################################


### Solve the equilibrium #####################################################

vacuum = AxisymmetricPolynomialMirror(
    center_field=CENTER_FIELD, half_length=HALF_LENGTH, mirror_strength=MIRROR_RATIO - 1.0
)
zb = np.linspace(-HALF_LENGTH, HALF_LENGTH, BOUNDARY_STATIONS)
inp = MirrorInput(
    ns=NS,
    elements=ELEMENTS,
    z_min=-HALF_LENGTH,
    z_max=HALF_LENGTH,
    phiedge=2.0 * np.pi * float(vacuum.poloidal_flux(MIDPLANE_RADIUS, 0.0)),
    zb=zb,
    rbc=[vacuum.boundary_radius(MIDPLANE_RADIUS, zb)],
    pres_scale=PRES_SCALE,
    am=[1.0, -1.0],
    ftol=1.0e-12,
    niter=1000,
)
mout = solve_mirror(inp).mout()

### Boundary traces: side wall (s = 1) and the two end planes ##################

# b_xyz is (Br, Bphi, Bz) in the v = 0 plane; the field at angle v is that vector rotated about z. The
# cubic splines of the samples are tabulated finely so that the traces are traceable JAX functions.
z = np.asarray(mout.z)


def tabulate(x, y):
    fine = np.linspace(x[0], x[-1], 4001)
    table = jnp.asarray(CubicSpline(x, y)(fine))
    fine = jnp.asarray(fine)
    return lambda q: jnp.stack([jnp.interp(q, fine, table[:, k]) for k in range(table.shape[1])])


def rotate(b, v):
    c, s = jnp.cos(v), jnp.sin(v)
    return jnp.stack((c * b[0] - s * b[1], s * b[0] + c * b[1], b[2]))


def cap_trace(index):
    r = np.sqrt(np.asarray(mout.s)) * np.asarray(mout.radius_scale)[-1, 0, index]  # r = sqrt(s) * a(z)
    table = tabulate(r, np.asarray(mout.b_xyz)[:, 0, index])
    return lambda w, v: rotate(table(w * r[-1]), v)


wall_radius = tabulate(z, np.asarray(mout.radius_scale)[-1, 0][:, None])
wall_field = tabulate(z, np.asarray(mout.b_xyz)[-1, 0])


def radius(zz, v):
    return wall_radius(zz)[0] + 0.0 * v


def side(zz, v):
    return rotate(wall_field(zz), v)


panels = mirror_panels(radius, z[0], z[-1], side, cap_trace(0), cap_trace(-1))
surface = MirrorExteriorField(panels)
flux_defect = float(surface.flux_defect())

### Fields outside and inside the plasma ######################################

zs = np.linspace(-0.9 * HALF_LENGTH, 0.9 * HALF_LENGTH, 9)
outside = jnp.asarray(np.stack((np.full_like(zs, 2.0 * MIDPLANE_RADIUS), 0.0 * zs, zs), axis=-1))
inside = jnp.asarray(np.stack((np.full_like(zs, 0.5 * MIDPLANE_RADIUS), 0.0 * zs, zs), axis=-1))
plasma_outside = np.asarray(surface.B_plasma_xyz(outside))
coil_inside = np.asarray(surface.B_plasma_xyz(inside, branch="external"))
coil_exact = np.asarray(jax.vmap(vacuum.field)(inside))
plasma_scale = float(np.max(np.linalg.norm(plasma_outside, axis=-1)))
coil_change = float(np.max(np.linalg.norm(coil_inside - coil_exact, axis=-1)) / CENTER_FIELD)

### Field lines of the vacuum field plus the plasma field ######################


def total_field(x):
    return vacuum.field(x) + surface.B_plasma_xyz(x[None])[0]


def trace(x0):
    """RK4 steps in arc length along B, from the midplane to the end of the device."""
    h = 1.2 * HALF_LENGTH / LINE_STEPS

    def unit(x):
        return total_field(x) / jnp.linalg.norm(total_field(x))

    def step(x, _):
        k1 = unit(x)
        k2 = unit(x + 0.5 * h * k1)
        k3 = unit(x + 0.5 * h * k2)
        k4 = unit(x + h * k3)
        x = x + h / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return x, x

    return jnp.concatenate((x0[None], jax.lax.scan(step, x0, None, length=LINE_STEPS)[1]))


lines = [np.asarray(trace(jnp.array([r, 0.0, 0.0]))) for r in LAUNCH_RADII]

### Report and plot ###########################################################

summary = {
    "pres_scale": PRES_SCALE,
    "net_flux_defect": flux_defect,
    "plasma_field_outside_max_T": plasma_scale,
    "coil_field_inside_change_from_vacuum_relative_to_center_field": coil_change,
    "field_line_end_radii_m": [float(np.hypot(line[-1, 0], line[-1, 1])) for line in lines],
}
print(json.dumps(summary, indent=2))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(7.0, 3.6))
zz = np.linspace(z[0], z[-1], 200)
edge = np.array([float(radius(q, 0.0)) for q in zz])
ax.plot(zz, edge, "k", label="plasma boundary")
ax.plot(zz, -edge, "k")
for line in lines:
    ax.plot(line[:, 2], np.hypot(line[:, 0], line[:, 1]), color="C0")
ax.set_xlabel("z [m]")
ax.set_ylabel("r [m]")
ax.legend()
fig.tight_layout()
figure_path = OUTPUT_DIR / "mirror_exterior_field_lines.png"
fig.savefig(figure_path, dpi=150)
print(f"Wrote {figure_path}")
summary_path = OUTPUT_DIR / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2) + "\n")
print(f"Wrote {summary_path}")
