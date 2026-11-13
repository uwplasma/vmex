#!/usr/bin/env python
"""Free-boundary tokamak held by eight circular PF loops: how the plasma answers a coil move.

A shaped, finite-beta tokamak (R0 = 1.7 m, a = 0.5 m, Ip = 1 MA, B_phi = 2 T) sits in the field of eight circular
poloidal-field (PF) loops plus a vacuum toroidal field F / R.  The loops are ESSOS circles, so their radius, height
and current are differentiable.  The script solves the free-boundary equilibrium (the LCFS is solved for), then moves
one coil current, one coil height and one coil radius and reports, for the volume, the aspect ratio and the axis and
edge rotational transform

* the implicit derivative of the coupled plasma-vacuum root (one adjoint per observable, no step size), and
* a central difference of two fully reconverged solves.

A plasma frozen in place would not respond to a coil move at all: the whole response is the plasma re-equilibrating.
The pressure and current profiles are tabulated in the normalized toroidal flux and held fixed, as are the toroidal
flux PHIEDGE and the total current CURTOR.  The profiles and coil currents are those of an independent
Grad-Shafranov solution of the same coil set (a plasma bounded by a smooth flux surface inside the separatrix, no
wall), so the solved LCFS lies on that solution's flux surface.  The static equilibrium is not claimed vertically
stable."""

import os

import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import replace

import vmex as vj
from vmex.core import implicit as im
from vmex.core.freeboundary_implicit import make_free_boundary_config, solve_free_boundary_implicit

from essos.coils import Coils, Curves
from essos.fields import BiotSavart

# Eight PF loops (R [m], Z [m], current [A]) and the vacuum toroidal field F = R B_phi [T m]:
LOOP_R = np.array([0.9, 0.9, 1.7, 1.7, 2.6, 2.6, 2.9, 0.55])
LOOP_Z = np.array([1.5, -1.5, 2.0, -2.0, 1.3, -1.3, 0.0, 0.0])
LOOP_I = np.array([209101.0, 337989.0, 39367.0, 608581.0, -191170.0, -517298.0, -207005.0, -108351.0])
F_VAC = 3.4

# Fixed profiles on s = normalized toroidal flux (11 knots), total current [A] and toroidal flux [Wb]:
PRESSURE = [19966.0, 15706.5, 11688.3, 8174.1, 5312.0, 3153.0, 1660.1, 732.9, 238.1, 37.4, 0.0]
CURRENT_DERIVATIVE = [1.0, 0.98352, 0.91995, 0.81558, 0.68276, 0.53357, 0.38407, 0.24687, 0.13113, 0.04574, 0.0]
CURTOR, PHIEDGE = 1.0e6, 2.0489

# Initial magnetic axis (R, Z) [m], and the LCFS guess (the solve moves it) as R, Z cosine/sine coefficients, m = 0..7:
AXIS = (1.7654, 0.0602)
RBC = [1.7458, 0.5346, -0.0134, -0.0273, 0.0044, 0.0015, -0.0014, -0.0001]
RBS = [0.0, 0.0081, 0.0003, 0.0061, 0.0016, -0.0015, -0.0002, 0.0004]
ZBC = [0.0589, 0.0081, 0.003, -0.01, -0.0017, 0.0024, 0.0, -0.0006]
ZBS = [0.0, 0.5923, -0.0258, -0.0304, 0.0079, 0.0015, -0.0021, 0.0]

# Perturbation: current of loop 3 [A], height of loop 3 [m], radius of loop 4 [m]; central-difference step (x units):
UNITS = np.array([1.0e4, 1.0e-2, 1.0e-2])
STEP = 1.0e-2

# Resolution and tolerances of the solve (every solve, the implicit one and the re-solves, uses the same):
NS, MPOL, NSEG = 16, 8, 128
NITER, FTOL = 20000, 1.0e-10

# VMEX_EXAMPLES_CI=1 is the short smoke pass the test suite runs:
if os.environ.get("VMEX_EXAMPLES_CI") == "1":
    NS, MPOL, NSEG = 12, 6, 64

###############################################################################
# End of input parameters.
###############################################################################

jax.config.update("jax_enable_x64", True)


class TokamakField:
    """PF loops (ESSOS Biot-Savart) plus the vacuum toroidal field F / R, as a field pytree with ``b_cyl``."""

    def __init__(self, loops, f_vac):
        self.loops, self.f_vac = loops, f_vac

    def b_cyl(self, r, phi, z):
        b_r, b_phi, b_z = self.loops.b_cyl(r, phi, z)
        return b_r, b_phi + self.f_vac / r, b_z


jax.tree_util.register_pytree_node(
    TokamakField, lambda f: ((f.loops, f.f_vac), None), lambda _, c: TokamakField(*c))

# Circle i: x = R cos t, y = R sin t, z = Z, in ESSOS' [constant, sin, cos] Fourier layout.
zeros = np.zeros_like(LOOP_R)
dofs = np.stack([np.stack([zeros, zeros, LOOP_R], -1), np.stack([zeros, LOOP_R, zeros], -1),
                 np.stack([LOOP_Z, zeros, zeros], -1)], 1)
coils = Coils(Curves(jnp.asarray(dofs), n_segments=NSEG, nfp=1, stellsym=False), jnp.asarray(LOOP_I))
base, n_curve_dofs, current_scale = coils.dofs, coils.dofs_curves.size, float(coils.currents_scale)


def field_from_parameters(theta):
    """theta = (dI3 [x 10 kA], dZ3 [x 1 cm], dR4 [x 1 cm]) -> field of the moved loops."""
    moved = (base.at[n_curve_dofs + 3].add(theta[0] * UNITS[0] / current_scale)
             .at[3 * 9 + 6].add(theta[1] * UNITS[1])
             .at[4 * 9 + 2].add(theta[2] * UNITS[2]).at[4 * 9 + 4].add(theta[2] * UNITS[2]))
    return TokamakField(BiotSavart(coils.with_dofs(moved)), F_VAC)


knots = np.linspace(0.0, 1.0, len(PRESSURE))
DECK = os.path.join(os.path.dirname(vj.__file__), "..", "examples", "data", "input.DIII-D_lasym_false")
inp = replace(
    vj.VmecInput.from_file(DECK), lasym=True, mpol=MPOL, ntor=0, ntheta=48, nzeta=0, ncurr=1, lfreeb=True, mgrid_file="direct ESSOS field", extcur=np.ones(1),
    rbc=np.asarray(RBC[:MPOL])[None, :], rbs=np.asarray(RBS[:MPOL])[None, :], zbc=np.asarray(ZBC[:MPOL])[None, :], zbs=np.asarray(ZBS[:MPOL])[None, :],
    ns_array=np.array([NS]), ftol_array=np.array([FTOL]), niter_array=np.array([NITER]),
    pmass_type="cubic_spline", am_aux_s=knots, am_aux_f=np.asarray(PRESSURE), pres_scale=1.0,
    pcurr_type="cubic_spline_ip", ac_aux_s=knots, ac_aux_f=np.asarray(CURRENT_DERIVATIVE),
    curtor=CURTOR, phiedge=PHIEDGE)
axis_r, axis_z = np.zeros(np.size(inp.raxis_c)), np.zeros(np.size(inp.raxis_c))
axis_r[0], axis_z[0] = AXIS
inp = replace(inp, raxis_c=axis_r, zaxis_c=axis_z, raxis_s=np.zeros_like(axis_r), zaxis_s=np.zeros_like(axis_r))
params = im.params_from_input(inp)
config = make_free_boundary_config(inp, field_from_parameters(jnp.zeros(3)), ns=NS, ftol=FTOL, max_iterations=NITER,
                                   field_from_parameters=field_from_parameters)
context = im.runtime_from_params(params, config.implicit)


def observables(theta):
    """Volume [m^3], aspect ratio and axis and edge rotational transform of the reconverged equilibrium."""
    state = solve_free_boundary_implicit(params, theta, config)
    return jnp.stack([im.plasma_volume(state, context), im.aspect_ratio(state, context),
                      im.iota_axis(state, context), im.iota_edge(state, context)])


NAMES = ("volume [m^3]", "aspect", "iota axis", "iota edge")
print("Solving the free-boundary equilibrium and its coupled adjoint (first call compiles)...", flush=True)
theta0 = jnp.zeros(3)
values, pullback = jax.vjp(observables, theta0)
jacobian = np.array([np.asarray(pullback(jnp.zeros(4).at[k].set(1.0))[0]) for k in range(4)])
print("Central differences of reconverged solves...", flush=True)
solve = jax.jit(observables)
central = np.array([(np.asarray(solve(STEP * e)) - np.asarray(solve(-STEP * e))) / (2 * STEP)
                    for e in jnp.eye(3)]).T

print("\nresponse per unit of: 10 kA on loop 3 | 1 cm on loop 3 height | 1 cm on loop 4 radius")
for k, name in enumerate(NAMES):
    print(f"{name:>13s} = {float(values[k]):10.6f}  implicit {np.array2string(jacobian[k], precision=5)}")
    print(f"{'':>13s}   {'':10s}  central  {np.array2string(central[k], precision=5)}")
worst = np.abs(jacobian - central).max(axis=1) / np.abs(jacobian).max(axis=1)
print(f"largest implicit-vs-central difference, relative to each row's largest entry: {worst.max():.1e}")
