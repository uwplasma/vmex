#!/usr/bin/env python
"""Evaluate the coil plus finite-beta plasma field outside a VMEX boundary.

"""

from pathlib import Path

import jax
import jax.numpy as jnp

import vmex as vj
from vmex import optimize as opt

from essos.coils import Coils
from essos.fields import BiotSavart

DATA = Path(__file__).resolve().parent / "data"
INPUT = DATA / "input.LandremanPaul2021_QA_beta0p5_bootstrap"
COILS = DATA / "ESSOS_biot_savart_LandremanPaulQA_beta0p5_bootstrap.json"

print("Building the optimized 0.5%-beta QA equilibrium and its derivative graph...")
inp = vj.VmecInput.from_file(INPUT)
problem = opt.VmecProblem.from_input(inp, max_mode=1, use_ess=True, progress=True)
final_equilibrium = problem.equilibrium_from_x(problem.x0)

print("Loading the ESSOS coils optimized against this finite-beta equilibrium...")
coils = Coils.from_json(str(COILS))

def coil_field_from_dofs(dofs):
    field = BiotSavart(coils.with_dofs(dofs))
    return lambda points: jax.vmap(field.B)(points)

# The exterior total field is the actual ESSOS coil field plus the plasma-current
# field from virtual casing, a periodic trapezoid rule over the LCFS whose error
# decays as exp(-2 pi d / h). The default 32 x 32 source grid ends on 64 toroidal
# points over the whole torus, h = 2 pi R / 64, about 0.15 m here, so the point is
# placed 0.25 m (1.5 minor radii) outboard of the LCFS, still well inside the coils.
# On a 12 x 12 grid at 0.03 m the plasma field came out near 15 T against about
# 0.03 T; closer points need with_near_surface_continuation().
DISTANCE = 0.25  # metres along the axis-to-edge ray at theta = phi = 0
DIGITS = 4
final_equilibrium.set_points_flux([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
axis, edge = final_equilibrium.field.get_points_cart()
xyz = edge + DISTANCE * (edge - axis) / jnp.linalg.norm(edge - axis)
print("Building the coil + virtual-casing exterior field...")
outside = final_equilibrium.exterior_field(
    external_parameters=coils.dofs, external_field_from_parameters=coil_field_from_dofs,
    external_dof_names=coils.dof_names, nphi=32, ntheta=32,
    digits=DIGITS).set_points_xyz(xyz[None])
# Refuse, rather than return, a field whose estimated error exceeds 10**-DIGITS.
outside.accuracy_check = "raise"

# All returned field components and spatial derivative axes are Cartesian.
# VJPs hold xyz fixed and return boundary modes followed by ESSOS coil modes.
print("Evaluating B and its spatial derivatives...")
B = outside.B()
error_estimate = outside.B_error_estimate()
print(f"estimated relative error of the plasma field = {float(error_estimate.max()):.1e} "
      f"(requested {10.0 ** -DIGITS:.0e})")
absB = outside.absB()
gradB = outside.gradB()
gradgradB = outside.gradgradB()
gradgradgradB = outside.gradgradgradB()
print("Evaluating B VJP...")
dBdx = outside.B_vjp(jnp.ones_like(B))
print("Evaluating gradB VJP...")
dgradBdx = outside.gradB_vjp(jnp.ones_like(gradB))
print("Evaluating gradgradB VJP...")
d2Bdx = outside.gradgradB_vjp(jnp.ones_like(gradgradB))
print("Evaluating gradgradgradB VJP (the most expensive derivative)...")
d3Bdx = outside.gradgradgradB_vjp(jnp.ones_like(gradgradgradB))

print("outside Cartesian point (x, y, z) =", xyz)
print("B [T] =", B)
print("|B| [T] =", absB)
print("uses virtual casing =", outside.uses_virtual_casing)
print("gradB, gradgradB, gradgradgradB shapes =",
      gradB.shape, gradgradB.shape, gradgradgradB.shape)
print("dof_names =", outside.dof_names)
print("B, gradB, gradgradB, gradgradgradB VJP shapes =",
      dBdx.shape, dgradBdx.shape, d2Bdx.shape, d3Bdx.shape)
